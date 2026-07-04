"""
openutau-remote-inference — FastAPI 远程推理服务器

支持 DiffSinger 声学模型、方差子模型 (dur/pitch/variance)、
声码器 (ONNX/JIT) 的远程推理。自动识别 OpenUtau 模型目录结构。
"""

import argparse
import logging
import os
import sys
import time
import uuid
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import onnxruntime as ort
import torch
import uvicorn
import yaml
from fastapi import FastAPI, HTTPException, Request

from lib.dsconfig_parser import (
    AcousticConfig,
    ModelType,
    VocoderConfig,
    VarianceSubConfig,
)
from lib.model_registry import (
    ModelRegistry,
    OnnxModelInfo,
    SingerInfo,
    find_model_by_relative_path,
    find_singer_by_model_path,
    scan_all,
)

# ====================================================================
# 日志
# ====================================================================

logging.basicConfig(level=logging.INFO, format="[%(asctime)s][%(levelname)s] %(message)s")
LOGGER = logging.getLogger("openutau-remote-inference")
LOGGER.setLevel(logging.INFO)

REGISTRY: Optional[ModelRegistry] = None

# ====================================================================
# Session 管理器
# ====================================================================

class SessionManager:
    """管理客户端会话及其 ONNX/TorchScript 模型缓存，各会话隔离。"""

    def __init__(self, max_sessions: int = 10, session_ttl: int = 3600):
        self.max_sessions = max_sessions
        self.session_ttl = session_ttl
        self._sessions: Dict[str, Dict[Path, Dict[str, ort.InferenceSession]]] = {}
        self._torch_sessions: Dict[str, Dict[Path, torch.jit.ScriptModule]] = {}
        self._last_active: Dict[str, float] = {}

    def get_or_create(self, session_id: str, allow_evict: bool = True) -> Dict[Path, ort.InferenceSession]:
        """获取或创建 session。如果 allow_evict=False 且已达上限，返回 None 表示已满。"""
        self._evict_stale()
        if session_id not in self._sessions:
            if len(self._sessions) >= self.max_sessions:
                if not allow_evict:
                    return None  # 告知调用方 session 已满
                oldest = min(self._last_active, key=self._last_active.get)
                LOGGER.warning(f"[{session_id[:8]}] Max sessions ({self.max_sessions}), evicting {oldest[:8]}")
                self.release_all(oldest)
            self._sessions[session_id] = {}
            self._torch_sessions[session_id] = {}
            LOGGER.info(f"[{session_id[:8]}] Session created (active={self.active_count+1})")
        self._last_active[session_id] = time.time()
        return self._sessions[session_id]

    def get_torch_sessions(self, session_id: str) -> Dict[Path, torch.jit.ScriptModule]:
        self.get_or_create(session_id)
        return self._torch_sessions[session_id]

    def release_model(self, session_id: str, model_path: Path):
        sess_dict = self._sessions.get(session_id)
        if sess_dict and model_path in sess_dict:
            del sess_dict[model_path]
        torch_dict = self._torch_sessions.get(session_id)
        if torch_dict and model_path in torch_dict:
            del torch_dict[model_path]
        LOGGER.info(f"[{session_id[:8]}] Released model: {model_path}")

    def release_all(self, session_id: str):
        count_onnx = len(self._sessions.pop(session_id, {}))
        count_jit = len(self._torch_sessions.pop(session_id, {}))
        self._last_active.pop(session_id, None)
        LOGGER.info(f"[{session_id[:8]}] Released all: {count_onnx} ONNX, {count_jit} JIT (active={self.active_count})")

    def _evict_stale(self):
        now = time.time()
        stale = [sid for sid, last in self._last_active.items() if now - last > self.session_ttl]
        for sid in stale:
            self.release_all(sid)

    @property
    def active_count(self) -> int:
        return len(self._sessions)

    @property
    def active_sessions(self) -> list:
        return list(self._sessions.keys())


SESSION_MGR = SessionManager(max_sessions=10)

# ====================================================================
# 参数解析
# ====================================================================

parser = argparse.ArgumentParser()
parser.add_argument("-d", "--root_dir", type=str, default=str(Path(__file__).parent),
                    help="root directory containing models (Singers/, Dependencies/)")
parser.add_argument("--host", type=str, default="0.0.0.0", help="binding host")
parser.add_argument("--port", type=int, default=7889, help="binding port")
parser.add_argument("--max_sessions", type=int, default=10, help="max concurrent client sessions")
parser.add_argument("--precision", type=str, default="fp32", choices=["fp32", "fp16", "int8"],
                    help="inference precision: fp32 (default), fp16, or int8 (weights not converted)")
args = parser.parse_args()
args.root_dir = Path(args.root_dir).resolve()
SESSION_MGR.max_sessions = args.max_sessions
LOGGER.info(f"Model root directory: {args.root_dir}")
LOGGER.info(f"Max sessions: {args.max_sessions}")
LOGGER.info(f"Inference precision: {args.precision} (weights kept as-is)")

# ====================================================================
# FastAPI 应用
# ====================================================================

app = FastAPI(title="OpenUtau Remote Inference Server", version="2.0.0")


@app.on_event("startup")
async def startup():
    """启动时扫描模型目录"""
    global REGISTRY
    LOGGER.info("Scanning model registry...")
    REGISTRY = scan_all(args.root_dir)


def process_path(path_str: str) -> Path:
    """解析并校验路径安全性"""
    filepath = Path(path_str)
    if not filepath.is_absolute():
        filepath = args.root_dir / filepath
    # 使用 abspath 而非 realpath，避免符号链接/目录交接点导致路径偏离 root_dir
    abs_filepath = os.path.abspath(filepath)
    abs_root_dir = os.path.abspath(args.root_dir)
    common = Path(os.path.commonprefix([abs_filepath, abs_root_dir]))
    LOGGER.debug(f"process_path: path={path_str!r} -> filepath={filepath} abs={abs_filepath} root={abs_root_dir} common={common}")
    if common != Path(abs_root_dir):
        LOGGER.warning(f"Path rejected: {abs_filepath} is not under {abs_root_dir}")
        raise HTTPException(status_code=403, detail="Path provided is not a subpath of the root directory.")
    return filepath


def _ensure_registry():
    """确保注册表已初始化"""
    global REGISTRY
    if REGISTRY is None:
        REGISTRY = scan_all(args.root_dir)
    return REGISTRY


# ====================================================================
# 获取 Session ID（从 Header / Query）
# ====================================================================

def _get_session_id(request: Request) -> str:
    """从请求中提取或生成 session ID，优先使用客户端提供的"""
    session_id = request.headers.get("X-Session-Id", "")
    if not session_id:
        session_id = request.query_params.get("session_id", "")
    if not session_id:
        session_id = f"anon-{uuid.uuid4().hex[:12]}"
    SESSION_MGR.get_or_create(session_id)
    return session_id


# ====================================================================
# ONNX Runtime 提供器
# ====================================================================


def _build_providers(precision: str = "fp32") -> list:
    """按优先级构建 provider 列表，只包含当前运行时实际可用的 provider"""
    available = ort.get_available_providers()

    # 各 provider 的配置选项
    provider_opts = {
        'TensorrtExecutionProvider': lambda: (
            {
                'device_id': 0,
                'trt_fp16_enable': precision == "fp16",
                'trt_int8_enable': precision == "int8",
                'trt_engine_cache_enable': True,
                'trt_engine_cache_path': str(args.root_dir / ".trt_cache"),
                'trt_max_workspace_size': 4 * 1024 * 1024 * 1024,
            }
            if precision in ("fp16", "int8")
            else {}
        ),
        'CUDAExecutionProvider': lambda: {
            'device_id': 0,
            'arena_extend_strategy': 'kNextPowerOfTwo',
            'cuda_mem_limit': 4 * 1024 * 1024 * 1024,
            'do_copy_in_default_stream': True,
        },
        'DmlExecutionProvider': lambda: {},
        'CPUExecutionProvider': lambda: {},
    }

    # 优先级顺序
    priority_order = [
        'TensorrtExecutionProvider',
        'CUDAExecutionProvider',
        'DmlExecutionProvider',
        'CPUExecutionProvider',
    ]

    providers = []
    for name in priority_order:
        if name in available:
            providers.append((name, provider_opts[name]()))

    if not providers:
        providers.append(('CPUExecutionProvider', {}))

    return providers



def _load_onnx_session(
    model_path: Path,
    session_id: str,
    model_type: str = "acoustic",
    precision: Optional[str] = None,
) -> ort.InferenceSession:
    """加载（或从会话缓存获取）ONNX Runtime session"""
    if precision is None:
        precision = args.precision
    sess_key = f"precision_{precision}"
    sess_dict = SESSION_MGR.get_or_create(session_id)
    if model_path not in sess_dict:
        sess_dict[model_path] = {}
    if sess_key not in sess_dict[model_path]:
        so = ort.SessionOptions()

        # 图优化
        if model_type in ("acoustic", "variance", "pitch"):
            so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        else:
            so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED

        # 线程控制
        so.intra_op_num_threads = max(1, (os.cpu_count() or 4) - 1)
        so.inter_op_num_threads = 2

        # 执行模式
        so.execution_mode = ort.ExecutionMode.ORT_PARALLEL

        # 缓存优化
        so.enable_cpu_mem_arena = True
        so.enable_mem_pattern = True

        # 日志
        so.log_severity_level = 3

        model_bytes = model_path.read_bytes()
        model_size_mb = len(model_bytes) / (1024 * 1024)

        LOGGER.info(
            f"[{session_id[:8]}] Loading ONNX: {model_path.relative_to(args.root_dir)} "
            f"[size={model_size_mb:.1f}MB, precision={precision}]"
        )

        # 依次尝试 provider 列表，失败时自动 fallback
        providers = _build_providers(precision)
        provider_names = [p[0] for p in providers]
        LOGGER.info(f"[{session_id[:8]}] Provider chain: {' → '.join(provider_names)}")

        session = None
        last_error = None
        while providers and session is None:
            current_name = providers[0][0]
            LOGGER.info(f"[{session_id[:8]}]   ⏳ {current_name}...")
            try:
                session = ort.InferenceSession(
                    model_bytes,
                    sess_options=so,
                    providers=providers,
                )
                active = session.get_providers()
                LOGGER.info(f"[{session_id[:8]}]   ✅ {current_name} active (full: {active})")
            except Exception as e:
                last_error = e
                LOGGER.warning(
                    f"[{session_id[:8]}]   ❌ {current_name} FAILED: {e}"
                )
                remaining = [p[0] for p in providers[1:]]
                if remaining:
                    LOGGER.info(f"[{session_id[:8]}]   ⏩ Fallback: {' → '.join(remaining)}")
                providers = providers[1:]

        if session is None:
            raise RuntimeError(
                f"Failed to load model with any available provider. Last error: {last_error}"
            )

        sess_dict[model_path][sess_key] = session
        LOGGER.info(
            f"[{session_id[:8]}] Model loaded: {model_path.relative_to(args.root_dir)} "
            f"[precision={precision}]"
        )
    return sess_dict[model_path][sess_key]


def _load_torchscript_model(model_path: Path, session_id: str) -> torch.jit.ScriptModule:
    """加载（或从会话缓存获取）TorchScript 模型"""
    torch_dict = SESSION_MGR.get_torch_sessions(session_id)
    if model_path not in torch_dict:
        LOGGER.info(f"[{session_id[:8]}] Loading TorchScript: {model_path.relative_to(args.root_dir)}")
        torch_dict[model_path] = torch.jit.load(
            str(model_path), map_location=torch.device('cpu')
        )
        torch_dict[model_path].eval()
        LOGGER.info(f"[{session_id[:8]}] TorchScript loaded: {model_path.relative_to(args.root_dir)}")
    return torch_dict[model_path]


# ====================================================================
# 请求/响应 数据序列化工具
# ====================================================================

_TENSOR_TYPE_MAP = {
    "float": "float32",
    "double": "float64",
    "int8": "int8",
    "int16": "int16",
    "int32": "int32",
    "int64": "int64",
    "uint8": "uint8",
    "uint16": "uint16",
    "uint32": "uint32",
    "uint64": "uint64",
    "bool": "bool",
}


def _deserialize_input(body_input: dict, expected_type_str: str) -> np.ndarray:
    """将请求中的 tensor 数据反序列化为 numpy array"""
    type_str = expected_type_str[7:-1]  # "tensor(float)" -> "float"
    np_dtype_str = _TENSOR_TYPE_MAP.get(type_str, type_str)
    if np_dtype_str == "float":
        np_dtype_str = "float32"
    data_key = f"{type_str}_data"
    if data_key not in body_input:
        raise HTTPException(status_code=400, detail=f"Missing data field '{data_key}'")
    arr = np.array(body_input[data_key], dtype=np.dtype(np_dtype_str)).reshape(body_input["shape"])
    return arr


def _maybe_squeeze(arr: np.ndarray, model_shape) -> np.ndarray:
    """如果模型期望标量但数组是 [1]，降维"""
    if len(model_shape) == 0 and arr.ndim == 1 and arr.shape[0] == 1:
        return arr.squeeze()
    return arr


def _serialize_output(output_name: str, output: np.ndarray, onnx_type: str) -> dict:
    """将 numpy 输出序列化为响应格式"""
    type_str = onnx_type[7:-1]
    data_key = f"{type_str}_data"
    return {
        "type": onnx_type,
        "shape": list(output.shape),
        data_key: output.flatten().tolist(),
    }


# ====================================================================
# 输入校验 (根据 dsconfig.yaml)
# ====================================================================

_ACOUSTIC_FIXED_INPUTS = {
    "tokens": "tensor(int64)",
    "durations": "tensor(int64)",
    "f0": "tensor(float)",
}

_ACOUSTIC_OPTIONAL_INPUTS = {
    "gender": ("tensor(float)", "use_key_shift_embed"),
    "velocity": ("tensor(float)", "use_speed_embed"),
    "spk_embed": ("tensor(float)", None),
    "languages": ("tensor(int64)", "use_lang_id"),
}

# 参与扩散推理的内部输入
_ACOUSTIC_INTERNAL_INPUTS = {"condition", "x_aux", "depth", "steps"}


def _validate_acoustic_inputs(
    model_info: OnnxModelInfo,
    body_inputs: dict,
    session: ort.InferenceSession,
    session_id: str = "",
):
    """校验声学模型输入与 dsconfig.yaml 一致"""
    config = model_info.config
    if not isinstance(config, AcousticConfig):
        return

    # 检查必需输入
    for name, expected_type in _ACOUSTIC_FIXED_INPUTS.items():
        if name not in body_inputs:
            raise HTTPException(
                status_code=400,
                detail=f"Acoustic model requires input '{name}' ({expected_type})"
            )
        if body_inputs[name].get("type") != expected_type:
            raise HTTPException(
                status_code=400,
                detail=f"Acoustic input '{name}' should be {expected_type}, got {body_inputs[name].get('type')}"
            )

    # 检查可选输入是否按配置启用
    for name, (expected_type, config_key) in _ACOUSTIC_OPTIONAL_INPUTS.items():
        if name in body_inputs:
            if config_key and not getattr(config, config_key, False):
                LOGGER.warning(
                    f"[{session_id[:8]}] Input '{name}' provided but dsconfig '_{config_key}' is false; "
                    f"model may ignore it"
                )
            if body_inputs[name].get("type") != expected_type:
                raise HTTPException(
                    status_code=400,
                    detail=f"Acoustic input '{name}' should be {expected_type}"
                )


# ====================================================================
# 核心推理函数
# ====================================================================

def _run_onnx_inference(
    model_path: Path,
    body_inputs: dict,
    model_info: Optional[OnnxModelInfo] = None,
    session_id: str = "",
    model_type: str = "acoustic",
) -> dict:
    """通用 ONNX 推理核心"""
    if not model_path.exists():
        raise HTTPException(
            status_code=404,
            detail=f"ONNX model {model_path.relative_to(args.root_dir)} not found"
        )

    session = _load_onnx_session(model_path, session_id, model_type=model_type)

    # 模型类型感知的输入校验
    if model_info and model_info.model_type == ModelType.ACOUSTIC:
        _validate_acoustic_inputs(model_info, body_inputs, session, session_id)

    # 反序列化输入
    inputs = {}
    for model_input in session.get_inputs():
        input_name = model_input.name
        if input_name not in body_inputs:
            LOGGER.warning(
                f"[{session_id[:8]}] Missing input '{input_name}' in request. "
                f"Available inputs: {list(body_inputs.keys())}"
            )
            raise HTTPException(
                status_code=400,
                detail=f"Input '{input_name}' not found in request body"
            )
        raw = body_inputs[input_name]
        # 基本格式校验
        if "type" not in raw or "shape" not in raw:
            LOGGER.warning(
                f"[{session_id[:8]}] Input '{input_name}' has invalid format. "
                f"Keys: {list(raw.keys()) if isinstance(raw, dict) else type(raw).__name__}"
            )
            raise HTTPException(
                status_code=400,
                detail=f"Input '{input_name}' has invalid format (missing type/shape)"
            )
        if raw["type"] != model_input.type:
            LOGGER.warning(
                f"[{session_id[:8]}] Input '{input_name}' type mismatch: "
                f"got '{raw['type']}', expected '{model_input.type}'"
            )
        inputs[input_name] = _deserialize_input(raw, model_input.type)

    # 推理
    raw_outputs = session.run(None, inputs)
    model_outputs = session.get_outputs()

    result = {}
    for i in range(len(model_outputs)):
        out = raw_outputs[i]
        if isinstance(out, np.ndarray):
            result[model_outputs[i].name] = _serialize_output(
                model_outputs[i].name, out, model_outputs[i].type
            )
        else:
            result[model_outputs[i].name] = {"type": model_outputs[i].type}
    return result


def _run_jit_vocoder(
    model_path: Path,
    body_inputs: dict,
    session_id: str = "",
) -> dict:
    """TorchScript (JIT) 声码器推理"""
    jit_path = model_path.with_suffix(".jit")
    if not jit_path.exists():
        raise HTTPException(
            status_code=404,
            detail=f"TorchScript vocoder {jit_path.relative_to(args.root_dir)} not found"
        )

    model = _load_torchscript_model(jit_path, session_id)

    for k in ("mel", "f0"):
        if k not in body_inputs:
            raise HTTPException(status_code=400, detail=f"Input '{k}' not found for JIT vocoder")
        if body_inputs[k].get("type") != "tensor(float)":
            raise HTTPException(status_code=400, detail=f"Input '{k}' must be tensor(float)")

    mel = _deserialize_input(body_inputs["mel"], "tensor(float)")
    f0 = _deserialize_input(body_inputs["f0"], "tensor(float)")

    with torch.no_grad():
        signal, _, _ = model(
            torch.from_numpy(mel),
            torch.from_numpy(f0).unsqueeze(-1),
        )

    return {
        "waveform": {
            "type": "tensor(float)",
            "shape": list(signal.shape),
            "float_data": signal.flatten().tolist(),
        }
    }


def _run_onnx_vocoder(
    model_path: Path,
    body_inputs: dict,
    config: VocoderConfig,
    session_id: str = "",
) -> dict:
    """ONNX 声码器推理"""
    return _run_onnx_inference(model_path, body_inputs, session_id=session_id, model_type="vocoder")


# ====================================================================
# API 端点 — 基础
# ====================================================================

@app.get("/ping")
async def ping() -> str:
    return "pong"


@app.get("/check_variance")
async def check_variance(singer: str = ""):
    """检查端点是否可达"""
    return {"status": "ok", "endpoint": "check_variance", "singer": singer}


@app.get("/exists")
async def exists(model_path: str) -> bool:
    resolved = process_path(model_path)
    return resolved.exists()


@app.get("/registry")
async def get_registry() -> dict:
    """获取完整的模型注册信息"""
    registry = _ensure_registry()
    result = {
        "root_dir": str(registry.root_dir),
        "singers": {},
        "dependencies": {},
    }

    for sname, sinfo in registry.singers.items():
        singer_entry = {
            "name": sinfo.name,
            "display_name": sinfo.display_name,
            "base_dir": str(sinfo.base_dir.relative_to(registry.root_dir)),
            "models": {},
        }

        # 声学模型
        if sinfo.acoustic_model:
            singer_entry["models"]["acoustic"] = {
                "path": sinfo.acoustic_model.relative_path,
                "type": sinfo.acoustic_model.model_type,
            }
            if sinfo.acoustic_config:
                cfg = sinfo.acoustic_config
                singer_entry["config"] = {
                    "sample_rate": cfg.sample_rate,
                    "hop_size": cfg.hop_size,
                    "num_mel_bins": cfg.num_mel_bins,
                    "hidden_size": cfg.hidden_size,
                    "use_key_shift_embed": cfg.use_key_shift_embed,
                    "use_speed_embed": cfg.use_speed_embed,
                    "use_breathiness_embed": cfg.use_breathiness_embed,
                    "use_voicing_embed": cfg.use_voicing_embed,
                    "use_tension_embed": cfg.use_tension_embed,
                    "use_lang_id": cfg.use_lang_id,
                    "speakers": cfg.speakers,
                    "vocoder": cfg.vocoder,
                }

        # 方差子模型
        for sub_key, sub_model, sub_config in [
            ("linguistic", sinfo.linguistic_model, None),
            ("dur", sinfo.dur_model, sinfo.dur_config),
            ("pitch", sinfo.pitch_model, sinfo.pitch_config),
            ("variance", sinfo.variance_model, sinfo.variance_config),
        ]:
            if sub_model:
                entry = {
                    "path": sub_model.relative_path,
                    "type": sub_model.model_type,
                }
                if sub_config:
                    entry["predict_dur"] = sub_config.predict_dur
                    entry["use_expr"] = sub_config.use_expr
                    entry["use_note_rest"] = sub_config.use_note_rest
                    entry["predict_breathiness"] = sub_config.predict_breathiness
                    entry["predict_voicing"] = sub_config.predict_voicing
                    entry["predict_tension"] = sub_config.predict_tension
                singer_entry["models"][sub_key] = entry

        # 声码器
        if sinfo.vocoder_model:
            voc_entry: Dict[str, object] = {
                "path": sinfo.vocoder_model.relative_path,
                "type": sinfo.vocoder_model.model_type,
            }
            if sinfo.vocoder_config:
                voc_entry["model_type"] = sinfo.vocoder_config.model_type
                voc_entry["pitch_controllable"] = sinfo.vocoder_config.pitch_controllable
            singer_entry["models"]["vocoder"] = voc_entry

        result["singers"][sname] = singer_entry

    for dname, dinfo in registry.dependencies.items():
        dep_entry = {
            "base_dir": str(dinfo.base_dir.relative_to(registry.root_dir)),
            "models": [str(m.relative_path) for m in dinfo.models.values()],
        }
        if dinfo.config:
            dep_entry["id"] = dinfo.config.id
            dep_entry["version"] = dinfo.config.version
            dep_entry["class"] = dinfo.config.class_name
        result["dependencies"][dname] = dep_entry

    return result


@app.get("/singer_info")
async def singer_info(singer_name: str, request: Request) -> dict:
    """获取指定歌手的详细模型信息"""
    session_id = _get_session_id(request)
    registry = _ensure_registry()
    from lib.model_registry import find_singer_by_name
    sinfo = find_singer_by_name(registry, singer_name)
    if sinfo is None:
        raise HTTPException(status_code=404, detail=f"Singer '{singer_name}' not found")

    result = {
        "name": sinfo.name,
        "display_name": sinfo.display_name,
        "base_dir": str(sinfo.base_dir.relative_to(registry.root_dir)),
        "models": [],
    }

    for attr, label in [
        ("acoustic_model", "acoustic"),
        ("linguistic_model", "linguistic"),
        ("dur_model", "dur"),
        ("pitch_model", "pitch"),
        ("variance_model", "variance"),
        ("vocoder_model", "vocoder"),
    ]:
        m = getattr(sinfo, attr, None)
        if m:
            result["models"].append({
                "type": m.model_type,
                "path": m.relative_path,
                "inputs": [i.name for i in _load_onnx_session(m.file_path, session_id).get_inputs()],
                "outputs": [o.name for o in _load_onnx_session(m.file_path, session_id).get_outputs()],
            })

    return result


# ====================================================================
# API 端点 — ONNX 元信息
# ====================================================================

_META_SESSION = "meta"

@app.get("/onnx_info/inputs")
async def onnx_input_names(model_path: str) -> List[str]:
    """获取 ONNX 模型的输入名称列表"""
    resolved = process_path(model_path)
    if not resolved.exists():
        raise HTTPException(status_code=404, detail=f"ONNX model {resolved.relative_to(args.root_dir)} not found")
    session = _load_onnx_session(resolved, _META_SESSION)
    return [input.name for input in session.get_inputs()]


@app.get("/onnx_info/outputs")
async def onnx_output_names(model_path: str) -> List[str]:
    """获取 ONNX 模型的输出名称列表"""
    resolved = process_path(model_path)
    if not resolved.exists():
        raise HTTPException(status_code=404, detail=f"ONNX model {resolved.relative_to(args.root_dir)} not found")
    session = _load_onnx_session(resolved, _META_SESSION)
    return [output.name for output in session.get_outputs()]


@app.get("/onnx_info/details")
async def onnx_details(model_path: str) -> dict:
    """获取 ONNX 模型的完整输入/输出签名"""
    resolved = process_path(model_path)
    if not resolved.exists():
        raise HTTPException(status_code=404, detail=f"ONNX model {resolved.relative_to(args.root_dir)} not found")
    session = _load_onnx_session(resolved, _META_SESSION)
    return {
        "inputs": [
            {"name": i.name, "type": i.type, "shape": str(i.shape)}
            for i in session.get_inputs()
        ],
        "outputs": [
            {"name": o.name, "type": o.type, "shape": str(o.shape)}
            for o in session.get_outputs()
        ],
    }


# ====================================================================
# 请求日志工具
# ====================================================================

def _log_inference_request(model_path: Path, body_inputs: dict, session_id: str = ""):
    """记录推理请求日志"""
    rel_path = str(model_path.relative_to(args.root_dir))
    parts = rel_path.replace("\\", "/").split("/")

    # 提取角色名
    if len(parts) >= 2 and parts[0] == "Singers":
        model_name = parts[1]
        singer_eng = parts[2] if len(parts) > 2 else parts[1]
    else:
        model_name = parts[-1].rsplit(".", 1)[0] if parts else rel_path
        singer_eng = model_name

    # 基于路径推断模型类型（与扫描逻辑一致：看所在目录）
    path_lower = rel_path.replace("\\", "/").lower()
    if "/dsvocoder/" in path_lower:
        mtype = "Vocoder"
    elif "/dsdur/" in path_lower:
        mtype = "Dur"
    elif "/dspitch/" in path_lower:
        mtype = "Pitch"
    elif "/dsvariance/" in path_lower:
        mtype = "Variance"
    elif "dependencies/" in path_lower:
        mtype = "Dependency"
    else:
        mtype = "Acoustic"

    # 提取 steps（仅声学模型需要）
    if mtype == "Vocoder":
        LOGGER.info(f"[{session_id[:8]}] {model_name}({singer_eng}) | {mtype}: -")
        return

    steps_data = body_inputs.get("steps", {})
    steps = 0
    if steps_data:
        for key in ("int64_data", "float_data"):
            arr = steps_data.get(key, [])
            if arr:
                steps = int(arr[0])
                break

    LOGGER.info(f"[{session_id[:8]}] {model_name}({singer_eng}) | {mtype}: {steps}")


# ====================================================================
# API 端点 — 推理
# ====================================================================


def _prepare_onnx_inputs(
    session: ort.InferenceSession,
    body_inputs: dict,
    session_id: str = "",
) -> dict:
    """反序列化 ONNX 模型输入，缺失的用零填充"""
    inputs = {}
    for model_input in session.get_inputs():
        input_name = model_input.name
        if input_name not in body_inputs:
            # 缺失输入 → 零填充
            shape = [
                int(d.size if hasattr(d, "size") else (d if isinstance(d, int) else 1))
                for d in model_input.shape
            ]
            if model_input.type == "tensor(float)":
                inputs[input_name] = np.zeros(shape, dtype=np.float32)
            elif "int64" in model_input.type:
                inputs[input_name] = np.zeros(shape, dtype=np.int64)
            elif model_input.type == "tensor(bool)":
                inputs[input_name] = np.zeros(shape, dtype=bool)
            LOGGER.info(
                f"[{session_id[:8]}] Input '{input_name}' not provided, zero-filled "
                f"shape={shape}"
            )
            continue
        raw = body_inputs[input_name]
        if "type" not in raw or "shape" not in raw:
            raise HTTPException(
                status_code=400,
                detail=f"Input '{input_name}' has invalid format (missing type/shape)",
            )
        if raw["type"] != model_input.type:
            raise HTTPException(
                status_code=400,
                detail=f"Input '{input_name}' has type {raw['type']} but expected {model_input.type}",
            )
        inputs[input_name] = _deserialize_input(raw, model_input.type)
        inputs[input_name] = _maybe_squeeze(inputs[input_name], model_input.shape)
    return inputs


def _run_onnx_session(
    session: ort.InferenceSession,
    inputs: dict,
) -> dict:
    """执行 ONNX 会话并序列化输出"""
    raw_outputs = session.run(None, inputs)
    model_outputs = session.get_outputs()
    result = {}
    for i in range(len(model_outputs)):
        out = raw_outputs[i]
        if isinstance(out, np.ndarray):
            result[model_outputs[i].name] = _serialize_output(
                model_outputs[i].name, out, model_outputs[i].type
            )
        else:
            result[model_outputs[i].name] = {"type": model_outputs[i].type}
    return result


@app.post("/inference_acoustic")
async def inference_acoustic(body: Dict, request: Request):
    """
    声学模型推理专用端点。

    请求格式：
      { "model_path": "...", "inputs": { "tokens": {...}, "durations": {...}, ... } }

    缺失的可选输入（breathiness/voicing/tension/gender/velocity/depth）自动零填充。
    """
    session_id = _get_session_id(request)
    model_path = process_path(body["model_path"])
    body_inputs = body.get("inputs", {})

    _log_inference_request(model_path, body_inputs, session_id)

    registry = _ensure_registry()
    model_info = find_model_by_relative_path(
        registry, str(model_path.relative_to(args.root_dir))
    )

    session = _load_onnx_session(model_path, session_id, model_type="acoustic")

    # 声学模型输入校验（只检查必需输入，不阻止推理）
    if model_info and model_info.model_type == ModelType.ACOUSTIC:
        _validate_acoustic_inputs(model_info, body_inputs, session, session_id)

    inputs = _prepare_onnx_inputs(session, body_inputs, session_id)
    return _run_onnx_session(session, inputs)


@app.post("/inference_acoustic_batch")
async def inference_acoustic_batch(body: Dict, request: Request):
    """
    批量声学模型推理 — 一次请求推理多个片段，复用模型 session。

    请求格式：
      {
        "model_path": "...",
        "segments": [
          { "inputs": { "tokens": {...}, "durations": {...}, ... } },
          { "inputs": { "tokens": {...}, "durations": {...}, ... } }
        ]
      }

    返回格式：
      {
        "outputs": [ { "mel": {...}, ... }, { "mel": {...}, ... } ]
      }

    所有片段使用同一个模型，各片段独立推理，结果按顺序返回。
    """
    session_id = _get_session_id(request)
    model_path = process_path(body["model_path"])
    segments: list = body.get("segments", [])

    if not segments:
        raise HTTPException(status_code=400, detail="No segments provided for batch inference")

    LOGGER.info(
        f"[{session_id[:8]}] Batch inference: {model_path.relative_to(args.root_dir)} "
        f"({len(segments)} segments)"
    )

    registry = _ensure_registry()
    model_info = find_model_by_relative_path(
        registry, str(model_path.relative_to(args.root_dir))
    )

    session = _load_onnx_session(model_path, session_id, model_type="acoustic")

    # 校验第一个片段的声学输入（后续片段同模型，格式校验一致）
    if model_info and model_info.model_type == ModelType.ACOUSTIC:
        first_inputs = segments[0].get("inputs", {})
        _validate_acoustic_inputs(model_info, first_inputs, session, session_id)

    outputs = []
    for idx, segment in enumerate(segments):
        body_inputs = segment.get("inputs", {})
        LOGGER.debug(
            f"[{session_id[:8]}] Batch segment {idx + 1}/{len(segments)} "
            f"inputs: {list(body_inputs.keys())}"
        )
        inputs = _prepare_onnx_inputs(session, body_inputs, session_id)
        result = _run_onnx_session(session, inputs)
        outputs.append(result)

    LOGGER.info(
        f"[{session_id[:8]}] Batch complete: {len(segments)} segments "
        f"for {model_path.relative_to(args.root_dir)}"
    )
    return {"outputs": outputs, "count": len(segments)}


@app.post("/inference_vocoder")
async def inference_vocoder(body: Dict, request: Request):
    """
    声码器推理专用端点。

    请求格式：
      { "model_path": "...", "inputs": { "mel": {...}, "f0": {...} } }

    支持 ONNX 和 TorchScript (JIT) 声码器。
    """
    session_id = _get_session_id(request)
    model_path = process_path(body["model_path"])
    body_inputs = body.get("inputs", {})

    _log_inference_request(model_path, body_inputs, session_id)

    # 检查 JIT 声码器
    if (model_path.parent / "vocoder.yaml").exists():
        vocoder_cfg = yaml.safe_load(
            (model_path.parent / "vocoder.yaml").read_text(encoding="utf-8")
        ) or {}
        if vocoder_cfg.get("model_type", "onnx") == "jit":
            return _run_jit_vocoder(model_path, body_inputs, session_id)

    # ONNX 声码器
    registry = _ensure_registry()
    model_info = find_model_by_relative_path(
        registry, str(model_path.relative_to(args.root_dir))
    )
    vc = model_info.config if model_info and hasattr(model_info, "config") and isinstance(model_info.config, VocoderConfig) else VocoderConfig()
    return _run_onnx_vocoder(model_path, body_inputs, vc, session_id)


@app.post("/inference_vocoder_batch")
async def inference_vocoder_batch(body: Dict, request: Request):
    """
    批量声码器推理 — 一次请求推理多个片段的波形。

    请求格式：
      {
        "model_path": "...",
        "segments": [
          { "inputs": { "mel": {...}, "f0": {...} } },
          { "inputs": { "mel": {...}, "f0": {...} } }
        ]
      }

    支持 ONNX 和 TorchScript (JIT) 声码器。
    所有片段使用同一个模型，结果按顺序返回。
    """
    session_id = _get_session_id(request)
    model_path = process_path(body["model_path"])
    segments: list = body.get("segments", [])

    if not segments:
        raise HTTPException(status_code=400, detail="No segments provided for batch vocoder inference")

    LOGGER.info(
        f"[{session_id[:8]}] Batch vocoder: {model_path.relative_to(args.root_dir)} "
        f"({len(segments)} segments)"
    )

    # 检查 JIT 声码器
    is_jit = False
    if (model_path.parent / "vocoder.yaml").exists():
        vocoder_cfg = yaml.safe_load(
            (model_path.parent / "vocoder.yaml").read_text(encoding="utf-8")
        ) or {}
        if vocoder_cfg.get("model_type", "onnx") == "jit":
            is_jit = True

    # JIT 声码器暂不支持批量
    if is_jit:
        LOGGER.warning(
            f"[{session_id[:8]}] JIT vocoder does not support batch, "
            f"processing {len(segments)} segments sequentially"
        )
        outputs = []
        for idx, segment in enumerate(segments):
            body_inputs = segment.get("inputs", {})
            result = _run_jit_vocoder(model_path, body_inputs, session_id)
            outputs.append(result)
        return {"outputs": outputs, "count": len(segments)}

    # ONNX 声码器批量推理
    registry = _ensure_registry()
    model_info = find_model_by_relative_path(
        registry, str(model_path.relative_to(args.root_dir))
    )
    vc = model_info.config if model_info and hasattr(model_info, "config") and isinstance(model_info.config, VocoderConfig) else VocoderConfig()

    session = _load_onnx_session(model_path, session_id, model_type="vocoder")

    outputs = []
    for idx, segment in enumerate(segments):
        body_inputs = segment.get("inputs", {})
        inputs = _prepare_onnx_inputs(session, body_inputs, session_id)
        result = _run_onnx_session(session, inputs)
        outputs.append(result)

    LOGGER.info(
        f"[{session_id[:8]}] Batch vocoder complete: {len(segments)} segments "
        f"for {model_path.relative_to(args.root_dir)}"
    )
    return {"outputs": outputs, "count": len(segments)}


@app.post("/inference_dependency")
async def inference_dependency(body: Dict, request: Request):
    """
    依赖模块推理专用端点。

    请求格式：
      { "model_path": "...", "inputs": { ... } }
    """
    session_id = _get_session_id(request)
    model_path = process_path(body["model_path"])
    body_inputs = body.get("inputs", {})

    _log_inference_request(model_path, body_inputs, session_id)

    session = _load_onnx_session(model_path, session_id, model_type="dependency")
    inputs = _prepare_onnx_inputs(session, body_inputs, session_id)
    return _run_onnx_session(session, inputs)


@app.post("/inference")
async def inference(body: Dict, request: Request):
    """
    通用推理端点（向后兼容）— 根据模型路径自动识别类型并路由到专用端点。
    """
    session_id = _get_session_id(request)
    model_path = process_path(body["model_path"])
    body_inputs = body.get("inputs", {})

    _log_inference_request(model_path, body_inputs, session_id)

    # ---- 1. 检查 JIT 声码器 (DDSP) ----
    if (model_path.parent / "vocoder.yaml").exists():
        vocoder_cfg = yaml.safe_load(
            (model_path.parent / "vocoder.yaml").read_text(encoding="utf-8")
        ) or {}
        if vocoder_cfg.get("model_type", "onnx") == "jit":
            return _run_jit_vocoder(model_path, body_inputs, session_id)

    registry = _ensure_registry()
    model_info = find_model_by_relative_path(
        registry, str(model_path.relative_to(args.root_dir))
    )

    # 按类型路由
    if model_info and model_info.model_type == ModelType.VOCODER:
        vc = model_info.config
        if isinstance(vc, VocoderConfig) and vc.model_type == "jit":
            return _run_jit_vocoder(model_path, body_inputs, session_id)
        return _run_onnx_vocoder(
            model_path, body_inputs,
            vc if isinstance(vc, VocoderConfig) else VocoderConfig(),
            session_id,
        )

    model_type_str = "acoustic" if (model_info and model_info.model_type == ModelType.ACOUSTIC) else "dependency"
    session = _load_onnx_session(model_path, session_id, model_type=model_type_str)

    if model_info and model_info.model_type == ModelType.ACOUSTIC:
        _validate_acoustic_inputs(model_info, body_inputs, session, session_id)

    inputs = _prepare_onnx_inputs(session, body_inputs, session_id)
    return _run_onnx_session(session, inputs)


@app.post("/inference_chain")
async def inference_chain(body: Dict, request: Request):
    """
    链式推理端点 — 自动串联 linguistic → dur/pitch/variance 的多步推理。
    """
    session_id = _get_session_id(request)
    registry = _ensure_registry()
    singer_name = body.get("singer", "")
    pipeline: list = body.get("pipeline", [])
    body_inputs: dict = body.get("inputs", {})

    from lib.model_registry import find_singer_by_name
    sinfo = find_singer_by_name(registry, singer_name)
    if sinfo is None:
        raise HTTPException(status_code=404, detail=f"Singer '{singer_name}' not found")

    step_map = {
        "linguistic": sinfo.linguistic_model,
        "dur": sinfo.dur_model,
        "pitch": sinfo.pitch_model,
        "variance": sinfo.variance_model,
        "acoustic": sinfo.acoustic_model,
        "vocoder": sinfo.vocoder_model,
    }

    intermediate: dict = {}
    all_outputs: dict = {}

    for step_name in pipeline:
        model_info = step_map.get(step_name)
        if model_info is None:
            raise HTTPException(status_code=404, detail=f"Model for step '{step_name}' not found for singer '{singer_name}'")

        step_inputs = {}
        sess = _load_onnx_session(model_info.file_path, session_id, model_type=step_name)
        for mi in sess.get_inputs():
            if mi.name in body_inputs:
                step_inputs[mi.name] = body_inputs[mi.name]
            elif mi.name in intermediate:
                step_inputs[mi.name] = intermediate[mi.name]
            else:
                raise HTTPException(status_code=400, detail=f"Step '{step_name}' requires input '{mi.name}' which is missing")

        step_result = _run_onnx_inference(model_info.file_path, step_inputs, session_id=session_id)
        all_outputs[step_name] = step_result

        for oname, ovalue in step_result.items():
            intermediate[oname] = ovalue

    return {"singer": singer_name, "pipeline": pipeline, "steps": all_outputs}


# ====================================================================
# API 端点 — 子模型推理 (variance / pitch)
# ====================================================================

@app.post("/inference_variance")
async def inference_variance(body: Dict, request: Request):
    """
    方差子模型推理 — 自动串联 linguistic encoder → variance predictor。

    请求体:
    ```json
    {
        "singer": "fu2_ning2_na4",
        "inputs": {
            "tokens": {...},
            "ph_dur": {...},
            "pitch": {...},
            "languages": {...},
            "steps": {...}
        }
    }
    ```
    """
    session_id = _get_session_id(request)
    registry = _ensure_registry()
    from lib.model_registry import find_singer_by_name
    singer_name = body.get("singer", "")
    body_inputs: dict = body.get("inputs", {})

    sinfo = find_singer_by_name(registry, singer_name)
    if sinfo is None:
        raise HTTPException(status_code=404, detail=f"Singer '{singer_name}' not found")

    # 方差子模型有自己的语义编码器，不能使用共享的 linguistic_model
    var_model = sinfo.variance_model
    if var_model is None:
        raise HTTPException(status_code=404, detail=f"Variance sub-model not found for singer '{singer_name}'")

    # 从 dsvariance 目录查找对应的 linguistic ONNX
    var_dir = var_model.file_path.parent
    ling_path = None
    for f in var_dir.glob("*.linguistic.onnx"):
        ling_path = f
        break
    if ling_path is None:
        raise HTTPException(status_code=404, detail=f"Variance linguistic encoder not found in {var_dir}")

    ling_model = OnnxModelInfo(
        model_type=ModelType.LINGUISTIC,
        file_path=ling_path.resolve(),
        relative_path=str(ling_path.relative_to(args.root_dir)),
    )
    LOGGER.info(f"[{session_id[:8]}] Variance chain for {singer_name} (ling={ling_path.name}, var={var_model.file_path.name})")

    # 步骤2: 运行 linguistic encoder
    sess_ling = _load_onnx_session(ling_model.file_path, session_id, model_type="variance")
    ling_inputs = {}
    for mi in sess_ling.get_inputs():
        if mi.name == "ph_dur":
            if mi.name in body_inputs:
                ling_inputs[mi.name] = _deserialize_input(body_inputs[mi.name], mi.type)
            elif "durations" in body_inputs:
                ling_inputs[mi.name] = _deserialize_input(body_inputs["durations"], mi.type)
            else:
                raise HTTPException(status_code=400, detail=f"Linguistic encoder requires '{mi.name}'")
        elif mi.name in body_inputs:
            ling_inputs[mi.name] = _deserialize_input(body_inputs[mi.name], mi.type)
        else:
            raise HTTPException(status_code=400, detail=f"Linguistic encoder requires input '{mi.name}' which is missing")

    raw_ling = sess_ling.run(None, ling_inputs)

    # 找到 encoder_out
    encoder_out = None
    for i, out in enumerate(sess_ling.get_outputs()):
        if out.name == "encoder_out":
            encoder_out = raw_ling[i]
            break
    if encoder_out is None:
        encoder_out = raw_ling[0]

    # 步骤3: 运行 variance predictor
    sess_var = _load_onnx_session(var_model.file_path, session_id, model_type="variance")
    var_inputs = {}
    for mi in sess_var.get_inputs():
        if mi.name == "encoder_out":
            var_inputs[mi.name] = encoder_out
        elif mi.name == "ph_dur":
            if "ph_dur" in body_inputs:
                var_inputs[mi.name] = _deserialize_input(body_inputs["ph_dur"], mi.type)
            elif "durations" in body_inputs:
                var_inputs[mi.name] = _deserialize_input(body_inputs["durations"], mi.type)
        elif mi.name == "pitch" and "pitch" in body_inputs:
            var_inputs[mi.name] = _deserialize_input(body_inputs["pitch"], mi.type)
        elif mi.name in ("steps", "speedup") and mi.name in body_inputs:
            var_inputs[mi.name] = _deserialize_input(body_inputs[mi.name], mi.type)
        elif mi.name in body_inputs:
            var_inputs[mi.name] = _deserialize_input(body_inputs[mi.name], mi.type)
        else:
            # 提供默认零值
            shape = [int(d.size if hasattr(d, 'size') else (d if isinstance(d, int) else 1))
                     for d in mi.shape]
            # 尝试从 pitch/f0 获取实际帧数填充动态维度
            if "pitch" in var_inputs:
                n_frames = var_inputs["pitch"].shape[1]
                for idx, d in enumerate(mi.shape):
                    if not isinstance(d, int) or d <= 0:
                        shape[idx] = n_frames
            elif "f0" in var_inputs:
                n_frames = var_inputs["f0"].shape[1]
                for idx, d in enumerate(mi.shape):
                    if not isinstance(d, int) or d <= 0:
                        shape[idx] = n_frames
            if mi.type == "tensor(float)":
                var_inputs[mi.name] = np.zeros(shape, dtype=np.float32)
            elif "int64" in mi.type:
                var_inputs[mi.name] = np.zeros(shape, dtype=np.int64)
            elif mi.type == "tensor(bool)":
                # retake 默认为 True（全部重新预测），其他 bool 默认为 False
                is_retake = mi.name == "retake"
                var_inputs[mi.name] = np.full(shape, is_retake, dtype=bool)
        # 矫正标量形状（客户端发 shape=[1] 但模型期望 shape=[]）
        if mi.name in var_inputs:
            var_inputs[mi.name] = _maybe_squeeze(var_inputs[mi.name], mi.shape)

    raw_var = sess_var.run(None, var_inputs)

    # 步骤4: 序列化输出
    result = {}
    for i, out in enumerate(sess_var.get_outputs()):
        if isinstance(raw_var[i], np.ndarray):
            result[out.name] = _serialize_output(out.name, raw_var[i], out.type)
    result["_chain"] = "variance"

    # 记录 variance 推理步数
    steps_var = body_inputs.get("steps", {}).get("int64_data", [0])
    step_count = int(steps_var[0]) if steps_var else 0
    LOGGER.info(f"[{session_id[:8]}] {singer_name} | Variance chain done (steps={step_count})")
    return result


@app.post("/inference_pitch")
async def inference_pitch(body: Dict, request: Request):
    """
    音高子模型推理 — 自动串联 linguistic encoder → pitch predictor。
    """
    session_id = _get_session_id(request)
    registry = _ensure_registry()
    from lib.model_registry import find_singer_by_name
    singer_name = body.get("singer", "")
    body_inputs: dict = body.get("inputs", {})

    sinfo = find_singer_by_name(registry, singer_name)
    if sinfo is None:
        raise HTTPException(status_code=404, detail=f"Singer '{singer_name}' not found")

    pit_model = sinfo.pitch_model
    if pit_model is None:
        raise HTTPException(status_code=404, detail=f"Pitch sub-model not found for singer '{singer_name}'")

    # 从 dspitch 目录查找该子模型自己的 linguistic ONNX
    pit_dir = pit_model.file_path.parent
    ling_path = None
    for f in pit_dir.glob("*.linguistic.onnx"):
        ling_path = f
        break
    if ling_path is None:
        raise HTTPException(status_code=404, detail=f"Pitch linguistic encoder not found in {pit_dir}")

    ling_model = OnnxModelInfo(
        model_type=ModelType.LINGUISTIC,
        file_path=ling_path.resolve(),
        relative_path=str(ling_path.relative_to(args.root_dir)),
    )

    # Linguistic encoder
    sess_ling = _load_onnx_session(ling_model.file_path, session_id, model_type="pitch")
    ling_inputs = {}
    for mi in sess_ling.get_inputs():
        if mi.name in body_inputs:
            ling_inputs[mi.name] = _deserialize_input(body_inputs[mi.name], mi.type)
        elif mi.name == "ph_dur" and "durations" in body_inputs:
            ling_inputs[mi.name] = _deserialize_input(body_inputs["durations"], mi.type)

    raw_ling = sess_ling.run(None, ling_inputs)
    encoder_out = None
    for i, out in enumerate(sess_ling.get_outputs()):
        if out.name == "encoder_out":
            encoder_out = raw_ling[i]
            break
    if encoder_out is None:
        encoder_out = raw_ling[0]

    # Pitch predictor
    sess_pit = _load_onnx_session(pit_model.file_path, session_id, model_type="pitch")
    pit_inputs = {}
    for mi in sess_pit.get_inputs():
        if mi.name == "encoder_out":
            pit_inputs[mi.name] = encoder_out
        elif mi.name in body_inputs:
            pit_inputs[mi.name] = _deserialize_input(body_inputs[mi.name], mi.type)
        else:
            shape = [int(d.size if hasattr(d, 'size') else (d if isinstance(d, int) else 1))
                     for d in mi.shape]
            # 从已有的输入中获取实际帧数/音符数填充动态维度
            if "pitch" in pit_inputs:
                n_frames = pit_inputs["pitch"].shape[1]
                for idx, d in enumerate(mi.shape):
                    if not isinstance(d, int) or d <= 0:
                        shape[idx] = n_frames
            elif "note_midi" in pit_inputs:
                n_notes = pit_inputs["note_midi"].shape[1]
                for idx, d in enumerate(mi.shape):
                    if not isinstance(d, int) or d <= 0:
                        shape[idx] = n_notes
            if mi.type == "tensor(float)":
                # expr 默认 1.0（中性值），其他默认 0
                fill_val = 1.0 if mi.name == "expr" else 0.0
                pit_inputs[mi.name] = np.full(shape, fill_val, dtype=np.float32)
            elif "int64" in mi.type:
                pit_inputs[mi.name] = np.zeros(shape, dtype=np.int64)
            elif mi.type == "tensor(bool)":
                # retake 默认为 True（全部重新预测），其他 bool 默认为 False
                is_retake = mi.name == "retake"
                pit_inputs[mi.name] = np.full(shape, is_retake, dtype=bool)
        # 矫正标量形状（客户端发 shape=[1] 但模型期望 shape=[]）
        if mi.name in pit_inputs:
            pit_inputs[mi.name] = _maybe_squeeze(pit_inputs[mi.name], mi.shape)

    raw_pit = sess_pit.run(None, pit_inputs)

    result = {}
    for i, out in enumerate(sess_pit.get_outputs()):
        if isinstance(raw_pit[i], np.ndarray):
            result[out.name] = _serialize_output(out.name, raw_pit[i], out.type)
    result["_chain"] = "pitch"

    # 记录 pitch 推理步数
    steps_pit = body_inputs.get("steps", {}).get("int64_data", [0])
    step_count = int(steps_pit[0]) if steps_pit else 0
    LOGGER.info(f"[{session_id[:8]}] {singer_name} | Pitch chain done (steps={step_count})")
    return result


# ====================================================================
# API 端点 — 资源管理
# ====================================================================

@app.post("/release")
async def release(body: Dict, request: Request):
    """释放指定模型（当前会话中）"""
    session_id = _get_session_id(request)
    resolved = process_path(body["model_path"])
    SESSION_MGR.release_model(session_id, resolved)
    return "ok"


@app.post("/release_session")
async def release_session(request: Request):
    """释放当前会话的所有模型缓存"""
    session_id = _get_session_id(request)
    SESSION_MGR.release_all(session_id)
    return {"status": "ok", "session": session_id[:8]}


@app.post("/release_all")
async def release_all(request: Request):
    """释放所有会话的所有模型缓存（管理用）"""
    # 只允许本地或带 token 的管理员调用
    total_onnx = 0
    total_jit = 0
    for sid in list(SESSION_MGR.active_sessions):
        total_onnx += len(SESSION_MGR._sessions.get(sid, {}))
        total_jit += len(SESSION_MGR._torch_sessions.get(sid, {}))
        SESSION_MGR.release_all(sid)
    return {"released_onnx": total_onnx, "released_jit": total_jit}


@app.get("/session_available")
async def session_available(request: Request):
    """
    检查当前 session 是否可用（不会创建新 session，不会驱逐旧 session）。
    客户端在开始推理前调用此端点决定是排队还是回退到本地。
    """
    session_id = request.headers.get("X-Session-Id", "")
    if not session_id:
        session_id = request.query_params.get("session_id", "")

    # 已有 session → 直接可用
    if session_id and session_id in SESSION_MGR._sessions:
        return {
            "available": True,
            "existing": True,
            "active_count": SESSION_MGR.active_count,
            "max_sessions": SESSION_MGR.max_sessions,
        }

    # 还有空位
    if SESSION_MGR.active_count < SESSION_MGR.max_sessions:
        return {
            "available": True,
            "existing": False,
            "active_count": SESSION_MGR.active_count,
            "max_sessions": SESSION_MGR.max_sessions,
        }

    # 已满
    return {
        "available": False,
        "existing": False,
        "active_count": SESSION_MGR.active_count,
        "max_sessions": SESSION_MGR.max_sessions,
    }


@app.get("/sessions")
async def list_sessions():
    """列出当前活跃会话"""
    return {
        "active_count": SESSION_MGR.active_count,
        "max_sessions": SESSION_MGR.max_sessions,
        "sessions": SESSION_MGR.active_sessions,
    }


@app.post("/rescan")
async def rescan():
    """重新扫描模型目录"""
    global REGISTRY
    REGISTRY = scan_all(args.root_dir)
    return {
        "status": "ok",
        "singers": len(REGISTRY.singers),
        "dependencies": len(REGISTRY.dependencies),
    }


# ====================================================================
# 入口
# ====================================================================

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host=args.host,
        port=args.port,
        workers=1,
        log_level="info",
    )
