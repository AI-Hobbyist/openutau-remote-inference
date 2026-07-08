"""
model_registry.py — DiffSinger 模型注册与目录扫描

自动发现并注册所有可用的 singer 声库和依赖模块。
根据 OpenUtau_Models 的目录结构约定进行识别：
  Singers/{SingerName}-DiffSinger/{SingerName}/
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from lib.dsconfig_parser import (
    AcousticConfig,
    ModelType,
    VocoderConfig,
    VarianceSubConfig,
    parse_model_config,
    parse_vocoder_config,
    parse_sub_model_config,
)

LOGGER = logging.getLogger("openutau-remote-inference.model_registry")


# ====================================================================
# 注册信息
# ====================================================================

@dataclass
class OnnxModelInfo:
    """单个 ONNX 模型的注册信息"""
    model_type: str                    # ModelType 枚举值
    file_path: Path                    # ONNX 文件绝对路径
    relative_path: str                 # 相对于 root_dir 的路径
    inputs: List[dict] = field(default_factory=list)   # 输入签名 (延迟加载)
    outputs: List[dict] = field(default_factory=list)  # 输出签名 (延迟加载)
    config: Optional[object] = None    # 关联配置对象


@dataclass
class SingerInfo:
    """一个歌手的完整注册信息"""
    name: str                          # 歌手名称 (如 fu2_ning2_na4)
    display_name: str                  # 显示名称
    base_dir: Path                     # Singer 模型目录

    # 声学模型
    acoustic_model: Optional[OnnxModelInfo] = None
    acoustic_config: Optional[AcousticConfig] = None

    # 方差子模型
    linguistic_model: Optional[OnnxModelInfo] = None   # 共享语言编码器
    dur_model: Optional[OnnxModelInfo] = None
    pitch_model: Optional[OnnxModelInfo] = None
    variance_model: Optional[OnnxModelInfo] = None

    # 子模型配置
    dur_config: Optional[VarianceSubConfig] = None
    pitch_config: Optional[VarianceSubConfig] = None
    variance_config: Optional[VarianceSubConfig] = None

    # 声码器
    vocoder_model: Optional[OnnxModelInfo] = None
    vocoder_config: Optional[VocoderConfig] = None


@dataclass
class ModelRegistry:
    """全局模型注册表"""
    root_dir: Path
    singers: Dict[str, SingerInfo] = field(default_factory=dict)


# ====================================================================
# 扫描器
# ====================================================================

def scan_singers(root_dir: Path) -> Dict[str, SingerInfo]:
    """扫描 Singers/ 目录，识别所有声库"""
    singers: Dict[str, SingerInfo] = {}
    singers_dir = root_dir / "Singers"

    if not singers_dir.exists():
        LOGGER.warning(f"Singers directory not found: {singers_dir}")
        return singers

    for actual_singer_dir in _iter_singer_dirs(singers_dir):
        relative_display = actual_singer_dir.relative_to(singers_dir).as_posix()

        singer_name = actual_singer_dir.name
        display_name = relative_display

        info = SingerInfo(
            name=singer_name,
            display_name=display_name,
            base_dir=actual_singer_dir,
        )

        # 1. 解析声学主配置
        aco_config = parse_model_config(actual_singer_dir)
        info.acoustic_config = aco_config or AcousticConfig()

        # 2. 注册声学 ONNX — 主目录下任意 .onnx 文件即认为存在
        if info.acoustic_model is None:
            for f in sorted(actual_singer_dir.glob("*.onnx")):
                info.acoustic_model = OnnxModelInfo(
                    model_type=ModelType.ACOUSTIC,
                    file_path=f.resolve(),
                    relative_path=str(f.relative_to(root_dir)),
                    config=aco_config,
                )
                LOGGER.info(f"  Acoustic model: {f.name}")
                break

        # 3. 扫描方差子模型
        _scan_sub_model(info, actual_singer_dir / "dsdur", "dur", root_dir)
        _scan_sub_model(info, actual_singer_dir / "dspitch", "pitch", root_dir)
        _scan_sub_model(info, actual_singer_dir / "dsvariance", "variance", root_dir)

        # 4. 扫描声码器
        _scan_vocoder(info, actual_singer_dir / "dsvocoder", root_dir)

        registry_key = singer_name
        if registry_key in singers:
            registry_key = relative_display
            LOGGER.warning(
                f"Duplicate singer name '{singer_name}', using registry key '{registry_key}'"
            )
        singers[registry_key] = info
        LOGGER.info(
            f"Registered singer '{display_name}': "
            f"acoustic={'✓' if info.acoustic_model else '✗'}, "
            f"dur={'✓' if info.dur_model else '✗'}, "
            f"pitch={'✓' if info.pitch_model else '✗'}, "
            f"variance={'✓' if info.variance_model else '✗'}, "
            f"vocoder={'✓' if info.vocoder_model else '✗'}"
        )

    return singers


def _iter_singer_dirs(folder: Path) -> List[Path]:
    """递归扫描所有实际声库目录，支持 Singers/dir1/声库 等多层嵌套。"""
    result: List[Path] = []
    if not folder.exists() or not folder.is_dir():
        return result
    if _is_singer_dir(folder):
        return [folder]
    for child in sorted(folder.iterdir()):
        if not child.is_dir() or child.name == "backup":
            continue
        result.extend(_iter_singer_dirs(child))
    return result


def _is_singer_dir(folder: Path) -> bool:
    """判断一个目录是否为歌手模型目录（含有歌手特征文件）"""
    if not folder.is_dir():
        return False
    # 排除已知的子模型/非歌手目录名
    if folder.name in ("dsdur", "dspitch", "dsvariance", "dsvocoder", "backup"):
        return False
    # OpenUtau 歌手元数据文件
    if (folder / "character.yaml").exists() or (folder / "character.txt").exists():
        return True
    # 含有 dsconfig.yaml 且根目录下含有 .onnx 模型文件
    if (folder / "dsconfig.yaml").exists():
        if any(f.is_file() and f.suffix == ".onnx" for f in folder.iterdir()):
            return True
    return False


def _find_actual_singer_dir(singer_folder: Path) -> Optional[Path]:
    """找到实际的歌手模型目录，兼容各种嵌套层级

    支持以下目录结构:
      1) Singers/{Name}/                           — 平铺结构，无嵌套
      2) Singers/{Name}-DiffSinger/{Name}/         — 一层嵌套
      3) Singers/Category/{Name}-DiffSinger/{Name}/ — 多层嵌套
    """
    # 情况1：当前目录本身就是歌手目录
    if _is_singer_dir(singer_folder):
        return singer_folder

    # 情况2：递归查找子目录（兼容任意嵌套层级）
    for child in sorted(singer_folder.iterdir()):
        if not child.is_dir() or child.name == "backup":
            continue
        result = _find_actual_singer_dir(child)
        if result is not None:
            return result

    return None


def _scan_sub_model(info: SingerInfo, sub_dir: Path, sub_type: str, root_dir: Path):
    """扫描方差子模型目录 — 目录存在即认为有该模块"""
    if not sub_dir.exists():
        return

    # 子目录中存在即可标记为有对应模块，路径留空由 /inference 按需查找
    placeholder = list(sub_dir.glob("*.onnx"))
    model_file = placeholder[0] if placeholder else None

    # 优先选择 predictor 模型 (*.variance.onnx / *.pitch.onnx / *.dur.onnx)
    # 而非 *.linguistic.onnx（字母序可能在前面）
    if model_file and "linguistic" in model_file.name:
        for f in placeholder:
            if "linguistic" not in f.name:
                model_file = f
                break

    model_key_map = {
        "dur": ("dur", ModelType.DURATION),
        "pitch": ("pitch", ModelType.PITCH),
        "variance": ("variance", ModelType.VARIANCE),
    }
    if sub_type in model_key_map:
        _cfg_key, mtype = model_key_map[sub_type]
        if model_file and model_file.exists():
            onnx_info = OnnxModelInfo(
                model_type=mtype,
                file_path=model_file.resolve(),
                relative_path=str(model_file.relative_to(root_dir)),
            )
            if sub_type == "dur":
                info.dur_model = onnx_info
            elif sub_type == "pitch":
                info.pitch_model = onnx_info
            elif sub_type == "variance":
                info.variance_model = onnx_info


def _scan_vocoder(info: SingerInfo, vocoder_dir: Path, root_dir: Path):
    """扫描声码器目录"""
    if not vocoder_dir.exists():
        return

    # 解析 vocoder.yaml
    vocoder_config = parse_vocoder_config(vocoder_dir)
    info.vocoder_config = vocoder_config or VocoderConfig()

    # 寻找 ONNX/JIT 模型
    # dsvocoder 目录存在即认为有声码器模块，取第一个 ONNX 或 JIT 文件
    model_path = None
    for f in vocoder_dir.glob("*.onnx"):
        model_path = f
        break
    if model_path is None:
        for f in vocoder_dir.glob("*.jit"):
            model_path = f
            break

    if model_path and model_path.exists():
        info.vocoder_model = OnnxModelInfo(
            model_type=ModelType.VOCODER,
            file_path=model_path.resolve(),
            relative_path=str(model_path.relative_to(root_dir)),
            config=vocoder_config or VocoderConfig(),
        )


# ====================================================================
# 查找 API
# ====================================================================

def find_model_by_relative_path(
    registry: ModelRegistry,
    relative_path: str,
) -> Optional[OnnxModelInfo]:
    """根据相对路径查找 ONNX 模型信息"""
    path = Path(relative_path)
    parts = path.parts

    # 处理 Singers/xxx/.../xxx.onnx
    if len(parts) >= 2 and parts[0] == "Singers":
        # 需要匹配到具体的 singer
        singer_name = None
        for sname, sinfo in registry.singers.items():
            rel = sinfo.acoustic_model.relative_path if sinfo.acoustic_model else ""
            if sname in str(path) or rel and rel == str(path):
                singer_name = sname
                break

        if singer_name is None:
            # 尝试在路径中匹配 singer 名
            for sname in registry.singers:
                if sname in str(path):
                    singer_name = sname
                    break

        if singer_name and singer_name in registry.singers:
            sinfo = registry.singers[singer_name]
            path_str = str(path)
            # 逐一匹配已知模型
            candidates = [
                ("acoustic_model", sinfo.acoustic_model),
                ("linguistic_model", sinfo.linguistic_model),
                ("dur_model", sinfo.dur_model),
                ("pitch_model", sinfo.pitch_model),
                ("variance_model", sinfo.variance_model),
                ("vocoder_model", sinfo.vocoder_model),
            ]
            for _, model_info in candidates:
                if model_info and model_info.relative_path == path_str:
                    return model_info
                if model_info and model_info.file_path.name == path.name:
                    return model_info

    return None


def find_singer_by_name(
    registry: ModelRegistry,
    name: str,
) -> Optional[SingerInfo]:
    """根据歌手名称查找"""
    # 精确匹配
    if name in registry.singers:
        return registry.singers[name]
    # 模糊匹配
    name_lower = name.lower()
    for sname, sinfo in registry.singers.items():
        if name_lower in sname.lower():
            return sinfo
        if name_lower in sinfo.display_name.lower():
            return sinfo
    return None


def find_singer_by_model_path(
    registry: ModelRegistry,
    model_path: Path,
) -> Optional[Tuple[str, SingerInfo]]:
    """根据模型文件路径查找所属歌手"""
    resolved = model_path.resolve()
    for sname, sinfo in registry.singers.items():
        if sinfo.base_dir in resolved.parents or sinfo.base_dir == resolved.parent:
            return sname, sinfo
    return None


# ====================================================================
# 一次性扫描
# ====================================================================

def scan_all(root_dir: Path) -> ModelRegistry:
    """扫描根目录下所有模型"""
    LOGGER.info(f"Scanning model registry at: {root_dir}")
    singers = scan_singers(root_dir)
    registry = ModelRegistry(
        root_dir=root_dir,
        singers=singers,
    )
    LOGGER.info(
        f"Registry scan complete: "
        f"{len(singers)} singer(s)"
    )
    return registry
