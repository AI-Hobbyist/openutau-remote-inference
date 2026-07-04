"""
dsconfig_parser.py — DiffSinger 配置文件解析器

解析 OpenUtau DiffSinger 模型的各类 YAML/JSON 配置文件：
  - dsconfig.yaml    (声学模型/方差子模型配置)
  - vocoder.yaml     (声码器配置)
  - oudep.yaml       (依赖声明)
  - character.yaml   (歌手元数据)
  - config.json      (依赖详细配置)
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Any

import yaml


# ====================================================================
# 声学模型配置 (dsconfig.yaml — 歌手根目录)
# ====================================================================

@dataclass
class AcousticConfig:
    """歌手主 dsconfig.yaml 解析结果"""
    # 模型文件
    acoustic: str = ""
    phonemes: str = ""
    languages: str = ""

    # 基础参数
    hidden_size: int = 256
    sample_rate: int = 44100
    hop_size: int = 512
    win_size: int = 2048
    fft_size: int = 2048
    num_mel_bins: int = 128
    mel_fmin: float = 40.0
    mel_fmax: float = 16000.0
    mel_base: str = "e"
    mel_scale: str = "slaney"

    # 声音特征开关
    use_key_shift_embed: bool = False   # gender 输入
    use_speed_embed: bool = False       # velocity 输入
    use_breathiness_embed: bool = False
    use_voicing_embed: bool = False
    use_tension_embed: bool = False
    use_energy_embed: bool = False
    use_lang_id: bool = False

    # 扩散相关
    use_variable_depth: bool = False
    max_depth: float = 0.0
    use_continuous_acceleration: bool = True

    # 声码器引用
    vocoder: str = ""

    # 多说话人
    speakers: List[str] = field(default_factory=list)

    # 原始完整字典
    raw: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_path(cls, path: Path) -> "AcousticConfig":
        """从 dsconfig.yaml 文件路径加载"""
        if not path.exists():
            return cls()
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        return cls.from_dict(data)

    @classmethod
    def from_dict(cls, data: dict) -> "AcousticConfig":
        return cls(
            acoustic=data.get("acoustic", ""),
            phonemes=data.get("phonemes", ""),
            languages=data.get("languages", ""),
            hidden_size=data.get("hidden_size", 256),
            sample_rate=data.get("sample_rate", 44100),
            hop_size=data.get("hop_size", 512),
            win_size=data.get("win_size", 2048),
            fft_size=data.get("fft_size", 2048),
            num_mel_bins=data.get("num_mel_bins", 128),
            mel_fmin=data.get("mel_fmin", 40.0),
            mel_fmax=data.get("mel_fmax", 16000.0),
            mel_base=data.get("mel_base", "e"),
            mel_scale=data.get("mel_scale", "slaney"),
            use_key_shift_embed=data.get("use_key_shift_embed", False),
            use_speed_embed=data.get("use_speed_embed", False),
            use_breathiness_embed=data.get("use_breathiness_embed", False),
            use_voicing_embed=data.get("use_voicing_embed", False),
            use_tension_embed=data.get("use_tension_embed", False),
            use_energy_embed=data.get("use_energy_embed", False),
            use_lang_id=data.get("use_lang_id", False),
            use_variable_depth=data.get("use_variable_depth", False),
            max_depth=data.get("max_depth", 0.0),
            use_continuous_acceleration=data.get("use_continuous_acceleration", True),
            vocoder=data.get("vocoder", ""),
            speakers=data.get("speakers", []),
            raw=data,
        )


# ====================================================================
# 方差子模型配置 (dsconfig.yaml — dsdur/dspitch/dsvariance 目录)
# ====================================================================

@dataclass
class VarianceSubConfig:
    """方差子模型 dsconfig.yaml 解析结果"""
    # 模型文件
    linguistic: str = ""
    dur: str = ""
    pitch: str = ""
    variance: str = ""
    phonemes: str = ""
    languages: str = ""

    # 基础参数
    hidden_size: int = 256
    sample_rate: int = 44100
    hop_size: int = 512

    # 功能开关
    predict_dur: bool = False
    use_expr: bool = False
    use_note_rest: bool = False
    use_lang_id: bool = False
    use_continuous_acceleration: bool = True

    # 方差预测列表
    predict_energy: bool = False
    predict_breathiness: bool = False
    predict_voicing: bool = False
    predict_tension: bool = False

    # 多说话人
    speakers: List[str] = field(default_factory=list)

    # 原始完整字典
    raw: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_path(cls, path: Path) -> "VarianceSubConfig":
        if not path.exists():
            return cls()
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        return cls.from_dict(data)

    @classmethod
    def from_dict(cls, data: dict) -> "VarianceSubConfig":
        return cls(
            linguistic=data.get("linguistic", ""),
            dur=data.get("dur", ""),
            pitch=data.get("pitch", ""),
            variance=data.get("variance", ""),
            phonemes=data.get("phonemes", ""),
            languages=data.get("languages", ""),
            hidden_size=data.get("hidden_size", 256),
            sample_rate=data.get("sample_rate", 44100),
            hop_size=data.get("hop_size", 512),
            predict_dur=data.get("predict_dur", False),
            use_expr=data.get("use_expr", False),
            use_note_rest=data.get("use_note_rest", False),
            use_lang_id=data.get("use_lang_id", False),
            use_continuous_acceleration=data.get("use_continuous_acceleration", True),
            predict_energy=data.get("predict_energy", False),
            predict_breathiness=data.get("predict_breathiness", False),
            predict_voicing=data.get("predict_voicing", False),
            predict_tension=data.get("predict_tension", False),
            speakers=data.get("speakers", []),
            raw=data,
        )


# ====================================================================
# 声码器配置 (vocoder.yaml)
# ====================================================================

@dataclass
class VocoderConfig:
    """vocoder.yaml 解析结果"""
    name: str = ""
    model: str = ""
    model_type: str = "onnx"  # onnx / jit

    sample_rate: int = 44100
    hop_size: int = 512
    win_size: int = 2048
    fft_size: int = 2048
    num_mel_bins: int = 128
    mel_fmin: float = 40.0
    mel_fmax: float = 16000.0
    mel_base: str = "e"
    mel_scale: str = "slaney"

    pitch_controllable: bool = False
    force_on_cpu: bool = False

    raw: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_path(cls, path: Path) -> "VocoderConfig":
        if not path.exists():
            return cls()
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        return cls.from_dict(data)

    @classmethod
    def from_dict(cls, data: dict) -> "VocoderConfig":
        return cls(
            name=data.get("name", ""),
            model=data.get("model", ""),
            model_type=data.get("model_type", "onnx"),
            sample_rate=data.get("sample_rate", 44100),
            hop_size=data.get("hop_size", 512),
            win_size=data.get("win_size", 2048),
            fft_size=data.get("fft_size", 2048),
            num_mel_bins=data.get("num_mel_bins", 128),
            mel_fmin=data.get("mel_fmin", 40.0),
            mel_fmax=data.get("mel_fmax", 16000.0),
            mel_base=data.get("mel_base", "e"),
            mel_scale=data.get("mel_scale", "slaney"),
            pitch_controllable=data.get("pitch_controllable", False),
            force_on_cpu=data.get("force_on_cpu", False),
            raw=data,
        )


# ====================================================================
# 依赖声明 (oudep.yaml)
# ====================================================================

@dataclass
class DependencyConfig:
    """oudep.yaml 解析结果"""
    id: str = ""
    version: str = ""
    name: str = ""
    description: str = ""
    class_name: str = ""

    raw: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_path(cls, path: Path) -> "DependencyConfig":
        if not path.exists():
            return cls()
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        return cls.from_dict(data)

    @classmethod
    def from_dict(cls, data: dict) -> "DependencyConfig":
        return cls(
            id=data.get("id", ""),
            version=data.get("version", ""),
            name=data.get("name", ""),
            description=data.get("description", ""),
            class_name=data.get("class", ""),
            raw=data,
        )


# ====================================================================
# 模型类型枚举
# ====================================================================

class ModelType:
    """DiffSinger 模型类型"""
    ACOUSTIC = "acoustic"       # 声学模型 (FS2 + Diffusion)
    LINGUISTIC = "linguistic"   # 语言编码器
    DURATION = "dur"            # 时长预测器
    PITCH = "pitch"             # 音高预测器
    VARIANCE = "variance"       # 多方差预测器
    VOCODER = "vocoder"         # 声码器
    DEPENDENCY = "dependency"   # 依赖 (GAME/RMVPE)

    @classmethod
    def infer_from_path(cls, model_path: Path) -> str:
        """根据文件名推测模型类型"""
        name = model_path.name.lower()
        parent_name = model_path.parent.name.lower()

        # Dependencies 目录下的属于依赖
        if "dependencies" in model_path.parts:
            return cls.DEPENDENCY

        # dsvocoder 目录下的
        if parent_name == "dsvocoder":
            return cls.VOCODER

        # 方差子模型
        if parent_name in ("dsdur",):
            if ".linguistic." in name:
                return cls.LINGUISTIC
            return cls.DURATION
        if parent_name in ("dspitch",):
            if ".linguistic." in name:
                return cls.LINGUISTIC
            return cls.PITCH
        if parent_name in ("dsvariance",):
            if ".linguistic." in name:
                return cls.LINGUISTIC
            return cls.VARIANCE

        # 声学主模型 (根目录下的 .onnx)
        if "_aco." in name:
            return cls.ACOUSTIC

        return "unknown"

    @classmethod
    def sub_model_dir(cls, model_type: str) -> str:
        """模型类型对应的子目录名"""
        mapping = {
            cls.DURATION: "dsdur",
            cls.PITCH: "dspitch",
            cls.VARIANCE: "dsvariance",
            cls.VOCODER: "dsvocoder",
        }
        return mapping.get(model_type, "")


# ====================================================================
# 便捷解析函数
# ====================================================================

def parse_model_config(base_dir: Path) -> Optional[AcousticConfig]:
    """尝试在 base_dir 下查找并解析 dsconfig.yaml (声学主配置)"""
    candidates = [
        base_dir / "dsconfig.yaml",
        base_dir.parent / "dsconfig.yaml",
    ]
    for path in candidates:
        if path.exists():
            return AcousticConfig.from_path(path)
    return None


def parse_vocoder_config(vocoder_dir: Path) -> Optional[VocoderConfig]:
    """在声码器目录下查找并解析 vocoder.yaml"""
    path = vocoder_dir / "vocoder.yaml"
    if path.exists():
        return VocoderConfig.from_path(path)
    return None


def parse_dependency_config(dep_dir: Path) -> Optional[DependencyConfig]:
    """在依赖目录下查找并解析 oudep.yaml"""
    path = dep_dir / "oudep.yaml"
    if path.exists():
        return DependencyConfig.from_path(path)
    return None


def parse_sub_model_config(sub_dir: Path) -> Optional[VarianceSubConfig]:
    """在方差子模型目录 (dsdur/dspitch/dsvariance) 下解析 dsconfig.yaml"""
    path = sub_dir / "dsconfig.yaml"
    if path.exists():
        return VarianceSubConfig.from_path(path)
    return None
