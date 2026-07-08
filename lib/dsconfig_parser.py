"""DiffSinger configuration parsers."""










from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Any

import yaml


# ====================================================================
# 澹板妯″瀷閰嶇疆 (dsconfig.yaml 鈥?姝屾墜鏍圭洰褰?
# ====================================================================

@dataclass
class AcousticConfig:
    """Acoustic dsconfig.yaml data."""
    # 妯″瀷鏂囦欢
    acoustic: str = ""
    phonemes: str = ""
    languages: str = ""

    # 鍩虹鍙傛暟
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

    # 澹伴煶鐗瑰緛寮€鍏?
    use_key_shift_embed: bool = False   # gender 杈撳叆
    use_speed_embed: bool = False       # velocity 杈撳叆
    use_breathiness_embed: bool = False
    use_voicing_embed: bool = False
    use_tension_embed: bool = False
    use_energy_embed: bool = False
    use_lang_id: bool = False

    # 鎵╂暎鐩稿叧
    use_variable_depth: bool = False
    max_depth: float = 0.0
    use_continuous_acceleration: bool = True

    # 澹扮爜鍣ㄥ紩鐢?
    vocoder: str = ""

    # 澶氳璇濅汉
    speakers: List[str] = field(default_factory=list)

    # 鍘熷瀹屾暣瀛楀吀
    raw: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_path(cls, path: Path) -> "AcousticConfig":
        """Load an acoustic config from a dsconfig.yaml file."""
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
# 鏂瑰樊瀛愭ā鍨嬮厤缃?(dsconfig.yaml 鈥?dsdur/dspitch/dsvariance 鐩綍)
# ====================================================================

@dataclass
class VarianceSubConfig:
    """Variance sub-model dsconfig.yaml data."""
    # 妯″瀷鏂囦欢
    linguistic: str = ""
    dur: str = ""
    pitch: str = ""
    variance: str = ""
    phonemes: str = ""
    languages: str = ""

    # 鍩虹鍙傛暟
    hidden_size: int = 256
    sample_rate: int = 44100
    hop_size: int = 512

    # 鍔熻兘寮€鍏?
    predict_dur: bool = False
    use_expr: bool = False
    use_note_rest: bool = False
    use_lang_id: bool = False
    use_continuous_acceleration: bool = True

    # 鏂瑰樊棰勬祴鍒楄〃
    predict_energy: bool = False
    predict_breathiness: bool = False
    predict_voicing: bool = False
    predict_tension: bool = False

    # 澶氳璇濅汉
    speakers: List[str] = field(default_factory=list)

    # 鍘熷瀹屾暣瀛楀吀
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
# 澹扮爜鍣ㄩ厤缃?(vocoder.yaml)
# ====================================================================

@dataclass
class VocoderConfig:
    """Vocoder config data."""
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
# 妯″瀷绫诲瀷鏋氫妇
# ====================================================================

class ModelType:
    """DiffSinger model type names."""
    ACOUSTIC = "acoustic"       # 澹板妯″瀷 (FS2 + Diffusion)
    LINGUISTIC = "linguistic"   # 璇█缂栫爜鍣?
    DURATION = "dur"            # 鏃堕暱棰勬祴鍣?
    PITCH = "pitch"             # 闊抽珮棰勬祴鍣?
    VARIANCE = "variance"       # 澶氭柟宸娴嬪櫒
    VOCODER = "vocoder"         # 澹扮爜鍣?

    @classmethod
    def infer_from_path(cls, model_path: Path) -> str:
        """Infer model type from a model path."""
        name = model_path.name.lower()
        parent_name = model_path.parent.name.lower()


        # dsvocoder 鐩綍涓嬬殑
        if parent_name == "dsvocoder":
            return cls.VOCODER

        # 鏂瑰樊瀛愭ā鍨?
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

        # 澹板涓绘ā鍨?(鏍圭洰褰曚笅鐨?.onnx)
        if "_aco." in name:
            return cls.ACOUSTIC

        return "unknown"

    @classmethod
    def sub_model_dir(cls, model_type: str) -> str:
        """Return the subdirectory name for a model type."""
        mapping = {
            cls.DURATION: "dsdur",
            cls.PITCH: "dspitch",
            cls.VARIANCE: "dsvariance",
            cls.VOCODER: "dsvocoder",
        }
        return mapping.get(model_type, "")


# ====================================================================
# 渚挎嵎瑙ｆ瀽鍑芥暟
# ====================================================================

def parse_model_config(base_dir: Path) -> Optional[AcousticConfig]:
    """Find and parse acoustic dsconfig.yaml."""
    candidates = [
        base_dir / "dsconfig.yaml",
        base_dir.parent / "dsconfig.yaml",
    ]
    for path in candidates:
        if path.exists():
            return AcousticConfig.from_path(path)
    return None


def parse_vocoder_config(vocoder_dir: Path) -> Optional[VocoderConfig]:
    """Find and parse vocoder.yaml."""
    path = vocoder_dir / "vocoder.yaml"
    if path.exists():
        return VocoderConfig.from_path(path)
    return None



def parse_sub_model_config(sub_dir: Path) -> Optional[VarianceSubConfig]:
    """Find and parse sub-model dsconfig.yaml."""
    path = sub_dir / "dsconfig.yaml"
    if path.exists():
        return VarianceSubConfig.from_path(path)
    return None
