# :satisfied: OpenUtau Remote Inference Server v2

> 基于 FastAPI 的 DiffSinger 远程推理服务器，支持声学模型、方差子模型（时长/音高/音色）、声码器（ONNX/JIT）的远程推理。

将 OpenUtau 的 DiffSinger 模型部署到远程 GPU 服务器上推理，本地 OpenUtau 仅作为前端使用。需要配合 [修改版 OpenUtau](https://github.com/AI-Hobbyist/OpenUtau/tree/diffs-remote-new) 使用。

---

## :zap: 快速开始

### 1. 安装依赖

根据你的硬件选择安装：

```bash
# NVIDIA GPU (CUDA)
pip install -r requirements-cuda.txt

# AMD GPU (DirectML)
pip install -r requirements-dml.txt
```

> **注意**: `requirements-cuda.txt` 包含 `onnxruntime-gpu` 和 `tensorrt`，`requirements-dml.txt` 包含 `onnxruntime-directml`。两种环境下都会回退到 CPU 推理。

### 2. 准备模型

在服务器上创建模型根目录，复制 `Singers/` 中的模型文件（保持相对目录结构）。例如：

```
/path/to/models/
├── Singers/
│   └── {SingerName}-DiffSinger/
│       └── {SingerName}/
│           ├── dsconfig.yaml
│           ├── acoustic.onnx
│           ├── dsdur/              # 时长子模型
│           ├── dspitch/            # 音高子模型
│           ├── dsvariance/         # 音色方差子模型
│           └── dsvocoder/          # 声码器
└── Dependencies/
    ├── game/                       # GAME MIDI 提取器
    └── rmvpe/                      # RMVPE 音高提取
```

### 3. 启动服务器

```bash
python main.py [-d <模型根目录>] [--host 0.0.0.0] [--port 7889] [--max_sessions 10] [--precision fp32]
```

**参数说明**:

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `-d` / `--root_dir` | `.`（当前目录） | 模型根目录（包含 `Singers/`、`Dependencies/`） |
| `--host` | `0.0.0.0` | 绑定地址 |
| `--port` | `7889` | 绑定端口 |
| `--max_sessions` | `10` | 最大并发客户端会话数 |
| `--precision` | `fp32` | 推理精度：`fp32` / `fp16` / `int8` |

---

## :bookmark_tabs: API 文档

### 基础端点

| 方法 | 路径 | 说明 |
|------|------|------|
| `GET` | `/ping` | 健康检查，返回 `"pong"` |
| `GET` | `/exists?model_path={path}` | 检查模型文件是否存在 |
| `GET` | `/registry` | 获取完整的模型注册信息（所有歌手、依赖） |
| `GET` | `/singer_info?singer_name={name}` | 获取指定歌手的详细模型信息 |

### ONNX 元信息

| 方法 | 路径 | 说明 |
|------|------|------|
| `GET` | `/onnx_info/inputs?model_path={path}` | 获取 ONNX 模型的输入名称列表 |
| `GET` | `/onnx_info/outputs?model_path={path}` | 获取 ONNX 模型的输出名称列表 |
| `GET` | `/onnx_info/details?model_path={path}` | 获取 ONNX 模型的完整输入/输出签名 |

### 推理端点

| 方法 | 路径 | 说明 |
|------|------|------|
| `POST` | `/inference` | **通用推理** — 自动识别模型类型并路由到专用端点（向后兼容） |
| `POST` | `/inference_acoustic` | **声学模型推理** — 缺失的可选输入自动零填充 |
| `POST` | `/inference_acoustic_batch` | **批量声学推理** — 一次请求推理多个片段 |
| `POST` | `/inference_vocoder` | **声码器推理** — 支持 ONNX 和 JIT (DDSP) 声码器 |
| `POST` | `/inference_vocoder_batch` | **批量声码器推理** — 支持 ONNX 和 JIT，一次推理多个片段 |
| `POST` | `/inference_variance` | **方差子模型推理** — 自动串联 linguistic encoder → variance predictor |
| `POST` | `/inference_dependency` | **依赖模块推理** — GAME / RMVPE 等依赖模型 |
| `POST` | `/inference_chain` | **链式推理** — 自动串联 linguistic → dur/pitch/variance 多步推理 |

### 推理请求格式

所有推理端点通用请求结构：

```json
{
    "model_path": "Singers/{SingerName}/{model}.onnx",
    "inputs": {
        "tokens":       { "type": "tensor(int64)",  "shape": [1, N], "int64_data": [...] },
        "durations":    { "type": "tensor(int64)",  "shape": [1, N], "int64_data": [...] },
        "f0":           { "type": "tensor(float)",  "shape": [1, M], "float_data": [...] },
        "gender":       { "type": "tensor(float)",  "shape": [1, M], "float_data": [...] },
        "velocity":     { "type": "tensor(float)",  "shape": [1, M], "float_data": [...] },
        "breathiness":  { "type": "tensor(float)",  "shape": [1, M], "float_data": [...] },
        "voicing":      { "type": "tensor(float)",  "shape": [1, M], "float_data": [...] },
        "tension":      { "type": "tensor(float)",  "shape": [1, M], "float_data": [...] },
        "steps":        { "type": "tensor(int32)",  "shape": [1],    "int32_data": [10] }
    }
}
```

> **可选输入自动填充**: 声学模型中，未提供的可选输入（`gender`、`velocity`、`breathiness`、`voicing`、`tension`、`depth`、`spk_embed`、`languages`）会自动零填充，无需客户端额外处理。

### 批量推理格式

```json
{
    "model_path": "Singers/{SingerName}/{model}.onnx",
    "segments": [
        { "inputs": { ... } },
        { "inputs": { ... } }
    ]
}
```

### 链式推理格式

```json
{
    "singer": "{SingerName}",
    "pipeline": ["linguistic", "dur", "pitch", "variance"],
    "inputs": {
        "tokens":     { ... },
        "ph_midi":    { ... },
        "ph_dur":     { ... },
        "steps":      { ... }
    }
}
```

---

## :star: 核心特性

### 自动模型发现与注册

启动时自动扫描 `Singers/` 和 `Dependencies/` 目录，识别所有声库和依赖模块，支持两种目录结构：

- `Singers/{Name}-DiffSinger/{Name}/` — OpenUtau 标准结构
- `Singers/{Name}/` — 简化结构

通过 `GET /registry` 可查看完整的注册信息。

### 智能计算提供器选择

自动检测并按优先级选择可用的 ONNX Runtime 提供器：

```
TensorRT (fp16/int8) → CUDA → DirectML → CPU
```

提供器异常时自动 fallback，确保服务可用性。

### 多精度推理

支持 `--precision` 参数指定推理精度（`fp32`/`fp16`/`int8`），在 TensorRT 提供器下可启用 TensorRT 缓存加速。

### 会话管理

每个客户端通过 `X-Session-Id` 头或 `session_id` 查询参数维持独立会话，会话内模型缓存复用。超时（默认 1 小时）自动释放，减少内存占用。

### DDSP / JIT 声码器支持

支持 `vocoder.yaml` 中 `model_type: jit` 的 DDSP TorchScript 声码器。将 JIT 文件命名为 `{model}.jit`，在 `vocoder.yaml` 中添加一行 `model_type: jit` 即可使用。

### 安全校验

`process_path()` 函数严格校验路径，防止路径穿越攻击，确保所有模型文件访问均在根目录范围内。

---

## :file_folder: 项目结构

```
openutau-remote-inference/
├── main.py                     # FastAPI 服务器主入口（含推理核心、会话管理、API 端点）
├── lib/
│   ├── __init__.py
│   ├── dsconfig_parser.py      # 配置文件解析器（dsconfig.yaml / vocoder.yaml / oudep.yaml）
│   ├── model_registry.py       # 模型注册与目录扫描（自动发现歌手和依赖）
│   └── nvSTFT.py               # STFT 工具函数
├── docs/
│   └── ONNX_EXPORT_ANALYSIS.md # ONNX 导出与远程推理适配分析文档
├── requirements-cuda.txt       # NVIDIA GPU 依赖
├── requirements-dml.txt        # AMD GPU 依赖
├── LICENSE                     # Apache 2.0
└── README.md
```

---

## :memo: 配置文件

### `dsconfig.yaml`（声学主配置）

```yaml
acoustic: {model}.onnx
phonemes: {model}.phonemes.json
languages: {model}.languages.json
hidden_size: 256
sample_rate: 44100
hop_size: 512
num_mel_bins: 128
use_key_shift_embed: true    # gender 输入
use_speed_embed: true        # velocity 输入
use_breathiness_embed: true
use_voicing_embed: true
use_tension_embed: true
use_variable_depth: true
max_depth: 0.6
speakers: [...]              # 多说话人（可选）
```

### `vocoder.yaml`（声码器配置）

```yaml
name: nsf_hifigan_44.1k
model: nsf_hifigan.onnx
sample_rate: 44100
hop_size: 512
num_mel_bins: 128
pitch_controllable: true
model_type: onnx             # 可选: onnx / jit
```

## :hamburger: 动机

- 在 OpenUtau 中使用 DDSP 声码器
- 直接在远程 GPU 服务器上使用 `torch.compile` 优化检查点
- 将计算密集型推理从本地转移到远程服务器

## :bangbang: 免责声明

这不是一个开箱即用的打包仓库，你需要自行搭建 Python 环境并编译 [修改版 OpenUtau](https://github.com/AI-Hobbyist/OpenUtau/tree/diffs-remote-new)。后续会根据反馈持续完善。

## :yum: 致谢

- 原始思路：[fishaudio/openutau-remote-host](https://github.com/fishaudio/openutau-remote-host)
- OpenUtau 修改参考：[fishaudio/OpenUtau](https://github.com/fishaudio/OpenUtau)
- 上游参考：[hrukalive/openutau-remote-inference](https://github.com/hrukalive/openutau-remote-inference)