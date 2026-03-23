# TTS → Edge Impulse → TFLite 模型训练流水线

自动将中文文本标签转换为训练好的 TensorFlow Lite 关键词识别模型，整个过程通过 Edge Impulse API 完成，无需手动操作 Studio 界面（首次配置除外）。

支持两种输入模式：**文本（TTS）** 模式通过 Google TTS 自动生成音频；**录音** 模式可在浏览器中直接使用麦克风录制真实人声。

## 工作原理

```
用户输入标签（文本或麦克风录音）
        ↓
音频准备：TTS 自动生成 WAV 文件  或  浏览器录音（增强至目标数量）
        ↓
WAV 文件上传至 Edge Impulse（训练集 + 测试集自动划分）
        ↓
通过 Edge Impulse API 触发训练任务
        ↓
下载训练好的 TFLite 模型 → exported_model/model.tflite
```

## 前置条件

### 1. Python 3.10 及以上版本

### 2. 安装依赖

```bash
pip install -r requirements.txt
```

### 3. Edge Impulse 项目配置（仅需一次）

本流水线依赖已存在的 Edge Impulse 项目。首次运行前请完成以下配置：

1. 在 [studio.edgeimpulse.com](https://studio.edgeimpulse.com) 创建一个项目
2. 进入 **Impulse Design**，按如下配置：
   - 输入块：**Audio（1 秒窗口）**
   - 处理块：**MFCC**
   - 学习块：**Classification（Keras）**
3. 保存 Impulse
4. 记录 **Project ID**（在 Studio 页面 URL 中可以找到）

### 4. 环境变量配置

将 `.env.example` 复制为 `.env`，并填写您的凭据：

```bash
cp .env.example .env
```

```env
EDGE_IMPULSE_API_KEY=ei_xxxxxxxxxxxxxxxxxxxx
EDGE_IMPULSE_PROJECT_ID=123456
```

API Key 可在 Edge Impulse Studio → **Dashboard → Keys** 中获取。

## 使用方法

### Web UI（推荐）

```bash
python app.py
```

在浏览器中打开 `http://localhost:5000`。

```
┌──────────────────────────────────────────────┐
│  TTS → Edge Impulse Pipeline                 │
├──────────────────────────────────────────────┤
│  模式：  ● 文本（TTS）    ○ 录音             │
├──────────────────────────────────────────────┤
│  标签                             [+ 添加]   │
│  ┌────────────────────────────────────────┐  │
│  │  [开始     ]  [🎙 录音]  [✕]          │  │
│  │    ▶ 第1次  ▶ 第2次  ▶ 第3次  ✓      │  │
│  └────────────────────────────────────────┘  │
├──────────────────────────────────────────────┤
│  设置 ▼  样本数 | 时长 | 训练轮次 ...        │
├──────────────────────────────────────────────┤
│  [▶ 运行流水线]                              │
├──────────────────────────────────────────────┤
│  日志  ────────────────────────────────      │
│  === 第 1/4 步：处理音频 ...                 │
│  [Audio] '开始' → 增强 3 条录音 → 20 个文件 │
├──────────────────────────────────────────────┤
│  [⬇ 下载 model.tflite]  （完成后显示）      │
└──────────────────────────────────────────────┘
```

**TTS 模式** — 输入标签，点击运行。Google TTS 自动合成音频。

**录音模式** — 点击 🎙 录音，说出词语，点击 ⏹ 停止。每个标签至少录制 3 次后即可点击运行。系统会通过数据增强将录音数量自动填充至 `样本数/标签` 设定值。

### 命令行（备用方式）

#### 交互模式

```bash
python main.py
```

程序会提示您逐行输入中文标签：

```
请输入标签 (留空结束): 喂食
请输入标签 (留空结束): 开灯
请输入标签 (留空结束):
```

#### 非交互模式

```bash
python main.py --labels "喂食,开灯,关灯" --samples 30
```

#### 命令行参数

| 参数 | 说明 |
|---|---|
| `--labels` | 逗号分隔的标签列表 |
| `--samples` | 每个标签的样本数（覆盖 `TTS_SAMPLES_PER_LABEL`） |
| `--training-cycles` | 训练轮次（覆盖 `TRAINING_CYCLES`） |
| `--learning-rate` | 学习率（覆盖 `LEARNING_RATE`） |
| `--batch-size` | 批次大小（覆盖 `BATCH_SIZE`） |
| `--autotune` | 训练前运行 DSP 自动调优 |
| `--skip-tts` | 跳过 TTS 生成，使用 `dataset/` 中已有的 WAV 文件 |
| `--skip-upload` | 跳过上传，对已上传的数据直接触发训练 |
| `--skip-training` | 跳过训练，仅下载模型 |
| `--export-only` | 仅下载已训练好的模型 |

示例 — 训练已完成，仅重新下载模型：

```bash
python main.py --labels "喂食,开灯" --export-only
```

## 输出结果

运行成功后，输出目录如下：

```
exported_model/
├── model.tflite      # 可直接部署的 TFLite 模型
└── labels.json       # 标签元数据，例如 {"labels": ["喂食", "开灯"]}
```

### 模型输出格式

TFLite 模型输出形状为 `[1, N]` 的 float32 数组，N 为标签数量：

```python
原始输出：[[0.95, 0.03, 0.02]]
```

使用 `ModelExporter` 中的 `run_inference()` 辅助方法：

```python
from modules import ModelExporter

exporter = ModelExporter(api_key="...", project_id="...", output_dir="exported_model")
result = exporter.run_inference("audio.wav", labels=["喂食", "开灯", "关灯"])

# 返回结果：
# {
#   "predicted_label": "喂食",
#   "confidence": 0.95,
#   "classification": {"喂食": 0.95, "开灯": 0.03, "关灯": 0.02}
# }
```

## 测试模型

对本地测试集进行准确率评估：

```bash
python test_model.py
```

对单个 WAV 文件进行推理：

```bash
python test_model.py --wav "path/to/audio.wav"
```

自定义路径：

```bash
python test_model.py --model exported_model/model.tflite --labels exported_model/labels.json --dataset dataset
```

> **注意：** 本地测试准确率可能低于 Edge Impulse Studio 显示的结果（约 50% vs 70%），原因是本地 MFCC 特征提取与 Edge Impulse 内部 DSP 存在细微差异。**以 Studio 上显示的准确率为准。**

## 参数配置

所有配置项均可在 `.env` 文件中覆盖：

| 变量名 | 默认值 | 说明 |
|---|---|---|
| `EDGE_IMPULSE_API_KEY` | — | 必填。您的 Edge Impulse API Key |
| `EDGE_IMPULSE_PROJECT_ID` | — | 必填。您的 Edge Impulse 项目 ID |
| `TTS_SAMPLES_PER_LABEL` | `20` | 每个标签生成的音频样本数 |
| `SAMPLE_RATE` | `16000` | 音频采样率（Hz） |
| `SAMPLE_DURATION` | `1.5` | 每段音频时长（秒） |
| `DATASET_DIR` | `dataset` | 生成的 WAV 文件保存目录 |
| `TTS_VOLUME` | `1.0` | TTS 输出音量（0.0–1.0） |
| `TRAINING_CYCLES` | `100` | 训练轮次 |
| `LEARNING_RATE` | `0.0005` | 优化器学习率 |
| `BATCH_SIZE` | `32` | 训练批次大小 |
| `AUTOTUNE_DSP` | `false` | 训练前是否运行 DSP 自动调优 |

## 项目结构

```
tts_to_model_pipeline/
├── app.py                         # Flask Web UI 服务器
├── main.py                        # 命令行流水线入口
├── test_model.py                  # 离线准确率评估脚本
├── requirements.txt
├── .env.example
├── templates/
│   └── index.html                 # 单页 Web UI
├── modules/
│   ├── tts_generator.py           # gTTS 音频生成、数据增强、录音支持
│   ├── dataset_builder.py         # 80/20 训练集/测试集划分
│   ├── edge_impulse_client.py     # 上传与训练 API 调用
│   └── model_exporter.py          # 构建与下载 TFLite 模型
├── dataset/                       # 生成的 WAV 文件（已加入 .gitignore）
└── exported_model/                # 输出模型文件（已加入 .gitignore）
```

## 提升准确率

| 方法 | 预期效果 |
|---|---|
| 将 `TTS_SAMPLES_PER_LABEL` 增加至 50 及以上 | 中等 — 增加增强变体的多样性 |
| 提高 `TRAINING_CYCLES`（默认 100） | 中等 |
| 使用录音模式录制真实人声 | 显著提升 |
| 每个标签录制 5–10 次（而非最少 3 次） | 中等偏高 |

## API 调用说明

### gTTS — Google 翻译 TTS
| 调用 | 时机 |
|---|---|
| `gTTS(text, lang).save()` | 每个标签调用两次（普通话 + 台湾口音） |

- **免费，无需 API Key**
- 非官方 API（与 translate.google.com 使用同一后端）—— 短时间内大量请求可能触发限流，代码内已有 0.3 秒延迟缓解此问题
- 风险：Google 可能随时修改该接口，无服务保障

---

### Edge Impulse 数据上传 API — `https://ingestion.edgeimpulse.com/api`
| 调用 | 接口 | 时机 |
|---|---|---|
| 上传 WAV 文件 | `POST /api/{training\|testing}/files` | 每个 WAV 文件上传时调用一次 |

- **免费版限制：每个项目约 3 小时音频存储空间**
- 每次运行会先清除旧数据再重新上传，存储用量不会持续增长

---

### Edge Impulse Studio API — `https://studio.edgeimpulse.com/v1/api`
| 调用 | 接口 | 时机 |
|---|---|---|
| 获取学习块 ID | `GET /{projectId}/impulse` | 每次训练前调用一次 |
| 启动训练任务 | `POST /{projectId}/jobs/train/keras/{learnId}` | 每次训练调用一次 |
| 轮询训练状态 | `GET /{projectId}/jobs/{jobId}/status` | 每 10 秒一次，直到训练完成 |
| 触发模型构建 | `POST /{projectId}/jobs/build-ondevice-model?type=zip` | 每次导出调用一次 |
| 轮询构建状态 | `GET /{projectId}/jobs/{jobId}/status` | 每 10 秒一次，直到构建完成 |
| 下载模型 ZIP | `GET /{projectId}/deployment/history/{version}/download` | 每次导出调用一次 |

- **免费版限制：最多 4 个项目，每月约 20 分钟训练计算时间**
- 模型构建/下载及状态轮询接口无明显限制
- 付费版（$20/月）可取消计算时间限制 —— 若需在生产环境中频繁重训建议升级

---

### 汇总对比

| API | 需要 Key | 是否免费 | 需注意的限制 |
|---|---|---|---|
| gTTS | 否 | 是 | 非官方接口，避免短时间大量调用 |
| Edge Impulse 数据上传 | 是 | 是 | 每个项目约 3 小时音频存储 |
| Edge Impulse 训练 | 是 | 是 | 每月约 20 分钟计算时间 |
| Edge Impulse 模型构建/下载 | 是 | 是 | 无明显限制 |
| Edge Impulse 状态轮询 | 是 | 是 | 无明显限制 |

## 依赖要求

- Python 3.10+
- 网络连接（gTTS 需调用 Google TTS API；Edge Impulse API 需联网）
- Edge Impulse 免费账户即可用于开发测试
