# TTS → Edge Impulse → TFLite Pipeline

Automatically converts Chinese text labels into a trained TensorFlow Lite keyword-spotting model via Edge Impulse — no manual Studio interaction required after the initial one-time setup.

## How it works

```
User inputs Chinese text labels
        ↓
TTS generates WAV audio samples (gTTS, zh-CN + zh-TW accents + augmentation)
        ↓
WAV files uploaded to Edge Impulse (training + testing split)
        ↓
Training job triggered via Edge Impulse API
        ↓
Trained TFLite model downloaded → exported_model/model.tflite
```

## Prerequisites

### 1. Python 3.10+

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Edge Impulse project (one-time manual setup)

The pipeline uses an existing Edge Impulse project. Before running for the first time:

1. Create a project at [studio.edgeimpulse.com](https://studio.edgeimpulse.com)
2. Go to **Impulse Design** and configure:
   - Input: **Audio (1 second window)**
   - Processing block: **MFCC**
   - Learning block: **Classification (Keras)**
3. Save the impulse
4. Note your **Project ID** (visible in the Studio URL)

### 4. Environment variables

Copy `.env.example` to `.env` and fill in your credentials:

```bash
cp .env.example .env
```

```env
EDGE_IMPULSE_API_KEY=ei_xxxxxxxxxxxxxxxxxxxx
EDGE_IMPULSE_PROJECT_ID=123456
```

Get your API key from Edge Impulse Studio → **Dashboard → Keys**.

## Usage

### Interactive mode

```bash
python main.py
```

You will be prompted to enter Chinese labels one per line:

```
请输入标签 (留空结束): 喂食
请输入标签 (留空结束): 开灯
请输入标签 (留空结束):
```

### Non-interactive mode

```bash
python main.py --labels "喂食,开灯,关灯" --samples 30
```

### Skip flags (useful for resuming a failed run)

| Flag | Description |
|---|---|
| `--skip-tts` | Skip TTS generation, use existing WAV files in `dataset/` |
| `--skip-upload` | Skip upload, trigger training on already-uploaded data |
| `--skip-training` | Skip training, only download the model |
| `--export-only` | Only download the already-trained model |

Example — re-download model after training is already done:

```bash
python main.py --labels "喂食,开灯" --export-only
```

## Output

After a successful run:

```
exported_model/
├── model.tflite      # TFLite model ready for deployment
└── labels.json       # Label metadata  e.g. {"labels": ["喂食", "开灯"]}
```

### Model output format

The TFLite model outputs a float32 array of shape `[1, N]` where N is the number of labels:

```python
raw output: [[0.95, 0.03, 0.02]]
```

Using the `run_inference()` helper in `ModelExporter`:

```python
from modules import ModelExporter

exporter = ModelExporter(api_key="...", project_id="...", output_dir="exported_model")
result = exporter.run_inference("audio.wav", labels=["喂食", "开灯", "关灯"])

# result:
# {
#   "predicted_label": "喂食",
#   "confidence": 0.95,
#   "classification": {"喂食": 0.95, "开灯": 0.03, "关灯": 0.02}
# }
```

## Testing the model

Run accuracy evaluation on the local test split:

```bash
python test_model.py
```

Run inference on a single WAV file:

```bash
python test_model.py --wav "path/to/audio.wav"
```

Custom paths:

```bash
python test_model.py --model exported_model/model.tflite --labels exported_model/labels.json --dataset dataset
```

> **Note:** Local accuracy may read lower than Edge Impulse Studio (~50% vs ~70%) due to small differences between our local MFCC implementation and Edge Impulse's internal DSP. The Studio accuracy is the authoritative number.

## Configuration

All settings can be overridden in `.env`:

| Variable | Default | Description |
|---|---|---|
| `EDGE_IMPULSE_API_KEY` | — | Required. Your Edge Impulse API key |
| `EDGE_IMPULSE_PROJECT_ID` | — | Required. Your Edge Impulse project ID |
| `TTS_SAMPLES_PER_LABEL` | `20` | Audio samples to generate per label |
| `SAMPLE_RATE` | `16000` | Audio sample rate in Hz |
| `SAMPLE_DURATION` | `1.5` | Length of each audio clip in seconds |
| `DATASET_DIR` | `dataset` | Directory to save generated WAV files |
| `TTS_VOLUME` | `1.0` | TTS output volume (0.0–1.0) |

## Project structure

```
tts_to_model_pipeline/
├── main.py                        # Pipeline entry point
├── test_model.py                  # Offline accuracy evaluation
├── requirements.txt
├── .env.example
├── modules/
│   ├── tts_generator.py           # gTTS audio generation + augmentation
│   ├── dataset_builder.py         # 80/20 train/test split
│   ├── edge_impulse_client.py     # Upload + training API calls
│   └── model_exporter.py          # Build + download TFLite model
├── dataset/                       # Generated WAV files (git-ignored)
└── exported_model/                # Output model files (git-ignored)
```

## Growing model design

Each pipeline run **adds** new data to the existing Edge Impulse project rather than replacing it. This means:

- Previously trained labels are retained
- The model is retrained on the full accumulated dataset each run
- New labels can be added incrementally over time

## Improving accuracy

| Action | Expected impact |
|---|---|
| Increase `TTS_SAMPLES_PER_LABEL` to 50+ | Medium — more augmentation variety |
| Increase `trainingCycles` in `edge_impulse_client.py` (default 100) | Medium |
| Use real human voice recordings instead of TTS | High |

## API reference

### gTTS — Google Translate TTS
| Call | When |
|---|---|
| `gTTS(text, lang).save()` | Twice per label (zh-CN + zh-TW accent) |

- **Free, no API key required**
- Unofficial API (same backend as translate.google.com) — Google can rate-limit if too many rapid requests; mitigated by the built-in 0.3s delay
- Risk: no SLA — Google could change it without notice

---

### Edge Impulse Ingestion API — `https://ingestion.edgeimpulse.com/api`
| Call | Endpoint | When |
|---|---|---|
| Upload WAV file | `POST /api/{training\|testing}/files` | Once per WAV file |

- **Free tier: ~3 hours of audio storage per project**
- Each run appends data (growing model) — monitor storage usage over time

---

### Edge Impulse Studio API — `https://studio.edgeimpulse.com/v1/api`
| Call | Endpoint | When |
|---|---|---|
| Get learn block ID | `GET /{projectId}/impulse` | Once per training run |
| Start training job | `POST /{projectId}/jobs/train/keras/{learnId}` | Once per training run |
| Poll training status | `GET /{projectId}/jobs/{jobId}/status` | Every 10s until done |
| Trigger model build | `POST /{projectId}/jobs/build-ondevice-model?type=zip` | Once per export |
| Poll build status | `GET /{projectId}/jobs/{jobId}/status` | Every 10s until done |
| Download model ZIP | `GET /{projectId}/deployment/history/{version}/download` | Once per export |

- **Free tier: 4 projects max, ~20 minutes of training compute/month**
- Build/download and status polling calls have no meaningful limit
- Paid plan ($20/month) removes the compute cap — recommended if retraining frequently in production

---

### Summary

| API | Key Required | Free | Limit to Watch |
|---|---|---|---|
| gTTS | No | Yes | Unofficial — avoid rapid bulk calls |
| Edge Impulse Ingestion | Yes | Yes | ~3 hours audio storage per project |
| Edge Impulse Training | Yes | Yes | ~20 min compute/month |
| Edge Impulse Build/Download | Yes | Yes | No meaningful limit |
| Edge Impulse Status Polling | Yes | Yes | No meaningful limit |

## Requirements

- Python 3.10+
- Internet connection (gTTS calls Google TTS API; Edge Impulse API calls)
- Edge Impulse free plan is sufficient for development
