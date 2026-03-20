"""
TTS → Edge Impulse → TensorFlow Model Pipeline
================================================

Full pipeline:
  1. User enters text labels (words/phrases to classify)
  2. TTS generates WAV audio samples for each label
  3. WAV files are uploaded to Edge Impulse (training + testing split)
  4. A training job is triggered via the Edge Impulse API
  5. The trained TFLite model is downloaded to `exported_model/`

Usage:
    python main.py

    Or non-interactively:
    python main.py --labels "hello,goodbye,stop" --samples 15

Environment variables (see .env.example):
    EDGE_IMPULSE_API_KEY    required
    EDGE_IMPULSE_PROJECT_ID required
"""

import argparse
import os
import sys

from dotenv import load_dotenv

from modules import TTSGenerator, DatasetBuilder, EdgeImpulseClient, ModelExporter


def _is_chinese(text: str) -> bool:
    """Return True if text contains at least one Chinese character."""
    for ch in text:
        cp = ord(ch)
        if (
            0x4E00 <= cp <= 0x9FFF
            or 0x3400 <= cp <= 0x4DBF
            or 0x20000 <= cp <= 0x2A6DF
            or 0xF900 <= cp <= 0xFAFF
            or 0x3000 <= cp <= 0x303F
        ):
            return True
    return False


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

def load_config() -> dict:
    load_dotenv()
    api_key = os.getenv("EDGE_IMPULSE_API_KEY", "")
    project_id = os.getenv("EDGE_IMPULSE_PROJECT_ID", "")

    if not api_key or not project_id:
        print("[ERROR] EDGE_IMPULSE_API_KEY and EDGE_IMPULSE_PROJECT_ID must be set in .env")
        sys.exit(1)

    return {
        "api_key": api_key,
        "project_id": project_id,
        "dataset_dir": os.getenv("DATASET_DIR", "dataset"),
        "sample_rate": int(os.getenv("SAMPLE_RATE", "16000")),
        "duration": float(os.getenv("SAMPLE_DURATION", "1.5")),
        "samples_per_label": int(os.getenv("TTS_SAMPLES_PER_LABEL", "20")),
        "tts_volume": float(os.getenv("TTS_VOLUME", "1.0")),
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="TTS → Edge Impulse → TFLite Model Pipeline")
    parser.add_argument(
        "--labels",
        type=str,
        default="",
        help='Comma-separated labels (e.g. "hello,goodbye,stop"). '
             "Leave empty for interactive prompt.",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=0,
        help="Audio samples to generate per label (overrides TTS_SAMPLES_PER_LABEL).",
    )
    parser.add_argument(
        "--skip-tts",
        action="store_true",
        help="Skip TTS generation; upload files already in --dataset-dir.",
    )
    parser.add_argument(
        "--skip-upload",
        action="store_true",
        help="Skip upload; only trigger training (files already uploaded).",
    )
    parser.add_argument(
        "--skip-training",
        action="store_true",
        help="Skip training; only download the model (training already done).",
    )
    parser.add_argument(
        "--export-only",
        action="store_true",
        help="Only download the already-trained model.",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    cfg = load_config()

    # Override samples if provided via CLI
    if args.samples > 0:
        cfg["samples_per_label"] = args.samples

    # ---- Step 0: Collect labels ----------------------------------------
    if args.labels:
        labels = []
        for l in args.labels.split(","):
            l = l.strip()
            if not l:
                continue
            if not _is_chinese(l):
                print(f"[ERROR] '{l}' 不是中文。请输入中文文本。")
                sys.exit(1)
            labels.append(l)
    else:
        print("\n=== TTS → Edge Impulse → TFLite Pipeline ===")
        print("请输入您想训练的中文词语，每行一个，完成后按两次回车。\n")
        labels = []
        while True:
            label = input("请输入标签 (留空结束): ").strip()
            if not label:
                break
            if not _is_chinese(label):
                print("请输入中文文本。")
                continue
            labels.append(label)

    if not labels:
        print("[ERROR] 没有提供标签，程序退出。")
        sys.exit(1)

    print(f"\n[Config] Labels ({len(labels)}): {labels}")
    print(f"[Config] Project: {cfg['project_id']}")
    print(f"[Config] Samples per label: {cfg['samples_per_label']}")
    print(f"[Config] Dataset dir: {cfg['dataset_dir']}\n")

    # ---- Step 1: TTS generation ----------------------------------------
    if not args.skip_tts and not args.export_only:
        print("=== Step 1/4: Generating audio with TTS ===")
        tts = TTSGenerator(
            output_dir=cfg["dataset_dir"],
            sample_rate=cfg["sample_rate"],
            duration=cfg["duration"],
            samples_per_label=cfg["samples_per_label"],
            tts_volume=cfg["tts_volume"],
        )
        tts.generate(labels)
    else:
        print("=== Step 1/4: TTS generation skipped ===")

    # ---- Step 2: Build dataset split -----------------------------------
    print("=== Step 2/4: Splitting dataset (80/20 train/test) ===")
    builder = DatasetBuilder(dataset_dir=cfg["dataset_dir"])
    split = builder.build()

    # ---- Step 3: Upload + Train ----------------------------------------
    client = EdgeImpulseClient(api_key=cfg["api_key"], project_id=cfg["project_id"])

    if not args.skip_upload and not args.export_only:
        print("=== Step 3a/4: Uploading dataset to Edge Impulse ===")
        client.upload_dataset(split)

    if not args.skip_training and not args.export_only:
        print("\n=== Step 3b/4: Starting training job ===")
        job_id = client.start_training()
        success = client.wait_for_training(job_id)
        if not success:
            print("\n[ERROR] Training failed. Check Edge Impulse Studio for details.")
            sys.exit(1)

    # ---- Step 4: Export model ------------------------------------------
    print("\n=== Step 4/4: Downloading trained TFLite model ===")
    exporter = ModelExporter(
        api_key=cfg["api_key"],
        project_id=cfg["project_id"],
        output_dir="exported_model",
    )
    tflite_path = exporter.download(labels)

    # ---- Done ----------------------------------------------------------
    print("\n" + "=" * 50)
    print("Pipeline complete!")
    print(f"  Model:  {tflite_path}")
    print(f"  Labels: exported_model/labels.json")
    print()
    print("Model output format:")
    print("  - Raw TFLite tensor : float32 array  e.g. [[0.95, 0.03, 0.02]]")
    print("  - run_inference()   : {'predicted_label': 'hello',")
    print("                         'confidence': 0.95,")
    print("                         'classification': {'hello': 0.95, ...}}")
    print()
    print("To test inference on a WAV file:")
    print("  result = exporter.run_inference('your_audio.wav', labels)")
    print("=" * 50)


if __name__ == "__main__":
    main()
