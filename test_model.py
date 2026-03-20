"""
test_model.py — Offline accuracy test for the exported TFLite model.

Usage:
    python test_model.py                         # uses exported_model/ and dataset/
    python test_model.py --model path/to/model.tflite --dataset path/to/dataset
    python test_model.py --wav my_audio.wav      # single-file inference

What it does:
    1. Loads exported_model/model.tflite + exported_model/labels.json
    2. Rebuilds the same 80/20 test split used during upload (seed=42)
    3. Runs MFCC extraction + TFLite inference on every test WAV
    4. Reports per-label accuracy and overall accuracy
    5. Prints a confusion matrix
"""

import argparse
import json
import os
import sys

import numpy as np

# ---------------------------------------------------------------------------
# Feature extraction (must match Edge Impulse project settings)
# ---------------------------------------------------------------------------

def extract_mfcc(
    wav_path: str,
    sample_rate: int = 16000,
    num_coefficients: int = 13,
    num_filters: int = 40,
    frame_length: float = 0.02,
    frame_stride: float = 0.02,   # Edge Impulse default: 20ms non-overlapping frames
    fft_length: int = 512,
    low_freq: int = 300,
    high_freq: int = 8000,
    window_size_ms: int = 1000,   # Edge Impulse default window: 1 second
) -> np.ndarray:
    import wave
    import scipy.fftpack

    with wave.open(wav_path, "rb") as wf:
        raw = wf.readframes(wf.getnframes())
    signal = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0

    # Trim/pad to the model's expected window size
    window_samples = int(sample_rate * window_size_ms / 1000)
    if len(signal) < window_samples:
        signal = np.pad(signal, (0, window_samples - len(signal)))
    else:
        signal = signal[:window_samples]

    frame_len = int(round(frame_length * sample_rate))
    frame_step = int(round(frame_stride * sample_rate))

    signal[1:] -= 0.97 * signal[:-1]

    num_frames = 1 + (len(signal) - frame_len) // frame_step
    indices = (
        np.arange(frame_len)[None, :]
        + np.arange(num_frames)[:, None] * frame_step
    )
    frames = signal[indices] * np.hamming(frame_len)

    mag = np.abs(np.fft.rfft(frames, n=fft_length)) ** 2

    mel_filters = _mel_filterbank(sample_rate, fft_length, num_filters, low_freq, high_freq)
    mel_energy = np.dot(mag, mel_filters.T)
    mel_energy = np.where(mel_energy == 0, np.finfo(float).eps, mel_energy)
    log_mel = np.log(mel_energy)

    mfcc = scipy.fftpack.dct(log_mel, type=2, axis=1, norm="ortho")[:, :num_coefficients]
    return mfcc.flatten().astype(np.float32)


def _mel_filterbank(sample_rate, fft_length, num_filters, low_freq, high_freq):
    def hz_to_mel(hz):
        return 2595 * np.log10(1 + hz / 700)

    def mel_to_hz(mel):
        return 700 * (10 ** (mel / 2595) - 1)

    low_mel = hz_to_mel(low_freq)
    high_mel = hz_to_mel(high_freq)
    mel_points = np.linspace(low_mel, high_mel, num_filters + 2)
    hz_points = mel_to_hz(mel_points)
    bin_points = np.floor((fft_length + 1) * hz_points / sample_rate).astype(int)

    filters = np.zeros((num_filters, fft_length // 2 + 1))
    for m in range(1, num_filters + 1):
        f_m_minus = bin_points[m - 1]
        f_m = bin_points[m]
        f_m_plus = bin_points[m + 1]
        for k in range(f_m_minus, f_m):
            filters[m - 1, k] = (k - f_m_minus) / (f_m - f_m_minus)
        for k in range(f_m, f_m_plus):
            filters[m - 1, k] = (f_m_plus - k) / (f_m_plus - f_m)
    return filters


# ---------------------------------------------------------------------------
# TFLite inference
# ---------------------------------------------------------------------------

def run_tflite(model_path: str, features: np.ndarray) -> np.ndarray:
    try:
        import tflite_runtime.interpreter as tflite
        Interpreter = tflite.Interpreter
    except ImportError:
        import tensorflow as tf
        Interpreter = tf.lite.Interpreter

    interpreter = Interpreter(model_path=model_path)
    interpreter.allocate_tensors()

    inp = interpreter.get_input_details()[0]
    out = interpreter.get_output_details()[0]

    input_data = features.reshape(inp["shape"]).astype(inp["dtype"])
    interpreter.set_tensor(inp["index"], input_data)
    interpreter.invoke()
    return interpreter.get_tensor(out["index"])


def _softmax(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float64)
    e = np.exp(x - np.max(x))
    return (e / e.sum()).astype(np.float32)


def predict(model_path: str, wav_path: str, labels: list[str]) -> dict:
    features = extract_mfcc(wav_path)
    raw = run_tflite(model_path, features)
    raw_scores = raw[0]

    # Apply softmax if values are outside [0, 1] (raw logits from Edge Impulse)
    if raw_scores.min() < 0 or raw_scores.max() > 1.01:
        probs = _softmax(raw_scores).tolist()
    else:
        probs = raw_scores.tolist()

    classification = dict(zip(labels, probs))
    idx = int(np.argmax(probs))
    return {
        "predicted_label": labels[idx],
        "confidence": float(probs[idx]),
        "classification": classification,
    }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _safe_filename(text: str) -> str:
    """Same logic as TTSGenerator._safe_filename — Chinese chars → u{hex} stems."""
    parts = []
    for ch in text:
        if ch.isascii() and (ch.isalnum() or ch == "_"):
            parts.append(ch)
        elif ch == " ":
            parts.append("_")
        else:
            parts.append(f"u{ord(ch):04x}")
    return "".join(parts) or "label"


# ---------------------------------------------------------------------------
# Dataset test split (mirrors DatasetBuilder seed=42 / 80:20 split)
# ---------------------------------------------------------------------------

def build_test_split(
    dataset_dir: str,
    model_labels: list[str],
    train_split: float = 0.8,
    seed: int = 42,
) -> dict[str, list[str]]:
    """
    Returns {chinese_label: [test_wav_paths]} using the same split as main.py.

    Dataset folders use ASCII-encoded names (u5582u98df for 喂食).
    This function maps them back to the original Chinese label strings by
    comparing folder names against _safe_filename(label) for each model label.
    """
    import random
    random.seed(seed)

    # Build a reverse map: safe_name → original_label
    safe_to_label = {_safe_filename(lbl): lbl for lbl in model_labels}

    test_split: dict[str, list[str]] = {}

    folder_names = sorted([
        d for d in os.listdir(dataset_dir)
        if os.path.isdir(os.path.join(dataset_dir, d))
    ])

    for folder in folder_names:
        # Resolve to original label (may be the folder name itself for ASCII labels)
        original_label = safe_to_label.get(folder, folder)

        label_dir = os.path.join(dataset_dir, folder)
        wav_files = sorted([
            os.path.join(label_dir, f)
            for f in os.listdir(label_dir)
            if f.lower().endswith(".wav")
        ])
        if not wav_files:
            continue
        random.shuffle(wav_files)
        split_at = max(1, int(len(wav_files) * train_split))
        test_split[original_label] = wav_files[split_at:]

    return test_split


# ---------------------------------------------------------------------------
# Accuracy evaluation
# ---------------------------------------------------------------------------

def evaluate(model_path: str, labels: list[str], test_split: dict[str, list[str]]) -> None:
    total = 0
    correct = 0

    # confusion[true_label][predicted_label] = count
    confusion: dict[str, dict[str, int]] = {l: {l2: 0 for l2 in labels} for l in labels}

    per_label_correct: dict[str, int] = {l: 0 for l in labels}
    per_label_total: dict[str, int] = {l: 0 for l in labels}

    errors = []

    for true_label, wav_paths in test_split.items():
        if true_label not in confusion:
            # folder name not in model labels — skip
            print(f"[WARN] Label '{true_label}' not in model labels, skipping.")
            continue

        for wav_path in wav_paths:
            try:
                result = predict(model_path, wav_path, labels)
            except Exception as e:
                errors.append(f"  {os.path.basename(wav_path)}: {e}")
                continue

            predicted = result["predicted_label"]
            confidence = result["confidence"]

            confusion[true_label][predicted] += 1
            per_label_total[true_label] += 1
            total += 1

            if predicted == true_label:
                per_label_correct[true_label] += 1
                correct += 1
            else:
                errors.append(
                    f"  WRONG  {os.path.basename(wav_path)}"
                    f"  true={true_label}  predicted={predicted}  conf={confidence:.2f}"
                )

    # ------------------------------------------------------------------
    # Results
    # ------------------------------------------------------------------
    print("\n" + "=" * 56)
    print("  MODEL ACCURACY REPORT")
    print("=" * 56)

    print(f"\n  Overall accuracy: {correct}/{total} = ", end="")
    if total > 0:
        print(f"{correct / total * 100:.1f}%")
    else:
        print("N/A (no test files)")

    print("\n  Per-label accuracy:")
    for label in labels:
        n = per_label_total.get(label, 0)
        c = per_label_correct.get(label, 0)
        bar = "#" * c + "-" * (n - c)
        pct = f"{c / n * 100:.1f}%" if n > 0 else "N/A"
        print(f"    {label:<22}  {c:>2}/{n:<2}  {pct:>6}  [{bar}]")

    # Confusion matrix
    print(f"\n  Confusion matrix (rows=true, cols=predicted):")
    col_w = max(len(l) for l in labels) + 2
    header = " " * (col_w + 2) + "".join(f"{l:>{col_w}}" for l in labels)
    print(f"    {header}")
    for true_label in labels:
        row = f"    {true_label:<{col_w}}  "
        row += "".join(
            f"{confusion[true_label].get(pred, 0):>{col_w}}"
            for pred in labels
        )
        print(row)

    if errors:
        print(f"\n  Misclassifications / errors ({len(errors)}):")
        for e in errors[:20]:   # cap at 20 lines
            print(e)
        if len(errors) > 20:
            print(f"  ... and {len(errors) - 20} more")

    print("\n" + "=" * 56)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Test TFLite model accuracy")
    parser.add_argument("--model", default="exported_model/model.tflite",
                        help="Path to model.tflite (default: exported_model/model.tflite)")
    parser.add_argument("--labels", default="exported_model/labels.json",
                        help="Path to labels.json (default: exported_model/labels.json)")
    parser.add_argument("--dataset", default="dataset",
                        help="Dataset directory (default: dataset)")
    parser.add_argument("--wav", default="",
                        help="Run inference on a single WAV file instead of the full test set")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # Load labels
    if not os.path.exists(args.labels):
        print(f"[ERROR] Labels file not found: {args.labels}")
        sys.exit(1)
    with open(args.labels, encoding="utf-8") as f:
        labels = json.load(f)["labels"]
    print(f"[Test] Model labels: {labels}")

    if not os.path.exists(args.model):
        print(f"[ERROR] Model file not found: {args.model}")
        sys.exit(1)

    # Single-file mode
    if args.wav:
        if not os.path.exists(args.wav):
            print(f"[ERROR] WAV file not found: {args.wav}")
            sys.exit(1)
        result = predict(args.model, args.wav, labels)
        print(f"\n  File      : {args.wav}")
        print(f"  Prediction: {result['predicted_label']}  (confidence {result['confidence']:.2%})")
        print("  All scores:")
        for label, score in sorted(result["classification"].items(), key=lambda x: -x[1]):
            bar = "#" * int(score * 30)
            print(f"    {label:<22} {score:.4f}  {bar}")
        return

    # Full test set mode
    if not os.path.isdir(args.dataset):
        print(f"[ERROR] Dataset directory not found: {args.dataset}")
        sys.exit(1)

    print(f"[Test] Building test split from: {args.dataset}")
    test_split = build_test_split(args.dataset, labels)

    total_test = sum(len(v) for v in test_split.values())
    print(f"[Test] Test files: {total_test} across {len(test_split)} labels\n")

    if total_test == 0:
        print("[ERROR] No test files found. Make sure the dataset directory has label subdirectories with .wav files.")
        sys.exit(1)

    evaluate(args.model, labels, test_split)


if __name__ == "__main__":
    main()
