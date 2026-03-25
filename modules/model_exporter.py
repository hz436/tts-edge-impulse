"""
model_exporter.py — Build + download TFLite model from Edge Impulse.

Improvements over baseline:
  • Streaming ZIP download (64 KB chunks) — no large in-memory buffer.
  • Download progress bar printed to stdout (visible in SSE log stream).
  • Reuses the same requests.Session pattern as EdgeImpulseClient.
  • Saves labels.json alongside the model file.
"""

from __future__ import annotations

import io
import json
import os
import time
import zipfile
from pathlib import Path
from typing import Optional

import numpy as np
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

STUDIO_URL     = "https://studio.edgeimpulse.com/v1/api"
POLL_INTERVAL  = 10       # seconds
DOWNLOAD_CHUNK = 65_536   # 64 KB


class ModelExporter:
    """Trigger the on-device model build and download the TFLite ZIP."""

    def __init__(self, api_key: str, project_id: str, output_dir: str = "exported_model") -> None:
        self.api_key    = api_key
        self.project_id = project_id
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

        self._session = requests.Session()
        retry = Retry(
            total=3,
            backoff_factor=0.5,
            status_forcelist=[429, 500, 502, 503, 504],
        )
        adapter = HTTPAdapter(max_retries=retry, pool_connections=4, pool_maxsize=4)
        self._session.mount("https://", adapter)
        self._session.headers.update({"x-api-key": api_key})

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def download(self, labels: list[str]) -> str:
        """
        Trigger a TFLite build job, wait, download the ZIP, extract
        model.tflite, and write labels.json.

        Returns the path to the extracted model.tflite.
        """
        # 1. Trigger build
        # type=zip  → deployment format (query param, required)
        # engine    → ML engine (request body, required)
        build_url = f"{STUDIO_URL}/{self.project_id}/jobs/build-ondevice-model?type=zip"
        print("[Export] Triggering TFLite model build …")
        resp = self._session.post(build_url, json={"engine": "tflite"}, timeout=30)
        self._raise(resp, "trigger build")
        data = resp.json()
        job_id         = data.get("id")
        deploy_version = data.get("deploymentVersion")
        if job_id is None:
            raise RuntimeError(f"[Export] Could not find job ID in build response: {data}")
        print(f"[Export] Build job started (jobId={job_id}, version={deploy_version}).")

        # 2. Wait for build to complete
        self._wait_for_job(job_id, label="Model build")

        # 3. Download ZIP (streaming) using the versioned historic deployment URL
        download_url = (
            f"{STUDIO_URL}/{self.project_id}/deployment/history/{deploy_version}/download"
        )
        tflite_path = self._stream_download_zip(download_url)

        # 4. Write labels.json
        labels_path = os.path.join(self.output_dir, "labels.json")
        with open(labels_path, "w", encoding="utf-8") as f:
            json.dump({"labels": labels}, f, ensure_ascii=False, indent=2)
        print(f"[Export] Labels written → {labels_path}")

        print(f"[Export] Model ready → {tflite_path}")
        return tflite_path

    def run_inference(self, wav_path: str, labels: list[str]) -> dict:
        """
        Run local TFLite inference on a single WAV file.

        Returns
        -------
        {
            "predicted_label": str,
            "confidence": float,
            "classification": {label: score, …}
        }
        """
        model_path = os.path.join(self.output_dir, "model.tflite")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found: {model_path}")

        features = _extract_mfcc(wav_path)
        raw      = _run_tflite(model_path, features)
        scores   = raw[0]

        if scores.min() < 0 or scores.max() > 1.01:
            probs = _softmax(scores).tolist()
        else:
            probs = scores.tolist()

        idx = int(np.argmax(probs))
        return {
            "predicted_label": labels[idx],
            "confidence":      float(probs[idx]),
            "classification":  dict(zip(labels, probs)),
        }

    # ------------------------------------------------------------------
    # Private
    # ------------------------------------------------------------------

    def _stream_download_zip(self, url: str) -> str:
        """Download the deployment ZIP in chunks, extract, return tflite path."""
        print(f"[Export] Downloading model ZIP …")
        resp = self._session.get(url, stream=True, timeout=120)
        self._raise(resp, "download model ZIP")

        total    = int(resp.headers.get("content-length", 0))
        received = 0
        buf      = io.BytesIO()

        for chunk in resp.iter_content(chunk_size=DOWNLOAD_CHUNK):
            buf.write(chunk)
            received += len(chunk)
            if total:
                pct = received / total * 100
                # \r keeps the line in place; SSE will receive the final update
                print(f"[Export] Download progress: {pct:.1f}%  "
                      f"({received // 1024} / {total // 1024} KB)", flush=True)

        print(f"[Export] Download complete ({received // 1024} KB).")
        buf.seek(0)

        # Extract model.tflite (and any other files) from the ZIP
        tflite_path: Optional[str] = None
        with zipfile.ZipFile(buf) as zf:
            for name in zf.namelist():
                zf.extract(name, self.output_dir)
                dest = os.path.join(self.output_dir, name)
                print(f"[Export] Extracted → {dest}")
                if name.endswith(".tflite"):
                    tflite_path = dest

        if tflite_path is None:
            raise RuntimeError(
                "No .tflite file found in the downloaded ZIP. "
                "Check Edge Impulse Studio for build errors."
            )
        return tflite_path

    def _wait_for_job(self, job_id: int, label: str = "Job") -> None:
        """Poll until a job completes or fails."""
        url = f"{STUDIO_URL}/{self.project_id}/jobs/{job_id}/status"
        print(f"[Export] Waiting for '{label}' (jobId={job_id}) …", flush=True)
        while True:
            time.sleep(POLL_INTERVAL)
            resp = self._session.get(url, timeout=15)
            self._raise(resp, f"poll job {job_id}")
            job = resp.json().get("job", {})

            # Shape A: {"job": {"status": "completed"}}
            status = job.get("status", "")
            if status in ("completed",):
                print(f"[Export] {label} finished successfully.")
                return
            if status in ("failed", "cancelled"):
                raise RuntimeError(
                    f"[Export] {label} ended with status '{status}'. "
                    "Check Edge Impulse Studio for details."
                )

            # Shape B: {"job": {"finished": "...", "finishedSuccessful": true}}
            if "finished" in job:
                if job.get("finishedSuccessful", False):
                    print(f"[Export] {label} finished successfully.")
                    return
                raise RuntimeError(
                    f"[Export] {label} failed. Check Edge Impulse Studio for details."
                )

            print(f"[Export] {label} running …")

    @staticmethod
    def _raise(resp: requests.Response, context: str) -> None:
        if not resp.ok:
            try:
                detail = resp.json()
            except Exception:
                detail = resp.text[:200]
            raise RuntimeError(
                f"[Export] {context} failed — HTTP {resp.status_code}: {detail}"
            )


# ---------------------------------------------------------------------------
# Local MFCC + TFLite helpers (mirrors test_model.py)
# ---------------------------------------------------------------------------

def _extract_mfcc(
    wav_path: str,
    sample_rate: int = 16000,
    num_coefficients: int = 13,
    num_filters: int = 40,
    frame_length: float = 0.02,
    frame_stride: float = 0.02,
    fft_length: int = 512,
    low_freq: int = 300,
    high_freq: int = 8000,
    window_size_ms: int = 1000,
) -> np.ndarray:
    import wave
    import scipy.fftpack

    with wave.open(wav_path, "rb") as wf:
        raw = wf.readframes(wf.getnframes())
    signal = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0

    window_samples = int(sample_rate * window_size_ms / 1000)
    if len(signal) < window_samples:
        signal = np.pad(signal, (0, window_samples - len(signal)))
    else:
        signal = signal[:window_samples]

    frame_len  = int(round(frame_length * sample_rate))
    frame_step = int(round(frame_stride * sample_rate))
    signal[1:] -= 0.97 * signal[:-1]

    num_frames = 1 + (len(signal) - frame_len) // frame_step
    indices    = (
        np.arange(frame_len)[None, :]
        + np.arange(num_frames)[:, None] * frame_step
    )
    frames  = signal[indices] * np.hamming(frame_len)
    mag     = np.abs(np.fft.rfft(frames, n=fft_length)) ** 2
    filters = _mel_filterbank(sample_rate, fft_length, num_filters, low_freq, high_freq)
    mel_e   = np.dot(mag, filters.T)
    mel_e   = np.where(mel_e == 0, np.finfo(float).eps, mel_e)
    log_mel = np.log(mel_e)
    mfcc    = scipy.fftpack.dct(log_mel, type=2, axis=1, norm="ortho")[:, :num_coefficients]
    return mfcc.flatten().astype(np.float32)


def _mel_filterbank(sample_rate, fft_length, num_filters, low_freq, high_freq):
    def hz_to_mel(hz):  return 2595 * np.log10(1 + hz / 700)
    def mel_to_hz(mel): return 700 * (10 ** (mel / 2595) - 1)

    mel_pts  = np.linspace(hz_to_mel(low_freq), hz_to_mel(high_freq), num_filters + 2)
    hz_pts   = mel_to_hz(mel_pts)
    bin_pts  = np.floor((fft_length + 1) * hz_pts / sample_rate).astype(int)
    filters  = np.zeros((num_filters, fft_length // 2 + 1))
    for m in range(1, num_filters + 1):
        lo, mid, hi = bin_pts[m - 1], bin_pts[m], bin_pts[m + 1]
        for k in range(lo,  mid): filters[m - 1, k] = (k - lo)  / (mid - lo)
        for k in range(mid, hi):  filters[m - 1, k] = (hi - k)  / (hi - mid)
    return filters


def _run_tflite(model_path: str, features: np.ndarray) -> np.ndarray:
    try:
        import tflite_runtime.interpreter as tflite
        Interpreter = tflite.Interpreter
    except ImportError:
        import tensorflow as tf
        Interpreter = tf.lite.Interpreter

    interp = Interpreter(model_path=model_path)
    interp.allocate_tensors()
    inp = interp.get_input_details()[0]
    out = interp.get_output_details()[0]
    interp.set_tensor(inp["index"], features.reshape(inp["shape"]).astype(inp["dtype"]))
    interp.invoke()
    return interp.get_tensor(out["index"])


def _softmax(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float64)
    e = np.exp(x - np.max(x))
    return (e / e.sum()).astype(np.float32)
