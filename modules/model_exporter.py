"""
ModelExporter: Downloads the trained TFLite model from Edge Impulse
and (optionally) converts it to a full TensorFlow SavedModel for deployment.

Edge Impulse produces TFLite files. This module:
  1. Downloads the .tflite from the Edge Impulse deployment API
  2. Saves it locally as `model.tflite`
  3. Optionally wraps it in a TF Lite Interpreter and saves label metadata
     so the model is immediately runnable.

Output explained
----------------
The TFLite model output tensor is a **float32 numpy array** of shape [1, N]
where N is the number of labels.  Example for labels ["hello", "stop", "noise"]:

    raw output: [[0.95, 0.03, 0.02]]

To get a human-readable dict, this module provides `run_inference()` which
returns:
    {"hello": 0.95, "stop": 0.03, "noise": 0.02}

along with a `predicted_label` and `confidence` field.
"""

import io
import os
import json
import time
import zipfile

import numpy as np
import requests


class ModelExporter:
    BASE_URL = "https://studio.edgeimpulse.com/v1/api"

    def __init__(self, api_key: str, project_id: str, output_dir: str = "exported_model"):
        self.api_key = api_key
        self.project_id = project_id
        self.output_dir = output_dir
        self._headers = {"x-api-key": self.api_key}
        os.makedirs(self.output_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # Download — three-step: find target → build → download
    # ------------------------------------------------------------------

    def download(self, labels: list[str]) -> str:
        """
        Build and download the trained TFLite model from Edge Impulse.

        Flow (per official spec):
          1. POST /api/{projectId}/jobs/build-ondevice-model → trigger build job
          2. Response contains deploymentVersion (or poll job until finished)
          3. GET  /api/{projectId}/deployment/history/{deploymentVersion}/download
        """
        deployment_version = self._build_deployment()
        return self._download_build(deployment_version, labels)

    def _build_deployment(self, retries: int = 3) -> str:
        """
        Trigger a TFLite on-device model build job.
        Returns the deploymentVersion string for the download step.

        Response: {"id": <jobId>, "deploymentVersion": <versionId>}
        If deploymentVersion is absent, poll the job until finished.
        """
        print("[Export] Triggering deployment build (type=zip, engine=tflite) ...")
        url = f"{self.BASE_URL}/{self.project_id}/jobs/build-ondevice-model"

        resp = None
        for attempt in range(1, retries + 1):
            try:
                resp = requests.post(
                    url,
                    params={"type": "zip"},
                    headers={**self._headers, "Content-Type": "application/json"},
                    json={"engine": "tflite", "modelType": "int8"},
                    timeout=60,
                )
                break
            except (requests.exceptions.SSLError, requests.exceptions.ConnectionError) as e:
                if attempt < retries:
                    wait = 2 ** attempt
                    print(f"[Export] Connection error (attempt {attempt}/{retries}), retrying in {wait}s ...")
                    time.sleep(wait)
                else:
                    raise RuntimeError(
                        f"[Edge Impulse] Build request failed after {retries} attempts.\n"
                        f"  Error: {e}\n"
                        f"  Check your internet connection."
                    )

        if not resp.ok:
            try:
                detail = resp.json().get("error", resp.text[:200])
            except Exception:
                detail = resp.text[:200]
            raise RuntimeError(
                f"[Edge Impulse] Build request failed — HTTP {resp.status_code}\n"
                f"  Detail: {detail}"
            )

        body = resp.json()
        print(f"[Export] Build response: {body}")

        job_id = str(body.get("id", ""))
        if not job_id:
            raise RuntimeError(
                f"[Edge Impulse] Build response had no job id.\n"
                f"  Full response: {body}"
            )

        # deploymentVersion may be pre-assigned in the response; pass it as a hint
        deployment_version_hint = str(body.get("deploymentVersion", ""))

        # Always wait for the async job to complete before downloading
        return self._wait_for_build_job(job_id, deployment_version_hint=deployment_version_hint)

    def _wait_for_build_job(
        self,
        job_id: str,
        timeout: int = 300,
        poll_interval: int = 10,
        deployment_version_hint: str = "",
    ) -> str:
        """Poll a build job until complete and return the resulting deploymentVersion."""
        url = f"{self.BASE_URL}/{self.project_id}/jobs/{job_id}/status"
        deadline = time.time() + timeout
        print(f"[Export] Waiting for build job {job_id} ...")

        while time.time() < deadline:
            resp = None
            for attempt in range(1, 4):
                try:
                    resp = requests.get(url, headers=self._headers, timeout=30)
                    break
                except (requests.exceptions.SSLError, requests.exceptions.ConnectionError) as e:
                    if attempt < 3:
                        print(f"[Export] Connection error polling job (attempt {attempt}/3), retrying in {2**attempt}s ...")
                        time.sleep(2 ** attempt)
                    else:
                        print(f"[Export] Connection error polling job, will retry on next poll interval — {e}")
                        time.sleep(poll_interval)
                        continue

            if resp is None:
                continue
            resp.raise_for_status()
            job = resp.json().get("job", {})

            if job.get("finishedSuccessful"):
                # Prefer version from job status; fall back to the hint from the build response
                deployment_version = str(
                    job.get("deploymentVersion", "")
                    or job.get("artifactId", "")
                    or deployment_version_hint
                )
                print(f"[Export] Build job completed. Deployment version: {deployment_version}")
                return deployment_version

            if job.get("finished") and not job.get("finishedSuccessful"):
                raise RuntimeError(
                    f"[Edge Impulse] Build job {job_id} failed.\n"
                    f"  See logs: https://studio.edgeimpulse.com/studio/{self.project_id}/jobs/{job_id}"
                )

            elapsed = int(time.time() - (deadline - timeout))
            print(f"[Export]   ... build running (elapsed {elapsed}s)")
            time.sleep(poll_interval)

        raise RuntimeError(f"[Edge Impulse] Build job {job_id} timed out after {timeout}s.")

    def _download_build(self, deployment_version: str, labels: list[str], retries: int = 3) -> str:
        """Download the built deployment ZIP and extract the .tflite file."""
        print(f"[Export] Downloading deployment version {deployment_version} ...")
        url = f"{self.BASE_URL}/{self.project_id}/deployment/history/{deployment_version}/download"

        resp = None
        for attempt in range(1, retries + 1):
            try:
                resp = requests.get(
                    url,
                    headers={**self._headers, "Accept": "application/zip"},
                    timeout=120,
                )
                break
            except (requests.exceptions.SSLError, requests.exceptions.ConnectionError) as e:
                if attempt < retries:
                    wait = 2 ** attempt
                    print(f"[Export] Connection error on download (attempt {attempt}/{retries}), retrying in {wait}s ...")
                    time.sleep(wait)
                else:
                    raise RuntimeError(
                        f"[Edge Impulse] Download failed after {retries} attempts.\n"
                        f"  Error: {e}"
                    )

        if not resp.ok:
            try:
                detail = resp.json().get("error", resp.text[:200])
            except Exception:
                detail = resp.text[:200]
            raise RuntimeError(
                f"[Edge Impulse] Download failed — HTTP {resp.status_code}\n"
                f"  Detail: {detail}"
            )

        # Extract .tflite from ZIP
        tflite_path = None
        with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
            print(f"[Export] ZIP contents: {zf.namelist()}")
            for name in zf.namelist():
                if name.endswith(".tflite"):
                    tflite_path = os.path.join(self.output_dir, "model.tflite")
                    with open(tflite_path, "wb") as f:
                        f.write(zf.read(name))
                    print(f"[Export] Saved model → {tflite_path}")
                    break

        if tflite_path is None:
            raise RuntimeError(
                "[Edge Impulse] No .tflite file found in the downloaded ZIP.\n"
                f"  ZIP contained: {zipfile.ZipFile(io.BytesIO(resp.content)).namelist()}"
            )

        labels_path = os.path.join(self.output_dir, "labels.json")
        with open(labels_path, "w", encoding="utf-8") as f:
            json.dump({"labels": labels}, f, indent=2, ensure_ascii=False)
        print(f"[Export] Saved labels → {labels_path}")

        return tflite_path

    # ------------------------------------------------------------------
    # Inference helper (demonstrates model output format)
    # ------------------------------------------------------------------

    def run_inference(self, wav_path: str, labels: list[str]) -> dict:
        """
        Run the exported TFLite model on a WAV file and return a structured result.

        Returns:
            {
                "predicted_label": "hello",
                "confidence": 0.95,
                "classification": {"hello": 0.95, "stop": 0.03, "noise": 0.02}
            }

        This is the deployment-ready inference function — copy it into your
        embedded application and replace the WAV loading with whatever audio
        source you use.
        """
        tflite_path = os.path.join(self.output_dir, "model.tflite")
        if not os.path.exists(tflite_path):
            raise FileNotFoundError(f"Model not found at {tflite_path}")

        features = self._extract_mfcc(wav_path)
        raw_output = self._run_tflite(tflite_path, features)

        # raw_output shape: (1, num_labels)  dtype: float32
        probabilities = raw_output[0].tolist()

        # Map to dict — this is the "classification dictionary" your boss asked about
        classification = dict(zip(labels, probabilities))
        predicted_label = labels[int(np.argmax(probabilities))]
        confidence = float(np.max(probabilities))

        return {
            "predicted_label": predicted_label,
            "confidence": confidence,
            "classification": classification,   # <-- the dict with all class probabilities
        }

    # ------------------------------------------------------------------
    # DSP: MFCC feature extraction (must match Edge Impulse project settings)
    # ------------------------------------------------------------------

    def _extract_mfcc(
        self,
        wav_path: str,
        sample_rate: int = 16000,
        num_coefficients: int = 13,
        num_filters: int = 40,
        frame_length: float = 0.02,
        frame_stride: float = 0.01,
        fft_length: int = 512,
        low_freq: int = 300,
        high_freq: int = 8000,
    ) -> np.ndarray:
        """Extract MFCC features from a WAV file — identical to Edge Impulse defaults."""
        import wave
        import scipy.fftpack

        with wave.open(wav_path, "rb") as wf:
            n_frames = wf.getnframes()
            raw = wf.readframes(n_frames)
        signal = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0

        frame_len = int(round(frame_length * sample_rate))
        frame_step = int(round(frame_stride * sample_rate))

        # Pre-emphasis
        signal[1:] -= 0.97 * signal[:-1]

        # Framing
        num_frames = 1 + (len(signal) - frame_len) // frame_step
        indices = (
            np.arange(frame_len)[None, :]
            + np.arange(num_frames)[:, None] * frame_step
        )
        frames = signal[indices] * np.hamming(frame_len)

        # FFT + power spectrum
        mag = np.abs(np.fft.rfft(frames, n=fft_length)) ** 2

        # Mel filterbank
        mel_filters = self._mel_filterbank(
            sample_rate, fft_length, num_filters, low_freq, high_freq
        )
        mel_energy = np.dot(mag, mel_filters.T)
        mel_energy = np.where(mel_energy == 0, np.finfo(float).eps, mel_energy)
        log_mel = np.log(mel_energy)

        # DCT → MFCC
        mfcc = scipy.fftpack.dct(log_mel, type=2, axis=1, norm="ortho")[:, :num_coefficients]
        return mfcc.flatten().astype(np.float32)

    @staticmethod
    def _mel_filterbank(
        sample_rate: int,
        fft_length: int,
        num_filters: int,
        low_freq: int,
        high_freq: int,
    ) -> np.ndarray:
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

    def _run_tflite(self, model_path: str, features: np.ndarray) -> np.ndarray:
        """Run TFLite inference; tries tflite-runtime first, falls back to tensorflow."""
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

        # Reshape features to match model input
        input_data = features.reshape(inp["shape"]).astype(inp["dtype"])
        interpreter.set_tensor(inp["index"], input_data)
        interpreter.invoke()

        return interpreter.get_tensor(out["index"])
