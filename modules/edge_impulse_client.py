"""
EdgeImpulseClient: Thin wrapper around the Edge Impulse REST API.

Handles:
  - Uploading .wav files with labels (training + testing splits)
  - Triggering a training job
  - Polling until the training job finishes
"""

import os
import time
import requests
from requests import Response


def _ei_error(resp: Response) -> str:
    """Extract a readable error message from an Edge Impulse API response."""
    try:
        body = resp.json()
        # Edge Impulse wraps errors in {"success": false, "error": "..."}
        return body.get("error", body.get("message", resp.text[:200]))
    except Exception:
        return resp.text[:200]


class EdgeImpulseClient:
    STUDIO_URL = "https://studio.edgeimpulse.com/v1/api"
    INGESTION_URL = "https://ingestion.edgeimpulse.com/api"

    def __init__(self, api_key: str, project_id: str):
        self.api_key = api_key
        self.project_id = project_id
        self._headers = {"x-api-key": self.api_key}

    # ------------------------------------------------------------------
    # Upload
    # ------------------------------------------------------------------

    def upload_dataset(self, split: dict[str, dict[str, list[str]]]) -> None:
        """
        Upload all WAV files to Edge Impulse.

        `split` must be in the format returned by DatasetBuilder.build():
            {"training": {label: [paths]}, "testing": {label: [paths]}}
        """
        for category in ("training", "testing"):
            label_map = split.get(category, {})
            for label, paths in label_map.items():
                print(f"[Upload] {category}/{label}: {len(paths)} files ...")
                for path in paths:
                    self._upload_file(path, label, category)

    def _upload_file(self, path: str, label: str, category: str, retries: int = 3) -> None:
        # Edge Impulse ingestion API: label and category go in headers, not form data
        url = f"{self.INGESTION_URL}/{category}/files"
        filename = os.path.basename(path)
        headers = {
            **self._headers,
            "x-label": label,
        }

        for attempt in range(1, retries + 1):
            try:
                with open(path, "rb") as f:
                    resp = requests.post(
                        url,
                        headers=headers,
                        files={"data": (filename, f, "audio/wav")},
                        timeout=30,
                    )

                if resp.status_code not in (200, 201):
                    print(
                        f"[Edge Impulse] Upload failed for {filename} — HTTP {resp.status_code}\n"
                        f"  Detail: {_ei_error(resp)}"
                    )
                else:
                    print(f"[Upload]   OK: {filename}")
                return  # success or non-retryable HTTP error — either way stop retrying

            except (requests.exceptions.SSLError, requests.exceptions.ConnectionError) as e:
                if attempt < retries:
                    wait = 2 ** attempt  # 2s, 4s
                    print(f"[Upload]   Connection error on {filename} (attempt {attempt}/{retries}), retrying in {wait}s ...")
                    time.sleep(wait)
                else:
                    print(
                        f"[Edge Impulse] Upload failed for {filename} after {retries} attempts.\n"
                        f"  Error: {e}\n"
                        f"  Check your internet connection and try again."
                    )

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def _get_impulse_block_ids(self) -> tuple[int, int]:
        """
        Fetch the impulse once and return (learn_block_id, dsp_block_id).
        Raises if either block type is missing.
        """
        url = f"{self.STUDIO_URL}/{self.project_id}/impulse"
        resp = requests.get(url, headers=self._headers, timeout=30)
        if not resp.ok:
            raise RuntimeError(
                f"[Edge Impulse] Could not fetch impulse — HTTP {resp.status_code}\n"
                f"  Detail: {_ei_error(resp)}\n"
                f"  Make sure your impulse is saved in Studio: "
                f"https://studio.edgeimpulse.com/studio/{self.project_id}/impulse"
            )
        impulse = resp.json().get("impulse", {})

        learn_id = None
        for block in impulse.get("learnBlocks", []):
            if "keras" in block.get("type", "").lower() or "classification" in block.get("title", "").lower():
                learn_id = block["id"]
                print(f"[Train] Learn block: id={learn_id} title='{block.get('title', '')}'")
                break

        if learn_id is None:
            raise RuntimeError(
                "No Keras learning block found in your project's impulse.\n"
                "Please set up your impulse in Edge Impulse Studio first:\n"
                "  1. Go to your project → Impulse Design\n"
                "  2. Add an Audio (MFE or MFCC) processing block\n"
                "  3. Add a Classification (Keras) learning block\n"
                "  4. Save the impulse, then re-run this pipeline."
            )

        dsp_id = None
        for block in impulse.get("dspBlocks", []):
            dsp_id = block["id"]
            print(f"[Train] DSP block:   id={dsp_id} title='{block.get('title', '')}'")
            break

        if dsp_id is None:
            raise RuntimeError(
                "No DSP block found in your project's impulse.\n"
                "Please add an Audio (MFE or MFCC) processing block in Impulse Design."
            )

        return learn_id, dsp_id

    def get_learn_id(self) -> int:
        """Backward-compatible wrapper — returns just the learn block ID."""
        learn_id, _ = self._get_impulse_block_ids()
        return learn_id

    def run_dsp_autotune(self, timeout: int = 300, poll_interval: int = 10) -> bool:
        """
        Trigger DSP parameter auto-tune and wait for it to complete.
        Uses POST /api/{projectId}/jobs/autotune-dsp with the project's DSP block ID.
        Returns True on success, False on failure.
        """
        _, dsp_id = self._get_impulse_block_ids()

        url = f"{self.STUDIO_URL}/{self.project_id}/jobs/autotune-dsp"
        resp = requests.post(url, headers=self._headers, json={"dspId": dsp_id}, timeout=30)
        if not resp.ok:
            print(
                f"[Autotune] Failed to start auto-tune job — HTTP {resp.status_code}\n"
                f"  Detail: {_ei_error(resp)}"
            )
            return False

        job_id = str(resp.json().get("id", ""))
        if not job_id:
            print(f"[Autotune] No job ID in response: {resp.json()}")
            return False

        print(f"[Autotune] Job started: id={job_id} (dspId={dsp_id})")

        # Poll until done — same pattern as wait_for_training
        status_url = f"{self.STUDIO_URL}/{self.project_id}/jobs/{job_id}/status"
        deadline = time.time() + timeout
        while time.time() < deadline:
            poll_resp = self._request_with_retry("GET", status_url, timeout=30)
            if poll_resp is None or not poll_resp.ok:
                time.sleep(poll_interval)
                continue
            job = poll_resp.json().get("job", {})
            if job.get("finishedSuccessful"):
                print("[Autotune] Auto-tune completed successfully.")
                return True
            if job.get("finished") and not job.get("finishedSuccessful"):
                print(f"[Autotune] Auto-tune job {job_id} failed on the EI side.")
                return False
            elapsed = int(time.time() - (deadline - timeout))
            print(f"[Autotune]   ... running (elapsed {elapsed}s)")
            time.sleep(poll_interval)

        print(f"[Autotune] Timed out after {timeout}s.")
        return False

    def start_training(
        self,
        training_cycles: int = 100,
        learning_rate: float = 0.0005,
        batch_size: int = 32,
        train_test_split: float = 0.8,
        auto_class_weights: bool = False,
    ) -> str:
        """
        Fetch the Keras learn block ID, then trigger a training job.
        Returns the job_id string.
        """
        learn_id, _ = self._get_impulse_block_ids()
        url = f"{self.STUDIO_URL}/{self.project_id}/jobs/train/keras/{learn_id}"
        training_params = {
            "trainingCycles": training_cycles,
            "learningRate": learning_rate,
            "batchSize": batch_size,
            "trainTestSplit": train_test_split,
            "autoClassWeights": auto_class_weights,
        }
        print(f"[Train] Params: cycles={training_cycles} lr={learning_rate} batch={batch_size}")
        resp = requests.post(url, headers=self._headers, json=training_params, timeout=30)
        if not resp.ok:
            raise RuntimeError(
                f"[Edge Impulse] Failed to start training job — HTTP {resp.status_code}\n"
                f"  Detail: {_ei_error(resp)}\n"
                f"  Common causes:\n"
                f"    - Not enough training data uploaded\n"
                f"    - DSP auto-tune not run yet (use --autotune flag)\n"
                f"    - MFCC parameters produced no features\n"
                f"  Studio link: https://studio.edgeimpulse.com/studio/{self.project_id}/impulse"
            )
        body = resp.json()
        job_id = str(
            body.get("id")
            or body.get("jobId")
            or (body.get("job") or {}).get("id")
            or ""
        )
        if not job_id:
            raise RuntimeError(
                f"[Edge Impulse] Training job started but no job ID returned.\n"
                f"  Full response: {body}"
            )
        print(f"[Train] Job started: id={job_id}")
        return job_id

    def wait_for_training(self, job_id: str, timeout: int = 600, poll_interval: int = 10) -> bool:
        """
        Poll job status until it succeeds, fails, or times out.
        Returns True on success, False otherwise.
        """
        # Spec: GET /api/{projectId}/jobs/{jobId}/status
        url = f"{self.STUDIO_URL}/{self.project_id}/jobs/{job_id}/status"
        deadline = time.time() + timeout
        print(f"[Train] Waiting for job {job_id} (timeout {timeout}s) ...")

        while time.time() < deadline:
            resp = requests.get(url, headers=self._headers, timeout=30)
            resp.raise_for_status()
            job = resp.json().get("job", {})

            # Spec field is 'finishedSuccessful' (no trailing 'ly')
            finished = job.get("finished") is not None
            success = job.get("finishedSuccessful", False)

            if finished and success:
                print(f"[Train] Job {job_id} completed successfully.")
                return True

            if finished and not success:
                print(
                    f"[Edge Impulse] Training job {job_id} failed on the Edge Impulse side.\n"
                    f"  Common causes:\n"
                    f"    - MFCC parameters don't fit the audio (no DSP features generated)\n"
                    f"    - Not enough samples per label (minimum ~10 recommended)\n"
                    f"    - Audio clips too short for the configured window size\n"
                    f"  See full logs in Studio: "
                    f"https://studio.edgeimpulse.com/studio/{self.project_id}/jobs/{job_id}"
                )
                return False

            elapsed = int(time.time() - (deadline - timeout))
            print(f"[Train]   ... still running (elapsed {elapsed}s)")
            time.sleep(poll_interval)

        print(f"[Train] Timeout waiting for job {job_id}.")
        return False

    # ------------------------------------------------------------------
    # Sample management
    # ------------------------------------------------------------------

    def clear_all_project_data(self) -> int:
        """
        Delete ALL training and testing samples from the project.
        Called before every upload to ensure EI contains exactly what was just generated.
        Returns total samples deleted.
        """
        all_ids: list[int] = []
        for category in ("training", "testing"):
            offset = 0
            while True:
                samples = self._fetch_page(category, limit=200, offset=offset)
                if not samples:
                    break
                all_ids.extend(s["id"] for s in samples if s.get("id") is not None)
                if len(samples) < 200:
                    break
                offset += len(samples)

        if not all_ids:
            print("[Delete] Project is already empty.")
            return 0

        print(f"[Delete] Removing {len(all_ids)} existing sample(s) from project ...")
        deleted = 0
        for sample_id in all_ids:
            del_url = f"{self.STUDIO_URL}/{self.project_id}/raw-data/{sample_id}"
            del_resp = self._request_with_retry("DELETE", del_url, timeout=30)
            if del_resp is not None and del_resp.ok:
                deleted += 1
            else:
                status = del_resp.status_code if del_resp is not None else "no response"
                print(f"[Delete] Warning: failed to delete sample {sample_id} — HTTP {status}")

        print(f"[Delete] Cleared {deleted}/{len(all_ids)} sample(s).")
        return deleted

    def delete_label_samples(self, label: str) -> int:
        """
        Delete all samples (training + testing) for the given label.

        Phase 1: collect all matching sample IDs by paging through the dataset.
        Phase 2: delete each collected ID.

        Returns total samples deleted.
        """
        all_ids: list[int] = []
        for category in ("training", "testing"):
            all_ids.extend(self._collect_ids_for_label(label, category))

        if not all_ids:
            print(f"[Delete] No samples found for label '{label}' — nothing to delete.")
            return 0

        deleted = 0
        for sample_id in all_ids:
            del_url = f"{self.STUDIO_URL}/{self.project_id}/raw-data/{sample_id}"
            del_resp = self._request_with_retry("DELETE", del_url, timeout=30)
            if del_resp is not None and del_resp.ok:
                deleted += 1
            else:
                status = del_resp.status_code if del_resp is not None else "no response"
                print(f"[Delete] Warning: failed to delete sample {sample_id} — HTTP {status}")

        print(f"[Delete] Removed {deleted}/{len(all_ids)} sample(s) for label '{label}'")
        return deleted

    def _collect_ids_for_label(self, label: str, category: str) -> list[int]:
        """Page through all samples in a category and collect IDs matching label."""
        ids: list[int] = []
        offset = 0
        while True:
            samples = self._fetch_page(category, limit=200, offset=offset)
            if samples is None:
                break
            if not samples:
                break
            for s in samples:
                if s.get("label") == label:
                    sid = s.get("id")
                    if sid is not None:
                        ids.append(sid)
            if len(samples) < 200:
                break
            offset += len(samples)
        return ids

    def _fetch_page(self, category: str, limit: int, offset: int) -> list | None:
        """Fetch one page of raw-data samples. Returns None on unrecoverable error."""
        url = f"{self.STUDIO_URL}/{self.project_id}/raw-data"
        params = {"category": category, "limit": limit, "offset": offset}
        resp = self._request_with_retry("GET", url, params=params, timeout=30)
        if resp is None:
            return None
        if not resp.ok:
            print(
                f"[Delete] Warning: could not list {category} samples at offset={offset} "
                f"— HTTP {resp.status_code}: {_ei_error(resp)}"
            )
            return None
        return resp.json().get("samples", [])

    def _request_with_retry(self, method: str, url: str, retries: int = 3, **kwargs) -> "requests.Response | None":
        """Execute an HTTP request with exponential back-off on SSL/connection errors."""
        for attempt in range(1, retries + 1):
            try:
                return requests.request(method, url, headers=self._headers, **kwargs)
            except (requests.exceptions.SSLError, requests.exceptions.ConnectionError) as e:
                if attempt < retries:
                    wait = 2 ** attempt
                    print(f"[HTTP] Connection error (attempt {attempt}/{retries}), retrying in {wait}s — {e}")
                    time.sleep(wait)
                else:
                    print(f"[HTTP] Failed after {retries} attempts: {e}")
                    return None
        return None

    # ------------------------------------------------------------------
    # Misc
    # ------------------------------------------------------------------

    def get_project_info(self) -> dict:
        url = f"{self.STUDIO_URL}/{self.project_id}"
        resp = requests.get(url, headers=self._headers, timeout=30)
        resp.raise_for_status()
        return resp.json()
