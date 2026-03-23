"""
LabelsRegistry: Tracks which labels have been uploaded to Edge Impulse.

Persists state in `<dataset_dir>/labels_state.json`.

- Skips TTS generation and upload for labels that haven't changed.
- Identifies labels whose config changed so old EI data can be deleted
  before a fresh upload, preventing unbounded dataset growth.
"""

import hashlib
import json
import os
from datetime import date


class LabelsRegistry:
    _STATE_FILE = "labels_state.json"

    def __init__(self, dataset_dir: str):
        self.dataset_dir = dataset_dir
        self._state_path = os.path.join(dataset_dir, self._STATE_FILE)
        self._state: dict = self._load()

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _load(self) -> dict:
        if os.path.exists(self._state_path):
            try:
                with open(self._state_path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except (json.JSONDecodeError, OSError):
                return {}
        return {}

    def save(self) -> None:
        os.makedirs(self.dataset_dir, exist_ok=True)
        with open(self._state_path, "w", encoding="utf-8") as f:
            json.dump(self._state, f, ensure_ascii=False, indent=2)

    # ------------------------------------------------------------------
    # Config fingerprint
    # ------------------------------------------------------------------

    @staticmethod
    def _config_key(cfg: dict) -> str:
        """MD5 of the TTS config fields that affect audio output."""
        relevant = {
            "samples_per_label": cfg.get("samples_per_label"),
            "sample_rate": cfg.get("sample_rate"),
            "duration": cfg.get("duration"),
        }
        return hashlib.md5(json.dumps(relevant, sort_keys=True).encode()).hexdigest()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def status(self, label: str, cfg: dict) -> str:
        """Return 'new', 'changed', or 'unchanged' for a label."""
        if label not in self._state:
            return "new"
        if self._state[label].get("config_key") != self._config_key(cfg):
            return "changed"
        return "unchanged"

    def record(self, label: str, safe_name: str, cfg: dict) -> None:
        """Mark a label as successfully uploaded with the given config."""
        self._state[label] = {
            "safe_name": safe_name,
            "config_key": self._config_key(cfg),
            "samples_per_label": cfg.get("samples_per_label"),
            "sample_rate": cfg.get("sample_rate"),
            "duration": cfg.get("duration"),
            "uploaded_at": str(date.today()),
        }

    def get_safe_name(self, label: str) -> str | None:
        """Return the stored safe folder name for a label, or None if not recorded."""
        return self._state.get(label, {}).get("safe_name")

    def remove(self, label: str) -> None:
        self._state.pop(label, None)

    def all_labels(self) -> list[str]:
        return list(self._state.keys())
