"""
DatasetBuilder: Scans the dataset directory and reports what's ready to upload.

Splits files into training (80%) and testing (20%) sets so Edge Impulse
gets a balanced evaluation dataset alongside training data.
"""

import os
import random


class DatasetBuilder:
    def __init__(self, dataset_dir: str, train_split: float = 0.8, seed: int = 42):
        self.dataset_dir = dataset_dir
        self.train_split = train_split
        self.seed = seed

    def build(
        self,
        only_labels: list[str] | None = None,
    ) -> dict[str, dict[str, list[str]]]:
        """
        Scan `dataset_dir` for label subdirectories and split files into
        training / testing.

        Args:
            only_labels: If provided, only process these folder names (safe ASCII
                         names used by TTSGenerator). Useful to upload only new or
                         changed labels without re-uploading unchanged ones.

        Returns:
            {
                "training": {"hello": [path1, path2, ...], ...},
                "testing":  {"hello": [path3, ...], ...},
            }
        """
        random.seed(self.seed)
        result = {"training": {}, "testing": {}}

        if not os.path.isdir(self.dataset_dir):
            raise FileNotFoundError(f"Dataset directory not found: {self.dataset_dir}")

        labels = [
            d for d in os.listdir(self.dataset_dir)
            if os.path.isdir(os.path.join(self.dataset_dir, d))
            and d != "labels_state.json"  # skip the state file if misdetected
        ]

        if only_labels is not None:
            only_set = set(only_labels)
            labels = [l for l in labels if l in only_set]

        if not labels:
            raise ValueError(f"No label subdirectories found in {self.dataset_dir}")

        for label in sorted(labels):
            label_dir = os.path.join(self.dataset_dir, label)
            wav_files = [
                os.path.join(label_dir, f)
                for f in os.listdir(label_dir)
                if f.lower().endswith(".wav")
            ]

            if not wav_files:
                print(f"[Dataset] Warning: no .wav files in '{label}', skipping.")
                continue

            random.shuffle(wav_files)
            split_at = max(1, int(len(wav_files) * self.train_split))
            result["training"][label] = wav_files[:split_at]
            result["testing"][label] = wav_files[split_at:]

        self._print_summary(result)
        return result

    # ------------------------------------------------------------------

    def _print_summary(self, split: dict) -> None:
        labels = set(split["training"]) | set(split["testing"])
        print(f"\n[Dataset] Split summary ({len(labels)} labels):")
        for label in sorted(labels):
            n_train = len(split["training"].get(label, []))
            n_test = len(split["testing"].get(label, []))
            print(f"  {label:<20} train={n_train}  test={n_test}")
        print()
