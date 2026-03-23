"""
TTSGenerator: Converts Chinese text labels into diverse audio WAV files.

All input is expected to be Chinese. gTTS (Google TTS) is used as the sole
synthesis backend — it has excellent Chinese quality and avoids the SAPI5
voice-switching deadlock that pyttsx3 suffers on Windows mid-session.

Augmentation strategy
----------------------
For each label:
  1. Generate 2 BASE clips via gTTS (zh-CN accent + zh-TW accent)
  2. Augment those bases to reach `samples_per_label` total:
       - Pitch shift     : ±0–3 semitones via resampling
       - Volume variation: 55–100% amplitude
       - Gaussian noise  : low-level background noise

Only 2 gTTS HTTP requests are made per label, so rate-limiting is not a concern.

Output format: mono 16-bit PCM 16 kHz (Edge Impulse requirement).

Dependencies:
    pip install gtts miniaudio
"""

import os
import tempfile
import time
import wave

import numpy as np
import scipy.signal
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _contains_chinese(text: str) -> bool:
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


def _safe_filename(text: str) -> str:
    """Convert a Chinese label to a safe ASCII filename stem."""
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
# Main class
# ---------------------------------------------------------------------------

class TTSGenerator:

    # Two accents give natural variety without extra gTTS requests
    GTTS_LANGS = ["zh-CN", "zh-TW"]

    def __init__(
        self,
        output_dir: str,
        sample_rate: int = 16000,
        duration: float = 1.5,
        samples_per_label: int = 20,
        tts_volume: float = 1.0,
    ):
        self.output_dir = output_dir
        self.sample_rate = sample_rate
        self.duration = duration
        self.samples_per_label = samples_per_label
        self.tts_volume = tts_volume

        self._check_dependencies()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate(self, labels: list[str]) -> dict[str, list[str]]:
        """
        Generate `samples_per_label` diverse audio clips for every label.
        Returns {label: [wav_path, ...]}
        """
        results: dict[str, list[str]] = {}

        for label in tqdm(labels, desc="Labels", unit="label"):
            safe_name = _safe_filename(label)
            label_dir = os.path.join(self.output_dir, safe_name)

            # Use cached audio if the folder already has enough WAV files
            if os.path.isdir(label_dir):
                cached = [
                    os.path.join(label_dir, f)
                    for f in sorted(os.listdir(label_dir))
                    if f.lower().endswith(".wav")
                ]
                if len(cached) >= self.samples_per_label:
                    tqdm.write(
                        f"[TTS] '{label}' → using {len(cached)} cached file(s) "
                        f"(skipping gTTS)"
                    )
                    results[label] = cached[:self.samples_per_label]
                    continue

            os.makedirs(label_dir, exist_ok=True)
            base_clips = self._generate_base_clips(label)

            # Compute adaptive duration: use actual speech length + 0.5s buffer so
            # long phrases (e.g. "打开猫砂盆") are never trimmed mid-word.
            max_base_secs = max(len(c) for c in base_clips) / self.sample_rate
            effective_duration = max(self.duration, max_base_secs + 0.5)
            if effective_duration > self.duration:
                tqdm.write(
                    f"[TTS] '{label}' → auto-extended duration "
                    f"{self.duration}s → {effective_duration:.2f}s"
                )

            paths = self._augment_to_target(base_clips, label_dir, safe_name, effective_duration)

            results[label] = paths
            tqdm.write(f"[TTS] '{label}' → {len(paths)} files in {label_dir}")

        return results

    # ------------------------------------------------------------------
    # Base clip generation (gTTS)
    # ------------------------------------------------------------------

    def _generate_base_clips(self, text: str) -> list[np.ndarray]:
        """
        Generate one base clip per accent (zh-CN, zh-TW).
        Only 2 HTTP requests per label total.
        """
        base_clips: list[np.ndarray] = []

        for lang in tqdm(self.GTTS_LANGS, desc="  TTS base", unit="accent", leave=False):
            audio = self._synth_gtts(text, lang)
            if audio is not None:
                base_clips.append(audio)
            time.sleep(0.3)  # small pause to be polite to Google's API

        if not base_clips:
            raise RuntimeError(
                f"gTTS produced no audio for '{text}'.\n"
                "Check your internet connection."
            )

        return base_clips

    def _synth_gtts(self, text: str, lang: str) -> np.ndarray | None:
        """Fetch audio from gTTS, decode MP3 → float32 numpy array via miniaudio (no ffmpeg needed)."""
        try:
            from gtts import gTTS
            import miniaudio
        except ImportError:
            raise RuntimeError(
                "gTTS and miniaudio are required.\n"
                "Run: pip install gtts miniaudio"
            )

        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp:
            mp3_path = tmp.name

        try:
            gTTS(text=text, lang=lang, slow=False).save(mp3_path)

            # miniaudio decodes MP3 natively — no external tools required
            decoded = miniaudio.decode_file(
                mp3_path,
                output_format=miniaudio.SampleFormat.SIGNED16,
                nchannels=1,
                sample_rate=self.sample_rate,
            )
            samples = np.array(decoded.samples, dtype=np.int16).astype(np.float32) / 32768.0
            return samples

        except Exception as e:
            tqdm.write(f"[TTS] gTTS error for '{text}' ({lang}): {e}")
            return None
        finally:
            if os.path.exists(mp3_path):
                os.remove(mp3_path)

    # ------------------------------------------------------------------
    # Augmentation
    # ------------------------------------------------------------------

    def _augment_to_target(
        self, base_clips: list[np.ndarray], label_dir: str, safe_name: str, duration: float
    ) -> list[str]:
        """Save base clips, then fill remaining slots with augmented variants."""
        paths: list[str] = []
        sample_idx = 0
        aug_seed = 0

        with tqdm(total=self.samples_per_label, desc="  Augmenting", unit="clip", leave=False) as pbar:
            for base in base_clips:
                if sample_idx >= self.samples_per_label:
                    break
                path = os.path.join(label_dir, f"{safe_name}_{sample_idx:03d}.wav")
                self._save_wav(base, path, duration)
                paths.append(path)
                sample_idx += 1
                pbar.update(1)

            while sample_idx < self.samples_per_label:
                base = base_clips[aug_seed % len(base_clips)]
                augmented = self._augment(base, seed=aug_seed)
                path = os.path.join(label_dir, f"{safe_name}_{sample_idx:03d}.wav")
                self._save_wav(augmented, path, duration)
                paths.append(path)
                sample_idx += 1
                aug_seed += 1
                pbar.update(1)

        return paths

    def _augment(self, samples: np.ndarray, seed: int) -> np.ndarray:
        """
        Deterministic augmentation:
          - Pitch shift  : ±0–3 semitones
          - Volume       : 55–100%
          - Gaussian noise: low-level
        """
        rng = np.random.RandomState(seed * 17 + 3)
        result = samples.copy()

        # Pitch shift via resampling
        semitones = rng.uniform(-3.0, 3.0)
        factor = 2 ** (semitones / 12.0)
        shifted_len = max(1, int(len(result) / factor))
        shifted = scipy.signal.resample(result, shifted_len)
        if len(shifted) < len(result):
            shifted = np.pad(shifted, (0, len(result) - len(shifted)))
        else:
            shifted = shifted[:len(result)]
        result = shifted

        # Volume variation
        result = result * rng.uniform(0.55, 1.0)

        # Background noise
        noise_amplitude = rng.uniform(0.002, 0.015)
        result = result + rng.randn(len(result)).astype(np.float32) * noise_amplitude

        return np.clip(result, -1.0, 1.0).astype(np.float32)

    # ------------------------------------------------------------------
    # WAV I/O
    # ------------------------------------------------------------------

    def _save_wav(self, samples: np.ndarray, path: str, duration: float) -> None:
        """Pad/trim float32 audio to `duration` seconds and save as 16-bit WAV."""
        target_len = int(self.sample_rate * duration)

        if len(samples) < target_len:
            samples = np.pad(samples, (0, target_len - len(samples)))
        else:
            samples = samples[:target_len]

        max_val = np.max(np.abs(samples))
        if max_val > 0:
            samples = samples / max_val
        pcm = (samples * 32767).astype(np.int16)

        with wave.open(path, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(self.sample_rate)
            wf.writeframes(pcm.tobytes())

    # ------------------------------------------------------------------
    # Recording-based generation (web UI input)
    # ------------------------------------------------------------------

    def _load_audio_file(self, path: str) -> np.ndarray | None:
        """Decode any audio file (webm, wav, mp3, ogg) to float32 numpy array at self.sample_rate."""
        try:
            import miniaudio
            decoded = miniaudio.decode_file(
                path,
                output_format=miniaudio.SampleFormat.SIGNED16,
                nchannels=1,
                sample_rate=self.sample_rate,
            )
            return np.array(decoded.samples, dtype=np.int16).astype(np.float32) / 32768.0
        except Exception as e:
            print(f"[Audio] Failed to load {path}: {e}")
            return None

    def generate_from_recordings(self, label: str, audio_paths: list[str]) -> list[str]:
        """
        Build `samples_per_label` WAV files from user recordings.

        Augmentation fills up to `samples_per_label` just like generate(),
        but the base clips come from real microphone recordings instead of gTTS.

        Returns list of WAV paths (same interface as generate()).
        """
        safe_name = _safe_filename(label)
        label_dir = os.path.join(self.output_dir, safe_name)
        os.makedirs(label_dir, exist_ok=True)

        base_clips: list[np.ndarray] = []
        for path in audio_paths:
            audio = self._load_audio_file(path)
            if audio is not None and len(audio) > 0:
                base_clips.append(audio)

        if not base_clips:
            raise RuntimeError(f"[Audio] No valid audio loaded for label '{label}'")

        max_base_secs = max(len(c) for c in base_clips) / self.sample_rate
        effective_duration = max(self.duration, max_base_secs + 0.5)

        paths = self._augment_to_target(base_clips, label_dir, safe_name, effective_duration)
        print(f"[Audio] '{label}' → augmented {len(audio_paths)} clips → {len(paths)} files")
        return paths

    # ------------------------------------------------------------------
    # Dependency check
    # ------------------------------------------------------------------

    @staticmethod
    def _check_dependencies() -> None:
        missing = []
        try:
            import gtts  # noqa: F401
        except ImportError:
            missing.append("gtts")
        try:
            import miniaudio  # noqa: F401
        except ImportError:
            missing.append("miniaudio")
        if missing:
            raise RuntimeError(
                f"Missing required packages: {', '.join(missing)}\n"
                f"Run: pip install {' '.join(missing)}"
            )
