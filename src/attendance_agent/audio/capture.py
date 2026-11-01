"""Audio capture and extraction module.

Supports loading audio from dedicated audio files (WAV, MP3, FLAC, OGG),
extracting audio tracks from video files (MP4, AVI, MOV) using ffmpeg,
and capturing live microphone input using sounddevice.
"""

from __future__ import annotations

import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import librosa
import numpy as np
import sounddevice as sd
import soundfile as sf

from attendance_agent.utils.logger import get_logger

logger = get_logger("attendance_agent.audio.capture")


def convert_to_mono(audio: np.ndarray) -> np.ndarray:
    """Ensure audio signal is single-channel (mono) float32 array.

    Args:
        audio: 1D or 2D numpy audio array.

    Returns:
        np.ndarray: 1D float32 mono audio array.
    """
    if audio.ndim == 1:
        return audio.astype(np.float32)

    # 2D stereo: determine channel axis
    # If channels-first: e.g. shape (2, N) where N > 2
    if audio.shape[0] <= 8 and audio.shape[0] < audio.shape[1]:
        mono = np.mean(audio, axis=0)
    else:
        # Samples-first: e.g. shape (N, 2)
        mono = np.mean(audio, axis=1)

    return mono.astype(np.float32)


def extract_audio_from_video(video_path: Path | str, target_sr: int = 16000) -> np.ndarray:
    """Extract audio track from a video container using ffmpeg.

    Args:
        video_path: Path to video file (MP4, AVI, MOV).
        target_sr: Target sample rate for extracted audio.

    Returns:
        np.ndarray: Extracted 1D mono float32 audio signal.

    Raises:
        RuntimeError: If ffmpeg is unavailable or audio extraction fails.
    """
    ffmpeg_bin = shutil.which("ffmpeg")
    if not ffmpeg_bin:
        raise RuntimeError("ffmpeg binary not found in system PATH. Cannot extract audio from video.")

    vpath = Path(video_path)
    if not vpath.exists():
        raise FileNotFoundError(f"Video file not found at: {vpath}")

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_audio:
        tmp_wav_path = Path(tmp_audio.name)

    try:
        cmd = [
            ffmpeg_bin,
            "-y",  # overwrite
            "-i",
            str(vpath),
            "-vn",  # disable video
            "-acodec",
            "pcm_s16le",  # 16-bit PCM WAV
            "-ar",
            str(target_sr),
            "-ac",
            "1",  # mono
            str(tmp_wav_path),
        ]

        logger.debug("Running ffmpeg extraction: %s", " ".join(cmd))
        proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=False)
        if proc.returncode != 0:
            raise RuntimeError(f"ffmpeg extraction failed with code {proc.returncode}: {proc.stderr[:300]}")

        # Load the generated WAV
        audio_data, sr = sf.read(str(tmp_wav_path), dtype="float32")
        return convert_to_mono(audio_data)
    finally:
        tmp_wav_path.unlink(missing_ok=True)


class AudioSourceLoader:
    """Loads and standardizes audio from files, videos, or raw buffers."""

    VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
    AUDIO_EXTENSIONS = {".wav", ".mp3", ".flac", ".ogg", ".m4a"}

    @classmethod
    def load_audio(
        cls,
        source_path: Path | str,
        target_sr: int = 16000,
    ) -> Tuple[np.ndarray, int]:
        """Load audio from file and convert to consistent mono float32 at target_sr.

        Args:
            source_path: Path to audio or video file.
            target_sr: Desired sample rate (default: 16,000 Hz).

        Returns:
            Tuple[np.ndarray, int]: (mono_audio_signal, sample_rate)

        Raises:
            FileNotFoundError: If source file is not found.
            ValueError: If file is empty or unsupported format.
        """
        path = Path(source_path)
        if not path.exists():
            raise FileNotFoundError(f"Audio source file not found at: {path}")

        ext = path.suffix.lower()

        # Handle video container by extracting audio track
        if ext in cls.VIDEO_EXTENSIONS:
            logger.info("Extracting audio from video file: %s", path.name)
            signal = extract_audio_from_video(path, target_sr=target_sr)
            return signal, target_sr

        # Handle standard audio file
        logger.info("Loading audio file: %s", path.name)
        try:
            # First attempt fast soundfile read
            audio_data, orig_sr = sf.read(str(path), dtype="float32")
            mono = convert_to_mono(audio_data)

            if orig_sr != target_sr:
                mono = librosa.resample(mono, orig_sr=orig_sr, target_sr=target_sr)

            if len(mono) == 0:
                raise ValueError("Loaded audio file contains zero samples.")

            return mono, target_sr
        except Exception as sf_exc:
            # Fallback to librosa.load
            logger.debug("soundfile loading failed (%s), falling back to librosa.load", sf_exc)
            try:
                mono, _ = librosa.load(str(path), sr=target_sr, mono=True)
                if len(mono) == 0:
                    raise ValueError("Loaded audio file contains zero samples.")
                return mono.astype(np.float32), target_sr
            except Exception as lib_exc:
                raise ValueError(f"Failed to read audio file '{path.name}': {lib_exc}") from lib_exc


class MicrophoneCapture:
    """Optional live microphone capture using sounddevice."""

    @staticmethod
    def is_available() -> bool:
        """Check if at least one audio input device is available."""
        try:
            devices = sd.query_devices()
            return any(d.get("max_input_channels", 0) > 0 for d in devices)
        except Exception:
            return False

    @staticmethod
    def list_input_devices() -> List[Dict[str, Any]]:
        """List all valid audio recording input devices.

        Returns:
            List[Dict[str, Any]]: List of device descriptor dictionaries.
        """
        try:
            devices = sd.query_devices()
            results = []
            for idx, dev in enumerate(devices):
                if dev.get("max_input_channels", 0) > 0:
                    results.append(
                        {
                            "index": idx,
                            "name": dev["name"],
                            "channels": dev["max_input_channels"],
                            "default_samplerate": dev.get("default_samplerate", 16000),
                        }
                    )
            return results
        except Exception as exc:
            logger.warning("Unable to list audio input devices: %s", exc)
            return []

    @classmethod
    def record_chunk(
        cls,
        duration_seconds: float = 5.0,
        sample_rate: int = 16000,
        device_index: Optional[int] = None,
    ) -> np.ndarray:
        """Synchronously record an audio chunk from microphone.

        Args:
            duration_seconds: Length of recording in seconds.
            sample_rate: Recording sample rate.
            device_index: Optional specific input device index.

        Returns:
            np.ndarray: 1D mono float32 recorded audio signal.

        Raises:
            RuntimeError: If recording fails or microphone access is denied.
        """
        num_samples = int(duration_seconds * sample_rate)
        logger.debug("Recording %s seconds from mic (device=%s)", duration_seconds, device_index)

        try:
            recording = sd.rec(
                frames=num_samples,
                samplerate=sample_rate,
                channels=1,
                dtype="float32",
                device=device_index,
                blocking=True,
            )
            return recording.flatten().astype(np.float32)
        except Exception as exc:
            logger.error("Microphone recording failed: %s", exc)
            raise RuntimeError(f"Microphone recording failed: {exc}") from exc
