"""Audio feature extraction and classroom acoustic activity analyzer.

Computes Root Mean Square (RMS) energy, Zero-Crossing Rate (ZCR),
speech activity presence, silence estimation, and noise levels over time windows.
Does NOT perform speech recognition, speaker identification, or audio biometric profiling.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from attendance_agent.models import NoiseLevel
from attendance_agent.utils.logger import get_logger

logger = get_logger("attendance_agent.audio.analyzer")


@dataclass
class WindowAudioMetric:
    """Acoustic activity metric computed over an individual time window."""

    window_index: int
    timestamp: str
    window_start: float
    window_end: float
    rms_energy: float
    zero_crossing_rate: float
    activity_detected: bool
    noise_level: str
    is_silence: bool

    def to_dict(self) -> Dict[str, Any]:
        """Convert window metric to dictionary representation."""
        return {
            "window_index": self.window_index,
            "timestamp": self.timestamp,
            "window_start": round(self.window_start, 2),
            "window_end": round(self.window_end, 2),
            "rms_energy": round(self.rms_energy, 5),
            "zero_crossing_rate": round(self.zero_crossing_rate, 4),
            "activity_detected": self.activity_detected,
            "noise_level": self.noise_level,
            "is_silence": self.is_silence,
        }


@dataclass
class AudioAnalysisReport:
    """Aggregated acoustic activity report for an entire audio stream or session."""

    total_duration_seconds: float
    window_count: int
    active_window_count: int
    activity_ratio: float
    silence_duration_seconds: float
    mean_rms_energy: float
    mean_zcr: float
    noise_distribution: Dict[str, int] = field(default_factory=dict)
    metrics: List[WindowAudioMetric] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert report to dictionary."""
        return {
            "total_duration_seconds": round(self.total_duration_seconds, 2),
            "window_count": self.window_count,
            "active_window_count": self.active_window_count,
            "activity_ratio": round(self.activity_ratio, 1),
            "silence_duration_seconds": round(self.silence_duration_seconds, 2),
            "mean_rms_energy": round(self.mean_rms_energy, 5),
            "mean_zcr": round(self.mean_zcr, 4),
            "noise_distribution": self.noise_distribution,
            "metrics": [m.to_dict() for m in self.metrics],
        }

    def to_dataframe(self) -> pd.DataFrame:
        """Convert time window metrics into a pandas DataFrame."""
        if not self.metrics:
            return pd.DataFrame(
                columns=[
                    "Window",
                    "Start (s)",
                    "End (s)",
                    "RMS Energy",
                    "ZCR",
                    "Activity Detected",
                    "Noise Level",
                ]
            )

        rows = [
            {
                "Window": m.window_index,
                "Start (s)": round(m.window_start, 1),
                "End (s)": round(m.window_end, 1),
                "RMS Energy": round(m.rms_energy, 4),
                "ZCR": round(m.zero_crossing_rate, 3),
                "Activity Detected": "Active" if m.activity_detected else "Quiet",
                "Noise Level": m.noise_level.capitalize(),
            }
            for m in self.metrics
        ]
        return pd.DataFrame(rows)


class AudioAnalyzer:
    """Analyzes classroom audio signals for activity, noise levels, and silence."""

    def __init__(
        self,
        sample_rate: int = 16000,
        window_duration_seconds: float = 5.0,
        rms_activity_threshold: float = 0.015,
        zcr_activity_threshold: float = 0.04,
        silence_threshold: float = 0.005,
        high_noise_threshold: float = 0.12,
    ) -> None:
        """Initialize AudioAnalyzer with configurable acoustic thresholds.

        Args:
            sample_rate: Expected audio sample rate in Hz (default: 16,000 Hz).
            window_duration_seconds: Time slice length per window in seconds.
            rms_activity_threshold: Minimum RMS energy to flag presence of acoustic activity.
            zcr_activity_threshold: Minimum Zero-Crossing Rate to flag voice-like frequency content.
            silence_threshold: Upper RMS bound for classifying an interval as silence.
            high_noise_threshold: Lower RMS bound for classifying an interval as high noise.
        """
        self.sample_rate = sample_rate
        self.window_duration = window_duration_seconds
        self.rms_threshold = rms_activity_threshold
        self.zcr_threshold = zcr_activity_threshold
        self.silence_threshold = silence_threshold
        self.high_noise_threshold = high_noise_threshold

    @staticmethod
    def compute_rms(signal: np.ndarray) -> float:
        """Calculate Root Mean Square (RMS) energy of an audio segment.

        Args:
            signal: 1D audio sample array.

        Returns:
            float: RMS energy (0.0 to 1.0).
        """
        if len(signal) == 0:
            return 0.0
        mean_sq = float(np.mean(np.square(signal)))
        return float(math.sqrt(max(0.0, mean_sq)))

    @staticmethod
    def compute_zcr(signal: np.ndarray) -> float:
        """Calculate normalized Zero-Crossing Rate (ZCR) of an audio segment.

        Args:
            signal: 1D audio sample array.

        Returns:
            float: Fraction of zero-crossings (0.0 to 1.0).
        """
        if len(signal) < 2:
            return 0.0
        zero_crossings = np.nonzero(np.diff(signal > 0))[0]
        return float(len(zero_crossings) / (len(signal) - 1))

    def classify_noise_level(self, rms: float) -> str:
        """Classify acoustic noise amplitude into low, moderate, or high.

        Args:
            rms: Window RMS energy.

        Returns:
            str: "low", "moderate", or "high".
        """
        if rms >= self.high_noise_threshold:
            return NoiseLevel.HIGH.value
        if rms >= self.rms_threshold:
            return NoiseLevel.MODERATE.value
        return NoiseLevel.LOW.value

    def analyze_window(
        self,
        samples: np.ndarray,
        window_start: float,
        window_end: float,
        window_idx: int,
    ) -> WindowAudioMetric:
        """Analyze a single audio window.

        Args:
            samples: Audio samples for this window.
            window_start: Start timestamp in seconds.
            window_end: End timestamp in seconds.
            window_idx: 1-based sequential window index.

        Returns:
            WindowAudioMetric: Structured metrics for the window.
        """
        rms = self.compute_rms(samples)
        zcr = self.compute_zcr(samples)

        is_silence = rms <= self.silence_threshold
        noise = self.classify_noise_level(rms)

        # Activity detected when acoustic energy and frequency variation exceed thresholds
        activity_detected = bool(rms >= self.rms_threshold and zcr >= self.zcr_threshold)

        timestamp_iso = datetime.now(timezone.utc).isoformat()

        return WindowAudioMetric(
            window_index=window_idx,
            timestamp=timestamp_iso,
            window_start=window_start,
            window_end=window_end,
            rms_energy=rms,
            zero_crossing_rate=zcr,
            activity_detected=activity_detected,
            noise_level=noise,
            is_silence=is_silence,
        )

    def analyze_audio_signal(
        self,
        signal: np.ndarray,
        sample_rate: Optional[int] = None,
        start_offset: float = 0.0,
    ) -> AudioAnalysisReport:
        """Divide continuous audio signal into time windows and compute metric timeline.

        Args:
            signal: 1D mono audio array.
            sample_rate: Audio sample rate (defaults to self.sample_rate).
            start_offset: Initial timestamp offset in seconds.

        Returns:
            AudioAnalysisReport: Complete metric timeline and aggregated summary.
        """
        sr = sample_rate or self.sample_rate
        total_samples = len(signal)
        total_duration = total_samples / sr if sr > 0 else 0.0

        if total_samples == 0:
            return AudioAnalysisReport(
                total_duration_seconds=0.0,
                window_count=0,
                active_window_count=0,
                activity_ratio=0.0,
                silence_duration_seconds=0.0,
                mean_rms_energy=0.0,
                mean_zcr=0.0,
                noise_distribution={"low": 0, "moderate": 0, "high": 0},
                metrics=[],
            )

        window_size_samples = int(self.window_duration * sr)
        metrics: List[WindowAudioMetric] = []
        noise_dist = {"low": 0, "moderate": 0, "high": 0}

        cur_sample = 0
        window_idx = 1

        while cur_sample < total_samples:
            end_sample = min(total_samples, cur_sample + window_size_samples)
            slice_samples = signal[cur_sample:end_sample]

            w_start = start_offset + (cur_sample / sr)
            w_end = start_offset + (end_sample / sr)

            metric = self.analyze_window(
                samples=slice_samples,
                window_start=w_start,
                window_end=w_end,
                window_idx=window_idx,
            )
            metrics.append(metric)

            # Update noise distribution count
            noise_dist[metric.noise_level] = noise_dist.get(metric.noise_level, 0) + 1

            cur_sample += window_size_samples
            window_idx += 1

        active_count = sum(1 for m in metrics if m.activity_detected)
        silence_dur = sum((m.window_end - m.window_start) for m in metrics if m.is_silence)
        mean_rms = float(np.mean([m.rms_energy for m in metrics])) if metrics else 0.0
        mean_zcr = float(np.mean([m.zero_crossing_rate for m in metrics])) if metrics else 0.0
        ratio = (active_count / len(metrics) * 100.0) if metrics else 0.0

        return AudioAnalysisReport(
            total_duration_seconds=total_duration,
            window_count=len(metrics),
            active_window_count=active_count,
            activity_ratio=ratio,
            silence_duration_seconds=silence_dur,
            mean_rms_energy=mean_rms,
            mean_zcr=mean_zcr,
            noise_distribution=noise_dist,
            metrics=metrics,
        )

    @staticmethod
    def generate_synthetic_audio(
        duration_seconds: float = 30.0,
        sample_rate: int = 16000,
        signal_type: str = "classroom_lecture",
    ) -> np.ndarray:
        r"""Generate synthetic audio signals for testing and demonstration.

        Args:
            duration_seconds: Desired audio length in seconds.
            sample_rate: Audio sampling frequency.
            signal_type: Signal archetype:
                - 'silence': Pure ambient room noise ($\sigma = 0.001$).
                - 'sine_tone': Pure 440 Hz tone.
                - 'classroom_lecture': Alternating speech bursts (teacher talking) and student pauses.
                - 'high_noise': Classroom interruptions / heavy background commotion.

        Returns:
            np.ndarray: 1D mono float32 synthetic signal.
        """
        num_samples = int(duration_seconds * sample_rate)
        t = np.linspace(0.0, duration_seconds, num_samples, endpoint=False)

        if signal_type == "silence":
            # Very low Gaussian noise floor
            return np.random.normal(0, 0.001, num_samples).astype(np.float32)

        elif signal_type == "sine_tone":
            # 440 Hz sine wave with amplitude 0.2
            return (0.2 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)

        elif signal_type == "high_noise":
            # Loud broadband commotion
            noise = np.random.normal(0, 0.22, num_samples)
            return np.clip(noise, -1.0, 1.0).astype(np.float32)

        else:
            # "classroom_lecture": Periodic speech activity and silence pauses
            signal = np.zeros(num_samples, dtype=np.float32)
            # Add low room ambient floor
            signal += np.random.normal(0, 0.002, num_samples).astype(np.float32)

            # Modulate speech bursts every 5-10 seconds
            cycle_duration = 10.0  # 6s speech, 4s pause
            cycle_samples = int(cycle_duration * sample_rate)
            speech_samples = int(6.0 * sample_rate)

            for start_idx in range(0, num_samples, cycle_samples):
                end_idx = min(num_samples, start_idx + speech_samples)
                sub_t = t[start_idx:end_idx]
                # Harmonic speech envelope simulation: fundamental (150Hz) + formants
                speech_burst = (
                    0.08 * np.sin(2 * np.pi * 180 * sub_t)
                    + 0.05 * np.sin(2 * np.pi * 360 * sub_t)
                    + 0.03 * np.sin(2 * np.pi * 720 * sub_t)
                    + np.random.normal(0, 0.01, len(sub_t))
                )
                signal[start_idx:end_idx] += speech_burst.astype(np.float32)

            return np.clip(signal, -1.0, 1.0)
