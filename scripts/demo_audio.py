"""Command-line demonstration of the Audio Processing and Sound Analysis Module.

Generates synthetic classroom audio signals (lecture with speech bursts and silence pauses),
computes windowed RMS energy and Zero-Crossing Rate, and displays structured telemetry.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure src is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from attendance_agent.audio.analyzer import AudioAnalyzer
from attendance_agent.config import get_config


def run_audio_demonstration() -> None:
    """Run simulated classroom audio activity analysis."""
    print("=" * 75)
    print("MULTIMODAL ATTENDANCE & ENGAGEMENT AGENT — AUDIO MODULE DEMO")
    print("=" * 75)

    config = get_config()
    analyzer = AudioAnalyzer(
        sample_rate=config.audio.sample_rate,
        window_duration_seconds=config.audio.window_duration_seconds,
        rms_activity_threshold=config.audio.rms_activity_threshold,
        zcr_activity_threshold=config.audio.zcr_activity_threshold,
    )

    print("\nGenerating 30 seconds of synthetic classroom lecture audio...")
    print("Configuration:")
    print(f" - Sample Rate: {config.audio.sample_rate} Hz")
    print(f" - Window Size: {config.audio.window_duration_seconds} seconds")
    print(f" - RMS Activity Threshold: {config.audio.rms_activity_threshold}")
    print(f" - ZCR Activity Threshold: {config.audio.zcr_activity_threshold}\n")

    signal = AudioAnalyzer.generate_synthetic_audio(
        duration_seconds=30.0,
        sample_rate=config.audio.sample_rate,
        signal_type="classroom_lecture",
    )

    report = analyzer.analyze_audio_signal(signal)

    print(f"Total Duration: {report.total_duration_seconds:.1f}s | Total Windows: {report.window_count}")
    print(f"Active Windows: {report.active_window_count} | Activity Ratio: {report.activity_ratio:.1f}%")
    print(f"Silence Duration: {report.silence_duration_seconds:.1f}s")
    print(f"Mean RMS: {report.mean_rms_energy:.4f} | Mean ZCR: {report.mean_zcr:.4f}")
    print(f"Noise Distribution: {report.noise_distribution}\n")

    print(f"{'Window':<8} {'Time (s)':<14} {'RMS Energy':<12} {'ZCR':<10} {'Activity':<12} {'Noise Level'}")
    print("-" * 75)

    for m in report.metrics:
        time_range = f"{m.window_start:.1f}s - {m.window_end:.1f}s"
        activity_str = "ACTIVE" if m.activity_detected else "QUIET"
        print(
            f"#{m.window_index:<7} {time_range:<14} {m.rms_energy:<12.4f} "
            f"{m.zero_crossing_rate:<10.3f} {activity_str:<12} {m.noise_level.upper()}"
        )

    print("=" * 75)
    print("Audio processing demonstration completed successfully.")


if __name__ == "__main__":
    run_audio_demonstration()

