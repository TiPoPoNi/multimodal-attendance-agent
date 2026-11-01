"""Command-line demonstration of Multimodal Signal Fusion and Engagement Scoring.

Simulates aligned visual participant tracking and acoustic activity streams,
synchronizes time windows, normalizes metrics, and calculates explainable
session activity proxy scores with constituent contributions.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure src is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from attendance_agent.config import get_config
from attendance_agent.fusion.scoring import MultimodalFusionEngine
from attendance_agent.fusion.synchronizer import MultimodalSynchronizer


def run_fusion_demonstration() -> None:
    """Run multimodal fusion synchronization and scoring demonstration."""
    print("=" * 80)
    print("MULTIMODAL ATTENDANCE & ENGAGEMENT AGENT — FUSION & SCORING DEMO")
    print("=" * 80)

    config = get_config()
    print("Configuration:")
    print(f" - Window Duration: {config.fusion.window_duration_seconds}s")
    print(f" - Weights: Visual={config.fusion.weight_visual_presence:.2f}, Audio={config.fusion.weight_audio_activity:.2f}, Movement={config.fusion.weight_movement:.2f}")

    synchronizer = MultimodalSynchronizer(window_duration_seconds=config.fusion.window_duration_seconds)
    engine = MultimodalFusionEngine(
        weight_visual_presence=config.fusion.weight_visual_presence,
        weight_audio_activity=config.fusion.weight_audio_activity,
        weight_movement=config.fusion.weight_movement,
    )

    print("\nSimulating 20 seconds (4 windows) across different classroom phases:")
    print(" - Window 1 (0-5s):   Lecture Start (High Visual, Moderate Audio, Low Motion)")
    print(" - Window 2 (5-10s):  Q&A Interaction (High Visual, High Audio, High Motion)")
    print(" - Window 3 (10-15s): Individual Reading (Moderate Visual, Silent Audio, Low Motion)")
    print(" - Window 4 (15-20s): Missing Audio Modality (Camera-only fallback simulation)\n")

    # Synthetic tracking events
    events = [
        # Win 1: 2 students attentive
        {"timestamp": 1.0, "track_id": 1, "bbox_x": 100, "bbox_y": 100, "bbox_w": 80, "bbox_h": 80, "confidence": 0.95},
        {"timestamp": 1.0, "track_id": 2, "bbox_x": 300, "bbox_y": 100, "bbox_w": 80, "bbox_h": 80, "confidence": 0.92},
        {"timestamp": 3.0, "track_id": 1, "bbox_x": 102, "bbox_y": 101, "bbox_w": 80, "bbox_h": 80, "confidence": 0.94},
        # Win 2: high interaction & motion
        {"timestamp": 6.0, "track_id": 1, "bbox_x": 105, "bbox_y": 102, "bbox_w": 80, "bbox_h": 80, "confidence": 0.96},
        {"timestamp": 8.0, "track_id": 1, "bbox_x": 140, "bbox_y": 120, "bbox_w": 80, "bbox_h": 80, "confidence": 0.95},
        # Win 3: quiet reading
        {"timestamp": 12.0, "track_id": 1, "bbox_x": 140, "bbox_y": 120, "bbox_w": 80, "bbox_h": 80, "confidence": 0.90},
        # Win 4: student presence but camera only
        {"timestamp": 17.0, "track_id": 1, "bbox_x": 141, "bbox_y": 121, "bbox_w": 80, "bbox_h": 80, "confidence": 0.93},
    ]

    # Synthetic audio metrics (Win 4 has no audio data)
    audio_metrics = [
        {"window_start": 0.0, "window_end": 5.0, "rms_energy": 0.025, "zero_crossing_rate": 0.05, "activity_detected": True, "noise_level": "moderate"},
        {"window_start": 5.0, "window_end": 10.0, "rms_energy": 0.080, "zero_crossing_rate": 0.12, "activity_detected": True, "noise_level": "high"},
        {"window_start": 10.0, "window_end": 15.0, "rms_energy": 0.002, "zero_crossing_rate": 0.01, "activity_detected": False, "is_silence": True, "noise_level": "low"},
    ]

    windows = synchronizer.synchronize_streams(
        session_duration_seconds=20.0,
        tracking_events=events,
        audio_metrics=audio_metrics,
    )

    report = engine.score_session(windows)

    print(f"{'Win':<4} {'Time':<12} {'Visual':<8} {'Audio':<8} {'Motion':<8} {'Score':<8} {'Conf':<6} {'Label':<26} {'Contributions'}")
    print("-" * 105)

    for s in report.window_scores:
        t_range = f"{s.window_start:.0f}s-{s.window_end:.0f}s"
        vis_pct = f"{int(s.visual_presence * 100)}%"
        aud_pct = f"{int(s.audio_activity * 100)}%"
        mov_pct = f"{int(s.movement_level * 100)}%"
        conf_pct = f"{int(s.confidence * 100)}%"
        contribs = f"Vis:{s.visual_contribution:.1f} + Aud:{s.audio_contribution:.1f} + Mov:{s.movement_contribution:.1f}"

        print(
            f"#{s.window_index:<3} {t_range:<12} {vis_pct:<8} {aud_pct:<8} {mov_pct:<8} "
            f"{s.activity_score:<8.1f} {conf_pct:<6} {s.label:<26} {contribs}"
        )

    print("-" * 105)
    print(f"Overall Session Observed Activity: {report.mean_activity_score:.1f}/100 [{report.overall_label}] (Mean Conf: {int(report.mean_confidence * 100)}%)")
    print("=" * 80)
    print("Multimodal fusion demonstration completed successfully.")


if __name__ == "__main__":
    run_fusion_demonstration()

