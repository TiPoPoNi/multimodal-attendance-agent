"""Command-line demonstration of the Computer Vision & Anonymous Face Tracking module.

Generates synthetic video frames with moving faces, executes face detection
and anonymous tracking, and displays structured output metrics without requiring
a physical webcam.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

# Ensure src is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import cv2
import numpy as np

from attendance_agent.config import get_config
from attendance_agent.vision.capture import annotate_frame
from attendance_agent.vision.detector import FaceBoundingBox, FaceDetectorWrapper, RawDetection
from attendance_agent.vision.tracker import AnonymousFaceTracker


def run_synthetic_demonstration() -> None:
    """Run simulated multi-person face tracking demonstration."""
    print("=" * 70)
    print("MULTIMODAL ATTENDANCE & ENGAGEMENT AGENT — VISION MODULE DEMO")
    print("=" * 70)

    config = get_config()
    print(f"Loading local FaceDetector from: {config.storage.face_model_path.name}")

    tracker = AnonymousFaceTracker(
        iou_threshold=config.vision.track_iou_threshold,
        max_inactive_frames=config.vision.track_max_inactive_frames,
    )

    print("\nSimulating 10 frames of classroom video with 3 participants...")
    print("Scenario: 2 stable students, 1 student temporarily looking away on frame 5-6.\n")

    # Participant 1 initial position: (100, 150)
    # Participant 2 initial position: (320, 160)
    # Participant 3 initial position: (500, 140)

    for frame_idx in range(1, 11):
        timestamp = frame_idx * 0.2  # 5 FPS simulation
        detections = []

        # Student 1: Slight natural head motion
        x1 = 100 + int(np.sin(frame_idx * 0.5) * 6)
        y1 = 150 + int(np.cos(frame_idx * 0.5) * 4)
        detections.append(
            RawDetection(
                bbox=FaceBoundingBox(x=x1, y=y1, w=90, h=110),
                confidence=0.94,
                timestamp=timestamp,
            )
        )

        # Student 2: Stable posture
        x2 = 320 + (frame_idx % 2) * 2
        y2 = 160
        detections.append(
            RawDetection(
                bbox=FaceBoundingBox(x=x2, y=y2, w=85, h=105),
                confidence=0.91,
                timestamp=timestamp,
            )
        )

        # Student 3: Disappears on frames 5 and 6 (temporary occlusion)
        if frame_idx not in (5, 6):
            x3 = 500 + int(np.cos(frame_idx * 0.4) * 8)
            y3 = 140
            detections.append(
                RawDetection(
                    bbox=FaceBoundingBox(x=x3, y=y3, w=95, h=115),
                    confidence=0.88,
                    timestamp=timestamp,
                )
            )

        # Update tracker
        result = tracker.update(detections, timestamp=timestamp)

        active_ids = [t.track_id for t in result.active_tracks]
        lost_ids = [t.track_id for t in result.lost_tracks]

        status_str = f"Active IDs: {active_ids}"
        if lost_ids:
            status_str += f" | Lost (Recoverable) IDs: {lost_ids}"

        print(
            f"Frame {frame_idx:02d} (t={timestamp:.1f}s) -> "
            f"Detected: {result.raw_detection_count} faces | {status_str}"
        )

    print("\nFinal Tracking Summary:")
    print("-" * 70)
    for track in tracker.get_tracks():
        print(
            f"Participant Track #{track.track_id}: "
            f"Status={track.status.upper():<14} | "
            f"Detections={track.total_detections:<2} | "
            f"First Seen={track.first_seen:.1f}s | "
            f"Last Seen={track.last_seen:.1f}s | "
            f"Total Movement={track.movement_distance:.1f}px"
        )
    print("=" * 70)
    print("Vision demonstration completed successfully.")


if __name__ == "__main__":
    run_synthetic_demonstration()

