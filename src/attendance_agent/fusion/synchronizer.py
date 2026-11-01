"""Multimodal signal synchronization module.

Aligns asynchronous visual tracking events and acoustic metrics into discrete,
unified time windows across a classroom session.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from attendance_agent.utils.logger import get_logger

logger = get_logger("attendance_agent.fusion.synchronizer")


@dataclass
class SynchronizedWindow:
    """Synchronized visual and acoustic feature snapshot for an individual time window."""

    window_index: int
    window_start: float
    window_end: float

    # Visual modality indicators
    has_visual_data: bool = False
    face_count: float = 0.0
    visual_presence_ratio: float = 0.0
    movement_distance: float = 0.0
    movement_level: float = 0.0  # Normalized [0.0, 1.0]
    mean_detection_confidence: float = 0.0
    frame_count: int = 0

    # Audio modality indicators
    has_audio_data: bool = False
    rms_energy: float = 0.0
    zero_crossing_rate: float = 0.0
    audio_activity_ratio: float = 0.0
    silence_ratio: float = 0.0
    noise_level: str = "low"

    def to_dict(self) -> Dict[str, Any]:
        """Convert synchronized window state to dictionary."""
        return {
            "window_index": self.window_index,
            "window_start": round(self.window_start, 2),
            "window_end": round(self.window_end, 2),
            "has_visual_data": self.has_visual_data,
            "face_count": round(self.face_count, 1),
            "visual_presence_ratio": round(self.visual_presence_ratio, 3),
            "movement_distance": round(self.movement_distance, 1),
            "movement_level": round(self.movement_level, 3),
            "mean_detection_confidence": round(self.mean_detection_confidence, 3),
            "frame_count": self.frame_count,
            "has_audio_data": self.has_audio_data,
            "rms_energy": round(self.rms_energy, 4),
            "zero_crossing_rate": round(self.zero_crossing_rate, 3),
            "audio_activity_ratio": round(self.audio_activity_ratio, 3),
            "silence_ratio": round(self.silence_ratio, 3),
            "noise_level": self.noise_level,
        }


class MultimodalSynchronizer:
    """Aligns video detection events and audio metrics on a unified timeline."""

    def __init__(
        self,
        window_duration_seconds: float = 5.0,
        movement_scale_px: float = 40.0,
    ) -> None:
        """Initialize synchronizer.

        Args:
            window_duration_seconds: Time interval for each synchronization slice.
            movement_scale_px: Movement reference in pixels corresponding to movement_level 1.0.
        """
        self.window_duration = max(1.0, window_duration_seconds)
        self.movement_scale = max(1.0, movement_scale_px)

    def synchronize_streams(
        self,
        session_duration_seconds: float,
        tracking_events: List[Dict[str, Any]],
        audio_metrics: List[Dict[str, Any]],
        start_offset: float = 0.0,
    ) -> List[SynchronizedWindow]:
        """Synchronize video tracking events and audio metrics into contiguous time windows.

        Args:
            session_duration_seconds: Total duration of observation session in seconds.
            tracking_events: List of tracking event dictionaries (with 'timestamp', 'track_id', etc.).
            audio_metrics: List of window audio metric dictionaries (with 'window_start', 'rms_energy', etc.).
            start_offset: Global start timestamp offset in seconds.

        Returns:
            List[SynchronizedWindow]: Chronological sequence of synchronized window objects.
        """
        dur = max(self.window_duration, session_duration_seconds)
        total_windows = int(math.ceil(dur / self.window_duration))

        # Group tracking events by frame timestamp
        events_by_time: Dict[float, List[Dict[str, Any]]] = {}
        for ev in tracking_events:
            ts = float(ev.get("timestamp", 0.0))
            events_by_time.setdefault(ts, []).append(ev)

        windows: List[SynchronizedWindow] = []

        for idx in range(total_windows):
            w_start = start_offset + (idx * self.window_duration)
            w_end = w_start + self.window_duration

            # 1. Visual Feature Aggregation for [w_start, w_end]
            win_events = [
                ev
                for ev in tracking_events
                if w_start <= float(ev.get("timestamp", 0.0)) < w_end
            ]

            has_vision = len(win_events) > 0
            face_count = 0.0
            vis_presence = 0.0
            movement_px = 0.0
            mean_conf = 0.0
            frame_cnt = 0

            if has_vision:
                # Group by unique timestamps within window (frames)
                frames_in_win: Dict[float, List[Dict[str, Any]]] = {}
                for ev in win_events:
                    frames_in_win.setdefault(float(ev["timestamp"]), []).append(ev)

                frame_cnt = len(frames_in_win)
                face_counts = [len(evs) for evs in frames_in_win.values()]
                face_count = float(np.mean(face_counts)) if face_counts else 0.0

                # Fraction of frames with at least 1 face detected
                vis_presence = 1.0 if face_count >= 1.0 else (face_count / 1.0)
                vis_presence = min(1.0, max(0.0, vis_presence))

                # Compute motion distance across consecutive frames per track
                tracks_positions: Dict[int, List[Tuple[float, float]]] = {}
                for ev in win_events:
                    tid = int(ev.get("track_id", 1))
                    cx = float(ev.get("bbox_x", 0.0)) + float(ev.get("bbox_w", 0.0)) / 2.0
                    cy = float(ev.get("bbox_y", 0.0)) + float(ev.get("bbox_h", 0.0)) / 2.0
                    tracks_positions.setdefault(tid, []).append((cx, cy))

                total_dist = 0.0
                for pts in tracks_positions.values():
                    for i in range(1, len(pts)):
                        dx = pts[i][0] - pts[i - 1][0]
                        dy = pts[i][1] - pts[i - 1][1]
                        total_dist += math.sqrt(dx * dx + dy * dy)

                movement_px = total_dist
                confidences = [float(ev.get("confidence", 0.0)) for ev in win_events if "confidence" in ev]
                mean_conf = float(np.mean(confidences)) if confidences else 0.0

            norm_movement = min(1.0, max(0.0, movement_px / self.movement_scale))

            # 2. Audio Feature Aggregation for [w_start, w_end]
            overlapping_audio = [
                a
                for a in audio_metrics
                if (
                    float(a.get("window_start", 0.0)) < w_end
                    and float(a.get("window_end", 0.0)) > w_start
                )
            ]

            has_audio = len(overlapping_audio) > 0
            rms = 0.0
            zcr = 0.0
            activity_ratio = 0.0
            silence_ratio = 0.0
            noise_level = "low"

            if has_audio:
                rms_list = [float(a.get("rms_energy", 0.0)) for a in overlapping_audio]
                zcr_list = [float(a.get("zero_crossing_rate", 0.0)) for a in overlapping_audio]
                active_list = [1 if a.get("activity_detected", False) else 0 for a in overlapping_audio]
                silence_list = [1 if a.get("is_silence", False) else 0 for a in overlapping_audio]

                rms = float(np.mean(rms_list))
                zcr = float(np.mean(zcr_list))
                activity_ratio = float(np.mean(active_list))
                silence_ratio = float(np.mean(silence_list))

                # Determine predominant noise level
                noise_counts: Dict[str, int] = {}
                for a in overlapping_audio:
                    lvl = a.get("noise_level", "low")
                    noise_counts[lvl] = noise_counts.get(lvl, 0) + 1
                noise_level = max(noise_counts.items(), key=lambda x: x[1])[0]

            window = SynchronizedWindow(
                window_index=idx + 1,
                window_start=w_start,
                window_end=w_end,
                has_visual_data=has_vision,
                face_count=face_count,
                visual_presence_ratio=vis_presence,
                movement_distance=movement_px,
                movement_level=norm_movement,
                mean_detection_confidence=mean_conf,
                frame_count=frame_cnt,
                has_audio_data=has_audio,
                rms_energy=rms,
                zero_crossing_rate=zcr,
                audio_activity_ratio=activity_ratio,
                silence_ratio=silence_ratio,
                noise_level=noise_level,
            )
            windows.append(window)

        return windows
