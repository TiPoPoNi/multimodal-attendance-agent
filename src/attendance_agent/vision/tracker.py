"""Anonymous face tracking module using Intersection-over-Union (IoU) matching.

Maintains consistent participant IDs across frames locally without computing
biometric embeddings or matching against human identity databases.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

from attendance_agent.vision.detector import FaceBoundingBox, RawDetection
from attendance_agent.utils.logger import get_logger

logger = get_logger("attendance_agent.vision.tracker")


@dataclass
class TrackedParticipant:
    """Represents an ongoing tracked face throughout a session."""

    track_id: int
    bbox: FaceBoundingBox
    confidence: float
    first_seen: float
    last_seen: float
    inactive_frames: int = 0
    total_detections: int = 1
    status: str = "active"  # "active", "temporary_lost", "inactive"
    movement_distance: float = 0.0
    history: List[Tuple[float, float, float]] = field(default_factory=list)  # (time, cx, cy)

    def to_dict(self) -> Dict[str, Any]:
        """Convert track state to dictionary representation."""
        cx, cy = self.bbox.center
        return {
            "track_id": self.track_id,
            "status": self.status,
            "confidence": round(self.confidence, 3),
            "first_seen": self.first_seen,
            "last_seen": self.last_seen,
            "inactive_frames": self.inactive_frames,
            "total_detections": self.total_detections,
            "movement_distance": round(self.movement_distance, 2),
            "bbox": {
                "x": self.bbox.x,
                "y": self.bbox.y,
                "w": self.bbox.w,
                "h": self.bbox.h,
                "norm_x": round(self.bbox.norm_x, 4),
                "norm_y": round(self.bbox.norm_y, 4),
                "norm_w": round(self.bbox.norm_w, 4),
                "norm_h": round(self.bbox.norm_h, 4),
                "center_x": round(cx, 1),
                "center_y": round(cy, 1),
            },
        }


@dataclass
class TrackingFrameResult:
    """Complete structured tracking output for a single processed frame."""

    timestamp: float
    frame_index: int
    active_tracks: List[TrackedParticipant]
    lost_tracks: List[TrackedParticipant]
    all_tracks: List[TrackedParticipant]
    active_count: int
    raw_detection_count: int

    def to_dict(self) -> Dict[str, Any]:
        """Convert frame result to dictionary."""
        return {
            "timestamp": self.timestamp,
            "frame_index": self.frame_index,
            "active_count": self.active_count,
            "raw_detection_count": self.raw_detection_count,
            "tracks": [t.to_dict() for t in self.all_tracks],
        }


class AnonymousFaceTracker:
    """Tracks detected faces across consecutive video frames anonymously."""

    def __init__(
        self,
        iou_threshold: float = 0.3,
        max_inactive_frames: int = 15,
        max_disappearance_seconds: float = 2.0,
    ) -> None:
        """Initialize the anonymous tracker.

        Args:
            iou_threshold: Minimum IoU required to associate a detection with an existing track.
            max_inactive_frames: Number of consecutive frames without detection before marking inactive.
            max_disappearance_seconds: Maximum time gap in seconds for temporary disappearance recovery.
        """
        self.iou_threshold = iou_threshold
        self.max_inactive_frames = max_inactive_frames
        self.max_disappearance_seconds = max_disappearance_seconds

        self._next_track_id: int = 1
        self._tracks: Dict[int, TrackedParticipant] = {}
        self._frame_index: int = 0

    @property
    def next_track_id(self) -> int:
        """Get the next sequential track ID."""
        return self._next_track_id

    def update(
        self,
        detections: List[RawDetection],
        timestamp: float,
    ) -> TrackingFrameResult:
        """Update tracker state with detections from the current frame.

        Args:
            detections: List of RawDetection objects from the FaceDetector.
            timestamp: Frame timestamp in seconds.

        Returns:
            TrackingFrameResult: Structured tracking results for the frame.
        """
        self._frame_index += 1

        existing_track_ids = list(self._tracks.keys())
        matched_track_ids: Set[int] = set()
        matched_det_indices: Set[int] = set()

        # Step 1: Candidate matching via IoU matrix
        if existing_track_ids and detections:
            candidates: List[Tuple[float, int, int]] = []
            for track_id in existing_track_ids:
                track = self._tracks[track_id]
                for det_idx, det in enumerate(detections):
                    score = track.bbox.iou(det.bbox)
                    if score >= self.iou_threshold:
                        candidates.append((score, track_id, det_idx))

            # Sort by highest IoU first (greedy assignment)
            candidates.sort(key=lambda x: x[0], reverse=True)

            for score, track_id, det_idx in candidates:
                if track_id in matched_track_ids or det_idx in matched_det_indices:
                    continue

                # Match confirmed: update existing track
                matched_track_ids.add(track_id)
                matched_det_indices.add(det_idx)

                det = detections[det_idx]
                track = self._tracks[track_id]

                # Compute motion distance from previous center
                old_cx, old_cy = track.bbox.center
                new_cx, new_cy = det.bbox.center
                dist = math.sqrt((new_cx - old_cx) ** 2 + (new_cy - old_cy) ** 2)

                track.bbox = det.bbox
                track.confidence = det.confidence
                track.last_seen = timestamp
                track.inactive_frames = 0
                track.total_detections += 1
                track.status = "active"
                track.movement_distance += dist
                track.history.append((timestamp, new_cx, new_cy))
                # Keep sliding window of history positions
                if len(track.history) > 60:
                    track.history.pop(0)

        # Step 2: Handle unmatched existing tracks (disappearance / occlusion)
        for track_id in existing_track_ids:
            if track_id not in matched_track_ids:
                track = self._tracks[track_id]
                track.inactive_frames += 1
                gap_seconds = timestamp - track.last_seen

                if (
                    track.inactive_frames <= self.max_inactive_frames
                    and gap_seconds <= self.max_disappearance_seconds
                ):
                    track.status = "temporary_lost"
                else:
                    track.status = "inactive"

        # Step 3: Handle unmatched detections (new participants entering view)
        for det_idx, det in enumerate(detections):
            if det_idx not in matched_det_indices:
                new_id = self._next_track_id
                self._next_track_id += 1

                cx, cy = det.bbox.center
                new_track = TrackedParticipant(
                    track_id=new_id,
                    bbox=det.bbox,
                    confidence=det.confidence,
                    first_seen=timestamp,
                    last_seen=timestamp,
                    inactive_frames=0,
                    total_detections=1,
                    status="active",
                    movement_distance=0.0,
                    history=[(timestamp, cx, cy)],
                )
                self._tracks[new_id] = new_track
                logger.debug("New participant track assigned: ID #%d", new_id)

        # Step 4: Purge obsolete inactive tracks that exceeded max timeout
        purge_limit = self.max_inactive_frames * 3
        to_remove = [
            tid
            for tid, t in self._tracks.items()
            if t.status == "inactive" and t.inactive_frames > purge_limit
        ]
        for tid in to_remove:
            del self._tracks[tid]

        # Step 5: Build structured frame output
        active_list = [t for t in self._tracks.values() if t.status == "active"]
        lost_list = [t for t in self._tracks.values() if t.status == "temporary_lost"]
        all_list = list(self._tracks.values())

        return TrackingFrameResult(
            timestamp=timestamp,
            frame_index=self._frame_index,
            active_tracks=active_list,
            lost_tracks=lost_list,
            all_tracks=all_list,
            active_count=len(active_list),
            raw_detection_count=len(detections),
        )

    def get_tracks(self) -> List[TrackedParticipant]:
        """Return all currently managed tracks."""
        return list(self._tracks.values())

    def reset(self) -> None:
        """Reset all tracked participants and counters."""
        self._tracks.clear()
        self._next_track_id = 1
        self._frame_index = 0
        logger.info("AnonymousFaceTracker reset to initial state.")
