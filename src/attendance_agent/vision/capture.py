"""Video input capture and visualization module.

Supports live webcams and uploaded video files using OpenCV.
Handles video decoding, frame resizing, resource safety, and overlay annotation
without persisting raw video frames to disk.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import cv2
import numpy as np

from attendance_agent.vision.tracker import TrackedParticipant, TrackingFrameResult
from attendance_agent.utils.logger import get_logger

logger = get_logger("attendance_agent.vision.capture")


class VideoSource:
    """Manages OpenCV video capture for live camera or recorded video file."""

    def __init__(
        self,
        source: Union[int, str, Path],
        target_width: int = 640,
        target_height: int = 480,
    ) -> None:
        """Initialize video source.

        Args:
            source: Camera device index (int) or file system path (str/Path).
            target_width: Desired frame width for resizing.
            target_height: Desired frame height for resizing.
        """
        self.source = source
        self.target_width = target_width
        self.target_height = target_height
        self._cap: Optional[cv2.VideoCapture] = None
        self.is_live: bool = isinstance(source, int)

    def open(self) -> bool:
        """Open the video capture device or file.

        Returns:
            bool: True if video capture was successfully initialized.
        """
        self.release()

        src_arg = self.source
        if isinstance(src_arg, Path):
            src_arg = str(src_arg)

        logger.info("Opening video source: %s", src_arg)
        try:
            if self.is_live:
                # Use DirectShow on Windows for fast webcam init
                self._cap = cv2.VideoCapture(int(src_arg), cv2.CAP_DSHOW)
                if not self._cap.isOpened():
                    # Fallback to default backend
                    self._cap = cv2.VideoCapture(int(src_arg))
            else:
                self._cap = cv2.VideoCapture(str(src_arg))

            if not self._cap.isOpened():
                logger.warning("Failed to open video source: %s", src_arg)
                return False

            return True
        except Exception as exc:
            logger.error("Exception opening video source %s: %s", src_arg, exc)
            return False

    def is_opened(self) -> bool:
        """Check if capture stream is currently open."""
        return self._cap is not None and self._cap.isOpened()

    def read_frame(self) -> Tuple[bool, Optional[np.ndarray]]:
        """Read and resize the next frame from video stream.

        Returns:
            Tuple[bool, Optional[np.ndarray]]: (success_flag, bgr_frame)
        """
        if not self.is_opened():
            return False, None

        ret, frame = self._cap.read()
        if not ret or frame is None:
            return False, None

        # Resize if dimensions differ from target
        h, w = frame.shape[:2]
        if w != self.target_width or h != self.target_height:
            frame = cv2.resize(
                frame,
                (self.target_width, self.target_height),
                interpolation=cv2.INTER_AREA,
            )

        return True, frame

    def get_properties(self) -> Dict[str, Any]:
        """Retrieve video stream metadata.

        Returns:
            Dict[str, Any]: Dimensions, FPS, and total frames count.
        """
        if not self.is_opened():
            return {"opened": False, "is_live": self.is_live}

        fps = self._cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT)) if not self.is_live else -1
        width = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        return {
            "opened": True,
            "is_live": self.is_live,
            "fps": fps if fps > 0 else 15.0,
            "total_frames": total_frames,
            "width": width,
            "height": height,
        }

    def release(self) -> None:
        """Release underlying OpenCV VideoCapture resources."""
        if self._cap is not None:
            self._cap.release()
            self._cap = None
            logger.debug("Released video source: %s", self.source)

    def __enter__(self) -> "VideoSource":
        self.open()
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.release()


def annotate_frame(
    frame: np.ndarray,
    tracking_result: TrackingFrameResult,
    low_light: bool = False,
    show_confidence: bool = True,
) -> np.ndarray:
    """Draw bounding boxes and participant track IDs onto a copy of the frame.

    Args:
        frame: Original BGR frame.
        tracking_result: Frame tracking output from AnonymousFaceTracker.
        low_light: If True, draws a subtle lighting warning banner.
        show_confidence: If True, prints detection confidence percentage.

    Returns:
        np.ndarray: Annotated BGR frame for preview.
    """
    canvas = frame.copy()

    for track in tracking_result.all_tracks:
        if track.status == "inactive":
            continue

        bbox = track.bbox
        x, y, w, h = bbox.x, bbox.y, bbox.w, bbox.h

        # Color coding: Green for active, Amber/Orange for temporary_lost
        if track.status == "active":
            color = (40, 180, 50)  # BGR green
            label_text = f"ID #{track.track_id}"
            if show_confidence and track.confidence > 0:
                label_text += f" ({int(track.confidence * 100)}%)"
        else:
            color = (30, 160, 240)  # BGR amber/orange
            label_text = f"ID #{track.track_id} (Lost)"

        # Draw bounding box
        cv2.rectangle(canvas, (x, y), (x + w, y + h), color, 2)

        # Label background banner
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        font_thickness = 1
        (tw, th), baseline = cv2.getTextSize(label_text, font, font_scale, font_thickness)

        label_y1 = max(0, y - th - 8)
        label_y2 = y
        cv2.rectangle(
            canvas,
            (x, label_y1),
            (x + tw + 8, label_y2),
            color,
            -1,
        )

        # Draw label text
        cv2.putText(
            canvas,
            label_text,
            (x + 4, y - 4),
            font,
            font_scale,
            (255, 255, 255),
            font_thickness,
            cv2.LINE_AA,
        )

    # Optional low lighting warning banner
    if low_light:
        cv2.rectangle(canvas, (10, 10), (220, 38), (20, 20, 20), -1)
        cv2.rectangle(canvas, (10, 10), (220, 38), (0, 180, 255), 1)
        cv2.putText(
            canvas,
            "Low Lighting Detected",
            (18, 29),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (0, 220, 255),
            1,
            cv2.LINE_AA,
        )

    # Participant count badge
    count_text = f"Participants: {tracking_result.active_count}"
    fw = canvas.shape[1]
    cv2.putText(
        canvas,
        count_text,
        (fw - 160, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        canvas,
        count_text,
        (fw - 160, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (40, 180, 50),
        1,
        cv2.LINE_AA,
    )

    return canvas

