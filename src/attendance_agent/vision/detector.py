"""Face detection module using MediaPipe Face Detector Tasks API.

Provides local-first, privacy-preserving face detection without biometric
feature extraction, facial recognition, or cloud transmission.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

from attendance_agent.utils.logger import get_logger

logger = get_logger("attendance_agent.vision.detector")


@dataclass
class FaceBoundingBox:
    """Bounding box coordinates in pixel and normalized units."""

    x: int
    y: int
    w: int
    h: int
    norm_x: float = 0.0
    norm_y: float = 0.0
    norm_w: float = 0.0
    norm_h: float = 0.0

    @property
    def x2(self) -> int:
        """Right coordinate in pixels."""
        return self.x + self.w

    @property
    def y2(self) -> int:
        """Bottom coordinate in pixels."""
        return self.y + self.h

    @property
    def area(self) -> float:
        """Area in square pixels."""
        return float(self.w * self.h)

    @property
    def center(self) -> Tuple[float, float]:
        """Center coordinate (cx, cy) in pixels."""
        return (self.x + self.w / 2.0, self.y + self.h / 2.0)

    def iou(self, other: "FaceBoundingBox") -> float:
        """Compute Intersection over Union (IoU) with another bounding box.

        Args:
            other: Comparison FaceBoundingBox.

        Returns:
            float: IoU value between 0.0 and 1.0.
        """
        inter_x1 = max(self.x, other.x)
        inter_y1 = max(self.y, other.y)
        inter_x2 = min(self.x2, other.x2)
        inter_y2 = min(self.y2, other.y2)

        inter_w = max(0, inter_x2 - inter_x1)
        inter_h = max(0, inter_y2 - inter_y1)
        inter_area = float(inter_w * inter_h)

        if inter_area <= 0.0:
            return 0.0

        union_area = self.area + other.area - inter_area
        if union_area <= 0.0:
            return 0.0

        return inter_area / union_area


@dataclass
class RawDetection:
    """Structured detection event from the detector before track association."""

    bbox: FaceBoundingBox
    confidence: float
    timestamp: float


class FaceDetectorWrapper:
    """Local face detector using MediaPipe Tasks FaceDetector."""

    def __init__(
        self,
        model_path: Path | str,
        min_detection_confidence: float = 0.5,
    ) -> None:
        """Initialize the local FaceDetector.

        Args:
            model_path: Path to the local .task or .tflite model file.
            min_detection_confidence: Minimum score threshold (0.0 to 1.0).

        Raises:
            FileNotFoundError: If the model file is not present locally.
        """
        self.model_path = Path(model_path)
        self.min_confidence = min_detection_confidence

        if not self.model_path.exists():
            try:
                import urllib.request
                self.model_path.parent.mkdir(parents=True, exist_ok=True)
                url = "https://storage.googleapis.com/mediapipe-models/face_detector/blaze_face_short_range/float16/1/blaze_face_short_range.task"
                logger.info("Local model not found at '%s'. Downloading MediaPipe weights...", self.model_path)
                urllib.request.urlretrieve(url, str(self.model_path))
            except Exception as e:
                logger.warning("Auto-download of face detector model failed: %s", e)
                raise FileNotFoundError(
                    f"MediaPipe face detector model file not found at '{self.model_path}'. "
                    "Download the offline model file into data/models/ before starting."
                ) from e

        logger.info(
            "Initializing MediaPipe FaceDetector using model '%s' (min_confidence=%.2f)",
            self.model_path.name,
            self.min_confidence,
        )

        base_options = mp_python.BaseOptions(model_asset_path=str(self.model_path))
        options = vision.FaceDetectorOptions(
            base_options=base_options,
            min_detection_confidence=self.min_confidence,
        )
        self._detector: Optional[vision.FaceDetector] = vision.FaceDetector.create_from_options(options)

    def detect_faces(
        self,
        frame: np.ndarray,
        timestamp: float = 0.0,
    ) -> Tuple[List[RawDetection], Dict[str, Any]]:
        """Process an OpenCV BGR frame and return detected face bounding boxes.

        Args:
            frame: BGR numpy ndarray from OpenCV.
            timestamp: Relative or wall-clock timestamp in seconds.

        Returns:
            Tuple[List[RawDetection], Dict[str, Any]]: Detected faces and metadata.
        """
        if self._detector is None:
            raise RuntimeError("FaceDetector has already been closed.")

        if frame is None or frame.size == 0:
            return [], {"error": "Empty frame", "low_light": False, "brightness": 0.0}

        frame_h, frame_w = frame.shape[:2]

        # Check lighting condition via mean grayscale luminance
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        brightness = float(np.mean(gray))
        is_low_light = brightness < 35.0

        # MediaPipe expects RGB format
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

        # Run offline inference
        result = self._detector.detect(mp_image)

        detections: List[RawDetection] = []
        if result and result.detections:
            for det in result.detections:
                score = 0.0
                if det.categories:
                    score = float(det.categories[0].score or 0.0)

                # Clamp bounding box to frame boundaries
                box = det.bounding_box
                x = max(0, min(int(box.origin_x), frame_w - 1))
                y = max(0, min(int(box.origin_y), frame_h - 1))
                w = max(1, min(int(box.width), frame_w - x))
                h = max(1, min(int(box.height), frame_h - y))

                bbox = FaceBoundingBox(
                    x=x,
                    y=y,
                    w=w,
                    h=h,
                    norm_x=x / frame_w,
                    norm_y=y / frame_h,
                    norm_w=w / frame_w,
                    norm_h=h / frame_h,
                )
                detections.append(
                    RawDetection(
                        bbox=bbox,
                        confidence=score,
                        timestamp=timestamp,
                    )
                )

        metadata = {
            "frame_width": frame_w,
            "frame_height": frame_h,
            "face_count": len(detections),
            "low_light": is_low_light,
            "brightness": brightness,
        }

        return detections, metadata

    def close(self) -> None:
        """Release MediaPipe detector resources."""
        if self._detector is not None:
            self._detector.close()
            self._detector = None
            logger.info("FaceDetector resources released.")

    def __enter__(self) -> "FaceDetectorWrapper":
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.close()
