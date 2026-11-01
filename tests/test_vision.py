"""Unit tests for FaceDetectorWrapper, VideoSource, and frame annotation."""

from pathlib import Path
import numpy as np
import pytest

from attendance_agent.config import get_config
from attendance_agent.vision.capture import VideoSource, annotate_frame
from attendance_agent.vision.detector import (
    FaceBoundingBox,
    FaceDetectorWrapper,
    RawDetection,
)
from attendance_agent.vision.tracker import AnonymousFaceTracker


def test_bounding_box_properties() -> None:
    """Verify FaceBoundingBox geometry properties and calculations."""
    bbox = FaceBoundingBox(x=50, y=80, w=120, h=160, norm_x=0.1, norm_y=0.2, norm_w=0.25, norm_h=0.33)
    assert bbox.x2 == 170
    assert bbox.y2 == 240
    assert bbox.area == 120 * 160
    assert bbox.center == (110.0, 160.0)


def test_detector_missing_model_raises_error(tmp_path: Path) -> None:
    """Verify that FaceDetectorWrapper raises FileNotFoundError when model is missing."""
    non_existent = tmp_path / "does_not_exist.task"
    with pytest.raises(FileNotFoundError) as exc_info:
        FaceDetectorWrapper(model_path=non_existent)
    assert "MediaPipe face detector model file not found" in str(exc_info.value)


def test_detector_inference_on_synthetic_frame() -> None:
    """Verify detector processes synthetic black frame without error and flags low lighting."""
    config = get_config()
    model_path = config.storage.face_model_path

    if not model_path.exists():
        pytest.skip("Local model file not yet available for detector inference test.")

    with FaceDetectorWrapper(model_path=model_path, min_detection_confidence=0.5) as detector:
        # Create dark/black frame (640x480, 3 channels)
        black_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        detections, metadata = detector.detect_faces(black_frame, timestamp=0.0)

        assert isinstance(detections, list)
        assert len(detections) == 0
        assert metadata["face_count"] == 0
        assert metadata["low_light"] is True
        assert metadata["brightness"] == 0.0


def test_video_source_handles_invalid_path(tmp_path: Path) -> None:
    """Verify VideoSource gracefully handles non-existent or invalid video files."""
    invalid_file = tmp_path / "corrupt_or_missing.mp4"
    source = VideoSource(source=invalid_file)
    success = source.open()

    assert success is False
    assert source.is_opened() is False
    ret, frame = source.read_frame()
    assert ret is False
    assert frame is None


def test_annotate_frame_rendering() -> None:
    """Verify annotate_frame produces valid annotated canvas without dimension distortion."""
    tracker = AnonymousFaceTracker()
    det = RawDetection(
        bbox=FaceBoundingBox(x=100, y=100, w=80, h=80),
        confidence=0.92,
        timestamp=0.0,
    )
    result = tracker.update([det], timestamp=0.0)

    # Blank canvas
    frame = np.ones((480, 640, 3), dtype=np.uint8) * 200
    annotated = annotate_frame(frame, result, low_light=True, show_confidence=True)

    assert annotated.shape == frame.shape
    assert annotated.dtype == np.uint8
    # Canvas should have modified pixels from drawings (not identical to raw canvas)
    assert not np.array_equal(annotated, frame)

