"""Comprehensive edge-case and robustness test suite.

Validates application resilience against:
- Empty databases and non-existent sessions
- Corrupt, invalid, or truncated media files
- Silent audio and zero-energy signals
- Extreme or missing modality inputs
- Zero face detections and intermittent tracking loss
- Duplicate session identifiers and unique constraints
- Interrupted session recovery
"""

from pathlib import Path
import tempfile
import numpy as np
import pytest

from attendance_agent.audio.analyzer import AudioAnalyzer
from attendance_agent.audio.capture import AudioSourceLoader
from attendance_agent.config import AppConfig, get_config
from attendance_agent.fusion.scoring import MultimodalFusionEngine
from attendance_agent.fusion.synchronizer import SynchronizedWindow
from attendance_agent.models import AttendanceStatus, SessionStatus
from attendance_agent.storage.database import DatabaseManager, init_db
from attendance_agent.vision.attendance import PresenceAttendanceEngine
from attendance_agent.vision.capture import VideoSource
from attendance_agent.vision.detector import FaceBoundingBox, RawDetection
from attendance_agent.vision.tracker import AnonymousFaceTracker


@pytest.fixture
def clean_db(tmp_path: Path) -> DatabaseManager:
    """Fixture providing initialized empty test database."""
    db_file = tmp_path / "test_edge_cases.db"
    init_db(db_file)
    return DatabaseManager(db_file)


# ==============================================================================
# 1. DATABASE EDGE CASES
# ==============================================================================

def test_empty_database_queries(clean_db: DatabaseManager) -> None:
    """Verify clean behavior when querying an empty database."""
    stats = clean_db.count_overview_stats()
    assert stats["total_sessions"] == 0
    assert stats["total_attendance_records"] == 0
    assert stats["avg_presence_percentage"] == 0.0

    # Non-existent session
    assert clean_db.get_session("non-existent-id") is None
    assert clean_db.get_attendance_records("non-existent-id") == []
    assert clean_db.get_audio_metrics("non-existent-id") == []
    assert clean_db.get_fusion_scores("non-existent-id") == []
    assert clean_db.get_recommendations("non-existent-id") == []
    assert clean_db.get_unclosed_sessions() == []


def test_duplicate_session_creation(clean_db: DatabaseManager) -> None:
    """Verify duplicate session creation handles constraints gracefully."""
    sess_id = "test-session-duplicate"
    clean_db.create_session(sess_id, "Initial Title")

    # Creating session with duplicate ID should raise IntegrityError
    import sqlite3
    with pytest.raises(sqlite3.IntegrityError):
        clean_db.create_session(sess_id, "Duplicate Title")


def test_interrupted_session_recovery(clean_db: DatabaseManager) -> None:
    """Verify recovery of session that crashed before clean completion."""
    sess_id = "crashed-session-001"
    clean_db.create_session(sess_id, "Crashed Lecture")

    # Record some tracking events before crash
    events = [
        {"session_id": sess_id, "track_id": 1, "timestamp": 10.0, "bbox_x": 50.0, "bbox_y": 50.0, "bbox_w": 80.0, "bbox_h": 90.0, "confidence": 0.9},
        {"session_id": sess_id, "track_id": 1, "timestamp": 20.0, "bbox_x": 52.0, "bbox_y": 50.0, "bbox_w": 80.0, "bbox_h": 90.0, "confidence": 0.9},
        {"session_id": sess_id, "track_id": 2, "timestamp": 15.0, "bbox_x": 200.0, "bbox_y": 60.0, "bbox_w": 80.0, "bbox_h": 90.0, "confidence": 0.9},
    ]
    clean_db.record_tracking_events_batch(events)

    # Session is unclosed
    unclosed = clean_db.get_unclosed_sessions()
    assert any(s["id"] == sess_id for s in unclosed)

    # Recovery engine parses events and reconstructs attendance
    engine = PresenceAttendanceEngine(min_presence_percentage=50.0, min_presence_duration_seconds=5.0)
    summary = engine.recover_session(clean_db, sess_id)

    assert summary is not None
    assert summary.session_id == sess_id
    assert summary.total_participants == 2

    # Attendance records should now be in database
    records = clean_db.get_attendance_records(sess_id)
    assert len(records) == 2


# ==============================================================================
# 2. MEDIA AND CAPTURE EDGE CASES
# ==============================================================================

def test_invalid_video_source() -> None:
    """Verify invalid or non-existent video path fails gracefully."""
    v = VideoSource("non_existent_video_file_xyz.mp4")
    assert v.open() is False
    ret, frame = v.read_frame()
    assert ret is False
    assert frame is None
    v.release()


def test_corrupt_video_file(tmp_path: Path) -> None:
    """Verify corrupt video file does not crash the application."""
    corrupt_file = tmp_path / "corrupt.mp4"
    corrupt_file.write_bytes(b"NOT_A_REAL_VIDEO_HEADER_OR_DATA" * 100)

    v = VideoSource(str(corrupt_file))
    # OpenCV should either fail to open or fail to read first frame
    opened = v.open()
    if opened:
        ret, frame = v.read_frame()
        assert ret is False
    v.release()


def test_corrupt_audio_file(tmp_path: Path) -> None:
    """Verify AudioSourceLoader raises ValueError or RuntimeError on corrupt audio data."""
    corrupt_file = tmp_path / "corrupt.wav"
    corrupt_file.write_bytes(b"INVALID_WAV_BYTES_DATA_STREAM")

    with pytest.raises((ValueError, RuntimeError, Exception)):
        AudioSourceLoader.load_audio(corrupt_file, target_sample_rate=16000)


def test_silent_audio_signal() -> None:
    """Verify completely silent audio is correctly identified without NaN errors."""
    analyzer = AudioAnalyzer(sample_rate=16000, window_duration_seconds=2.0)
    # Zero signal
    silent = np.zeros(32000, dtype=np.float32)
    report = analyzer.analyze_audio_signal(silent)

    assert report.window_count == 1
    assert report.metrics[0].rms_energy == pytest.approx(0.0, abs=1e-6)
    assert report.metrics[0].activity_detected is False
    assert report.metrics[0].noise_level == "low"
    assert report.activity_ratio == 0.0


# ==============================================================================
# 3. VISION & TRACKING EDGE CASES
# ==============================================================================

def test_tracker_zero_detections() -> None:
    """Verify tracker handles consecutive empty frames without crashing."""
    tracker = AnonymousFaceTracker(max_inactive_frames=3)

    # 5 frames of zero detections
    for i in range(5):
        result = tracker.update([], timestamp=i * 0.1)
        assert len(result.active_tracks) == 0
        assert result.active_count == 0
        assert result.raw_detection_count == 0


def test_tracker_multiple_faces_and_occlusion() -> None:
    """Verify tracker manages multiple concurrent faces and recovers after 1-frame disappearance."""
    tracker = AnonymousFaceTracker(iou_threshold=0.3, max_inactive_frames=5)

    # Frame 1: 3 faces
    f1 = [
        RawDetection(FaceBoundingBox(10, 10, 50, 50), 0.9, 0.0),
        RawDetection(FaceBoundingBox(100, 10, 50, 50), 0.9, 0.0),
        RawDetection(FaceBoundingBox(200, 10, 50, 50), 0.9, 0.0),
    ]
    r1 = tracker.update(f1, timestamp=0.0)
    assert len(r1.active_tracks) == 3
    track_ids = {t.track_id for t in r1.active_tracks}
    assert len(track_ids) == 3

    # Frame 2: Face #2 disappeared (momentary occlusion)
    f2 = [
        RawDetection(FaceBoundingBox(12, 11, 50, 50), 0.9, 0.1),
        RawDetection(FaceBoundingBox(202, 10, 50, 50), 0.9, 0.1),
    ]
    r2 = tracker.update(f2, timestamp=0.1)
    assert len(r2.active_tracks) == 2
    assert len(r2.all_tracks) == 3  # occluded track preserved as inactive

    # Frame 3: Face #2 reappears
    f3 = [
        RawDetection(FaceBoundingBox(14, 12, 50, 50), 0.9, 0.2),
        RawDetection(FaceBoundingBox(102, 11, 50, 50), 0.9, 0.2),
        RawDetection(FaceBoundingBox(204, 11, 50, 50), 0.9, 0.2),
    ]
    r3 = tracker.update(f3, timestamp=0.2)
    assert len(r3.active_tracks) == 3
    # Same original track IDs preserved!
    assert {t.track_id for t in r3.active_tracks} == track_ids


# ==============================================================================
# 4. FUSION MISSING MODALITIES
# ==============================================================================

def test_fusion_neither_modality_present() -> None:
    """Verify fusion engine handles windows where both video and audio are absent."""
    engine = MultimodalFusionEngine()
    empty_win = SynchronizedWindow(
        window_index=0,
        window_start=0.0,
        window_end=5.0,
        visual_presence_ratio=0.0,
        movement_level=0.0,
        mean_detection_confidence=0.0,
        audio_activity_ratio=0.0,
        rms_energy=0.0,
        noise_level="low",
        has_visual_data=False,
        has_audio_data=False,
    )
    score = engine.score_window(empty_win)
    assert score.activity_score == 0.0
    assert score.confidence == 0.0
    assert score.label == "Insufficient data"
    assert score.modalities_present == []
