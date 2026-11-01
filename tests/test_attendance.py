"""Unit tests for the presence-based attendance evaluation engine.

Validates interval calculations, sustained presence thresholds, short appearance filtering,
borderline status evaluation, session recovery from SQLite, and CSV reporting.
"""

from pathlib import Path
import pytest

from attendance_agent.models import AttendanceStatus
from attendance_agent.storage.database import DatabaseManager, init_db
from attendance_agent.vision.attendance import (
    PresenceAttendanceEngine,
    SessionAttendanceSummary,
)


@pytest.fixture
def test_db(tmp_path: Path) -> DatabaseManager:
    """Fixture providing an initialized SQLite test database."""
    db_file = tmp_path / "test_attendance_engine.db"
    return DatabaseManager(db_file)


def test_continuous_presence_marks_present() -> None:
    """Verify continuous presence across a session marks participant as Present."""
    engine = PresenceAttendanceEngine(
        min_presence_percentage=75.0,
        min_presence_duration_seconds=300.0,
        max_disappearance_seconds=30.0,
    )
    session_duration = 600.0  # 10 minutes

    # Continuous detections every 2 seconds from 0 to 600s
    timestamps = [float(t) for t in range(0, 601, 2)]

    result = engine.evaluate_track(
        track_id=1,
        timestamps=timestamps,
        session_duration_seconds=session_duration,
    )

    assert result.status == AttendanceStatus.PRESENT.value
    assert result.detection_intervals == 1
    assert result.presence_duration_seconds >= 600.0
    assert result.presence_percentage == pytest.approx(100.0, rel=1e-2)


def test_short_appearance_marks_insufficient() -> None:
    """Verify a momentary appearance (< threshold) is marked Insufficient Presence."""
    engine = PresenceAttendanceEngine(
        min_presence_percentage=75.0,
        min_presence_duration_seconds=300.0,
        max_disappearance_seconds=30.0,
    )
    session_duration = 600.0

    # Face detected only for 10 seconds at the beginning
    timestamps = [float(t) for t in range(0, 11, 2)]

    result = engine.evaluate_track(
        track_id=2,
        timestamps=timestamps,
        session_duration_seconds=session_duration,
    )

    assert result.status == AttendanceStatus.INSUFFICIENT_PRESENCE.value
    assert result.presence_duration_seconds == pytest.approx(10.0, abs=1.0)
    assert result.presence_percentage < 5.0


def test_intermittent_appearances_with_multiple_intervals() -> None:
    """Verify gaps exceeding max_disappearance create separate intervals and are deducted."""
    engine = PresenceAttendanceEngine(
        min_presence_percentage=60.0,
        min_presence_duration_seconds=200.0,
        max_disappearance_seconds=20.0,
    )
    session_duration = 600.0

    # Interval 1: 0 to 120s (duration = 120s)
    block1 = [float(t) for t in range(0, 121, 5)]
    # Big absence: from 120s to 300s (gap = 180s)
    # Interval 2: 300 to 450s (duration = 150s)
    block2 = [float(t) for t in range(300, 451, 5)]

    all_ts = block1 + block2
    result = engine.evaluate_track(
        track_id=3,
        timestamps=all_ts,
        session_duration_seconds=session_duration,
    )

    assert result.detection_intervals == 2
    # Total duration should be 120 + 150 = 270s (absence is deducted)
    assert result.presence_duration_seconds == pytest.approx(270.0, abs=2.0)
    # Presence %: 270 / 600 = 45%
    assert result.presence_percentage == pytest.approx(45.0, rel=1e-2)
    # 45% is below 60% requirement -> Insufficient Presence
    assert result.status == AttendanceStatus.INSUFFICIENT_PRESENCE.value


def test_borderline_presence_marks_pending_confirmation() -> None:
    """Verify borderline presence is classified as Pending Confirmation for teacher review."""
    engine = PresenceAttendanceEngine(
        min_presence_percentage=75.0,
        min_presence_duration_seconds=300.0,
        borderline_tolerance_percentage=10.0,  # 65% to 75% is borderline
    )
    session_duration = 500.0

    # Student present for 350s (70% presence, > 300s duration, but < 75%)
    timestamps = [float(t) for t in range(0, 351, 2)]

    result = engine.evaluate_track(
        track_id=4,
        timestamps=timestamps,
        session_duration_seconds=session_duration,
        is_in_progress=False,
    )

    assert result.status == AttendanceStatus.PENDING_CONFIRMATION.value
    assert result.presence_percentage == pytest.approx(70.0, rel=1e-2)


def test_timestamp_validation_and_cleaning() -> None:
    """Verify invalid or unsorted timestamps are cleaned gracefully."""
    engine = PresenceAttendanceEngine()

    # Unsorted, negative, and duplicate timestamps
    noisy_ts = [50.0, -10.0, 20.0, 50.0, 10.0, 0.0, -1.0, 30.0]
    intervals, total_dur = engine.calculate_presence_intervals(noisy_ts)

    # Valid sorted sequence: 0, 10, 20, 30, 50 (gap between 30 and 50 is 20 <= 30) -> 1 interval [0, 50]
    assert len(intervals) == 1
    assert intervals[0].start == 0.0
    assert intervals[0].end == 50.0
    assert total_dur == 50.0


def test_roster_mapping_preservation() -> None:
    """Verify instructor roster mappings are correctly associated with track metrics."""
    engine = PresenceAttendanceEngine(min_presence_percentage=50.0, min_presence_duration_seconds=100.0)
    timestamps = [float(t) for t in range(0, 300, 2)]

    summary = engine.evaluate_track(
        track_id=1,
        timestamps=timestamps,
        session_duration_seconds=300.0,
        roster_name="Alice Smith",
        student_id="CS-2024-01",
        notes="Attentive front row",
    )

    assert summary.roster_name == "Alice Smith"
    assert summary.student_id == "CS-2024-01"
    assert summary.notes == "Attentive front row"
    assert summary.status == AttendanceStatus.PRESENT.value


def test_session_recovery_from_database(test_db: DatabaseManager) -> None:
    """Verify recovering attendance records for an unclosed session from SQLite tracking events."""
    session_id = "sess-recovery-test-01"
    test_db.create_session(session_id, "Operating Systems Lecture")

    # Manually map participant 1 to Bob
    test_db.map_roster_participant(
        session_id=session_id,
        track_id=1,
        roster_name="Bob Jones",
        student_id="CS-2024-02",
    )

    # Simulate 50 tracking events for track 1 and 10 events for track 2
    events = []
    for t in range(0, 50, 1):
        events.append(
            {
                "session_id": session_id,
                "track_id": 1,
                "timestamp": float(t),
                "bbox_x": 100.0,
                "bbox_y": 100.0,
                "bbox_w": 80.0,
                "bbox_h": 80.0,
                "confidence": 0.95,
            }
        )

    for t in range(0, 10, 1):
        events.append(
            {
                "session_id": session_id,
                "track_id": 2,
                "timestamp": float(t),
                "bbox_x": 300.0,
                "bbox_y": 100.0,
                "bbox_w": 80.0,
                "bbox_h": 80.0,
                "confidence": 0.92,
            }
        )

    inserted_count = test_db.record_tracking_events_batch(events)
    assert inserted_count == 60

    # Recover session using engine
    engine = PresenceAttendanceEngine(
        min_presence_percentage=50.0,
        min_presence_duration_seconds=30.0,
    )
    recovered = engine.recover_session(test_db, session_id)

    assert recovered is not None
    assert recovered.total_participants == 2
    assert len(recovered.records) == 2

    track1_rec = next(r for r in recovered.records if r.track_id == 1)
    track2_rec = next(r for r in recovered.records if r.track_id == 2)

    assert track1_rec.roster_name == "Bob Jones"
    assert track1_rec.presence_duration_seconds >= 49.0
    assert track1_rec.status == AttendanceStatus.PRESENT.value

    assert track2_rec.presence_duration_seconds >= 9.0
    # Track 2 presence (9s) is under min_presence_duration_seconds (30s)
    assert track2_rec.status == AttendanceStatus.INSUFFICIENT_PRESENCE.value

    # Verify database was updated
    db_records = test_db.get_attendance_records(session_id)
    assert len(db_records) == 2


def test_csv_export_format(tmp_path: Path) -> None:
    """Verify session attendance summary correctly exports to CSV file."""
    engine = PresenceAttendanceEngine(min_presence_percentage=50.0, min_presence_duration_seconds=100.0)
    summary = engine.evaluate_session(
        session_id="sess-export-01",
        title="Database Systems",
        session_duration_seconds=300.0,
        track_events={
            1: [float(t) for t in range(0, 250, 5)],
            2: [float(t) for t in range(0, 50, 5)],
        },
        roster_mappings={
            1: {"roster_name": "Charlie Day", "student_id": "CS-03", "notes": ""},
        },
    )

    csv_path = tmp_path / "attendance_report.csv"
    exported_path = engine.export_summary_to_csv(summary, csv_path)

    assert exported_path.exists()
    content = exported_path.read_text(encoding="utf-8")
    assert "session_id,session_title,session_duration_seconds" in content
    assert "Charlie Day" in content
    assert "Unassigned" in content
    assert "Present" in content

