"""Unit tests for SQLite database initialization, schemas, and CRUD operations."""

from pathlib import Path
import sqlite3
import pytest

from attendance_agent.models import AttendanceStatus, SessionStatus
from attendance_agent.storage.database import (
    DatabaseManager,
    get_connection,
    init_db,
)


@pytest.fixture
def test_db_path(tmp_path: Path) -> Path:
    """Provide a clean temporary database path for each test."""
    return tmp_path / "test_attendance.db"


def test_init_db_creates_tables(test_db_path: Path) -> None:
    """Verify that init_db creates all required application tables and indexes."""
    init_db(test_db_path)

    expected_tables = {
        "sessions",
        "participants",
        "attendance_records",
        "tracking_events",
        "audio_metrics",
        "fusion_scores",
        "recommendations",
    }

    with get_connection(test_db_path) as conn:
        cursor = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%';"
        )
        tables = {row[0] for row in cursor.fetchall()}

    assert expected_tables.issubset(tables)


def test_foreign_keys_enforced(test_db_path: Path) -> None:
    """Verify that foreign key constraints are enabled and enforced."""
    init_db(test_db_path)

    with get_connection(test_db_path) as conn:
        cursor = conn.execute("PRAGMA foreign_keys;")
        fk_status = cursor.fetchone()[0]
        assert fk_status == 1

        # Attempt inserting a participant with a non-existent session_id
        with pytest.raises(sqlite3.IntegrityError):
            conn.execute(
                """
                INSERT INTO participants (session_id, track_id, roster_name, created_at)
                VALUES ('non-existent-session', 1, 'Alice Smith', '2026-11-01T10:00:00Z');
                """
            )


def test_session_lifecycle(test_db_path: Path) -> None:
    """Verify creating, querying, and updating a session."""
    db = DatabaseManager(test_db_path)

    # 1. Create session
    session_id = "session-20261101-01"
    created = db.create_session(session_id=session_id, title="Computer Science 101")
    assert created["id"] == session_id
    assert created["title"] == "Computer Science 101"
    assert created["status"] == SessionStatus.ACTIVE.value

    # 2. Retrieve session
    fetched = db.get_session(session_id)
    assert fetched is not None
    assert fetched["title"] == "Computer Science 101"

    # 3. List sessions
    sessions_list = db.list_sessions()
    assert len(sessions_list) == 1
    assert sessions_list[0]["id"] == session_id

    # 4. Update status
    updated = db.update_session_status(
        session_id=session_id,
        status=SessionStatus.COMPLETED.value,
        end_time="2026-11-01T11:00:00Z",
        duration_seconds=3600.0,
    )
    assert updated is True

    completed_session = db.get_session(session_id)
    assert completed_session is not None
    assert completed_session["status"] == SessionStatus.COMPLETED.value
    assert completed_session["duration_seconds"] == 3600.0


def test_participant_roster_mapping(test_db_path: Path) -> None:
    """Verify manual mapping of anonymous track ID to roster entry."""
    db = DatabaseManager(test_db_path)
    session_id = "session-math-202"
    db.create_session(session_id, "Discrete Math")

    # Map track 1 to John Doe
    participant_id = db.map_roster_participant(
        session_id=session_id,
        track_id=1,
        roster_name="John Doe",
        student_id="STU-001",
        notes="First row, left side",
    )
    assert participant_id > 0

    participants = db.get_participants(session_id)
    assert len(participants) == 1
    assert participants[0]["roster_name"] == "John Doe"
    assert participants[0]["track_id"] == 1

    # Update mapping (upsert behavior)
    db.map_roster_participant(
        session_id=session_id,
        track_id=1,
        roster_name="Jonathan Doe",
        student_id="STU-001",
        notes="Updated note",
    )
    participants_updated = db.get_participants(session_id)
    assert len(participants_updated) == 1
    assert participants_updated[0]["roster_name"] == "Jonathan Doe"


def test_attendance_records_and_stats(test_db_path: Path) -> None:
    """Verify recording presence attendance and calculating aggregated stats."""
    db = DatabaseManager(test_db_path)
    session_id = "session-physics-301"
    db.create_session(session_id, "Quantum Physics")

    # Record attendance for track 1 (Present)
    db.record_attendance(
        session_id=session_id,
        track_id=1,
        status=AttendanceStatus.PRESENT.value,
        first_seen=0.0,
        last_seen=600.0,
        presence_duration_seconds=540.0,
        presence_percentage=90.0,
        detection_intervals=3,
    )

    # Record attendance for track 2 (Insufficient Presence)
    db.record_attendance(
        session_id=session_id,
        track_id=2,
        status=AttendanceStatus.INSUFFICIENT_PRESENCE.value,
        first_seen=10.0,
        last_seen=120.0,
        presence_duration_seconds=110.0,
        presence_percentage=18.3,
        detection_intervals=1,
    )

    records = db.get_attendance_records(session_id)
    assert len(records) == 2
    assert records[0]["track_id"] == 1
    assert records[0]["status"] == AttendanceStatus.PRESENT.value
    assert records[0]["presence_percentage"] == 90.0

    stats = db.count_overview_stats()
    assert stats["total_sessions"] == 1
    assert stats["total_attendance_records"] == 2
    assert stats["avg_presence_percentage"] == 90.0

