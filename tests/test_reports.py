"""Unit tests for Session Analytics, Reports, and Demo Data Generator."""

from pathlib import Path
import csv
import pytest

from attendance_agent.analytics.reports import (
    export_timeline_csv,
    generate_session_summary,
    generate_synthetic_demo_session,
)
from attendance_agent.storage.database import DatabaseManager, init_db


@pytest.fixture
def test_db(tmp_path: Path) -> DatabaseManager:
    """Fixture providing initialized SQLite test database."""
    db_path = tmp_path / "test_reports.db"
    init_db(db_path)
    return DatabaseManager(db_path)


def test_synthetic_demo_session_creation(test_db: DatabaseManager) -> None:
    """Verify synthetic demo session is completely populated and isolated."""
    demo_id = generate_synthetic_demo_session(test_db, title="[DEMO] Test AI Session")
    assert demo_id.startswith("demo-session-")

    # Verify session recorded
    session = test_db.get_session(demo_id)
    assert session is not None
    assert "[DEMO]" in session["title"]
    assert session["status"] == "completed"

    # Verify participants mapped
    participants = test_db.get_participants(demo_id)
    assert len(participants) == 5
    assert all("Synthetic" in p["roster_name"] for p in participants)

    # Verify attendance
    attendance = test_db.get_attendance_records(demo_id)
    assert len(attendance) == 5

    # Verify fusion scores
    fusion_scores = test_db.get_fusion_scores(demo_id)
    assert len(fusion_scores) == 20

    # Verify recommendations
    recs = test_db.get_recommendations(demo_id)
    assert len(recs) == 3


def test_generate_session_summary(test_db: DatabaseManager) -> None:
    """Verify analytical session summary compiles correct aggregates."""
    demo_id = generate_synthetic_demo_session(test_db)
    summary = generate_session_summary(test_db, demo_id)

    assert "session_id" in summary
    assert summary["session_id"] == demo_id
    assert summary["attendance"]["total_participants"] == 5
    assert summary["attendance"]["present"] == 3
    assert summary["attendance"]["insufficient"] == 1
    assert summary["attendance"]["pending"] == 1
    assert summary["attendance"]["average_presence_percentage"] > 70.0

    assert summary["activity"]["total_windows"] == 20
    assert summary["activity"]["mean_activity_score"] > 0.0
    assert summary["recommendations"]["total"] == 3


def test_export_timeline_csv(test_db: DatabaseManager, tmp_path: Path) -> None:
    """Verify timeline CSV export format and completeness."""
    demo_id = generate_synthetic_demo_session(test_db)
    csv_file = tmp_path / "timeline_export.csv"

    exported = export_timeline_csv(test_db, demo_id, csv_file)
    assert exported.exists()

    with open(exported, mode="r", encoding="utf-8") as f:
        reader = list(csv.reader(f))
        headers = reader[0]
        rows = reader[1:]

    assert "window_start" in headers
    assert "activity_score" in headers
    assert len(rows) == 20
