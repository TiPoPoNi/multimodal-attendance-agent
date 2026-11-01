"""Local SQLite database management module.

Encapsulates database initialization, schema creation, connection lifecycle,
and parameterized CRUD operations for sessions, participants, tracking events,
attendance records, audio metrics, and multimodal fusion scores.
"""

from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional, Tuple

from attendance_agent.models import AttendanceStatus, SessionStatus
from attendance_agent.utils.logger import get_logger

logger = get_logger("attendance_agent.storage")


def get_current_iso_timestamp() -> str:
    """Generate current UTC timestamp in ISO 8601 format."""
    return datetime.now(timezone.utc).isoformat()


@contextmanager
def get_connection(db_path: Path | str) -> Generator[sqlite3.Connection, None, None]:
    """Provide a transactional SQLite database connection context.

    Ensures foreign keys are enabled and automatically commits or rolls back
    upon block completion or exception.

    Args:
        db_path: File system path to the SQLite database file.

    Yields:
        sqlite3.Connection: Active database connection with row factory enabled.
    """
    path_obj = Path(db_path)
    path_obj.parent.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(str(path_obj))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON;")
    try:
        yield conn
        conn.commit()
    except Exception as exc:
        conn.rollback()
        logger.error("Database transaction rolled back due to error: %s", exc)
        raise
    finally:
        conn.close()


def init_db(db_path: Path | str) -> None:
    """Initialize database schema if tables do not exist.

    Args:
        db_path: Target SQLite database path.
    """
    logger.info("Initializing SQLite database at %s", db_path)

    schema_statements = [
        """
        CREATE TABLE IF NOT EXISTS sessions (
            id TEXT PRIMARY KEY,
            title TEXT NOT NULL,
            start_time TEXT NOT NULL,
            end_time TEXT,
            duration_seconds REAL DEFAULT 0.0,
            status TEXT NOT NULL DEFAULT 'created',
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        );
        """,
        """
        CREATE TABLE IF NOT EXISTS participants (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT NOT NULL,
            track_id INTEGER NOT NULL,
            roster_name TEXT,
            student_id TEXT,
            notes TEXT,
            created_at TEXT NOT NULL,
            FOREIGN KEY (session_id) REFERENCES sessions(id) ON DELETE CASCADE,
            UNIQUE(session_id, track_id)
        );
        """,
        """
        CREATE TABLE IF NOT EXISTS attendance_records (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT NOT NULL,
            participant_id INTEGER,
            track_id INTEGER NOT NULL,
            status TEXT NOT NULL,
            first_seen REAL,
            last_seen REAL,
            presence_duration_seconds REAL DEFAULT 0.0,
            presence_percentage REAL DEFAULT 0.0,
            detection_intervals INTEGER DEFAULT 0,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            FOREIGN KEY (session_id) REFERENCES sessions(id) ON DELETE CASCADE,
            FOREIGN KEY (participant_id) REFERENCES participants(id) ON DELETE SET NULL,
            UNIQUE(session_id, track_id)
        );
        """,
        """
        CREATE TABLE IF NOT EXISTS tracking_events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT NOT NULL,
            track_id INTEGER NOT NULL,
            timestamp REAL NOT NULL,
            bbox_x REAL NOT NULL,
            bbox_y REAL NOT NULL,
            bbox_w REAL NOT NULL,
            bbox_h REAL NOT NULL,
            confidence REAL NOT NULL,
            status TEXT NOT NULL DEFAULT 'active',
            FOREIGN KEY (session_id) REFERENCES sessions(id) ON DELETE CASCADE
        );
        """,
        """
        CREATE TABLE IF NOT EXISTS audio_metrics (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT NOT NULL,
            timestamp REAL NOT NULL,
            window_start REAL NOT NULL,
            window_end REAL NOT NULL,
            rms_energy REAL NOT NULL,
            zero_crossing_rate REAL NOT NULL,
            activity_detected INTEGER NOT NULL,
            noise_level TEXT NOT NULL,
            FOREIGN KEY (session_id) REFERENCES sessions(id) ON DELETE CASCADE
        );
        """,
        """
        CREATE TABLE IF NOT EXISTS fusion_scores (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT NOT NULL,
            window_start REAL NOT NULL,
            window_end REAL NOT NULL,
            visual_presence REAL NOT NULL,
            audio_activity REAL NOT NULL,
            movement_level REAL NOT NULL,
            activity_score REAL NOT NULL,
            confidence REAL NOT NULL,
            label TEXT NOT NULL,
            FOREIGN KEY (session_id) REFERENCES sessions(id) ON DELETE CASCADE
        );
        """,
        """
        CREATE TABLE IF NOT EXISTS recommendations (
            id TEXT PRIMARY KEY,
            session_id TEXT NOT NULL,
            timestamp REAL NOT NULL,
            condition TEXT NOT NULL,
            explanation TEXT NOT NULL,
            suggestion TEXT NOT NULL,
            confidence TEXT NOT NULL,
            rule_source TEXT NOT NULL,
            priority TEXT DEFAULT 'Medium',
            supporting_metrics TEXT DEFAULT '{}',
            FOREIGN KEY (session_id) REFERENCES sessions(id) ON DELETE CASCADE
        );
        """,
        # Indexes for efficient session queries
        "CREATE INDEX IF NOT EXISTS idx_tracking_session_track ON tracking_events(session_id, track_id);",
        "CREATE INDEX IF NOT EXISTS idx_attendance_session ON attendance_records(session_id);",
        "CREATE INDEX IF NOT EXISTS idx_audio_session ON audio_metrics(session_id);",
        "CREATE INDEX IF NOT EXISTS idx_fusion_session ON fusion_scores(session_id);",
        "CREATE INDEX IF NOT EXISTS idx_recommendations_session ON recommendations(session_id);",
    ]

    with get_connection(db_path) as conn:
        for stmt in schema_statements:
            conn.execute(stmt)

        # Migrations for existing tables if columns are missing
        for col, col_type, default_val in [
            ("priority", "TEXT", "'Medium'"),
            ("supporting_metrics", "TEXT", "'{}'"),
        ]:
            try:
                conn.execute(f"ALTER TABLE recommendations ADD COLUMN {col} {col_type} DEFAULT {default_val};")
            except sqlite3.OperationalError:
                pass

    logger.info("Database schema initialized successfully.")


class DatabaseManager:
    """High-level database access layer executing parameterized operations."""

    def __init__(self, db_path: Path | str) -> None:
        """Initialize database manager.

        Args:
            db_path: Path to SQLite database file.
        """
        self.db_path = Path(db_path)
        init_db(self.db_path)

    def create_session(self, session_id: str, title: str) -> Dict[str, Any]:
        """Create a new session record.

        Args:
            session_id: Unique session identifier.
            title: Human-readable session title.

        Returns:
            Dict[str, Any]: Created session attributes.
        """
        now = get_current_iso_timestamp()
        query = """
            INSERT INTO sessions (id, title, start_time, duration_seconds, status, created_at, updated_at)
            VALUES (?, ?, ?, 0.0, ?, ?, ?);
        """
        params = (session_id, title, now, SessionStatus.ACTIVE.value, now, now)

        with get_connection(self.db_path) as conn:
            conn.execute(query, params)

        logger.info("Created session %s ('%s')", session_id, title)
        return {
            "id": session_id,
            "title": title,
            "start_time": now,
            "duration_seconds": 0.0,
            "status": SessionStatus.ACTIVE.value,
            "created_at": now,
            "updated_at": now,
        }

    def get_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve a session by its unique ID.

        Args:
            session_id: Session identifier.

        Returns:
            Optional[Dict[str, Any]]: Session data dictionary or None if not found.
        """
        query = "SELECT * FROM sessions WHERE id = ?;"
        with get_connection(self.db_path) as conn:
            cursor = conn.execute(query, (session_id,))
            row = cursor.fetchone()
            if row:
                return dict(row)
        return None

    def list_sessions(self, limit: int = 50) -> List[Dict[str, Any]]:
        """List recent sessions sorted by start time descending.

        Args:
            limit: Maximum number of sessions to return.

        Returns:
            List[Dict[str, Any]]: List of session records.
        """
        query = "SELECT * FROM sessions ORDER BY created_at DESC LIMIT ?;"
        with get_connection(self.db_path) as conn:
            cursor = conn.execute(query, (limit,))
            return [dict(row) for row in cursor.fetchall()]

    def update_session_status(
        self,
        session_id: str,
        status: str,
        end_time: Optional[str] = None,
        duration_seconds: Optional[float] = None,
    ) -> bool:
        """Update the lifecycle status and completion metrics of a session.

        Args:
            session_id: Session identifier.
            status: New session status string.
            end_time: Optional session completion timestamp.
            duration_seconds: Optional session total duration in seconds.

        Returns:
            bool: True if a record was updated, False otherwise.
        """
        now = get_current_iso_timestamp()
        fields = ["status = ?", "updated_at = ?"]
        params: List[Any] = [status, now]

        if end_time is not None:
            fields.append("end_time = ?")
            params.append(end_time)

        if duration_seconds is not None:
            fields.append("duration_seconds = ?")
            params.append(duration_seconds)

        params.append(session_id)
        query = f"UPDATE sessions SET {', '.join(fields)} WHERE id = ?;"

        with get_connection(self.db_path) as conn:
            cursor = conn.execute(query, tuple(params))
            return cursor.rowcount > 0

    def map_roster_participant(
        self,
        session_id: str,
        track_id: int,
        roster_name: str,
        student_id: Optional[str] = None,
        notes: Optional[str] = None,
    ) -> int:
        """Manually map an anonymous track ID to a student roster entry.

        Args:
            session_id: Session identifier.
            track_id: Anonymous vision track ID.
            roster_name: Student name from roster.
            student_id: Optional student identification code.
            notes: Optional instructor notes.

        Returns:
            int: Participant record ID.
        """
        now = get_current_iso_timestamp()
        query = """
            INSERT INTO participants (session_id, track_id, roster_name, student_id, notes, created_at)
            VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(session_id, track_id) DO UPDATE SET
                roster_name = excluded.roster_name,
                student_id = excluded.student_id,
                notes = excluded.notes;
        """
        params = (session_id, track_id, roster_name, student_id, notes, now)

        with get_connection(self.db_path) as conn:
            cursor = conn.execute(query, params)
            participant_id = cursor.lastrowid

            # Associate with existing attendance record if present
            conn.execute(
                """
                UPDATE attendance_records
                SET participant_id = ?, updated_at = ?
                WHERE session_id = ? AND track_id = ?;
                """,
                (participant_id, now, session_id, track_id),
            )

        logger.info(
            "Mapped track %d in session %s to roster name '%s'",
            track_id,
            session_id,
            roster_name,
        )
        return participant_id or 0

    def get_participants(self, session_id: str) -> List[Dict[str, Any]]:
        """Get all participant mappings for a session.

        Args:
            session_id: Session identifier.

        Returns:
            List[Dict[str, Any]]: Participant mappings.
        """
        query = "SELECT * FROM participants WHERE session_id = ? ORDER BY track_id ASC;"
        with get_connection(self.db_path) as conn:
            cursor = conn.execute(query, (session_id,))
            return [dict(row) for row in cursor.fetchall()]

    def record_attendance(
        self,
        session_id: str,
        track_id: int,
        status: str,
        first_seen: Optional[float] = None,
        last_seen: Optional[float] = None,
        presence_duration_seconds: float = 0.0,
        presence_percentage: float = 0.0,
        detection_intervals: int = 0,
        participant_id: Optional[int] = None,
    ) -> None:
        """Insert or update an attendance record for a participant track.

        Args:
            session_id: Session identifier.
            track_id: Track ID.
            status: Attendance status classification.
            first_seen: First detection timestamp in seconds.
            last_seen: Last detection timestamp in seconds.
            presence_duration_seconds: Total detected presence duration.
            presence_percentage: Presence percentage over session.
            detection_intervals: Number of distinct presence intervals.
            participant_id: Optional participant ID if already mapped.
        """
        now = get_current_iso_timestamp()
        query = """
            INSERT INTO attendance_records (
                session_id, participant_id, track_id, status, first_seen,
                last_seen, presence_duration_seconds, presence_percentage,
                detection_intervals, created_at, updated_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(session_id, track_id) DO UPDATE SET
                participant_id = COALESCE(excluded.participant_id, attendance_records.participant_id),
                status = excluded.status,
                first_seen = excluded.first_seen,
                last_seen = excluded.last_seen,
                presence_duration_seconds = excluded.presence_duration_seconds,
                presence_percentage = excluded.presence_percentage,
                detection_intervals = excluded.detection_intervals,
                updated_at = excluded.updated_at;
        """
        params = (
            session_id,
            participant_id,
            track_id,
            status,
            first_seen,
            last_seen,
            presence_duration_seconds,
            presence_percentage,
            detection_intervals,
            now,
            now,
        )

        with get_connection(self.db_path) as conn:
            conn.execute(query, params)

    def get_attendance_records(self, session_id: str) -> List[Dict[str, Any]]:
        """Retrieve attendance records with participant mapping information.

        Args:
            session_id: Session identifier.

        Returns:
            List[Dict[str, Any]]: Attendance records with roster information if mapped.
        """
        query = """
            SELECT 
                a.id,
                a.session_id,
                a.track_id,
                a.status,
                a.first_seen,
                a.last_seen,
                a.presence_duration_seconds,
                a.presence_percentage,
                a.detection_intervals,
                p.roster_name,
                p.student_id,
                p.notes
            FROM attendance_records a
            LEFT JOIN participants p 
                ON a.session_id = p.session_id AND a.track_id = p.track_id
            WHERE a.session_id = ?
            ORDER BY a.track_id ASC;
        """
        with get_connection(self.db_path) as conn:
            cursor = conn.execute(query, (session_id,))
            return [dict(row) for row in cursor.fetchall()]

    def count_overview_stats(self) -> Dict[str, Any]:
        """Compute aggregated statistics for system dashboard overview.

        Returns:
            Dict[str, Any]: Summary metrics (total sessions, total records, average presence).
        """
        with get_connection(self.db_path) as conn:
            total_sessions = conn.execute("SELECT COUNT(*) FROM sessions;").fetchone()[0]
            total_records = conn.execute("SELECT COUNT(*) FROM attendance_records;").fetchone()[0]
            avg_presence_row = conn.execute(
                "SELECT AVG(presence_percentage) FROM attendance_records WHERE status = ?;",
                (AttendanceStatus.PRESENT.value,),
            ).fetchone()
            avg_presence = avg_presence_row[0] if avg_presence_row and avg_presence_row[0] is not None else 0.0

        return {
            "total_sessions": total_sessions,
            "total_attendance_records": total_records,
            "avg_presence_percentage": round(float(avg_presence), 1),
        }

    def record_tracking_events_batch(self, events: List[Dict[str, Any]]) -> int:
        """Batch insert raw tracking events for auditing and attendance recovery.

        Args:
            events: List of tracking event dictionaries.

        Returns:
            int: Number of inserted events.
        """
        if not events:
            return 0

        query = """
            INSERT INTO tracking_events (
                session_id, track_id, timestamp, bbox_x, bbox_y, bbox_w, bbox_h, confidence, status
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?);
        """
        rows = [
            (
                e["session_id"],
                e["track_id"],
                e["timestamp"],
                e["bbox_x"],
                e["bbox_y"],
                e["bbox_w"],
                e["bbox_h"],
                e.get("confidence", 0.0),
                e.get("status", "active"),
            )
            for e in events
        ]

        with get_connection(self.db_path) as conn:
            cursor = conn.executemany(query, rows)
            return cursor.rowcount

    def get_tracking_events(
        self, session_id: str, track_id: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """Retrieve tracking events for a session, optionally filtered by track ID.

        Args:
            session_id: Session identifier.
            track_id: Optional track ID filter.

        Returns:
            List[Dict[str, Any]]: List of recorded tracking events.
        """
        if track_id is not None:
            query = """
                SELECT * FROM tracking_events
                WHERE session_id = ? AND track_id = ?
                ORDER BY timestamp ASC;
            """
            params: Tuple[Any, ...] = (session_id, track_id)
        else:
            query = """
                SELECT * FROM tracking_events
                WHERE session_id = ?
                ORDER BY timestamp ASC;
            """
            params = (session_id,)

        with get_connection(self.db_path) as conn:
            cursor = conn.execute(query, params)
            return [dict(row) for row in cursor.fetchall()]

    def get_unclosed_sessions(self) -> List[Dict[str, Any]]:
        """Retrieve sessions that were active or paused but not marked completed or aborted.

        Returns:
            List[Dict[str, Any]]: Unclosed sessions eligible for recovery.
        """
        query = """
            SELECT * FROM sessions
            WHERE status IN ('active', 'paused')
            ORDER BY created_at DESC;
        """
        with get_connection(self.db_path) as conn:
            cursor = conn.execute(query)
            return [dict(row) for row in cursor.fetchall()]

    def export_attendance_to_csv(self, session_id: str, output_path: Path | str) -> Path:
        """Export session attendance records to a CSV file.

        Args:
            session_id: Session identifier.
            output_path: Destination path for the CSV report.

        Returns:
            Path: Path to the generated CSV file.
        """
        import csv

        out_path = Path(output_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        records = self.get_attendance_records(session_id)
        session = self.get_session(session_id) or {"title": "Unknown", "id": session_id}

        fieldnames = [
            "session_id",
            "session_title",
            "track_id",
            "roster_name",
            "student_id",
            "status",
            "first_seen_seconds",
            "last_seen_seconds",
            "presence_duration_seconds",
            "presence_percentage",
            "detection_intervals",
            "notes",
        ]

        with open(out_path, mode="w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for r in records:
                writer.writerow(
                    {
                        "session_id": session_id,
                        "session_title": session.get("title", ""),
                        "track_id": r.get("track_id"),
                        "roster_name": r.get("roster_name") or "Unassigned",
                        "student_id": r.get("student_id") or "",
                        "status": r.get("status"),
                        "first_seen_seconds": r.get("first_seen"),
                        "last_seen_seconds": r.get("last_seen"),
                        "presence_duration_seconds": r.get("presence_duration_seconds"),
                        "presence_percentage": r.get("presence_percentage"),
                        "detection_intervals": r.get("detection_intervals"),
                        "notes": r.get("notes") or "",
                    }
                )

        logger.info("Exported attendance report for session %s to %s", session_id, out_path)
        return out_path

    def record_audio_metrics_batch(self, session_id: str, metrics: List[Dict[str, Any]]) -> int:
        """Batch record audio window metrics for a session.

        Args:
            session_id: Session identifier.
            metrics: List of audio metric dictionaries.

        Returns:
            int: Number of rows inserted.
        """
        if not metrics:
            return 0

        query = """
            INSERT INTO audio_metrics (
                session_id, timestamp, window_start, window_end,
                rms_energy, zero_crossing_rate, activity_detected, noise_level
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?);
        """
        rows = [
            (
                session_id,
                m.get("timestamp", m.get("window_end", 0.0)),
                m["window_start"],
                m["window_end"],
                m["rms_energy"],
                m["zero_crossing_rate"],
                1 if m.get("activity_detected", False) else 0,
                m.get("noise_level", "low"),
            )
            for m in metrics
        ]

        with get_connection(self.db_path) as conn:
            cursor = conn.executemany(query, rows)
            return cursor.rowcount

    def get_audio_metrics(self, session_id: str) -> List[Dict[str, Any]]:
        """Retrieve all recorded audio metrics for a session in chronological order.

        Args:
            session_id: Session identifier.

        Returns:
            List[Dict[str, Any]]: List of recorded audio window records.
        """
        query = """
            SELECT * FROM audio_metrics
            WHERE session_id = ?
            ORDER BY window_start ASC;
        """
        with get_connection(self.db_path) as conn:
            cursor = conn.execute(query, (session_id,))
            return [dict(row) for row in cursor.fetchall()]

    def record_recommendations_batch(self, session_id: str, recs: List[Dict[str, Any]]) -> int:
        """Batch insert generated pedagogical recommendations into database.

        Args:
            session_id: Session identifier.
            recs: List of recommendation dictionaries.

        Returns:
            int: Number of rows inserted.
        """
        if not recs:
            return 0

        query = """
            INSERT INTO recommendations (
                id, session_id, timestamp, condition, explanation, suggestion, confidence, rule_source, priority, supporting_metrics
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(id) DO NOTHING;
        """
        rows = [
            (
                r["id"],
                session_id,
                r.get("timestamp", 0.0),
                r["condition"],
                r["explanation"],
                r["suggestion"],
                r.get("confidence", "Medium"),
                r.get("rule_source", "unknown"),
                r.get("priority", "Medium"),
                json.dumps(r.get("supporting_metrics", {})) if isinstance(r.get("supporting_metrics"), dict) else str(r.get("supporting_metrics", "{}")),
            )
            for r in recs
        ]

        with get_connection(self.db_path) as conn:
            cursor = conn.executemany(query, rows)
            return cursor.rowcount

    def get_recommendations(self, session_id: str) -> List[Dict[str, Any]]:
        """Retrieve all recorded teaching recommendations for a session.

        Args:
            session_id: Session identifier.

        Returns:
            List[Dict[str, Any]]: Recommendations sorted chronologically.
        """
        query = """
            SELECT * FROM recommendations
            WHERE session_id = ?
            ORDER BY timestamp ASC;
        """
        with get_connection(self.db_path) as conn:
            cursor = conn.execute(query, (session_id,))
            results = []
            for row in cursor.fetchall():
                d = dict(row)
                if "supporting_metrics" in d and isinstance(d["supporting_metrics"], str):
                    try:
                        d["supporting_metrics"] = json.loads(d["supporting_metrics"])
                    except Exception:
                        pass
                results.append(d)
            return results

    def record_fusion_scores_batch(self, session_id: str, scores: List[Dict[str, Any]]) -> int:
        """Batch record multimodal fusion scores for a session.

        Args:
            session_id: Session identifier.
            scores: List of fusion score dictionaries.

        Returns:
            int: Number of rows inserted.
        """
        if not scores:
            return 0

        query = """
            INSERT INTO fusion_scores (
                session_id, window_start, window_end,
                visual_presence, audio_activity, movement_level,
                activity_score, confidence, label
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?);
        """
        rows = [
            (
                session_id,
                s["window_start"],
                s["window_end"],
                s.get("visual_presence", 0.0),
                s.get("audio_activity", 0.0),
                s.get("movement_level", 0.0),
                s.get("activity_score", 0.0),
                s.get("confidence", 0.0),
                s.get("label", "Insufficient data"),
            )
            for s in scores
        ]

        with get_connection(self.db_path) as conn:
            cursor = conn.executemany(query, rows)
            return cursor.rowcount

    def get_fusion_scores(self, session_id: str) -> List[Dict[str, Any]]:
        """Retrieve recorded multimodal fusion scores for a session.

        Args:
            session_id: Session identifier.

        Returns:
            List[Dict[str, Any]]: List of fusion records in chronological order.
        """
        query = """
            SELECT * FROM fusion_scores
            WHERE session_id = ?
            ORDER BY window_start ASC;
        """
        with get_connection(self.db_path) as conn:
            cursor = conn.execute(query, (session_id,))
            return [dict(row) for row in cursor.fetchall()]



