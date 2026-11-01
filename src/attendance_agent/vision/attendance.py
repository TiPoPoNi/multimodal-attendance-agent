"""Presence-based attendance evaluation engine.

Converts raw face detection and tracking events into sustained presence metrics
and attendance classifications (Present, Insufficient Presence, Pending Confirmation,
Not Detected) without reliance on external cloud services or biometric databases.
"""

from __future__ import annotations

import csv
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from attendance_agent.models import AttendanceStatus
from attendance_agent.storage.database import DatabaseManager
from attendance_agent.utils.logger import get_logger

logger = get_logger("attendance_agent.vision.attendance")


@dataclass
class PresenceInterval:
    """A contiguous interval during which a participant was detected."""

    start: float
    end: float
    duration: float

    def to_dict(self) -> Dict[str, Any]:
        """Convert interval to dictionary."""
        return {
            "start": round(self.start, 2),
            "end": round(self.end, 2),
            "duration": round(self.duration, 2),
        }


@dataclass
class ParticipantAttendanceSummary:
    """Evaluated attendance metrics for an individual participant track."""

    track_id: int
    status: str
    first_seen: Optional[float]
    last_seen: Optional[float]
    presence_duration_seconds: float
    presence_percentage: float
    detection_intervals: int
    intervals: List[PresenceInterval] = field(default_factory=list)
    roster_name: Optional[str] = None
    student_id: Optional[str] = None
    notes: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert summary to dictionary representation."""
        return {
            "track_id": self.track_id,
            "roster_name": self.roster_name or "Unassigned",
            "student_id": self.student_id or "",
            "status": self.status,
            "first_seen": round(self.first_seen, 2) if self.first_seen is not None else None,
            "last_seen": round(self.last_seen, 2) if self.last_seen is not None else None,
            "presence_duration_seconds": round(self.presence_duration_seconds, 2),
            "presence_percentage": round(self.presence_percentage, 1),
            "detection_intervals": self.detection_intervals,
            "notes": self.notes or "",
            "intervals": [i.to_dict() for i in self.intervals],
        }


@dataclass
class SessionAttendanceSummary:
    """Aggregated attendance summary for an entire classroom session."""

    session_id: str
    title: str
    session_duration_seconds: float
    total_participants: int
    present_count: int
    insufficient_count: int
    pending_count: int
    attendance_rate: float
    records: List[ParticipantAttendanceSummary] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert session summary to dictionary."""
        return {
            "session_id": self.session_id,
            "title": self.title,
            "session_duration_seconds": round(self.session_duration_seconds, 1),
            "total_participants": self.total_participants,
            "present_count": self.present_count,
            "insufficient_count": self.insufficient_count,
            "pending_count": self.pending_count,
            "attendance_rate": round(self.attendance_rate, 1),
            "records": [r.to_dict() for r in self.records],
        }

    def to_dataframe(self) -> pd.DataFrame:
        """Convert participant records into a tabular pandas DataFrame."""
        if not self.records:
            return pd.DataFrame(
                columns=[
                    "Track ID",
                    "Roster Name",
                    "Student ID",
                    "Status",
                    "First Seen",
                    "Last Seen",
                    "Presence (s)",
                    "Presence %",
                    "Intervals",
                    "Notes",
                ]
            )

        rows = [
            {
                "Track ID": f"#{r.track_id}",
                "Roster Name": r.roster_name or "Unassigned",
                "Student ID": r.student_id or "",
                "Status": r.status,
                "First Seen": f"{r.first_seen:.1f}s" if r.first_seen is not None else "-",
                "Last Seen": f"{r.last_seen:.1f}s" if r.last_seen is not None else "-",
                "Presence (s)": round(r.presence_duration_seconds, 1),
                "Presence %": f"{r.presence_percentage:.1f}%",
                "Intervals": r.detection_intervals,
                "Notes": r.notes or "",
            }
            for r in self.records
        ]
        return pd.DataFrame(rows)


class PresenceAttendanceEngine:
    """Evaluates presence durations and calculates classroom attendance records."""

    def __init__(
        self,
        min_presence_percentage: float = 75.0,
        min_presence_duration_seconds: float = 300.0,
        max_disappearance_seconds: float = 30.0,
        min_detection_quantum_seconds: float = 1.0,
        borderline_tolerance_percentage: float = 10.0,
    ) -> None:
        """Initialize the presence attendance evaluation engine.

        Args:
            min_presence_percentage: Required presence percentage to mark 'Present'.
            min_presence_duration_seconds: Cumulative presence in seconds required.
            max_disappearance_seconds: Maximum tolerated disappearance before splitting intervals.
            min_detection_quantum_seconds: Minimum duration credited to an isolated single detection.
            borderline_tolerance_percentage: Tolerance below min_presence_percentage classified as 'Pending Confirmation'.
        """
        self.min_presence_percentage = min_presence_percentage
        self.min_presence_duration_seconds = min_presence_duration_seconds
        self.max_disappearance_seconds = max_disappearance_seconds
        self.min_detection_quantum_seconds = min_detection_quantum_seconds
        self.borderline_tolerance_percentage = borderline_tolerance_percentage

    def calculate_presence_intervals(
        self,
        timestamps: List[float],
    ) -> Tuple[List[PresenceInterval], float]:
        """Group detection timestamps into contiguous presence intervals.

        Handles intermittent detections, brief head-turns, and long departures.

        Args:
            timestamps: Raw sequence of detection timestamps.

        Returns:
            Tuple[List[PresenceInterval], float]: (intervals_list, total_duration_seconds)
        """
        # Validate and clean timestamps
        valid_ts = sorted([t for t in timestamps if t is not None and t >= 0.0])
        if not valid_ts:
            return [], 0.0

        intervals: List[PresenceInterval] = []
        cur_start = valid_ts[0]
        cur_end = valid_ts[0]

        for ts in valid_ts[1:]:
            gap = ts - cur_end
            if gap <= self.max_disappearance_seconds:
                # Part of the same contiguous presence interval
                cur_end = ts
            else:
                # Gap exceeded: close current interval and start new one
                interval_dur = max(self.min_detection_quantum_seconds, cur_end - cur_start)
                intervals.append(
                    PresenceInterval(
                        start=cur_start,
                        end=cur_end,
                        duration=interval_dur,
                    )
                )
                cur_start = ts
                cur_end = ts

        # Close final interval
        final_dur = max(self.min_detection_quantum_seconds, cur_end - cur_start)
        intervals.append(
            PresenceInterval(
                start=cur_start,
                end=cur_end,
                duration=final_dur,
            )
        )

        total_duration = sum(i.duration for i in intervals)
        return intervals, total_duration

    def classify_status(
        self,
        presence_duration_seconds: float,
        presence_percentage: float,
        is_in_progress: bool = False,
    ) -> str:
        """Classify presence into an AttendanceStatus enum value.

        Args:
            presence_duration_seconds: Cumulative presence in seconds.
            presence_percentage: Presence percentage over session duration.
            is_in_progress: Whether the session is currently active.

        Returns:
            str: Attendance status string.
        """
        if presence_duration_seconds <= 0.0:
            return AttendanceStatus.NOT_DETECTED.value

        # Criteria for full Present status
        meets_pct = presence_percentage >= self.min_presence_percentage
        meets_dur = presence_duration_seconds >= self.min_presence_duration_seconds

        if meets_pct and meets_dur:
            return AttendanceStatus.PRESENT.value

        # In-progress sessions or borderline presence (e.g. within 10% of threshold)
        borderline_pct = max(0.0, self.min_presence_percentage - self.borderline_tolerance_percentage)
        is_borderline = presence_percentage >= borderline_pct

        if is_in_progress or is_borderline:
            return AttendanceStatus.PENDING_CONFIRMATION.value

        return AttendanceStatus.INSUFFICIENT_PRESENCE.value

    def evaluate_track(
        self,
        track_id: int,
        timestamps: List[float],
        session_duration_seconds: float,
        roster_name: Optional[str] = None,
        student_id: Optional[str] = None,
        notes: Optional[str] = None,
        is_in_progress: bool = False,
    ) -> ParticipantAttendanceSummary:
        """Evaluate attendance for a single participant track.

        Args:
            track_id: Track identifier.
            timestamps: List of detection timestamps.
            session_duration_seconds: Total duration of the session in seconds.
            roster_name: Optional teacher-assigned roster name.
            student_id: Optional student identification number.
            notes: Optional instructor notes.
            is_in_progress: Whether the session is currently underway.

        Returns:
            ParticipantAttendanceSummary: Detailed attendance evaluation.
        """
        intervals, presence_duration = self.calculate_presence_intervals(timestamps)

        first_seen = timestamps[0] if timestamps else None
        last_seen = timestamps[-1] if timestamps else None

        # Compute presence percentage
        if session_duration_seconds > 0.0:
            presence_pct = min(100.0, (presence_duration / session_duration_seconds) * 100.0)
        else:
            presence_pct = 100.0 if presence_duration > 0 else 0.0

        status = self.classify_status(
            presence_duration_seconds=presence_duration,
            presence_percentage=presence_pct,
            is_in_progress=is_in_progress,
        )

        return ParticipantAttendanceSummary(
            track_id=track_id,
            status=status,
            first_seen=first_seen,
            last_seen=last_seen,
            presence_duration_seconds=presence_duration,
            presence_percentage=presence_pct,
            detection_intervals=len(intervals),
            intervals=intervals,
            roster_name=roster_name,
            student_id=student_id,
            notes=notes,
        )

    def evaluate_session(
        self,
        session_id: str,
        title: str,
        session_duration_seconds: float,
        track_events: Dict[int, List[float]],
        roster_mappings: Optional[Dict[int, Dict[str, str]]] = None,
        is_in_progress: bool = False,
    ) -> SessionAttendanceSummary:
        """Evaluate attendance across all participants in a session.

        Args:
            session_id: Session identifier.
            title: Human-readable session title.
            session_duration_seconds: Total session duration.
            track_events: Mapping from track_id to sequence of timestamps.
            roster_mappings: Optional instructor mappings: track_id -> {roster_name, student_id, notes}.
            is_in_progress: Flag indicating whether session is still active.

        Returns:
            SessionAttendanceSummary: Aggregated session attendance report.
        """
        mappings = roster_mappings or {}
        records: List[ParticipantAttendanceSummary] = []

        for track_id, ts_list in track_events.items():
            m = mappings.get(track_id, {})
            record = self.evaluate_track(
                track_id=track_id,
                timestamps=ts_list,
                session_duration_seconds=session_duration_seconds,
                roster_name=m.get("roster_name"),
                student_id=m.get("student_id"),
                notes=m.get("notes"),
                is_in_progress=is_in_progress,
            )
            records.append(record)

        # Sort records by track_id
        records.sort(key=lambda r: r.track_id)

        present_count = sum(1 for r in records if r.status == AttendanceStatus.PRESENT.value)
        insufficient_count = sum(1 for r in records if r.status == AttendanceStatus.INSUFFICIENT_PRESENCE.value)
        pending_count = sum(1 for r in records if r.status == AttendanceStatus.PENDING_CONFIRMATION.value)

        total = len(records)
        rate = (present_count / total * 100.0) if total > 0 else 0.0

        return SessionAttendanceSummary(
            session_id=session_id,
            title=title,
            session_duration_seconds=session_duration_seconds,
            total_participants=total,
            present_count=present_count,
            insufficient_count=insufficient_count,
            pending_count=pending_count,
            attendance_rate=rate,
            records=records,
        )

    def recover_session(
        self,
        db: DatabaseManager,
        session_id: str,
    ) -> Optional[SessionAttendanceSummary]:
        """Reconstruct attendance records from persisted tracking events after interruption.

        Args:
            db: Initialized DatabaseManager.
            session_id: Identifier of the interrupted session.

        Returns:
            Optional[SessionAttendanceSummary]: Reconstructed attendance state, or None if session missing.
        """
        session_data = db.get_session(session_id)
        if not session_data:
            logger.warning("Cannot recover session %s: not found in database.", session_id)
            return None

        # Fetch stored tracking events
        events = db.get_tracking_events(session_id)
        logger.info("Recovering session %s with %d stored tracking events.", session_id, len(events))

        # Group timestamps by track ID
        track_events: Dict[int, List[float]] = {}
        for ev in events:
            tid = ev["track_id"]
            ts = ev["timestamp"]
            track_events.setdefault(tid, []).append(ts)

        # Fetch instructor mappings
        participants = db.get_participants(session_id)
        roster_mappings: Dict[int, Dict[str, str]] = {}
        for p in participants:
            roster_mappings[p["track_id"]] = {
                "roster_name": p.get("roster_name") or "",
                "student_id": p.get("student_id") or "",
                "notes": p.get("notes") or "",
            }

        duration = session_data.get("duration_seconds") or 0.0
        if duration <= 0.0 and events:
            # Estimate duration from timestamps
            duration = max(ev["timestamp"] for ev in events) - min(ev["timestamp"] for ev in events)

        summary = self.evaluate_session(
            session_id=session_id,
            title=session_data.get("title", "Recovered Session"),
            session_duration_seconds=max(1.0, duration),
            track_events=track_events,
            roster_mappings=roster_mappings,
            is_in_progress=False,
        )

        # Persist updated attendance records to database
        for r in summary.records:
            db.record_attendance(
                session_id=session_id,
                track_id=r.track_id,
                status=r.status,
                first_seen=r.first_seen,
                last_seen=r.last_seen,
                presence_duration_seconds=r.presence_duration_seconds,
                presence_percentage=r.presence_percentage,
                detection_intervals=r.detection_intervals,
            )

        logger.info("Successfully recovered and updated attendance for session %s.", session_id)
        return summary

    def export_summary_to_csv(
        self,
        summary: SessionAttendanceSummary,
        output_path: Path | str,
    ) -> Path:
        """Export session attendance summary to CSV file.

        Args:
            summary: Evaluated SessionAttendanceSummary.
            output_path: Target CSV file path.

        Returns:
            Path: Destination path of exported CSV.
        """
        out_path = Path(output_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        fieldnames = [
            "session_id",
            "session_title",
            "session_duration_seconds",
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
            for r in summary.records:
                writer.writerow(
                    {
                        "session_id": summary.session_id,
                        "session_title": summary.title,
                        "session_duration_seconds": round(summary.session_duration_seconds, 1),
                        "track_id": r.track_id,
                        "roster_name": r.roster_name or "Unassigned",
                        "student_id": r.student_id or "",
                        "status": r.status,
                        "first_seen_seconds": round(r.first_seen, 2) if r.first_seen is not None else "",
                        "last_seen_seconds": round(r.last_seen, 2) if r.last_seen is not None else "",
                        "presence_duration_seconds": round(r.presence_duration_seconds, 2),
                        "presence_percentage": round(r.presence_percentage, 1),
                        "detection_intervals": r.detection_intervals,
                        "notes": r.notes or "",
                    }
                )

        logger.info("Exported session summary CSV to %s", out_path)
        return out_path
