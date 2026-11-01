"""Session analytics, reporting, and CSV export module.

Generates unified session reports synthesizing presence-based attendance,
multimodal activity timelines, noise metrics, and pedagogical recommendations.
Also provides isolated synthetic demo sessions for hardware-free demonstration.
"""

from __future__ import annotations

import csv
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from attendance_agent.models import AttendanceStatus, SessionStatus
from attendance_agent.storage.database import DatabaseManager, get_current_iso_timestamp
from attendance_agent.utils.logger import get_logger

logger = get_logger("attendance_agent.analytics")


def generate_session_summary(db: DatabaseManager, session_id: str) -> Dict[str, Any]:
    """Compile comprehensive analytical summary of a completed or active session.

    Args:
        db: Initialized DatabaseManager instance.
        session_id: Session identifier.

    Returns:
        Dict[str, Any]: Structured summary dictionary with attendance, timeline, and suggestions.
    """
    session = db.get_session(session_id)
    if not session:
        return {"error": f"Session '{session_id}' not found."}

    attendance_records = db.get_attendance_records(session_id)
    fusion_scores = db.get_fusion_scores(session_id)
    recommendations = db.get_recommendations(session_id)
    audio_metrics = db.get_audio_metrics(session_id)

    # Attendance aggregates
    total_participants = len(attendance_records)
    present_count = sum(1 for r in attendance_records if r["status"] == AttendanceStatus.PRESENT.value)
    insufficient_count = sum(
        1 for r in attendance_records if r["status"] == AttendanceStatus.INSUFFICIENT_PRESENCE.value
    )
    pending_count = sum(
        1 for r in attendance_records if r["status"] == AttendanceStatus.PENDING_CONFIRMATION.value
    )
    not_detected_count = sum(
        1 for r in attendance_records if r["status"] == AttendanceStatus.NOT_DETECTED.value
    )
    avg_presence = (
        sum(r["presence_percentage"] for r in attendance_records) / total_participants
        if total_participants > 0
        else 0.0
    )

    # Activity timeline aggregates
    activity_scores = [f["activity_score"] for f in fusion_scores]
    mean_activity = sum(activity_scores) / len(activity_scores) if activity_scores else 0.0
    max_activity = max(activity_scores) if activity_scores else 0.0
    min_activity = min(activity_scores) if activity_scores else 0.0

    # Noise aggregates
    high_noise_windows = sum(1 for a in audio_metrics if a.get("noise_level") == "high")

    # Recommendation aggregates
    recs_by_priority: Dict[str, int] = {}
    for r in recommendations:
        p = r.get("priority", "Medium")
        recs_by_priority[p] = recs_by_priority.get(p, 0) + 1

    return {
        "session_id": session_id,
        "title": session["title"],
        "status": session["status"],
        "start_time": session["start_time"],
        "end_time": session.get("end_time"),
        "duration_seconds": session.get("duration_seconds", 0.0),
        "attendance": {
            "total_participants": total_participants,
            "present": present_count,
            "insufficient": insufficient_count,
            "pending": pending_count,
            "not_detected": not_detected_count,
            "average_presence_percentage": round(avg_presence, 1),
            "records": attendance_records,
        },
        "activity": {
            "total_windows": len(fusion_scores),
            "mean_activity_score": round(mean_activity, 1),
            "max_activity_score": round(max_activity, 1),
            "min_activity_score": round(min_activity, 1),
            "timeline": fusion_scores,
        },
        "acoustic": {
            "total_audio_windows": len(audio_metrics),
            "high_noise_windows": high_noise_windows,
        },
        "recommendations": {
            "total": len(recommendations),
            "by_priority": recs_by_priority,
            "items": recommendations,
        },
    }


def export_timeline_csv(db: DatabaseManager, session_id: str, output_path: Path | str) -> Path:
    """Export window-by-window multimodal activity timeline scores to CSV.

    Args:
        db: DatabaseManager instance.
        session_id: Session identifier.
        output_path: Target CSV file path.

    Returns:
        Path: Resolved Path to written CSV file.
    """
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fusion_scores = db.get_fusion_scores(session_id)

    fieldnames = [
        "window_start",
        "window_end",
        "visual_presence",
        "audio_activity",
        "movement_level",
        "activity_score",
        "confidence",
        "label",
    ]

    with open(path, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for score in fusion_scores:
            writer.writerow(score)

    logger.info("Exported %d timeline windows to '%s'", len(fusion_scores), path)
    return path


def generate_synthetic_demo_session(
    db: DatabaseManager,
    title: str = "[DEMO] Multimodal AI & Classroom Dynamics",
) -> str:
    """Generate a fully realized, non-judgmental synthetic demo session in SQLite.

    Guaranteed never to overwrite real attendance records. All participant records
    and session entries are explicitly marked as synthetic demonstration data.

    Args:
        db: DatabaseManager instance.
        title: Title of synthetic demonstration session.

    Returns:
        str: Created demo session identifier.
    """
    demo_session_id = f"demo-session-{datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S')}"
    duration_seconds = 600.0  # 10 minutes

    # 1. Create session
    db.create_session(session_id=demo_session_id, title=title)
    db.update_session_status(
        session_id=demo_session_id,
        status=SessionStatus.COMPLETED.value,
        end_time=get_current_iso_timestamp(),
        duration_seconds=duration_seconds,
    )

    # 2. Map synthetic roster participants
    synthetic_students = [
        (1, "Synthetic Student A (Alice)", "DEMO-001", "Row 1 Left (Synthetic Demo)"),
        (2, "Synthetic Student B (Bob)", "DEMO-002", "Row 1 Center (Synthetic Demo)"),
        (3, "Synthetic Student C (Charlie)", "DEMO-003", "Row 2 Right (Synthetic Demo)"),
        (4, "Synthetic Student D (Diana)", "DEMO-004", "Row 2 Left (Synthetic Demo)"),
        (5, "Synthetic Student E (Evan)", "DEMO-005", "Row 3 Center (Synthetic Demo)"),
    ]

    for track_id, name, stu_id, notes in synthetic_students:
        db.map_roster_participant(
            session_id=demo_session_id,
            track_id=track_id,
            roster_name=name,
            student_id=stu_id,
            notes=notes,
        )

    # 3. Create attendance records
    attendance_data = [
        # Track 1: Present (95% presence)
        (1, AttendanceStatus.PRESENT.value, 0.0, 595.0, 570.0, 95.0, 1),
        # Track 2: Present (88% presence)
        (2, AttendanceStatus.PRESENT.value, 10.0, 590.0, 528.0, 88.0, 2),
        # Track 3: Present (80% presence)
        (3, AttendanceStatus.PRESENT.value, 15.0, 580.0, 480.0, 80.0, 2),
        # Track 4: Insufficient Presence (35% presence)
        (4, AttendanceStatus.INSUFFICIENT_PRESENCE.value, 30.0, 250.0, 210.0, 35.0, 1),
        # Track 5: Pending Confirmation (68% presence)
        (5, AttendanceStatus.PENDING_CONFIRMATION.value, 45.0, 480.0, 408.0, 68.0, 3),
    ]

    for tid, status, first, last, pres_sec, pres_pct, intervals in attendance_data:
        db.record_attendance(
            session_id=demo_session_id,
            track_id=tid,
            status=status,
            first_seen=first,
            last_seen=last,
            presence_duration_seconds=pres_sec,
            presence_percentage=pres_pct,
            detection_intervals=intervals,
        )

    # 4. Create synthetic timeline fusion scores (20 windows of 30s)
    fusion_scores_batch: List[Dict[str, Any]] = []
    audio_metrics_batch: List[Dict[str, Any]] = []

    # Dynamic scenario progression:
    # 0-120s: Baseline lecture (activity ~55)
    # 120-240s: Quiet independent reading (activity ~28)
    # 240-420s: Collaborative group activity (activity ~75)
    # 420-540s: Lecture Q&A (activity ~60)
    # 540-600s: Session wrap-up (activity ~45)
    for i in range(20):
        w_start = i * 30.0
        w_end = (i + 1) * 30.0

        if w_end <= 120.0:
            vis = 0.85
            aud = 0.35
            mov = 0.20
            score = 52.0
            noise = "low"
            label = "Moderate observed activity"
        elif w_end <= 240.0:
            vis = 0.80
            aud = 0.02
            mov = 0.05
            score = 25.0
            noise = "low"
            label = "Low observed activity"
        elif w_end <= 420.0:
            vis = 0.90
            aud = 0.75
            mov = 0.55
            score = 77.0
            noise = "high" if i in (10, 11) else "moderate"
            label = "High observed activity"
        elif w_end <= 540.0:
            vis = 0.85
            aud = 0.50
            mov = 0.30
            score = 60.0
            noise = "moderate"
            label = "Moderate observed activity"
        else:
            vis = 0.70
            aud = 0.25
            mov = 0.15
            score = 42.0
            noise = "low"
            label = "Moderate observed activity"

        fusion_scores_batch.append(
            {
                "window_start": w_start,
                "window_end": w_end,
                "visual_presence": vis,
                "audio_activity": aud,
                "movement_level": mov,
                "activity_score": score,
                "confidence": 0.95,
                "label": label,
            }
        )
        audio_metrics_batch.append(
            {
                "timestamp": w_end,
                "window_start": w_start,
                "window_end": w_end,
                "rms_energy": 0.02 if noise == "low" else (0.09 if noise == "high" else 0.04),
                "zero_crossing_rate": 0.06,
                "activity_detected": aud > 0.1,
                "noise_level": noise,
            }
        )

    db.record_fusion_scores_batch(demo_session_id, fusion_scores_batch)
    db.record_audio_metrics_batch(demo_session_id, audio_metrics_batch)

    # 5. Create synthetic recommendations
    synthetic_recs = [
        {
            "id": f"rec-{demo_session_id}-sustained_low-180000",
            "session_id": demo_session_id,
            "timestamp": 180.0,
            "condition": "Sustained Low Classroom Activity",
            "explanation": "Observed activity remained below 30.0 for 3 consecutive intervals (90.0s).",
            "suggestion": "Consider a quick think-pair-share or informal polling question to verify comprehension.",
            "confidence": "High",
            "rule_source": "sustained_low_activity",
            "priority": "Medium",
            "supporting_metrics": {"consecutive_windows": 3, "avg_activity": 25.0},
        },
        {
            "id": f"rec-{demo_session_id}-high_interaction-330000",
            "session_id": demo_session_id,
            "timestamp": 330.0,
            "condition": "High Multimodal Interaction Interval",
            "explanation": "Concurrent elevation of acoustic activity (75.0%) and motion (55.0%) observed.",
            "suggestion": "Strong collaborative engagement detected. Ideal interval to let peer discussion develop before summarizing.",
            "confidence": "High",
            "rule_source": "high_interaction_activity",
            "priority": "Info",
            "supporting_metrics": {"audio_activity": 75.0, "movement_level": 55.0},
        },
        {
            "id": f"rec-{demo_session_id}-noise_spike-360000",
            "session_id": demo_session_id,
            "timestamp": 360.0,
            "condition": "Elevated Classroom Ambient Noise",
            "explanation": "Classroom acoustic activity reached high ambient volume thresholds.",
            "suggestion": "Check whether active group discussion is wrapping up or if room acoustics require transitioning back to lecture mode.",
            "confidence": "High",
            "rule_source": "increasing_noise",
            "priority": "Warning",
            "supporting_metrics": {"audio_activity": 75.0, "noise_level": "high"},
        },
    ]
    db.record_recommendations_batch(demo_session_id, synthetic_recs)

    logger.info("Generated synthetic demonstration session '%s' successfully.", demo_session_id)
    return demo_session_id
