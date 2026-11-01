"""Multimodal Attendance & Engagement Agent - Streamlit Application.

Local-first, privacy-preserving classroom presence tracking, multimodal activity
scoring, and transparent rule-based teaching recommendations.
Operates 100% offline without external LLM APIs, biometric profiling, or cloud retention.
"""

from __future__ import annotations

import io
import json
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# Ensure src directory is available on sys.path
SRC_DIR = Path(__file__).resolve().parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from attendance_agent.agent.recommendations import (
    PRIORITY_RANK,
    TeachingRecommendationEngine,
)
from attendance_agent.analytics.reports import (
    export_timeline_csv,
    generate_session_summary,
    generate_synthetic_demo_session,
)
from attendance_agent.audio.analyzer import AudioAnalysisReport, AudioAnalyzer
from attendance_agent.audio.capture import AudioSourceLoader
from attendance_agent.config import get_config
from attendance_agent.fusion.scoring import (
    MultimodalFusionEngine,
    SessionActivityReport,
    WindowActivityScore,
)
from attendance_agent.fusion.synchronizer import MultimodalSynchronizer
from attendance_agent.models import AttendanceStatus, SessionStatus
from attendance_agent.storage.database import DatabaseManager, init_db
from attendance_agent.utils.logger import get_logger, setup_logger
from attendance_agent.vision.attendance import (
    PresenceAttendanceEngine,
    export_attendance_csv,
)
from attendance_agent.vision.capture import VideoSource, annotate_frame
from attendance_agent.vision.detector import (
    FaceBoundingBox,
    FaceDetectorWrapper,
    RawDetection,
)
from attendance_agent.vision.tracker import AnonymousFaceTracker, TrackingFrameResult

# Set page configuration
st.set_page_config(
    page_title="Multimodal Attendance & Engagement Agent",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Initialize application settings and database
config = get_config()
config.ensure_directories()
init_db(config.storage.database_path)
db = DatabaseManager(config.storage.database_path)
logger = setup_logger(
    name="attendance_agent",
    log_level=config.log_level,
    log_file=config.storage.sessions_dir / "agent.log",
)


# ==============================================================================
# Helper UI Components
# ==============================================================================

def render_header() -> None:
    """Render top application brand banner and security status."""
    st.markdown(
        """
        <div style="padding: 1.1rem 1.4rem; background-color: #f8fafc; border-left: 5px solid #2563eb; border-radius: 6px; margin-bottom: 1.2rem; border-top: 1px solid #e2e8f0; border-right: 1px solid #e2e8f0; border-bottom: 1px solid #e2e8f0;">
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <div>
                    <h2 style="margin: 0; font-size: 1.65rem; color: #0f172a; font-weight: 700;">
                        Multimodal Attendance & Engagement Agent
                    </h2>
                    <p style="margin: 0.3rem 0 0 0; font-size: 0.92rem; color: #475569;">
                        Local-first, privacy-preserving classroom presence tracking and transparent activity analytics.
                    </p>
                </div>
                <div style="text-align: right;">
                    <span style="background-color: #ecfdf5; color: #059669; font-weight: 600; font-size: 0.8rem; padding: 0.35rem 0.65rem; border-radius: 9999px; border: 1px solid #a7f3d0;">
                        🔒 100% Local & Offline
                    </span>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_sidebar() -> None:
    """Render sidebar system info and quick links."""
    with st.sidebar:
        st.subheader("🎓 System Roadmap")
        st.markdown(
            """
            - ✅ **Phase 1:** Project Foundation & Setup
            - ✅ **Phase 2:** Anonymous Face Tracking
            - ✅ **Phase 3:** Presence-Based Attendance
            - ✅ **Phase 4:** Classroom Sound Analysis
            - ✅ **Phase 5:** Multimodal Signal Fusion
            - ✅ **Phase 6:** Rule-Based Teaching Agent
            - ✅ **Phase 7:** Integrated Dashboard
            - ✅ **Phase 8:** Testing & Optimization
            """
        )

        st.divider()
        st.subheader("⚡ Quick Actions")
        if st.button("🚀 Load Synthetic Demo Session", use_container_width=True, type="primary"):
            with st.spinner("Generating synthetic classroom demonstration data..."):
                demo_id = generate_synthetic_demo_session(db)
                st.session_state["selected_session_id"] = demo_id
                st.session_state["current_session_id"] = demo_id
                st.success(f"Loaded demo session `{demo_id}`!")
                st.rerun()

        st.divider()
        st.subheader("🔒 Privacy Guarantee")
        st.info(
            "**Zero Cloud Transmission**\n\n"
            "• Frames & audio processed locally.\n"
            "• No biometric recognition against identity databases.\n"
            "• No raw video recordings stored.\n"
            "• Deterministic rule-based advice."
        )

        st.caption(f"DB Path: `{config.storage.database_path.name}`")
        st.caption(f"Environment: `{config.env}` | Python: `{sys.version.split()[0]}`")


def get_all_sessions_options() -> List[Dict[str, Any]]:
    """Retrieve all available sessions for dropdown selection."""
    return db.list_sessions(limit=100)


def session_selector(label: str = "Select Session to Inspect:", key: str = "session_select") -> Optional[str]:
    """Render a reusable session dropdown and return the selected session ID."""
    sessions = get_all_sessions_options()
    if not sessions:
        st.info("No recorded sessions found in the database. Use Section 1 or Section 2 to start or generate a session.")
        return None

    # Determine default index
    default_id = st.session_state.get("selected_session_id") or st.session_state.get("current_session_id")
    sess_ids = [s["id"] for s in sessions]
    default_idx = 0
    if default_id in sess_ids:
        default_idx = sess_ids.index(default_id)

    selected = st.selectbox(
        label,
        options=sess_ids,
        index=default_idx,
        format_func=lambda sid: f"{sid} — {next((s['title'] for s in sessions if s['id'] == sid), '')} ({next((s['status'] for s in sessions if s['id'] == sid), '')})",
        key=key,
    )
    st.session_state["selected_session_id"] = selected
    return selected


# ==============================================================================
# SECTION 1: DASHBOARD OVERVIEW
# ==============================================================================

def render_dashboard_overview() -> None:
    """Render top-level executive metrics, recent sessions, and quick demo launcher."""
    st.markdown("### 📊 System Dashboard Overview")
    st.caption("High-level classroom analytics summary and recent session health.")

    # 1. Overall aggregated statistics from SQLite
    stats = db.count_overview_stats()
    recent_sessions = db.list_sessions(limit=10)

    latest_session_title = recent_sessions[0]["title"] if recent_sessions else "None"
    latest_session_id = recent_sessions[0]["id"] if recent_sessions else "—"

    # Overall activity score from recent fusion scores
    all_fusion_scores = []
    if recent_sessions:
        all_fusion_scores = db.get_fusion_scores(recent_sessions[0]["id"])
    avg_recent_activity = (
        round(sum(s["activity_score"] for s in all_fusion_scores) / len(all_fusion_scores), 1)
        if all_fusion_scores
        else 0.0
    )

    kpi1, kpi2, kpi3, kpi4, kpi5 = st.columns(5)
    with kpi1:
        st.metric("Total Sessions", stats["total_sessions"])
    with kpi2:
        st.metric("Attendance Records", stats["total_attendance_records"])
    with kpi3:
        st.metric("Avg Presence %", f"{stats['avg_presence_percentage']:.1f}%")
    with kpi4:
        st.metric("Latest Activity Proxy", f"{avg_recent_activity:.1f} / 100" if avg_recent_activity > 0 else "—")
    with kpi5:
        st.metric("Active Sessions", sum(1 for s in recent_sessions if s["status"] == SessionStatus.ACTIVE.value))

    st.markdown("---")

    # 2. Action Banner & Demo Launcher
    col_banner, col_action = st.columns([3, 1])
    with col_banner:
        st.markdown(
            f"""
            **Latest Observation:** `{latest_session_title}` (`{latest_session_id}`)  
            To test the full analytics suite without hardware cameras or microphones, click the button on the right to load synthetic data.
            """
        )
    with col_action:
        if st.button("🚀 Load Synthetic Demo Session", use_container_width=True):
            with st.spinner("Generating isolated demonstration records..."):
                demo_id = generate_synthetic_demo_session(db)
                st.session_state["selected_session_id"] = demo_id
                st.session_state["current_session_id"] = demo_id
                st.success(f"Generated demo `{demo_id}`!")
                st.rerun()

    st.markdown("---")
    st.markdown("#### 🗃️ Recent Observation Sessions")

    if not recent_sessions:
        st.info("No recorded sessions found in the database. Start a new session or load a synthetic demo.")
        return

    df_sessions = pd.DataFrame(
        [
            {
                "Session ID": s["id"],
                "Title": s["title"],
                "Status": s["status"].upper(),
                "Start Time": s["start_time"][:19].replace("T", " ") if s.get("start_time") else "—",
                "Duration": f"{int(s.get('duration_seconds', 0.0) // 60)}m {int(s.get('duration_seconds', 0.0) % 60)}s",
                "Created At": s["created_at"][:19].replace("T", " ") if s.get("created_at") else "—",
            }
            for s in recent_sessions
        ]
    )
    st.dataframe(df_sessions, use_container_width=True, hide_index=True)


# ==============================================================================
# SECTION 2: LIVE / RECORDED SESSION
# ==============================================================================

def render_live_recorded_session() -> None:
    """Render camera, video upload, and live session processing controls."""
    st.markdown("### 📹 Live Classroom & Recorded Video Session")
    st.caption("Capture live camera video or upload pre-recorded classroom footage for anonymous presence and activity analysis.")

    # Initialize tracker and detector in session state
    if "tracker" not in st.session_state:
        st.session_state.tracker = AnonymousFaceTracker(
            iou_threshold=config.vision.track_iou_threshold,
            max_inactive_frames=config.vision.track_max_inactive_frames,
        )
    if "detector" not in st.session_state:
        model_path = config.storage.face_model_path
        if model_path.exists():
            try:
                st.session_state.detector = FaceDetectorWrapper(
                    model_path=model_path,
                    min_detection_confidence=config.vision.min_detection_confidence,
                )
            except Exception as e:
                st.error(f"Error loading MediaPipe FaceDetector: {e}")
                st.session_state.detector = None
        else:
            st.session_state.detector = None

    # Session State tracking
    if "session_active" not in st.session_state:
        st.session_state.session_active = False
    if "current_session_id" not in st.session_state:
        st.session_state.current_session_id = None
    if "session_start_time" not in st.session_state:
        st.session_state.session_start_time = None
    if "live_track_events" not in st.session_state:
        st.session_state.live_track_events = {}

    col_ctrl1, col_ctrl2 = st.columns([3, 1])

    with col_ctrl1:
        session_title = st.text_input(
            "Session Title:",
            value=st.session_state.get("session_title", "Lecture: Multimodal Data Systems"),
            key="input_session_title",
        )
        st.session_state["session_title"] = session_title

    with col_ctrl2:
        st.write("")
        st.write("")
        if not st.session_state.session_active:
            if st.button("▶ Start New Session", type="primary", use_container_width=True):
                new_sid = f"sess-{datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S')}"
                st.session_state.current_session_id = new_sid
                st.session_state.session_start_time = time.time()
                st.session_state.session_active = True
                st.session_state.live_track_events = {}
                db.create_session(new_sid, session_title)
                st.success(f"Started session `{new_sid}`")
                st.rerun()
        else:
            if st.button("⏹ Finalize Session", type="secondary", use_container_width=True):
                elapsed = time.time() - (st.session_state.session_start_time or time.time())
                sid = st.session_state.current_session_id

                # Calculate presence attendance
                attendance_engine = PresenceAttendanceEngine(
                    min_presence_percentage=config.attendance.min_presence_percentage,
                    min_presence_duration_seconds=min(elapsed * 0.5, config.attendance.min_presence_duration_seconds),
                    max_disappearance_seconds=config.attendance.max_disappearance_seconds,
                )
                summary = attendance_engine.evaluate_session(
                    session_id=sid,
                    title=session_title,
                    session_duration_seconds=elapsed,
                    track_events=st.session_state.live_track_events,
                    is_in_progress=False,
                )
                for r in summary.records:
                    db.record_attendance(
                        session_id=sid,
                        track_id=r.track_id,
                        status=r.status,
                        first_seen=r.first_seen,
                        last_seen=r.last_seen,
                        presence_duration_seconds=r.presence_duration_seconds,
                        presence_percentage=r.presence_percentage,
                        detection_intervals=r.detection_intervals,
                    )

                db.update_session_status(
                    session_id=sid,
                    status=SessionStatus.COMPLETED.value,
                    end_time=datetime.now(timezone.utc).isoformat(),
                    duration_seconds=elapsed,
                )
                st.session_state.session_active = False
                st.success(f"Finalized session `{sid}` ({int(elapsed)}s duration, {len(summary.records)} attendees).")
                st.rerun()

    # Session Status Banner
    if st.session_state.session_active:
        elapsed_sec = int(time.time() - st.session_state.session_start_time)
        track_cnt = len(st.session_state.live_track_events)
        st.info(
            f"🟢 **SESSION ACTIVE** | ID: `{st.session_state.current_session_id}` | "
            f"Elapsed: **{elapsed_sec // 60}m {elapsed_sec % 60}s** | Active Track Count: **{track_cnt}**"
        )

    st.markdown("---")

    # Input method selection
    input_method = st.radio(
        "Choose Video Processing Source:",
        options=["Synthetic Classroom Simulation", "Upload Video File", "Live Webcam"],
        horizontal=True,
    )

    col_view, col_metrics = st.columns([3, 2])

    if input_method == "Synthetic Classroom Simulation":
        with col_view:
            st.markdown("##### Simulated Multi-Participant Stream")
            st.caption("Demonstrates tracking logic, IoU re-association, and temporary disappearance recovery.")
            sim_steps = st.slider("Simulation Steps", min_value=5, max_value=30, value=12, step=1)
            run_sim = st.button("▶ Run Simulation Sequence", type="primary")
            img_placeholder = st.empty()

        with col_metrics:
            st.markdown("##### Real-Time Anonymous Telemetry")
            telemetry_placeholder = st.empty()

        if run_sim:
            tracker: AnonymousFaceTracker = st.session_state.tracker
            tracker.reset()

            for step in range(1, sim_steps + 1):
                timestamp = step * 0.5
                frame = np.ones((480, 640, 3), dtype=np.uint8) * 240
                # Classroom blackboard background
                cv2.rectangle(frame, (60, 20), (580, 90), (35, 55, 45), -1)
                cv2.putText(frame, "CLASSROOM SIMULATION", (170, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 240, 200), 2)

                dets = []
                # Participant 1 (Alice): smooth sine motion
                x1 = 120 + int(np.sin(step * 0.5) * 12)
                y1 = 180 + int(np.cos(step * 0.5) * 6)
                cv2.circle(frame, (x1 + 45, y1 + 55), 45, (180, 150, 120), -1)
                dets.append(RawDetection(FaceBoundingBox(x1, y1, 90, 110), 0.94, timestamp))

                # Participant 2 (Bob): center desk
                x2 = 280 + (step % 2) * 2
                y2 = 190
                cv2.circle(frame, (x2 + 45, y2 + 55), 45, (160, 150, 130), -1)
                dets.append(RawDetection(FaceBoundingBox(x2, y2, 90, 110), 0.91, timestamp))

                # Participant 3 (Charlie): occluded/disappeared on steps 4, 5, 6
                if step not in (4, 5, 6):
                    x3 = 450 + int(np.cos(step * 0.4) * 8)
                    y3 = 185
                    cv2.circle(frame, (x3 + 45, y3 + 55), 45, (190, 140, 140), -1)
                    dets.append(RawDetection(FaceBoundingBox(x3, y3, 90, 110), 0.88, timestamp))

                result = tracker.update(dets, timestamp=timestamp)
                annotated = annotate_frame(frame, result, low_light=False, show_confidence=True)
                img_placeholder.image(annotated, channels="BGR", use_container_width=True)

                # Record tracking events if session is active
                if st.session_state.session_active:
                    for t in result.active_tracks:
                        tid = t.track_id
                        if tid not in st.session_state.live_track_events:
                            st.session_state.live_track_events[tid] = []
                        st.session_state.live_track_events[tid].append(timestamp)

                track_rows = [
                    {
                        "Track ID": f"#{t.track_id}",
                        "Status": t.status.capitalize(),
                        "Confidence": f"{int(t.confidence * 100)}%",
                        "Detections": t.total_detections,
                        "Movement": f"{t.movement_distance} px",
                    }
                    for t in result.all_tracks
                ]
                if track_rows:
                    telemetry_placeholder.dataframe(pd.DataFrame(track_rows), use_container_width=True, hide_index=True)

                time.sleep(0.06)

            st.success(f"Simulation completed ({sim_steps} frames evaluated).")

    elif input_method == "Upload Video File":
        uploaded_video = st.file_uploader("Upload Classroom Video (MP4, AVI, MOV)", type=["mp4", "avi", "mov"])
        with col_view:
            video_placeholder = st.empty()
        with col_metrics:
            video_telemetry = st.empty()

        if uploaded_video:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp_file:
                tmp_file.write(uploaded_video.read())
                temp_path = tmp_file.name

            try:
                vsource = VideoSource(temp_path)
                if vsource.open():
                    props = vsource.get_properties()
                    st.info(f"Loaded: `{uploaded_video.name}` ({props['width']}x{props['height']} @ {props['fps']:.1f} FPS, {props['frame_count']} frames)")

                    max_frames = st.slider("Process Frame Budget", min_value=10, max_value=min(120, int(props["frame_count"])), value=30)
                    if st.button("▶ Process Uploaded Video Frames", type="primary"):
                        tracker = AnonymousFaceTracker()
                        detector = st.session_state.detector

                        if detector is None:
                            st.error("FaceDetectorWrapper not initialized. Ensure local model exists.")
                        else:
                            frame_idx = 0
                            while frame_idx < max_frames:
                                ret, frame = vsource.read_frame()
                                if not ret or frame is None:
                                    break
                                frame_idx += 1
                                ts = frame_idx / props["fps"]

                                detections, meta = detector.detect_faces(frame, timestamp=ts)
                                result = tracker.update(detections, timestamp=ts)

                                annotated = annotate_frame(frame, result, low_light=meta.get("low_light", False), show_confidence=True)
                                video_placeholder.image(annotated, channels="BGR", use_container_width=True)

                                if st.session_state.session_active:
                                    for t in result.active_tracks:
                                        tid = t.track_id
                                        if tid not in st.session_state.live_track_events:
                                            st.session_state.live_track_events[tid] = []
                                        st.session_state.live_track_events[tid].append(ts)

                                track_rows = [
                                    {
                                        "Track ID": f"#{t.track_id}",
                                        "Status": t.status.capitalize(),
                                        "Confidence": f"{int(t.confidence * 100)}%",
                                        "Detections": t.total_detections,
                                    }
                                    for t in result.all_tracks
                                ]
                                if track_rows:
                                    video_telemetry.dataframe(pd.DataFrame(track_rows), use_container_width=True, hide_index=True)

                            st.success(f"Processed {frame_idx} video frames successfully.")
                    vsource.release()
                else:
                    st.error("Could not open video file.")
            finally:
                Path(temp_path).unlink(missing_ok=True)

    elif input_method == "Live Webcam":
        cam_idx = st.number_input("Camera Index", min_value=0, max_value=5, value=0)
        with col_view:
            cam_placeholder = st.empty()
            enable_cam = st.toggle("Activate Webcam Feed", value=False)
        with col_metrics:
            cam_telemetry = st.empty()

        if enable_cam:
            source = VideoSource(source=int(cam_idx))
            if not source.open():
                st.warning(f"Could not connect to camera index {cam_idx}. Verify device connection and permissions.")
            else:
                detector = st.session_state.detector
                tracker = st.session_state.tracker
                frame_count = 0

                # Read up to 40 frames per toggle iteration
                while frame_count < 40:
                    ret, frame = source.read_frame()
                    if not ret or frame is None:
                        break
                    frame_count += 1
                    ts = time.time()

                    if detector:
                        detections, meta = detector.detect_faces(frame, timestamp=ts)
                        result = tracker.update(detections, timestamp=ts)
                        annotated = annotate_frame(frame, result, low_light=meta.get("low_light", False), show_confidence=True)
                    else:
                        annotated = frame

                    cam_placeholder.image(annotated, channels="BGR", use_container_width=True)
                    time.sleep(0.04)

                source.release()


# ==============================================================================
# SECTION 3: ATTENDANCE
# ==============================================================================

def render_attendance_section() -> None:
    """Render presence attendance ledger, manual roster mapping, and CSV export."""
    st.markdown("### 📋 Presence-Based Attendance System")
    st.caption("Evaluates sustained classroom presence over time. Eliminates single-frame false positives.")

    sid = session_selector(label="Choose Session to Review Attendance:")
    if not sid:
        return

    records = db.get_attendance_records(sid)
    session_info = db.get_session(sid) or {}

    # Attendance KPIs
    total_rec = len(records)
    present_cnt = sum(1 for r in records if r["status"] == AttendanceStatus.PRESENT.value)
    insufficient_cnt = sum(1 for r in records if r["status"] == AttendanceStatus.INSUFFICIENT_PRESENCE.value)
    pending_cnt = sum(1 for r in records if r["status"] == AttendanceStatus.PENDING_CONFIRMATION.value)
    rate = (present_cnt / total_rec * 100.0) if total_rec > 0 else 0.0

    k1, k2, k3, k4 = st.columns(4)
    with k1:
        st.metric("Total Observed", total_rec)
    with k2:
        st.metric("Marked Present", present_cnt, f"{rate:.1f}% Rate")
    with k3:
        st.metric("Insufficient Presence", insufficient_cnt)
    with k4:
        st.metric("Pending Review", pending_cnt)

    st.markdown("---")
    st.markdown(f"#### Attendance Ledger for `{sid}`")

    if records:
        df_records = pd.DataFrame(
            [
                {
                    "Track ID": f"#{r['track_id']}",
                    "Student Name (Roster)": r.get("roster_name") or "— (Unmapped)",
                    "Student ID": r.get("student_id") or "—",
                    "Status": r["status"].replace("_", " ").title(),
                    "Presence Duration": f"{r['presence_duration_seconds']:.1f} s",
                    "Presence %": f"{r['presence_percentage']:.1f}%",
                    "Intervals": r["detection_intervals"],
                    "First Seen": f"{r['first_seen']:.1f}s" if r.get("first_seen") is not None else "—",
                    "Last Seen": f"{r['last_seen']:.1f}s" if r.get("last_seen") is not None else "—",
                    "Notes": r.get("notes") or "",
                }
                for r in records
            ]
        )
        st.dataframe(df_records, use_container_width=True, hide_index=True)

        # CSV Download Button
        csv_buffer = io.StringIO()
        df_records.to_csv(csv_buffer, index=False)
        st.download_button(
            label="📥 Download Attendance CSV Report",
            data=csv_buffer.getvalue(),
            file_name=f"attendance_{sid}.csv",
            mime="text/csv",
        )
    else:
        st.info("No attendance records recorded yet for this session. Complete a session in Section 2 to populate records.")

    st.markdown("---")
    st.markdown("#### 🧑‍🏫 Teacher-Operated Manual Roster Mapping")
    st.caption("Associate anonymous track IDs with named student roster entries. No biometric face recognition is performed.")

    if records:
        map_c1, map_c2, map_c3, map_c4 = st.columns([1, 2, 1, 1])
        with map_c1:
            track_opts = [r["track_id"] for r in records]
            sel_track = st.selectbox("Select Track ID", options=track_opts, key="map_track_sel")
        with map_c2:
            s_name = st.text_input("Student Name", placeholder="e.g. Samantha Vance", key="map_name_inp")
        with map_c3:
            s_id = st.text_input("Student ID", placeholder="e.g. STU-108", key="map_id_inp")
        with map_c4:
            st.write("")
            st.write("")
            if st.button("Save Mapping", type="primary", use_container_width=True):
                if s_name.strip():
                    db.map_roster_participant(
                        session_id=sid,
                        track_id=int(sel_track),
                        roster_name=s_name.strip(),
                        student_id=s_id.strip() if s_id else None,
                    )
                    st.success(f"Track #{sel_track} mapped to **{s_name}**.")
                    st.rerun()


# ==============================================================================
# SECTION 4: ENGAGEMENT ANALYTICS
# ==============================================================================

def render_engagement_analytics() -> None:
    """Render multimodal activity timeline, visual presence, audio activity, and noise charts."""
    st.markdown("### 📈 Multimodal Engagement & Activity Analytics")
    st.caption("Explainable proxy scores combining visual presence coverage, acoustic activity, and physical movement.")

    sid = session_selector(label="Choose Session to Inspect Engagement:")
    if not sid:
        return

    fusion_scores = db.get_fusion_scores(sid)
    audio_metrics = db.get_audio_metrics(sid)

    if not fusion_scores:
        st.info(f"No multimodal fusion metrics recorded for session `{sid}` yet. Run a session or load a demo.")
        return

    # Calculate Summary Metrics
    act_scores = [f["activity_score"] for f in fusion_scores]
    mean_score = sum(act_scores) / len(act_scores) if act_scores else 0.0
    mean_conf = sum(f["confidence"] for f in fusion_scores) / len(fusion_scores) if fusion_scores else 0.0

    k1, k2, k3, k4 = st.columns(4)
    with k1:
        st.metric("Mean Activity Score", f"{mean_score:.1f} / 100")
    with k2:
        st.metric("Activity Classification", "High" if mean_score >= 70 else ("Moderate" if mean_score >= 30 else "Low"))
    with k3:
        st.metric("Mean Data Confidence", f"{int(mean_conf * 100)}%")
    with k4:
        st.metric("Timeline Windows", len(fusion_scores))

    times = [f["window_end"] for f in fusion_scores]
    scores = [f["activity_score"] for f in fusion_scores]
    vis_vals = [f["visual_presence"] * 100 for f in fusion_scores]
    aud_vals = [f["audio_activity"] * 100 for f in fusion_scores]
    mov_vals = [f["movement_level"] * 100 for f in fusion_scores]

    # Chart 1: Main Activity Timeline with Shaded Zones
    fig_timeline = go.Figure()
    fig_timeline.add_hrect(y0=0, y1=30, fillcolor="#94a3b8", opacity=0.15, line_width=0, annotation_text="Low Activity Zone")
    fig_timeline.add_hrect(y0=30, y1=70, fillcolor="#10b981", opacity=0.15, line_width=0, annotation_text="Moderate Activity Zone")
    fig_timeline.add_hrect(y0=70, y1=100, fillcolor="#f59e0b", opacity=0.15, line_width=0, annotation_text="High Activity Zone")

    fig_timeline.add_trace(
        go.Scatter(
            x=times,
            y=scores,
            mode="lines+markers",
            name="Activity Score Proxy",
            line=dict(color="#2563eb", width=3),
            marker=dict(size=6),
        )
    )
    fig_timeline.update_layout(
        title="Multimodal Activity Timeline (0 - 100 Proxy)",
        xaxis_title="Session Time (seconds)",
        yaxis_title="Activity Score",
        yaxis_range=[0, 105],
        height=320,
        margin=dict(l=20, r=20, t=40, b=20),
    )
    st.plotly_chart(fig_timeline, use_container_width=True)

    col_c1, col_c2 = st.columns(2)

    with col_c1:
        # Chart 2: Modality Signals Comparison
        fig_signals = go.Figure()
        fig_signals.add_trace(go.Scatter(x=times, y=vis_vals, name="Visual Presence %", line=dict(color="#2563eb", width=2)))
        fig_signals.add_trace(go.Scatter(x=times, y=aud_vals, name="Audio Activity %", line=dict(color="#10b981", width=2)))
        fig_signals.add_trace(go.Scatter(x=times, y=mov_vals, name="Physical Motion %", line=dict(color="#f59e0b", width=2, dash="dot")))
        fig_signals.update_layout(
            title="Individual Modality Signal Trajectories",
            xaxis_title="Session Time (seconds)",
            yaxis_title="Normalized Ratio (%)",
            yaxis_range=[0, 105],
            height=300,
            margin=dict(l=20, r=20, t=40, b=20),
        )
        st.plotly_chart(fig_signals, use_container_width=True)

    with col_c2:
        # Chart 3: Constituent Point Contribution Breakdown
        w_v = config.fusion.weight_visual_presence
        w_a = config.fusion.weight_audio_activity
        w_m = config.fusion.weight_movement

        c_vis = [v * w_v for v in vis_vals]
        c_aud = [a * w_a for a in aud_vals]
        c_mov = [m * w_m for m in mov_vals]

        fig_bar = go.Figure()
        fig_bar.add_trace(go.Bar(x=times, y=c_vis, name="Vision Contribution", marker_color="#2563eb"))
        fig_bar.add_trace(go.Bar(x=times, y=c_aud, name="Audio Contribution", marker_color="#10b981"))
        fig_bar.add_trace(go.Bar(x=times, y=c_mov, name="Movement Contribution", marker_color="#f59e0b"))
        fig_bar.update_layout(
            barmode="stack",
            title="Constituent Score Contributions",
            xaxis_title="Session Time (seconds)",
            yaxis_title="Score Contribution",
            height=300,
            margin=dict(l=20, r=20, t=40, b=20),
        )
        st.plotly_chart(fig_bar, use_container_width=True)

    # Telemetry Ledger Table & CSV Download
    st.markdown("##### Synchronized Window Telemetry Ledger")
    df_timeline = pd.DataFrame(
        [
            {
                "Window End": f"{f['window_end']:.1f} s",
                "Visual Presence": f"{f['visual_presence'] * 100:.1f}%",
                "Audio Activity": f"{f['audio_activity'] * 100:.1f}%",
                "Movement Level": f"{f['movement_level'] * 100:.1f}%",
                "Activity Score": f"{f['activity_score']:.1f}",
                "Confidence": f"{int(f['confidence'] * 100)}%",
                "Label": f["label"],
            }
            for f in fusion_scores
        ]
    )
    st.dataframe(df_timeline, use_container_width=True, hide_index=True)

    csv_buf = io.StringIO()
    df_timeline.to_csv(csv_buf, index=False)
    st.download_button(
        label="📥 Download Timeline CSV",
        data=csv_buf.getvalue(),
        file_name=f"timeline_{sid}.csv",
        mime="text/csv",
    )


# ==============================================================================
# SECTION 5: TEACHING SUGGESTIONS
# ==============================================================================

def render_teaching_suggestions() -> None:
    """Render rule-based pedagogical suggestions, evidence cards, and advice history."""
    st.markdown("### 💡 Teaching Suggestions & Pedagogical Agent")
    st.caption("Context-aware, rule-based instructional decision support based on observed multimodal trends.")

    sid = session_selector(label="Choose Session to Review Suggestions:")
    if not sid:
        return

    recommendations = db.get_recommendations(sid)

    # Top metrics
    urgent_cnt = sum(1 for r in recommendations if r.get("priority") == "Urgent")
    warning_cnt = sum(1 for r in recommendations if r.get("priority") == "Warning")
    medium_cnt = sum(1 for r in recommendations if r.get("priority") == "Medium")
    info_cnt = sum(1 for r in recommendations if r.get("priority") == "Info")

    k1, k2, k3, k4, k5 = st.columns(5)
    with k1:
        st.metric("Total Suggestions", len(recommendations))
    with k2:
        st.metric("Urgent Actions", urgent_cnt)
    with k3:
        st.metric("Warnings", warning_cnt)
    with k4:
        st.metric("Recommendations", medium_cnt)
    with k5:
        st.metric("Observations", info_cnt)

    st.info(
        "ℹ️ **Pedagogical Decision Support Only** — All suggestions are generated by transparent, offline deterministic rules. "
        "They do not make automated judgments, draw biometric conclusions, or evaluate individual student capability."
    )

    if not recommendations:
        st.info("No teaching suggestions recorded for this session yet. Run a session with multimodal activity to generate suggestions.")
        return

    st.markdown("---")
    st.markdown("#### Contextual Suggestion Cards")

    # Priority styling map
    p_colors = {
        "Urgent": ("#ef4444", "#fef2f2"),
        "Warning": ("#f59e0b", "#fffbeb"),
        "Medium": ("#2563eb", "#eff6ff"),
        "Info": ("#10b981", "#ecfdf5"),
    }

    for rec in recommendations:
        p = rec.get("priority", "Medium")
        border_color, bg_color = p_colors.get(p, ("#64748b", "#f8fafc"))
        timestamp_str = f"{rec['timestamp']:.1f}s" if isinstance(rec.get("timestamp"), (int, float)) else str(rec.get("timestamp"))

        st.markdown(
            f"""
            <div style="border-left: 5px solid {border_color}; background-color: {bg_color}; padding: 1rem 1.2rem; border-radius: 6px; margin-bottom: 1rem; border-top: 1px solid #e2e8f0; border-right: 1px solid #e2e8f0; border-bottom: 1px solid #e2e8f0;">
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <span style="font-weight: 700; font-size: 1.05rem; color: #0f172a;">
                        {rec['condition']}
                    </span>
                    <div>
                        <span style="background-color: {border_color}; color: white; font-weight: 600; font-size: 0.75rem; padding: 0.2rem 0.5rem; border-radius: 4px; margin-right: 0.4rem;">
                            {p.upper()}
                        </span>
                        <span style="background-color: #e2e8f0; color: #334155; font-size: 0.75rem; padding: 0.2rem 0.5rem; border-radius: 4px;">
                            ⏱ {timestamp_str}
                        </span>
                    </div>
                </div>
                <p style="margin: 0.5rem 0 0.3rem 0; font-size: 0.92rem; color: #334155;">
                    <strong>Observation:</strong> {rec['explanation']}
                </p>
                <p style="margin: 0.3rem 0 0.3rem 0; font-size: 0.92rem; color: #0f172a; font-weight: 600;">
                    💡 <strong>Suggested Teaching Action:</strong> {rec['suggestion']}
                </p>
                <div style="margin-top: 0.4rem; font-size: 0.8rem; color: #64748b;">
                    Rule Source: <code>{rec.get('rule_source', 'heuristic')}</code> | Confidence: <strong>{rec.get('confidence', 'Medium')}</strong>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        with st.expander(f"View Supporting Evidence & Metrics ({rec['id']})"):
            supp_metrics = rec.get("supporting_metrics", {})
            st.json(supp_metrics)


# ==============================================================================
# SECTION 6: SESSION REPORTS
# ==============================================================================

def render_session_reports() -> None:
    """Render comprehensive session analytical summary and multi-table CSV downloads."""
    st.markdown("### 📑 Session Analytical Reports & Exports")
    st.caption("Consolidated pedagogical reports combining attendance, acoustic profiles, activity trends, and suggestions.")

    sid = session_selector(label="Choose Session to Generate Report:")
    if not sid:
        return

    summary = generate_session_summary(db, sid)
    if "error" in summary:
        st.error(summary["error"])
        return

    st.markdown(f"#### Executive Report: `{summary['title']}`")
    st.caption(f"Session ID: `{sid}` | Status: **{summary['status'].upper()}** | Duration: **{int(summary.get('duration_seconds', 0.0) // 60)}m {int(summary.get('duration_seconds', 0.0) % 60)}s**")

    # High-level Summary Metrics
    att = summary["attendance"]
    act = summary["activity"]
    recs = summary["recommendations"]

    r1, r2, r3, r4 = st.columns(4)
    with r1:
        st.metric("Total Attendees", att["total_participants"])
    with r2:
        st.metric("Present Count", att["present"], f"{att['average_presence_percentage']:.1f}% Avg")
    with r3:
        st.metric("Mean Activity Score", f"{act['mean_activity_score']:.1f} / 100")
    with r4:
        st.metric("Teaching Suggestions", recs["total"])

    tab_att, tab_act, tab_recs = st.tabs(["Attendance Ledger", "Activity Profile", "Teaching Advice Record"])

    with tab_att:
        if att["records"]:
            df_att = pd.DataFrame(
                [
                    {
                        "Track ID": f"#{r['track_id']}",
                        "Roster Name": r.get("roster_name") or "Unmapped",
                        "Student ID": r.get("student_id") or "—",
                        "Status": r["status"].replace("_", " ").title(),
                        "Presence Duration": f"{r['presence_duration_seconds']:.1f}s",
                        "Presence %": f"{r['presence_percentage']:.1f}%",
                    }
                    for r in att["records"]
                ]
            )
            st.dataframe(df_att, use_container_width=True, hide_index=True)
        else:
            st.info("No attendance records for this session.")

    with tab_act:
        if act["timeline"]:
            df_timeline = pd.DataFrame(
                [
                    {
                        "Window End": f"{f['window_end']:.1f}s",
                        "Visual Presence": f"{f['visual_presence'] * 100:.1f}%",
                        "Audio Activity": f"{f['audio_activity'] * 100:.1f}%",
                        "Activity Score": f"{f['activity_score']:.1f}",
                        "Label": f["label"],
                    }
                    for f in act["timeline"]
                ]
            )
            st.dataframe(df_timeline, use_container_width=True, hide_index=True)
        else:
            st.info("No timeline scores recorded.")

    with tab_recs:
        if recs["items"]:
            df_recs = pd.DataFrame(
                [
                    {
                        "Time": f"{r['timestamp']:.1f}s" if isinstance(r.get("timestamp"), (int, float)) else str(r.get("timestamp")),
                        "Priority": r.get("priority", "Medium"),
                        "Condition": r["condition"],
                        "Suggested Action": r["suggestion"],
                        "Confidence": r.get("confidence", "Medium"),
                    }
                    for r in recs["items"]
                ]
            )
            st.dataframe(df_recs, use_container_width=True, hide_index=True)
        else:
            st.info("No recommendations recorded.")

    st.markdown("---")
    # CSV Export
    csv_buf = io.StringIO()
    if att["records"]:
        pd.DataFrame(att["records"]).to_csv(csv_buf, index=False)
    st.download_button(
        label="📥 Download Complete Session Report CSV",
        data=csv_buf.getvalue(),
        file_name=f"report_{sid}.csv",
        mime="text/csv",
    )


# ==============================================================================
# SECTION 7: SETTINGS
# ==============================================================================

def render_settings_section() -> None:
    """Render configuration parameters, thresholds, scoring weights, and privacy guarantees."""
    st.markdown("### ⚙️ System Configuration & Privacy Controls")
    st.caption("Adjust processing parameters, detection thresholds, and multimodal fusion weights.")

    col_s1, col_s2 = st.columns(2)

    with col_s1:
        st.markdown("#### 🎥 Computer Vision & Tracking Parameters")
        cam_idx = st.number_input("Camera Device Index", min_value=0, max_value=5, value=0, key="set_cam_idx")
        min_det_conf = st.slider("MediaPipe Face Detection Confidence", min_value=0.3, max_value=0.95, value=float(config.vision.min_detection_confidence), step=0.05, key="set_det_conf")
        iou_thresh = st.slider("Tracking IoU Overlap Threshold", min_value=0.1, max_value=0.7, value=float(config.vision.track_iou_threshold), step=0.05, key="set_iou")
        max_inact = st.slider("Max Inactive Tracking Frames Before Timeout", min_value=5, max_value=60, value=int(config.vision.track_max_inactive_frames), step=5, key="set_inact")

        st.markdown("#### 🎙️ Classroom Sound Analysis Parameters")
        audio_win = st.slider("Audio Window Duration (seconds)", min_value=1.0, max_value=15.0, value=float(config.audio.window_duration_seconds), step=1.0, key="set_aud_win")
        rms_th = st.slider("RMS Energy Activity Threshold", min_value=0.005, max_value=0.050, value=float(config.audio.rms_activity_threshold), step=0.005, format="%.3f", key="set_rms")

    with col_s2:
        st.markdown("#### 📋 Presence Attendance Rules")
        min_pct = st.slider("Minimum Presence Percentage (%)", min_value=10.0, max_value=100.0, value=float(config.attendance.min_presence_percentage), step=5.0, key="set_min_pct")
        min_dur = st.number_input("Minimum Presence Duration (seconds)", min_value=10, max_value=3600, value=int(config.attendance.min_presence_duration_seconds), step=30, key="set_min_dur")
        max_disap = st.slider("Maximum Temporary Disappearance Gap (seconds)", min_value=5.0, max_value=60.0, value=float(config.attendance.max_disappearance_seconds), step=5.0, key="set_disap")

        st.markdown("#### ⚡ Multimodal Signal Fusion Weights")
        w_v = st.slider("Visual Presence Weight", min_value=0.0, max_value=1.0, value=float(config.fusion.weight_visual_presence), step=0.05, key="set_w_v")
        w_a = st.slider("Audio Activity Weight", min_value=0.0, max_value=1.0, value=float(config.fusion.weight_audio_activity), step=0.05, key="set_w_a")
        w_m = st.slider("Physical Movement Weight", min_value=0.0, max_value=1.0, value=float(config.fusion.weight_movement), step=0.05, key="set_w_m")

        total_w = w_v + w_a + w_m
        if total_w > 0:
            st.caption(f"Normalized Weights: Vision **{w_v/total_w:.2f}** | Audio **{w_a/total_w:.2f}** | Motion **{w_m/total_w:.2f}**")
        else:
            st.warning("Weights sum to zero.")

    st.markdown("---")
    st.markdown("#### 🔒 Security & Privacy Audit Checklist")

    st.markdown(
        """
        - ✅ **Zero Cloud Transmission:** All video frames and acoustic audio windows are processed exclusively on this local machine.
        - ✅ **No Biometric Identification:** No facial recognition against identity databases or biometric embeddings.
        - ✅ **Ephemeral Processing:** Raw video frames are discarded immediately after feature extraction.
        - ✅ **SQL Injection Safeguard:** All SQLite database queries use parameterized prepared statements.
        - ✅ **Teacher Agency:** All pedagogical suggestions are purely advisory decision support.
        """
    )


# ==============================================================================
# MAIN APPLICATION CONTROLLER
# ==============================================================================

def main() -> None:
    """Main application layout controller with the 7 core sections."""
    render_header()
    render_sidebar()

    tabs = st.tabs(
        [
            "📊 Dashboard Overview",
            "📹 Live / Recorded Session",
            "📋 Attendance",
            "📈 Engagement Analytics",
            "💡 Teaching Suggestions",
            "📑 Session Reports",
            "⚙️ Settings",
        ]
    )

    with tabs[0]:
        render_dashboard_overview()

    with tabs[1]:
        render_live_recorded_session()

    with tabs[2]:
        render_attendance_section()

    with tabs[3]:
        render_engagement_analytics()

    with tabs[4]:
        render_teaching_suggestions()

    with tabs[5]:
        render_session_reports()

    with tabs[6]:
        render_settings_section()


if __name__ == "__main__":
    main()
