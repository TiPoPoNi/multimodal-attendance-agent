# Multimodal Attendance & Engagement Agent

[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/release/python-3110/)
[![Tests Passing](https://img.shields.io/badge/pytest-70%20passed-success.svg)](file:///tests)
[![Privacy First](https://img.shields.io/badge/privacy-100%25%20local%20offline-green.svg)](#privacy-and-ethical-safeguards)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

A local-first, privacy-preserving classroom analytics system that combines computer vision, acoustic sound processing, presence-based attendance verification, and rule-based pedagogical advice. Operates **100% offline** with zero external LLM dependencies, no cloud data transmission, and no biometric student identification.

---

## Table of Contents

- [Problem Statement](#problem-statement)
- [Key Features](#key-features)
- [Technical Architecture](#technical-architecture)
- [Project Structure](#project-structure)
- [System Requirements](#system-requirements)
- [Installation & Setup](#installation--setup)
- [MediaPipe Model Setup](#mediapipe-model-setup)
- [Running the Application](#running-the-application)
- [Dashboard Walkthrough](#dashboard-walkthrough)
- [Testing Suite](#testing-suite)
- [Privacy and Ethical Safeguards](#privacy-and-ethical-safeguards)
- [Known Limitations](#known-limitations)
- [Future Enhancements](#future-enhancements)
- [Publishing to GitHub](#publishing-to-github)

---

## Problem Statement

Traditional attendance methods (manual roll calls, sign-in sheets, RFID card taps) are slow, error-prone, and prone to proxy attendance. Conversely, commercial facial-recognition surveillance platforms present severe privacy and legal liabilities by generating permanent biometric templates and uploading sensitive student images to remote cloud servers. Furthermore, instructors lack real-time, non-invasive indicators of whether classroom discussions are active, stagnant, or disrupted.

**Multimodal Attendance & Engagement Agent** resolves this tension by providing:
1. **Sustained Presence Attendance:** Evaluates temporal presence intervals across an entire lecture rather than a momentary single-frame snapshot.
2. **Anonymous Tracking:** Tracks participants using localized bounding boxes and temporary IDs (`#1`, `#2`, ...); never stores facial embeddings.
3. **Multimodal Activity Proxy:** Fuses normalized visual presence, ambient sound energy, and physical movement into an explainable 0–100 activity index.
4. **Assistive Pedagogical Engine:** Transparent rule-based decision support that suggests timely instructional interventions (e.g. think-pair-share, comprehension checks) without evaluating individual student psychology.

---

## Key Features

- **Local-First & Offline:** Runs entirely on the instructor's local machine; no API keys, cloud tokens, or internet connection required.
- **Anonymous Face Tracking:** OpenCV + MediaPipe Face Detector (Tasks API) paired with Intersection-over-Union (IoU) spatial matching, timeout pruning, and occlusion recovery.
- **Presence-Based Attendance Engine:** Configurable presence duration, disappearance tolerance, and 4-tier attendance classification (`Present`, `Insufficient Presence`, `Pending Confirmation`, `Not Detected`).
- **Teacher Roster Association:** Manual instructor mapping of anonymous track IDs to student roster entries; no automated biometric face identification.
- **Acoustic Classroom Analysis:** Windowed RMS energy, Zero-Crossing Rate (ZCR), silence detection, and noise-level classification via Librosa and SoundFile. Does **not** perform speech-to-text or voiceprint profiling.
- **Explainable Multimodal Fusion:** Window-by-window alignment with explicit component point contributions (`visual_contribution`, `audio_contribution`, `movement_contribution`).
- **Rule-Based Teaching Agent:** Deterministic pedagogical rules prioritizing alerts (Urgent > Warning > Medium > Info) with cooldown deduplication to avoid alert fatigue.
- **Comprehensive 7-Section Dashboard:** Interactive Streamlit interface with Plotly charts, session timers, and instant CSV exports.
- **Synthetic Demonstration Mode:** Complete hardware-free demo generator allowing users to evaluate all 7 tabs without physical webcams or microphones.

---

## Technical Architecture

```
                    ┌─────────────────────────┐
                    │ Video Input (Webcam/MP4)│
                    └────────────┬────────────┘
                                 │
                     ┌───────────▼───────────┐
                     │ MediaPipe FaceDetector│
                     └───────────┬───────────┘
                                 │ Detections
                     ┌───────────▼───────────┐
                     │ AnonymousFaceTracker  │  (IoU Spatial Re-association)
                     └─────┬───────────┬─────┘
                           │           │
       Tracking Events     │           │ Track Timestamps
┌──────────────────────────▼───┐   ┌───▼──────────────────────────┐
│ SQLite Database              │   │ PresenceAttendanceEngine     │
│ (sessions, attendance, etc.) │   └───┬──────────────────────────┘
└──────────────────────────┬───┘       │ Presence Statuses
                           │           ▼
                           │       ┌──────────────────────────────┐
┌───────────────────────┐  │       │ Manual Teacher Roster Map    │
│ Audio Input (Mic/WAV) │  │       └──────────────────────────────┘
└───────────┬───────────┘  │
            │              │
┌───────────▼───────────┐  │
│ AudioAnalyzer         │  │ (RMS Energy, ZCR, Noise Level)
└───────────┬───────────┘  │
            │              │
            ▼              │
┌──────────────────────────▼───┐
│ MultimodalSynchronizer       │ (Discrete Time-Window Binning)
└───────────┬──────────────────┘
            │ Synchronized Windows
┌───────────▼──────────────────┐
│ MultimodalFusionEngine       │ (Explainable 0-100 Activity Proxy)
└───────────┬──────────────────┘
            │ Window Scores
┌───────────▼──────────────────┐
│ TeachingRecommendationEngine │ (Rule Evaluation, Cooldowns, Priority)
└───────────┬──────────────────┘
            │ Recommendations
┌───────────▼─────────────────────────────────────────────────────┐
│ Streamlit Interactive Analytics Dashboard (7 Core Sections)     │
└─────────────────────────────────────────────────────────────────┘
```

---

## Project Structure

```text
multimodal-attendance-agent/
│
├── app.py                      # Main Streamlit application entry point (7 sections)
├── requirements.txt            # Python dependencies (pinned versions)
├── pytest.ini                  # Pytest test discovery and path configuration
├── README.md                   # Full system documentation and guide
├── LICENSE                     # MIT License placeholder
├── .gitignore                  # Git exclusions for models, databases, and caches
├── .env.example                # Local environment configuration template
│
├── src/
│   └── attendance_agent/
│       ├── __init__.py         # Package initialization
│       ├── config.py           # Typed dataclass application settings
│       ├── models.py           # Domain dataclasses and enumerations
│       │
│       ├── vision/             # Computer vision & face tracking
│       │   ├── __init__.py
│       │   ├── detector.py     # MediaPipe FaceDetector Tasks API wrapper
│       │   ├── tracker.py      # Anonymous tracking engine with IoU matching
│       │   ├── capture.py      # Video capture and frame annotation utilities
│       │   └── attendance.py   # Presence-based attendance calculation engine
│       │
│       ├── audio/              # Audio capture and acoustic analysis
│       │   ├── __init__.py
│       │   ├── capture.py      # Audio loading, mono conversion, ffmpeg extraction
│       │   └── analyzer.py     # Windowed RMS energy, ZCR, and noise classification
│       │
│       ├── fusion/             # Multimodal signal fusion
│       │   ├── __init__.py
│       │   ├── synchronizer.py # Cross-modal timeline window alignment
│       │   └── scoring.py      # Explainable 0-100 activity proxy scoring
│       │
│       ├── agent/              # Pedagogical recommendation agent
│       │   ├── __init__.py
│       │   └── recommendations.py # Deterministic rules, priority, and cooldowns
│       │
│       ├── analytics/          # Session analytics & CSV exports
│       │   ├── __init__.py
│       │   └── reports.py      # Summary generation, timeline export, demo sessions
│       │
│       ├── storage/            # Local data persistence
│       │   ├── __init__.py
│       │   └── database.py     # Parameterized SQLite DatabaseManager
│       │
│       └── utils/              # Shared utilities
│           ├── __init__.py
│           └── logger.py       # Standard structured logging configuration
│
├── data/
│   ├── models/                 # Local MediaPipe model weights (.task)
│   ├── sessions/               # SQLite database (attendance.db) and logs
│   └── samples/                # Sample video and audio media files
│
├── scripts/                    # Verification demonstration scripts
│   ├── demo_vision.py          # Vision detection & anonymous tracking CLI demo
│   ├── demo_audio.py           # Audio extraction & acoustic analysis CLI demo
│   ├── demo_fusion.py          # Multimodal timeline fusion & scoring CLI demo
│   └── demo_recommendations.py # Rule-based teaching advice CLI demo
│
└── tests/                      # Automated Pytest suite (70 tests)
    ├── __init__.py
    ├── test_config.py          # Configuration and directory initialization tests
    ├── test_database.py        # SQLite schema, foreign keys, and CRUD tests
    ├── test_vision.py          # Frame capture and annotation unit tests
    ├── test_tracker.py         # Face tracking, IoU matching, timeout tests
    ├── test_attendance.py      # Presence intervals and attendance rule tests
    ├── test_audio.py           # Audio extraction, RMS energy, and ZCR tests
    ├── test_fusion.py          # Timeline synchronization and fusion scoring tests
    ├── test_recommendations.py # Teaching recommendation rules and cooldown tests
    ├── test_reports.py         # Summary generation and synthetic demo tests
    └── test_edge_cases.py      # Resilience tests (empty DB, corrupt files, silences)
```

---

## System Requirements

- **Operating System:** Windows 10/11, macOS 12+, or Linux (Ubuntu 22.04+).
- **Python:** Python 3.11.x (recommended).
- **Hardware:** Standard CPU (2+ cores), 4 GB RAM, integrated webcam or USB camera (optional).
- **External Binaries:** FFmpeg (optional, recommended for extracting audio from video files).

---

## Installation & Setup

### 1. Clone the Repository
```bash
git clone https://github.com/your-username/multimodal-attendance-agent.git
cd multimodal-attendance-agent
```

### 2. Create and Activate Virtual Environment
```bash
# Windows (PowerShell)
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# Linux / macOS
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Configure Environment Variables
Copy `.env.example` to `.env`:
```bash
# Windows
copy .env.example .env

# Linux / macOS
cp .env.example .env
```

---

## MediaPipe Model Setup

MediaPipe FaceDetector requires a local `.task` weights file inside `data/models/`. To maintain privacy and offline operation, the model is loaded locally:

```bash
# Verify the model file exists in data/models/
data/models/blaze_face_short_range.task
```

If not present, download the short-range face detection model from Google's official repository:
```bash
# PowerShell
Invoke-WebRequest -Uri "https://storage.googleapis.com/mediapipe-models/face_detector/blaze_face_short_range/float16/1/blaze_face_short_range.task" -OutFile "data/models/blaze_face_short_range.task"

# Linux / macOS
curl -L -o data/models/blaze_face_short_range.task https://storage.googleapis.com/mediapipe-models/face_detector/blaze_face_short_range/float16/1/blaze_face_short_range.task
```

---

## Running the Application

Launch the Streamlit web dashboard:
```bash
streamlit run app.py
```
Open your browser at `http://localhost:8501`.

---

## Dashboard Walkthrough

The dashboard is structured into 7 functional sections:

### 1. 📊 Dashboard Overview
- Executive summary metrics: Total Sessions, Attendance Records, Average Presence %, and Active Sessions count.
- **One-Click Demo Generator:** Click **"🚀 Load Synthetic Demo Session"** to instantly populate realistic multi-student lecture data without needing a physical camera or microphone.
- Recent observation sessions table with live status badges.

### 2. 📹 Live / Recorded Session
- Choose video input source: **Synthetic Classroom Simulation**, **Upload Video File** (MP4, AVI, MOV), or **Live Webcam**.
- Real-time session controls: Start session, live elapsed timer, participant count, and video preview annotated with tracking IDs and bounding boxes.
- Finalizing a session automatically runs the attendance, fusion, and pedagogical advice pipelines and commits results to SQLite.

### 3. 📋 Attendance
- Session selector dropdown.
- Attendance metrics: Total Observed, Marked Present (with % rate), Insufficient Presence, and Pending Review.
- **Manual Roster Mapping:** Form allowing teachers to assign anonymous track IDs (`#1`, `#2`) to real student names and IDs.
- Comprehensive attendance table with presence duration, percentage, and detection intervals.
- Instant CSV export of session attendance records.

### 4. 📈 Engagement Analytics
- Multimodal Activity Timeline (0–100 Proxy) plotted against Low (`< 30`), Moderate (`30–70`), and High (`> 70`) engagement zones.
- Modality signal comparison comparing Visual Presence % vs Audio Activity % vs Physical Motion %.
- Constituent score contribution breakdown (stacked bar chart showing point attribution).
- Synchronized window telemetry ledger with CSV download.

### 5. 💡 Teaching Suggestions
- Alert metrics broken down by Priority: Urgent, Warning, Medium, and Info.
- Distinctly styled pedagogical cards displaying:
  - Timestamp of detection.
  - Detected condition (e.g., *Sustained Low Classroom Activity*, *Elevated Classroom Ambient Noise*).
  - Clear observational explanation.
  - Actionable suggested teaching strategy (e.g., think-pair-share prompt, formative question).
  - Supporting evidence metrics drawer.

### 6. 📑 Session Reports
- Executive multi-tab reports synthesizing attendance, activity trends, acoustic noise profiles, and teaching records.
- One-click **Download Complete Session Report CSV** button.

### 7. ⚙️ Settings
- Configurable computer vision parameters (FPS, confidence, IoU overlap).
- Presence attendance thresholds (minimum presence %, duration, disappearance gap).
- Audio processing thresholds (RMS threshold, ZCR threshold, window duration).
- Multimodal scoring weights (Visual, Audio, Movement).
- Local Privacy & Security verification checklist.

---

## Testing Suite

The project includes an extensive automated test suite covering all modules, calculations, and edge cases.

### Run All 70 Automated Tests
```bash
pytest -v
```

### Run Tests by Functional Area
```bash
# Vision & anonymous tracking tests
pytest tests/test_vision.py tests/test_tracker.py -v

# Presence attendance engine tests
pytest tests/test_attendance.py -v

# Acoustic signal analysis tests
pytest tests/test_audio.py -v

# Multimodal fusion scoring tests
pytest tests/test_fusion.py -v

# Rule-based recommendation engine tests
pytest tests/test_recommendations.py -v

# Edge-case and robustness tests
pytest tests/test_edge_cases.py -v
```

### Run Standalone CLI Demos
```bash
python scripts/demo_vision.py
python scripts/demo_audio.py
python scripts/demo_fusion.py
python scripts/demo_recommendations.py
```

---

## Privacy and Ethical Safeguards

1. **Local-First Execution:** Visual and acoustic processing execute strictly on the local CPU/GPU. No data leaves the classroom device.
2. **No Facial Recognition Database:** Faces are localized with temporary bounding-box tracking IDs (`#1`, `#2`, ...). The system **never** computes or stores facial embeddings.
3. **No Biometric Surveillance:** The system does not classify emotion, attentiveness, or psychological states. The activity score is explicitly defined as an *observable multimodal signal proxy*.
4. **No Raw Media Retention:** Video frames and audio windows are processed in memory and cleared immediately.
5. **Human-in-the-Loop Roster Association:** Attendance is mapped manually by the instructor. The system never automatically identifies students.
6. **Teacher Agency:** All pedagogical suggestions are advisory decision-support heuristics designed to assist teachers, never to replace instructional judgment.

---

## Known Limitations

- **Camera Field of View:** Fixed webcam angles may suffer from visual occlusion in large or multi-tiered lecture halls.
- **Ambient Microphone Noise:** In non-acoustic rooms, external ambient noise (HVAC, street traffic) can elevate audio energy metrics without classroom interaction.
- **Single-Channel Microphones:** Directional acoustic isolation of individual student speakers is not supported (by design to safeguard student privacy).
- **Lighting Sensitivity:** MediaPipe face detection confidence degrades under severe backlighting or extremely low illumination.

---

## Future Enhancements

- Support for multiple USB camera feeds with homography stitching.
- Adaptive baseline calibration for classroom acoustic background noise.
- Export to institutional LMS platforms (Canvas, Blackboard, Moodle) via standard CSV/LTI formats.
- Offline visual heatmaps showing aggregate classroom movement density.

---

## Publishing to GitHub

To publish this project to GitHub as a clean, production-ready repository:

```bash
# 1. Initialize git repository
git init

# 2. Stage files (excluding models, cache, and db per .gitignore)
git add .

# 3. Check staged files
git status

# 4. Commit initial release
git commit -m "Initial release: Multimodal Attendance & Engagement Agent v1.0.0"

# 5. Link to your remote GitHub repository
git remote add origin https://github.com/<your-username>/multimodal-attendance-agent.git
git branch -M main
git push -u origin main
```

---

## License

This project is licensed under the terms of the [MIT License](LICENSE).
