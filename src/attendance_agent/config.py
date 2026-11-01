"""Application configuration module.

Handles directory resolution, environment variable loading, and typed settings
for all submodules (Vision, Audio, Attendance, Fusion, Storage, Agent).
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

# Resolve project root relative to this configuration file
PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent.parent

# Load local .env file if available
load_dotenv(PROJECT_ROOT / ".env")


@dataclass(frozen=True)
class StorageConfig:
    """Storage and path settings."""

    root_dir: Path = PROJECT_ROOT
    data_dir: Path = field(default_factory=lambda: PROJECT_ROOT / "data")
    models_dir: Path = field(default_factory=lambda: PROJECT_ROOT / "data" / "models")
    sessions_dir: Path = field(default_factory=lambda: PROJECT_ROOT / "data" / "sessions")
    samples_dir: Path = field(default_factory=lambda: PROJECT_ROOT / "data" / "samples")
    database_path: Path = field(
        default_factory=lambda: Path(
            os.getenv("DATABASE_PATH", str(PROJECT_ROOT / "data" / "sessions" / "attendance.db"))
        )
    )
    @staticmethod
    def _resolve_default_model_path() -> Path:
        """Resolve the local MediaPipe face detection model path."""
        env_path = os.getenv("MODEL_PATH")
        if env_path:
            return Path(env_path)
        default_task = PROJECT_ROOT / "data" / "models" / "blaze_face_short_range.task"
        if default_task.exists():
            return default_task
        default_tflite = PROJECT_ROOT / "data" / "models" / "blaze_face_short_range.tflite"
        if default_tflite.exists():
            return default_tflite
        return default_task

    face_model_path: Path = field(
        default_factory=lambda: StorageConfig._resolve_default_model_path()
    )


@dataclass(frozen=True)
class VisionConfig:
    """Computer vision and face detection/tracking parameters."""

    camera_index: int = int(os.getenv("CAMERA_INDEX", "0"))
    frame_width: int = int(os.getenv("FRAME_WIDTH", "640"))
    frame_height: int = int(os.getenv("FRAME_HEIGHT", "480"))
    target_fps: int = int(os.getenv("TARGET_FPS", "15"))
    min_detection_confidence: float = float(os.getenv("DETECTION_CONFIDENCE_THRESHOLD", "0.5"))
    track_iou_threshold: float = float(os.getenv("TRACK_IOU_THRESHOLD", "0.3"))
    track_max_inactive_frames: int = int(os.getenv("TRACK_MAX_INACTIVE_FRAMES", "15"))


@dataclass(frozen=True)
class AudioConfig:
    """Audio capture and signal processing parameters."""

    sample_rate: int = int(os.getenv("AUDIO_SAMPLE_RATE", "16000"))
    window_duration_seconds: float = float(os.getenv("AUDIO_WINDOW_SECONDS", "5.0"))
    rms_activity_threshold: float = float(os.getenv("AUDIO_RMS_THRESHOLD", "0.015"))
    zcr_activity_threshold: float = float(os.getenv("AUDIO_ZCR_THRESHOLD", "0.05"))


@dataclass(frozen=True)
class AttendanceConfig:
    """Presence-based attendance evaluation rules."""

    min_presence_percentage: float = float(os.getenv("MIN_PRESENCE_PERCENTAGE", "75.0"))
    min_presence_duration_seconds: float = float(os.getenv("MIN_PRESENCE_DURATION_SECONDS", "300.0"))
    max_disappearance_seconds: float = float(os.getenv("MAX_DISAPPEARANCE_SECONDS", "30.0"))


@dataclass(frozen=True)
class FusionConfig:
    """Multimodal fusion scoring weights and parameters."""

    window_duration_seconds: float = float(os.getenv("FUSION_WINDOW_SECONDS", "5.0"))
    weight_visual_presence: float = float(os.getenv("WEIGHT_VISUAL_PRESENCE", "0.4"))
    weight_audio_activity: float = float(os.getenv("WEIGHT_AUDIO_ACTIVITY", "0.4"))
    weight_movement: float = float(os.getenv("WEIGHT_MOVEMENT", "0.2"))


@dataclass(frozen=True)
class AgentConfig:
    """Pedagogical recommendation rule thresholds and cooldown parameters."""

    sustained_low_threshold: float = float(os.getenv("AGENT_LOW_ACTIVITY_THRESHOLD", "30.0"))
    sustained_low_windows: int = int(os.getenv("AGENT_LOW_ACTIVITY_WINDOWS", "3"))
    high_noise_threshold: float = float(os.getenv("AGENT_HIGH_NOISE_THRESHOLD", "0.08"))
    high_interaction_audio: float = float(os.getenv("AGENT_HIGH_AUDIO_THRESHOLD", "60.0"))
    high_interaction_movement: float = float(os.getenv("AGENT_HIGH_MOVEMENT_THRESHOLD", "40.0"))
    reduced_visual_threshold: float = float(os.getenv("AGENT_REDUCED_VISUAL_THRESHOLD", "50.0"))
    prolonged_silence_windows: int = int(os.getenv("AGENT_SILENCE_WINDOWS", "4"))
    cooldown_seconds: float = float(os.getenv("AGENT_COOLDOWN_SECONDS", "30.0"))


@dataclass(frozen=True)
class AppConfig:
    """Aggregated application configuration."""

    env: str = os.getenv("APP_ENV", "development")
    log_level: str = os.getenv("LOG_LEVEL", "INFO")
    storage: StorageConfig = field(default_factory=StorageConfig)
    vision: VisionConfig = field(default_factory=VisionConfig)
    audio: AudioConfig = field(default_factory=AudioConfig)
    attendance: AttendanceConfig = field(default_factory=AttendanceConfig)
    fusion: FusionConfig = field(default_factory=FusionConfig)
    agent: AgentConfig = field(default_factory=AgentConfig)

    def ensure_directories(self) -> None:
        """Create required runtime data directories if they do not exist."""
        self.storage.data_dir.mkdir(parents=True, exist_ok=True)
        self.storage.models_dir.mkdir(parents=True, exist_ok=True)
        self.storage.sessions_dir.mkdir(parents=True, exist_ok=True)
        self.storage.samples_dir.mkdir(parents=True, exist_ok=True)
        # Ensure database parent directory exists
        self.storage.database_path.parent.mkdir(parents=True, exist_ok=True)


_config_instance: Optional[AppConfig] = None


def get_config(force_reload: bool = False) -> AppConfig:
    """Retrieve the singleton AppConfig instance.

    Args:
        force_reload: If True, re-instantiate configuration from current environment.

    Returns:
        AppConfig: The active application configuration instance.
    """
    global _config_instance
    if _config_instance is None or force_reload:
        _config_instance = AppConfig()
        _config_instance.ensure_directories()
    return _config_instance

