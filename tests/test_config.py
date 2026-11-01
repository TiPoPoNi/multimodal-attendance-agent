"""Unit tests for application configuration and directory management."""

from pathlib import Path

import pytest

from attendance_agent.config import (
    AppConfig,
    AttendanceConfig,
    AudioConfig,
    FusionConfig,
    StorageConfig,
    VisionConfig,
    get_config,
)


def test_storage_config_paths() -> None:
    """Verify that all paths are pathlib.Path instances and properly configured."""
    storage = StorageConfig()
    assert isinstance(storage.root_dir, Path)
    assert isinstance(storage.data_dir, Path)
    assert isinstance(storage.models_dir, Path)
    assert isinstance(storage.sessions_dir, Path)
    assert isinstance(storage.samples_dir, Path)
    assert isinstance(storage.database_path, Path)
    assert isinstance(storage.face_model_path, Path)

    # Check hierarchical consistency
    assert storage.models_dir.parent == storage.data_dir
    assert storage.sessions_dir.parent == storage.data_dir
    assert storage.samples_dir.parent == storage.data_dir


def test_default_config_values() -> None:
    """Verify sensible default parameters across all sub-configurations."""
    vision = VisionConfig()
    assert vision.frame_width == 640
    assert vision.frame_height == 480
    assert vision.target_fps > 0
    assert 0.0 < vision.min_detection_confidence <= 1.0

    audio = AudioConfig()
    assert audio.sample_rate == 16000
    assert audio.window_duration_seconds > 0.0
    assert audio.rms_activity_threshold > 0.0

    attendance = AttendanceConfig()
    assert 0.0 <= attendance.min_presence_percentage <= 100.0
    assert attendance.min_presence_duration_seconds > 0.0

    fusion = FusionConfig()
    total_weight = (
        fusion.weight_visual_presence
        + fusion.weight_audio_activity
        + fusion.weight_movement
    )
    assert pytest.approx(total_weight, 0.01) == 1.0


def test_ensure_directories(tmp_path: Path) -> None:
    """Verify that ensure_directories correctly creates required paths."""
    custom_storage = StorageConfig(
        root_dir=tmp_path,
        data_dir=tmp_path / "data",
        models_dir=tmp_path / "data" / "models",
        sessions_dir=tmp_path / "data" / "sessions",
        samples_dir=tmp_path / "data" / "samples",
        database_path=tmp_path / "data" / "sessions" / "test.db",
        face_model_path=tmp_path / "data" / "models" / "model.task",
    )
    app_config = AppConfig(storage=custom_storage)
    app_config.ensure_directories()

    assert custom_storage.data_dir.is_dir()
    assert custom_storage.models_dir.is_dir()
    assert custom_storage.sessions_dir.is_dir()
    assert custom_storage.samples_dir.is_dir()


def test_config_singleton() -> None:
    """Verify get_config returns a valid singleton AppConfig instance."""
    cfg1 = get_config()
    cfg2 = get_config()
    assert cfg1 is cfg2
    assert cfg1.env in ["development", "testing", "production"]

