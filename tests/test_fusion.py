"""Unit tests for Multimodal Signal Fusion and Engagement Scoring.

Validates time synchronization, missing modality handling, re-weighting,
explainable component contributions, score normalization, and SQLite persistence.
"""

from pathlib import Path
import pytest

from attendance_agent.fusion.scoring import (
    MultimodalFusionEngine,
    SynchronizedWindow,
)
from attendance_agent.fusion.synchronizer import MultimodalSynchronizer
from attendance_agent.storage.database import DatabaseManager


@pytest.fixture
def sample_tracking_events() -> list[dict]:
    """Provide synthetic tracking events across 3 frames."""
    return [
        # Window 1: 0.0s to 5.0s (Student 1 & 2 active)
        {"session_id": "sess-01", "track_id": 1, "timestamp": 1.0, "bbox_x": 100.0, "bbox_y": 100.0, "bbox_w": 80.0, "bbox_h": 80.0, "confidence": 0.95},
        {"session_id": "sess-01", "track_id": 2, "timestamp": 1.0, "bbox_x": 300.0, "bbox_y": 100.0, "bbox_w": 80.0, "bbox_h": 80.0, "confidence": 0.90},
        {"session_id": "sess-01", "track_id": 1, "timestamp": 3.0, "bbox_x": 115.0, "bbox_y": 105.0, "bbox_w": 80.0, "bbox_h": 80.0, "confidence": 0.94},
        # Window 2: 5.0s to 10.0s (Student 1 only)
        {"session_id": "sess-01", "track_id": 1, "timestamp": 6.5, "bbox_x": 120.0, "bbox_y": 105.0, "bbox_w": 80.0, "bbox_h": 80.0, "confidence": 0.91},
    ]


@pytest.fixture
def sample_audio_metrics() -> list[dict]:
    """Provide synthetic audio metrics across two 5-second windows."""
    return [
        # Window 1
        {"session_id": "sess-01", "window_start": 0.0, "window_end": 5.0, "rms_energy": 0.045, "zero_crossing_rate": 0.06, "activity_detected": True, "noise_level": "moderate"},
        # Window 2
        {"session_id": "sess-01", "window_start": 5.0, "window_end": 10.0, "rms_energy": 0.003, "zero_crossing_rate": 0.01, "activity_detected": False, "noise_level": "low"},
    ]


def test_multimodal_synchronization_matching_timestamps(
    sample_tracking_events: list[dict],
    sample_audio_metrics: list[dict],
) -> None:
    """Verify synchronizer aligns video events and audio metrics into exact time windows."""
    synchronizer = MultimodalSynchronizer(window_duration_seconds=5.0)
    windows = synchronizer.synchronize_streams(
        session_duration_seconds=10.0,
        tracking_events=sample_tracking_events,
        audio_metrics=sample_audio_metrics,
    )

    assert len(windows) == 2
    w1, w2 = windows[0], windows[1]

    # Window 1 checks
    assert w1.window_start == 0.0
    assert w1.window_end == 5.0
    assert w1.has_visual_data is True
    assert w1.has_audio_data is True
    assert w1.face_count > 0.0
    assert w1.audio_activity_ratio == 1.0  # Active in window 1
    assert w1.noise_level == "moderate"

    # Window 2 checks
    assert w2.window_start == 5.0
    assert w2.window_end == 10.0
    assert w2.has_visual_data is True
    assert w2.has_audio_data is True
    assert w2.audio_activity_ratio == 0.0  # Quiet in window 2
    assert w2.noise_level == "low"


def test_missing_audio_modality_video_only(sample_tracking_events: list[dict]) -> None:
    """Verify video-only input is re-weighted cleanly with penalized confidence."""
    synchronizer = MultimodalSynchronizer(window_duration_seconds=5.0)
    windows = synchronizer.synchronize_streams(
        session_duration_seconds=5.0,
        tracking_events=sample_tracking_events,
        audio_metrics=[],  # No audio available
    )

    assert len(windows) == 1
    w = windows[0]
    assert w.has_visual_data is True
    assert w.has_audio_data is False

    engine = MultimodalFusionEngine()
    score = engine.score_window(w)

    assert score.modalities_present == ["vision"]
    assert score.audio_contribution == 0.0
    assert score.confidence < 0.70  # Discounted confidence
    assert score.activity_score > 0.0


def test_missing_video_modality_audio_only(sample_audio_metrics: list[dict]) -> None:
    """Verify audio-only input is re-weighted cleanly with penalized confidence."""
    synchronizer = MultimodalSynchronizer(window_duration_seconds=5.0)
    windows = synchronizer.synchronize_streams(
        session_duration_seconds=5.0,
        tracking_events=[],  # No camera available
        audio_metrics=sample_audio_metrics,
    )

    assert len(windows) == 1
    w = windows[0]
    assert w.has_visual_data is False
    assert w.has_audio_data is True

    engine = MultimodalFusionEngine()
    score = engine.score_window(w)

    assert score.modalities_present == ["audio"]
    assert score.visual_contribution == 0.0
    assert score.movement_contribution == 0.0
    assert score.confidence < 0.65  # Discounted confidence
    assert score.activity_score > 0.0


def test_insufficient_data_empty_window() -> None:
    """Verify windows with neither visual nor audio metrics report Insufficient data."""
    empty_win = SynchronizedWindow(
        window_index=1,
        window_start=0.0,
        window_end=5.0,
        has_visual_data=False,
        has_audio_data=False,
    )

    engine = MultimodalFusionEngine()
    score = engine.score_window(empty_win)

    assert score.activity_score == 0.0
    assert score.confidence == 0.0
    assert score.label == "Insufficient data"
    assert len(score.modalities_present) == 0


def test_explainable_contributions_sum_to_total() -> None:
    """Verify constituent signal contributions sum to final aggregate activity score."""
    win = SynchronizedWindow(
        window_index=1,
        window_start=0.0,
        window_end=5.0,
        has_visual_data=True,
        has_audio_data=True,
        visual_presence_ratio=0.85,
        movement_level=0.30,
        audio_activity_ratio=0.75,
        mean_detection_confidence=0.92,
    )

    engine = MultimodalFusionEngine(
        weight_visual_presence=0.40,
        weight_audio_activity=0.40,
        weight_movement=0.20,
    )
    score = engine.score_window(win)

    # Check that contributions sum to score
    total_contrib = (
        score.visual_contribution
        + score.audio_contribution
        + score.movement_contribution
    )
    assert pytest.approx(total_contrib, 0.01) == score.activity_score
    assert 0.0 <= score.activity_score <= 100.0


def test_custom_weight_configuration() -> None:
    """Verify custom scoring weights ignore unweighted modalities."""
    win = SynchronizedWindow(
        window_index=1,
        window_start=0.0,
        window_end=5.0,
        has_visual_data=True,
        has_audio_data=True,
        visual_presence_ratio=0.90,
        movement_level=0.0,
        audio_activity_ratio=0.90,
    )

    # Audio weight explicitly set to 0.0
    engine_no_audio = MultimodalFusionEngine(
        weight_visual_presence=0.80,
        weight_audio_activity=0.0,
        weight_movement=0.20,
    )
    score = engine_no_audio.score_window(win)

    assert score.audio_contribution == 0.0
    assert score.visual_contribution == pytest.approx(score.activity_score, rel=1e-2)


def test_score_classification_labels() -> None:
    """Verify transparent classification boundaries for high, moderate, and low activity."""
    engine = MultimodalFusionEngine(
        low_activity_threshold=30.0,
        high_activity_threshold=70.0,
        min_confidence_threshold=0.30,
    )

    assert engine.classify_label(15.0, 0.8) == "Low observed activity"
    assert engine.classify_label(50.0, 0.8) == "Moderate observed activity"
    assert engine.classify_label(85.0, 0.8) == "High observed activity"
    assert engine.classify_label(85.0, 0.2) == "Insufficient data"  # confidence below 0.30


def test_fusion_scores_database_persistence(tmp_path: Path) -> None:
    """Verify batch insertion and query of fusion scores in SQLite database."""
    db = DatabaseManager(tmp_path / "fusion_test.db")
    session_id = "sess-fusion-01"
    db.create_session(session_id, "Multimodal Lecture")

    scores_to_insert = [
        {
            "window_start": 0.0,
            "window_end": 5.0,
            "visual_presence": 0.90,
            "audio_activity": 0.70,
            "movement_level": 0.20,
            "activity_score": 75.0,
            "confidence": 0.95,
            "label": "High observed activity",
        },
        {
            "window_start": 5.0,
            "window_end": 10.0,
            "visual_presence": 0.80,
            "audio_activity": 0.10,
            "movement_level": 0.10,
            "activity_score": 38.0,
            "confidence": 0.90,
            "label": "Moderate observed activity",
        },
    ]

    inserted = db.record_fusion_scores_batch(session_id, scores_to_insert)
    assert inserted == 2

    retrieved = db.get_fusion_scores(session_id)
    assert len(retrieved) == 2
    assert retrieved[0]["activity_score"] == 75.0
    assert retrieved[0]["label"] == "High observed activity"
    assert retrieved[1]["activity_score"] == 38.0
    assert retrieved[1]["label"] == "Moderate observed activity"

