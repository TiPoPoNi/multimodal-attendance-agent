"""Unit tests for Rule-Based Agent and Teaching Recommendation Engine."""

from pathlib import Path
from typing import List, Optional
import pytest

from attendance_agent.agent.recommendations import (
    HighInteractionPeriodRule,
    IncreasingNoiseRule,
    MissingModalityRule,
    ProlongedSilenceRule,
    ReducedVisualPresenceRule,
    SustainedLowActivityRule,
    TeachingRecommendationEngine,
)
from attendance_agent.config import AgentConfig
from attendance_agent.fusion.scoring import WindowActivityScore
from attendance_agent.storage.database import DatabaseManager, init_db


def make_window_score(
    window_start: float,
    window_end: float,
    visual_presence: float = 80.0,
    audio_activity: float = 30.0,
    movement_level: float = 20.0,
    activity_score: float = 45.0,
    confidence: float = 1.0,
    label: str = "Moderate activity",
    window_index: int = 0,
    modalities_present: Optional[List[str]] = None,
) -> WindowActivityScore:
    """Helper to synthesize a WindowActivityScore."""
    if modalities_present is None:
        modalities_present = ["vision", "audio"]
    return WindowActivityScore(
        window_index=window_index,
        window_start=window_start,
        window_end=window_end,
        visual_presence=visual_presence,
        audio_activity=audio_activity,
        movement_level=movement_level,
        activity_score=activity_score,
        confidence=confidence,
        label=label,
        modalities_present=modalities_present,
        visual_contribution=round(visual_presence * 0.4, 2),
        audio_contribution=round(audio_activity * 0.4, 2),
        movement_contribution=round(movement_level * 0.2, 2),
    )



def test_sustained_low_activity_rule() -> None:
    """Verify rule triggers only after consecutive low activity intervals."""
    rule = SustainedLowActivityRule(threshold=30.0, min_windows=3)

    w1 = make_window_score(0.0, 5.0, activity_score=20.0)
    w2 = make_window_score(5.0, 10.0, activity_score=25.0)
    w3 = make_window_score(10.0, 15.0, activity_score=22.0)

    # 1 window: not triggered
    assert rule.evaluate("sess-1", w1, []) is None

    # 2 windows: not triggered
    assert rule.evaluate("sess-1", w2, [w1]) is None

    # 3 windows: triggers
    for i, w in enumerate([w1, w2, w3]):
        aud = w.audio_activity
        vis = w.visual_presence
        print(f"w{i+1}: aud={aud}, vis={vis}")
    rec = rule.evaluate("sess-1", w3, [w1, w2])
    print("rec:", rec)
    assert rec is not None
    assert rec.condition == "Sustained Low Classroom Activity"
    assert rec.rule_source == "sustained_low_activity"
    assert rec.priority == "Medium"
    assert rec.supporting_metrics["consecutive_windows"] == 3
    assert "brief formative question" in rec.suggestion


def test_sustained_low_activity_urgent_priority() -> None:
    """Verify urgent priority assignment when low activity persists for 5+ windows."""
    rule = SustainedLowActivityRule(threshold=30.0, min_windows=3)
    windows = [make_window_score(i * 5.0, (i + 1) * 5.0, activity_score=15.0) for i in range(5)]

    rec = rule.evaluate("sess-1", windows[4], windows[:4])
    assert rec is not None
    assert rec.priority == "Urgent"


def test_increasing_noise_rule() -> None:
    """Verify rule triggers on elevated audio activity."""
    rule = IncreasingNoiseRule()

    normal = make_window_score(0.0, 5.0, audio_activity=40.0)
    assert rule.evaluate("sess-1", normal, []) is None

    high_noise = make_window_score(5.0, 10.0, audio_activity=85.0)
    rec = rule.evaluate("sess-1", high_noise, [normal])
    assert rec is not None
    assert rec.condition == "Elevated Classroom Ambient Noise"
    assert rec.priority == "Warning"
    assert "collaborative tasks" in rec.suggestion


def test_reduced_visual_presence_rule() -> None:
    """Verify rule triggers when presence drops below threshold after prior presence."""
    rule = ReducedVisualPresenceRule(threshold=50.0)

    w1 = make_window_score(0.0, 5.0, visual_presence=80.0)
    w2 = make_window_score(5.0, 10.0, visual_presence=30.0)

    # Dropped presence
    rec = rule.evaluate("sess-1", w2, [w1])
    assert rec is not None
    assert rec.condition == "Reduced Visual Participant Presence"
    assert rec.priority == "Warning"
    assert "camera positioning" in rec.suggestion


def test_high_interaction_period_rule() -> None:
    """Verify detection of concurrent high audio activity and physical motion."""
    rule = HighInteractionPeriodRule(min_audio=60.0, min_movement=40.0)

    # Only audio high
    w1 = make_window_score(0.0, 5.0, audio_activity=70.0, movement_level=20.0)
    assert rule.evaluate("sess-1", w1, []) is None

    # Both audio and movement high
    w2 = make_window_score(5.0, 10.0, audio_activity=65.0, movement_level=50.0)
    rec = rule.evaluate("sess-1", w2, [w1])
    assert rec is not None
    assert rec.condition == "High Multimodal Interaction Interval"
    assert rec.priority == "Info"
    assert "collaborative energy" in rec.suggestion


def test_missing_modality_rule() -> None:
    """Verify rule warns when confidence indicates missing sensor data."""
    rule = MissingModalityRule(max_confidence=0.6)

    # Normal confidence
    w_normal = make_window_score(0.0, 5.0, confidence=1.0)
    assert rule.evaluate("sess-1", w_normal, []) is None

    # Low confidence due to video missing
    w_missing_video = make_window_score(5.0, 10.0, visual_presence=0.0, confidence=0.5)
    rec = rule.evaluate("sess-1", w_missing_video, [w_normal])
    assert rec is not None
    assert rec.condition == "Partial Modality Sensor Observation"
    assert rec.confidence == "Low"
    assert "video" in rec.supporting_metrics["missing_streams"]


def test_prolonged_silence_rule() -> None:
    """Verify rule triggers when audio is silent while visual presence continues."""
    rule = ProlongedSilenceRule(min_windows=3, max_audio=5.0, min_visual=50.0)

    w1 = make_window_score(0.0, 5.0, visual_presence=0.75, audio_activity=0.02)
    w2 = make_window_score(5.0, 10.0, visual_presence=0.70, audio_activity=0.01)
    w3 = make_window_score(10.0, 15.0, visual_presence=0.80, audio_activity=0.0)

    rec = rule.evaluate("sess-1", w3, [w1, w2])
    assert rec is not None
    assert rec.condition == "Prolonged Classroom Silence"
    assert rec.priority == "Info"
    assert "independent reading" in rec.suggestion


def test_recommendation_engine_cooldown_and_deduplication() -> None:
    """Verify engine suppresses duplicate recommendations within cooldown window."""
    config = AgentConfig(
        sustained_low_threshold=30.0,
        sustained_low_windows=2,
        cooldown_seconds=30.0,
    )
    engine = TeachingRecommendationEngine(config=config)

    w1 = make_window_score(0.0, 5.0, activity_score=20.0)
    w2 = make_window_score(5.0, 10.0, activity_score=20.0)
    w3 = make_window_score(10.0, 15.0, activity_score=20.0)

    # Window 2 triggers sustained low activity
    recs_w2 = engine.evaluate_window("sess-1", w2, [w1])
    assert len(recs_w2) >= 1
    assert recs_w2[0].rule_source == "sustained_low_activity"

    # Window 3 occurs at 15.0s (< 30s cooldown), so it must be suppressed
    recs_w3 = engine.evaluate_window("sess-1", w3, [w1, w2])
    low_activity_recs = [r for r in recs_w3 if r.rule_source == "sustained_low_activity"]
    assert len(low_activity_recs) == 0

    # Window 8 at 45.0s (> 30s cooldown), so it can trigger again
    w8 = make_window_score(40.0, 45.0, activity_score=20.0)
    recs_w8 = engine.evaluate_window("sess-1", w8, [w1, w2, w3])
    low_activity_recs_w8 = [r for r in recs_w8 if r.rule_source == "sustained_low_activity"]
    assert len(low_activity_recs_w8) == 1


def test_recommendation_prioritization() -> None:
    """Verify that returned recommendations are sorted by priority: Urgent > Warning > Info."""
    engine = TeachingRecommendationEngine(config=AgentConfig(cooldown_seconds=0.0))

    # Window triggering both warning (noise >= 75%) and info (interaction)
    w_prior = make_window_score(0.0, 5.0, visual_presence=80.0, audio_activity=20.0)
    w_multi = make_window_score(
        5.0, 10.0, visual_presence=80.0, audio_activity=85.0, movement_level=60.0
    )

    recs = engine.evaluate_window("sess-1", w_multi, [w_prior])
    priorities = [r.priority for r in recs]
    # Warning should precede Info
    if "Warning" in priorities and "Info" in priorities:
        assert priorities.index("Warning") < priorities.index("Info")


def test_recommendations_database_persistence(tmp_path: Path) -> None:
    """Verify batch inserting recommendations into SQLite and querying them back."""
    db_path = tmp_path / "test_recs.db"
    init_db(db_path)
    db = DatabaseManager(db_path)
    session_id = "session-recs-001"
    db.create_session(session_id, "Pedagogical Agent Test")

    engine = TeachingRecommendationEngine(config=AgentConfig(sustained_low_windows=2))
    w1 = make_window_score(0.0, 5.0, activity_score=15.0)
    w2 = make_window_score(5.0, 10.0, activity_score=15.0)

    recs = engine.evaluate_window(session_id, w2, [w1])
    assert len(recs) > 0

    # Persist batch
    recs_dicts = [r.to_dict() for r in recs]
    inserted = db.record_recommendations_batch(session_id, recs_dicts)
    assert inserted == len(recs)

    # Retrieve from database
    stored = db.get_recommendations(session_id)
    assert len(stored) == len(recs)
    assert stored[0]["session_id"] == session_id
    assert stored[0]["rule_source"] == recs[0].rule_source
    assert stored[0]["priority"] == recs[0].priority
    assert isinstance(stored[0]["supporting_metrics"], dict)
