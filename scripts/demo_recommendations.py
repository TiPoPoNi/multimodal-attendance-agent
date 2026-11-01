"""Demonstration script for Rule-Based Agent and Pedagogical Recommendation Engine.

Simulates a representative classroom session timeline with shifting dynamics:
1. Normal lecture baseline
2. Sustained low activity interval
3. High collaboration / interactive groupwork
4. Ambient noise spike
5. Prolonged quiet study / exam period
6. Degraded camera coverage

Evaluates rules, enforces cooldowns, sorts recommendations by priority,
and persists them into the SQLite database.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from attendance_agent.agent.recommendations import TeachingRecommendationEngine
from attendance_agent.config import AgentConfig, get_config
from attendance_agent.fusion.scoring import WindowActivityScore
from attendance_agent.storage.database import DatabaseManager, init_db


def build_scenario_score(
    idx: int,
    start: float,
    end: float,
    vis: float,
    aud: float,
    mov: float,
    score: float,
    conf: float = 1.0,
    label: str = "Moderate observed activity",
) -> WindowActivityScore:
    """Helper to synthesize a timeline window score."""
    return WindowActivityScore(
        window_index=idx,
        window_start=start,
        window_end=end,
        visual_presence=vis,
        audio_activity=aud,
        movement_level=mov,
        activity_score=score,
        confidence=conf,
        label=label,
        modalities_present=["vision", "audio"] if conf > 0.6 else ["audio"],
        visual_contribution=round(vis * 40.0, 1),
        audio_contribution=round(aud * 40.0, 1),
        movement_contribution=round(mov * 20.0, 1),
    )


def run_demo() -> None:
    print("=" * 78)
    print("  Multimodal Attendance & Engagement Agent - Recommendation Engine Demo")
    print("=" * 78)

    cfg = get_config()
    cfg.ensure_directories()
    init_db(cfg.storage.database_path)
    db = DatabaseManager(cfg.storage.database_path)

    session_id = "demo-pedagogical-session-001"
    db.create_session(session_id, "Pedagogical Agent Demonstration")

    agent_cfg = AgentConfig(
        sustained_low_threshold=30.0,
        sustained_low_windows=3,
        high_interaction_audio=60.0,
        high_interaction_movement=40.0,
        reduced_visual_threshold=50.0,
        prolonged_silence_windows=3,
        cooldown_seconds=15.0,
    )
    engine = TeachingRecommendationEngine(config=agent_cfg)

    # 10 Windows (50 seconds total, 5s windows)
    timeline = [
        # W0-W1: Normal lecture baseline
        build_scenario_score(0, 0.0, 5.0, 0.85, 0.40, 0.20, 54.0),
        build_scenario_score(1, 5.0, 10.0, 0.80, 0.35, 0.15, 49.0),
        # W2-W4: Sustained low activity (scores < 30 for 3 consecutive windows)
        build_scenario_score(2, 10.0, 15.0, 0.70, 0.10, 0.05, 25.0, label="Low observed activity"),
        build_scenario_score(3, 15.0, 20.0, 0.65, 0.08, 0.04, 22.0, label="Low observed activity"),
        build_scenario_score(4, 20.0, 25.0, 0.70, 0.05, 0.05, 21.0, label="Low observed activity"),
        # W5: High interaction interval (audio 70%, movement 50%)
        build_scenario_score(5, 25.0, 30.0, 0.85, 0.70, 0.50, 72.0, label="High observed activity"),
        # W6: Ambient noise spike (audio 85%)
        build_scenario_score(6, 30.0, 35.0, 0.85, 0.85, 0.30, 74.0, label="High observed activity"),
        # W7-W9: Prolonged silence during quiet independent work
        build_scenario_score(7, 35.0, 40.0, 0.80, 0.02, 0.10, 36.0),
        build_scenario_score(8, 40.0, 45.0, 0.80, 0.01, 0.08, 35.0),
        build_scenario_score(9, 45.0, 50.0, 0.80, 0.00, 0.05, 34.0),
    ]

    print(f"\nEvaluating {len(timeline)} simulated 5-second session windows...\n")

    all_recommendations = engine.evaluate_timeline(session_id, timeline)

    print(f"Generated {len(all_recommendations)} context-aware teaching recommendations:\n")
    print(f"{'Time':<8} | {'Priority':<9} | {'Confidence':<10} | {'Condition':<35} | {'Rule Source'}")
    print("-" * 95)
    for r in all_recommendations:
        time_str = f"{r.timestamp:.1f}s"
        print(f"{time_str:<8} | {r.priority:<9} | {r.confidence:<10} | {r.condition:<35} | {r.rule_source}")
        print(f"  -> Explanation : {r.explanation}")
        print(f"  -> Suggestion  : {r.suggestion}\n")

    # Persist in SQLite
    rec_dicts = [r.to_dict() for r in all_recommendations]
    inserted = db.record_recommendations_batch(session_id, rec_dicts)
    print(f"Successfully recorded {inserted} recommendations into SQLite database.")

    # Retrieve verify
    retrieved = db.get_recommendations(session_id)
    print(f"Verified database query: retrieved {len(retrieved)} persisted recommendations for session {session_id}.\n")
    print("=" * 78)
    print("Phase 6 Demonstration Complete.")
    print("=" * 78)


if __name__ == "__main__":
    run_demo()
