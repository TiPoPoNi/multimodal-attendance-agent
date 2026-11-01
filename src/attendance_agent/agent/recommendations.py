"""Rule-based pedagogical recommendation engine.

Analyzes multimodal activity timelines and generates contextual,
transparent, non-judgmental teaching suggestions for classroom instructors.
Operates completely offline without external LLM APIs.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
import uuid

from attendance_agent.config import AgentConfig, get_config
from attendance_agent.fusion.scoring import WindowActivityScore
from attendance_agent.models import Recommendation
from attendance_agent.utils.logger import get_logger

logger = get_logger("attendance_agent.agent")

# Priority ranking for sorting recommendations (lower number = higher priority)
PRIORITY_RANK = {
    "Urgent": 1,
    "Warning": 2,
    "Medium": 3,
    "Info": 4,
}


def _to_percentage(value: float) -> float:
    """Normalize input value to 0-100 scale regardless of whether passed as ratio [0, 1] or percentage."""
    if 0.0 <= value <= 1.0 and value > 0.0:
        return value * 100.0
    return value



class BaseRecommendationRule(ABC):
    """Abstract base class for all pedagogical recommendation rules."""

    rule_id: str = "base_rule"
    name: str = "Base Rule"
    description: str = "Base rule description"

    @abstractmethod
    def evaluate(
        self,
        session_id: str,
        current_window: WindowActivityScore,
        history: List[WindowActivityScore],
    ) -> Optional[Recommendation]:
        """Evaluate rule conditions against the current window and historical metrics.

        Args:
            session_id: Unique session identifier.
            current_window: Latest window activity score and constituent signals.
            history: Chronological list of previous window scores in current session.

        Returns:
            Optional[Recommendation]: Generated recommendation if triggered, else None.
        """
        pass


class SustainedLowActivityRule(BaseRecommendationRule):
    """Rule triggering when observed activity proxy remains low for consecutive windows."""

    rule_id = "sustained_low_activity"
    name = "Sustained Low Classroom Activity"
    description = "Detects prolonged periods of low multimodal activity proxy."

    def __init__(self, threshold: float = 30.0, min_windows: int = 3) -> None:
        self.threshold = threshold
        self.min_windows = min_windows

    def evaluate(
        self,
        session_id: str,
        current_window: WindowActivityScore,
        history: List[WindowActivityScore],
    ) -> Optional[Recommendation]:
        # Consider the sequence of recent windows including the current one
        relevant_windows = history + [current_window]
        if len(relevant_windows) < self.min_windows:
            return None

        recent = relevant_windows[-self.min_windows :]
        if all(w.activity_score < self.threshold for w in recent):
            avg_activity = sum(w.activity_score for w in recent) / len(recent)
            duration = sum(w.window_end - w.window_start for w in recent)

            # Assign higher priority if low activity persists even longer
            extended_count = sum(
                1 for w in reversed(relevant_windows) if w.activity_score < self.threshold
            )
            priority = "Urgent" if extended_count >= 5 else "Medium"
            confidence = "High" if all(w.confidence >= 0.7 for w in recent) else "Medium"

            return Recommendation(
                id=f"rec-{session_id}-{self.rule_id}-{int(current_window.window_end * 1000)}",
                session_id=session_id,
                timestamp=current_window.window_end,
                condition="Sustained Low Classroom Activity",
                explanation=(
                    f"Observed multimodal activity has remained below {self.threshold:.1f} "
                    f"for {self.min_windows} consecutive intervals ({duration:.1f}s), averaging {avg_activity:.1f}."
                ),
                suggestion=(
                    "Consider introducing a brief formative question, a quick think-pair-share prompt, "
                    "or an interactive comprehension check to re-engage the room."
                ),
                confidence=confidence,
                rule_source=self.rule_id,
                priority=priority,
                supporting_metrics={
                    "consecutive_windows": self.min_windows,
                    "avg_activity": round(avg_activity, 1),
                    "threshold": self.threshold,
                    "duration_seconds": round(duration, 1),
                },
            )
        return None


class IncreasingNoiseRule(BaseRecommendationRule):
    """Rule triggering when ambient noise or acoustic energy is significantly elevated."""

    rule_id = "increasing_noise"
    name = "Elevated Classroom Ambient Noise"
    description = "Detects elevated or spiking ambient acoustic energy."

    def __init__(self, high_noise_threshold: float = 0.08) -> None:
        self.high_noise_threshold = high_noise_threshold

    def evaluate(
        self,
        session_id: str,
        current_window: WindowActivityScore,
        history: List[WindowActivityScore],
    ) -> Optional[Recommendation]:
        # High audio activity combined with high energy
        aud_pct = _to_percentage(current_window.audio_activity)
        if aud_pct >= 75.0:
            return Recommendation(
                id=f"rec-{session_id}-{self.rule_id}-{int(current_window.window_end * 1000)}",
                session_id=session_id,
                timestamp=current_window.window_end,
                condition="Elevated Classroom Ambient Noise",
                explanation=(
                    f"Classroom acoustic activity is significantly elevated ({aud_pct:.1f}%). "
                    "High ambient volume or vocal energy observed."
                ),
                suggestion=(
                    "Observe whether students are actively discussing collaborative tasks or if "
                    "ambient classroom disruption warrants drawing attention back to the front."
                ),
                confidence="High",
                rule_source=self.rule_id,
                priority="Warning",
                supporting_metrics={
                    "audio_activity": round(aud_pct, 1),
                    "window_start": current_window.window_start,
                    "window_end": current_window.window_end,
                },
            )
        return None


class ReducedVisualPresenceRule(BaseRecommendationRule):
    """Rule triggering when visual face tracking coverage decreases notably."""

    rule_id = "reduced_visual_presence"
    name = "Reduced Visual Participant Presence"
    description = "Detects drops in detected participant presence coverage."

    def __init__(self, threshold: float = 50.0) -> None:
        self.threshold = threshold

    def evaluate(
        self,
        session_id: str,
        current_window: WindowActivityScore,
        history: List[WindowActivityScore],
    ) -> Optional[Recommendation]:
        # Trigger only if tracking was active in recent history to avoid triggering before session starts
        has_prior_presence = any(_to_percentage(w.visual_presence) > 0.0 for w in history)
        vis_pct = _to_percentage(current_window.visual_presence)
        if has_prior_presence and vis_pct < self.threshold:
            return Recommendation(
                id=f"rec-{session_id}-{self.rule_id}-{int(current_window.window_end * 1000)}",
                session_id=session_id,
                timestamp=current_window.window_end,
                condition="Reduced Visual Participant Presence",
                explanation=(
                    f"Visual participant tracking coverage dropped to {vis_pct:.1f}%, "
                    f"which is below the expected visibility threshold of {self.threshold:.1f}%."
                ),
                suggestion=(
                    "Check camera positioning, room lighting, or line of sight before drawing conclusions "
                    "regarding student absence or disengagement."
                ),
                confidence="Medium",
                rule_source=self.rule_id,
                priority="Warning",
                supporting_metrics={
                    "visual_presence": round(vis_pct, 1),
                    "threshold": self.threshold,
                },
            )
        return None


class HighInteractionPeriodRule(BaseRecommendationRule):
    """Rule identifying periods where acoustic activity and physical movement rise together."""

    rule_id = "high_interaction_activity"
    name = "High Multimodal Interaction Interval"
    description = "Detects concurrent elevation of audio activity and visual movement."

    def __init__(self, min_audio: float = 60.0, min_movement: float = 40.0) -> None:
        self.min_audio = min_audio
        self.min_movement = min_movement

    def evaluate(
        self,
        session_id: str,
        current_window: WindowActivityScore,
        history: List[WindowActivityScore],
    ) -> Optional[Recommendation]:
        aud_pct = _to_percentage(current_window.audio_activity)
        mov_pct = _to_percentage(current_window.movement_level)
        if aud_pct >= self.min_audio and mov_pct >= self.min_movement:
            return Recommendation(
                id=f"rec-{session_id}-{self.rule_id}-{int(current_window.window_end * 1000)}",
                session_id=session_id,
                timestamp=current_window.window_end,
                condition="High Multimodal Interaction Interval",
                explanation=(
                    f"Concurrent elevation of acoustic activity ({aud_pct:.1f}%) and "
                    f"physical motion ({mov_pct:.1f}%) observed."
                ),
                suggestion=(
                    "Strong collaborative energy detected. This is a favorable interval for peer discussions "
                    "or hands-on problem solving before transitioning to guided instruction."
                ),
                confidence="High",
                rule_source=self.rule_id,
                priority="Info",
                supporting_metrics={
                    "audio_activity": round(aud_pct, 1),
                    "movement_level": round(mov_pct, 1),
                    "activity_score": round(current_window.activity_score, 1),
                },
            )
        return None


class MissingModalityRule(BaseRecommendationRule):
    """Rule detecting sensor degradation or missing modalities in the fusion pipeline."""

    rule_id = "missing_modality"
    name = "Partial Modality Sensor Observation"
    description = "Warns when sensory input streams are degraded or missing."

    def __init__(self, max_confidence: float = 0.6) -> None:
        self.max_confidence = max_confidence

    def evaluate(
        self,
        session_id: str,
        current_window: WindowActivityScore,
        history: List[WindowActivityScore],
    ) -> Optional[Recommendation]:
        if current_window.confidence <= self.max_confidence:
            missing_parts = []
            if current_window.visual_presence == 0.0:
                missing_parts.append("video")
            if current_window.audio_activity == 0.0:
                missing_parts.append("audio")

            desc = " and ".join(missing_parts) if missing_parts else "unspecified modality"

            return Recommendation(
                id=f"rec-{session_id}-{self.rule_id}-{int(current_window.window_end * 1000)}",
                session_id=session_id,
                timestamp=current_window.window_end,
                condition="Partial Modality Sensor Observation",
                explanation=(
                    f"Activity metrics are currently evaluated with degraded confidence ({current_window.confidence:.2f}). "
                    f"Stream(s) affected: {desc}."
                ),
                suggestion=(
                    "Pedagogical suggestions are operating on partial sensor information. "
                    "Verify physical camera and microphone connections."
                ),
                confidence="Low",
                rule_source=self.rule_id,
                priority="Warning",
                supporting_metrics={
                    "confidence": round(current_window.confidence, 2),
                    "missing_streams": missing_parts,
                },
            )
        return None


class ProlongedSilenceRule(BaseRecommendationRule):
    """Rule identifying extended quiet periods while students remain visually present."""

    rule_id = "prolonged_silence"
    name = "Prolonged Classroom Silence"
    description = "Detects extended silence intervals while visual presence is active."

    def __init__(self, min_windows: int = 4, max_audio: float = 5.0, min_visual: float = 50.0) -> None:
        self.min_windows = min_windows
        self.max_audio = max_audio
        self.min_visual = min_visual

    def evaluate(
        self,
        session_id: str,
        current_window: WindowActivityScore,
        history: List[WindowActivityScore],
    ) -> Optional[Recommendation]:
        relevant = history + [current_window]
        if len(relevant) < self.min_windows:
            return None

        recent = relevant[-self.min_windows :]
        if all(
            _to_percentage(w.audio_activity) <= self.max_audio
            and _to_percentage(w.visual_presence) >= self.min_visual
            for w in recent
        ):
            duration = sum(w.window_end - w.window_start for w in recent)
            return Recommendation(
                id=f"rec-{session_id}-{self.rule_id}-{int(current_window.window_end * 1000)}",
                session_id=session_id,
                timestamp=current_window.window_end,
                condition="Prolonged Classroom Silence",
                explanation=(
                    f"Zero or negligible acoustic activity recorded across {self.min_windows} "
                    f"consecutive intervals ({duration:.1f}s) despite active visual presence."
                ),
                suggestion=(
                    "Expected during independent reading, quizzes, or quiet work. If currently lecturing, "
                    "consider pausing for open questions or soliciting verbal reactions."
                ),
                confidence="Medium",
                rule_source=self.rule_id,
                priority="Info",
                supporting_metrics={
                    "silence_windows": self.min_windows,
                    "duration_seconds": round(duration, 1),
                    "audio_activity": round(current_window.audio_activity, 1),
                    "visual_presence": round(current_window.visual_presence, 1),
                },
            )
        return None


class TeachingRecommendationEngine:
    """Orchestrates rule evaluation, deduplication, cooldowns, and prioritization."""

    def __init__(
        self,
        config: Optional[AgentConfig] = None,
        rules: Optional[List[BaseRecommendationRule]] = None,
    ) -> None:
        """Initialize recommendation engine with configuration and rules.

        Args:
            config: Optional AgentConfig instance. If None, loaded from application config.
            rules: Optional custom rule collection. If None, default 6 rules are instantiated.
        """
        self.config = config or get_config().agent
        self.rules: List[BaseRecommendationRule] = rules or [
            SustainedLowActivityRule(
                threshold=self.config.sustained_low_threshold,
                min_windows=self.config.sustained_low_windows,
            ),
            IncreasingNoiseRule(
                high_noise_threshold=self.config.high_noise_threshold,
            ),
            ReducedVisualPresenceRule(
                threshold=self.config.reduced_visual_threshold,
            ),
            HighInteractionPeriodRule(
                min_audio=self.config.high_interaction_audio,
                min_movement=self.config.high_interaction_movement,
            ),
            MissingModalityRule(max_confidence=0.6),
            ProlongedSilenceRule(
                min_windows=self.config.prolonged_silence_windows,
                max_audio=5.0,
                min_visual=self.config.reduced_visual_threshold,
            ),
        ]
        self._last_triggered: Dict[str, float] = {}
        self._history: List[Recommendation] = []

    def add_rule(self, rule: BaseRecommendationRule) -> None:
        """Register a new recommendation rule into the engine."""
        self.rules.append(rule)
        logger.info("Registered rule '%s' (%s)", rule.name, rule.rule_id)

    def evaluate_window(
        self,
        session_id: str,
        current_window: WindowActivityScore,
        history: List[WindowActivityScore],
    ) -> List[Recommendation]:
        """Evaluate all active rules on the current window with cooldown checks.

        Args:
            session_id: Unique session identifier.
            current_window: Latest window activity score.
            history: Chronological list of previous window scores in session.

        Returns:
            List[Recommendation]: Triggered recommendations sorted by priority.
        """
        triggered: List[Recommendation] = []

        for rule in self.rules:
            # Check cooldown before invoking rule logic
            last_time = self._last_triggered.get(rule.rule_id, -999999.0)
            if (current_window.window_end - last_time) < self.config.cooldown_seconds:
                continue

            rec = rule.evaluate(session_id, current_window, history)
            if rec is not None:
                self._last_triggered[rule.rule_id] = current_window.window_end
                self._history.append(rec)
                triggered.append(rec)
                logger.debug(
                    "Rule '%s' triggered at %.1fs (Priority: %s)",
                    rule.rule_id,
                    current_window.window_end,
                    rec.priority,
                )

        # Sort recommendations by priority: Urgent (1) -> Warning (2) -> Medium (3) -> Info (4)
        triggered.sort(key=lambda r: PRIORITY_RANK.get(r.priority, 99))
        return triggered

    def evaluate_timeline(
        self,
        session_id: str,
        timeline: List[WindowActivityScore],
    ) -> List[Recommendation]:
        """Evaluate rules sequentially across an entire session activity timeline.

        Args:
            session_id: Session identifier.
            timeline: Full chronological sequence of window activity scores.

        Returns:
            List[Recommendation]: All recommendations produced throughout the timeline.
        """
        all_recs: List[Recommendation] = []
        history: List[WindowActivityScore] = []

        for score in timeline:
            recs = self.evaluate_window(session_id, score, history)
            all_recs.extend(recs)
            history.append(score)

        return all_recs

    def get_history(self) -> List[Recommendation]:
        """Return all historical recommendations generated by this engine instance."""
        return list(self._history)

    def clear_history(self) -> None:
        """Reset internal recommendation history and cooldown timers."""
        self._history.clear()
        self._last_triggered.clear()
