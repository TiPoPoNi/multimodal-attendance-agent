"""Pedagogical recommendation agent and teaching suggestions.

Exports the rule-based recommendation engine, individual transparent rules,
and priority rankings for decision-support in the classroom.
"""

from attendance_agent.agent.recommendations import (
    PRIORITY_RANK,
    BaseRecommendationRule,
    HighInteractionPeriodRule,
    IncreasingNoiseRule,
    MissingModalityRule,
    ProlongedSilenceRule,
    ReducedVisualPresenceRule,
    SustainedLowActivityRule,
    TeachingRecommendationEngine,
)

__all__ = [
    "PRIORITY_RANK",
    "BaseRecommendationRule",
    "HighInteractionPeriodRule",
    "IncreasingNoiseRule",
    "MissingModalityRule",
    "ProlongedSilenceRule",
    "ReducedVisualPresenceRule",
    "SustainedLowActivityRule",
    "TeachingRecommendationEngine",
]
