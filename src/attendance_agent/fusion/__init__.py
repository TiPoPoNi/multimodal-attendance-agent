"""Multimodal signal fusion and activity proxy scoring engine."""

from attendance_agent.fusion.scoring import (
    MultimodalFusionEngine,
    SessionActivityReport,
    WindowActivityScore,
)
from attendance_agent.fusion.synchronizer import (
    MultimodalSynchronizer,
    SynchronizedWindow,
)

__all__ = [
    "MultimodalSynchronizer",
    "SynchronizedWindow",
    "WindowActivityScore",
    "SessionActivityReport",
    "MultimodalFusionEngine",
]
