"""Multimodal engagement and activity proxy scoring engine.

Combines synchronized visual presence, participant motion, and acoustic activity
into an explainable, transparent 0-to-100 session activity proxy score.
Does NOT measure emotions, intelligence, individual student attention, or motivation.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from attendance_agent.fusion.synchronizer import SynchronizedWindow
from attendance_agent.utils.logger import get_logger

logger = get_logger("attendance_agent.fusion.scoring")


@dataclass
class WindowActivityScore:
    """Evaluated multimodal activity score for an individual time window."""

    window_index: int
    window_start: float
    window_end: float

    # Observable Normalized Signals [0.0, 1.0]
    visual_presence: float
    audio_activity: float
    movement_level: float

    # Resulting Aggregate Metrics
    activity_score: float  # 0.0 to 100.0
    confidence: float  # 0.0 to 1.0
    label: str  # "Low observed activity", "Moderate observed activity", etc.

    # Available Modalities
    modalities_present: List[str]

    # Explainability Component Contributions (sum to activity_score)
    visual_contribution: float
    audio_contribution: float
    movement_contribution: float

    def to_dict(self) -> Dict[str, Any]:
        """Convert score object to structured dictionary."""
        return {
            "window_index": self.window_index,
            "window_start": round(self.window_start, 2),
            "window_end": round(self.window_end, 2),
            "visual_presence": round(self.visual_presence, 3),
            "audio_activity": round(self.audio_activity, 3),
            "movement_level": round(self.movement_level, 3),
            "activity_score": round(self.activity_score, 1),
            "confidence": round(self.confidence, 2),
            "label": self.label,
            "modalities_present": self.modalities_present,
            "contributions": {
                "visual": round(self.visual_contribution, 1),
                "audio": round(self.audio_contribution, 1),
                "movement": round(self.movement_contribution, 1),
            },
        }


@dataclass
class SessionActivityReport:
    """Aggregated multimodal activity profile for an entire session."""

    total_windows: int
    mean_activity_score: float
    mean_confidence: float
    overall_label: str
    active_modalities: List[str]
    window_scores: List[WindowActivityScore] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert session report to dictionary."""
        return {
            "total_windows": self.total_windows,
            "mean_activity_score": round(self.mean_activity_score, 1),
            "mean_confidence": round(self.mean_confidence, 2),
            "overall_label": self.overall_label,
            "active_modalities": self.active_modalities,
            "window_scores": [s.to_dict() for s in self.window_scores],
        }

    def to_dataframe(self) -> pd.DataFrame:
        """Convert window scores into a tabular pandas DataFrame."""
        if not self.window_scores:
            return pd.DataFrame(
                columns=[
                    "Window",
                    "Start (s)",
                    "End (s)",
                    "Activity Score",
                    "Label",
                    "Confidence",
                    "Visual Presence",
                    "Audio Activity",
                    "Movement",
                ]
            )

        rows = [
            {
                "Window": s.window_index,
                "Start (s)": round(s.window_start, 1),
                "End (s)": round(s.window_end, 1),
                "Activity Score": round(s.activity_score, 1),
                "Label": s.label,
                "Confidence": f"{int(s.confidence * 100)}%",
                "Visual Presence": f"{int(s.visual_presence * 100)}%",
                "Audio Activity": f"{int(s.audio_activity * 100)}%",
                "Movement": f"{int(s.movement_level * 100)}%",
            }
            for s in self.window_scores
        ]
        return pd.DataFrame(rows)


class MultimodalFusionEngine:
    """Combines synchronized video and audio metrics into explainable activity proxy scores."""

    def __init__(
        self,
        weight_visual_presence: float = 0.40,
        weight_audio_activity: float = 0.40,
        weight_movement: float = 0.20,
        low_activity_threshold: float = 30.0,
        high_activity_threshold: float = 70.0,
        min_confidence_threshold: float = 0.30,
    ) -> None:
        """Initialize the multimodal fusion scoring engine.

        Args:
            weight_visual_presence: Weight for visual presence ratio (default: 0.40).
            weight_audio_activity: Weight for audio activity ratio (default: 0.40).
            weight_movement: Weight for participant movement level (default: 0.20).
            low_activity_threshold: Boundary below which label is 'Low observed activity'.
            high_activity_threshold: Boundary above which label is 'High observed activity'.
            min_confidence_threshold: Minimum confidence required before flagging 'Insufficient data'.
        """
        self.w_visual = max(0.0, weight_visual_presence)
        self.w_audio = max(0.0, weight_audio_activity)
        self.w_movement = max(0.0, weight_movement)

        self.low_threshold = low_activity_threshold
        self.high_threshold = high_activity_threshold
        self.min_confidence = min_confidence_threshold

    def classify_label(self, score: float, confidence: float) -> str:
        """Map score and confidence to transparent activity label.

        Args:
            score: Computed 0-to-100 activity score.
            confidence: Data confidence (0.0 to 1.0).

        Returns:
            str: Descriptive label.
        """
        if confidence < self.min_confidence:
            return "Insufficient data"
        if score < self.low_threshold:
            return "Low observed activity"
        if score < self.high_threshold:
            return "Moderate observed activity"
        return "High observed activity"

    def score_window(self, window: SynchronizedWindow) -> WindowActivityScore:
        """Calculate explainable multimodal activity score for a single synchronized window.

        Handles missing modalities by re-normalizing weights and discounting confidence.

        Args:
            window: Input SynchronizedWindow containing visual and audio metrics.

        Returns:
            WindowActivityScore: Structured score with component breakdown and confidence.
        """
        modalities: List[str] = []
        if window.has_visual_data:
            modalities.append("vision")
        if window.has_audio_data:
            modalities.append("audio")

        # Case 1: Insufficient data (neither modality present)
        if not modalities:
            return WindowActivityScore(
                window_index=window.window_index,
                window_start=window.window_start,
                window_end=window.window_end,
                visual_presence=0.0,
                audio_activity=0.0,
                movement_level=0.0,
                activity_score=0.0,
                confidence=0.0,
                label="Insufficient data",
                modalities_present=[],
                visual_contribution=0.0,
                audio_contribution=0.0,
                movement_contribution=0.0,
            )

        # Normalize observable inputs
        norm_vis = min(1.0, max(0.0, window.visual_presence_ratio))
        norm_mov = min(1.0, max(0.0, window.movement_level))
        norm_aud = min(1.0, max(0.0, window.audio_activity_ratio))

        # Re-weight dynamically based on available modalities
        if window.has_visual_data and window.has_audio_data:
            # Both modalities available
            total_w = self.w_visual + self.w_audio + self.w_movement
            w_v = (self.w_visual / total_w) if total_w > 0 else 0.4
            w_a = (self.w_audio / total_w) if total_w > 0 else 0.4
            w_m = (self.w_movement / total_w) if total_w > 0 else 0.2

            # Confidence is high when both are active and detections are clear
            det_conf = max(0.5, window.mean_detection_confidence) if window.mean_detection_confidence > 0 else 0.8
            confidence = min(1.0, det_conf)

        elif window.has_visual_data and not window.has_audio_data:
            # Vision only: re-weight visual presence and movement
            total_w = self.w_visual + self.w_movement
            w_v = (self.w_visual / total_w) if total_w > 0 else 0.7
            w_m = (self.w_movement / total_w) if total_w > 0 else 0.3
            w_a = 0.0

            # Confidence penalty for missing audio modality
            confidence = 0.65 * (window.mean_detection_confidence if window.mean_detection_confidence > 0 else 0.8)

        else:
            # Audio only: audio receives full weight
            w_v = 0.0
            w_m = 0.0
            w_a = 1.0

            # Confidence penalty for missing visual modality
            confidence = 0.55

        # Compute constituent contributions (scaled to 100)
        c_v = w_v * norm_vis * 100.0
        c_a = w_a * norm_aud * 100.0
        c_m = w_m * norm_mov * 100.0

        activity_score = min(100.0, max(0.0, c_v + c_a + c_m))
        label = self.classify_label(activity_score, confidence)

        return WindowActivityScore(
            window_index=window.window_index,
            window_start=window.window_start,
            window_end=window.window_end,
            visual_presence=norm_vis,
            audio_activity=norm_aud,
            movement_level=norm_mov,
            activity_score=activity_score,
            confidence=confidence,
            label=label,
            modalities_present=modalities,
            visual_contribution=c_v,
            audio_contribution=c_a,
            movement_contribution=c_m,
        )

    def score_session(
        self,
        windows: List[SynchronizedWindow],
    ) -> SessionActivityReport:
        """Score an entire sequence of synchronized windows and summarize session activity.

        Args:
            windows: Sequence of SynchronizedWindow instances.

        Returns:
            SessionActivityReport: Session-level aggregated activity profile.
        """
        if not windows:
            return SessionActivityReport(
                total_windows=0,
                mean_activity_score=0.0,
                mean_confidence=0.0,
                overall_label="Insufficient data",
                active_modalities=[],
                window_scores=[],
            )

        scores = [self.score_window(w) for w in windows]

        # Calculate session averages excluding empty windows
        valid_scores = [s.activity_score for s in scores if s.confidence >= self.min_confidence]
        mean_score = float(np.mean(valid_scores)) if valid_scores else 0.0
        mean_conf = float(np.mean([s.confidence for s in scores])) if scores else 0.0

        overall_label = self.classify_label(mean_score, mean_conf)

        active_mods = set()
        for s in scores:
            active_mods.update(s.modalities_present)

        return SessionActivityReport(
            total_windows=len(scores),
            mean_activity_score=mean_score,
            mean_confidence=mean_conf,
            overall_label=overall_label,
            active_modalities=sorted(list(active_mods)),
            window_scores=scores,
        )
