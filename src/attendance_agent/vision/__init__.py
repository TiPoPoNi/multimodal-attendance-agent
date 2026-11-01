"""Computer Vision module for face detection, anonymous tracking, and presence attendance."""

from attendance_agent.vision.attendance import (
    ParticipantAttendanceSummary,
    PresenceAttendanceEngine,
    PresenceInterval,
    SessionAttendanceSummary,
)
from attendance_agent.vision.capture import (
    VideoSource,
    annotate_frame,
)
from attendance_agent.vision.detector import (
    FaceBoundingBox,
    FaceDetectorWrapper,
    RawDetection,
)
from attendance_agent.vision.tracker import (
    AnonymousFaceTracker,
    TrackedParticipant,
    TrackingFrameResult,
)

__all__ = [
    "FaceBoundingBox",
    "FaceDetectorWrapper",
    "RawDetection",
    "AnonymousFaceTracker",
    "TrackedParticipant",
    "TrackingFrameResult",
    "VideoSource",
    "annotate_frame",
    "PresenceInterval",
    "ParticipantAttendanceSummary",
    "SessionAttendanceSummary",
    "PresenceAttendanceEngine",
]
