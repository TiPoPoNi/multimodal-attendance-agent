"""Audio processing module for classroom acoustic activity and noise analysis."""

from attendance_agent.audio.analyzer import (
    AudioAnalysisReport,
    AudioAnalyzer,
    WindowAudioMetric,
)
from attendance_agent.audio.capture import (
    AudioSourceLoader,
    MicrophoneCapture,
    convert_to_mono,
    extract_audio_from_video,
)

__all__ = [
    "AudioSourceLoader",
    "MicrophoneCapture",
    "convert_to_mono",
    "extract_audio_from_video",
    "WindowAudioMetric",
    "AudioAnalysisReport",
    "AudioAnalyzer",
]
