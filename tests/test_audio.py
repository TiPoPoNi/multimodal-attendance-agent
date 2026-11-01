"""Unit tests for the audio processing module.

Validates RMS energy, Zero-Crossing Rate, silence detection, noise classification,
synthetic signal generation, stereo-to-mono conversion, and SQLite metrics storage.
"""

from pathlib import Path
import numpy as np
import pytest
import soundfile as sf

from attendance_agent.audio.analyzer import AudioAnalyzer
from attendance_agent.audio.capture import (
    AudioSourceLoader,
    MicrophoneCapture,
    convert_to_mono,
)
from attendance_agent.storage.database import DatabaseManager


def test_convert_to_mono() -> None:
    """Verify stereo audio in different shapes is converted to 1D mono float32."""
    # 1D array
    mono_in = np.array([0.1, 0.2, 0.3], dtype=np.float32)
    assert np.array_equal(convert_to_mono(mono_in), mono_in)

    # 2D stereo (samples, channels): 3 samples, 2 channels
    stereo_samples_first = np.array([[0.2, 0.4], [0.6, 0.8], [0.1, 0.3]], dtype=np.float32)
    mono_res1 = convert_to_mono(stereo_samples_first)
    assert mono_res1.ndim == 1
    assert len(mono_res1) == 3
    assert mono_res1[0] == pytest.approx(0.3)
    assert mono_res1[1] == pytest.approx(0.7)
    assert mono_res1[2] == pytest.approx(0.2)

    # 2D stereo (channels, samples): 2 channels, 3 samples
    stereo_ch_first = np.array([[0.2, 0.6, 0.1], [0.4, 0.8, 0.3]], dtype=np.float32)
    mono_res2 = convert_to_mono(stereo_ch_first)
    assert mono_res2.ndim == 1
    assert len(mono_res2) == 3
    assert mono_res2[0] == pytest.approx(0.3)
    assert mono_res2[1] == pytest.approx(0.7)
    assert mono_res2[2] == pytest.approx(0.2)


def test_rms_computation_accuracy() -> None:
    """Verify RMS energy calculation against theoretical sine wave value."""
    # Pure silence -> RMS = 0
    silence = np.zeros(16000, dtype=np.float32)
    assert AudioAnalyzer.compute_rms(silence) == 0.0

    # Sine wave with amplitude A = 0.5: Theoretical RMS = A / sqrt(2) ≈ 0.35355
    sr = 16000
    t = np.linspace(0, 1.0, sr, endpoint=False)
    sine = (0.5 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
    rms = AudioAnalyzer.compute_rms(sine)
    assert rms == pytest.approx(0.5 / np.sqrt(2), rel=1e-2)


def test_zcr_computation_accuracy() -> None:
    """Verify Zero-Crossing Rate calculation matches theoretical frequency."""
    sr = 16000
    freq = 400.0  # 400 Hz
    t = np.linspace(0, 1.0, sr, endpoint=False)
    sine = np.sin(2 * np.pi * freq * t).astype(np.float32)

    # Theoretical ZCR = 2 * freq / sr = 800 / 16000 = 0.05
    zcr = AudioAnalyzer.compute_zcr(sine)
    assert zcr == pytest.approx(0.05, abs=0.002)


def test_silence_detection() -> None:
    """Verify that pure room silence is correctly flagged as silence and inactive."""
    analyzer = AudioAnalyzer(
        sample_rate=16000,
        window_duration_seconds=5.0,
        silence_threshold=0.005,
    )
    silence = AudioAnalyzer.generate_synthetic_audio(duration_seconds=10.0, signal_type="silence")

    report = analyzer.analyze_audio_signal(silence, sample_rate=16000)

    assert report.window_count == 2
    assert report.active_window_count == 0
    assert report.activity_ratio == 0.0
    assert report.silence_duration_seconds == pytest.approx(10.0, abs=0.1)
    for m in report.metrics:
        assert m.is_silence is True
        assert m.activity_detected is False
        assert m.noise_level == "low"


def test_high_noise_detection() -> None:
    """Verify that high-amplitude noise is classified with noise_level 'high'."""
    analyzer = AudioAnalyzer(
        sample_rate=16000,
        window_duration_seconds=5.0,
        high_noise_threshold=0.12,
    )
    high_noise = AudioAnalyzer.generate_synthetic_audio(duration_seconds=10.0, signal_type="high_noise")

    report = analyzer.analyze_audio_signal(high_noise, sample_rate=16000)

    assert report.window_count == 2
    for m in report.metrics:
        assert m.noise_level == "high"
        assert m.rms_energy > 0.12


def test_speech_lecture_simulation() -> None:
    """Verify speech burst simulation generates active windows with moderate noise."""
    analyzer = AudioAnalyzer(
        sample_rate=16000,
        window_duration_seconds=5.0,
        rms_activity_threshold=0.015,
        zcr_activity_threshold=0.03,
    )
    lecture_audio = AudioAnalyzer.generate_synthetic_audio(
        duration_seconds=30.0,
        signal_type="classroom_lecture",
    )

    report = analyzer.analyze_audio_signal(lecture_audio, sample_rate=16000)

    assert report.window_count == 6
    assert report.active_window_count > 0
    assert 0.0 < report.activity_ratio <= 100.0


def test_windowing_timeline_coverage() -> None:
    """Verify audio is evenly sliced into non-overlapping consecutive time windows."""
    analyzer = AudioAnalyzer(window_duration_seconds=5.0)
    signal = np.zeros(16000 * 20, dtype=np.float32)  # 20 seconds

    report = analyzer.analyze_audio_signal(signal, sample_rate=16000, start_offset=10.0)

    assert report.window_count == 4
    assert report.metrics[0].window_start == pytest.approx(10.0)
    assert report.metrics[0].window_end == pytest.approx(15.0)
    assert report.metrics[3].window_start == pytest.approx(25.0)
    assert report.metrics[3].window_end == pytest.approx(30.0)


def test_audio_file_save_and_load(tmp_path: Path) -> None:
    """Verify AudioSourceLoader correctly loads and resamples audio from disk."""
    wav_path = tmp_path / "test_sample.wav"
    # Write synthetic 22050 Hz audio to disk
    orig_sr = 22050
    synth_audio = AudioAnalyzer.generate_synthetic_audio(duration_seconds=2.0, sample_rate=orig_sr, signal_type="sine_tone")
    sf.write(str(wav_path), synth_audio, samplerate=orig_sr)

    # Load with AudioSourceLoader with target 16000 Hz
    loaded_signal, target_sr = AudioSourceLoader.load_audio(wav_path, target_sr=16000)

    assert target_sr == 16000
    assert loaded_signal.ndim == 1
    # Duration ~ 2.0s -> ~32,000 samples
    assert len(loaded_signal) == pytest.approx(32000, abs=100)


def test_database_persistence_of_audio_metrics(tmp_path: Path) -> None:
    """Verify batch insertion and retrieval of audio metrics in SQLite database."""
    db = DatabaseManager(tmp_path / "audio_test.db")
    session_id = "sess-audio-test-01"
    db.create_session(session_id, "Audio Lab")

    analyzer = AudioAnalyzer(window_duration_seconds=5.0)
    signal = AudioAnalyzer.generate_synthetic_audio(duration_seconds=15.0)
    report = analyzer.analyze_audio_signal(signal)

    metrics_dicts = [m.to_dict() for m in report.metrics]
    inserted = db.record_audio_metrics_batch(session_id, metrics_dicts)
    assert inserted == 3

    retrieved = db.get_audio_metrics(session_id)
    assert len(retrieved) == 3
    assert retrieved[0]["window_start"] == 0.0
    assert retrieved[0]["window_end"] == 5.0
    assert retrieved[0]["session_id"] == session_id


def test_microphone_device_listing() -> None:
    """Verify microphone device querying runs without raising uncaught exceptions."""
    devices = MicrophoneCapture.list_input_devices()
    assert isinstance(devices, list)
    # is_available should return a bool
    avail = MicrophoneCapture.is_available()
    assert isinstance(avail, bool)
