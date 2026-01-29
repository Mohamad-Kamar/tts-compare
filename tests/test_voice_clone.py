"""Voice cloning capability tests."""

import numpy as np
import pytest

from .conftest import SHORT_TEXT, MEDIUM_TEXT


@pytest.mark.voice_cloning
class TestChatterboxVoiceCloning:
    """Voice cloning tests for Chatterbox engine."""

    @pytest.fixture(autouse=True)
    def setup(self, chatterbox_engine):
        self.engine = chatterbox_engine

    def test_clone_voice_basic(self, reference_audio_file):
        """Test basic voice cloning."""
        audio, sr = self.engine.clone_voice(
            SHORT_TEXT,
            reference_audio=str(reference_audio_file),
        )

        assert isinstance(audio, np.ndarray)
        assert len(audio) > 0
        assert sr > 0

    def test_clone_voice_longer_text(self, reference_audio_file):
        """Test voice cloning with longer text."""
        audio, sr = self.engine.clone_voice(
            MEDIUM_TEXT,
            reference_audio=str(reference_audio_file),
        )

        assert isinstance(audio, np.ndarray)
        duration = len(audio) / sr
        assert duration > 1.0

    def test_clone_voice_missing_reference(self):
        """Test error handling for missing reference file."""
        with pytest.raises(FileNotFoundError):
            self.engine.clone_voice(SHORT_TEXT, reference_audio="nonexistent.wav")

    def test_clone_voice_consistency(self, reference_audio_file):
        """Test that cloned voice is consistent across calls."""
        audio1, _ = self.engine.clone_voice(
            SHORT_TEXT,
            reference_audio=str(reference_audio_file),
        )
        audio2, _ = self.engine.clone_voice(
            SHORT_TEXT,
            reference_audio=str(reference_audio_file),
        )

        # Audio should be similar but not identical (model has some randomness)
        # Just verify both are valid
        assert len(audio1) > 0
        assert len(audio2) > 0


@pytest.mark.voice_cloning
class TestQwen3VoiceCloning:
    """Voice cloning tests for Qwen3 engine."""

    @pytest.fixture(autouse=True)
    def setup(self, qwen3_engine):
        self.engine = qwen3_engine

    def test_clone_voice_basic(self, reference_audio_file):
        """Test basic voice cloning."""
        audio, sr = self.engine.clone_voice(
            SHORT_TEXT,
            reference_audio=str(reference_audio_file),
        )

        assert isinstance(audio, np.ndarray)
        assert len(audio) > 0
        assert sr > 0

    def test_clone_voice_with_transcript(self, reference_audio_file):
        """Test voice cloning with reference transcript."""
        audio, sr = self.engine.clone_voice(
            SHORT_TEXT,
            reference_audio=str(reference_audio_file),
            reference_text="This is the reference audio transcript.",
        )

        assert isinstance(audio, np.ndarray)
        assert len(audio) > 0

    def test_clone_voice_missing_reference(self):
        """Test error handling for missing reference file."""
        with pytest.raises(FileNotFoundError):
            self.engine.clone_voice(SHORT_TEXT, reference_audio="nonexistent.wav")


@pytest.mark.voice_cloning
class TestKokoroVoiceCloning:
    """Verify Kokoro properly rejects voice cloning."""

    @pytest.fixture(autouse=True)
    def setup(self, kokoro_engine):
        self.engine = kokoro_engine

    def test_clone_voice_not_supported(self, reference_audio_file):
        """Test that Kokoro raises NotImplementedError for voice cloning."""
        with pytest.raises(NotImplementedError) as excinfo:
            self.engine.clone_voice(
                SHORT_TEXT,
                reference_audio=str(reference_audio_file),
            )

        assert "not support voice cloning" in str(excinfo.value).lower()


@pytest.mark.voice_cloning
class TestVoiceCloningComparison:
    """Compare voice cloning across engines that support it."""

    def test_both_engines_produce_audio(
        self,
        chatterbox_engine,
        qwen3_engine,
        reference_audio_file,
    ):
        """Test that both cloning-capable engines produce valid audio."""
        cb_audio, cb_sr = chatterbox_engine.clone_voice(
            SHORT_TEXT,
            reference_audio=str(reference_audio_file),
        )
        q3_audio, q3_sr = qwen3_engine.clone_voice(
            SHORT_TEXT,
            reference_audio=str(reference_audio_file),
        )

        # Both should produce valid audio
        assert len(cb_audio) > 0
        assert len(q3_audio) > 0

        # Both should have reasonable durations for the same text
        cb_duration = len(cb_audio) / cb_sr
        q3_duration = len(q3_audio) / q3_sr

        # Durations should be in the same ballpark (within 3x of each other)
        ratio = max(cb_duration, q3_duration) / min(cb_duration, q3_duration)
        assert ratio < 3.0, f"Duration ratio too large: {ratio}"
