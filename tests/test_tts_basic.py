"""Basic TTS capability tests."""

import numpy as np
import pytest

from .conftest import SHORT_TEXT, MEDIUM_TEXT, LONG_TEXT


class TestKokoroBasic:
    """Basic tests for Kokoro engine."""

    @pytest.fixture(autouse=True)
    def setup(self, kokoro_engine):
        self.engine = kokoro_engine

    def test_generate_short_text(self):
        """Test generating audio from short text."""
        audio, sr = self.engine.generate(SHORT_TEXT)

        assert isinstance(audio, np.ndarray)
        assert len(audio) > 0
        assert sr == 24000

    def test_generate_medium_text(self):
        """Test generating audio from medium text."""
        audio, sr = self.engine.generate(MEDIUM_TEXT)

        assert isinstance(audio, np.ndarray)
        assert len(audio) > 0
        # Medium text should produce more audio
        duration = len(audio) / sr
        assert duration > 1.0

    @pytest.mark.slow
    def test_generate_long_text(self):
        """Test generating audio from long text."""
        audio, sr = self.engine.generate(LONG_TEXT)

        assert isinstance(audio, np.ndarray)
        assert len(audio) > 0
        duration = len(audio) / sr
        assert duration > 5.0

    def test_different_voices(self):
        """Test generating with different voices."""
        voices = ["af_heart", "am_adam"]

        for voice in voices:
            audio, sr = self.engine.generate(SHORT_TEXT, voice=voice)
            assert len(audio) > 0, f"Failed for voice: {voice}"

    def test_speed_control(self):
        """Test speech speed control."""
        audio_normal, _ = self.engine.generate(SHORT_TEXT, speed=1.0)
        audio_fast, _ = self.engine.generate(SHORT_TEXT, speed=1.5)
        audio_slow, _ = self.engine.generate(SHORT_TEXT, speed=0.7)

        # Faster speech should be shorter
        assert len(audio_fast) < len(audio_normal)
        # Slower speech should be longer
        assert len(audio_slow) > len(audio_normal)

    def test_list_voices(self):
        """Test listing available voices."""
        voices = self.engine.list_voices()

        assert isinstance(voices, list)
        assert len(voices) > 0
        assert "af_heart" in voices

    def test_get_info(self):
        """Test getting engine info."""
        info = self.engine.get_info()

        assert info["name"] == "kokoro"
        assert info["parameters"] == "82M"
        assert info["supports_voice_cloning"] is False
        assert info["sample_rate"] == 24000

    def test_voice_cloning_not_supported(self):
        """Test that voice cloning raises NotImplementedError."""
        with pytest.raises(NotImplementedError):
            self.engine.clone_voice(SHORT_TEXT, "reference.wav")


class TestChatterboxBasic:
    """Basic tests for Chatterbox engine."""

    @pytest.fixture(autouse=True)
    def setup(self, chatterbox_engine):
        self.engine = chatterbox_engine

    def test_generate_short_text(self):
        """Test generating audio from short text."""
        audio, sr = self.engine.generate(SHORT_TEXT)

        assert isinstance(audio, np.ndarray)
        assert len(audio) > 0
        assert sr > 0

    def test_paralinguistic_tags(self):
        """Test paralinguistic tag support (Turbo model)."""
        text_with_tag = "Hello [laugh] how are you?"
        audio, sr = self.engine.generate(text_with_tag)

        assert isinstance(audio, np.ndarray)
        assert len(audio) > 0

    def test_get_info(self):
        """Test getting engine info."""
        info = self.engine.get_info()

        assert info["name"] == "chatterbox"
        assert info["supports_voice_cloning"] is True
        assert "turbo" in info["available_models"]


class TestQwen3Basic:
    """Basic tests for Qwen3 engine."""

    @pytest.fixture(autouse=True)
    def setup(self, qwen3_engine):
        self.engine = qwen3_engine

    def test_generate_short_text(self):
        """Test generating audio from short text."""
        audio, sr = self.engine.generate(SHORT_TEXT)

        assert isinstance(audio, np.ndarray)
        assert len(audio) > 0
        assert sr > 0

    def test_different_speakers(self):
        """Test generating with different preset speakers."""
        speakers = ["Ryan", "Aiden"]

        for speaker in speakers:
            audio, sr = self.engine.generate(SHORT_TEXT, voice=speaker)
            assert len(audio) > 0, f"Failed for speaker: {speaker}"

    def test_voice_instruction(self):
        """Test voice instruction for CustomVoice model."""
        audio, sr = self.engine.generate(
            SHORT_TEXT,
            instruct="Speak slowly and clearly",
        )

        assert isinstance(audio, np.ndarray)
        assert len(audio) > 0

    def test_list_voices(self):
        """Test listing available speakers."""
        voices = self.engine.list_voices()

        assert isinstance(voices, list)
        assert len(voices) > 0
        assert "Ryan" in voices

    def test_get_info(self):
        """Test getting engine info."""
        info = self.engine.get_info()

        assert info["name"] == "qwen3"
        assert info["supports_voice_cloning"] is True
        assert info["model_size"] in ["0.6B", "1.7B"]


class TestOutputFormat:
    """Test output format consistency across engines."""

    def test_audio_dtype(self, any_engine):
        """Test that audio output is float32."""
        audio, _ = any_engine.generate(SHORT_TEXT)
        assert audio.dtype in [np.float32, np.float64]

    def test_audio_range(self, any_engine):
        """Test that audio values are in valid range."""
        audio, _ = any_engine.generate(SHORT_TEXT)
        assert np.abs(audio).max() <= 1.1  # Allow small margin

    def test_audio_mono(self, any_engine):
        """Test that audio is mono (1D array)."""
        audio, _ = any_engine.generate(SHORT_TEXT)
        assert len(audio.shape) == 1

    def test_sample_rate_valid(self, any_engine):
        """Test that sample rate is valid."""
        _, sr = any_engine.generate(SHORT_TEXT)
        assert sr in [16000, 22050, 24000, 44100, 48000]
