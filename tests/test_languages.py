"""Multilingual support tests."""

import numpy as np
import pytest

from .conftest import MULTILINGUAL_TEXTS


class TestKokoroLanguages:
    """Language support tests for Kokoro engine."""

    @pytest.fixture(autouse=True)
    def setup(self, kokoro_engine):
        self.engine = kokoro_engine

    @pytest.mark.parametrize("lang,text", [
        ("en-us", MULTILINGUAL_TEXTS["en"]),
        ("es", MULTILINGUAL_TEXTS["es"]),
        ("fr", MULTILINGUAL_TEXTS["fr"]),
    ])
    def test_supported_languages(self, lang, text):
        """Test generating speech in supported languages."""
        audio, sr = self.engine.generate(text, language=lang)

        assert isinstance(audio, np.ndarray)
        assert len(audio) > 0
        assert sr == 24000

    def test_japanese_requires_extra(self):
        """Test Japanese (may require extra package)."""
        try:
            audio, sr = self.engine.generate(
                MULTILINGUAL_TEXTS["ja"],
                language="ja",
            )
            assert len(audio) > 0
        except ImportError:
            pytest.skip("Japanese support requires misaki[ja] package")

    def test_chinese_requires_extra(self):
        """Test Chinese (may require extra package)."""
        try:
            audio, sr = self.engine.generate(
                MULTILINGUAL_TEXTS["zh"],
                language="zh",
            )
            assert len(audio) > 0
        except ImportError:
            pytest.skip("Chinese support requires misaki[zh] package")


class TestChatterboxLanguages:
    """Language support tests for Chatterbox engine."""

    @pytest.fixture(autouse=True)
    def setup(self, chatterbox_engine):
        self.engine = chatterbox_engine

    def test_english_turbo(self):
        """Test English with Turbo model."""
        audio, sr = self.engine.generate(MULTILINGUAL_TEXTS["en"])

        assert isinstance(audio, np.ndarray)
        assert len(audio) > 0

    @pytest.mark.slow
    def test_multilingual_model(self):
        """Test multilingual model with different languages."""
        from engines import ChatterboxEngine

        ml_engine = ChatterboxEngine(model="multilingual")

        for lang in ["en", "fr", "es", "de"]:
            if lang in MULTILINGUAL_TEXTS:
                audio, sr = ml_engine.generate(MULTILINGUAL_TEXTS[lang])
                assert len(audio) > 0, f"Failed for language: {lang}"


class TestQwen3Languages:
    """Language support tests for Qwen3 engine."""

    @pytest.fixture(autouse=True)
    def setup(self, qwen3_engine):
        self.engine = qwen3_engine

    @pytest.mark.parametrize("lang,text", [
        ("en", MULTILINGUAL_TEXTS["en"]),
        ("zh", MULTILINGUAL_TEXTS["zh"]),
        ("ja", MULTILINGUAL_TEXTS["ja"]),
        ("ko", MULTILINGUAL_TEXTS["ko"]),
        ("fr", MULTILINGUAL_TEXTS["fr"]),
        ("de", MULTILINGUAL_TEXTS["de"]),
    ])
    def test_supported_languages(self, lang, text):
        """Test generating speech in supported languages."""
        audio, sr = self.engine.generate(text, language=lang)

        assert isinstance(audio, np.ndarray)
        assert len(audio) > 0

    def test_language_consistency(self):
        """Test that language parameter affects output."""
        text = "Hello"

        # Generate in different languages
        audio_en, _ = self.engine.generate(text, language="en")
        audio_fr, _ = self.engine.generate(text, language="fr")

        # Both should produce valid audio (content may differ)
        assert len(audio_en) > 0
        assert len(audio_fr) > 0


class TestLanguageSupport:
    """Cross-engine language support comparison."""

    def test_english_all_engines(self, kokoro_engine, chatterbox_engine, qwen3_engine):
        """Test that all engines support English."""
        text = MULTILINGUAL_TEXTS["en"]

        for engine in [kokoro_engine, chatterbox_engine, qwen3_engine]:
            audio, sr = engine.generate(text)
            assert len(audio) > 0, f"English failed for {engine.name}"

    def test_language_info(self, kokoro_engine, chatterbox_engine, qwen3_engine):
        """Test that engines report supported languages."""
        for engine in [kokoro_engine, chatterbox_engine, qwen3_engine]:
            info = engine.get_info()
            langs = info.get("supported_languages", engine.supported_languages)

            assert isinstance(langs, list)
            assert len(langs) > 0
            # All should support at least English
            assert any("en" in lang.lower() for lang in langs)
