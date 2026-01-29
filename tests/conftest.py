"""Pytest configuration and fixtures for TTS tests.

Tests automatically skip for engines that are not installed.
"""

import os
import sys
import tempfile
from pathlib import Path
from typing import Generator

import numpy as np
import pytest
import soundfile as sf

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from engines import get_available_engines, get_engine


# Test text samples
SHORT_TEXT = "Hello, this is a test."
MEDIUM_TEXT = "The quick brown fox jumps over the lazy dog. This sentence contains every letter of the English alphabet."
LONG_TEXT = """
In a hole in the ground there lived a hobbit. Not a nasty, dirty, wet hole,
filled with the ends of worms and an oozy smell, nor yet a dry, bare, sandy hole
with nothing in it to sit down on or to eat: it was a hobbit-hole, and that means comfort.
"""

# Multilingual test texts
MULTILINGUAL_TEXTS = {
    "en": "Hello, how are you today?",
    "es": "Hola, como estas hoy?",
    "fr": "Bonjour, comment allez-vous aujourd'hui?",
    "de": "Hallo, wie geht es Ihnen heute?",
    "ja": "こんにちは、今日はお元気ですか？",
    "zh": "你好，今天好吗？",
    "ko": "안녕하세요, 오늘 어떠세요?",
}


# Cache availability check at module load
_AVAILABLE_ENGINES = get_available_engines()


def pytest_collection_modifyitems(config, items):
    """Auto-skip tests for unavailable engines based on test name/class."""
    for item in items:
        # Check if the test name or class contains an engine name
        nodeid_lower = item.nodeid.lower()

        for engine_name, is_available in _AVAILABLE_ENGINES.items():
            # Skip if engine name appears in test path and engine is not installed
            if engine_name in nodeid_lower and not is_available:
                skip_marker = pytest.mark.skip(
                    reason=f"Engine '{engine_name}' not installed. "
                           f"Install with: pip install tts-compare[{engine_name}]"
                )
                item.add_marker(skip_marker)
                break


def pytest_configure(config):
    """Configure custom markers."""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "gpu: marks tests that require GPU"
    )
    config.addinivalue_line(
        "markers", "voice_cloning: marks tests that require voice cloning capability"
    )


@pytest.fixture(scope="session")
def project_root() -> Path:
    """Get project root directory."""
    return Path(__file__).parent.parent


@pytest.fixture(scope="session")
def samples_dir(project_root) -> Path:
    """Get samples directory."""
    samples = project_root / "samples"
    samples.mkdir(exist_ok=True)
    return samples


@pytest.fixture(scope="session")
def output_dir(project_root) -> Path:
    """Get output directory."""
    output = project_root / "output"
    output.mkdir(exist_ok=True)
    return output


@pytest.fixture
def temp_audio_file() -> Generator[Path, None, None]:
    """Create a temporary audio file for testing."""
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        # Generate a simple sine wave
        sr = 24000
        duration = 3.0
        t = np.linspace(0, duration, int(sr * duration), dtype=np.float32)
        audio = 0.5 * np.sin(2 * np.pi * 440 * t)  # 440 Hz tone

        sf.write(f.name, audio, sr)
        yield Path(f.name)

    # Cleanup
    os.unlink(f.name)


@pytest.fixture
def reference_audio_file(samples_dir) -> Path:
    """Get or create a reference audio file for voice cloning tests."""
    ref_path = samples_dir / "reference_voice.wav"

    if not ref_path.exists():
        # Create a synthetic reference (in real tests, use actual speech)
        sr = 24000
        duration = 5.0
        t = np.linspace(0, duration, int(sr * duration), dtype=np.float32)

        # Create a more speech-like signal with harmonics
        audio = np.zeros_like(t)
        for i, freq in enumerate([220, 440, 660, 880]):
            audio += (0.3 / (i + 1)) * np.sin(2 * np.pi * freq * t)

        # Add envelope
        envelope = np.exp(-t / duration)
        audio = audio * envelope

        sf.write(str(ref_path), audio.astype(np.float32), sr)

    return ref_path


# Engine fixtures with automatic skip if not available

@pytest.fixture(scope="module")
def kokoro_engine():
    """Create Kokoro engine instance. Skips if not installed."""
    if not _AVAILABLE_ENGINES.get("kokoro", False):
        pytest.skip("Kokoro not installed. Install with: pip install tts-compare[kokoro]")
    return get_engine("kokoro")


@pytest.fixture(scope="module")
def chatterbox_engine():
    """Create Chatterbox engine instance. Skips if not installed."""
    if not _AVAILABLE_ENGINES.get("chatterbox", False):
        pytest.skip("Chatterbox not installed. Install with: pip install tts-compare[chatterbox]")
    return get_engine("chatterbox", model="turbo")


@pytest.fixture(scope="module")
def qwen3_engine():
    """Create Qwen3 engine instance. Skips if not installed."""
    if not _AVAILABLE_ENGINES.get("qwen3", False):
        pytest.skip("Qwen3 not installed. Install with: pip install tts-compare[qwen3]")
    return get_engine("qwen3", model_size="0.6B", model_type="CustomVoice")


@pytest.fixture(params=["kokoro", "chatterbox", "qwen3"])
def any_available_engine(request):
    """Parametrized fixture that runs tests only for INSTALLED engines."""
    engine_name = request.param

    if not _AVAILABLE_ENGINES.get(engine_name, False):
        pytest.skip(f"Engine '{engine_name}' not installed")

    kwargs = {}
    if engine_name == "chatterbox":
        kwargs = {"model": "turbo"}
    elif engine_name == "qwen3":
        kwargs = {"model_size": "0.6B", "model_type": "CustomVoice"}

    return get_engine(engine_name, **kwargs)


# Legacy fixture for backward compatibility
@pytest.fixture(params=["kokoro", "chatterbox", "qwen3"])
def any_engine(request, kokoro_engine, chatterbox_engine, qwen3_engine):
    """Parametrized fixture for all engines (skips unavailable)."""
    engines = {
        "kokoro": kokoro_engine if _AVAILABLE_ENGINES.get("kokoro") else None,
        "chatterbox": chatterbox_engine if _AVAILABLE_ENGINES.get("chatterbox") else None,
        "qwen3": qwen3_engine if _AVAILABLE_ENGINES.get("qwen3") else None,
    }

    engine = engines.get(request.param)
    if engine is None:
        pytest.skip(f"Engine '{request.param}' not installed")

    return engine
