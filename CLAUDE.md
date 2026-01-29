# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

TTS Compare is a Python tool for comparing text-to-speech engines (Kokoro, Chatterbox, Qwen3). Each engine runs in its own virtual environment due to dependency conflicts.

## Development Commands

### Environment Setup
```bash
# Each engine needs its own venv (chatterbox and qwen3 conflict)
make setup-kokoro       # Creates .venv-kokoro
make setup-chatterbox   # Creates .venv-chatterbox (requires Python 3.11)
make setup-qwen3        # Creates .venv-qwen3

# Activate before working
source .venv-kokoro/bin/activate
```

### Testing & Linting
```bash
make test              # Run all tests (pytest tests/ -v)
make test-fast         # Skip slow tests (pytest tests/ -v -m "not slow")
pytest tests/ -k kokoro -v  # Run tests for specific engine
make lint              # ruff check
make lint-fix          # ruff check --fix
```

### CLI Usage
```bash
tts engines                                    # List installed engines
tts generate -e kokoro -t "Hello" -o out.wav   # Generate speech
tts clone -e chatterbox -r ref.wav -t "Hi" -o cloned.wav  # Voice cloning
tts generate -e kokoro --list-voices           # List voices
```

## Architecture

### Engine System (`engines/`)
- **Registry pattern with lazy loading** - Engines only import when requested
- `get_engine(name)` returns engine instance; raises `EngineNotAvailableError` if not installed
- `get_available_engines()` returns dict of availability status
- All engines implement `TTSEngine` base class from `engines/base.py`

### TTSEngine Interface (`engines/base.py`)
```python
class TTSEngine(ABC):
    def generate(text, voice, speed, language) -> (audio_array, sample_rate)
    def clone_voice(text, reference_audio, reference_text) -> (audio_array, sample_rate)
    def list_voices() -> List[str]
    def get_info() -> Dict
    def detect_device() -> str  # Returns cuda/mps/cpu
```

### CLI (`cli/main.py`)
Click-based CLI with commands: `engines`, `generate`, `clone`, `benchmark`

### Test Framework (`tests/conftest.py`)
- Fixtures auto-skip when engine unavailable
- `any_available_engine` fixture parametrizes across all installed engines
- Markers: `@pytest.mark.slow`, `@pytest.mark.gpu`, `@pytest.mark.voice_cloning`

## Key Constraints

1. **Chatterbox and Qwen3 cannot coexist** - Different transformers versions (4.46.3 vs 4.57.3)
2. **Chatterbox requires Python 3.11** - numpy<1.26 incompatible with 3.12
3. **Kokoro requires espeak-ng** - `brew install espeak-ng`
4. **MPS fallback** - Set `PYTORCH_ENABLE_MPS_FALLBACK=1` for Apple Silicon issues

## Python API

```python
from engines import get_engine
import soundfile as sf

engine = get_engine("kokoro")
audio, sr = engine.generate("Hello")
sf.write("out.wav", audio, sr)

# Voice cloning (chatterbox/qwen3 only)
engine = get_engine("chatterbox")
audio, sr = engine.clone_voice("Hello", reference_audio="voice.wav")
```
