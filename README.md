# TTS Compare

Compare text-to-speech engines: Kokoro, Chatterbox, and Qwen3.

## Quick Start (macOS)

### Step 1: Install dependencies

```bash
brew install python@3.11 espeak-ng
```

### Step 2: Clone the repo

```bash
git clone <repo>
cd tts-compare
```

### Step 3: Pick an engine and set it up

**Option A: Kokoro** (simplest, works on CPU)
```bash
make setup-kokoro
source .venv-kokoro/bin/activate
```

**Option B: Chatterbox** (voice cloning)
```bash
make setup-chatterbox
source .venv-chatterbox/bin/activate
```

**Option C: Qwen3** (highest quality, voice cloning)
```bash
make setup-qwen3
source .venv-qwen3/bin/activate
```

### Step 4: Generate speech

```bash
tts generate -e kokoro -t "Hello, world!" -o hello.wav
```

## Commands

```bash
# Generate speech
tts generate -e kokoro -t "Hello, world!" -o hello.wav
tts generate -e kokoro --stdin -o hello.wav

# Generate from text file
tts generate -e kokoro -i notes.txt -o output.wav

# Clone a voice (chatterbox or qwen3 only)
tts clone -e chatterbox -r voice.wav -t "Hello in my voice" -o out.wav

# List available voices
tts generate -e kokoro --list-voices

# Check which engines are installed
tts engines

# Inspect resolved settings
tts generate -e kokoro -t "Hello" --show-settings
```

## Switching Engines

Each engine has its own environment. To switch:

```bash
source .venv-kokoro/bin/activate      # switch to Kokoro
source .venv-chatterbox/bin/activate  # switch to Chatterbox
source .venv-qwen3/bin/activate       # switch to Qwen3
```

## Which Engine Should I Use?

| Engine | Size | Best For | Requires |
|--------|------|----------|----------|
| **Kokoro** | 82M | Quick TTS, low resources | CPU OK |
| **Chatterbox** | 350-500M | Voice cloning | GPU (4-6GB) |
| **Qwen3** | 0.6-1.7B | Highest quality | GPU (4-8GB) |

## Python API

```python
from engines import get_engine
import soundfile as sf

engine = get_engine("kokoro")
audio, sr = engine.generate("Hello, world!")
sf.write("output.wav", audio, sr)

# Voice cloning (chatterbox or qwen3)
engine = get_engine("chatterbox")
audio, sr = engine.clone_voice("Hello in my voice", reference_audio="voice.wav")
sf.write("cloned.wav", audio, sr)
```

## Troubleshooting

**"command not found: make"**
```bash
xcode-select --install
```

**"espeak-ng not found"**
```bash
brew install espeak-ng
```

**"Python 3.11 required" error**
```bash
brew install python@3.11
```

**MPS errors on Apple Silicon**
```bash
export PYTORCH_ENABLE_MPS_FALLBACK=1
```

## License

MIT
