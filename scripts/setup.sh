#!/bin/bash
# TTS Compare - Setup Script
set -e

cd "$(dirname "$0")/.."

ENGINE="${1:-kokoro}"
VENV_DIR=".venv-$ENGINE"

echo "TTS Compare Setup"
echo "================="
echo ""
echo "Engine: $ENGINE"
echo "Venv:   $VENV_DIR"
echo ""

# Validate engine
case "$ENGINE" in
    kokoro|chatterbox|qwen3) ;;
    *)
        echo "Error: Unknown engine '$ENGINE'"
        echo "Usage: ./scripts/setup.sh [kokoro|chatterbox|qwen3]"
        exit 1
        ;;
esac

# Find appropriate Python version
# Chatterbox requires Python 3.11 (numpy<1.26 constraint doesn't support 3.12)
PYTHON=""
if [ "$ENGINE" = "chatterbox" ]; then
    # Chatterbox needs Python 3.11
    for cmd in python3.11 python3; do
        if command -v $cmd &> /dev/null; then
            version=$($cmd -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
            if [[ "$version" == "3.11" ]]; then
                PYTHON=$cmd
                echo "Using: $PYTHON ($version)"
                break
            fi
        fi
    done
    if [ -z "$PYTHON" ]; then
        echo "Error: Chatterbox requires Python 3.11 (numpy<1.26 doesn't support 3.12)"
        echo ""
        echo "Install with:"
        echo "  brew install python@3.11  # macOS"
        echo "  uv python install 3.11    # or use uv"
        exit 1
    fi
else
    # Kokoro and Qwen3 work with 3.11 or 3.12
    for cmd in python3.12 python3.11 python3; do
        if command -v $cmd &> /dev/null; then
            version=$($cmd -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
            if [[ "$version" == "3.11" || "$version" == "3.12" ]]; then
                PYTHON=$cmd
                echo "Using: $PYTHON ($version)"
                break
            fi
        fi
    done
    if [ -z "$PYTHON" ]; then
        echo "Error: Python 3.11 or 3.12 required (not 3.13+)"
        echo ""
        echo "Install with:"
        echo "  brew install python@3.12  # macOS"
        echo "  uv python install 3.12    # or use uv"
        exit 1
    fi
fi

# Create venv
if [ -d "$VENV_DIR" ]; then
    echo ""
    read -p "Remove existing $VENV_DIR? [y/N] " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        rm -rf "$VENV_DIR"
    else
        echo "Aborted."
        exit 1
    fi
fi

echo ""
echo "Creating $VENV_DIR..."
$PYTHON -m venv "$VENV_DIR"
source "$VENV_DIR/bin/activate"

echo "Installing tts-compare[$ENGINE]..."
pip install --upgrade pip setuptools wheel -q
pip install -e ".[$ENGINE]"

# Check espeak-ng
if ! command -v espeak-ng &> /dev/null; then
    echo ""
    echo "Warning: espeak-ng not found (required by Kokoro)"
    if [[ "$OSTYPE" == "darwin"* ]]; then
        echo "  Install: brew install espeak-ng"
    else
        echo "  Install: apt install espeak-ng"
    fi
fi

echo ""
echo "Done! Activate with:"
echo "  source $VENV_DIR/bin/activate"
echo ""
echo "Test:"
echo "  tts engines"
