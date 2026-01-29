"""CLI tool for voice cloning.

Usage:
    python -m cli.clone --engine chatterbox --reference voice.wav --text "Clone this" --output out.wav
    python -m cli.clone --engine qwen3 --reference voice.wav --ref-text "transcript" --text "New text" --output out.wav
"""

import argparse
import sys
from pathlib import Path

import soundfile as sf


def main():
    parser = argparse.ArgumentParser(
        description="Voice Cloning - Clone a voice from reference audio",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Clone with Chatterbox
  python -m cli.clone -e chatterbox -r my_voice.wav -t "Hello in my voice" -o cloned.wav

  # Clone with Qwen3 (with transcript)
  python -m cli.clone -e qwen3 -r my_voice.wav --ref-text "Original speech" -t "New text" -o cloned.wav

  # Clone from file input
  python -m cli.clone -e chatterbox -r voice.wav -i script.txt -o narration.wav

Note: Kokoro does not support voice cloning.
        """,
    )

    parser.add_argument(
        "--engine", "-e",
        type=str,
        required=True,
        choices=["chatterbox", "qwen3"],
        help="TTS engine (kokoro not supported for cloning)",
    )
    parser.add_argument(
        "--reference", "-r",
        type=str,
        required=True,
        help="Reference audio file for voice cloning",
    )
    parser.add_argument(
        "--ref-text",
        type=str,
        help="Transcript of reference audio (recommended for Qwen3)",
    )
    parser.add_argument(
        "--text", "-t",
        type=str,
        help="Text to synthesize",
    )
    parser.add_argument(
        "--input", "-i",
        type=str,
        help="Input text file",
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default="cloned_output.wav",
        help="Output audio file (default: cloned_output.wav)",
    )
    parser.add_argument(
        "--language", "-l",
        type=str,
        default="en",
        help="Language code (default: en)",
    )
    parser.add_argument(
        "--device", "-d",
        type=str,
        default="auto",
        help="Device (cuda/mps/cpu/auto)",
    )

    # Chatterbox-specific
    parser.add_argument(
        "--model", "-m",
        type=str,
        default="turbo",
        choices=["turbo", "multilingual", "standard"],
        help="Chatterbox model variant (default: turbo)",
    )
    parser.add_argument(
        "--exaggeration",
        type=float,
        default=0.5,
        help="Exaggeration level 0-1 (Chatterbox, default: 0.5)",
    )

    # Qwen3-specific
    parser.add_argument(
        "--size",
        type=str,
        default="0.6B",
        choices=["0.6B", "1.7B"],
        help="Qwen3 model size (default: 0.6B)",
    )

    args = parser.parse_args()

    # Validate reference file
    if not Path(args.reference).exists():
        print(f"Error: Reference audio not found: {args.reference}")
        sys.exit(1)

    # Import engines
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from engines import get_engine, get_available_engines, ENGINE_INFO, EngineNotAvailableError

    # Early availability check with helpful error message
    available = get_available_engines()
    if not available.get(args.engine, False):
        info = ENGINE_INFO.get(args.engine, {})
        print(f"\nError: Engine '{args.engine}' is not installed.")
        print(f"\nTo install: {info.get('install', f'pip install tts-compare[{args.engine}]')}")
        if info.get("conflict"):
            print(f"\nNote: {info.get('conflict_reason', '')}")
            print(f"You cannot have both {args.engine} and {info['conflict']} in the same environment.")
        installed = [n for n, a in available.items() if a]
        if installed:
            print(f"\nInstalled engines: {', '.join(installed)}")
        else:
            print("\nQuick start: pip install tts-compare[kokoro]")
        sys.exit(1)

    # Create engine
    engine_kwargs = {"device": args.device}

    if args.engine == "chatterbox":
        engine_kwargs["model"] = args.model
    elif args.engine == "qwen3":
        engine_kwargs["model_size"] = args.size
        engine_kwargs["model_type"] = "Base"  # Voice cloning requires Base model

    engine = get_engine(args.engine, **engine_kwargs)

    # Get text
    if args.input:
        with open(args.input, "r", encoding="utf-8") as f:
            text = f.read()
    elif args.text:
        text = args.text
    else:
        parser.error("Either --text or --input is required")

    print(f"Engine: {args.engine}")
    print(f"Reference: {args.reference}")
    print(f"Text length: {len(text)} characters")

    # Clone voice
    clone_kwargs = {
        "text": text,
        "reference_audio": args.reference,
    }

    if args.engine == "qwen3":
        clone_kwargs["reference_text"] = args.ref_text
        clone_kwargs["language"] = args.language
    elif args.engine == "chatterbox":
        clone_kwargs["exaggeration"] = args.exaggeration

    audio, sr = engine.clone_voice(**clone_kwargs)

    # Save output
    sf.write(args.output, audio, sr)

    duration = len(audio) / sr
    print(f"Generated {duration:.2f}s of audio with cloned voice")
    print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()
