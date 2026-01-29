"""CLI tool for text-to-speech generation.

Usage:
    python -m cli.generate --engine kokoro --text "Hello world" --output hello.wav
    python -m cli.generate --engine chatterbox --model turbo --text "Hello" --output cb.wav
    python -m cli.generate --engine qwen3 --size 1.7B --text "Hello" --voice Ryan --output qwen.wav
"""

import argparse
import sys
from pathlib import Path

import soundfile as sf


def main():
    parser = argparse.ArgumentParser(
        description="TTS Generator - Generate speech from text using various engines",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
            Examples:
            # Generate with Kokoro
            python -m cli.generate -e kokoro -t "Hello world" -o hello.wav

            # Generate with Chatterbox Turbo
            python -m cli.generate -e chatterbox --model turbo -t "Hello" -o cb.wav

            # Generate with Chatterbox Multilingual (French)
            python -m cli.generate -e chatterbox --model multilingual -t "Bonjour" -l fr -o bonjour.wav

            # Generate with Qwen3 CustomVoice
            python -m cli.generate -e qwen3 --size 0.6B -t "Hello" --voice Ryan -o qwen.wav

            # Generate with Qwen3 VoiceDesign
            python -m cli.generate -e qwen3 --size 1.7B --type VoiceDesign --instruct "warm male voice" -t "Hello" -o warm.wav

            # Generate from file
            python -m cli.generate -e kokoro -i input.txt -o output.wav
        """,
        )

    # Common arguments
    parser.add_argument(
        "--engine", "-e",
        type=str,
        required=True,
        choices=["kokoro", "chatterbox", "qwen3"],
        help="TTS engine to use",
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
        default="output.wav",
        help="Output audio file (default: output.wav)",
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

    # Kokoro-specific
    parser.add_argument(
        "--voice", "-v",
        type=str,
        help="Voice ID (Kokoro) or Speaker name (Qwen3)",
    )
    parser.add_argument(
        "--speed", "-s",
        type=float,
        default=1.0,
        help="Speech speed (Kokoro only, default: 1.0)",
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
        "--reference", "-r",
        type=str,
        help="Reference audio for voice cloning",
    )
    parser.add_argument(
        "--exaggeration",
        type=float,
        default=0.5,
        help="Exaggeration level 0-1 (Chatterbox, default: 0.5)",
    )
    parser.add_argument(
        "--cfg-weight",
        type=float,
        default=0.5,
        help="CFG weight 0-1 (Chatterbox, default: 0.5)",
    )

    # Qwen3-specific
    parser.add_argument(
        "--size",
        type=str,
        default="0.6B",
        choices=["0.6B", "1.7B"],
        help="Qwen3 model size (default: 0.6B)",
    )
    parser.add_argument(
        "--type",
        type=str,
        default="CustomVoice",
        choices=["Base", "CustomVoice", "VoiceDesign"],
        help="Qwen3 model type (default: CustomVoice)",
    )
    parser.add_argument(
        "--instruct",
        type=str,
        help="Voice instruction (Qwen3 CustomVoice/VoiceDesign)",
    )

    # Utility arguments
    parser.add_argument(
        "--list-voices",
        action="store_true",
        help="List available voices and exit",
    )
    parser.add_argument(
        "--info",
        action="store_true",
        help="Show engine info and exit",
    )

    args = parser.parse_args()

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

    # Create engine based on selection
    engine_kwargs = {"device": args.device}

    if args.engine == "kokoro":
        if args.voice:
            engine_kwargs["default_voice"] = args.voice
        engine_kwargs["speed"] = args.speed

    elif args.engine == "chatterbox":
        engine_kwargs["model"] = args.model

    elif args.engine == "qwen3":
        engine_kwargs["model_size"] = args.size
        engine_kwargs["model_type"] = args.type
        if args.voice:
            engine_kwargs["default_speaker"] = args.voice

    engine = get_engine(args.engine, **engine_kwargs)

    # Handle utility commands
    if args.list_voices:
        print(f"Available voices for {args.engine}:")
        for voice in engine.list_voices():
            print(f"  - {voice}")
        return

    if args.info:
        import json
        print(json.dumps(engine.get_info(), indent=2))
        return

    # Get text
    if args.input:
        with open(args.input, "r", encoding="utf-8") as f:
            text = f.read()
    elif args.text:
        text = args.text
    else:
        parser.error("Either --text or --input is required")

    print(f"Engine: {args.engine}")
    print(f"Text length: {len(text)} characters")

    # Generate audio
    generate_kwargs = {"language": args.language}

    if args.engine == "kokoro":
        if args.voice:
            generate_kwargs["voice"] = args.voice
        generate_kwargs["speed"] = args.speed

    elif args.engine == "chatterbox":
        if args.reference:
            print(f"Using reference audio: {args.reference}")
            audio, sr = engine.clone_voice(
                text,
                reference_audio=args.reference,
                exaggeration=args.exaggeration,
                cfg_weight=args.cfg_weight,
            )
        else:
            audio, sr = engine.generate(
                text,
                exaggeration=args.exaggeration,
                cfg_weight=args.cfg_weight,
            )
        sf.write(args.output, audio, sr)
        print(f"Saved to {args.output}")
        return

    elif args.engine == "qwen3":
        if args.voice:
            generate_kwargs["voice"] = args.voice
        if args.instruct:
            generate_kwargs["instruct"] = args.instruct

        if args.reference:
            print(f"Using reference audio: {args.reference}")
            audio, sr = engine.clone_voice(
                text,
                reference_audio=args.reference,
                language=args.language,
            )
        else:
            audio, sr = engine.generate(text, **generate_kwargs)
        sf.write(args.output, audio, sr)
        print(f"Saved to {args.output}")
        return

    # Default generation (Kokoro)
    audio, sr = engine.generate(text, **generate_kwargs)
    sf.write(args.output, audio, sr)

    duration = len(audio) / sr
    print(f"Generated {duration:.2f}s of audio")
    print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()
