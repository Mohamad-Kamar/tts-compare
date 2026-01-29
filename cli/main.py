"""TTS Compare - Main CLI entry point.

Unified command-line interface for TTS generation, voice cloning, and benchmarking.

Usage:
    tts engines                    # List available engines
    tts generate -e kokoro ...     # Generate speech
    tts clone -e chatterbox ...    # Clone voice
    tts benchmark ...              # Run benchmarks
"""

import sys
from pathlib import Path

import click

# Ensure the project root is in the path
sys.path.insert(0, str(Path(__file__).parent.parent))

from engines import (
    ENGINE_INFO,
    EngineNotAvailableError,
    get_available_engines,
    get_engine,
    list_engines,
)


def check_engine_available(engine_name: str) -> bool:
    """Check if engine is available and print helpful message if not.

    Returns True if available, False otherwise (after printing error).
    """
    available = get_available_engines()

    if available.get(engine_name, False):
        return True

    # Print helpful error message
    info = ENGINE_INFO.get(engine_name, {})

    click.echo()
    click.secho(f"Engine '{engine_name}' is not installed.", fg="red", bold=True)
    click.echo()
    click.echo(f"To install: {info.get('install', f'pip install tts-compare[{engine_name}]')}")

    if info.get("conflict"):
        click.echo()
        click.secho("Note: ", fg="yellow", nl=False)
        click.echo(info.get("conflict_reason", ""))
        click.echo(f"You cannot have both {engine_name} and {info['conflict']} in the same environment.")

    # Show what IS installed
    installed = [name for name, avail in available.items() if avail]
    if installed:
        click.echo()
        click.echo(f"Installed engines: {', '.join(installed)}")
    else:
        click.echo()
        click.echo("Quick start: pip install tts-compare[kokoro]")

    return False


@click.group()
@click.version_option(version="0.1.0")
def main():
    """TTS Compare - Generate speech with multiple TTS engines.

    \b
    Quick start:
      pip install tts-compare[kokoro]
      tts generate -e kokoro -t "Hello world" -o hello.wav

    \b
    For voice cloning, install one of:
      pip install tts-compare[chatterbox]  # 350-500M, fast
      pip install tts-compare[qwen3]       # 0.6B-1.7B, highest quality
    """
    pass


@main.command("engines")
def cmd_engines():
    """List available TTS engines and their installation status."""
    list_engines()


@main.command("generate")
@click.option("--engine", "-e", required=True,
              type=click.Choice(["kokoro", "chatterbox", "qwen3"]),
              help="TTS engine to use")
@click.option("--text", "-t", help="Text to synthesize")
@click.option("--input", "-i", "input_file", type=click.Path(exists=True),
              help="Input text file")
@click.option("--output", "-o", default="output.wav",
              help="Output audio file (default: output.wav)")
@click.option("--voice", "-v", help="Voice/speaker name")
@click.option("--language", "-l", default="en", help="Language code (default: en)")
@click.option("--speed", "-s", default=1.0, type=float,
              help="Speech speed (Kokoro only, default: 1.0)")
@click.option("--device", "-d", default="auto",
              help="Device (cuda/mps/cpu/auto)")
@click.option("--model", "-m", default="turbo",
              type=click.Choice(["turbo", "multilingual", "standard"]),
              help="Chatterbox model variant (default: turbo)")
@click.option("--size", default="0.6B", type=click.Choice(["0.6B", "1.7B"]),
              help="Qwen3 model size (default: 0.6B)")
@click.option("--type", "model_type", default="CustomVoice",
              type=click.Choice(["Base", "CustomVoice", "VoiceDesign"]),
              help="Qwen3 model type (default: CustomVoice)")
@click.option("--instruct", help="Voice instruction (Qwen3 VoiceDesign)")
@click.option("--8bit", "load_8bit", is_flag=True,
              help="Load model in 8-bit precision (Qwen3, ~50%% memory reduction)")
@click.option("--4bit", "load_4bit", is_flag=True,
              help="Load model in 4-bit precision (Qwen3, ~75%% memory reduction)")
@click.option("--list-voices", is_flag=True, help="List available voices and exit")
@click.option("--info", "show_info", is_flag=True, help="Show engine info and exit")
def cmd_generate(engine, text, input_file, output, voice, language, speed,
                 device, model, size, model_type, instruct, load_8bit, load_4bit,
                 list_voices, show_info):
    """Generate speech from text.

    \b
    Examples:
      tts generate -e kokoro -t "Hello world" -o hello.wav
      tts generate -e kokoro -v af_heart --speed 1.2 -t "Fast speech" -o fast.wav
      tts generate -e chatterbox --model turbo -t "Hello" -o cb.wav
      tts generate -e qwen3 --size 1.7B --type VoiceDesign --instruct "warm male" -t "Hi" -o warm.wav
      tts generate -e qwen3 --4bit -t "Memory efficient" -o efficient.wav
    """
    import soundfile as sf

    # Early availability check
    if not check_engine_available(engine):
        sys.exit(1)

    # Build engine kwargs
    engine_kwargs = {"device": device}

    if engine == "kokoro":
        if voice:
            engine_kwargs["default_voice"] = voice
        engine_kwargs["speed"] = speed
    elif engine == "chatterbox":
        engine_kwargs["model"] = model
    elif engine == "qwen3":
        engine_kwargs["model_size"] = size
        engine_kwargs["model_type"] = model_type
        if voice:
            engine_kwargs["default_speaker"] = voice
        if load_8bit:
            engine_kwargs["load_in_8bit"] = True
        if load_4bit:
            engine_kwargs["load_in_4bit"] = True

    # Load engine
    try:
        tts_engine = get_engine(engine, **engine_kwargs)
    except EngineNotAvailableError as e:
        click.echo(str(e), err=True)
        sys.exit(1)

    # Handle utility commands
    if list_voices:
        click.echo(f"Available voices for {engine}:")
        for v in tts_engine.list_voices():
            click.echo(f"  - {v}")
        return

    if show_info:
        import json
        click.echo(json.dumps(tts_engine.get_info(), indent=2))
        return

    # Get text
    if input_file:
        with open(input_file, "r", encoding="utf-8") as f:
            text = f.read()
    elif not text:
        click.echo("Error: Either --text or --input is required", err=True)
        sys.exit(1)

    click.echo(f"Engine: {engine}")
    click.echo(f"Text length: {len(text)} characters")

    # Generate
    generate_kwargs = {"language": language}

    if engine == "kokoro":
        if voice:
            generate_kwargs["voice"] = voice
        generate_kwargs["speed"] = speed
    elif engine == "qwen3":
        if voice:
            generate_kwargs["voice"] = voice
        if instruct:
            generate_kwargs["instruct"] = instruct

    audio, sr = tts_engine.generate(text, **generate_kwargs)
    sf.write(output, audio, sr)

    duration = len(audio) / sr
    click.secho(f"Generated {duration:.2f}s of audio", fg="green")
    click.echo(f"Saved to {output}")


@main.command("clone")
@click.option("--engine", "-e", required=True,
              type=click.Choice(["chatterbox", "qwen3"]),
              help="TTS engine (kokoro does not support cloning)")
@click.option("--reference", "-r", required=True, type=click.Path(exists=True),
              help="Reference audio file for voice cloning")
@click.option("--ref-text", help="Transcript of reference audio (recommended for Qwen3)")
@click.option("--text", "-t", help="Text to synthesize")
@click.option("--input", "-i", "input_file", type=click.Path(exists=True),
              help="Input text file")
@click.option("--output", "-o", default="cloned.wav",
              help="Output audio file (default: cloned.wav)")
@click.option("--language", "-l", default="en", help="Language code (default: en)")
@click.option("--device", "-d", default="auto",
              help="Device (cuda/mps/cpu/auto)")
@click.option("--model", "-m", default="turbo",
              type=click.Choice(["turbo", "multilingual", "standard"]),
              help="Chatterbox model variant (default: turbo)")
@click.option("--size", default="0.6B", type=click.Choice(["0.6B", "1.7B"]),
              help="Qwen3 model size (default: 0.6B)")
@click.option("--exaggeration", default=0.5, type=float,
              help="Exaggeration level 0-1 (Chatterbox, default: 0.5)")
@click.option("--8bit", "load_8bit", is_flag=True,
              help="Load model in 8-bit precision (Qwen3, ~50%% memory reduction)")
@click.option("--4bit", "load_4bit", is_flag=True,
              help="Load model in 4-bit precision (Qwen3, ~75%% memory reduction)")
def cmd_clone(engine, reference, ref_text, text, input_file, output, language,
              device, model, size, exaggeration, load_8bit, load_4bit):
    """Clone a voice from reference audio.

    \b
    Examples:
      tts clone -e chatterbox -r voice.wav -t "Hello in my voice" -o cloned.wav
      tts clone -e qwen3 -r voice.wav --ref-text "transcript" -t "New text" -o cloned.wav
    """
    import soundfile as sf

    # Early availability check
    if not check_engine_available(engine):
        sys.exit(1)

    # Build engine kwargs
    engine_kwargs = {"device": device}

    if engine == "chatterbox":
        engine_kwargs["model"] = model
    elif engine == "qwen3":
        engine_kwargs["model_size"] = size
        engine_kwargs["model_type"] = "Base"  # Voice cloning requires Base
        if load_8bit:
            engine_kwargs["load_in_8bit"] = True
        if load_4bit:
            engine_kwargs["load_in_4bit"] = True

    # Load engine
    try:
        tts_engine = get_engine(engine, **engine_kwargs)
    except EngineNotAvailableError as e:
        click.echo(str(e), err=True)
        sys.exit(1)

    # Get text
    if input_file:
        with open(input_file, "r", encoding="utf-8") as f:
            text = f.read()
    elif not text:
        click.echo("Error: Either --text or --input is required", err=True)
        sys.exit(1)

    click.echo(f"Engine: {engine}")
    click.echo(f"Reference: {reference}")
    click.echo(f"Text length: {len(text)} characters")

    # Clone voice
    clone_kwargs = {
        "text": text,
        "reference_audio": reference,
    }

    if engine == "qwen3":
        clone_kwargs["reference_text"] = ref_text
        clone_kwargs["language"] = language
    elif engine == "chatterbox":
        clone_kwargs["exaggeration"] = exaggeration

    audio, sr = tts_engine.clone_voice(**clone_kwargs)
    sf.write(output, audio, sr)

    duration = len(audio) / sr
    click.secho(f"Generated {duration:.2f}s of audio with cloned voice", fg="green")
    click.echo(f"Saved to {output}")


@main.command("benchmark")
@click.option("--engines", default="all",
              help="Engines to benchmark (comma-separated or 'all')")
@click.option("--output", "-o", default="benchmark_results.json",
              help="Output JSON file (default: benchmark_results.json)")
@click.option("--runs", default=3, type=int, help="Number of runs per test (default: 3)")
@click.option("--device", "-d", default="auto", help="Device (cuda/mps/cpu/auto)")
def cmd_benchmark(engines, output, runs, device):
    """Run performance benchmarks on TTS engines.

    \b
    Examples:
      tts benchmark --engines kokoro -o results.json
      tts benchmark --engines all --runs 5
    """
    import json
    from cli.benchmark import run_benchmarks

    available = get_available_engines()

    if engines == "all":
        engine_list = [name for name, avail in available.items() if avail]
    else:
        engine_list = [e.strip() for e in engines.split(",")]
        # Check availability
        for e in engine_list:
            if not available.get(e, False):
                click.secho(f"Warning: Engine '{e}' not available, skipping", fg="yellow")
        engine_list = [e for e in engine_list if available.get(e, False)]

    if not engine_list:
        click.echo("No engines available to benchmark.")
        click.echo("Install at least one: pip install tts-compare[kokoro]")
        sys.exit(1)

    click.echo(f"Benchmarking engines: {', '.join(engine_list)}")
    click.echo(f"Runs per test: {runs}")

    # Standard test texts
    texts = {
        "short": "Hello, this is a test.",
        "medium": "The quick brown fox jumps over the lazy dog. This sentence contains every letter.",
    }

    results = run_benchmarks(
        engines=engine_list,
        texts=texts,
        num_runs=runs,
        warmup_runs=1,
        qwen3_sizes=["0.6B"],
        chatterbox_models=["turbo"],
        device=device,
    )

    # Save results
    if results:
        with open(output, "w") as f:
            json.dump(results, f, indent=2)
        click.secho(f"\nResults saved to: {output}", fg="green")
    else:
        click.echo("No benchmark results generated.")


if __name__ == "__main__":
    main()
