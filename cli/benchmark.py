"""CLI tool for benchmarking TTS engines.

Usage:
    python -m cli.benchmark --engines all --output results.json
    python -m cli.benchmark --engines kokoro,chatterbox --runs 5
    python -m cli.benchmark --engines qwen3 --sizes 0.6B,1.7B
"""

import argparse
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List

import numpy as np


def run_benchmarks(
    engines: List[str],
    texts: Dict[str, str],
    num_runs: int,
    warmup_runs: int,
    qwen3_sizes: List[str],
    chatterbox_models: List[str],
    device: str,
) -> List[Dict]:
    """Run benchmarks on specified engines."""
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from engines import get_engine, get_available_engines, EngineNotAvailableError
    from utils.metrics import get_memory_usage, calculate_rtf

    # Filter to only available engines
    available = get_available_engines()
    engines = [e for e in engines if available.get(e, False)]
    if not engines:
        print("No engines available to benchmark.")
        return []

    results = []

    for engine_name in engines:
        print(f"\n{'='*50}")
        print(f"Benchmarking: {engine_name}")
        print(f"{'='*50}")

        # Handle multiple model variants
        if engine_name == "qwen3":
            variants = [{"model_size": s, "model_type": "CustomVoice"} for s in qwen3_sizes]
        elif engine_name == "chatterbox":
            variants = [{"model": m} for m in chatterbox_models]
        else:
            variants = [{}]

        for variant in variants:
            variant_str = "_".join(f"{k}={v}" for k, v in variant.items()) if variant else "default"
            print(f"\nVariant: {variant_str}")

            try:
                engine = get_engine(engine_name, device=device, **variant)

                for text_name, text in texts.items():
                    print(f"  Text: {text_name} ({len(text)} chars)")

                    # Warmup
                    for _ in range(warmup_runs):
                        try:
                            engine.generate(text)
                        except Exception as e:
                            print(f"    Warmup failed: {e}")
                            break

                    # Benchmark runs
                    run_times = []
                    audio_durations = []

                    for run in range(num_runs):
                        mem_before = get_memory_usage()

                        start = time.perf_counter()
                        try:
                            audio, sr = engine.generate(text)
                            elapsed = time.perf_counter() - start

                            run_times.append(elapsed)
                            audio_durations.append(len(audio) / sr)

                        except Exception as e:
                            print(f"    Run {run+1} failed: {e}")
                            continue

                        mem_after = get_memory_usage()

                    if run_times:
                        avg_time = np.mean(run_times)
                        avg_duration = np.mean(audio_durations)
                        rtf = calculate_rtf(avg_time, avg_duration)

                        result = {
                            "engine": engine_name,
                            "variant": variant_str,
                            "text_name": text_name,
                            "text_length": len(text),
                            "num_runs": len(run_times),
                            "avg_generation_time_s": round(avg_time, 4),
                            "std_generation_time_s": round(np.std(run_times), 4),
                            "avg_audio_duration_s": round(avg_duration, 4),
                            "rtf": round(rtf, 4),
                            "memory_mb": round(get_memory_usage(), 2),
                            "sample_rate": sr,
                        }
                        results.append(result)

                        print(f"    Time: {avg_time:.3f}s | Audio: {avg_duration:.2f}s | RTF: {rtf:.3f}")

            except Exception as e:
                print(f"  Failed to load engine: {e}")
                continue

    return results


def main():
    parser = argparse.ArgumentParser(
        description="TTS Benchmark - Compare performance across engines",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Benchmark all engines
  python -m cli.benchmark --engines all

  # Benchmark specific engines
  python -m cli.benchmark --engines kokoro,chatterbox

  # Compare Qwen3 model sizes
  python -m cli.benchmark --engines qwen3 --sizes 0.6B,1.7B

  # More runs for better statistics
  python -m cli.benchmark --engines all --runs 5

  # Save results to file
  python -m cli.benchmark --engines all --output benchmark_results.json
        """,
    )

    parser.add_argument(
        "--engines", "-e",
        type=str,
        default="all",
        help="Engines to benchmark (comma-separated or 'all')",
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        help="Output JSON file for results",
    )
    parser.add_argument(
        "--runs", "-n",
        type=int,
        default=3,
        help="Number of benchmark runs per test (default: 3)",
    )
    parser.add_argument(
        "--warmup", "-w",
        type=int,
        default=1,
        help="Number of warmup runs (default: 1)",
    )
    parser.add_argument(
        "--sizes",
        type=str,
        default="0.6B",
        help="Qwen3 model sizes to test (comma-separated, default: 0.6B)",
    )
    parser.add_argument(
        "--models",
        type=str,
        default="turbo",
        help="Chatterbox models to test (comma-separated, default: turbo)",
    )
    parser.add_argument(
        "--device", "-d",
        type=str,
        default="auto",
        help="Device (cuda/mps/cpu/auto)",
    )
    parser.add_argument(
        "--text-only",
        type=str,
        choices=["short", "medium", "long"],
        help="Only test specific text length",
    )

    args = parser.parse_args()

    # Parse engine list
    all_engines = ["kokoro", "chatterbox", "qwen3"]
    if args.engines.lower() == "all":
        engines = all_engines
    else:
        engines = [e.strip() for e in args.engines.split(",")]
        for e in engines:
            if e not in all_engines:
                print(f"Unknown engine: {e}")
                sys.exit(1)

    # Test texts
    texts = {
        "short": "Hello, this is a test.",
        "medium": "The quick brown fox jumps over the lazy dog. This sentence contains every letter of the English alphabet and is commonly used for testing.",
        "long": """In a hole in the ground there lived a hobbit. Not a nasty, dirty, wet hole,
filled with the ends of worms and an oozy smell, nor yet a dry, bare, sandy hole
with nothing in it to sit down on or to eat: it was a hobbit-hole, and that means comfort.
It had a perfectly round door like a porthole, painted green, with a shiny yellow brass knob in the exact middle.""",
    }

    if args.text_only:
        texts = {args.text_only: texts[args.text_only]}

    # Parse model variants
    qwen3_sizes = [s.strip() for s in args.sizes.split(",")]
    chatterbox_models = [m.strip() for m in args.models.split(",")]

    print("TTS Benchmark")
    print("=" * 50)
    print(f"Engines: {engines}")
    print(f"Runs per test: {args.runs}")
    print(f"Warmup runs: {args.warmup}")
    print(f"Device: {args.device}")
    if "qwen3" in engines:
        print(f"Qwen3 sizes: {qwen3_sizes}")
    if "chatterbox" in engines:
        print(f"Chatterbox models: {chatterbox_models}")

    # Run benchmarks
    results = run_benchmarks(
        engines=engines,
        texts=texts,
        num_runs=args.runs,
        warmup_runs=args.warmup,
        qwen3_sizes=qwen3_sizes,
        chatterbox_models=chatterbox_models,
        device=args.device,
    )

    # Print summary
    print("\n" + "=" * 50)
    print("SUMMARY")
    print("=" * 50)

    if results:
        # Group by engine
        by_engine = {}
        for r in results:
            key = f"{r['engine']}_{r['variant']}"
            if key not in by_engine:
                by_engine[key] = []
            by_engine[key].append(r)

        for key, engine_results in by_engine.items():
            avg_rtf = np.mean([r["rtf"] for r in engine_results])
            print(f"{key}: Average RTF = {avg_rtf:.3f}")

    # Save results
    if args.output:
        output_path = Path(args.output)
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to: {args.output}")


if __name__ == "__main__":
    main()
