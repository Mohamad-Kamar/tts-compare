"""Performance and quality metrics for TTS evaluation."""

import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional, Tuple

import numpy as np


@dataclass
class LatencyResult:
    """Result of latency measurement."""

    first_token_ms: float  # Time to first audio sample
    total_ms: float  # Total generation time
    audio_duration_s: float  # Duration of generated audio
    rtf: float  # Real-time factor


@dataclass
class BenchmarkResult:
    """Result of a benchmark run."""

    engine: str
    text_length: int
    audio_duration_s: float
    generation_time_s: float
    rtf: float
    memory_mb: float
    sample_rate: int


def calculate_rtf(
    generation_time: float,
    audio_duration: float,
) -> float:
    """Calculate Real-Time Factor (RTF).

    RTF < 1 means faster than real-time.
    RTF = 1 means real-time.
    RTF > 1 means slower than real-time.

    Args:
        generation_time: Time to generate audio (seconds)
        audio_duration: Duration of generated audio (seconds)

    Returns:
        Real-time factor
    """
    if audio_duration <= 0:
        return float("inf")
    return generation_time / audio_duration


def measure_latency(
    generate_fn: Callable[[], Tuple[np.ndarray, int]],
    warmup_runs: int = 1,
) -> LatencyResult:
    """Measure TTS generation latency.

    Args:
        generate_fn: Function that generates audio, returns (audio, sample_rate)
        warmup_runs: Number of warmup runs before measurement

    Returns:
        LatencyResult with timing information
    """
    # Warmup
    for _ in range(warmup_runs):
        generate_fn()

    # Measure
    start_time = time.perf_counter()
    audio, sr = generate_fn()
    end_time = time.perf_counter()

    total_time = end_time - start_time
    audio_duration = len(audio) / sr if sr > 0 else 0

    return LatencyResult(
        first_token_ms=total_time * 1000,  # Simplified - no streaming
        total_ms=total_time * 1000,
        audio_duration_s=audio_duration,
        rtf=calculate_rtf(total_time, audio_duration),
    )


def get_memory_usage() -> float:
    """Get current memory usage in MB.

    Returns:
        Memory usage in megabytes
    """
    try:
        import torch

        if torch.cuda.is_available():
            return torch.cuda.memory_allocated() / 1024 / 1024
        elif torch.backends.mps.is_available():
            # MPS doesn't have direct memory query, use system memory
            pass
    except ImportError:
        pass

    # Fallback to process memory
    try:
        import psutil
        process = psutil.Process()
        return process.memory_info().rss / 1024 / 1024
    except ImportError:
        return 0.0


def get_memory_usage_gb() -> float:
    """Get current memory usage in GB.

    Returns:
        Memory usage in gigabytes
    """
    return get_memory_usage() / 1024


@dataclass
class MemoryProfile:
    """Result of memory profiling."""

    before_mb: float
    after_mb: float
    peak_mb: float
    delta_mb: float

    @property
    def before_gb(self) -> float:
        return self.before_mb / 1024

    @property
    def after_gb(self) -> float:
        return self.after_mb / 1024

    @property
    def peak_gb(self) -> float:
        return self.peak_mb / 1024

    @property
    def delta_gb(self) -> float:
        return self.delta_mb / 1024

    def __str__(self) -> str:
        return (
            f"Memory: {self.before_gb:.2f}GB → {self.after_gb:.2f}GB "
            f"(Δ{self.delta_gb:+.2f}GB, peak: {self.peak_gb:.2f}GB)"
        )


class MemoryProfiler:
    """Context manager for profiling memory usage.

    Usage:
        with MemoryProfiler() as profiler:
            # Do memory-intensive work
            audio, sr = engine.generate("Hello")
        print(profiler.result)
    """

    def __init__(self, gc_before: bool = True, gc_after: bool = True):
        """Initialize profiler.

        Args:
            gc_before: Run garbage collection before starting
            gc_after: Run garbage collection after finishing
        """
        self._gc_before = gc_before
        self._gc_after = gc_after
        self._before_mb = 0.0
        self._peak_mb = 0.0
        self._after_mb = 0.0
        self.result: Optional[MemoryProfile] = None

    def __enter__(self) -> "MemoryProfiler":
        import gc
        if self._gc_before:
            gc.collect()

        self._before_mb = get_memory_usage()
        self._peak_mb = self._before_mb
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        import gc

        # Capture final memory
        self._after_mb = get_memory_usage()

        if self._gc_after:
            gc.collect()
            # Re-measure after GC
            self._after_mb = get_memory_usage()

        # Track peak (simplified - just use max of before/after)
        self._peak_mb = max(self._before_mb, self._after_mb)

        self.result = MemoryProfile(
            before_mb=self._before_mb,
            after_mb=self._after_mb,
            peak_mb=self._peak_mb,
            delta_mb=self._after_mb - self._before_mb,
        )
        return False


def profile_memory(func: Callable) -> Callable:
    """Decorator to profile memory usage of a function.

    Usage:
        @profile_memory
        def my_function():
            # Memory-intensive work
            pass

        result, profile = my_function()
        print(profile)
    """
    import functools

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        with MemoryProfiler() as profiler:
            result = func(*args, **kwargs)
        return result, profiler.result

    return wrapper


def format_memory(mb: float) -> str:
    """Format memory size for display.

    Args:
        mb: Memory in megabytes

    Returns:
        Formatted string (e.g., "1.5GB" or "512MB")
    """
    if mb >= 1024:
        return f"{mb / 1024:.2f}GB"
    return f"{mb:.0f}MB"


def compare_audio_similarity(
    audio1: np.ndarray,
    audio2: np.ndarray,
    sr: int = 24000,
) -> Dict[str, float]:
    """Compare similarity between two audio samples.

    Args:
        audio1: First audio array
        audio2: Second audio array
        sr: Sample rate

    Returns:
        Dictionary with similarity metrics
    """
    metrics = {}

    # Ensure same length for comparison
    min_len = min(len(audio1), len(audio2))
    a1 = audio1[:min_len]
    a2 = audio2[:min_len]

    # Mean Squared Error
    mse = np.mean((a1 - a2) ** 2)
    metrics["mse"] = float(mse)

    # Correlation
    if np.std(a1) > 0 and np.std(a2) > 0:
        correlation = np.corrcoef(a1, a2)[0, 1]
        metrics["correlation"] = float(correlation)
    else:
        metrics["correlation"] = 0.0

    # Energy ratio
    e1 = np.sum(a1 ** 2)
    e2 = np.sum(a2 ** 2)
    if e2 > 0:
        metrics["energy_ratio"] = float(e1 / e2)
    else:
        metrics["energy_ratio"] = float("inf")

    # Try to compute mel-spectrogram similarity if librosa available
    try:
        import librosa

        # Compute mel spectrograms
        mel1 = librosa.feature.melspectrogram(y=a1, sr=sr, n_mels=80)
        mel2 = librosa.feature.melspectrogram(y=a2, sr=sr, n_mels=80)

        # Convert to dB
        mel1_db = librosa.power_to_db(mel1)
        mel2_db = librosa.power_to_db(mel2)

        # Compute cosine similarity
        mel1_flat = mel1_db.flatten()
        mel2_flat = mel2_db.flatten()
        min_len = min(len(mel1_flat), len(mel2_flat))

        dot = np.dot(mel1_flat[:min_len], mel2_flat[:min_len])
        norm1 = np.linalg.norm(mel1_flat[:min_len])
        norm2 = np.linalg.norm(mel2_flat[:min_len])

        if norm1 > 0 and norm2 > 0:
            metrics["mel_cosine_similarity"] = float(dot / (norm1 * norm2))
        else:
            metrics["mel_cosine_similarity"] = 0.0

    except ImportError:
        pass

    return metrics


def run_benchmark(
    engine: "TTSEngine",
    text: str,
    num_runs: int = 3,
    warmup_runs: int = 1,
    **generate_kwargs,
) -> BenchmarkResult:
    """Run a benchmark on a TTS engine.

    Args:
        engine: TTS engine instance
        text: Text to synthesize
        num_runs: Number of benchmark runs
        warmup_runs: Number of warmup runs
        **generate_kwargs: Additional arguments for generate()

    Returns:
        BenchmarkResult with performance metrics
    """
    # Warmup
    for _ in range(warmup_runs):
        engine.generate(text, **generate_kwargs)

    # Benchmark runs
    times = []
    audio_durations = []

    for _ in range(num_runs):
        start_mem = get_memory_usage()

        start_time = time.perf_counter()
        audio, sr = engine.generate(text, **generate_kwargs)
        end_time = time.perf_counter()

        end_mem = get_memory_usage()

        times.append(end_time - start_time)
        audio_durations.append(len(audio) / sr)

    avg_time = np.mean(times)
    avg_duration = np.mean(audio_durations)

    return BenchmarkResult(
        engine=engine.name,
        text_length=len(text),
        audio_duration_s=avg_duration,
        generation_time_s=avg_time,
        rtf=calculate_rtf(avg_time, avg_duration),
        memory_mb=get_memory_usage(),
        sample_rate=sr,
    )
