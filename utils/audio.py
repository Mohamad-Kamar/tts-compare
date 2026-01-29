"""Audio utility functions."""

from pathlib import Path
from typing import Optional, Tuple, Union

import numpy as np
import soundfile as sf


def load_audio(
    path: Union[str, Path],
    target_sr: Optional[int] = None,
) -> Tuple[np.ndarray, int]:
    """Load audio from file.

    Args:
        path: Path to audio file
        target_sr: Target sample rate (resamples if different)

    Returns:
        Tuple of (audio_array, sample_rate)
    """
    audio, sr = sf.read(str(path))

    # Convert stereo to mono if needed
    if len(audio.shape) > 1:
        audio = audio.mean(axis=1)

    # Resample if needed
    if target_sr is not None and sr != target_sr:
        audio = resample_audio(audio, sr, target_sr)
        sr = target_sr

    return audio.astype(np.float32), sr


def save_audio(
    audio: np.ndarray,
    path: Union[str, Path],
    sample_rate: int,
    normalize: bool = True,
) -> None:
    """Save audio to file.

    Args:
        audio: Audio array
        path: Output path
        sample_rate: Sample rate
        normalize: Normalize audio to prevent clipping
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if normalize:
        max_val = np.abs(audio).max()
        if max_val > 0:
            audio = audio / max_val * 0.95

    sf.write(str(path), audio, sample_rate)


def get_audio_duration(
    audio: Union[np.ndarray, str, Path],
    sample_rate: Optional[int] = None,
) -> float:
    """Get audio duration in seconds.

    Args:
        audio: Audio array or path to audio file
        sample_rate: Sample rate (required if audio is array)

    Returns:
        Duration in seconds
    """
    if isinstance(audio, (str, Path)):
        audio, sample_rate = load_audio(audio)

    if sample_rate is None:
        raise ValueError("sample_rate required when audio is array")

    return len(audio) / sample_rate


def resample_audio(
    audio: np.ndarray,
    orig_sr: int,
    target_sr: int,
) -> np.ndarray:
    """Resample audio to target sample rate.

    Args:
        audio: Input audio array
        orig_sr: Original sample rate
        target_sr: Target sample rate

    Returns:
        Resampled audio array
    """
    if orig_sr == target_sr:
        return audio

    try:
        import librosa
        return librosa.resample(audio, orig_sr=orig_sr, target_sr=target_sr)
    except ImportError:
        # Fallback to simple linear interpolation
        duration = len(audio) / orig_sr
        new_length = int(duration * target_sr)
        indices = np.linspace(0, len(audio) - 1, new_length)
        return np.interp(indices, np.arange(len(audio)), audio).astype(np.float32)


def concatenate_audio(
    audio_segments: list,
    silence_duration: float = 0.0,
    sample_rate: int = 24000,
) -> np.ndarray:
    """Concatenate multiple audio segments.

    Args:
        audio_segments: List of audio arrays
        silence_duration: Duration of silence between segments (seconds)
        sample_rate: Sample rate

    Returns:
        Concatenated audio array
    """
    if not audio_segments:
        return np.array([], dtype=np.float32)

    if silence_duration > 0:
        silence = np.zeros(int(silence_duration * sample_rate), dtype=np.float32)
        segments_with_silence = []
        for i, seg in enumerate(audio_segments):
            segments_with_silence.append(seg)
            if i < len(audio_segments) - 1:
                segments_with_silence.append(silence)
        return np.concatenate(segments_with_silence)

    return np.concatenate(audio_segments)


def trim_silence(
    audio: np.ndarray,
    threshold_db: float = -40.0,
    frame_length: int = 2048,
    hop_length: int = 512,
) -> np.ndarray:
    """Trim leading and trailing silence from audio.

    Args:
        audio: Input audio array
        threshold_db: Silence threshold in dB
        frame_length: Frame length for energy calculation
        hop_length: Hop length for energy calculation

    Returns:
        Trimmed audio array
    """
    try:
        import librosa
        trimmed, _ = librosa.effects.trim(
            audio,
            top_db=-threshold_db,
            frame_length=frame_length,
            hop_length=hop_length,
        )
        return trimmed
    except ImportError:
        # Simple fallback
        threshold = 10 ** (threshold_db / 20) * np.abs(audio).max()
        above_threshold = np.abs(audio) > threshold

        if not above_threshold.any():
            return audio

        start = np.argmax(above_threshold)
        end = len(audio) - np.argmax(above_threshold[::-1])
        return audio[start:end]
