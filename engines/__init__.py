"""TTS Engine Registry with lazy loading and graceful degradation.

This module provides a registry of TTS engines that are loaded lazily,
allowing the package to work even when some engines aren't installed.

Usage:
    from engines import get_engine, get_available_engines, list_engines

    # Check what's available
    available = get_available_engines()
    list_engines()  # Prints formatted status

    # Get an engine (raises EngineNotAvailableError if not installed)
    engine = get_engine("kokoro")
    audio, sr = engine.generate("Hello, world!")
"""

from typing import TYPE_CHECKING, Dict, Optional, Type

from .base import TTSEngine, EngineNotAvailableError

if TYPE_CHECKING:
    from .kokoro_engine import KokoroEngine
    from .chatterbox_engine import ChatterboxEngine
    from .qwen3_engine import Qwen3Engine


# Engine metadata registry
# This defines all known engines and their properties
ENGINE_INFO: Dict[str, Dict] = {
    "kokoro": {
        "package": "kokoro",
        "install": "pip install tts-compare[kokoro]",
        "description": "Lightweight 82M model, fast, CPU-friendly",
        "voice_cloning": False,
        "conflict": None,
        "conflict_reason": None,
    },
    "chatterbox": {
        "package": "chatterbox",
        "install": "pip install tts-compare[chatterbox]",
        "description": "350-500M model, voice cloning, paralinguistic tags",
        "voice_cloning": True,
        "conflict": "qwen3",
        "conflict_reason": "Chatterbox requires transformers==4.46.3, Qwen3 requires transformers==4.57.3",
    },
    "qwen3": {
        "package": "qwen_tts",
        "install": "pip install tts-compare[qwen3]",
        "description": "0.6B-1.7B model, highest quality, voice cloning, voice design",
        "voice_cloning": True,
        "conflict": "chatterbox",
        "conflict_reason": "Qwen3 requires transformers==4.57.3, Chatterbox requires transformers==4.46.3",
    },
}


def _check_engine_available(name: str) -> bool:
    """Check if an engine's dependencies are available.

    Args:
        name: Engine name from ENGINE_INFO

    Returns:
        True if the engine can be imported, False otherwise
    """
    info = ENGINE_INFO.get(name, {})
    package = info.get("package", name)

    try:
        __import__(package)
        return True
    except ImportError:
        return False


def get_available_engines() -> Dict[str, bool]:
    """Return dict of engine names to availability status.

    Returns:
        Dictionary mapping engine names to True/False availability

    Example:
        >>> get_available_engines()
        {'kokoro': True, 'chatterbox': False, 'qwen3': False}
    """
    return {name: _check_engine_available(name) for name in ENGINE_INFO}


def list_engines(print_output: bool = True) -> str:
    """Print or return formatted status of all engines.

    Args:
        print_output: If True, print to stdout. If False, return string.

    Returns:
        Formatted string showing engine status
    """
    available = get_available_engines()

    lines = [
        "",
        "TTS Engines:",
        "-" * 60,
    ]

    for name, info in ENGINE_INFO.items():
        is_available = available[name]
        symbol = "[+]" if is_available else "[ ]"
        status = "INSTALLED" if is_available else "not installed"

        lines.append(f"{symbol} {name:12} - {info['description']}")

        if not is_available:
            lines.append(f"                  Install: {info['install']}")

        if info.get("conflict"):
            conflict_note = f"(conflicts with {info['conflict']})"
            if is_available:
                lines.append(f"                  {conflict_note}")

    lines.append("-" * 60)

    # Add quick start hint if nothing is installed
    if not any(available.values()):
        lines.append("")
        lines.append("Quick start: pip install tts-compare[kokoro]")

    output = "\n".join(lines)

    if print_output:
        print(output)

    return output


def get_engine(name: str, **kwargs) -> TTSEngine:
    """Factory function to get a TTS engine by name.

    This function lazily loads the engine module only when requested,
    and provides helpful error messages if the engine isn't installed.

    Args:
        name: Engine name ('kokoro', 'chatterbox', 'qwen3')
        **kwargs: Engine-specific configuration

    Returns:
        TTSEngine instance

    Raises:
        EngineNotAvailableError: If engine dependencies are not installed
        ValueError: If engine name is unknown

    Example:
        >>> engine = get_engine("kokoro")
        >>> audio, sr = engine.generate("Hello, world!")

        >>> engine = get_engine("chatterbox", model="turbo")
        >>> audio, sr = engine.clone_voice("Hello", reference_audio="voice.wav")
    """
    name = name.lower()

    if name not in ENGINE_INFO:
        available_names = list(ENGINE_INFO.keys())
        raise ValueError(f"Unknown engine: {name}. Available engines: {available_names}")

    info = ENGINE_INFO[name]

    # Lazy import with helpful error
    try:
        if name == "kokoro":
            from .kokoro_engine import KokoroEngine
            return KokoroEngine(**kwargs)
        elif name == "chatterbox":
            from .chatterbox_engine import ChatterboxEngine
            return ChatterboxEngine(**kwargs)
        elif name == "qwen3":
            from .qwen3_engine import Qwen3Engine
            return Qwen3Engine(**kwargs)
    except ImportError as e:
        raise EngineNotAvailableError(
            engine_name=name,
            install_cmd=info["install"],
            description=info["description"],
            conflict=info.get("conflict"),
            conflict_reason=info.get("conflict_reason"),
            original_error=e,
        ) from e


# Only export what users need - don't import engine classes at module level
__all__ = [
    "TTSEngine",
    "EngineNotAvailableError",
    "ENGINE_INFO",
    "get_engine",
    "get_available_engines",
    "list_engines",
]
