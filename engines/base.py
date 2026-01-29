"""Abstract base class for TTS engines."""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


class EngineNotAvailableError(ImportError):
    """Raised when a TTS engine is not installed.

    Provides helpful error messages with installation instructions.
    """

    def __init__(self, engine_name: str, install_cmd: str,
                 description: str = "", conflict: Optional[str] = None,
                 conflict_reason: Optional[str] = None,
                 original_error: Optional[Exception] = None):
        self.engine_name = engine_name
        self.install_cmd = install_cmd
        self.original_error = original_error

        lines = [
            "",
            "=" * 60,
            f"Engine '{engine_name}' is not installed.",
            "",
            f"To install: {install_cmd}",
        ]

        if description:
            lines.insert(3, f"({description})")

        if conflict and conflict_reason:
            lines.extend([
                "",
                f"NOTE: {conflict_reason}",
                f"You cannot have both {engine_name} and {conflict} in the same environment.",
            ])

        lines.append("=" * 60)

        super().__init__("\n".join(lines))


class TTSEngine(ABC):
    """Abstract base class for text-to-speech engines.

    All TTS engine implementations must inherit from this class and implement
    the required abstract methods.
    """

    name: str = "base"
    supports_voice_cloning: bool = False
    supported_languages: List[str] = []
    sample_rate: int = 24000

    @abstractmethod
    def __init__(self, device: str = "auto", **kwargs):
        """Initialize the TTS engine.

        Args:
            device: Device to run on ('cuda', 'mps', 'cpu', or 'auto')
            **kwargs: Engine-specific configuration
        """
        pass

    @abstractmethod
    def generate(
        self,
        text: str,
        voice: Optional[str] = None,
        speed: float = 1.0,
        language: Optional[str] = None,
        **kwargs,
    ) -> Tuple[np.ndarray, int]:
        """Generate speech from text.

        Args:
            text: Text to synthesize
            voice: Voice identifier (engine-specific)
            speed: Speech speed multiplier (1.0 = normal)
            language: Language code (engine-specific)
            **kwargs: Additional engine-specific parameters

        Returns:
            Tuple of (audio_array, sample_rate)
        """
        pass

    @abstractmethod
    def clone_voice(
        self,
        text: str,
        reference_audio: str,
        reference_text: Optional[str] = None,
        **kwargs,
    ) -> Tuple[np.ndarray, int]:
        """Generate speech using a cloned voice.

        Args:
            text: Text to synthesize
            reference_audio: Path to reference audio file
            reference_text: Transcript of reference audio (if required)
            **kwargs: Additional engine-specific parameters

        Returns:
            Tuple of (audio_array, sample_rate)

        Raises:
            NotImplementedError: If engine doesn't support voice cloning
        """
        pass

    @abstractmethod
    def list_voices(self) -> List[str]:
        """List available voices for this engine.

        Returns:
            List of voice identifiers
        """
        pass

    @abstractmethod
    def get_info(self) -> Dict[str, Any]:
        """Get information about this engine.

        Returns:
            Dictionary containing engine metadata
        """
        pass

    def is_loaded(self) -> bool:
        """Check if the model is loaded and ready.

        Returns:
            True if model is loaded
        """
        return hasattr(self, "_model") and self._model is not None

    @staticmethod
    def detect_device() -> str:
        """Auto-detect the best available device.

        Returns:
            Device string ('cuda', 'mps', or 'cpu')
        """
        import torch

        if torch.cuda.is_available():
            return "cuda"
        elif torch.backends.mps.is_available():
            return "mps"
        return "cpu"
