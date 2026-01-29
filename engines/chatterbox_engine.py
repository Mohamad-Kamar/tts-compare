"""Chatterbox TTS Engine implementation.

Chatterbox by Resemble AI supports both standard TTS and zero-shot voice cloning.
Available models:
- Turbo (350M params, English, fast)
- Multilingual (500M params, 23+ languages)

Standalone usage:
    python chatterbox_engine.py --input text.txt --output audio.wav
    python chatterbox_engine.py --text "Hello" --reference voice.wav --output clone.wav
"""

import gc
import logging
import os
import re
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

from .base import TTSEngine

# Module logger for production-ready output control
logger = logging.getLogger(__name__)

# Recommended chunk size from Chatterbox maintainers (GitHub Issue #191)
# Smaller chunks improve performance and reduce memory pressure
DEFAULT_CHUNK_SIZE = 300


def get_memory_gb() -> float:
    """Get current process memory usage in GB (macOS: bytes, Linux: KB)."""
    try:
        import platform
        import resource
        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        if platform.system() == 'Darwin':
            return rss / 1024 / 1024 / 1024  # bytes to GB
        return rss / 1024 / 1024  # KB to GB
    except ImportError:
        return 0.0


class ChatterboxEngine(TTSEngine):
    """Chatterbox TTS engine wrapper.

    Supports both Turbo (350M, English) and Multilingual (500M, 23+ languages) models.
    Features zero-shot voice cloning and paralinguistic tags.
    """

    name = "chatterbox"
    supports_voice_cloning = True
    sample_rate = 24000

    MODELS = {
        "turbo": {
            "class": "ChatterboxTurboTTS",
            "module": "chatterbox.tts_turbo",
            "params": "350M",
            "languages": ["en"],
            "features": ["paralinguistics"],
        },
        "multilingual": {
            "class": "ChatterboxTTS",
            "module": "chatterbox.tts",
            "params": "500M",
            "languages": [
                "ar", "zh", "cs", "nl", "en", "fr", "de", "hi", "it",
                "ja", "ko", "pl", "pt", "ru", "es", "tr", "vi",
            ],
            "features": ["multilingual"],
        },
        "standard": {
            "class": "ChatterboxTTS",
            "module": "chatterbox.tts",
            "params": "500M",
            "languages": ["en"],
            "features": ["cfg_tuning", "exaggeration"],
        },
    }

    # Paralinguistic tags supported by Turbo model
    PARALINGUISTIC_TAGS = ["[laugh]", "[chuckle]", "[cough]", "[sigh]", "[gasp]"]

    def __init__(
        self,
        device: str = "auto",
        model: str = "turbo",
        **kwargs,
    ):
        """Initialize Chatterbox engine.

        Args:
            device: Device to run on ('cuda', 'mps', 'cpu', or 'auto')
            model: Model variant ('turbo', 'multilingual', 'standard')
        """
        self._device = device if device != "auto" else self.detect_device()
        self._model_name = model
        self._model = None
        self._model_config = self.MODELS.get(model, self.MODELS["turbo"])

        # Update supported languages based on model
        self.supported_languages = self._model_config["languages"]

    def _clear_memory(self):
        """Clear memory after generation to mitigate memory leak.

        Based on Chatterbox GitHub Issue #218: The model retains intermediate
        tensors from diffusion sampling. This cleanup reduces the leak by ~70%.
        """
        # Force synchronization before clearing cache
        if self._device == "mps" and torch.backends.mps.is_available():
            torch.mps.synchronize()
            torch.mps.empty_cache()
        elif self._device == "cuda" and torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()

        # Aggressive Python garbage collection
        gc.collect()
        gc.collect()  # Second pass catches cycles

    def _check_memory_available(self, required_gb: float = 4.0) -> bool:
        """Check if sufficient memory is available for generation.

        Based on Chatterbox Issue #205 observations: large generations can consume
        massive amounts of memory. This provides early warning/failure.

        Args:
            required_gb: Minimum recommended memory in GB (default: 4GB for turbo)

        Returns:
            True if sufficient memory available

        Raises:
            MemoryError: If memory is critically low and cleanup doesn't help
        """
        try:
            import psutil
            available_gb = psutil.virtual_memory().available / (1024**3)
        except ImportError:
            # Can't check without psutil - proceed optimistically
            logger.debug("psutil not installed, skipping memory check")
            return True

        if available_gb < required_gb:
            logger.warning(
                f"Low memory: {available_gb:.1f}GB available, {required_gb:.1f}GB recommended"
            )

            # Try to free memory
            self._clear_memory()

            # Re-check after cleanup
            available_gb = psutil.virtual_memory().available / (1024**3)

            if available_gb < required_gb * 0.5:  # Still critically low
                raise MemoryError(
                    f"Insufficient memory: {available_gb:.1f}GB available, "
                    f"need at least {required_gb * 0.5:.1f}GB. "
                    "Consider using smaller chunk sizes or unloading other models."
                )

        return True

    def _chunk_text(self, text: str, max_chars: int = DEFAULT_CHUNK_SIZE) -> List[str]:
        """Split text into chunks at sentence boundaries.

        Based on Chatterbox GitHub Issue #191: Maintainers recommend generating
        smaller text chunks (~300 characters) for better performance on Mac.

        Args:
            text: Text to split
            max_chars: Maximum characters per chunk (default: 300)

        Returns:
            List of text chunks
        """
        if len(text) <= max_chars:
            return [text]

        # Split on sentence boundaries (., !, ?)
        sentences = re.split(r'(?<=[.!?])\s+', text)
        chunks = []
        current_chunk = ""

        for sentence in sentences:
            # If single sentence exceeds max, we have to include it as-is
            if len(sentence) > max_chars:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                    current_chunk = ""
                chunks.append(sentence)
            elif len(current_chunk) + len(sentence) + 1 <= max_chars:
                current_chunk += (" " if current_chunk else "") + sentence
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = sentence

        if current_chunk:
            chunks.append(current_chunk.strip())

        return chunks

    def _load_model(self):
        """Load the Chatterbox model."""
        if self._model is not None:
            return

        import importlib

        logger.info(f"Loading Chatterbox model on device: {self._device}")

        module = importlib.import_module(self._model_config["module"])
        model_class = getattr(module, self._model_config["class"])

        self._model = model_class.from_pretrained(device=self._device)
        self.sample_rate = self._model.sr

        logger.info(f"Model loaded successfully (sample rate: {self.sample_rate}Hz)")

    def _generate_chunk(
        self,
        text: str,
        audio_prompt_path: Optional[str] = None,
        exaggeration: float = 0.5,
        cfg_weight: float = 0.5,
    ) -> np.ndarray:
        """Generate audio for a single text chunk.

        Internal method that handles the actual model call.
        Uses inference_mode for better memory efficiency.
        """
        generate_kwargs = {"text": text}

        if audio_prompt_path:
            generate_kwargs["audio_prompt_path"] = audio_prompt_path

        # Model-specific parameters
        if self._model_name == "turbo":
            pass  # Turbo has simpler interface
        else:
            generate_kwargs["exaggeration"] = exaggeration
            generate_kwargs["cfg_weight"] = cfg_weight

        # Use inference_mode for better memory efficiency (no gradient tracking)
        with torch.inference_mode():
            wav = self._model.generate(**generate_kwargs)

            # Convert to numpy immediately while in inference_mode context
            if isinstance(wav, torch.Tensor):
                wav_np = wav.squeeze().cpu().numpy()
                del wav  # Explicitly delete tensor
            else:
                wav_np = wav

        return wav_np

    def generate(
        self,
        text: str,
        voice: Optional[str] = None,
        speed: float = 1.0,
        language: Optional[str] = None,
        audio_prompt_path: Optional[str] = None,
        exaggeration: float = 0.5,
        cfg_weight: float = 0.5,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
        **kwargs,
    ) -> Tuple[np.ndarray, int]:
        """Generate speech from text.

        For long texts, automatically chunks at sentence boundaries to improve
        performance and reduce memory pressure (per Chatterbox Issue #191).

        Args:
            text: Text to synthesize (can include paralinguistic tags for Turbo)
            voice: Not used directly; use audio_prompt_path for voice
            speed: Not directly supported; adjust via cfg_weight
            language: Language code (for multilingual model)
            audio_prompt_path: Reference audio for voice style
            exaggeration: Exaggeration level (0-1, standard model only)
            cfg_weight: CFG weight (0-1, affects expressiveness)
            chunk_size: Max characters per chunk (default: 300, per Issue #191)

        Returns:
            Tuple of (audio_array, sample_rate)
        """
        self._load_model()

        # Memory check based on model size (turbo=4GB, multilingual/standard=6GB)
        required_mem = 4.0 if self._model_name == "turbo" else 6.0
        self._check_memory_available(required_gb=required_mem)

        # Chunk text for better performance (Issue #191 recommendation)
        chunks = self._chunk_text(text, max_chars=chunk_size)
        num_chunks = len(chunks)

        if num_chunks > 1:
            logger.info(f"Processing {num_chunks} chunks ({len(text)} chars total)")

        audio_segments = []

        try:
            for i, chunk in enumerate(chunks):
                if num_chunks > 1:
                    mem = get_memory_gb()
                    logger.debug(f"Chunk {i + 1}/{num_chunks}: {len(chunk)} chars (mem: {mem:.1f}GB)")

                wav = self._generate_chunk(
                    text=chunk,
                    audio_prompt_path=audio_prompt_path,
                    exaggeration=exaggeration,
                    cfg_weight=cfg_weight,
                )
                audio_segments.append(wav)

                # Clear memory after each chunk (Issue #218 workaround)
                self._clear_memory()

            # Concatenate all audio segments
            if len(audio_segments) == 1:
                final_audio = audio_segments[0]
            else:
                final_audio = np.concatenate(audio_segments)

            return final_audio, self.sample_rate

        finally:
            # Always clean up segments list to prevent memory leaks on exception
            if audio_segments:
                del audio_segments
            self._clear_memory()
            gc.collect()

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
            reference_audio: Path to reference audio file (~10s recommended)
            reference_text: Not used by Chatterbox
            **kwargs: Additional generation parameters

        Returns:
            Tuple of (audio_array, sample_rate)
        """
        if not os.path.exists(reference_audio):
            raise FileNotFoundError(f"Reference audio not found: {reference_audio}")

        return self.generate(text, audio_prompt_path=reference_audio, **kwargs)

    def list_voices(self) -> List[str]:
        """List available voice options.

        Chatterbox uses reference audio for voices, not preset voices.

        Returns:
            List with informational message
        """
        return [
            "default (no reference)",
            "custom (provide reference audio via audio_prompt_path)",
        ]

    def get_info(self) -> Dict[str, Any]:
        """Get Chatterbox engine information.

        Returns:
            Engine metadata dictionary
        """
        return {
            "name": self.name,
            "model": self._model_name,
            "parameters": self._model_config["params"],
            "supports_voice_cloning": self.supports_voice_cloning,
            "supported_languages": self.supported_languages,
            "sample_rate": self.sample_rate,
            "features": self._model_config["features"],
            "paralinguistic_tags": self.PARALINGUISTIC_TAGS if self._model_name == "turbo" else [],
            "device": self._device,
            "available_models": list(self.MODELS.keys()),
        }

    def switch_model(self, model: str):
        """Switch to a different Chatterbox model variant.

        Args:
            model: Model name ('turbo', 'multilingual', 'standard')
        """
        if model not in self.MODELS:
            raise ValueError(f"Unknown model: {model}. Available: {list(self.MODELS.keys())}")

        if model != self._model_name:
            # Clear GPU memory before releasing model to prevent leaks
            if self._model is not None:
                self._clear_memory()
                del self._model

            self._model_name = model
            self._model_config = self.MODELS[model]
            self.supported_languages = self._model_config["languages"]
            self._model = None
            self._clear_memory()  # Final cleanup pass

    def unload_model(self) -> None:
        """Explicitly unload the model to free GPU/MPS memory.

        Useful for long-running processes that need to release memory between
        operations. The model will be automatically reloaded on the next
        generate() call.

        This addresses Chatterbox Issue #205 where models remain in memory
        indefinitely without an explicit release mechanism.
        """
        if self._model is not None:
            logger.info("Unloading Chatterbox model to free memory")

            # Clear any pending GPU operations first
            self._clear_memory()

            # Move to CPU first - helps with MPS memory release
            try:
                if hasattr(self._model, 'to'):
                    self._model.cpu()
            except Exception:
                pass  # Ignore errors during CPU transfer

            # Delete the model
            del self._model
            self._model = None

            # Aggressive cleanup
            self._clear_memory()

            logger.info("Model unloaded successfully")

    def __enter__(self):
        """Context manager entry - load model."""
        self._load_model()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - unload model to free memory."""
        self.unload_model()
        return False


def main():
    """Standalone CLI for Chatterbox TTS."""
    import argparse
    import torchaudio as ta

    parser = argparse.ArgumentParser(description="Chatterbox TTS - Standalone Generator")
    parser.add_argument("--input", "-i", type=str, help="Input text file")
    parser.add_argument("--text", "-t", type=str, help="Direct text input")
    parser.add_argument("--output", "-o", type=str, default="output.wav", help="Output WAV file")
    parser.add_argument("--reference", "-r", type=str, help="Reference audio for voice cloning")
    parser.add_argument("--model", "-m", type=str, default="turbo",
                        choices=["turbo", "multilingual", "standard"],
                        help="Model variant")
    parser.add_argument("--device", "-d", type=str, default="auto", help="Device (cuda/mps/cpu/auto)")
    parser.add_argument("--exaggeration", type=float, default=0.5, help="Exaggeration level (0-1)")
    parser.add_argument("--cfg-weight", type=float, default=0.5, help="CFG weight (0-1)")
    parser.add_argument("--info", action="store_true", help="Show model info")

    args = parser.parse_args()

    engine = ChatterboxEngine(device=args.device, model=args.model)

    if args.info:
        import json
        print(json.dumps(engine.get_info(), indent=2))
        return

    # Get text from file or argument
    if args.input:
        with open(args.input, "r", encoding="utf-8") as f:
            text = f.read()
    elif args.text:
        text = args.text
    else:
        parser.error("Either --input or --text is required")

    print(f"Using model: {args.model}")

    if args.reference:
        print(f"Cloning voice from: {args.reference}")
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

    # Save using torchaudio
    ta.save(args.output, torch.tensor(audio).unsqueeze(0), sr)
    print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()
