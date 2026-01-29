"""Qwen3-TTS Engine implementation.

Qwen3-TTS is a state-of-the-art TTS model with voice cloning and voice design.
Available models:
- 0.6B (lightweight, fast)
- 1.7B (high quality)

Model types:
- Base: Voice cloning capable
- CustomVoice: 9 preset voices with instruction control
- VoiceDesign: Natural language voice descriptions

Standalone usage:
    python qwen3_engine.py --input text.txt --output audio.wav --speaker Ryan
    python qwen3_engine.py --text "Hello" --reference voice.wav --ref-text "transcript" --output clone.wav
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

# Recommended chunk size for Qwen3-TTS
# MPS (Apple Silicon) needs smaller chunks due to float16 numerical instability
# CUDA/CPU can handle larger chunks
DEFAULT_CHUNK_SIZE = 500
MPS_CHUNK_SIZE = 200  # Smaller for MPS to prevent NaN/Inf errors


class Qwen3Engine(TTSEngine):
    """Qwen3-TTS engine wrapper.

    Supports multiple model sizes (0.6B, 1.7B) and types (Base, CustomVoice, VoiceDesign).
    Features voice cloning and natural language voice design.
    """

    name = "qwen3"
    supports_voice_cloning = True
    supported_languages = [
        "zh", "en", "ja", "ko", "de", "fr", "ru", "pt", "es", "it"
    ]
    sample_rate = 24000

    MODELS = {
        "0.6B": {
            "CustomVoice": "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice",
            "Base": "Qwen/Qwen3-TTS-12Hz-0.6B-Base",
        },
        "1.7B": {
            "CustomVoice": "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice",
            "Base": "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
            "VoiceDesign": "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign",
        },
    }

    # Preset speakers for CustomVoice models
    SPEAKERS = {
        "chinese": ["Vivian", "Serena", "Uncle_Fu", "Dylan", "Eric"],
        "english": ["Ryan", "Aiden"],
        "japanese": ["Ono_Anna"],
        "korean": ["Sohee"],
    }

    # Language name mapping
    LANGUAGE_NAMES = {
        "zh": "Chinese",
        "en": "English",
        "ja": "Japanese",
        "ko": "Korean",
        "de": "German",
        "fr": "French",
        "ru": "Russian",
        "pt": "Portuguese",
        "es": "Spanish",
        "it": "Italian",
    }

    def __init__(
        self,
        device: str = "auto",
        model_size: str = "0.6B",
        model_type: str = "CustomVoice",
        default_speaker: str = "Ryan",
        dtype: str = "auto",
        use_flash_attention: bool = False,
        load_in_8bit: bool = False,
        load_in_4bit: bool = False,
        **kwargs,
    ):
        """Initialize Qwen3-TTS engine.

        Args:
            device: Device to run on ('cuda', 'mps', 'cpu', or 'auto')
            model_size: Model size ('0.6B' or '1.7B')
            model_type: Model type ('Base', 'CustomVoice', 'VoiceDesign')
            default_speaker: Default speaker for CustomVoice models
            dtype: Data type ('float16', 'bfloat16', or 'auto')
            use_flash_attention: Enable flash attention (requires flash-attn, CUDA only)
            load_in_8bit: Load model in 8-bit precision (CUDA/CPU only, requires bitsandbytes>=0.49.0)
            load_in_4bit: Load model in 4-bit precision (CUDA/CPU only, requires bitsandbytes>=0.49.0)
        """
        self._device = device if device != "auto" else self.detect_device()
        self._model_size = model_size
        self._model_type = model_type
        self._default_speaker = default_speaker
        self._use_flash_attention = use_flash_attention
        self._load_in_8bit = load_in_8bit
        self._load_in_4bit = load_in_4bit
        self._model = None

        # Validate quantization options
        if load_in_8bit and load_in_4bit:
            raise ValueError("Cannot use both 8-bit and 4-bit quantization. Choose one.")

        # Validate quantization device compatibility
        # Bitsandbytes only supports CUDA and CPU, not MPS (Apple Silicon GPU)
        if (load_in_8bit or load_in_4bit) and self._device == "mps":
            raise ValueError(
                "Quantization (--4bit/--8bit) is not supported on MPS (Apple Silicon GPU). "
                "Bitsandbytes only supports CUDA and CPU. Alternatives:\n"
                "  - Use the smaller 0.6B model: --size 0.6B\n"
                "  - Run on CPU with quantization: --device cpu --4bit (slower)\n"
                "  - Use full precision on MPS (default, no flags needed)"
            )

        # Determine dtype based on device
        if dtype == "auto":
            # Use float32 on MPS to prevent numerical instability in multinomial sampling
            # Float16 causes "probability tensor contains inf/nan" errors on Apple Silicon
            self._dtype = torch.float32 if self._device == "mps" else torch.bfloat16
        else:
            self._dtype = getattr(torch, dtype)

        # Validate model configuration
        if model_size not in self.MODELS:
            raise ValueError(f"Invalid model size: {model_size}. Available: {list(self.MODELS.keys())}")

        if model_type not in self.MODELS[model_size]:
            available = list(self.MODELS[model_size].keys())
            raise ValueError(f"Invalid model type for {model_size}: {model_type}. Available: {available}")

    def _get_model_id(self) -> str:
        """Get the HuggingFace model ID for current configuration."""
        return self.MODELS[self._model_size][self._model_type]

    def _clear_memory(self):
        """Clear GPU/MPS memory cache after generation.

        Qwen3 uses transformer architecture which doesn't have the same
        memory leak issues as Chatterbox's diffusion model, but explicit
        cleanup ensures consistent memory behavior across platforms.
        """
        if self._device == "mps" and torch.backends.mps.is_available():
            torch.mps.synchronize()
            torch.mps.empty_cache()
        elif self._device == "cuda" and torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()

        gc.collect()

    def _chunk_text(self, text: str, max_chars: int = DEFAULT_CHUNK_SIZE) -> List[str]:
        """Split text into chunks at sentence boundaries.

        Long texts can cause numerical instability (NaN/Inf) during generation
        due to float16 precision limits. Chunking prevents this.

        Args:
            text: Text to split
            max_chars: Maximum characters per chunk (default: 500)

        Returns:
            List of text chunks
        """
        if len(text) <= max_chars:
            return [text]

        # Split on sentence boundaries (., !, ?, and Chinese/Japanese punctuation)
        sentences = re.split(r'(?<=[.!?。！？])\s*', text)
        chunks = []
        current_chunk = ""

        for sentence in sentences:
            # Skip empty sentences
            if not sentence.strip():
                continue

            # If single sentence exceeds max, split on commas or just include as-is
            if len(sentence) > max_chars:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                    current_chunk = ""
                # Try to split on commas for very long sentences
                if len(sentence) > max_chars * 2:
                    sub_parts = re.split(r'(?<=[,，、;；])\s*', sentence)
                    sub_chunk = ""
                    for part in sub_parts:
                        if len(sub_chunk) + len(part) + 1 <= max_chars:
                            sub_chunk += (" " if sub_chunk else "") + part
                        else:
                            if sub_chunk:
                                chunks.append(sub_chunk.strip())
                            sub_chunk = part
                    if sub_chunk:
                        chunks.append(sub_chunk.strip())
                else:
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

    def _generate_single(
        self,
        text: str,
        voice: str,
        lang_name: str,
        instruct: Optional[str] = None,
        retry_count: int = 0,
    ) -> Tuple[np.ndarray, int]:
        """Generate audio for a single text chunk.

        Internal method that handles the actual model call.
        Includes retry logic for numerical instability errors on MPS.
        """
        # Use more conservative generation parameters on MPS for stability
        gen_kwargs = {}
        if self._device == "mps":
            gen_kwargs = {"temperature": 0.7, "top_p": 0.9}

        try:
            if self._model_type == "CustomVoice":
                wavs, sr = self._model.generate_custom_voice(
                    text=text,
                    language=lang_name,
                    speaker=voice,
                    instruct=instruct or "",
                    **gen_kwargs,
                )
            elif self._model_type == "VoiceDesign":
                if not instruct:
                    instruct = "A natural, clear speaking voice"
                wavs, sr = self._model.generate_voice_design(
                    text=text,
                    language=lang_name,
                    instruct=instruct,
                    **gen_kwargs,
                )
            else:  # Base model - requires voice cloning
                raise ValueError(
                    "Base model requires reference audio. Use clone_voice() instead, "
                    "or switch to CustomVoice/VoiceDesign model type."
                )

            audio = wavs[0] if isinstance(wavs, list) else wavs

            if isinstance(audio, torch.Tensor):
                audio = audio.cpu().numpy()

            return audio, sr

        except RuntimeError as e:
            # Handle numerical instability (inf/nan in probability tensor)
            if "probability tensor contains" in str(e) and retry_count < 2:
                logger.warning(f"Numerical instability on chunk ({len(text)} chars), retrying with sub-chunks...")
                self._clear_memory()

                # Split this chunk in half and try again
                mid = len(text) // 2
                # Find a good split point (sentence or comma boundary)
                split_chars = '.!?。！？,，;；'
                best_split = mid
                for i in range(mid, min(mid + 50, len(text))):
                    if text[i] in split_chars:
                        best_split = i + 1
                        break
                for i in range(mid, max(mid - 50, 0), -1):
                    if text[i] in split_chars:
                        best_split = i + 1
                        break

                part1 = text[:best_split].strip()
                part2 = text[best_split:].strip()

                audio_parts = []
                sr = self.sample_rate
                if part1:
                    audio1, sr = self._generate_single(part1, voice, lang_name, instruct, retry_count + 1)
                    audio_parts.append(audio1)
                    self._clear_memory()
                if part2:
                    audio2, sr = self._generate_single(part2, voice, lang_name, instruct, retry_count + 1)
                    audio_parts.append(audio2)

                if len(audio_parts) == 1:
                    return audio_parts[0], sr
                return np.concatenate(audio_parts), sr
            else:
                raise

    def _load_model(self):
        """Load the Qwen3-TTS model."""
        if self._model is not None:
            return

        from qwen_tts import Qwen3TTSModel

        model_id = self._get_model_id()

        # Log quantization mode
        quant_mode = "4-bit" if self._load_in_4bit else "8-bit" if self._load_in_8bit else "none"
        logger.info(f"Loading Qwen3-TTS model: {model_id} on device: {self._device} (quantization: {quant_mode})")

        load_kwargs = {
            "dtype": self._dtype,
        }

        # Quantization support (requires bitsandbytes>=0.49.0 for Apple Silicon)
        if self._load_in_4bit:
            try:
                import bitsandbytes
                load_kwargs["load_in_4bit"] = True
                load_kwargs["bnb_4bit_compute_dtype"] = torch.float16
                logger.info("4-bit quantization enabled (75% memory reduction)")
            except ImportError:
                logger.warning("bitsandbytes not installed, falling back to full precision. "
                             "Install with: pip install bitsandbytes>=0.49.0")
        elif self._load_in_8bit:
            try:
                import bitsandbytes
                load_kwargs["load_in_8bit"] = True
                logger.info("8-bit quantization enabled (50% memory reduction)")
            except ImportError:
                logger.warning("bitsandbytes not installed, falling back to full precision. "
                             "Install with: pip install bitsandbytes>=0.49.0")

        # MPS-specific handling for Apple Silicon
        if self._device == "mps":
            load_kwargs["device_map"] = "auto"  # Let HuggingFace handle MPS placement
            # Ensure we don't use bfloat16 on MPS (not well supported)
            if self._dtype == torch.bfloat16:
                logger.warning("BFloat16 not supported on MPS, using float32 instead")
                self._dtype = torch.float32
                load_kwargs["dtype"] = torch.float32
        else:
            load_kwargs["device_map"] = self._device

            # Flash Attention only available for CUDA
            if self._use_flash_attention and self._device == "cuda":
                load_kwargs["attn_implementation"] = "flash_attention_2"
                logger.info("Flash Attention 2 enabled for faster inference")

        self._model = Qwen3TTSModel.from_pretrained(model_id, **load_kwargs)

        # Log SDPA status (PyTorch's memory-efficient attention)
        if hasattr(torch.backends, 'cuda') and hasattr(torch.backends.cuda, 'flash_sdp_enabled'):
            sdpa_flash = torch.backends.cuda.flash_sdp_enabled()
            sdpa_mem = torch.backends.cuda.mem_efficient_sdp_enabled()
            logger.debug(f"SDPA backends - Flash: {sdpa_flash}, Memory-efficient: {sdpa_mem}")

        logger.info(f"Model loaded successfully (dtype: {self._dtype})")

    def generate(
        self,
        text: str,
        voice: Optional[str] = None,
        speed: float = 1.0,
        language: Optional[str] = None,
        instruct: Optional[str] = None,
        chunk_size: Optional[int] = None,
        **kwargs,
    ) -> Tuple[np.ndarray, int]:
        """Generate speech from text.

        For long texts, automatically chunks at sentence boundaries to prevent
        numerical instability (NaN/Inf errors) that can occur with float16.

        Args:
            text: Text to synthesize
            voice: Speaker name for CustomVoice model
            speed: Not directly supported (model controls pacing)
            language: Language code (e.g., 'en', 'zh')
            instruct: Voice instruction for CustomVoice/VoiceDesign models
            chunk_size: Max characters per chunk (default: 200 on MPS, 500 on CUDA/CPU)

        Returns:
            Tuple of (audio_array, sample_rate)
        """
        self._load_model()

        voice = voice or self._default_speaker
        language = language or "en"
        lang_name = self.LANGUAGE_NAMES.get(language, "English")

        # Use smaller chunks on MPS to prevent numerical instability
        if chunk_size is None:
            chunk_size = MPS_CHUNK_SIZE if self._device == "mps" else DEFAULT_CHUNK_SIZE

        # Chunk text to prevent numerical instability with long sequences
        chunks = self._chunk_text(text, max_chars=chunk_size)
        num_chunks = len(chunks)

        if num_chunks > 1:
            logger.info(f"Processing {num_chunks} chunks ({len(text)} chars total)")

        audio_segments = []
        sr = self.sample_rate  # Default, will be updated from model

        try:
            for i, chunk in enumerate(chunks):
                if num_chunks > 1:
                    logger.debug(f"Chunk {i + 1}/{num_chunks}: {len(chunk)} chars")

                chunk_audio, sr = self._generate_single(
                    text=chunk,
                    voice=voice,
                    lang_name=lang_name,
                    instruct=instruct,
                )
                audio_segments.append(chunk_audio)

                # Clear memory after each chunk to prevent accumulation
                self._clear_memory()

            # Concatenate all audio segments
            if len(audio_segments) == 1:
                final_audio = audio_segments[0]
            else:
                final_audio = np.concatenate(audio_segments)
                logger.info(f"Generated {len(final_audio) / sr:.1f}s of audio from {num_chunks} chunks")

            return final_audio, sr

        finally:
            # Always clean up
            if audio_segments:
                del audio_segments
            self._clear_memory()
            gc.collect()

    def clone_voice(
        self,
        text: str,
        reference_audio: str,
        reference_text: Optional[str] = None,
        language: Optional[str] = None,
        chunk_size: Optional[int] = None,
        **kwargs,
    ) -> Tuple[np.ndarray, int]:
        """Generate speech using a cloned voice.

        For long texts, automatically chunks to prevent numerical instability.

        Args:
            text: Text to synthesize
            reference_audio: Path to reference audio (3+ seconds)
            reference_text: Transcript of reference audio (recommended)
            language: Language code
            chunk_size: Max characters per chunk (default: 200 on MPS, 500 on CUDA/CPU)

        Returns:
            Tuple of (audio_array, sample_rate)
        """
        if not os.path.exists(reference_audio):
            raise FileNotFoundError(f"Reference audio not found: {reference_audio}")

        # Voice cloning requires Base model
        if self._model_type != "Base":
            original_type = self._model_type
            # Clean up current model before switching
            if self._model is not None:
                self._clear_memory()
                del self._model
            self._model_type = "Base"
            self._model = None
            self._clear_memory()
            logger.info(f"Switching from {original_type} to Base model for voice cloning")

        self._load_model()

        language = language or "en"
        lang_name = self.LANGUAGE_NAMES.get(language, "English")

        # Use smaller chunks on MPS to prevent numerical instability
        if chunk_size is None:
            chunk_size = MPS_CHUNK_SIZE if self._device == "mps" else DEFAULT_CHUNK_SIZE

        # Chunk text to prevent numerical instability
        chunks = self._chunk_text(text, max_chars=chunk_size)
        num_chunks = len(chunks)

        if num_chunks > 1:
            logger.info(f"Processing {num_chunks} chunks for voice cloning ({len(text)} chars total)")

        audio_segments = []
        sr = self.sample_rate

        # Use more conservative generation parameters on MPS for stability
        gen_kwargs = {}
        if self._device == "mps":
            gen_kwargs = {"temperature": 0.7, "top_p": 0.9}

        def generate_clone_chunk(chunk_text: str, retry_count: int = 0) -> Tuple[np.ndarray, int]:
            """Generate a single clone chunk with retry on numerical instability."""
            try:
                wavs, sr = self._model.generate_voice_clone(
                    text=chunk_text,
                    language=lang_name,
                    ref_audio=reference_audio,
                    ref_text=reference_text or "",
                    **gen_kwargs,
                )
                audio = wavs[0] if isinstance(wavs, list) else wavs
                if isinstance(audio, torch.Tensor):
                    audio = audio.cpu().numpy()
                return audio, sr
            except RuntimeError as e:
                if "probability tensor contains" in str(e) and retry_count < 2:
                    logger.warning(f"Numerical instability on clone chunk ({len(chunk_text)} chars), retrying...")
                    self._clear_memory()
                    # Split and retry
                    mid = len(chunk_text) // 2
                    split_chars = '.!?。！？,，;；'
                    best_split = mid
                    for j in range(mid, min(mid + 50, len(chunk_text))):
                        if chunk_text[j] in split_chars:
                            best_split = j + 1
                            break
                    part1 = chunk_text[:best_split].strip()
                    part2 = chunk_text[best_split:].strip()
                    audio_parts = []
                    sr = self.sample_rate
                    if part1:
                        a1, sr = generate_clone_chunk(part1, retry_count + 1)
                        audio_parts.append(a1)
                        self._clear_memory()
                    if part2:
                        a2, sr = generate_clone_chunk(part2, retry_count + 1)
                        audio_parts.append(a2)
                    return np.concatenate(audio_parts) if len(audio_parts) > 1 else audio_parts[0], sr
                raise

        try:
            for i, chunk in enumerate(chunks):
                if num_chunks > 1:
                    logger.debug(f"Chunk {i + 1}/{num_chunks}: {len(chunk)} chars")

                audio, sr = generate_clone_chunk(chunk)
                audio_segments.append(audio)
                self._clear_memory()

            # Concatenate all audio segments
            if len(audio_segments) == 1:
                final_audio = audio_segments[0]
            else:
                final_audio = np.concatenate(audio_segments)
                logger.info(f"Generated {len(final_audio) / sr:.1f}s of cloned voice audio")

            return final_audio, sr

        finally:
            # Always clean up
            if audio_segments:
                del audio_segments
            self._clear_memory()
            gc.collect()

    def list_voices(self) -> List[str]:
        """List available preset speakers.

        Returns:
            List of speaker names (for CustomVoice models)
        """
        all_speakers = []
        for speakers in self.SPEAKERS.values():
            all_speakers.extend(speakers)
        return all_speakers

    def get_info(self) -> Dict[str, Any]:
        """Get Qwen3-TTS engine information.

        Returns:
            Engine metadata dictionary
        """
        quant_mode = "4-bit" if self._load_in_4bit else "8-bit" if self._load_in_8bit else "none"
        return {
            "name": self.name,
            "model_size": self._model_size,
            "model_type": self._model_type,
            "model_id": self._get_model_id(),
            "parameters": self._model_size,
            "supports_voice_cloning": self.supports_voice_cloning,
            "supported_languages": self.supported_languages,
            "sample_rate": self.sample_rate,
            "speakers": self.list_voices(),
            "speaker_categories": self.SPEAKERS,
            "device": self._device,
            "dtype": str(self._dtype),
            "quantization": quant_mode,
            "available_sizes": list(self.MODELS.keys()),
            "available_types": list(self.MODELS[self._model_size].keys()),
        }

    def switch_model(self, model_size: str = None, model_type: str = None):
        """Switch to a different model configuration.

        Args:
            model_size: New model size ('0.6B' or '1.7B')
            model_type: New model type ('Base', 'CustomVoice', 'VoiceDesign')
        """
        changed = False

        if model_size and model_size != self._model_size:
            if model_size not in self.MODELS:
                raise ValueError(f"Invalid model size: {model_size}")
            self._model_size = model_size
            changed = True

        if model_type and model_type != self._model_type:
            if model_type not in self.MODELS[self._model_size]:
                raise ValueError(f"Invalid model type for {self._model_size}: {model_type}")
            self._model_type = model_type
            changed = True

        if changed:
            # Clean up GPU/MPS memory before releasing model
            if self._model is not None:
                self._clear_memory()
                del self._model
            self._model = None
            self._clear_memory()  # Final cleanup pass

    def unload_model(self) -> None:
        """Explicitly unload the model to free GPU/MPS memory.

        Useful for long-running processes that need to release memory between
        operations. The model will be automatically reloaded on the next
        generate() or clone_voice() call.
        """
        if self._model is not None:
            logger.info("Unloading Qwen3-TTS model to free memory")

            # Clear any pending GPU operations first
            self._clear_memory()

            # Move to CPU first - helps with memory release
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
    """Standalone CLI for Qwen3-TTS."""
    import argparse
    import soundfile as sf

    parser = argparse.ArgumentParser(description="Qwen3-TTS - Standalone Generator")
    parser.add_argument("--input", "-i", type=str, help="Input text file")
    parser.add_argument("--text", "-t", type=str, help="Direct text input")
    parser.add_argument("--output", "-o", type=str, default="output.wav", help="Output WAV file")
    parser.add_argument("--reference", "-r", type=str, help="Reference audio for voice cloning")
    parser.add_argument("--ref-text", type=str, help="Transcript of reference audio")
    parser.add_argument("--speaker", "-s", type=str, default="Ryan", help="Speaker name")
    parser.add_argument("--lang", "-l", type=str, default="en", help="Language code")
    parser.add_argument("--instruct", type=str, help="Voice instruction/description")
    parser.add_argument("--size", type=str, default="0.6B", choices=["0.6B", "1.7B"],
                        help="Model size")
    parser.add_argument("--type", type=str, default="CustomVoice",
                        choices=["Base", "CustomVoice", "VoiceDesign"],
                        help="Model type")
    parser.add_argument("--device", "-d", type=str, default="auto", help="Device (cuda/mps/cpu/auto)")
    parser.add_argument("--flash-attn", action="store_true", help="Use flash attention")
    parser.add_argument("--info", action="store_true", help="Show model info")
    parser.add_argument("--list-speakers", action="store_true", help="List available speakers")

    args = parser.parse_args()

    engine = Qwen3Engine(
        device=args.device,
        model_size=args.size,
        model_type=args.type,
        default_speaker=args.speaker,
        use_flash_attention=args.flash_attn,
    )

    if args.info:
        import json
        print(json.dumps(engine.get_info(), indent=2))
        return

    if args.list_speakers:
        print("Available speakers:")
        for lang, speakers in Qwen3Engine.SPEAKERS.items():
            print(f"  {lang}:")
            for speaker in speakers:
                print(f"    - {speaker}")
        return

    # Get text from file or argument
    if args.input:
        with open(args.input, "r", encoding="utf-8") as f:
            text = f.read()
    elif args.text:
        text = args.text
    else:
        parser.error("Either --input or --text is required")

    print(f"Using model: {args.size} {args.type}")

    if args.reference:
        print(f"Cloning voice from: {args.reference}")
        audio, sr = engine.clone_voice(
            text,
            reference_audio=args.reference,
            reference_text=args.ref_text,
            language=args.lang,
        )
    else:
        audio, sr = engine.generate(
            text,
            voice=args.speaker,
            language=args.lang,
            instruct=args.instruct,
        )

    sf.write(args.output, audio, sr)
    print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()
