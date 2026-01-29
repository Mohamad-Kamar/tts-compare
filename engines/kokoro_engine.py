"""Kokoro TTS Engine implementation.

Kokoro is a lightweight 82M parameter TTS model with good quality
and fast inference. It does not support voice cloning.

Standalone usage:
    python kokoro_engine.py --input text.txt --output audio.wav --voice af_heart
"""

import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .base import TTSEngine


class KokoroEngine(TTSEngine):
    """Kokoro TTS engine wrapper.

    A lightweight 82M parameter TTS model supporting 8 languages and 54 voices.
    """

    name = "kokoro"
    supports_voice_cloning = False
    supported_languages = [
        "en-us",  # American English
        "en-gb",  # British English
        "es",     # Spanish
        "fr",     # French
        "hi",     # Hindi
        "it",     # Italian
        "ja",     # Japanese
        "pt-br",  # Portuguese (Brazil)
        "zh",     # Mandarin Chinese
    ]
    sample_rate = 24000

    # Language code mapping
    LANG_CODES = {
        "en-us": "a",
        "en-gb": "b",
        "es": "e",
        "fr": "f",
        "hi": "h",
        "it": "i",
        "ja": "j",
        "pt-br": "p",
        "zh": "z",
        "en": "a",  # Default English to American
    }

    # Sample voices by category
    VOICES = {
        "american_female": ["af_heart", "af_bella", "af_nicole", "af_sarah", "af_sky"],
        "american_male": ["am_adam", "am_michael", "am_fenrir"],
        "british_female": ["bf_emma", "bf_isabella"],
        "british_male": ["bm_george", "bm_lewis"],
    }

    def __init__(
        self,
        device: str = "auto",
        default_voice: str = "af_heart",
        default_lang: str = "en-us",
        speed: float = 1.0,
        **kwargs,
    ):
        """Initialize Kokoro engine.

        Args:
            device: Device to run on (note: Kokoro primarily uses CPU)
            default_voice: Default voice to use
            default_lang: Default language code
            speed: Default speech speed
        """
        self._device = device if device != "auto" else self.detect_device()
        self._default_voice = default_voice
        self._default_lang = default_lang
        self._default_speed = speed
        self._pipeline = None
        self._current_lang = None

        # Set MPS fallback for Apple Silicon
        if self._device == "mps":
            os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

    def _load_pipeline(self, lang_code: str):
        """Load or reload the Kokoro pipeline for a specific language."""
        from kokoro import KPipeline

        internal_code = self.LANG_CODES.get(lang_code, "a")

        if self._pipeline is None or self._current_lang != internal_code:
            self._pipeline = KPipeline(lang_code=internal_code)
            self._current_lang = internal_code

    def generate(
        self,
        text: str,
        voice: Optional[str] = None,
        speed: float = None,
        language: Optional[str] = None,
        split_pattern: str = r"\n+",
        **kwargs,
    ) -> Tuple[np.ndarray, int]:
        """Generate speech from text.

        Args:
            text: Text to synthesize
            voice: Voice identifier (e.g., 'af_heart', 'am_adam')
            speed: Speech speed multiplier
            language: Language code (e.g., 'en-us', 'ja')
            split_pattern: Regex pattern for splitting text into segments

        Returns:
            Tuple of (audio_array, sample_rate)
        """
        voice = voice or self._default_voice
        speed = speed if speed is not None else self._default_speed
        language = language or self._default_lang

        self._load_pipeline(language)

        generator = self._pipeline(
            text,
            voice=voice,
            speed=speed,
            split_pattern=split_pattern,
        )

        audio_chunks = []
        for _, _, audio in generator:
            audio_chunks.append(audio)

        if not audio_chunks:
            return np.array([], dtype=np.float32), self.sample_rate

        return np.concatenate(audio_chunks), self.sample_rate

    def clone_voice(
        self,
        text: str,
        reference_audio: str,
        reference_text: Optional[str] = None,
        **kwargs,
    ) -> Tuple[np.ndarray, int]:
        """Voice cloning is not supported by Kokoro.

        Raises:
            NotImplementedError: Always raised
        """
        raise NotImplementedError(
            "Kokoro does not support voice cloning. "
            "Use Chatterbox or Qwen3-TTS for voice cloning."
        )

    def list_voices(self) -> List[str]:
        """List all available Kokoro voices.

        Returns:
            List of voice identifiers
        """
        all_voices = []
        for voices in self.VOICES.values():
            all_voices.extend(voices)
        return sorted(all_voices)

    def get_info(self) -> Dict[str, Any]:
        """Get Kokoro engine information.

        Returns:
            Engine metadata dictionary
        """
        return {
            "name": self.name,
            "parameters": "82M",
            "supports_voice_cloning": self.supports_voice_cloning,
            "supported_languages": self.supported_languages,
            "sample_rate": self.sample_rate,
            "voices": self.list_voices(),
            "voice_categories": list(self.VOICES.keys()),
            "device": self._device,
            "default_voice": self._default_voice,
            "default_lang": self._default_lang,
        }


def main():
    """Standalone CLI for Kokoro TTS."""
    import argparse
    import soundfile as sf

    parser = argparse.ArgumentParser(description="Kokoro TTS - Standalone Generator")
    parser.add_argument("--input", "-i", type=str, help="Input text file")
    parser.add_argument("--text", "-t", type=str, help="Direct text input")
    parser.add_argument("--output", "-o", type=str, default="output.wav", help="Output WAV file")
    parser.add_argument("--voice", "-v", type=str, default="af_heart", help="Voice ID")
    parser.add_argument("--lang", "-l", type=str, default="en-us", help="Language code")
    parser.add_argument("--speed", "-s", type=float, default=1.0, help="Speech speed")
    parser.add_argument("--list-voices", action="store_true", help="List available voices")

    args = parser.parse_args()

    engine = KokoroEngine(
        default_voice=args.voice,
        default_lang=args.lang,
        speed=args.speed,
    )

    if args.list_voices:
        print("Available voices:")
        for category, voices in KokoroEngine.VOICES.items():
            print(f"  {category}:")
            for voice in voices:
                print(f"    - {voice}")
        return

    # Get text from file or argument
    if args.input:
        with open(args.input, "r", encoding="utf-8") as f:
            text = f.read()
    elif args.text:
        text = args.text
    else:
        parser.error("Either --input or --text is required")

    print(f"Generating audio with voice '{args.voice}'...")
    audio, sr = engine.generate(text)

    sf.write(args.output, audio, sr)
    print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()
