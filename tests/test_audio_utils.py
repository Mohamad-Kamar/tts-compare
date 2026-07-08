import tempfile
import unittest
from pathlib import Path

import numpy as np

from utils.audio import get_audio_duration, save_audio


class TestAudioUtils(unittest.TestCase):
    def test_save_audio_creates_parent_directories(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "nested" / "clip.wav"
            save_audio(np.zeros(2400, dtype=np.float32), path, 24000, normalize=False)
            self.assertTrue(path.exists())

    def test_get_audio_duration_from_array(self) -> None:
        audio = np.zeros(24000, dtype=np.float32)
        duration = get_audio_duration(audio, sample_rate=24000)
        self.assertAlmostEqual(duration, 1.0)


if __name__ == "__main__":
    unittest.main()
