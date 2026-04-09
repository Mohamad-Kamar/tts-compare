import tempfile
from pathlib import Path
import unittest

import numpy as np

from utils.audio import save_audio


class TestAudioUtils(unittest.TestCase):
    def test_save_audio_creates_parent_directories(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "nested" / "clip.wav"
            save_audio(np.zeros(2400, dtype=np.float32), path, 24000, normalize=False)
            self.assertTrue(path.exists())


if __name__ == "__main__":
    unittest.main()
