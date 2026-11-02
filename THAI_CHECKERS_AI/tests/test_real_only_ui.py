import unittest
from pathlib import Path
from unittest.mock import patch
from robot_io import LiveRunner, RobotFailure

ROOT = Path(__file__).resolve().parents[1]

class RealOnlyUITests(unittest.TestCase):
    def test_main_only_uses_real_runner(self):
        source = (ROOT / "main.py").read_text(encoding="utf-8")
        self.assertNotIn("SimulationRunner", source)
        self.assertNotIn("robot_mode", source)
        self.assertIn("LiveRunner(ROOT / 'live_safety_config.json')", source)
        # The dashboard may open with no board.npz, but real motion remains locked.
        self.assertIn("BoardCamera(require_board_file=False)", source)
        self.assertIn("if not (ROOT / 'board.npz').is_file():", source)
        self.assertIn("LiveRunner(ROOT / 'live_safety_config.json')", source)

    def test_missing_site_review_fails_closed(self):
        with patch('robot_io.Path.is_file', return_value=False):
            with self.assertRaises(RobotFailure):
                LiveRunner(ROOT / 'live_safety_config.json')

if __name__ == '__main__':
    unittest.main()
