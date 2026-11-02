import sys, unittest
from pathlib import Path
from unittest.mock import patch
sys.path.insert(0, str(Path(__file__).resolve().parent))
import _synth  # noqa: F401  (ตั้งไฟล์สอบเทียบจำลอง)
import numpy as np
import mg400_visionV4 as v
from vision_io import BoardCamera


class OccupancyVote(unittest.TestCase):
    def test_flickering_piece_is_kept(self):
        cam = BoardCamera()
        xs = {n: v.square_xy(cam.A, n) for n in (1, 3, 32)}
        def det(with32):
            objs = []
            for n in (1, 3):
                x, y = xs[n]; objs.append((x, y, 0, 0, 'black', False))
            if with32:
                x, y = xs[32]; objs.append((x, y, 0, 0, 'black', False))
            return objs
        # 32 อยู่ 4 จาก 5 เฟรม (กะพริบหาย 1 เฟรม) -> ต้องยังเห็นตลอด
        seq = [True, True, False, True, True]
        seen_list = []
        for flag in seq:
            with patch.object(v, 'detect_robot_objs', return_value=det(flag)):
                seen_list.append(cam.observe(None))
        self.assertTrue(all(32 in s for s in seen_list[1:]), seen_list)

    def test_gap_clears_history(self):
        cam = BoardCamera()
        x, y = v.square_xy(cam.A, 1)
        with patch.object(v, 'detect_robot_objs', return_value=[(x, y, 0, 0, 'black', False)]):
            cam.observe(None)
        cam._obs_time -= 5.0   # เว้นช่วงนาน (เช่น หุ่นทำงาน)
        with patch.object(v, 'detect_robot_objs', return_value=[]):
            self.assertEqual(cam.observe(None), {})


class SnapOffCenter(unittest.TestCase):
    def _xy(self, cam, col, row):
        return cam.A @ np.array([1.0, col, row])

    def test_piece_between_squares_snaps_to_dark(self):
        cam = BoardCamera()
        # ช่อง 32 = row 3, col 7 ; วางเยื้อง (+0.45 col, -0.5 row) คร่อมไปทางช่อง 24 (ช่องสว่าง)
        x, y = self._xy(cam, 7.45, 2.5)
        with patch.object(v, 'detect_robot_objs', return_value=[(x, y, 0, 0, 'black', False)]):
            seen = cam.observe(None)
        self.assertEqual(seen, {32: 'black'})

    def test_center_of_light_square_still_rejected(self):
        from vision_io import VisionFailure
        cam = BoardCamera()
        x, y = self._xy(cam, 7, 2)     # ช่อง 24 กลางช่อง = ช่องสว่าง
        with patch.object(v, 'detect_robot_objs', return_value=[(x, y, 0, 0, 'black', False)]):
            with self.assertRaises(VisionFailure):
                cam.observe(None)


if __name__ == '__main__':
    unittest.main()
