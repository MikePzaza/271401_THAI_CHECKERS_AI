"""V4 data and physical actions tested with fake robot sockets only."""
import json
import math
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import cv2
import numpy as np
import mg400_visionV4 as v
sys.path.insert(0, str(Path(__file__).resolve().parent))
import _synth
from engine import legal_moves
from game_logic import AI
from robot_io import LiveRunner, RobotPlanner, RobotFailure, moves_to_transfers
from vision_io import BoardCamera
from physical_match import PhysicalMatch, MatchError, king_signature

ROOT = Path(__file__).resolve().parents[1]


class FakeCamera:
    verified_board = True
    def __init__(self):
        self.z_plane = float(np.load(_synth.FIX / 'calib.npz')['z_plane'])
        self.A = _synth.A
        reserve = v.load_stock()['orange'][0]
        self.last_robot_objects = [(reserve[0], reserve[1], reserve[2], reserve[3], 'orange', True)]
    def xy(self, n): return tuple(map(float, v.square_xy(self.A, n)))


class FakeRobot:
    def __init__(self, ip):
        self.ip = ip
        self.moves = []
        self.suction = False
        self.closed = False
    def d(self, cmd):
        if cmd == 'RobotMode()': return '0,{5},RobotMode();'
        if cmd.startswith('DI('): return f'0,{{{1 if self.suction else 0}}},{cmd};'
        self.moves.append(cmd)
        return f'0,{{}}, {cmd};'
    def m(self, cmd):
        self.moves.append(cmd)
        return f'0,{{}}, {cmd};'
    def vacuum_on(self): self.suction = True; self.moves.append('vacuum_on')
    def release(self): self.suction = False; self.moves.append('release')
    def pump_off(self): self.suction = False; self.moves.append('pump_off')
    def close(self): self.closed = True


def approved_config():
    return {
      **{k:True for k in ('approved_for_this_setup', 'emergency_stop_hardware_tested',
         'human_exclusion_guard_tested','vacuum_sensor_tested',
         'board_alignment_physically_verified','tool_clearance_tested',
         'physical_king_identification_verified')},
      'vacuum_sensor_di':1, 'sensor_active_value':1,
      'tool_r':0.0, 'capture_discard_pose': None
    }


class V4Integration(unittest.TestCase):
    def test_calib_clear_discard_roi_stock_files(self):
        self.assertEqual(np.load(_synth.FIX/'calib.npz')['H'].shape,(3,3))
        self.assertEqual(_synth.A.shape,(2,3))
        self.assertEqual(np.load(_synth.FIX/'stock_pos.npz')['orange_xy'].shape,(4,2))
        self.assertEqual(np.load(_synth.FIX/'stock_pos.npz')['black_xy'].shape,(4,2))
        self.assertTrue(v.load_stock())
        d=v.load_discard();c=np.load(_synth.FIX/'clear_pos.npz')['pos']
        self.assertFalse(v.in_range(*d[:2]))  # V7 used to incorrectly reject this
        self.assertTrue(v.stock_reach_ok(*d[:2]))
        self.assertTrue(v.stock_reach_ok(*c[:2]))

    def test_v4_detects_king_dot_in_board_view(self):
        cam=BoardCamera()
        img=np.full((720,1280,3),240,np.uint8)
        pxy=cam.xy(49)
        p=cam.Hinv @ np.array([*pxy,1.0]);p/=p[2]
        u,w=map(int,np.rint(p[:2]))
        cv2.circle(img,(u,w),20,(0,135,250),-1)
        cv2.circle(img,(u,w),9,(0,180,0),-1)
        detected=cam.observe(img)
        self.assertEqual(detected.get(49),'orange')
        self.assertEqual(cam.last_kings,{49})

    def test_physical_match_rejects_wrong_king_marker(self):
        match=PhysicalMatch()
        with self.assertRaises(MatchError):
            match.start({},kings={49})
        from physical_match import signature
        match.start(signature(match.game.board),kings=set())
        with self.assertRaises(MatchError):
            match.require_board_ready(signature(match.game.board),kings={49})

    def test_crown_and_discard_are_planned_and_run_offline(self):
        b=[[0]*8 for _ in range(8)];b[6][1]=-1;b[2][1]=1
        b=tuple(tuple(x) for x in b)
        move=next(m for m in legal_moves(b,AI) if m.end[0]==7)
        camera=FakeCamera()
        planner=RobotPlanner(camera)
        steps=planner.plan(move,b)
        self.assertEqual([s.kind for s in steps],['piece','king_remove','king_place'])
        self.assertTrue(planner.king_source)
        with tempfile.TemporaryDirectory() as td:
            cfg=Path(td)/'safety.json'
            cfg.write_text(json.dumps(approved_config()),encoding='utf-8')
            runner=LiveRunner(cfg)
            robots=[]
            def factory(ip):
                r=FakeRobot(ip); robots.append(r); return r
            with patch('robot_io.v.MG400', side_effect=factory), patch('robot_io.time.sleep',return_value=None):
                result=runner.run(steps,planner,report=lambda s:None,stop=lambda:False)
            self.assertIn('AWAIT CAMERA',result)
            robot=robots[0]
            self.assertTrue(robot.closed)
            self.assertEqual(robot.moves.count('vacuum_on'),3)
            self.assertEqual(robot.moves.count('release'),3)
            discard=v.load_discard()
            self.assertIn(f'MovL({discard[0]:.1f},{discard[1]:.1f}', '\n'.join(robot.moves))
            self.assertFalse(robot.suction)

    def test_locked_without_site_review(self):
        with patch('robot_io.v.MG400') as mock:
            with self.assertRaises(RobotFailure):
                LiveRunner(ROOT/'live_safety_config.example.json')
            mock.assert_not_called()

    def test_king_supply_fails_closed_if_not_seen(self):
        cam=FakeCamera()
        cam.last_robot_objects=[]
        b=[[0]*8 for _ in range(8)];b[6][1]=-1;b[2][1]=1
        b=tuple(tuple(x) for x in b)
        move=next(m for m in legal_moves(b,AI) if m.end[0]==7)
        with self.assertRaisesRegex(RobotFailure,'กล้องไม่ยืนยันฮอส'):
            RobotPlanner(cam).plan(move,b)


if __name__=='__main__': unittest.main()


class KingFlicker(unittest.TestCase):
    def test_king_flag_survives_single_frame_dropout(self):
        cam = BoardCamera()
        def frame(dot):
            img = np.full((720, 1280, 3), 240, np.uint8)
            p = cam.Hinv @ np.array([*cam.xy(49), 1.0]); p /= p[2]
            u, w = map(int, np.rint(p[:2]))
            cv2.circle(img, (u, w), 20, (0, 135, 250), -1)
            if dot: cv2.circle(img, (u, w), 9, (0, 180, 0), -1)
            return img
        seq = [True, True, False, True, True]
        out = []
        for d in seq:
            cam.observe(frame(d)); out.append(frozenset(cam.last_kings))
        self.assertEqual(out[-1], {49})
        self.assertEqual(out[2], {49})    # เฟรมที่จุดเขียวหายชั่วคราวยังนับเป็นฮอส


class BoardcalDiagnostics(unittest.TestCase):
    """boardcal: ไม่กรองด้วย X/Y_RANGE, ไม่นับฮอสสำรอง, ถ้านับไม่ครบต้องรายงานชัด"""
    def _run(self, drop=0, outside=False):
        import io, contextlib, os, tempfile
        from unittest.mock import MagicMock
        sq = lambda r, c: (150.0 + (30.0 if outside else 0) + 31.5 * r, -110.0 + 31.5 * c)   # พิกัดหุ่นของช่อง
        objs = []
        for r, color in ((0, "black"), (1, "black"), (7, "orange"), (6, "orange")):
            for c in range(8):
                if (r + c) % 2:
                    continue
                x, y = sq(r, c)
                objs.append((x, y, 100 + 40 * c, 700 - 80 * r if color == "black" else 80 + 80 * (7 - r), color, False))
        objs.append((300.0, 250.0, 10, 10, "orange", True))     # ฮอสสำรองข้างกระดาน
        objs = objs[drop:]
        cap = MagicMock(); cap.release = MagicMock()
        frame = np.full((720, 1280, 3), 200, np.uint8)
        saved = tempfile.mkdtemp()
        with patch.object(v, "load_calib", return_value=(None, 0, None)), patch.object(v, "open_camera", return_value=cap), \
             patch.object(v, "read_frame", return_value=(True, frame)), patch.object(v, "detect_robot_objs", return_value=objs), \
             patch.object(v, "BOARD_FILE", os.path.join(saved, "board.npz")), patch.object(v, "BASE_DIR", saved):
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                try:
                    v.boardcal(); err = None
                except RuntimeError as e:
                    err = str(e)
        return err, buf.getvalue(), saved

    def test_ok_even_when_outside_range(self):
        err, out, saved = self._run(outside=True)
        self.assertIsNone(err, out)
        import os
        self.assertTrue(os.path.exists(os.path.join(saved, "board.npz")))
        self.assertTrue(os.path.exists(os.path.join(saved, "boardcal_debug.png")))
        A = np.load(os.path.join(saved, "board.npz"))["A"]
        self.assertAlmostEqual(float(A[0, 2]), 31.5, places=1)      # a2: ระยะต่อแถว

    def test_missing_piece_is_reported(self):
        err, out, _saved = self._run(drop=1)
        self.assertIsNotNone(err)
        self.assertIn("เจอดำ 7", err)
        self.assertIn("กล้องเห็นวัตถุ", out)


class BoardcalRealUserData(unittest.TestCase):
    """ข้อมูลจริงที่ผู้ใช้ส่งมา: ดำ 11 / ส้ม 11 (มีตัวแปลกปลอมและฮอสสำรองที่ไม่เห็นจุดเขียว) ต้องเลือก 8+8 บนกระดานได้"""
    RAW = """black K 799 349 271.2 117.9
black K 802 395 301.7 120.3
black K 802 444 333.6 120.3
black K 802 490 363.5 120.4
black - 612 507 374.9 -8.2
black - 540 507 375.2 -57.0
black - 757 507 374.8 90.0
black - 684 507 375.2 40.5
black - 720 545 399.1 65.2
black - 648 545 399.3 16.3
black - 577 546 400.1 -31.5
black - 502 546 400.7 -82.5
black - 796 588 426.6 116.8
black - 484 592 429.9 -93.5
black - 24 689 492.0 -397.3
orange - 754 291 232.2 86.6
orange - 682 291 232.8 37.3
orange - 611 291 233.2 -12.2
orange - 540 292 233.9 -61.4
orange - 719 326 255.8 62.9
orange - 576 327 257.3 -35.5
orange - 503 328 258.0 -86.1
orange - 647 328 257.8 13.2
orange - 444 354 275.4 -126.2
orange - 445 403 307.6 -124.0
orange - 446 447 336.5 -122.7
orange K 445 495 367.5 -121.8"""

    def test_picks_board_pieces(self):
        objs = []
        for line in self.RAW.splitlines():
            c, k, u, w, x, y = line.split()
            objs.append((float(x), float(y), float(u), float(w), c, k == 'K'))
        plain = [o for o in objs if not o[5]]
        pairs = v.assign_start_squares(plain, use_range=False)
        self.assertEqual(len(pairs), 16)
        by = {(r, c): p for p, r, c in pairs}
        self.assertEqual(by[(0, 0)][2:4], (502.0, 546.0))       # ดำ ช่อง 1
        self.assertEqual(by[(7, 7)][2:4], (754.0, 291.0))       # ส้ม ช่อง 64
        self.assertLess(v._layout_error(pairs), 3.0)
        self.assertTrue(all(p[2] > 480 and p[3] < 560 for p, _r, _c in pairs))   # ไม่มีตัวนอกกระดาน
