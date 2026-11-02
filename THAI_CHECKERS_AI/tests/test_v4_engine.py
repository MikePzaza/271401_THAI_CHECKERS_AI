"""motion_engine='v4': เดินหมากด้วย move_piece ของ mg400_visionV4 (ไม่ต้องมีเซนเซอร์สุญญากาศ) ทดสอบกับหุ่นจำลองเท่านั้น"""
import json, sys, tempfile, unittest
import numpy as np
from pathlib import Path
from unittest.mock import patch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import mg400_visionV4 as v
sys.path.insert(0, str(Path(__file__).resolve().parent))
import _synth
from engine import legal_moves
from game_logic import AI
from robot_io import LiveRunner, RobotPlanner, RobotFailure
sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_v4_integration import FakeCamera, FakeRobot, approved_config


def v4_config(**extra):
    cfg = approved_config()
    for k in ('vacuum_sensor_di', 'sensor_active_value', 'vacuum_sensor_tested'):
        cfg.pop(k)
    cfg.update(motion_engine='v4', **extra)
    return cfg


def run_steps(cfg, steps, planner, stop=lambda: False):
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / 'c.json'
        p.write_text(json.dumps(cfg), encoding='utf-8')
        runner = LiveRunner(p)
        robots = []
        def factory(ip):
            r = FakeRobot(ip); robots.append(r); return r
        with patch('robot_io.v.MG400', side_effect=factory), patch('mg400_visionV4.time.sleep', return_value=None), \
             patch('robot_io.time.sleep', return_value=None):
            runner.run(steps, planner, report=lambda s: None, stop=stop)
        return runner, robots[0]


def crown_setup():
    b = [[0] * 8 for _ in range(8)]; b[6][1] = -1; b[2][1] = 1
    b = tuple(tuple(x) for x in b)
    move = next(m for m in legal_moves(b, AI) if m.end[0] == 7)
    planner = RobotPlanner(FakeCamera())
    return planner, planner.plan(move, b)


class V4Engine(unittest.TestCase):
    def test_no_sensor_needed_and_crown_sequence(self):
        planner, steps = crown_setup()
        runner, robot = run_steps(v4_config(per_move_confirmation=False), steps, planner)
        self.assertFalse(runner.per_move_confirmation)
        self.assertEqual(robot.moves.count('vacuum_on'), 3)   # เดิน + ทิ้งหมากธรรมดา + หยิบฮอส
        self.assertEqual(robot.moves.count('release'), 3)
        text = '\n'.join(robot.moves)
        d = v.load_discard(); k = planner.king_source
        self.assertIn(f'MovL({d[0]:.1f},{d[1]:.1f}', text)
        self.assertIn(f'MovL({k[0]:.1f},{k[1]:.1f}', text)
        self.assertFalse(any(m.startswith('DI(') for m in robot.moves))
        self.assertTrue(robot.closed)

    def test_still_requires_site_attestations(self):
        cfg = v4_config(); cfg['emergency_stop_hardware_tested'] = False
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / 'c.json'; p.write_text(json.dumps(cfg), encoding='utf-8')
            with self.assertRaises(RobotFailure):
                LiveRunner(p)

    def test_stop_request_prevents_motion(self):
        planner, steps = crown_setup()
        with self.assertRaises(RobotFailure):
            run_steps(v4_config(), steps, planner, stop=lambda: True)

    def test_default_engine_is_sensor_and_needs_pin(self):
        cfg = approved_config(); cfg['vacuum_sensor_di'] = None
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / 'c.json'; p.write_text(json.dumps(cfg), encoding='utf-8')
            with self.assertRaises(RobotFailure):
                LiveRunner(p)


if __name__ == '__main__':
    unittest.main()


class CrownSkippedOnWin(unittest.TestCase):
    """ถึงแถวสุดท้ายแล้วชนะทันที -> ไม่หยิบฮอสมาวาง และกล้องต้องไม่เห็นฮอสที่ช่องนั้น"""
    def setUp(self):
        b = [[0] * 8 for _ in range(8)]; b[6][1] = -1      # เหลือส้มตัวเดียว ฝั่งคนไม่มีหมาก = ชนะทันที
        self.b = tuple(tuple(x) for x in b)
        self.move = next(m for m in legal_moves(self.b, AI) if m.end[0] == 7)

    def test_no_king_steps(self):
        from robot_io import moves_to_transfers
        from engine import crown_skipped
        self.assertTrue(crown_skipped(self.b, self.move))
        self.assertEqual([s.kind for s in moves_to_transfers(self.move, self.b)], ['piece'])

    def test_expected_kings_exclude_end_square(self):
        from physical_match import expected_kings_after, king_signature
        from engine import apply_move
        from game_logic import number_of
        end = number_of(self.move.end)
        self.assertIn(end, king_signature(apply_move(self.b, self.move)))
        self.assertNotIn(end, expected_kings_after(self.b, self.move))

    def test_crown_still_done_when_game_continues(self):
        from engine import crown_skipped
        b = [list(r) for r in self.b]; b[2][1] = 1
        b = tuple(tuple(x) for x in b)
        mv = next(m for m in legal_moves(b, AI) if m.end[0] == 7)
        self.assertFalse(crown_skipped(b, mv))


class NoClearBetweenSteps(unittest.TestCase):
    """ระหว่างขั้นของตาเดียวกัน (วางหมาก -> ทิ้ง -> วางฮอส) ไม่ไปจุดหลีกกล้อง ไปจุดหลีกเฉพาะหลังขั้นสุดท้าย"""
    def test_clear_only_after_last_step(self):
        planner, steps = crown_setup()
        _runner, robot = run_steps(v4_config(), steps, planner)
        c = planner.clear
        clear_cmd = f'MovL({c[0]:.1f},{c[1]:.1f},{c[2]:.1f},{v.TOOL_R:.1f})'   # Rotation ถูกล็อกเป็น TOOL_R เสมอ
        idx = [i for i, m in enumerate(robot.moves) if m == clear_cmd]
        self.assertEqual(len(idx), 1)                       # ครั้งเดียว
        self.assertEqual(idx[0], max(i for i, m in enumerate(robot.moves) if m.startswith('MovL')))   # และเป็นคำสั่งเดินสุดท้าย

    def test_capture_goes_straight_to_discard_pick(self):
        from engine import Move, legal_moves
        from game_logic import coord_of
        b = [[0] * 8 for _ in range(8)]
        b[5][2] = -1; b[6][3] = 1; b[2][1] = 1           # ส้มกินดำ (6,3) ลงแถวสุดท้ายได้
        b = tuple(tuple(x) for x in b)
        mv = next(m for m in legal_moves(b, AI) if m.captures)
        planner = RobotPlanner(FakeCamera())
        steps = planner.plan(mv, b)
        self.assertEqual([s.kind for s in steps][:2], ['piece', 'captured'])
        _r, robot = run_steps(v4_config(), steps, planner)
        c = planner.clear
        clear_cmd = f'MovL({c[0]:.1f},{c[1]:.1f},{c[2]:.1f},{v.TOOL_R:.1f})'   # Rotation ถูกล็อกเป็น TOOL_R เสมอ
        self.assertEqual(robot.moves.count(clear_cmd), 1)
        cap = planner.camera.xy(steps[1].source)
        first_cap = next(i for i, m in enumerate(robot.moves) if m.startswith(f'MovL({cap[0]:.1f},{cap[1]:.1f}'))
        last_release = max(i for i, m in enumerate(robot.moves[:first_cap]) if m == 'release')
        self.assertFalse(any(m == clear_cmd for m in robot.moves[last_release:first_cap]))   # ไม่แวะจุดหลีกก่อนไปหยิบตัวที่ถูกกิน


@patch.object(v, 'KING_CARRY_Z', -25.0)
class KingLiftSequence(unittest.TestCase):
    """ฮอสบนกระดาน: วิธียกเลือกได้ด้วย KING_LIFT_METHOD; หมากธรรมดายกแค่ hover"""
    def _run(self, piece, method="movl_high"):
        b = [[0] * 8 for _ in range(8)]; b[7][0] = piece; b[2][1] = 1
        if abs(piece) == 1:
            b[7][0] = 0; b[5][2] = piece
        b = tuple(tuple(x) for x in b)
        mv = next(m for m in legal_moves(b, AI) if not m.captures)
        planner = RobotPlanner(FakeCamera())
        steps = planner.plan(mv, b)
        with patch.object(v, 'KING_LIFT_METHOD', method):
            _r, robot = run_steps(v4_config(), steps, planner)
        return planner, [m for m in robot.moves if m.startswith('Mov')]

    def _after_pick(self, planner, cmds):
        src = planner.camera.xy(1)
        pick = f'MovL({src[0]:.1f},{src[1]:.1f},{v.PICK_Z + v.KING_PICK_Z_OFFSET:.1f}'
        i = next(k for k, c in enumerate(cmds) if c.startswith(pick))
        return src, i

    def test_default_single_movl_high(self):
        planner, cmds = self._run(-2)
        src, i = self._after_pick(planner, cmds)
        self.assertTrue(cmds[i + 1].startswith(f'MovL({src[0]:.1f},{src[1]:.1f},{v.KING_CARRY_Z:.1f}'))
        self.assertFalse(any(c.startswith('MovJ') for c in cmds))

    def test_movl_then_movj(self):
        planner, cmds = self._run(-2, "movl_movj")
        src, i = self._after_pick(planner, cmds)
        self.assertTrue(cmds[i + 1].startswith(f'MovL({src[0]:.1f},{src[1]:.1f},{v.PICK_Z + v.KING_LIFT_MOVL:.1f}'))
        self.assertTrue(cmds[i + 2].startswith(f'MovJ({src[0]:.1f},{src[1]:.1f},{v.KING_CARRY_Z:.1f}'))
        self.assertEqual(sum(c.startswith('MovJ') for c in cmds), 1)

    def test_plain_piece_unchanged(self):
        _p, cmds = self._run(-1, "movj")
        self.assertFalse(any(c.startswith('MovJ') for c in cmds))


@patch.object(v, 'KING_CARRY_Z', -25.0)
class KingLiftFallback(unittest.TestCase):
    def test_movl_hang_falls_back_to_movj(self):
        class Hang(FakeRobot):
            recovered = False
            hung = False
            def m(self, cmd):
                if cmd == 'Sync()' and not self.hung and self.moves and self.moves[-1].startswith('MovL') and \
                        f',{v.KING_CARRY_Z:.1f},' in self.moves[-1]:
                    self.hung = True
                    self.moves.append('HANG')
                    raise TimeoutError('timed out')
                return super().m(cmd)
            def recover_motion(self): self.recovered = True; self.moves.append('recover')
            def set_motion_timeout(self, s): pass
        b = [[0] * 8 for _ in range(8)]; b[7][0] = -2; b[2][1] = 1
        b = tuple(tuple(x) for x in b)
        mv = next(m for m in legal_moves(b, AI) if not m.captures)
        planner = RobotPlanner(FakeCamera()); steps = planner.plan(mv, b)
        robots = []
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / 'c.json'; p.write_text(json.dumps(v4_config()), encoding='utf-8')
            runner = LiveRunner(p)
            def factory(ip):
                r = Hang(ip); robots.append(r); return r
            with patch('robot_io.v.MG400', side_effect=factory), patch('mg400_visionV4.time.sleep', return_value=None), \
                 patch('robot_io.time.sleep', return_value=None):
                runner.run(steps, planner, report=lambda s: None, stop=lambda: False)
        mv_ = robots[0].moves
        self.assertTrue(robots[0].recovered)
        self.assertIn('HANG', mv_)
        self.assertTrue(any(c.startswith('MovJ(') for c in mv_))
        self.assertGreater(len(mv_), mv_.index('recover'))
        self.assertEqual(mv_.count('release'), 1)     # วางฮอสสำเร็จหลังสลับ


class ToolRotationLocked(unittest.TestCase):
    """ทุกคำสั่ง MovL/MovJ ต้องใช้ r = TOOL_R (0) แม้จุดหลีก/จุดทิ้งที่บันทึกไว้จะมี r อื่น"""
    def test_every_motion_uses_r_zero(self):
        self.assertEqual(v.TOOL_R, 0.0)
        self.assertTrue(abs(np.load(_synth.FIX / 'clear_pos.npz')['pos'][3]) > 1)    # fixture เก่ามี r ≠ 0 จริง
        for piece in (-1, -2):
            b = [[0] * 8 for _ in range(8)]; b[6][1] = -1; b[2][1] = 1
            if piece == -2:
                b[6][1] = 0; b[7][0] = -2
            b = tuple(tuple(x) for x in b)
            mv = next(m for m in legal_moves(b, AI) if not m.captures)
            planner = RobotPlanner(FakeCamera()); steps = planner.plan(mv, b)
            _r, robot = run_steps(v4_config(), steps, planner)
            movs = [m for m in robot.moves if m.startswith(('MovL', 'MovJ'))]
            self.assertTrue(movs)
            self.assertTrue(all(m.endswith(',0.0)') for m in movs), movs)

    def test_crown_and_discard_also_r_zero(self):
        planner, steps = crown_setup()                 # เดิน -> ทิ้งหมากที่จุดทิ้ง (r เก่า -46.5) -> หยิบฮอสจากที่เก็บ
        self.assertEqual([s.kind for s in steps], ['piece', 'king_remove', 'king_place'])
        _r, robot = run_steps(v4_config(), steps, planner)
        movs = [m for m in robot.moves if m.startswith(('MovL', 'MovJ'))]
        self.assertGreater(len(movs), 10)
        self.assertTrue(all(m.endswith(',0.0)') for m in movs), [m for m in movs if not m.endswith(',0.0)')])
