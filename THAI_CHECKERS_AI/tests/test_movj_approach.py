import math, re, unittest
from unittest.mock import patch
import mg400_visionV4 as v


class PoseRobot:
    def __init__(self):
        self.p = [300.0, 0.0, -100.0, 0.0]; self.moves = []
    def pose(self): return list(self.p)
    def m(self, cmd):
        self.moves.append(cmd)
        mm = re.match(r"Mov[JL]\(([-\d.]+),([-\d.]+),([-\d.]+)", cmd)
        if mm: self.p[:3] = [float(x) for x in mm.groups()]
        return "0,{},x;"
    def release(self): pass
    def pump_off(self): pass
    def vacuum_on(self): pass


class Approach(unittest.TestCase):
    def test_movj_then_short_movl(self):
        r = PoseRobot()
        with patch.object(v, "HOLD_SECONDS", 0), patch.object(v, "MOVJ_APPROACH_MM", 10.0):
            v.move_piece(r, (300.0, 50.0), (320.0, 70.0), (250, -200, -60, 0), -128.4, to_clear=False)
        cmds = [c for c in r.moves if c.startswith("Mov")]
        prev = [300.0, 0.0, -100.0]
        for i, c in enumerate(cmds):
            xyz = [float(x) for x in re.match(r"Mov[JL]\(([-\d.]+),([-\d.]+),([-\d.]+)", c).groups()]
            if c.startswith("MovL"):
                self.assertLessEqual(math.dist(prev, xyz), 11.5, (i, c))
            prev = xyz
        self.assertTrue(any(c.startswith("MovJ") for c in cmds))


if __name__ == "__main__":
    unittest.main()


class MinReach(unittest.TestCase):
    def test_pick_clamped_and_place_compensated(self):
        r = PoseRobot()
        with patch.object(v, "HOLD_SECONDS", 0):
            v.move_piece(r, (233.6, -9.7), (300.0, 40.0), (250, -200, -60, 0), -128.4, to_clear=False)
        cmds = [c for c in r.moves if c.startswith("Mov")]
        xs = [float(re.match(r"Mov[JL]\(([-\d.]+)", c).group(1)) for c in cmds]
        self.assertGreaterEqual(min(xs), v.MIN_REACH_X - 0.05)
        self.assertTrue(any(c.startswith("MovL(238.0,-9.7,%.1f" % (v.PICK_Z)) for c in cmds))
        self.assertTrue(any(c.startswith("MovL(304.4,40.0,") for c in cmds))


class AllMovLAndVia(unittest.TestCase):
    def test_default_all_movl(self):
        self.assertEqual(v.MOVJ_APPROACH_MM, 0)
        r = PoseRobot()
        with patch.object(v, "HOLD_SECONDS", 0):
            v.move_piece(r, (300.0, 50.0), (320.0, 70.0), (250, -200, -60, 0), -108.0, to_clear=False)
        self.assertFalse(any(c.startswith("MovJ") for c in r.moves))

    def test_via_dip(self):
        r = PoseRobot()
        with patch.object(v, "HOLD_SECONDS", 0):
            v.move_piece(r, (300.0, 50.0), (340.0, 90.0), (250, -200, -60, 0), -108.0, to_clear=False, via=[(320.0, 70.0)])
        c = [x for x in r.moves if x.startswith("Mov")]
        i = next(k for k, x in enumerate(c) if x.startswith("MovL(320.0,70.0,-108.0"))
        dip = max(v.Z_RANGE[0], v.PICK_Z + v.PLACE_Z_OFFSET, -108.0 - v.VIA_DIP_MM)
        self.assertTrue(c[i + 1].startswith("MovL(320.0,70.0,%.1f" % dip))
        self.assertGreaterEqual(dip, v.PICK_Z + v.PLACE_Z_OFFSET - 1e-6)
        self.assertTrue(c[i + 2].startswith("MovL(320.0,70.0,-108.0"))
        self.assertTrue(c[i + 3].startswith("MovL(340.0,90.0,-108.0"))

    def test_multicapture_transfer_has_via(self):
        import robot_io
        from engine import Move
        mv = Move(path=((7, 0), (5, 2), (3, 4)), captures=((6, 1), (4, 3)))
        steps = robot_io.moves_to_transfers(mv)
        piece = steps[0]
        self.assertEqual(len(piece.via), 1)
        self.assertEqual(piece.source, robot_io.number_of((7, 0)))
        self.assertEqual(piece.dest, robot_io.number_of((3, 4)))
        self.assertEqual(piece.via[0], robot_io.number_of((5, 2)))

    def test_speed(self):
        self.assertEqual(v.SPEED, 85)
