"""Regression coverage: men and kings share physical height and V4 motion."""
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from game_logic import AI
from engine import legal_moves
from robot_io import RobotPlanner, LiveRunner, RobotFailure
from test_v4_integration import FakeCamera, FakeRobot, approved_config
import mg400_visionV4 as v


def setup(board):
    board = tuple(map(tuple, board))
    cam = FakeCamera()
    move = legal_moves(board, AI)[0]
    planner = RobotPlanner(cam)
    steps = planner.plan(move,board)
    return cam, planner, steps


def simulate(planner,steps):
    with tempfile.TemporaryDirectory() as d:
        cfg = Path(d)/'cfg.json'
        config = approved_config()
        config['motion_engine'] = 'v4'
        config.pop('king_pick_z', None)
        config.pop('stock_king_pick_z', None)
        cfg.write_text(json.dumps(config))
        instance = LiveRunner(cfg)
        robots=[]
        def make(ip):
            r=FakeRobot(ip)
            robots.append(r)
            return r
        with patch('robot_io.v.MG400',side_effect=make), patch('robot_io.time.sleep',return_value=None):
            instance.run(steps,planner,stop=lambda: False,report=lambda message: None)
        return robots[0]


class KingSameSize(unittest.TestCase):
    def test_normal_man_uses_normal_pick_z(self):
        board=[[0]*6 for _ in range(6)]
        board[2][3]=-1
        board[0][0]=1
        cam, planner, steps=setup(board)
        r=simulate(planner,steps)
        xy=cam.xy(steps[0].source)
        self.assertIn(f'MovL({xy[0]:.1f},{xy[1]:.1f},{v.PICK_Z:.1f},0.0)',r.moves)

    def test_existing_king_uses_same_pick_z_without_king_config(self):
        board=[[0]*6 for _ in range(6)]
        board[2][3]=-2
        board[0][0]=1
        cam, planner, steps=setup(board)
        r=simulate(planner,steps)
        xy=cam.xy(steps[0].source)
        self.assertIn(f'MovL({xy[0]:.1f},{xy[1]:.1f},{v.PICK_Z:.1f},0.0)',r.moves)
        self.assertEqual(r.moves.count('vacuum_on'),1)
        self.assertEqual(r.moves.count('release'),1)

    def test_promotion_uses_original_z_and_saved_stock_xy(self):
        board=[[0]*6 for _ in range(6)]
        board[4][1]=-1
        board[2][1]=1
        cam, planner, steps=setup(board)
        self.assertEqual([s.kind for s in steps],['piece','king_remove','king_place'])
        reserve=v.load_stock()['orange'][0]
        self.assertEqual(planner.king_source,reserve[:2])
        r=simulate(planner,steps)
        x,y=planner.king_source
        self.assertIn(f'MovL({x:.1f},{y:.1f},{v.PICK_Z:.1f},0.0)',r.moves)
        self.assertEqual(r.moves.count('vacuum_on'),3)

    def test_far_king_marker_does_not_drive_robot_to_guessed_stock_position(self):
        board=[[0]*6 for _ in range(6)]
        board[4][1]=-1
        board[2][1]=1
        cam=FakeCamera()
        reserve=v.load_stock()['orange'][0]
        cam.last_robot_objects=[(reserve[0],reserve[1],reserve[2]+40,reserve[3],'orange',True)]
        planner=RobotPlanner(cam)
        move=legal_moves(tuple(map(tuple,board)),AI)[0]
        with self.assertRaisesRegex(RobotFailure,'กล้องไม่ยืนยันฮอส'):
            planner.plan(move,tuple(map(tuple,board)))

if __name__=='__main__': unittest.main()
