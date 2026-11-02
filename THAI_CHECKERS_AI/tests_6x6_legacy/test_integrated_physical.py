import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))

import numpy as np
from engine import apply_move,legal_moves, Move,initial_board
from game_logic import HUMAN, AI, coord_of, move_numbers
from physical_match import PhysicalMatch,signature,MatchError
from robot_io import RobotPlanner,moves_to_transfers,LiveRunner,RobotFailure
from vision_io import BoardCamera,VisionFailure
from adams import ADAMS,SearchConfig
from excel_history import export_history_xlsx

class FakeCamera:
    z_plane=-155.13978576660156
    verified_board=False
    def xy(self,n):
        from mg400_visionV2 import BOARD_AFFINE_DEFAULT,square_xy
        return tuple(square_xy(np.array(BOARD_AFFINE_DEFAULT),n))

class PhysicalTests(unittest.TestCase):
    def test_coordinates_and_sides(self):
        self.assertEqual(coord_of(1),(5,0))
        self.assertEqual(coord_of(36),(0,5))
        initial=signature(initial_board())
        self.assertEqual(sorted(n for n,color in initial.items() if color=='black'),[1,3,5,8,10,12])
        self.assertEqual(sorted(n for n,color in initial.items() if color=='orange'),[25,27,29,32,34,36])

    def test_real_calibrations_and_orientation(self):
        cam=BoardCamera()
        self.assertEqual(cam.H.shape,(3,3))
        self.assertEqual(cam.verified_board, (Path(__file__).resolve().parents[1] / "board.npz").is_file())
        self.assertAlmostEqual(cam.z_plane,-155.1398,places=2)
        self.assertEqual(cam.A.shape,(2,3))
        self.assertEqual(cam.xy(1),tuple(cam.A[:,0]))

    def test_tested_hsv_detector_on_camera_geometry(self):
        import cv2
        cam=BoardCamera()
        frame=np.full((720,1280,3),240,dtype=np.uint8)
        for n,color in signature(initial_board()).items():
            x,y=cam.xy(n)
            p=cam.Hinv @ np.array([x,y,1.0]);p/=p[2]
            px,py=map(int,np.rint(p[:2]))
            cv2.circle(frame,(px,py),18,(40,40,40) if color=='black' else (0,135,250),-1)
        self.assertEqual(cam.observe(frame),signature(initial_board()))

    def test_legal_ai_multijump_removal_before_motion(self):
        rows=[[0]*6 for _ in range(6)]
        rows[1][0]=-1  # AI orange starts at #25
        rows[2][1]=1  # human black at #20
        rows[4][3]=1  # human black at #10
        b=tuple(tuple(row) for row in rows)
        moves=legal_moves(b,AI)
        self.assertTrue(moves)
        self.assertIn((25,15,5),[move_numbers(m) for m in moves])
        move=next(m for m in moves if move_numbers(m)==(25,15,5))
        steps=moves_to_transfers(move)
        self.assertEqual([s.source for s in steps if s.kind=='captured'],[20,10])
        self.assertEqual([(s.source,s.dest) for s in steps if s.kind=='piece'],[(25,5)])

    def test_initial_board_validation(self):
        m=PhysicalMatch()
        with self.assertRaises(MatchError):m.start({1:'orange'})
        self.assertEqual(m.state.stage,'INITIAL')
        m.start(signature(m.game.board));self.assertEqual(m.state.stage,'HUMAN')

    def test_human_to_ai_camera_verification(self):
        m=PhysicalMatch();m.start(signature(m.game.board))
        h=legal_moves(m.game.board,HUMAN)[0]
        target=apply_move(m.game.board,h)
        entry=m.accept_human(signature(target))
        self.assertEqual(entry.move,h)
        self.assertEqual(m.state.stage,'AI_READY')
        self.assertEqual(len(m.game.history),1)
        ai=ADAMS(SearchConfig(max_depth=3,time_limit_ms=100, use_opening_book=False, use_tablebase=False,use_endgame_db=False))
        result=ai.choose(m.game.board,AI)
        self.assertIsNotNone(result.move)
        m.stage_ai_move(result.move)
        planner=RobotPlanner(FakeCamera())
        steps=planner.plan(result.move)
        self.assertTrue(steps)
        wrong=signature(initial_board())
        with self.assertRaises(MatchError):m.before_robot(wrong)
        self.assertEqual(m.state.stage,'AI_READY')
        m.before_robot(signature(m.game.board))
        # Unit-test the physical-board state transitions without commanding a robot.
        # The real controller is separately gated by site-reviewed interlocks.
        m.after_robot('ROBOT REPORTED DONE (AWAIT CAMERA)')
        self.assertEqual(len(m.game.history),1)  # no AI history before verification
        with self.assertRaises(MatchError):m.verify_ai(signature(m.game.board))
        exp=signature(apply_move(m.game.board,result.move))
        entry2=m.verify_ai(exp)
        self.assertEqual(entry2.move,result.move)
        self.assertEqual(len(m.game.history),2)
        self.assertEqual(m.state.stage,'HUMAN')
        self.assertIn('CAMERA VERIFIED',m.robot_status[2])

    def test_refuse_unfinished_or_illegal_turn(self):
        m=PhysicalMatch()
        with self.assertRaises(MatchError):m.accept_human({})
        m.start(signature(m.game.board))
        with self.assertRaises(MatchError):m.accept_human(signature(m.game.board))
        self.assertEqual(len(m.game.history),0)

    def test_multijump_plan(self):
        src=lambda n:coord_of(n)
        mv=Move((src(25),src(13),src(1)),(src(20),src(8)))
        steps=moves_to_transfers(mv)
        self.assertEqual([(x.kind,x.source,x.dest) for x in steps],
                         [('piece',25,1),('captured',20,None),('captured',8,None)])
        hop=moves_to_transfers(mv,direct=False)
        self.assertEqual([(x.kind,x.source,x.dest) for x in hop],[('piece',25,13),('piece',13,1),('captured',20,None),('captured',8,None)])
        old=moves_to_transfers(mv,capture_last=False,direct=False)
        self.assertEqual([x.kind for x in old],['captured','captured','piece','piece'])

    def test_robot_range_and_z_checks(self):
        c=FakeCamera();p=RobotPlanner(c)
        self.assertTrue(p.z_hover>p.z_place>-150)
        self.assertTrue(p.plan(legal_moves(initial_board(),AI)[0]))
        class Bad(FakeCamera):
            def xy(self,n):return (10000.,20000.)
        with self.assertRaises(RobotFailure):RobotPlanner(Bad()).plan(legal_moves(initial_board(),AI)[0])

    def test_live_stays_locked_without_review(self):
        with self.assertRaises(RobotFailure):LiveRunner(Path(__file__).parent/'nonexistent_config.json')
        with self.assertRaises(RobotFailure):LiveRunner(Path(__file__).resolve().parents[1]/'live_safety_config.example.json')

    def test_history_excel_zip_and_xml(self):
        import zipfile
        import xml.etree.ElementTree as ET
        m=PhysicalMatch();m.start(signature(m.game.board))
        h=legal_moves(m.game.board,HUMAN)[0];m.accept_human(signature(apply_move(m.game.board,h)))
        with tempfile.TemporaryDirectory() as td:
            f=Path(td)/'result.xlsx'
            export_history_xlsx(f,m.game.history,difficulty='ปานกลาง',game_status=m.game.status)
            with zipfile.ZipFile(f) as z:
                self.assertIsNone(z.testzip())
                for name in z.namelist():
                    if name.endswith('.xml') or name.endswith('.rels'):
                        ET.fromstring(z.read(name))
                self.assertIn('xl/worksheets/sheet1.xml',z.namelist())

if __name__=='__main__':unittest.main()
