import sys
import unittest
from pathlib import Path
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))

from engine import apply_move, legal_moves, initial_board, Move
from game_logic import AI, HUMAN, coord_of, move_numbers
from physical_match import PhysicalMatch, signature
from robot_io import moves_to_transfers, RobotFailure
from main import LEVELS

class NewSpecTests(unittest.TestCase):
    def test_levels_and_random(self):
        self.assertEqual(tuple(LEVELS),('ง่าย','ปานกลาง','ยาก','นรก'))
        self.assertLess(LEVELS['ง่าย'][0],LEVELS['ปานกลาง'][0])
        self.assertLess(LEVELS['ปานกลาง'][0],LEVELS['ยาก'][0])
        self.assertLess(LEVELS['ยาก'][0],LEVELS['นรก'][0])

    def test_starter_human_or_ai(self):
        for side,stage in [(HUMAN,'HUMAN'),(AI,'AI_READY')]:
            g=PhysicalMatch()
            g.start(signature(g.game.board),first_side=side)
            self.assertEqual(g.game.side_to_move,side)
            self.assertEqual(g.state.stage,stage)
            self.assertEqual(g.game.repetitions.total(),1)

    def test_classify_illegal_and_valid_recovery(self):
        g=PhysicalMatch()
        g.start(signature(g.game.board))
        self.assertEqual(g.classify_human(signature(g.game.board)),'unchanged')
        invalid=signature(g.game.board)
        del invalid[1]
        self.assertEqual(g.classify_human(invalid),'illegal')
        self.assertEqual(g.state.stage,'HUMAN')
        self.assertEqual(len(g.game.history),0)
        valid=legal_moves(g.game.board,HUMAN)[0]
        self.assertEqual(g.classify_human(signature(apply_move(g.game.board,valid))),'legal')
        g.accept_human(signature(apply_move(g.game.board,valid)))
        self.assertEqual(g.state.stage,'AI_READY')

    def test_promote_swap_plan_without_robot_motion(self):
        # Orange reaches bottom row (row 5) from row 4, promotion occurs.
        b=[[0]*6 for _ in range(6)]
        b[4][1]=-1;b[2][1]=1
        b=tuple(tuple(row) for row in b)
        moves=[m for m in legal_moves(b,AI) if len(m.path)==2 and m.end[0]==5]
        self.assertTrue(moves)
        from robot_io import moves_to_transfers
        tasks=moves_to_transfers(moves[0],b)
        self.assertEqual([t.kind for t in tasks],['piece','king_remove','king_place'])
        import numpy as np
        class FakeCamera:
            z_plane=-155.1398
            verified_board=False
            def xy(self,n):
                from mg400_visionV2 import BOARD_AFFINE_DEFAULT,square_xy
                return tuple(square_xy(np.array(BOARD_AFFINE_DEFAULT),n))
        from robot_io import RobotPlanner
        fc=FakeCamera()
        import mg400_visionV4 as v4
        entry=v4.load_stock()['orange'][0]
        fc.last_robot_objects=[(entry[0],entry[1],entry[2],entry[3],'orange',True)]
        plan=RobotPlanner(fc)
        steps=plan.plan(moves[0],b)
        self.assertEqual([t.kind for t in steps], ['piece', 'king_remove', 'king_place'])

    def test_capture_plan(self):
        rows=[[0]*6 for _ in range(6)]
        rows[1][0]=-1
        rows[2][1]=1
        rows[4][3]=1; rows[5][0]=1
        b=tuple(tuple(row) for row in rows)
        m=next(x for x in legal_moves(b,AI) if move_numbers(x)==(25,15,5))
        tasks=moves_to_transfers(m,b)
        self.assertEqual([s.kind for s in tasks],['piece','captured','captured','king_remove','king_place'])

    def test_wrong_first_player_fails(self):
        g=PhysicalMatch()
        with self.assertRaises(Exception):
            g.start(signature(g.game.board),first_side=10)

if __name__=='__main__':
    unittest.main()
