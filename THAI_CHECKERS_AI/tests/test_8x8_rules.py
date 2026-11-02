"""กติกา/พิกัด/สถานะเกมสำหรับหมากฮอสไทย 8x8 (ฝั่งละ 8 ตัว)"""
import sys, unittest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import _synth
from engine import initial_board, legal_moves, apply_move, BOARD_SIZE
from game_logic import HUMAN, AI, coord_of, number_of, move_numbers
from physical_match import PhysicalMatch, signature, king_signature, MatchError
from adams import ADAMS, SearchConfig, EvaluationWeights
from robot_io import moves_to_transfers


def empty():
    return [[0] * 8 for _ in range(8)]


def T(b):
    return tuple(tuple(r) for r in b)


class Rules8x8(unittest.TestCase):
    def test_initial_position_and_numbers(self):
        self.assertEqual(BOARD_SIZE, 8)
        sig = signature(initial_board())
        self.assertEqual(sorted(n for n, c in sig.items() if c == 'black'), [1, 3, 5, 7, 10, 12, 14, 16])
        self.assertEqual(sorted(n for n, c in sig.items() if c == 'orange'), [49, 51, 53, 55, 58, 60, 62, 64])
        self.assertEqual(coord_of(1), (7, 0))
        self.assertEqual(coord_of(64), (0, 7))
        for n in range(1, 65):
            self.assertEqual(number_of(coord_of(n)), n)
        self.assertEqual(len(legal_moves(initial_board(), HUMAN)), 7)
        self.assertEqual(len(legal_moves(initial_board(), AI)), 7)

    def test_man_moves_forward_only_and_captures_forward(self):
        b = empty(); b[4][3] = 1; b[3][2] = -1; b[3][4] = -1   # ดำที่ (4,3) กินได้ทั้งสองข้างถ้าด้านหลังว่าง
        ms = legal_moves(T(b), HUMAN)
        self.assertTrue(all(m.captures for m in ms))            # บังคับกิน
        self.assertEqual({m.end for m in ms}, {(2, 1), (2, 5)})
        b = empty(); b[2][3] = 1; b[3][2] = -1                  # หมากธรรมดาไม่กินถอยหลัง
        self.assertFalse(any(m.captures for m in legal_moves(T(b), HUMAN)))

    def test_multi_jump_and_promotion_on_row_zero(self):
        b = empty(); b[6][1] = 1; b[5][2] = -1; b[3][4] = -1; b[1][2] = 0
        ms = legal_moves(T(b), HUMAN)
        self.assertTrue(any(len(m.captures) == 2 for m in ms))
        b = empty(); b[1][2] = 1; b[7][0] = -1
        m = next(m for m in legal_moves(T(b), HUMAN) if m.end[0] == 0)
        self.assertEqual(abs(apply_move(T(b), m)[0][m.end[1]]), 2)

    def test_flying_king_8x8(self):
        b = empty(); b[7][0] = 2; b[2][5] = -1
        ms = legal_moves(T(b), HUMAN)
        self.assertTrue(ms and all(m.captures for m in ms))      # ฮอสเห็นหมากไกล 5 ช่องแล้วต้องกิน
        self.assertEqual(ms[0].end, (1, 6))

    def test_physical_match_flow(self):
        m = PhysicalMatch()
        m.start(signature(m.game.board), kings=set())
        mv = next(x for x in m.game.human_moves if move_numbers(x) == (10, 17))
        cam = signature(apply_move(m.game.board, mv))
        m.accept_human(cam, kings=set())
        self.assertEqual(m.state.stage, 'AI_READY')
        with self.assertRaises(MatchError):
            PhysicalMatch().start({1: 'black'})

    def test_ai_plays_and_full_selfplay_stays_legal(self):
        w = EvaluationWeights.from_file(str(Path(__file__).resolve().parents[1] / 'data/adams_weights.json'))
        ai = ADAMS(SearchConfig(max_depth=4, time_limit_ms=150), weights=w)
        board, side = initial_board(), HUMAN
        for ply in range(40):
            ms = legal_moves(board, side)
            if not ms:
                break
            res = ai.choose(board, side)
            mv = res.move if hasattr(res, 'move') else res
            self.assertIn(mv, ms)
            board = apply_move(board, mv)
            side = -side

    def test_transfers_for_8x8_capture(self):
        b = empty(); b[5][2] = -1; b[6][3] = 1; b[2][1] = 1
        mv = next(m for m in legal_moves(T(b), AI) if m.captures)
        steps = moves_to_transfers(mv, T(b))
        self.assertEqual([s.kind for s in steps][:2], ['piece', 'captured'])


if __name__ == '__main__':
    unittest.main()
