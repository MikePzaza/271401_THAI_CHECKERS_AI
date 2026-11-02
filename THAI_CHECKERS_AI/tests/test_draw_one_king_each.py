import unittest
from game_logic import CheckersGame, HUMAN, AI, number_of
from engine import legal_moves


class OneKingEach(unittest.TestCase):
    def test_draw_when_one_king_each(self):
        g = CheckersGame()
        b = [[0] * 8 for _ in range(8)]
        b[7][0] = 2     # ฮอสคน
        b[0][5] = -2    # ฮอส AI
        b[4][3] = -1    # หมากส้มที่ฮอสคนกิน -> เหลือฮอสละตัว
        g.board = tuple(tuple(x) for x in b)
        g.side_to_move = HUMAN
        mv = next(m for m in legal_moves(g.board, HUMAN) if m.is_capture)
        g.apply_human(mv) if hasattr(g, 'apply_human') else g._move(mv, HUMAN)
        self.assertEqual(g.status, 'draw')
        self.assertIn('ฮอสฝ่ายละ 1', g.draw_reason)

    def test_no_draw_with_man_left(self):
        g = CheckersGame()
        b = [[0] * 8 for _ in range(8)]
        b[7][0] = 2; b[0][5] = -2; b[6][1] = 1
        g.board = tuple(tuple(x) for x in b)
        g.side_to_move = HUMAN
        mv = legal_moves(g.board, HUMAN)[0]
        g._move(mv, HUMAN)
        self.assertEqual(g.status, 'playing')


if __name__ == '__main__':
    unittest.main()
