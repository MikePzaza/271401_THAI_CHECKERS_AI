import unittest
from engine import apply_move
from game_logic import HUMAN, number_of
from physical_match import PhysicalMatch, signature


class HumanCaptureOrder(unittest.TestCase):
    def _board_with_capture(self):
        pm = PhysicalMatch()
        b = [[0] * 8 for _ in range(8)]
        b[5][2] = 1       # คน (ดำ) เดินขึ้นด้านบนของกระดาน (แถวน้อย = บน)
        b[4][3] = -1      # ส้มที่ถูกกินได้
        b[0][7] = -1
        b[7][0] = 1
        pm.game.board = tuple(tuple(x) for x in b)
        pm.game.side_to_move = HUMAN
        pm.state.stage = 'HUMAN'
        return pm

    def test_all_three_orders(self):
        pm = self._board_with_capture()
        caps = [m for m in pm.game.human_moves if m.is_capture]
        self.assertTrue(caps)
        m = caps[0]
        final = signature(apply_move(pm.game.board, m))
        start = signature(pm.game.board)
        cap_sq = number_of(m.captures[0])
        # a) วางหมากดำถึงปลายทางก่อน ยังไม่หยิบหมากส้ม -> รอ ไม่ใช่ illegal
        a = dict(final); a[cap_sq] = 'orange'
        self.assertNotEqual(pm.classify_human(a), 'legal')
        self.assertIn(pm.diagnose_human(a)[0], ('capture_removal_pending',))
        self.assertIn('✓ เห็นแล้ว', pm.progress_human(a))
        # b) หยิบหมากส้มออกก่อน ยังไม่ขยับหมากดำ -> รอ
        b = dict(start); del b[cap_sq]
        self.assertEqual(pm.diagnose_human(b)[0], 'incomplete_scan')
        self.assertIn('✓ ครบแล้ว', pm.progress_human(b))
        # c) หยิบออก+วางทีเดียว -> ผ่านเลย
        self.assertEqual(pm.classify_human(final), 'legal')
        self.assertEqual(pm.diagnose_human(final)[0], 'legal')


if __name__ == '__main__':
    unittest.main()
