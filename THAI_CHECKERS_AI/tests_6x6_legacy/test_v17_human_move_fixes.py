"""Unit tests for V17 human-move detection, vision glare/shadow recovery, and diagnostics."""
import sys
import unittest
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from engine import apply_move, legal_moves
from game_logic import AI, HUMAN, coord_of, move_label, number_of
from physical_match import MatchError, PhysicalMatch, expected_kings_after, signature
from vision_io import BoardCamera, VisionFailure


class V17HumanMoveFixesTests(unittest.TestCase):
    def test_vision_recovers_all_four_lighting_failure_modes(self):
        """
        ทดสอบว่ากล้องสามารถตรวจจับหมากดำได้ครบทั้ง 6 ตัว แม้เกิดสภาวะ:
        1) ช่องสีฟ้าอยู่ในเงามืด (V=145, S=141)
        2) เงาเชื่อมระหว่างหมากดำ 2 ตัวที่อยู่ช่องติดกัน (#5 และ #12)
        3) แสงสะท้อนสว่างจ้าบนหน้าหมากดำ (#8, V=165)
        4) จุดเขียวฮอส + แสงสะท้อนที่ขอบหมากจนเป็นรูปตัว C (#15)
        """
        cam = BoardCamera()
        img = np.full((720, 1280, 3), 240, np.uint8)

        def sq_px(n):
            p = cam.Hinv @ np.array([*cam.xy(n), 1.0])
            p /= p[2]
            return int(round(p[0])), int(round(p[1]))

        # 1) ช่องสีฟ้าในเงามืดที่ช่อง #3
        u3, v3 = sq_px(3)
        cv2.rectangle(img, (u3 - 26, v3 - 26), (u3 + 26, v3 + 26), (145, 110, 65), -1)

        # 2) เงาเชื่อมระหว่างหมากดำช่อง #5 และ #12
        u5, v5 = sq_px(5)
        u12, v12 = sq_px(12)
        cv2.line(img, (u5, v5), (u12, v12), (135, 125, 115), 18)

        for n in (1, 3, 5, 8, 12, 15, 25, 27, 29, 32, 34, 36):
            u, w = sq_px(n)
            col = (40, 40, 40) if n < 20 else (0, 135, 250)
            cv2.circle(img, (u, w), 19, col, -1)

        # 3) แสงสะท้อนสว่างบนหมากดำช่อง #8
        u8, v8 = sq_px(8)
        cv2.circle(img, (u8 + 6, v8), 11, (165, 160, 155), -1)

        # 4) จุดเขียวฮอส + รอยแหว่งแสงสะท้อนที่ขอบหมากช่อง #15
        u15, v15 = sq_px(15)
        cv2.circle(img, (u15, v15), 9, (0, 180, 0), -1)
        cv2.rectangle(img, (u15 + 8, v15 - 3), (u15 + 19, v15 + 3), (200, 200, 200), -1)

        obs = cam.observe(img)
        self.assertEqual(sorted(k for k, v in obs.items() if v == 'black'), [1, 3, 5, 8, 12, 15])
        self.assertEqual(sorted(k for k, v in obs.items() if v == 'orange'), [25, 27, 29, 32, 34, 36])
        self.assertEqual(cam.last_kings, {15})

    def test_stock_green_dot_never_bleeds_onto_adjacent_board_square(self):
        """จุดเขียวของฮอสสำรองข้างกระดานต้องถูกจับคู่กับตัวฮอสสำรองเท่านั้น ไม่ไปติดหมากธรรมดาในช่อง #12 หรือ #24"""
        cam = BoardCamera()
        img = np.full((720, 1280, 3), 240, np.uint8)

        def sq_px(n):
            p = cam.Hinv @ np.array([*cam.xy(n), 1.0])
            p /= p[2]
            return int(round(p[0])), int(round(p[1]))

        for n in (1, 3, 5, 8, 10, 12, 25, 27, 29, 32, 34, 36):
            u, w = sq_px(n)
            col = (40, 40, 40) if n <= 12 else (0, 135, 250)
            cv2.circle(img, (u, w), 19, col, -1)

        # วาดฮอสสำรองสีดำพร้อมจุดเขียวที่จุดเก็บใกล้ช่อง #12 (u~801, v~491)
        cv2.circle(img, (801, 491), 19, (40, 40, 40), -1)
        cv2.circle(img, (790, 491), 9, (0, 180, 0), -1)

        obs = cam.observe(img)
        self.assertEqual(obs.get(12), 'black')
        self.assertEqual(cam.last_kings, set())

    def test_diagnose_mandatory_capture_on_recorded_match(self):
        """จำลองสถานการณ์จริงจาก history_physical_match.xlsx ตาที่ 7 ที่คนถูกบังคับกิน 15 -> 29"""
        m = PhysicalMatch()
        m.start(signature(m.game.board), first_side=HUMAN, kings=set())
        seq = [
            ('human', '12 → 17'),
            ('ai', '25 → 20'),
            ('human', '8 → 15'),
            ('ai', '20 → 13'),
            ('human', '17 → 24'),
            ('ai', '29 → 22'),
        ]
        for who, label in seq:
            if who == 'human':
                mv = next(x for x in m.game.human_moves if move_label(x) == label)
                m.accept_human(signature(apply_move(m.game.board, mv)), kings=set())
            else:
                mv = next(x for x in legal_moves(m.game.board, AI) if move_label(x) == label)
                m.stage_ai_move(mv)
                m.before_robot(signature(m.game.board), kings=set())
                m.after_robot('DONE')
                m.verify_ai(signature(apply_move(m.game.board, mv)), kings=set())

        # ตอนนี้ถึงตาคน และมีตาเดินที่ถูกกติกาเพียงตาเดียวคือ 15 -> 29 (บังคับกินช่อง 22)
        self.assertEqual([move_label(x) for x in m.game.human_moves], ['15 → 29'])

        # ถ้าคนขยับหมากตัวอื่น เช่น 24 -> 29 (ซึ่งดูเหมือนเดินเฉียงหน้าปกติ)
        cam_wrong = signature(m.game.board)
        del cam_wrong[24]
        cam_wrong[29] = 'black'
        self.assertEqual(m.classify_human(cam_wrong, kings=set()), 'illegal')
        kind, msg = m.diagnose_human(cam_wrong, kings=set())
        self.assertEqual(kind, 'must_capture')
        self.assertIn('บังคับกิน', msg)
        self.assertIn('15 → 29', msg)

        # ถ้าคนขยับ 15 -> 29 แล้วแต่ยังไม่ได้หยิบหมากส้มที่ช่อง 22 ออก
        cam_pending = signature(m.game.board)
        del cam_pending[15]
        cam_pending[29] = 'black'
        kind2, msg2 = m.diagnose_human(cam_pending, kings=set())
        self.assertEqual(kind2, 'capture_removal_pending')
        self.assertIn('22', msg2)

        # ถ้ากล้องมองไม่เห็นหมากที่เพิ่งเดินไป (หมากดำเหลือ 5 ตัว) ต้องจัดเป็น incomplete_scan ไม่ใช่ผิดกติกา
        cam_missing = signature(m.game.board)
        del cam_missing[15]
        kind3, msg3 = m.diagnose_human(cam_missing, kings=set())
        self.assertEqual(kind3, 'incomplete_scan')
        self.assertIn('5/6', msg3)

    def test_human_winning_promotion_skips_king_dot_requirement(self):
        """ถ้าคนเดินหมากเข้าแถวฮอสแล้วชนะทันที (AI ไม่เหลือหมาก/ตาเดิน) ไม่บังคับต้องวางจุดเขียว"""
        m = PhysicalMatch()
        board = [[0] * 6 for _ in range(6)]
        board[1][0] = 1   # คนที่ช่อง 25
        board[2][1] = -1  # AI ที่ช่อง 20 (ไม่มีตากิน แต่เดินต่อได้ถ้าไม่ถูกกิน)
        # ตั้งให้คนกิน 25 -> 32? ไม่ได้ 25 อยู่แถว 1 ต้องเดินขึ้นแถว 0 (ช่อง 32):
        board = [[0] * 6 for _ in range(6)]
        board[2][1] = 1   # คนที่ช่อง 20
        board[1][2] = -1  # AI ที่ช่อง 27 (ตัวสุดท้ายของ AI)
        m.game.board = tuple(tuple(r) for r in board)
        m.state.stage = 'HUMAN'
        # คนกระโดดกิน 20 -> 34 (เข้าแถวฮอสและกินหมาก AI ตัวสุดท้าย ชนะทันที!)
        mv = next(x for x in m.game.human_moves if move_label(x) == '20 → 34')
        after_cam = signature(apply_move(m.game.board, mv))
        self.assertEqual(expected_kings_after(m.game.board, mv), set())
        self.assertEqual(m.classify_human(after_cam, kings=set()), 'legal')
        entry = m.accept_human(after_cam, kings=set())
        self.assertEqual(move_label(entry.move), '20 → 34')
        self.assertEqual(m.state.stage, 'GAME_OVER')


if __name__ == '__main__':
    unittest.main()
