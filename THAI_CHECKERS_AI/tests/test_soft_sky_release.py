"""Focused regressions for the immediate draw and live popup integration."""
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from engine import apply_move, board_key, legal_moves
from game_logic import AI, HUMAN, CheckersGame, move_label
from physical_match import PhysicalMatch, expected_kings_after, king_signature, signature


def position(*pieces):
    b = [[0]*8 for _ in range(8)]
    for r, c, piece in pieces:
        b[r][c] = piece
    return tuple(tuple(row) for row in b)


class ImmediateDraw(unittest.TestCase):
    def test_both_sides_draw_after_one_move_without_another_turn(self):
        for side in (HUMAN, AI):
            with self.subTest(side=side):
                g = CheckersGame()
                g.board = position((7, 0, 2), (0, 5, -2))
                g.side_to_move = side
                move = legal_moves(g.board, side)[0]
                with patch('game_logic.terminal_winner', side_effect=AssertionError('draw must resolve first')):
                    if side == HUMAN:
                        g.play_human(move_label(move))
                    else:
                        g.play_ai(move)
                self.assertEqual((g.status, g.winner), ('draw', None))
                self.assertEqual(g.draw_reason, 'เหลือฮอสฝ่ายละ 1 ตัว')
                self.assertEqual(len(g.history), 1)
                self.assertEqual(g.human_moves, ())
                with self.assertRaises(ValueError):
                    g.play_ai(move)

    def test_promotion_to_the_last_pair_of_kings_draws(self):
        g = CheckersGame()
        g.board = position((1, 2, 1), (7, 6, -2))
        move = next(m for m in legal_moves(g.board, HUMAN) if m.end == (0, 1))
        entry = g.play_human(move_label(move))
        self.assertTrue(entry.promoted)
        self.assertEqual(g.status, 'draw')

    def test_ordinary_pieces_and_extra_pieces_do_not_trigger_two_king_rule(self):
        for pieces in [((7,0,1),(0,5,-1)), ((7,0,2),(0,5,-1)),
                       ((7,0,2),(0,5,-2),(6,3,1))]:
            with self.subTest(pieces=pieces):
                g = CheckersGame()
                g.board = position(*pieces)
                move = next(m for m in legal_moves(g.board, HUMAN) if not m.captures)
                g.play_human(move_label(move))
                self.assertEqual(g.status, 'playing')

    def test_human_capture_finishes_physical_match_without_ai_turn(self):
        match = PhysicalMatch()
        match.game.board = position((7,0,2),(0,5,-2),(4,3,-1))
        match.state.stage = 'HUMAN'
        move = next(m for m in legal_moves(match.game.board, HUMAN) if m.captures)
        after = apply_move(match.game.board, move)
        match.accept_human(signature(after), kings=expected_kings_after(match.game.board, move))
        self.assertEqual(match.game.status, 'draw')
        self.assertEqual(match.state.stage, 'GAME_OVER')

    def test_ai_capture_waits_for_camera_then_ends_without_human_turn(self):
        match = PhysicalMatch()
        match.game.board = position((0,7,-2),(7,2,2),(3,4,1))
        match.game.side_to_move = AI
        match.state.stage = 'AI_READY'
        move = next(m for m in legal_moves(match.game.board, AI) if m.captures)
        before = match.game.board
        after = apply_move(before, move)
        match.stage_ai_move(move)
        match.before_robot(signature(before), kings=king_signature(before))
        match.after_robot('mock robot completed')
        self.assertEqual(match.game.status, 'playing')
        match.verify_ai(signature(after), kings=expected_kings_after(before, move))
        self.assertEqual(match.game.status, 'draw')
        self.assertEqual(match.state.stage, 'GAME_OVER')
        self.assertIsNone(match.state.pending)

    def test_other_draw_rules_remain_enabled(self):
        for rule in ('repetition', 'quiet'):
            g = CheckersGame()
            g.board = position((7,0,2),(0,5,-2),(6,3,1))
            move = next(m for m in legal_moves(g.board, HUMAN) if not m.captures)
            if rule == 'repetition':
                g.repetitions[board_key(apply_move(g.board, move), AI)] = 2
            else:
                g.quiet_plies = 79
            g.play_human(move_label(move))
            self.assertEqual(g.status, 'draw')
            self.assertIn('3 ครั้ง' if rule == 'repetition' else '80 ครึ่งตา', g.draw_reason)


class PopupIntegration(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import tkinter as tk
        from tkinter import font
        try:
            cls.root = tk.Tk()
        except tk.TclError as exc:
            raise unittest.SkipTest('Tk display unavailable: '+str(exc))
        import soft_sky_dialogs as ui
        if 'Tahoma' not in font.families(cls.root) and 'Garuda' in font.families(cls.root):
            ui.FONT = 'Garuda'
        cls.tk, cls.ui = tk, ui
        cls.root.geometry('800x600+15+15')
        cls.root.update()

    @classmethod
    def tearDownClass(cls):
        cls.root.destroy()

    def tearDown(self):
        for w in self.root.winfo_children():
            if isinstance(w, self.tk.Toplevel):
                w.destroy()
        self.root.update()

    def assert_centered(self, top):
        self.root.update()
        self.assertLessEqual(abs(top.winfo_x()-(top.winfo_screenwidth()-top.winfo_width())//2), 2)
        self.assertLessEqual(abs(top.winfo_y()-(top.winfo_screenheight()-top.winfo_height())//2), 2)

    def app(self):
        from main import App
        app = object.__new__(App)
        app.root = self.root
        app.invalid_popup = app.invalid_label = None
        app.check_popup = None
        app.winner_shown = False
        app._autosave = Mock()
        return app

    def test_every_state_has_expected_position_and_controls_fit(self):
        for state in self.ui.STATES:
            with self.subTest(state=state):
                top, _ = self.ui.build_dialog(self.root, state)
                if state in ('human_turn', 'ai_turn'):
                    self.assert_bottom_right(top)
                else:
                    self.assert_centered(top)
                card = top._soft_sky_panel.card
                for w in card.winfo_children():
                    self.assertTrue(w.winfo_ismapped())
                    self.assertLessEqual(w.winfo_y()+w.winfo_height(), card.winfo_height())
                top.destroy()

    def test_live_human_status_reuses_window_and_does_not_grab(self):
        app = self.app()
        app._show_checking('เดินหมากได้เลย')
        first = app.check_popup
        self.assertIsNone(self.root.grab_current())
        text = 'รอหยิบหมากที่ถูกกินออก\nตรวจหมากส้มที่ช่อง 49 และ 51\nแล้วนำมือออกจากภาพกล้อง'
        app._show_checking(text)
        self.assertIs(app.check_popup, first)
        self.assertEqual(app.check_label.cget('text'), text)
        self.assert_bottom_right(first)
        app._hide_checking()
        self.assertFalse(first.winfo_exists())

    def test_illegal_warning_retains_grab_and_camera_controlled_dismissal(self):
        app = self.app()
        app._show_illegal('ต้องกินหมากก่อน')
        first = app.invalid_popup
        self.assertEqual(self.root.grab_current(), first)
        self.root.tk.eval(first.protocol('WM_DELETE_WINDOW'))
        self.assertTrue(first.winfo_exists())
        app._show_illegal('กรุณาคืนหมากให้ตรงตำแหน่งเดิม\n'+('ตรวจตำแหน่งหมากและจุดเขียวบนตัวฮอส\n'*5))
        self.assertIs(app.invalid_popup, first)
        self.assert_centered(first)
        app._clear_illegal()
        self.assertFalse(first.winfo_exists())
        self.assertIsNone(self.root.grab_current())

    def test_all_results_show_correct_text_save_history_and_close(self):
        for status, winner, expected in [('win',HUMAN,'คุณชนะ'),('win',AI,'AI ADOM ชนะ!'),('draw',None,'เสมอ!')]:
            with self.subTest(expected=expected):
                app = self.app()
                app.match = SimpleNamespace(state=SimpleNamespace(stage='GAME_OVER'),
                    game=SimpleNamespace(status=status,winner=winner,
                    draw_reason='เหลือฮอสฝ่ายละ 1 ตัว' if status == 'draw' else None))
                app._show_winner()
                windows = [w for w in self.root.winfo_children() if isinstance(w,self.tk.Toplevel)]
                top = windows[-1]
                self.assert_centered(top)
                children = top._soft_sky_panel.card.winfo_children()
                self.assertIn(expected,[w.cget('text') for w in children if isinstance(w,self.tk.Label)])
                app._autosave.assert_called_once()
                next(w for w in children if isinstance(w,self.tk.Button)).invoke()
                self.root.update()
                self.assertFalse(top.winfo_exists())


    def assert_bottom_right(self, top):
        self.root.update()
        left, upper, right, bottom = self.ui._screen_work_area(top)
        fx = max(0, top.winfo_rootx()-top.winfo_x())
        fy = max(0, top.winfo_rooty()-top.winfo_y())
        self.assertLessEqual(abs(top.winfo_x()-max(left,right-top.winfo_width()-2*fx-24)), 2)
        self.assertLessEqual(abs(top.winfo_y()-max(upper,bottom-top.winfo_height()-fy-fx-24)), 2)

    def test_ai_status_reuses_orange_non_modal_corner_popup(self):
        app = self.app()
        app._show_ai_moving('AI กำลังคำนวณตาเดิน...')
        first = app.ai_move_popup
        self.assertEqual(first._soft_sky_state, 'ai_turn')
        self.assertEqual(self.ui.STATES['ai_turn'][4], 'orange')
        self.assertIsNone(self.root.grab_current())
        app._show_ai_moving('MG400 กำลังเดินหมาก\nกรุณารอจนกล้องยืนยันตำแหน่งหมาก')
        self.assertIs(app.ai_move_popup, first)
        self.assertIn('MG400', app.ai_move_label.cget('text'))
        self.assert_bottom_right(first)
        app._hide_ai_moving()
        self.assertFalse(first.winfo_exists())

    def test_ai_popup_follows_calculation_motion_verification_and_cleanup(self):
        app = self.app()
        app.match = SimpleNamespace(state=SimpleNamespace(stage='AI_READY'))
        app.ai_busy, app.robot_busy, app.showing_intro = True, False, False
        app.status = self.tk.StringVar(self.root, 'MG400 กำลังเดินหมาก')
        app._sync_ai_popup()
        self.assertIn('คำนวณ', app.ai_move_label.cget('text'))
        first = app.ai_move_popup
        app.ai_busy, app.robot_busy = False, True
        app.match.state.stage = 'AI_IN_PROGRESS'
        app._sync_ai_popup()
        self.assertIs(app.ai_move_popup, first)
        self.assertEqual(app.ai_move_label.cget('text'), app.status.get())
        app.robot_busy = False
        app.match.state.stage = 'VERIFY_AI'
        app._sync_ai_popup()
        self.assertIn('รอกล้อง', app.ai_move_label.cget('text'))
        for stage in ('HUMAN', 'ERROR', 'GAME_OVER', 'INITIAL'):
            app.match.state.stage = stage
            app._sync_ai_popup()
            self.assertIsNone(app.ai_move_popup)
        app.match.state.stage, app.showing_intro = 'AI_READY', True
        app._sync_ai_popup()
        self.assertIsNone(app.ai_move_popup)

    def test_turn_popups_replace_each_other_without_overlap(self):
        app = self.app()
        app._show_checking('ถึงตาคน')
        human = app.check_popup
        app._show_ai_moving('ถึงตา AI')
        self.assertFalse(human.winfo_exists())
        ai = app.ai_move_popup
        app._show_checking('ถึงตาคนอีกครั้ง')
        self.assertFalse(ai.winfo_exists())
        self.assert_bottom_right(app.check_popup)
        self.assertIsNone(self.root.grab_current())

    def test_turn_popup_respects_usable_desktop_bounds(self):
        top, _ = self.ui.build_dialog(self.root, 'ai_turn')
        with patch('soft_sky_dialogs._screen_work_area', return_value=(100,50,1300,850)):
            self.ui.fit_dialog(top)
            self.assert_bottom_right(top)
            self.assertLess(top.winfo_y()+top.winfo_height(), 850)
            self.assertLess(top.winfo_x()+top.winfo_width(), 1300)


if __name__ == '__main__':
    unittest.main()
