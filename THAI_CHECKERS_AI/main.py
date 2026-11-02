"""Real-board 8x8 checkers: human hand + fixed camera + ADAMS + MG400.

This application has no simulation mode in the game UI. Real robot movement
is disabled until independently verified hardware safety interlocks pass.
The working user-supplied MG400 V4 hardware module is preserved unchanged.
"""
from __future__ import annotations

import secrets
import subprocess
import sys
import time
from datetime import datetime
import threading
from pathlib import Path
from queue import Empty, Queue
try:
    import tkinter as tk
    from tkinter import filedialog, messagebox, simpledialog, ttk
except ImportError:  # Allow headless unit tests to import constants like LEVELS
    tk = filedialog = messagebox = simpledialog = ttk = None

import cv2
import numpy as np
from PIL import Image
try:
    from PIL import ImageTk
except ImportError:
    ImageTk = None

from adams import ADAMS, EvaluationWeights, SearchConfig
from endgame_db import MicroEndgameDB
from engine import BOARD_SIZE as N, apply_move
from excel_history import export_history_xlsx
from game_logic import AI, HUMAN, move_label, number_of
from opening_book import OpeningBook
from physical_match import MatchError, PhysicalMatch, signature, king_signature, expected_kings_after
from robot_io import LiveRunner, RobotFailure, RobotPlanner
from robot_connection import (load_robot_ip,
                              save_robot_ip, probe_mg400)
from tablebase import ExactTwoPieceTablebase
from vision_io import BoardCamera, VisionFailure
import mg400_visionV4 as _v4
from modern_start_panel import ModernStartPanel
from soft_sky_dialogs import build_dialog, fit_dialog, center_dialog
from rules_window import RulesWindow

ROOT = Path(__file__).resolve().parent
# หลังกล้องเห็นกระดานเปลี่ยน ต้องนิ่งอย่างน้อยกี่วินาทีก่อนตรวจกติกา/แจ้งผิดกติกา (ปรับตรงนี้)
INTRO_POPUP_MS = 3000        # ms: popup ผลฝ่ายเริ่มก่อนค้างกี่ ms ก่อนเริ่มเล่น
CHECK_SETTLE_SECONDS = 2.0
VERIFY_STABLE_FRAMES = 3     # หลังหุ่นเดินเสร็จ: กล้องต้องเห็นกระดานตรงตา AI นิ่งกี่เฟรม (เฟรมละ ~0.11 วินาที) ถึงเปิดให้คนเล่น
VERIFY_SETTLE_SECONDS = 0.0  # วินาที: หน่วงเพิ่มหลังหุ่นเสร็จ (0 = ไม่หน่วง; ตาคนยังใช้ CHECK_SETTLE_SECONDS เดิม)
PENDING_STRIKES_LIMIT = 8   # รอบตรวจที่ยอมให้ "รอหยิบหมากที่ถูกกินออก/เปลี่ยนฮอส" ก่อนขึ้นจอแดงผิดกติกา (รอบละราว 2+ วินาที)
# Three supported AI difficulty levels; keep the selector and engine in sync.
LEVELS = {
    'ง่าย': (1, 100),
    'ปานกลาง': (5, 550),
    'ยาก': (12, 2000),
}
WHITE = '#F5FAFF'
SKY = '#6BB9EC'
INK = '#153653'
ORANGE = '#F18C30'
BLACK = '#24262E'
BACKGROUND = '#E6F2FF'
FONT = 'Leelawadee UI'
CARD = '#FFFFFF'
PALE = '#ECF5FE'
PRIMARY = '#0678E8'
NAVY = '#112F58'


def build_ai(level):
    depth, millis = LEVELS[level]
    weights = EvaluationWeights.from_file(ROOT / 'data/adams_weights.json')
    book = OpeningBook(ROOT / 'data/opening_book_8x8.json')
    tablebase = ExactTwoPieceTablebase(ROOT / 'data/tablebase_8x8.sqlite3')
    if not tablebase.is_built():
        tablebase = None
    endgame = MicroEndgameDB(ROOT / 'data/endgames_8x8.sqlite3', max_pieces=4,
                            proof_depth=22, node_limit=30000)
    return ADAMS(SearchConfig(max_depth=depth, time_limit_ms=millis), weights=weights,
                 opening_book=book, tablebase=tablebase, endgame_db=endgame)


class App:
    def __init__(self, root, *, enable_robot_diagnostics=True):
        self.root = root
        self.enable_robot_diagnostics = enable_robot_diagnostics
        root.title('AI ADOM VS YOU')
        try:
            root.iconbitmap(str(ROOT / 'assets' / 'AI_ADOM_VS_YOU.ico'))
        except tk.TclError:
            pass
        root.geometry('1536x960')
        root.minsize(1160, 760)
        root.configure(bg=BACKGROUND)
        self.match = PhysicalMatch()
        self.camera = None
        self.last_frame = None
        self.photo = None
        self.queue = Queue()
        self.generation = 0
        self.stop_event = threading.Event()
        self.ai_busy = False
        self.robot_busy = False
        self.robot_probe = None
        self._connection_check_running = False
        self._next_robot_check = 0.0
        try:
            self.robot_ip = load_robot_ip()
            self.robot_config_error = ''
        except ValueError as exc:
            self.robot_ip = ''
            self.robot_config_error = str(exc)
        self.robot_connection_text = tk.StringVar(value='MG400: ยังไม่ได้ตรวจสอบ')
        self.game_started_at = None
        self.pending_ai = None
        self.showing_intro = False
        self.auto_robot_started = False
        self.last_seen = None
        self.stable_count = 0
        self.invalid_popup = None
        self.invalid_label = None
        self.winner_shown = False
        self.selected_level = tk.StringVar(value='')  # user must choose explicitly
        self.actual_level = None
        self.camera_seen = None
        self.camera_seen_kings = frozenset()
        self.monitor_only = False
        self.history_times = {}
        self._rendered_history_key = None
        self._board_redraw_job = None
        self.status = tk.StringVar(value='เลือกความยากก่อนกดเริ่มเกม')
        self.saving_path = ROOT / 'history_physical_match.xlsx'
        self.start_page = None
        self.play_page = None
        self._show_start()
        root.after(100, self._tick)
        if enable_robot_diagnostics:
            root.after(150, self._check_robot_connection)
        root.protocol('WM_DELETE_WINDOW', self._close)

    def _button(self, master, label, fn, primary=False):
        return tk.Button(master, text=label, command=fn, font=('Tahoma', 11, 'bold'),
                         bg='#1878C9' if primary else '#E3F1FD',
                         fg='#FFFFFF' if primary else INK,
                         activebackground='#0C64AD', activeforeground='#FFFFFF',
                         relief='flat', padx=14, pady=10, cursor='hand2')

    def _obs_to_board(self, seen):
        board = [[0 for _ in range(N)] for _ in range(N)]
        if not seen:
            return board
        from game_logic import coord_of
        for cell, color in seen.items():
            try:
                r, c = coord_of(cell)
            except ValueError:
                continue
            board[r][c] = (2 if self.camera is not None and cell in self.camera.last_kings else 1) * (1 if color == 'black' else -1)
        return board


    def _refresh_history_table(self):
        if not hasattr(self, 'history_table'):
            return
        entries = self.match.game.history
        key = tuple((entry.ply, entry.side, tuple(entry.move.path), entry.captures, entry.promoted)
                    for entry in entries)
        if key == self._rendered_history_key:
            return
        self._rendered_history_key = key
        for row in self.history_table.get_children():
            self.history_table.delete(row)
        for entry in reversed(entries):
            cells = [str(number_of(coord)) for coord in entry.move.path]
            player = 'AI (สีส้ม)' if entry.side == AI else 'คน (สีดำ)'
            notes = []
            if entry.captures:
                notes.append('กินช่อง ' + ', '.join(map(str, entry.captures)))
            if entry.promoted:
                notes.append('เลื่อนเป็นฮอส')
            if entry.ply not in self.history_times:
                self.history_times[entry.ply] = datetime.now().strftime('%H:%M:%S')
            tag = 'ai' if entry.side == AI else 'human'
            self.history_table.insert('', 'end', values=(entry.ply, player, cells[0], cells[-1],
                                        ' / '.join(notes) if notes else 'เดินหมาก',
                                        self.history_times[entry.ply]), tags=(tag,))
        if hasattr(self, 'empty_history'):
            if entries:
                self.empty_history.place_forget()
            else:
                self.empty_history.place(relx=.5, rely=.55, anchor='center')

    def _small_action_button(self, parent, text, command, *, bg, fg, active):
        return tk.Button(parent, text=text, command=command, font=('Tahoma', 10, 'bold'),
                         bg=bg, fg=fg, activebackground=active, activeforeground=fg,
                         relief='flat', bd=0, cursor='hand2', padx=4, pady=5,
                         highlightthickness=1, highlightbackground=active)

    def _schedule_board_redraw(self):
        if not self.play_page or not self.play_page.winfo_exists():
            return
        if self._board_redraw_job is not None:
            self.root.after_cancel(self._board_redraw_job)
        self._board_redraw_job = self.root.after(55, self._board_draw)

    def _draw_board(self, canvas, board, size, ox=25, oy=25, show_labels=True):
        """Render the same numbered physical-board orientation everywhere."""
        canvas.delete('all')
        for r in range(N):
            for c in range(N):
                n = number_of((r, c))
                x, y = ox + c * size, oy + r * size
                playable = (r + c) % 2 == 1
                canvas.create_rectangle(x, y, x + size, y + size,
                                        fill='#66B9EE' if playable else '#F9FCFF',
                                        outline='#B2D2EA', width=1)
                canvas.create_text(x + 12, y + 14, text=str(n),
                                   fill=NAVY if playable else '#6784A6',
                                   font=(FONT, max(9, size // 6), 'bold'))
                piece = board[r][c]
                if piece:
                    color = '#20232C' if piece > 0 else '#FF8B1D'
                    trim = '#111B30' if piece > 0 else '#CB5B05'
                    d = size * 0.28
                    x1, y1 = x + d / 2 + 1, y + d / 2 + 2
                    x2, y2 = x + size - d / 2 - 1, y + size - d / 2
                    canvas.create_oval(x1 + 3, y1 + 5, x2 + 3, y2 + 5,
                                       fill='#84B2CD', outline='')
                    canvas.create_oval(x1, y1, x2, y2, fill=color,
                                       outline='#FFFFFF', width=3)
                    canvas.create_oval(x1 + 6, y1 + 6, x2 - 6, y2 - 6,
                                       fill=color, outline=trim, width=2)
                    if abs(piece) == 2:
                        # UI-only king marker: a compact green dot, not a star.
                        # This changes only the digital board rendering; camera/MG400
                        # king detection and physical marker logic remain untouched.
                        cx, cy = x + size / 2, y + size / 2
                        dot_r = max(3, min(5, int(size * 0.075)))
                        canvas.create_oval(cx - dot_r, cy - dot_r, cx + dot_r, cy + dot_r,
                                           fill='#16A34A', outline='#E7F9EC', width=1)
        if show_labels:
            canvas.create_text(ox + N // 2 * size, oy - 12, text='ด้านบน: AI สีส้ม',
                               font=(FONT, 10), fill=INK)
            canvas.create_text(ox + N // 2 * size, oy + N * size + 14,
                               text='ด้านล่าง: คน สีดำ', font=(FONT, 10), fill=INK)

    def _paint_difficulty(self):
        if hasattr(self, 'modern_start_panel') and self.start_page is not None:
            self.modern_start_panel.update_selection()

    def _choose_difficulty(self, level):
        self.selected_level.set(level)
        self._paint_difficulty()
        self._difficulty_changed()

    def _show_start(self):
        if self.play_page:
            self.play_page.destroy()
            self.play_page = None
        frm = tk.Frame(self.root, bg=BACKGROUND)
        frm.pack(fill='both', expand=True)
        self.start_page = frm

        # Approved visual banner (UI resource only); no game or robot logic changed.
        banner = tk.Label(frm, bg=BACKGROUND, bd=0, highlightthickness=0)
        banner.pack(fill='x')
        source = ROOT / 'assets' / 'title_banner.png'
        self._banner_source = Image.open(source).convert('RGB') if source.is_file() else None
        if self._banner_source is not None:
            def resize_header(event):
                w = max(event.width, 300)
                h = min(188, max(118, int(w * 186 / 1672)))
                picture = self._banner_source.resize((w, h), Image.Resampling.LANCZOS)
                # Lower the approved banner artwork slightly so the center title
                # does not sit too high, without changing any game logic.
                top_pad = max(8, min(18, h // 14))
                composed = Image.new('RGB', (w, h + top_pad), BACKGROUND)
                composed.paste(picture, (0, top_pad))
                self._banner_image = ImageTk.PhotoImage(composed)
                banner.configure(image=self._banner_image, height=h + top_pad)
            banner.bind('<Configure>', resize_header)
        else:
            tk.Label(banner, text='หมากฮอส 8 x 8', font=(FONT, 30, 'bold'),
                     bg=BACKGROUND, fg=NAVY).pack(pady=(12, 0))
            tk.Label(banner, text='AI ADOM VS YOU', font=(FONT, 20, 'bold'),
                     bg=BACKGROUND, fg='#1367E5').pack(pady=(0, 10))

        container = tk.Frame(frm, bg=BACKGROUND, padx=28, pady=15)
        container.pack(fill='both', expand=True)
        container.grid_columnconfigure(0, weight=1, uniform='home')
        container.grid_columnconfigure(1, weight=1, uniform='home')
        container.grid_rowconfigure(0, weight=1)

        left = tk.Frame(container, bg=CARD, padx=18, pady=10,
                        highlightbackground='#D6E7F6', highlightthickness=1)
        left.grid(row=0, column=0, sticky='nsew', padx=(0, 12))
        top_info = tk.Frame(left, bg='#EEF6FF')
        top_info.pack(fill='x', pady=(0, 4))
        tk.Label(top_info, text='ฝ่ายบน: AI ADOM', font=(FONT, 14, 'bold'),
                 bg='#EEF6FF', fg=NAVY).pack(anchor='center', pady=(3, 0))
        tk.Label(top_info, text='เดินหมากสีส้ม (AI)', font=(FONT, 10),
                 bg='#EEF6FF', fg='#557190').pack(anchor='center', pady=(0, 3))
        canvas_outer = tk.Frame(left, bg=CARD)
        canvas_outer.pack(pady=(7, 7))
        # All eight rows plus both side captions must fit in the left card.
        # Initial canvas dimensions and the first draw use the same square size.
        initial_square = 48
        board_canvas = tk.Canvas(canvas_outer, bg=CARD,
                                 width=initial_square * N + 28,
                                 height=initial_square * N + 28,
                                 borderwidth=0, highlightthickness=0)
        board_canvas.pack()
        def resize_home_board(event):
            # Use more of the left card for the 8x8 board, retaining enough
            # space for both captions and all eight rows on smaller screens.
            square = max(24, min(85, (event.width - 84) // N,
                                 (event.height - 188) // N))
            if getattr(self, '_home_board_square', None) != square:
                self._home_board_square = square
                board_canvas.configure(width=square * N + 28, height=square * N + 28)
                self._draw_board(board_canvas, self.match.game.board, square,
                                 14, 14, show_labels=False)
        self._home_board_square = initial_square
        left.bind('<Configure>', resize_home_board)
        self._draw_board(board_canvas, self.match.game.board, initial_square,
                         14, 14, show_labels=False)
        bottom_info = tk.Frame(left, bg='#EEF6FF')
        bottom_info.pack(fill='x', pady=(4, 0))
        tk.Label(bottom_info, text='ฝ่ายล่าง: คุณ (ผู้เล่น)', font=(FONT, 14, 'bold'),
                 fg=NAVY, bg='#EEF6FF').pack(anchor='center', pady=(3, 0))
        tk.Label(bottom_info, text='เดินหมากสีดำ (คน)', font=(FONT, 10),
                 bg='#EEF6FF', fg='#557190').pack(anchor='center', pady=(0, 3))

        # Redesigned right-side start panel. Banner and left board are unchanged.
        # All buttons call the existing game, diagnostic, settings and exit methods.
        self.modern_start_panel = ModernStartPanel(
            container,
            selected_level=self.selected_level,
            robot_connection_text=self.robot_connection_text,
            status_text=self.status,
            probe_getter=lambda: self.robot_probe,
            choose_level=self._choose_difficulty,
            check_connection=self._check_robot_connection,
            start_game=self._start_game,
            show_rules=self._rules,
            show_settings=self._settings,
            exit_program=self._close,
        )
        self.modern_start_panel.grid(row=0, column=1, sticky='nsew', padx=(12, 0))

        # Start-screen guidance now lives inside the scalable right control card.
        # No footer is packed beneath the two panels, so short windows cannot
        # crop it beyond the application boundary.

    def _settings(self):
        top = tk.Toplevel(self.root)
        top.title('ตั้งค่ากล้อง / MG400')
        top.geometry('900x860')
        top.configure(bg=CARD)
        top.transient(self.root)
        tk.Label(top, text='ตั้งค่ากล้องและ MG400 เครื่องจริง',
                 font=(FONT, 18, 'bold'), bg=CARD, fg=NAVY).pack(pady=(22, 15))
        # ---- ไฟล์สอบเทียบ: สร้างได้จากปุ่มในหน้านี้ (รัน mg400_visionV4.py ตามโหมด ในหน้าต่างคอนโซลแยก) ----
        setup_rows = (('ไฟล์สอบเทียบกล้อง', 'calib.npz', 'calibrate', 'สอบเทียบกล้อง',
                       'คลิกจุดบนภาพ + ลากหัวดูดไปจุดนั้น กด S ทีละจุด (ไม่สั่งแขนเคลื่อนที่)'),
                      ('จุดหลีกกล้อง', 'clear_pos.npz', 'setclear', 'ตั้งจุดหลีก',
                       'ลากหัวดูดไปจุดหลีกกล้อง แล้วกด Enter ในหน้าต่างคอนโซล'),
                      ('สอบเทียบหมายเลขช่อง', 'board.npz', 'boardcal', 'สอบเทียบกระดาน',
                       'วางหมากตำแหน่งเริ่มเกม (ดำ 8 ตัวล่าง ส้ม 8 ตัวบน) ก่อนกด'),
                      ('จุดทิ้งหมาก', 'discard_pos.npz', 'setdiscard', 'ตั้งจุดทิ้ง',
                       'ลากหัวดูดไปเหนือจุดทิ้ง แล้วกด Enter'),
                      ('กรอบตรวจจับ', 'roi.npz', 'setroi', 'ลากกรอบ',
                       'ลากครอบกระดาน + ที่เก็บฮอสซ้าย/ขวา (ตัดมือ/ถาดทิ้งออก)'),
                      ('จุดเก็บฮอส', 'stock_pos.npz', 'setstock', 'หาจากกล้อง',
                       'วางฮอสบนจุดเขียวครบ 8 จุดก่อนกด'))
        status_labels = {}
        buttons = []

        def file_status(fname):
            return ('✓ มีไฟล์แล้ว' if (ROOT / fname).is_file() else '✗ ยังไม่มี — กดปุ่มทางขวาเพื่อสร้าง')

        def refresh_status():
            for fname, lab in status_labels.items():
                try:
                    lab.config(text=file_status(fname), fg='#1B7F3A' if (ROOT / fname).is_file() else '#B3261E')
                except tk.TclError:
                    pass

        def run_mode(mode, title, hint):
            if self.robot_busy or self.match.state.stage not in ('INITIAL', 'GAME_OVER', 'ERROR'):
                messagebox.showwarning('กำลังเล่น', 'ห้ามสอบเทียบระหว่างเกม/หุ่นกำลังทำงาน', parent=top)
                return
            if not messagebox.askokcancel(
                    title,
                    f'{hint}\n\n'
                    'โปรแกรมจะเปิดหน้าต่างคอนโซลแยก (ทำตามข้อความในหน้าต่างนั้น) และปิดกล้องของ Dashboard ชั่วคราว\n'
                    'โหมดนี้อ่านตำแหน่งแขนอย่างเดียว ไม่สั่งแขนเคลื่อนที่ ต้องลากแขนด้วยมือเมื่อโปรแกรมบอก\n'
                    'ทำต่อหรือไม่?', parent=top):
                return
            if self.camera is not None:      # กล้อง USB เปิดซ้อนกันไม่ได้
                try:
                    self.camera.close()
                except Exception:
                    pass
                self.camera = None
            flags = getattr(subprocess, 'CREATE_NEW_CONSOLE', 0)
            try:
                import os as _os
                proc = subprocess.Popen([sys.executable, str(ROOT / 'mg400_visionV4.py'), mode],
                                        cwd=str(ROOT), creationflags=flags,
                                        env={**_os.environ, 'MG400_PAUSE_ON_ERROR': '1'})
            except OSError as exc:
                messagebox.showerror('เปิดโหมดไม่สำเร็จ', str(exc), parent=top)
                return
            for b in buttons:
                b.config(state='disabled')

            def wait():
                if proc.poll() is None:
                    try:
                        top.after(700, wait)
                    except tk.TclError:
                        pass
                    return
                _v4._ROI_CACHE.clear()       # โหลดกรอบ/ไฟล์ใหม่
                try:
                    for b in buttons:
                        b.config(state='normal')
                    refresh_status()
                    messagebox.showinfo(title, 'โหมดเสร็จแล้ว' if proc.returncode == 0 else
                                        f'โหมดจบด้วยรหัส {proc.returncode} (ดูข้อความในหน้าต่างคอนโซล; ถ้าเป็นโหมดอื่นที่ไม่ใช่ calibrate ต้องสอบเทียบกล้อง-หุ่นด้วย calibrate ก่อน)', parent=top)
                except tk.TclError:
                    pass
            top.after(700, wait)

        for label, fname, mode, btn_text, hint in setup_rows:
            row = tk.Frame(top, bg='#EDF6FF', padx=12, pady=6)
            row.pack(fill='x', padx=25, pady=2)
            tk.Label(row, text=label + ':', font=(FONT, 10, 'bold'),
                     bg='#EDF6FF', fg=NAVY, width=22, anchor='w').pack(side='left')
            lab = tk.Label(row, text='', font=(FONT, 10), bg='#EDF6FF', anchor='w', width=34)
            lab.pack(side='left')
            status_labels[fname] = lab
            b = tk.Button(row, text=btn_text, font=(FONT, 10, 'bold'), bg='#D8EAFB', fg=NAVY,
                          relief='flat', padx=10,
                          command=lambda m=mode, t=btn_text, h=hint: run_mode(m, t, h))
            b.pack(side='right')
            buttons.append(b)
            if mode == 'setstock':
                b2 = tk.Button(row, text='สอนจุดหยิบ', font=(FONT, 10, 'bold'), bg='#D8EAFB', fg=NAVY,
                               relief='flat', padx=10,
                               command=lambda: run_mode('teachstock', 'สอนจุดหยิบฮอส',
                                                        'ลากหัวดูดไปกลางฮอสแต่ละตัว (ส้ม 4 แล้วดำ 4) แล้วกด Enter ทีละตัว'))
                b2.pack(side='right', padx=(0, 6))
                buttons.append(b2)
        refresh_status()
        for label, value in (('กล้อง', 'กล้อง USB 0, หมุนภาพ 180° ตามไฟล์ V4'),
                             ('อนุมัติการใช้ MG400 จริง', 'live_safety_config.json — ตรวจอุปกรณ์จริงก่อน'),
                             ('การป้องกัน', 'E-stop และระบบกั้นพื้นที่ต้องเป็นอุปกรณ์จริง')):
            row = tk.Frame(top, bg='#EDF6FF', padx=12, pady=7)
            row.pack(fill='x', padx=25, pady=2)
            tk.Label(row, text=label + ':', font=(FONT, 10, 'bold'),
                     bg='#EDF6FF', fg=NAVY, width=22, anchor='w').pack(side='left')
            tk.Label(row, text=value, font=(FONT, 10),
                     bg='#EDF6FF', fg='#405F84', anchor='w').pack(side='left')
        ip_row = tk.Frame(top, bg=CARD)
        ip_row.pack(fill='x', padx=25, pady=(12, 3))
        tk.Label(ip_row, text='IP MG400:', bg=CARD, fg=NAVY,
                 font=(FONT, 11, 'bold')).pack(side='left')
        ip_value = tk.StringVar(value=self.robot_ip)
        tk.Entry(ip_row, textvariable=ip_value, font=(FONT, 12), width=19).pack(
            side='left', padx=10)
        def save_ip():
            if self.robot_busy or self.match.state.stage not in ('INITIAL', 'GAME_OVER', 'ERROR'):
                messagebox.showwarning('กำลังเล่น', 'ห้ามเปลี่ยน IP ระหว่างตาเกม', parent=top)
                return
            try:
                self.robot_ip = save_robot_ip(ip_value.get())
            except (ValueError, OSError) as exc:
                messagebox.showerror('IP ไม่ถูกต้อง', str(exc), parent=top)
                return
            self.robot_config_error = ''
            self.robot_probe = None
            self._next_robot_check = 0
            self._set_robot_connection_label()
            self._check_robot_connection()
            messagebox.showinfo('บันทึก IP แล้ว',
                                f'ใช้ IP {self.robot_ip} สำหรับการตรวจเชื่อมต่อและส่งคำสั่งจริง\n'
                                'การตรวจเชื่อมต่อไม่สั่งให้แขนเคลื่อนที่', parent=top)
        self._button(ip_row, 'บันทึก IP / ตรวจ', save_ip).pack(side='left', padx=5)
        tk.Label(top, text='ตรวจเฉพาะ TCP 29999 + 30003 / RobotMode(); ไม่ Enable ไม่ขยับแขน',
                 bg=CARD, fg='#56718D', font=(FONT, 9)).pack(pady=(3, 0))
        self._button(top, 'กลับหน้าแรก', top.destroy, primary=True).pack(pady=12)
        center_dialog(top)

    def _difficulty_changed(self):
        enabled = bool(self.selected_level.get())
        if self.start_page is not None and hasattr(self, 'modern_start_panel'):
            self.modern_start_panel.update_selection()
        if enabled:
            self.status.set('เลือกระดับ ' + self.selected_level.get() +
                            ' แล้ว — จัดหมากฝ่ายละ 8 ตัว และตรวจความพร้อม MG400')
        else:
            self.status.set('เลือกระดับความยากก่อนกดเริ่มเกม')

    def _show_play(self):
        if self.start_page:
            self.start_page.destroy()
            self.start_page = None
        if self.play_page:
            self.play_page.destroy()
            self.play_page = None
        self._rendered_history_key = None
        self._board_redraw_job = None

        p = tk.Frame(self.root, bg='#E6F2FC')
        p.pack(fill='both', expand=True)
        self.play_page = p

        # Blue title bar, matching the approved mock-up.
        top = tk.Frame(p, bg='#0B4B91', height=79)
        top.pack(fill='x')
        top.pack_propagate(False)
        emblem = tk.Canvas(top, width=54, height=54, highlightthickness=0, bg='#0B4B91')
        emblem.pack(side='left', padx=(20, 13), pady=12)
        emblem.create_oval(3, 3, 51, 51, fill='white', outline='#D9E7F6')
        for cx, cy, fill in ((22, 32, '#FF921E'), (31, 21, '#222832'), (39, 33, '#FF921E')):
            emblem.create_oval(cx-8, cy-8, cx+8, cy+8, fill=fill, outline='#DCE9F6', width=1)
        title = tk.Frame(top, bg='#0B4B91')
        title.pack(side='left', pady=10)
        tk.Label(title, text='AI ADOM VS YOU', font=('Tahoma', 23, 'bold'),
                 fg='white', bg='#0B4B91').pack(anchor='w')
        tk.Label(title, text='หมากฮอส 8 x 8 | กระดานจริง', font=('Tahoma', 10),
                 fg='#D8E8FB', bg='#0B4B91').pack(anchor='w')
        self.level_label = tk.Label(top, text=f'ADOM: {self.actual_level or "-"}   |   MG400: เครื่องจริง',
                                   bg='#103E79', fg='white', font=('Tahoma', 11, 'bold'),
                                   padx=16, pady=11)
        self.level_label.pack(side='right', padx=20, pady=17)

        layout = tk.Frame(p, bg='#E6F2FC')
        layout.pack(fill='both', expand=True, padx=10, pady=(10, 7))
        layout.grid_columnconfigure(0, weight=35, uniform='top')
        layout.grid_columnconfigure(1, weight=40, uniform='top')
        layout.grid_columnconfigure(2, weight=25, uniform='top')
        layout.grid_rowconfigure(0, weight=7)
        layout.grid_rowconfigure(1, weight=3, minsize=180)

        def card(row, column, *, span=1, padx=(0, 0), pady=(0, 0)):
            box = tk.Frame(layout, bg='white', highlightthickness=1, highlightbackground='#C9DEEF')
            box.grid(row=row, column=column, columnspan=span, sticky='nsew', padx=padx, pady=pady)
            return box

        # Left: camera fills most of its card, without forcing the panel to grow.
        camera_card = card(0, 0, padx=(0, 6), pady=(0, 8))
        cam_header = tk.Frame(camera_card, bg='white')
        cam_header.pack(fill='x', padx=15, pady=(13, 3))
        tk.Label(cam_header, text='▣  ภาพจากกล้อง', font=('Tahoma', 14, 'bold'),
                 bg='white', fg='#114481').pack(side='left')
        self.camera_live_label = tk.Label(cam_header, text='● รอเชื่อมต่อ',
                                          font=('Tahoma', 9, 'bold'), bg='white', fg='#A06B16')
        self.camera_live_label.pack(side='right')
        tk.Label(camera_card, text='แสดงภาพมุมมองกล้องแบบเรียลไทม์', font=('Tahoma', 9),
                 bg='white', fg='#617B9E').pack(anchor='w', padx=17)
        cam_well = tk.Frame(camera_card, bg='#16283C')
        cam_well.pack(fill='both', expand=True, padx=13, pady=(10, 12))
        self.video = tk.Label(cam_well, text='รอภาพจากกล้อง', fg='#DFECFA', bg='#16283C',
                              font=('Tahoma', 13))
        self.video.pack(fill='both', expand=True)

        # Centre: responsive 8x8 board, with no clipping at the bottom.
        board_card = card(0, 1, padx=(6, 6), pady=(0, 8))
        row_header = tk.Frame(board_card, bg='white')
        row_header.pack(fill='x', padx=15, pady=(13, 3))
        tk.Label(row_header, text='▦  กระดานจากกล้อง', bg='white', fg='#114481',
                 font=('Tahoma', 14, 'bold')).pack(side='left')
        legend = tk.Frame(row_header, bg='#F2F8FE', padx=7, pady=3)
        legend.pack(side='right')
        tk.Label(legend, text='● AI ส้ม  ● คน ดำ', font=('Tahoma', 9, 'bold'),
                 fg='#204872', bg='#F2F8FE').pack()
        tk.Label(board_card, text='อัปเดตตำแหน่งหมากที่ตรวจพบจากภาพกล้อง',
                 font=('Tahoma', 9), bg='white', fg='#617B9E').pack(anchor='w', padx=17)
        self.board_canvas = tk.Canvas(board_card, bg='white', highlightthickness=0, bd=0)
        self.board_canvas.pack(fill='both', expand=True, padx=12, pady=(6, 2))
        self.board_canvas.bind('<Configure>', lambda _e: self._schedule_board_redraw())
        self.turn_label = tk.Label(board_card, bg='white', fg='#385879',
                                   font=('Tahoma', 9), anchor='w', justify='left', wraplength=500)
        self.turn_label.pack(fill='x', padx=16, pady=(0, 10))

        # Right: a compact robot dashboard and four small, fully visible controls.
        side = card(0, 2, padx=(6, 0), pady=(0, 8))
        tk.Label(side, text='⚙  สถานะระบบ', bg='white', fg='#114481',
                 font=('Tahoma', 14, 'bold')).pack(anchor='w', padx=12, pady=(10, 5))
        self.camera_status = tk.StringVar(value='รอเชื่อมต่อ')
        self.robot_status_view = tk.StringVar(value='ยังไม่ตรวจสอบ')
        self.ai_status_view = tk.StringVar(value=f'ระดับ: {self.actual_level or "-"}')
        self.excel_status_view = tk.StringVar(value='ยังไม่มีประวัติ')

        def status_row(label_text, variable, color):
            r = tk.Frame(side, bg='white')
            r.pack(fill='x', padx=12, pady=2)
            tk.Label(r, text=label_text, bg='white', fg='#345576', font=('Tahoma', 10),
                     anchor='w').pack(side='left')
            val = tk.Label(r, textvariable=variable, bg=color, fg='#245482',
                           font=('Tahoma', 9, 'bold'), padx=7, pady=3)
            val.pack(side='right')
            return val

        self._camera_badge = status_row('▣  กล้อง', self.camera_status, '#E9F2FD')
        self._robot_badge = status_row('⚙  MG400', self.robot_status_view, '#E9F2FD')
        tk.Label(side, textvariable=self.robot_connection_text, font=('Tahoma', 8),
                 bg='white', fg='#506C88', anchor='w', justify='left',
                 wraplength=330).pack(fill='x', padx=12, pady=(0, 2))
        status_row('◉  AI (ADOM)', self.ai_status_view, '#E5F1FF')
        status_row('▤  บันทึก Excel', self.excel_status_view, '#F1F5FA')
        tk.Frame(side, bg='#DFE9F4', height=1).pack(fill='x', padx=12, pady=(5, 7))
        tk.Label(side, text='สถานะเกม', font=('Tahoma', 13, 'bold'),
                 bg='white', fg='#114481').pack(anchor='w', padx=12)
        self.game_state_label = tk.Label(side, text='รอเริ่มเกม', bg='#E5F0FF',
                                        fg='#225FAD', font=('Tahoma', 10, 'bold'),
                                        anchor='w', padx=7, pady=5, wraplength=195,
                                        justify='left')
        self.game_state_label.pack(fill='x', padx=12, pady=(5, 6))
        # Independent, unambiguous indicator: the UI can be open without a live game.
        self.game_live_label = tk.Label(side, text='● ยังไม่เริ่มเกม', bg='#FFF2D8',
                                        fg='#825500', font=('Tahoma', 11, 'bold'),
                                        anchor='w', padx=7, pady=6)
        self.game_live_label.pack(fill='x', padx=12, pady=(0, 5))
        tk.Frame(side, bg='#DFE9F4', height=1).pack(fill='x', padx=12, pady=(2, 6))
        tk.Label(side, text='คำสั่งเพิ่มเติม', bg='white', fg='#114481',
                 font=('Tahoma', 13, 'bold')).pack(anchor='w', padx=12, pady=(0, 4))
        controls = tk.Frame(side, bg='white')
        controls.pack(fill='x', padx=10, pady=(0, 5))
        controls.grid_columnconfigure(0, weight=1)
        controls.grid_columnconfigure(1, weight=1)
        buttons = (
            ('บันทึก Excel', self._export, '#E2F5E9', '#147A43', '#BFE8CF'),
            ('ดูกติกา', self._rules, '#E8F1FE', '#135EA8', '#C7E0FC'),
            ('เริ่มเกมใหม่', self._restart, '#D8EAFE', '#145DAA', '#A8D1FF'),
            ('หยุดการทำงาน', self._request_stop, '#FCE7E7', '#B92A2D', '#F4C3C4'),
        )
        self.action_buttons = []
        for i, (label, fn, bg, fg, active) in enumerate(buttons):
            b = self._small_action_button(controls, label, fn, bg=bg, fg=fg, active=active)
            b.grid(row=i//2, column=i%2, sticky='ew', padx=3, pady=4)
            self.action_buttons.append(b)
        diag_controls = tk.Frame(side, bg='white')
        diag_controls.pack(fill='x', padx=12, pady=(1, 5))
        diag_controls.grid_columnconfigure(0, weight=1)
        diag_controls.grid_columnconfigure(1, weight=1)
        self._small_action_button(diag_controls, '↻ ตรวจ MG400',
                                  self._check_robot_connection, bg='#EAF3FD',
                                  fg='#145DAA', active='#CADFF8').grid(
                                  row=0, column=0, sticky='ew', padx=2, pady=2)
        self._small_action_button(diag_controls, '⚙ ตั้งค่า IP', self._settings,
                                  bg='#EAF3FD', fg='#145DAA', active='#CADFF8').grid(
                                  row=0, column=1, sticky='ew', padx=2, pady=2)
        self.game_begin_button = self._small_action_button(
            diag_controls, '▶ ตรวจความพร้อม / เริ่มเกมจริง', self._start_game,
            bg='#E7F4E8', fg='#176C36', active='#C6E9D0')
        self.robot_retry_button = self._small_action_button(
            diag_controls, '▶ ทำตา AI ต่อ', self._retry_robot,
            bg='#E7F4E8', fg='#176C36', active='#C6E9D0')

        # Bottom: one uninterrupted history table across the whole window.
        hist = card(1, 0, span=3, padx=(0, 0), pady=(5, 0))
        hist_header = tk.Frame(hist, bg='white')
        hist_header.pack(fill='x', padx=17, pady=(11, 6))
        tk.Label(hist_header, text='▤  ประวัติการเดินของคนและ AI', bg='white',
                 fg='#114481', font=('Tahoma', 14, 'bold')).pack(side='left')
        tk.Label(hist_header, text='รายการล่าสุดอยู่ด้านบน', bg='white',
                 fg='#617B9E', font=('Tahoma', 9)).pack(side='right')

        st = ttk.Style()
        st.configure('RealCheckers.Treeview', background='#FBFDFF', fieldbackground='#FBFDFF',
                     foreground='#20456E', rowheight=27, borderwidth=0, font=('Tahoma', 10))
        st.configure('RealCheckers.Treeview.Heading', background='#DCEAFB', foreground='#154479',
                     font=('Tahoma', 10, 'bold'))
        st.map('RealCheckers.Treeview', background=[('selected', '#D8EAFD')],
               foreground=[('selected', '#154479')])
        cols = ('ply', 'player', 'src', 'dst', 'note', 'clock')
        wrap = tk.Frame(hist, bg='white')
        wrap.pack(fill='both', expand=True, padx=17, pady=(0, 12))
        self.history_table = ttk.Treeview(wrap, columns=cols, show='headings',
                                          style='RealCheckers.Treeview', height=5)
        widths = (85, 200, 105, 105, 520, 150)
        for name, heading, w in zip(cols, ('ลำดับ','ผู้เล่น','จาก','ไป','รายละเอียด','เวลา'), widths):
            self.history_table.heading(name, text=heading)
            self.history_table.column(name, width=w,
                                      anchor='w' if name=='note' else 'center',
                                      stretch=name=='note')
        self.history_table.tag_configure('ai', foreground='#D06A0C')
        self.history_table.tag_configure('human', foreground='#1C3956')
        scrollbar = ttk.Scrollbar(wrap, orient='vertical', command=self.history_table.yview)
        self.history_table.configure(yscrollcommand=scrollbar.set)
        self.history_table.pack(side='left', fill='both', expand=True)
        scrollbar.pack(side='right', fill='y')
        self.empty_history = tk.Label(wrap, text='ยังไม่มีประวัติการเดิน\nเริ่มเกมเพื่อแสดงรายการเดินหมาก',
                                      bg='#FBFDFF', fg='#748EA9', font=('Tahoma', 11),
                                      justify='center')
        self.empty_history.place(relx=.5, rely=.55, anchor='center')
        self.history_table.bind('<Configure>', lambda _e: self._position_empty_history())

        footer = tk.Frame(p, bg='#EAF3FB', height=32)
        footer.pack(fill='x', padx=10, pady=(0, 5))
        footer.pack_propagate(False)
        self.status_left = tk.Label(footer, textvariable=self.status, bg='#EAF3FB',
                                    fg='#496E97', font=('Tahoma', 9), anchor='w')
        self.status_left.pack(side='left', fill='x', expand=True, padx=6)
        self.status_right = tk.Label(footer, text='● รอการตรวจสอบ', bg='#EAF3FB',
                                     fg='#758DAB', font=('Tahoma', 9, 'bold'))
        self.status_right.pack(side='right', padx=8)
        self._board_draw()
        self.root.after(100, self._board_draw)

    def _position_empty_history(self):
        if hasattr(self, 'empty_history') and not self.match.game.history:
            self.empty_history.lift()

    def _board_draw(self):
        self._board_redraw_job = None
        if not self.play_page or not self.play_page.winfo_exists():
            return
        seen = self.camera_seen
        board = self._obs_to_board(seen) if seen is not None else self.match.game.board
        cv = self.board_canvas
        cw, ch = cv.winfo_width(), cv.winfo_height()
        if cw < 20 or ch < 20:
            cw, ch = 560, 430
        # Fit the entire board + top/bottom orientation labels in the ACTUAL canvas.
        margin = 8
        sq = max(12, min((cw - 2*margin)//N, (ch - 48)//N))
        ox, oy = (cw - N*sq)//2, (ch - N*sq)//2
        self._draw_board(cv, board, sq, ox, oy, show_labels=False)
        cv.create_text(cw//2, max(9, oy-11), text='ด้านบน: AI สีส้ม',
                       font=('Tahoma', 9, 'bold'), fill='#365C83')
        cv.create_text(cw//2, min(ch-9, oy+N*sq+12), text='ด้านล่าง: คน สีดำ',
                       font=('Tahoma', 9, 'bold'), fill='#365C83')

        stages = {
            'INITIAL':'แสดงภาพกระดาน (ยังไม่เริ่มเล่นจริง)',
            'HUMAN':'ถึงตาคน — เดินหมากดำบนกระดานจริง',
            'AI_READY':'AI ADOM กำลังคิดตาเดิน',
            'AI_IN_PROGRESS':'MG400 กำลังเคลื่อนที่ — ห้ามเข้าพื้นที่',
            'VERIFY_AI':'ตรวจสอบผลการเดินของ AI จากกล้อง',
            'GAME_OVER':'เกมจบแล้ว',
            'ERROR':'หยุดเกมเพื่อแก้ไขข้อผิดพลาด',
        }
        stage = self.match.state.stage
        summary = ('ยังไม่มีข้อมูลจากกล้อง' if seen is None else
                   f'ตรวจพบหมากดำ {sum(v=="black" for v in seen.values())} ตัว '
                   f'| หมากส้ม {sum(v=="orange" for v in seen.values())} ตัว | ฮอส {len(self.camera_seen_kings)} ตัว')
        if stage == 'HUMAN' and not self.monitor_only:
            hmoves = self.match.game.human_moves
            if hmoves:
                m_text = ', '.join(move_label(m) for m in hmoves[:8])
                if any(m.is_capture for m in hmoves):
                    summary += f'\n⚠ บังคับกิน! ตาที่ต้องเดิน: {m_text}'
                else:
                    summary += f'\nตาที่คนเดินได้: {m_text}'
        self.turn_label.config(text=summary)
        self.game_state_label.config(text=stages.get(stage,'-'),
                                     fg='#B22D30' if stage=='ERROR' else '#225FAD')
        if stage == 'INITIAL':
            started_text = '● ยังไม่เริ่มเกมจริง — ดูภาพเท่านั้น'
            bg, fg = '#FFF1D7', '#825500'
        elif stage == 'GAME_OVER':
            started_text = '● เกมจบแล้ว'
            bg, fg = '#E5EDFB', '#25588A'
        elif stage == 'ERROR':
            started_text = '● เกมหยุด — พบข้อผิดพลาด'
            bg, fg = '#FBE7E7', '#A6242B'
        else:
            started_text = '● เกมเริ่มแล้ว'
            if self.game_started_at:
                started_text += ' ' + self.game_started_at
            bg, fg = '#DDF4E5', '#18713D'
        self.game_live_label.configure(text=started_text, bg=bg, fg=fg)
        if stage == 'INITIAL':
            self.robot_retry_button.grid_forget()
            self.game_begin_button.grid(row=1, column=0, columnspan=2,
                                        sticky='ew', pady=(3, 0))
        elif stage == 'AI_READY' and self.pending_ai is not None:
            self.game_begin_button.grid_forget()
            self.robot_retry_button.grid(row=1, column=0, columnspan=2,
                                         sticky='ew', pady=(3, 0))
            self.robot_retry_button.configure(
                state='disabled' if self.robot_busy or self.ai_busy else 'normal')
        else:
            self.game_begin_button.grid_forget()
            self.robot_retry_button.grid_forget()
        self.ai_status_view.set('ระดับ: ' + (self.actual_level or '-'))
        self.excel_status_view.set('มีประวัติ' if self.match.game.history else 'ยังไม่มีประวัติ')
        self._refresh_history_table()
        if self.robot_busy:
            self.robot_status_view.set('กำลังเคลื่อนที่')
        elif self._connection_check_running:
            self.robot_status_view.set('กำลังตรวจ TCP...')
        elif self._robot_probe_fresh():
            p = self.robot_probe
            if p.connected:
                self.robot_status_view.set('ติดต่อได้ / ' + str(p.mode))
            elif not p.dashboard_ok:
                self.robot_status_view.set('พอร์ต 29999 ไม่ตอบ')
            else:
                self.robot_status_view.set('พอร์ต 30003 ไม่ตอบ')
        else:
            self.robot_status_view.set('ยังไม่ยืนยันการเชื่อมต่อ')
        if self.camera is not None:
            self.camera_status.set('เปิดกล้องแล้ว')
            self.camera_live_label.configure(text='● เปิดกล้องแล้ว', fg='#23924F')
        else:
            self.camera_status.set('ยังไม่เชื่อมต่อ')
            self.camera_live_label.configure(text='● ยังไม่เชื่อมต่อ', fg='#A16E22')
        if stage=='ERROR':
            self.status_right.configure(text='● พบข้อผิดพลาด', fg='#B52D33')
        elif self.robot_busy:
            self.status_right.configure(text='● MG400 กำลังทำงาน', fg='#B16C11')
        elif self._robot_probe_fresh() and self.robot_probe.connected and not self.robot_probe.available_for_game:
            self.status_right.configure(text='● MG400 มี Alarm/อยู่ในโหมดไม่พร้อม', fg='#B52D33')
        elif self.monitor_only or stage=='INITIAL':
            self.status_right.configure(text='● ยังไม่เริ่มเกม / ไม่ส่งคำสั่ง MG400', fg='#235EA2')
        else:
            self.status_right.configure(text='● เกมเริ่มแล้ว / ' + (
                'ติดต่อ MG400 ได้' if self._robot_probe_fresh() and self.robot_probe.connected
                else 'MG400 ยังไม่ยืนยัน'), fg='#267C49' if self._robot_probe_fresh()
                and self.robot_probe.connected else '#527291')

    def _robot_probe_fresh(self):
        return (self.robot_probe is not None and self.robot_probe.ip == self.robot_ip
                and time.monotonic() - self.robot_probe.checked_at < 25.0)

    def _set_robot_connection_label(self):
        if self.robot_config_error:
            self.robot_connection_text.set('MG400: ตั้งค่า IP ไม่ถูกต้อง — ' + self.robot_config_error)
        elif self._connection_check_running:
            self.robot_connection_text.set(f'MG400 ({self.robot_ip}): กำลังตรวจการเชื่อมต่อ...')
        elif self._robot_probe_fresh():
            self.robot_connection_text.set('MG400: ' + self.robot_probe.display())
        else:
            self.robot_connection_text.set(f'MG400 ({self.robot_ip}): ยังไม่ได้ยืนยันการเชื่อมต่อ')

    def _check_robot_connection(self):
        """No command except RobotMode(). Nonblocking: Tk controls remain responsive."""
        if not self.enable_robot_diagnostics or self._connection_check_running or self.robot_busy:
            return
        if not self.robot_ip:
            self._set_robot_connection_label()
            return
        self._connection_check_running = True
        self._set_robot_connection_label()
        ip = self.robot_ip
        ident = self.generation
        def worker():
            try:
                probe = probe_mg400(ip, timeout=2.0)
                self.queue.put(('probe', ident, probe, None))
            except Exception as exc:
                self.queue.put(('probe', ident, None, str(exc)))
        threading.Thread(target=worker, daemon=True).start()

    def _start_game(self):
        if self.selected_level.get() not in LEVELS:
            return
        self.actual_level = self.selected_level.get()
        self.monitor_only = False
        self.match.reset()
        self.pending_ai = None
        self.auto_robot_started = False
        self.showing_intro = False
        self.winner_shown = False
        self.game_started_at = None
        self.last_seen = None
        self.stable_count = 0
        self.camera_seen = None
        self.camera_seen_kings = frozenset()
        self._clear_illegal()
        self._show_play()
        self.level_label.config(text=f'ADOM: {self.actual_level}   |   MG400: เครื่องจริง')

        try:
            if self.camera is None:
                self.camera = BoardCamera(require_board_file=False)
                self.camera.open()
            self.status.set('เปิดหน้าจอเล่นแล้ว: ด้านซ้ายเป็นภาพกล้อง ด้านขวาเป็นกระดานตามที่กล้องตรวจพบ')
            self._board_draw()
        except Exception as exc:
            self.monitor_only = True
            self.status.set('เปิดหน้าจอเล่นแล้ว แต่ยังเปิดกล้องไม่ได้: ' + str(exc))
            self.turn_label.config(text='ยังเปิดกล้องไม่ได้ — ด้านซ้ายจะแสดงภาพเมื่อกล้องพร้อม ส่วนด้านขวาจะคงกระดานล่าสุดไว้')
            self._board_draw()
            return

        issues = []
        if not self._robot_probe_fresh() or not self.robot_probe.available_for_game:
            msg = (self.robot_probe.display() if self._robot_probe_fresh() else
                   'ยังไม่ได้ตรวจการเชื่อมต่อ MG400')
            issues.append('MG400 ไม่พร้อม: ' + msg)
            self._next_robot_check = 0  # request a new passive check
        try:
            LiveRunner(ROOT / 'live_safety_config.json')
        except Exception as exc:
            issues.append(str(exc))
        if not (ROOT / 'board.npz').is_file():
            issues.append('ยังไม่มี board.npz จาก boardcal')

        try:
            preview = self.camera.snapshot(frames=2)
            self.camera_seen = preview
            self.camera_seen_kings = frozenset(self.camera.last_kings)
        except Exception as exc:
            issues.append('กล้องยังอ่านกระดานเริ่มต้นไม่ได้: ' + str(exc))
        self._board_draw()

        if issues:
            self.monitor_only = True
            self.status.set('เปิดหน้าจอเล่นแล้ว แต่ยังไม่เริ่มเกมจริง: ' + ' | '.join(issues))
            self.turn_label.config(text='โหมดดูภาพจากกล้องเท่านั้น\nด้านซ้ายเป็นภาพกล้อง ด้านขวาเป็นกระดานตามกล้อง\nเมื่อพร้อมค่อยกลับไปเริ่มเกมใหม่อีกครั้ง')
            messagebox.showwarning('ยังเริ่มเกมจริงไม่ได้ — สิ่งที่ต้องแก้',
                                   '\n\n'.join(f'{i}. {t}' for i, t in enumerate(issues, 1)))
            return

        # (ตัดหน้าต่าง 'ยืนยันพื้นที่ปลอดภัยก่อนเริ่ม' ออกตามที่ผู้ใช้สั่ง — ผู้ควบคุมรับผิดชอบความปลอดภัยเอง)
        choice = self._ask_first_side()
        if choice is None:
            self.monitor_only = True
            self.status.set('ยกเลิกการเริ่มเกม — ยังไม่ได้เลือกฝ่ายที่เริ่มก่อน (ดูภาพกล้องได้อย่างเดียว)')
            self._board_draw()
            return
        try:
            initial = self.camera.snapshot(frames=6)
            self.camera_seen = initial
            self.camera_seen_kings = frozenset(self.camera.last_kings)
            was_random = choice == 'random'
            first = secrets.choice((HUMAN, AI)) if was_random else choice
            self.match.start(initial, first_side=first, kings=self.camera.last_kings)
            self.game_started_at = datetime.now().strftime('%H:%M:%S')
            self._board_draw()
            self.showing_intro = True
            who = 'คน (หมากดำ)' if first == HUMAN else 'AI ADOM (หมากส้ม)'
            self.status.set(f'{"สุ่มได้" if was_random else "เลือกให้"} {who} เริ่มก่อน | ความยาก: {self.actual_level}')
            popup, _message = build_dialog(
                self.root, ('human_first' if first == HUMAN else 'ai_first') + ('' if was_random else '_pick'))
            popup.transient(self.root)
            popup.grab_set()
            popup.protocol('WM_DELETE_WINDOW', lambda: None)
            self.root.after(INTRO_POPUP_MS, lambda: self._end_intro(popup, first))
        except Exception as exc:
            self.monitor_only = True
            self.match.reset()
            self.game_started_at = None
            self.status.set('ยังเริ่มเกมจริงไม่ได้ แต่ยังดูภาพกล้องและกระดานด้านขวาได้: ' + str(exc))
            self._board_draw()
            messagebox.showwarning('ยังเริ่มเกมจริงไม่ได้',
                                   'ตอนนี้ระบบเปิดเป็นโหมดดูภาพกล้อง/กระดานจากกล้องก่อน\n' + str(exc))

    def _ask_first_side(self):
        """Styled modal for choosing who starts: random, human, or AI."""
        result = {'value': None}
        top = tk.Toplevel(self.root)
        top.title('เลือกฝ่ายที่เริ่มก่อน')
        top.configure(bg='#DCEAF7')
        top.transient(self.root)
        top.resizable(False, False)

        def pick(value):
            result['value'] = value
            try:
                top.grab_release()
            except tk.TclError:
                pass
            top.destroy()

        # A restrained, card-based dialog that matches the dashboard palette.
        shell = tk.Frame(top, bg='#FFFFFF', highlightbackground='#B8CBDE',
                         highlightthickness=1, bd=0)
        shell.pack(fill='both', expand=True, padx=1, pady=1)

        header = tk.Frame(shell, bg='#EAF4FF')
        header.pack(fill='x')
        logo = tk.Canvas(header, width=64, height=52, bg='#EAF4FF',
                         highlightthickness=0, bd=0)
        logo.pack(pady=(12, 0))
        # Three-piece mark echoes the orange/black game pieces in the main UI.
        logo.create_oval(10, 18, 31, 39, fill='#F18C30', outline='#FFFFFF', width=2)
        logo.create_oval(27, 7, 48, 28, fill='#24262E', outline='#FFFFFF', width=2)
        logo.create_oval(30, 27, 51, 48, fill='#24262E', outline='#FFFFFF', width=2)

        tk.Label(header, text='ใครจะเริ่มก่อน?', font=(FONT, 20, 'bold'),
                 bg='#EAF4FF', fg=NAVY).pack(pady=(2, 4))
        tk.Label(header, text='เลือกฝ่ายที่เริ่มก่อน หรือให้ระบบสุ่ม',
                 font=(FONT, 10), bg='#EAF4FF', fg='#557190').pack(pady=(0, 16))

        body = tk.Frame(shell, bg='#FFFFFF')
        body.pack(fill='both', expand=True, padx=28, pady=(18, 8))

        def draw_card_icon(canvas, kind, bg):
            canvas.delete('all')
            canvas.configure(bg=bg)
            if kind == 'random':
                # White die with three blue pips.
                canvas.create_rectangle(9, 8, 39, 38, fill='#FFFFFF', outline='#FFFFFF', width=1)
                for px, py in ((16, 15), (31, 23), (16, 31)):
                    canvas.create_oval(px - 2.4, py - 2.4, px + 2.4, py + 2.4,
                                       fill=bg, outline=bg)
            elif kind == 'human':
                canvas.create_oval(7, 6, 41, 40, fill='#111827', outline='#FFFFFF', width=2)
                canvas.create_oval(12, 11, 36, 35, fill='#20232C', outline='#3F4652', width=1)
            else:
                canvas.create_oval(7, 6, 41, 40, fill='#E8790E', outline='#FFFFFF', width=2)
                canvas.create_oval(12, 11, 36, 35, fill='#F18C30', outline='#FFD7AE', width=1)

        def bind_click_tree(widget, value):
            widget.bind('<Button-1>', lambda _event, v=value: pick(v))
            widget.configure(cursor='hand2')
            for child in widget.winfo_children():
                bind_click_tree(child, value)

        def add_choice(kind, title, subtitle, value, bg, hover_bg):
            card = tk.Frame(body, bg=bg, height=68, highlightthickness=0, bd=0,
                            cursor='hand2')
            card.pack(fill='x', pady=5)
            card.pack_propagate(False)
            icon = tk.Canvas(card, width=48, height=48, bg=bg, highlightthickness=0, bd=0)
            icon.pack(side='left', padx=(16, 10), pady=10)
            draw_card_icon(icon, kind, bg)

            copy = tk.Frame(card, bg=bg)
            copy.pack(side='left', fill='both', expand=True, pady=9)
            title_label = tk.Label(copy, text=title, font=(FONT, 12, 'bold'),
                                   bg=bg, fg='#FFFFFF', anchor='w')
            title_label.pack(anchor='w')
            subtitle_label = tk.Label(copy, text=subtitle, font=(FONT, 9),
                                      bg=bg, fg='#F0F6FF', anchor='w')
            subtitle_label.pack(anchor='w', pady=(2, 0))

            # Subtle hover feedback, while keeping the three choices visually distinct.
            all_widgets = [card, icon, copy, title_label, subtitle_label]
            def paint_hover(_event=None, hovered=False):
                shade = hover_bg if hovered else bg
                for widget in all_widgets:
                    try:
                        widget.configure(bg=shade)
                    except tk.TclError:
                        pass
                draw_card_icon(icon, kind, shade)
            for widget in all_widgets:
                widget.bind('<Enter>', lambda event, fn=paint_hover: fn(event, True))
                widget.bind('<Leave>', lambda event, fn=paint_hover: fn(event, False))
            bind_click_tree(card, value)

        add_choice('random', 'สุ่มผู้เริ่มเล่น', 'ให้ระบบเลือกฝ่ายเริ่มเกม',
                   'random', '#0878E8', '#086ACB')
        add_choice('human', 'คนเริ่มก่อน', 'หมากสีดำ  ·  YOU',
                   HUMAN, '#202735', '#303B4D')
        add_choice('ai', 'AI ADOM เริ่มก่อน', 'หมากสีส้ม  ·  MG400',
                   AI, '#E8790E', '#D96B06')

        cancel = tk.Button(body, text='ยกเลิก', command=lambda: pick(None),
                           font=(FONT, 10), bg='#FFFFFF', fg='#557190',
                           activebackground='#F1F7FD', activeforeground=NAVY,
                           relief='flat', bd=0, cursor='hand2', padx=18, pady=7)
        cancel.pack(pady=(8, 4))
        top.protocol('WM_DELETE_WINDOW', lambda: pick(None))

        top.update_idletasks()
        try:
            width = max(420, top.winfo_reqwidth())
            height = top.winfo_reqheight()
            x = self.root.winfo_rootx() + (self.root.winfo_width() - width) // 2
            y = self.root.winfo_rooty() + (self.root.winfo_height() - height) // 2
            top.geometry(f'{width}x{height}+{max(0, x)}+{max(0, y)}')
        except tk.TclError:
            pass
        top.grab_set()
        self.root.wait_window(top)
        return result['value']

    def _end_intro(self,popup,first):
        try:
            if popup.winfo_exists():
                popup.grab_release()
                popup.destroy()
        except tk.TclError:
            pass
        self.showing_intro=False
        if self.match.state.stage=='AI_READY' and first==AI:
            self._compute_ai()
        elif self.match.state.stage=='HUMAN':
            self.status.set('เริ่มตาคน: ใช้มือเดินหมากดำบนกระดานจริง ยกมือออกเมื่อเดินเสร็จ')

    def _compute_ai(self):
        if self.ai_busy or self.robot_busy or self.showing_intro:
            return
        if self.match.state.stage!='AI_READY' or self.match.state.pending is not None:
            return
        self.ai_busy=True
        self.status.set('ADOM กำลังคำนวณตาเดิน...')
        self._sync_ai_popup()
        ident=self.generation
        board=self.match.game.board
        level=self.actual_level
        def worker():
            try:
                result=build_ai(level).choose(board,AI)
                self.queue.put(('ai',ident,result,None))
            except Exception as e:
                self.queue.put(('ai',ident,None,str(e)))
        threading.Thread(target=worker,daemon=True).start()

    def _run_robot(self):
        if self.showing_intro or self.robot_busy or self.ai_busy:
            return
        if self.match.state.stage!='AI_READY' or self.pending_ai is None or self.auto_robot_started:
            return
        self.auto_robot_started=True
        try:
            if not self._robot_probe_fresh() or not self.robot_probe.available_for_game:
                raise RobotFailure('ยังไม่ได้ยืนยันการเชื่อมต่อ MG400 ล่าสุด — กด ตรวจ MG400 และดูสถานะ')
            pre=self.camera.snapshot(5)
            self.match.require_board_ready(pre, kings=self.camera.last_kings)
            planner=RobotPlanner(self.camera)
            steps=planner.plan(self.pending_ai,self.match.game.board)
            runner=LiveRunner(ROOT / 'live_safety_config.json')
            # Human protection must be independently enforced by hardware.
            if runner.per_move_confirmation:
                if not messagebox.askyesno('ยืนยันตา AI ก่อนเคลื่อนจริง',
                    f'ให้ MG400 ทำตา {move_label(self.pending_ai)}?\n'
                    'ตรวจคนออกนอกพื้นที่และชุดป้องกันทำงานแล้ว'):
                    self.auto_robot_started=False
                    self.status.set('รอผู้ควบคุมอนุญาต MG400 ก่อนทำตา AI')
                    return
                phrase=simpledialog.askstring('MG400', 'พิมพ์ MOVE MG400 เพื่ออนุมัติให้เคลื่อนที่จริง')
                if phrase!='MOVE MG400':
                    self.auto_robot_started=False
                    self.status.set('ยังไม่อนุมัติการเคลื่อนที่จริง')
                    return
            # Refresh/verify again AFTER operator confirmation to prevent stale
            # king-reserve or board positions being used while dialogue was open.
            pre=self.camera.snapshot(5)
            self.match.require_board_ready(pre, kings=self.camera.last_kings)
            steps=planner.plan(self.pending_ai,self.match.game.board)
            self.match.before_robot(pre, kings=self.camera.last_kings)
            self.robot_busy=True
            self.stop_event.clear()
            self._board_draw()
            ident=self.generation
            self.status.set('MG400 เคลื่อนที่จริง — ห้ามเข้าพื้นที่')
            self._sync_ai_popup()
            def worker():
                try:
                    msg=runner.run(steps,planner,report=lambda s:self.queue.put(('note',ident,s,None)),
                                   stop=self.stop_event.is_set, robot_ip=self.robot_ip)
                    self.queue.put(('robot',ident,msg,None))
                except Exception as e:
                    self.queue.put(('robot',ident,None,str(e)))
            threading.Thread(target=worker,daemon=True).start()
        except Exception as exc:
            self.auto_robot_started=False
            self.status.set('ยังไม่อนุญาต MG400: '+str(exc))
            messagebox.showwarning('MG400 ไม่พร้อม',str(exc))

    def _retry_robot(self):
        self.auto_robot_started=False
        self._run_robot()

    def _on_human(self,seen):
        try:
            entry=self.match.accept_human(seen, kings=self.camera.last_kings)
            self._hide_checking()
            self._clear_illegal()
            self.last_seen=None
            self.stable_count=0
            self.status.set('กล้องยืนยันคนเดิน: '+move_label(entry.move))
            self._board_draw()
            self._autosave()
            if self.match.state.stage=='GAME_OVER':
                self._show_winner()
            else:
                self._compute_ai()
        except MatchError as e:
            self._show_illegal(str(e))

    def _on_verified(self,seen):
        try:
            entry=self.match.verify_ai(seen, kings=self.camera.last_kings)
            self._hide_ai_moving()
            self.pending_ai=None
            self.auto_robot_started=False
            self._clear_illegal()
            self.last_seen=None
            self.stable_count=0
            self.status.set('กล้องยืนยัน AI เดิน: '+move_label(entry.move)+' → ตาคน')
            self._board_draw()
            self._autosave()
            if self.match.state.stage=='GAME_OVER':
                self._show_winner()
        except MatchError as e:
            self._show_illegal('หมากหลัง MG400 เคลื่อนที่ไม่ตรงกับตา AI: '+str(e))

    def _show_illegal(self,reason):
        # The camera still controls when this persistent warning is dismissed.
        if self.invalid_popup is None or not self.invalid_popup.winfo_exists():
            top, self.invalid_label = build_dialog(self.root, 'illegal', text=reason)
            top.transient(self.root)
            top.protocol('WM_DELETE_WINDOW',lambda:None)
            top.grab_set()
            self.invalid_popup=top
        if self.invalid_label is not None:
            self.invalid_label.config(text=reason)
            fit_dialog(self.invalid_popup)

    def _show_checking(self,text):
        """Non-modal human-turn status in the bottom-right corner."""
        self._hide_ai_moving()
        try:
            popup=getattr(self,'check_popup',None)
            if popup is None or not popup.winfo_exists():
                top, self.check_label = build_dialog(self.root, 'human_turn', text=text)
                top.transient(self.root)
                top.protocol('WM_DELETE_WINDOW',lambda:None)
                self.check_popup=top
                self._check_text=None
            if text!=getattr(self,'_check_text',None):
                self._check_text=text
                self.check_label.config(text=text)
                fit_dialog(self.check_popup)
        except tk.TclError:
            self.check_popup=None

    def _hide_checking(self):
        popup=getattr(self,'check_popup',None)
        if popup is not None:
            try:
                popup.destroy()
            except tk.TclError:
                pass
        self.check_popup=None
        self._check_text=None

    def _show_ai_moving(self, text):
        """Render AI turn progress without changing game or robot state."""
        self._hide_checking()
        try:
            popup = getattr(self, 'ai_move_popup', None)
            if popup is None or not popup.winfo_exists():
                top, self.ai_move_label = build_dialog(self.root, 'ai_turn', text=text)
                top.transient(self.root)
                top.protocol('WM_DELETE_WINDOW', lambda: None)
                self.ai_move_popup = top
                self._ai_move_text = None
            if text != getattr(self, '_ai_move_text', None):
                self._ai_move_text = text
                self.ai_move_label.config(text=text)
                fit_dialog(self.ai_move_popup)
        except tk.TclError:
            self.ai_move_popup = None

    def _hide_ai_moving(self):
        popup = getattr(self, 'ai_move_popup', None)
        if popup is not None:
            try:
                popup.destroy()
            except tk.TclError:
                pass
        self.ai_move_popup = None
        self._ai_move_text = None

    def _sync_ai_popup(self):
        """Follow the existing turn phases; never initiate or retry movement."""
        stage = self.match.state.stage
        if getattr(self, 'showing_intro', False) or stage not in ('AI_READY', 'AI_IN_PROGRESS', 'VERIFY_AI'):
            self._hide_ai_moving()
            return
        if getattr(self, 'robot_busy', False) or stage == 'AI_IN_PROGRESS':
            status = getattr(self, 'status', None)
            text = status.get() if status is not None else 'MG400 กำลังเดินหมาก — ห้ามเข้าพื้นที่'
        elif stage == 'VERIFY_AI':
            text = 'AI เดินเสร็จแล้ว\nรอกล้องยืนยันตำแหน่งหมาก'
        elif getattr(self, 'ai_busy', False):
            text = 'AI กำลังคำนวณตาเดิน...'
        else:
            status = getattr(self, 'status', None)
            text = status.get() if status is not None else 'เตรียมการเดินหมากของ AI'
        self._show_ai_moving(text)

    def _clear_illegal(self):
        if self.invalid_popup is not None:
            try:
                self.invalid_popup.grab_release()
                self.invalid_popup.destroy()
            except tk.TclError:
                pass
        self.invalid_popup=None
        self.invalid_label=None

    def _show_winner(self):
        if self.winner_shown or self.match.state.stage!='GAME_OVER':
            return
        self.winner_shown=True
        self._hide_ai_moving()
        self._hide_checking()
        game=self.match.game
        state = ('human_win' if game.winner == HUMAN else 'ai_win') if game.status == 'win' else 'draw'
        top, _message = build_dialog(self.root, state, text=game.draw_reason)
        self._autosave()

    def _rules(self):
        RulesWindow(self.root)

    def _autosave(self):
        try:
            export_history_xlsx(self.saving_path,self.match.game.history,
                                difficulty=self.actual_level or self.selected_level.get(),
                                game_status=self.match.game.status,
                                robot_status=self.match.robot_status)
        except Exception as e:
            self.status.set('Excel บันทึกอัตโนมัติไม่ได้: '+str(e))

    def _export(self):
        name=filedialog.asksaveasfilename(defaultextension='.xlsx',
                    filetypes=[('Excel workbook','*.xlsx')],
                    initialfile='thai_checkers_8x8_history.xlsx')
        if name:
            try:
                export_history_xlsx(name,self.match.game.history,
                                    difficulty=self.actual_level or self.selected_level.get(),
                                    game_status=self.match.game.status,
                                    robot_status=self.match.robot_status)
                messagebox.showinfo('Excel','บันทึกสำเร็จ:\n'+name)
            except Exception as e:
                messagebox.showerror('Excel',str(e))

    def _request_stop(self):
        self.stop_event.set()
        self.status.set('ส่งคำขอหยุดแล้ว — หากเป็นเหตุฉุกเฉินต้องใช้ E-stop ฮาร์ดแวร์จริง')

    def _restart(self):
        if self.ai_busy or self.robot_busy:
            messagebox.showwarning('รอ','ห้ามเริ่มเกมใหม่ระหว่าง MG400 เคลื่อนที่หรือ AI คำนวณ')
            return
        if not messagebox.askyesno('เริ่มใหม่','กรุณาจัดหมากบนกระดานจริงกลับ 12 ตัว แล้วเริ่มเกมใหม่ ยืนยันหรือไม่?'):
            return
        self._hide_ai_moving()
        self._hide_checking()
        self.generation+=1
        self.match.reset()
        self.history_times.clear()
        self.pending_ai=None
        self.auto_robot_started=False
        self.actual_level=None
        self.showing_intro=False
        self.winner_shown=False
        self.last_seen=None
        self.stable_count=0
        self.camera_seen=None
        self.monitor_only=False
        self.game_started_at = None
        self.selected_level.set('')
        self._clear_illegal()
        self.status.set('เลือกความยากแล้วกดเริ่มเกมใหม่')
        self._show_start()

    def _render_frame(self,frame,seen):
        if self.play_page is None:
            return
        display=self.camera.annotate(frame,seen)
        h,w=display.shape[:2]
        vw,vh=self.video.winfo_width(),self.video.winfo_height()
        if vw < 20 or vh < 20:
            return
        scale=min(vw/w,vh/h)
        rgb=cv2.cvtColor(cv2.resize(display,(max(1,int(w*scale)),max(1,int(h*scale)))),cv2.COLOR_BGR2RGB)
        self.photo=ImageTk.PhotoImage(Image.fromarray(rgb))
        self.video.configure(image=self.photo,text='')

    def _tick(self):
        try:
            self._sync_ai_popup()
            if self.enable_robot_diagnostics and not self.robot_busy and not self._connection_check_running and self.robot_ip \
                    and time.monotonic() >= self._next_robot_check:
                self._next_robot_check = time.monotonic() + 12.0
                self._check_robot_connection()
            if self.camera is not None and self.play_page is not None:
                try:
                    frame=self.camera.read_frame()
                    # Continue displaying the webcam even during MG400 motion,
                    # but never interpret occlusion by the arm as a human move.
                    seen=None
                    if not self.robot_busy:
                        try:
                            seen=self.camera.observe(frame)
                            self._vision_msg=''
                        except (VisionFailure,ValueError,np.linalg.LinAlgError) as vexc:
                            self.last_seen=None
                            self.stable_count=0
                            self._vision_msg=str(vexc)
                    self._render_frame(frame,seen)
                    if seen is not None and (seen != self.camera_seen or frozenset(self.camera.last_kings) != self.camera_seen_kings):
                        self.camera_seen = seen
                        self.camera_seen_kings = frozenset(self.camera.last_kings)
                        self._board_draw()
                    stage=self.match.state.stage
                    if stage!='HUMAN' or self.robot_busy or self.ai_busy or self.showing_intro:
                        self._hide_checking()
                    elif seen is None:
                        msg=getattr(self,'_vision_msg','') or ''
                        self._show_checking('ถึงตาคน (หมากสีดำ)\n'+('⚠ กล้องอ่านกระดานไม่ได้: '+msg if msg else 'รอกล้องอ่านกระดาน ...'))
                    elif (seen==signature(self.match.game.board) and self.camera.last_kings==king_signature(self.match.game.board)):
                        # ถึงตาคนแล้ว (กระดานยังตรงสถานะเกม): แสดงหน้าต่างตาคนเล่นทันที ไม่ต้องรอมือเข้าไป
                        self._show_checking('ถึงตาคน (หมากสีดำ) — เดินหมากได้เลย\nกล้องจะตรวจให้อัตโนมัติหลังวางหมากนิ่ง')
                    else:
                        self._show_checking(self.match.progress_human(seen,kings=self.camera.last_kings))
                    if self.monitor_only and stage == 'INITIAL':
                        self._clear_illegal()
                    elif not self.robot_busy and not self.ai_busy and not self.showing_intro and seen is not None and stage in ('HUMAN','VERIFY_AI','AI_READY'):
                        if (seen, frozenset(self.camera.last_kings))==self.last_seen:
                            self.stable_count+=1
                        else:
                            self.last_seen=(seen, frozenset(self.camera.last_kings))
                            self.stable_count=1
                            self.last_change_time=time.monotonic()
                        # ตาคน: รอให้นิ่งนานหน่อย (กันมือบัง) / หลังหุ่นเสร็จ (VERIFY_AI) ตรวจเร็ว เพื่อให้คนเล่นต่อได้ทันที
                        need_frames,need_settle=(12,CHECK_SETTLE_SECONDS) if stage=='HUMAN' else (VERIFY_STABLE_FRAMES,VERIFY_SETTLE_SECONDS)
                        if self.stable_count>=need_frames and time.monotonic()-getattr(self,'last_change_time',0.0)>=need_settle:
                            self.stable_count=0
                            if stage=='HUMAN':
                                classification=self.match.classify_human(seen,kings=self.camera.last_kings)
                                if classification=='unchanged':
                                    self._pending_action_strikes = 0
                                    self._clear_illegal()
                                elif classification=='legal':
                                    self._pending_action_strikes = 0
                                    self._on_human(seen)
                                else:
                                    diag_kind, diag_msg = self.match.diagnose_human(seen, kings=self.camera.last_kings)
                                    if diag_kind == 'incomplete_scan':
                                        # กล้องยังเห็นหมากไม่ครบ (เช่น มือกำลังถือหมาก/บังหมาก หรือแสงสะท้อน) -> ไม่เด้งผิดกติกา
                                        self._pending_action_strikes = 0
                                        self._clear_illegal()
                                        self.status.set(diag_msg)
                                        self.turn_label.config(text=diag_msg)
                                    elif diag_kind in ('capture_removal_pending', 'king_mismatch'):
                                        # เดินถูกช่องแล้ว แต่รอหยิบหมากส้มที่ถูกกินออก หรือรอเปลี่ยนตัวฮอสจุดเขียว
                                        self._pending_action_strikes = getattr(self, '_pending_action_strikes', 0) + 1
                                        self.status.set(diag_msg)
                                        self.turn_label.config(text=diag_msg)
                                        if self._pending_action_strikes >= PENDING_STRIKES_LIMIT:
                                            self._show_illegal(diag_msg)
                                    else:
                                        self._pending_action_strikes = 0
                                        self.status.set(diag_msg.replace('\n', ' | '))
                                        self._show_illegal(diag_msg)
                            elif stage=='AI_READY':
                                if seen!=signature(self.match.game.board) or self.camera.last_kings!=king_signature(self.match.game.board):
                                    self._show_illegal('มีการขยับหมากระหว่างรอ AI / MG400\nโปรดคืนหมากให้ตรงตำแหน่งล่าสุด')
                                else:
                                    self._clear_illegal()
                            else:
                                expected=signature(apply_move(self.match.game.board,self.match.state.pending))
                                if seen==expected and self.camera.last_kings==expected_kings_after(self.match.game.board,self.match.state.pending):
                                    self._on_verified(seen)
                                elif seen==expected:
                                    want=expected_kings_after(self.match.game.board,self.match.state.pending)
                                    self._show_illegal('หมากตรงตามตา AI แล้ว แต่กล้องเห็นตำแหน่งฮอส '
                                                       f'{sorted(self.camera.last_kings)} ไม่ตรงที่ควรเป็น {sorted(want)}\n'
                                                       'ตรวจว่าตัวฮอสมีจุดเขียวตรงกลางและไม่ถูกบัง / ปรับ GREEN_LOW-GREEN_HIGH ใน mg400_visionV4.py')
                                elif seen!=signature(self.match.game.board) or self.camera.last_kings!=king_signature(self.match.game.board):
                                    self._show_illegal('ตำแหน่งหมาก AI หลังการย้ายยังไม่ถูกต้อง\nกรุณาตรวจหมากส้มและหมากที่ถูกกิน')
                                else:
                                    self._clear_illegal()
                    else:
                        if self.robot_busy or self.ai_busy or self.showing_intro:
                            self.last_seen=None
                            self.stable_count=0
                except Exception as exc:
                    self.status.set('ปัญหาการอ่านกล้อง: '+str(exc))
            for _ in range(10):
                try:
                    kind,ident,result,error=self.queue.get_nowait()
                except Empty:
                    break
                if ident != self.generation:
                    if kind == 'probe':
                        self._connection_check_running = False
                    continue
                if kind == 'probe':
                    self._connection_check_running = False
                    if result is not None and result.ip == self.robot_ip:
                        self.robot_probe = result
                    elif error:
                        self.robot_probe = None
                        self.robot_connection_text.set('MG400: ตรวจสอบไม่ได้ — ' + error)
                    self._set_robot_connection_label()
                    if self.play_page is not None:
                        self._board_draw()
                elif kind=='note':
                    self.status.set(result)
                elif kind=='ai':
                    self.ai_busy=False
                    if error:
                        self.match.robot_failed('AI: '+error)
                        self.status.set('AI ผิดพลาด: '+error)
                        messagebox.showerror('AI',error)
                        continue
                    if result.move is None:
                        self.match.robot_failed('AI ไม่พบตาเดิน')
                        messagebox.showwarning('AI','ไม่พบตาเดิน โปรดตรวจตรรกะเกม')
                        continue
                    try:
                        self.match.stage_ai_move(result.move)
                        self.pending_ai=result.move
                        self.auto_robot_started=False
                        self.status.set('ADOM เลือก '+move_label(result.move)+' → เตรียม MG400 อัตโนมัติ')
                        self._board_draw()
                        self.root.after(70,self._run_robot)
                    except Exception as exc:
                        self.match.robot_failed(str(exc))
                        messagebox.showerror('การเดิน AI',str(exc))
                elif kind=='robot':
                    self.robot_busy=False
                    if error:
                        self.match.robot_failed(error)
                        self.status.set('MG400 ผิดพลาด: '+error+' — ต้องตรวจสอบก่อนเล่นต่อ')
                        messagebox.showerror('MG400 ทำงานไม่สำเร็จ',error)
                    else:
                        self.match.after_robot(result)
                        self.status.set('MG400 รายงานทำงานเสร็จ → กล้องตรวจตำแหน่งบนกระดานจริง')
                    self._board_draw()
        finally:
            try:
                self._sync_ai_popup()
                self.root.after(110,self._tick)
            except tk.TclError:
                pass

    def _close(self):
        if self.robot_busy:
            messagebox.showwarning('MG400 กำลังทำงาน',
                'ห้ามปิดโปรแกรมขณะที่หุ่นยนต์กำลังทำงาน ต้องใช้ E-stop ฮาร์ดแวร์เมื่อฉุกเฉิน')
            return
        self.stop_event.set()
        if self.camera is not None:
            try:
                self.camera.close()
            except Exception:
                pass
        self.root.destroy()


def main():
    root=tk.Tk()
    App(root)
    root.mainloop()


if __name__=='__main__':
    main()
