"""Real Tkinter smoke test: start button must open dashboard without devices.
No robot is ever instantiated, and no camera is accessed.
"""
import sys
from pathlib import Path
from unittest.mock import patch
import tkinter as tk
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from main import App, VisionFailure

root = tk.Tk()
app = App(root)
app.selected_level.set('ง่าย')
with patch('main.BoardCamera', side_effect=VisionFailure('ทดสอบ: ไม่มีอุปกรณ์')):
    with patch('main.LiveRunner', side_effect=AssertionError('must never connect robot')):
        app._start_game()
root.update()
assert app.play_page and app.play_page.winfo_exists(), 'Dashboard not opened'
assert app.monitor_only, 'Should be a passive dashboard without camera'
assert len(app.action_buttons) == 4
print('PASS: Dashboard opens without camera, and robot is not contacted')
root.destroy()
