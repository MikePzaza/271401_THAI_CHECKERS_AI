"""UI-only preview; never opens camera or connects to MG400."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import tkinter as tk
from PIL import ImageGrab
from main import App

out = Path(sys.argv[1]) if len(sys.argv)>1 else Path(__file__).resolve().parents[1] / 'DASHBOARD_V6_ACTUAL_PREVIEW.png'
size = sys.argv[2] if len(sys.argv)>2 else '1536x960'
w, h = [int(v) for v in size.split('x')]
root = tk.Tk()
app = App(root, enable_robot_diagnostics=False)
root.geometry(f'{w}x{h}+0+0')
app.actual_level = 'สุ่ม'
app.status.set('ตัวอย่างหน้าต่าง Desktop — ยังไม่เปิดกล้องหรือเชื่อมต่อ MG400')
app.monitor_only = True
app._show_play()
root.update()
# Allow debounce of responsive canvas and geometry to settle.
import time
time.sleep(.20)
root.update()
print('ROOT',root.winfo_width(),root.winfo_height())
print('CANVAS',app.board_canvas.winfo_width(),app.board_canvas.winfo_height())
print('BUTTONS',[(b.cget('text'),b.winfo_width(),b.winfo_height(),b.winfo_rooty()) for b in app.action_buttons])
print('HISTORY',app.history_table.winfo_width(),app.history_table.winfo_height())
print('RIGHT PANEL',app.game_state_label.winfo_width())
# capture actual GUI, not AI-generated mock-up
ImageGrab.grab(bbox=(0,0,w,h),xdisplay=root.winfo_screen()).save(out)
print('PREVIEW',out)
root.destroy()
