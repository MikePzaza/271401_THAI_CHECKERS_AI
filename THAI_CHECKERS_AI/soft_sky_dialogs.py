"""Soft Sky popup visuals; application retains game flow and hardware control."""
import tkinter as tk

SKY = '#E2F2FF'
WHITE = '#FFFFFF'
INK = '#193D61'
BLUE = '#126BCB'
MUTED = '#64809B'
RED = '#BF3445'
FONT = 'Tahoma'

STATES = {
    'human_first': ('คนเริ่มก่อน', 'ผลการสุ่มฝ่ายเริ่มเกม', 'ฝ่ายเริ่มก่อน',
                    'คน (หมากดำ)', 'black', 'คุณได้เริ่มก่อน'),
    'ai_first': ('AI เริ่มก่อน', 'ผลการสุ่มฝ่ายเริ่มเกม', 'ฝ่ายเริ่มก่อน',
                 'AI ADOM (หมากส้ม)', 'orange', 'AI ได้เริ่มก่อน'),
    'human_first_pick': ('คนเริ่มก่อน', 'ฝ่ายเริ่มเกม', 'ฝ่ายเริ่มก่อน',
                         'คน (หมากสีดำ)', 'black', 'คุณเลือกเริ่มก่อน'),
    'ai_first_pick': ('AI เริ่มก่อน', 'ฝ่ายเริ่มเกม', 'ฝ่ายเริ่มก่อน',
                      'AI ADOM (หมากสีส้ม)', 'orange', 'คุณเลือกให้ AI เริ่มก่อน'),
    'human_win': ('คนชนะ', 'เกมสิ้นสุด', 'จบเกม',
                  'คุณชนะ', 'black', ''),
    'ai_win': ('AI ชนะ', 'เกมสิ้นสุด', 'จบเกม',
               'AI ADOM ชนะ!', 'orange', ''),
    'illegal': ('ผิดกติกา', 'เดินผิดกติกา', 'เดินผิดกติกา',
                '', None, ''),
    'human_turn': ('คนกำลังเดิน', 'ตรวจตาคนเล่น', 'ตาคนเล่น',
                   '', 'black', ''),
    'ai_turn': ('AI กำลังเดิน', 'AI กำลังเดิน', 'AI กำลังเดิน',
                '', 'orange', ''),
    'draw': ('เสมอ', 'เกมสิ้นสุด', 'จบเกม', 'เสมอ!', None, ''),
}


def rounded(canvas, x1, y1, x2, y2, radius, **kwargs):
    return canvas.create_polygon(
        x1+radius, y1, x2-radius, y1, x2, y1, x2, y1+radius,
        x2, y2-radius, x2, y2, x2-radius, y2, x1+radius, y2,
        x1, y2, x1, y2-radius, x1, y1+radius, x1, y1,
        smooth=True, splinesteps=24, **kwargs)


def token(canvas, cx, cy, radius, color):
    orange = color == 'orange'
    halo = '#FFF0DB' if orange else '#DCEFFF'
    rim = '#FFD18D' if orange else '#77828C'
    outer = '#F59B2B' if orange else '#292F36'
    inner = '#FFAC3A' if orange else '#414A54'
    canvas.create_oval(cx-radius-9, cy-radius-9, cx+radius+9,
                       cy+radius+9, fill=halo, outline='')
    canvas.create_oval(cx-radius, cy-radius, cx+radius, cy+radius,
                       fill=outer, outline='')
    canvas.create_oval(cx-radius+6, cy-radius+6, cx+radius-6,
                       cy+radius-6, fill=inner, outline=rim, width=2)


def label(parent, text, size, color=INK, bold=False, **kwargs):
    return tk.Label(parent, text=text, bg=WHITE, fg=color,
                    font=(FONT, size, 'bold' if bold else 'normal'), **kwargs)


def make_panel(parent, state, close=None, text=None):
    """Reusable visual panel. Returns panel plus the dynamic message label."""
    _name, _title, heading, result, color, subtitle = STATES[state]
    is_turn = state in ('human_turn', 'ai_turn')
    is_error = state == 'illegal'
    bg = '#FFF0F2' if is_error else SKY
    shell = tk.Canvas(parent, bg=bg, highlightthickness=0, borderwidth=0)
    card = tk.Frame(shell, bg=WHITE)
    card.place(x=30, y=25, relwidth=1, width=-60,
               relheight=1, height=-50)

    def paint(event):
        shell.delete('card')
        rounded(shell, 14, 14, event.width-14, event.height-14,
                24, fill=WHITE, outline='', tags='card')
        shell.tag_lower('card')
        wrap = max(200, event.width-90)
        for child in card.winfo_children():
            if isinstance(child, tk.Label):
                child.configure(wraplength=wrap)

    shell.bind('<Configure>', paint)
    dynamic = None
    if is_turn:
        head = tk.Frame(card, bg=WHITE)
        head.pack(fill='x', pady=(4, 8))
        emblem = tk.Canvas(head, width=66, height=66, bg=WHITE,
                           highlightthickness=0)
        emblem.pack(side='left', padx=(0, 14))
        token(emblem, 33, 33, 23, color)
        title = tk.Frame(head, bg=WHITE)
        title.pack(side='left')
        label(title, heading, 20, BLUE, True).pack(anchor='w')
        label(title, 'หมากสีส้ม' if state == 'ai_turn' else 'หมากสีดำ', 11, MUTED).pack(anchor='w')
        default = 'AI กำลังเตรียมเดินหมาก' if state == 'ai_turn' else 'เดินหมากได้เลย แล้วนำมือออกจากภาพกล้อง'
        dynamic = label(card, text or default,
                        12, wraplength=430, justify='left', anchor='w')
        dynamic.pack(fill='x', pady=(2, 8))
        footer = 'กรุณารอ และอย่าขยับหมากระหว่างตา AI' if state == 'ai_turn' else 'กล้องจะตรวจให้อัตโนมัติเมื่อวางหมากนิ่ง'
        label(card, footer, 10,
              MUTED, wraplength=430, justify='left', anchor='w').pack(fill='x')
    elif is_error:
        emblem = tk.Canvas(card, width=76, height=76, bg=WHITE,
                           highlightthickness=0)
        emblem.pack(pady=(3, 4))
        emblem.create_oval(4, 4, 72, 72, fill='#FFE5E9', outline='')
        emblem.create_text(38, 37, text='!', fill=RED,
                            font=(FONT, 32, 'bold'))
        label(card, heading, 23, RED, True).pack(pady=(0, 8))
        dynamic = label(card, text or 'ต้องกินหมากก่อน\nกรุณาคืนหมากหรือเดินตามตาที่ถูกกติกา',
                        12, wraplength=450, justify='center')
        dynamic.pack(fill='x', pady=(0, 10))
        label(card, 'หน้าต่างนี้จะหายเมื่อกล้องเห็นกระดาน\nกลับมาถูกต้องเท่านั้น',
              10, MUTED, justify='center').pack(side='bottom', pady=(4, 9))
    else:
        label(card, heading, 16, INK, True).pack(pady=(2, 4))
        emblem = tk.Canvas(card, width=130, height=102, bg=WHITE,
                           highlightthickness=0)
        emblem.pack(pady=(0, 3))
        if color:
            token(emblem, 65, 51, 34, color)
        else:
            token(emblem, 43, 51, 23, 'black')
            token(emblem, 90, 51, 23, 'orange')
        label(card, result, 24 if state.startswith('ai_first') else 26,
              BLUE, True, wraplength=460, justify='center').pack(pady=(0, 5))
        if subtitle:
            label(card, subtitle, 12, INK, True).pack(pady=(0, 4))
        if '_first' in state:
            label(card, 'หน้าต่างนี้จะปิดอัตโนมัติใน 3 วินาที',
                  10, MUTED).pack(side='bottom', pady=(6, 12))
        else:
            if state == 'draw':
                label(card, text or 'เหลือฮอสฝ่ายละ 1 ตัว', 11,
                      MUTED, wraplength=450).pack(pady=(0, 4))
            tk.Button(card, text='กลับไปดูประวัติ', command=close,
                      font=(FONT, 12, 'bold'), bg=BLUE, fg=WHITE,
                      activebackground='#0758AE', activeforeground=WHITE,
                      relief='flat', bd=0, cursor='hand2', pady=10,
                      highlightthickness=2, highlightbackground=WHITE,
                      highlightcolor='#83BDF5').pack(side='bottom', fill='x',
                                                    padx=30, pady=(6, 10))
    shell.card = card
    return shell, dynamic



def center_dialog(top):
    """Center on the screen rather than on the dashboard window."""
    top.update_idletasks()
    width, height = top.winfo_width(), top.winfo_height()
    x = max(0, (top.winfo_screenwidth()-width)//2)
    y = max(0, (top.winfo_screenheight()-height)//2)
    top.geometry(f'{width}x{height}+{x}+{y}')


def _screen_work_area(top):
    """Read the Windows usable desktop so a taskbar cannot cover the popup."""
    area = (0, 0, top.winfo_screenwidth(), top.winfo_screenheight())
    if top.tk.call('tk', 'windowingsystem') == 'win32':
        try:
            import ctypes
            from ctypes import wintypes
            rect = wintypes.RECT()
            if ctypes.windll.user32.SystemParametersInfoW(48, 0, ctypes.byref(rect), 0):
                area = (rect.left, rect.top, rect.right, rect.bottom)
        except (OSError, AttributeError):
            pass
    return area


def bottom_right_dialog(top):
    """Keep a compact turn status within the usable desktop, with a margin."""
    top.update_idletasks()
    left, upper, right, bottom = _screen_work_area(top)
    width, height = top.winfo_width(), top.winfo_height()
    frame_x = max(0, top.winfo_rootx()-top.winfo_x())
    frame_y = max(0, top.winfo_rooty()-top.winfo_y())
    x = max(left, right-width-2*frame_x-24)
    y = max(upper, bottom-height-frame_y-frame_x-24)
    top.geometry(f'{width}x{height}+{x}+{y}')


def position_dialog(top):
    if getattr(top, '_soft_sky_state', None) in ('human_turn', 'ai_turn'):
        bottom_right_dialog(top)
    else:
        center_dialog(top)


def fit_dialog(top):
    """Fit changing text and retain the position assigned to its state."""
    top.update_idletasks()
    want = max(top._soft_sky_min_height,
               top._soft_sky_panel.card.winfo_reqheight()+60)
    height = min(want, max(top._soft_sky_min_height,
                           top.winfo_screenheight()-90))
    if top.winfo_height() != height:
        top.geometry(f'{top.winfo_width()}x{height}')
    position_dialog(top)


def build_dialog(parent, state, *, text=None):
    """No timers, modal grabs, game state updates or device calls here."""
    top = tk.Toplevel(parent)
    top.title(STATES[state][1])
    width, height = (500, 260) if state in ('human_turn', 'ai_turn') else (560, 400)
    if state == 'illegal':
        width, height = 630, 420
    if state == 'draw':
        height = 440
    top.configure(bg=SKY)
    top.geometry(f'{width}x{height}')
    top.minsize(width, height)
    panel, message = make_panel(top, state, top.destroy, text=text)
    panel.pack(fill='both', expand=True)
    top._soft_sky_panel = panel
    top._soft_sky_min_height = height
    top._soft_sky_state = state
    fit_dialog(top)
    return top, message
