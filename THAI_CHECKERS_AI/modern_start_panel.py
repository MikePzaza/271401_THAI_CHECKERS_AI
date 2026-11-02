"""Modern, responsive start-screen control panel for Thai Checkers 8x8.

Presentation only: all actions delegate to the original App handlers.
No camera, robotics, game rules, or network operations are implemented here.
"""
from __future__ import annotations

import sys
import tkinter as tk
from PIL import Image, ImageDraw, ImageFilter, ImageTk

# Logical geometry.  Images and hit regions share these coordinates.
W, H = 720, 710
DPR = 2
BLUE = '#0875F7'
NAVY = '#0F3168'
MUTED = '#6583A9'


class ModernStartPanel(tk.Canvas):
    LEVELS = ('ง่าย', 'ปานกลาง', 'ยาก')
    CAPTIONS = ('เหมาะสำหรับผู้เริ่มต้น', 'ระดับมาตรฐาน', 'เหมาะสำหรับผู้เล่นมีประสบการณ์')

    def __init__(self, master, *, selected_level, robot_connection_text, status_text,
                 probe_getter, choose_level, check_connection, start_game, show_rules,
                 show_settings, exit_program):
        super().__init__(master, bg='#E6F2FF', borderwidth=0,
                         highlightthickness=0, cursor='arrow', takefocus=1)
        self.selected_level = selected_level
        self.robot_connection_text = robot_connection_text
        self.status_text = status_text
        self.probe_getter = probe_getter
        self.actions = {
            'level:ง่าย': lambda: choose_level('ง่าย'),
            'level:ปานกลาง': lambda: choose_level('ปานกลาง'),
            'level:ยาก': lambda: choose_level('ยาก'),
            'check': check_connection,
            'start': start_game,
            'rules': show_rules,
            'settings': show_settings,
            'exit': exit_program,
        }
        self.font = 'Leelawadee UI' if sys.platform == 'win32' else 'Noto Sans Thai'
        self._scale = 1
        self._offset = (0, 0)
        self._areas = []
        self._hover = None
        self._photo = None
        self._draw_after = None
        self._image_cache = {}
        self.bind('<Configure>', self._on_resize)
        self.bind('<Motion>', self._motion)
        self.bind('<Leave>', self._leave)
        self.bind('<Button-1>', self._click)
        self.bind('<Return>', self._keyboard_start)
        self._status_trace = self.robot_connection_text.trace_add('write', self._update_status)
        self._hint_trace = self.status_text.trace_add('write', self._update_status)
        self.bind('<Destroy>', self._on_destroy)
        self._redraw()

    def _on_destroy(self, event):
        if event.widget is self:
            try:
                self.robot_connection_text.trace_remove('write', self._status_trace)
                self.status_text.trace_remove('write', self._hint_trace)
            except (tk.TclError, ValueError):
                pass
            if self._draw_after is not None:
                try:
                    self.after_cancel(self._draw_after)
                except tk.TclError:
                    pass
                self._draw_after = None

    def _on_resize(self, _event):
        self._schedule_redraw()

    def _update_status(self, *_args):
        if self.winfo_exists():
            self._schedule_redraw()

    def update_selection(self):
        self._schedule_redraw()

    def _schedule_redraw(self):
        if self._draw_after is None:
            self._draw_after = self.after_idle(self._redraw)

    def _colors(self):
        raw = self.robot_connection_text.get()
        probe = self.probe_getter()
        if 'กำลังตรวจ' in raw:
            return 'กำลังตรวจสอบ', '#E6F2FF', '#125FC4', 'กำลังตรวจสอบการเชื่อมต่อ...'
        if probe is not None and probe.connected:
            return ('ติดต่อได้', '#E3F7EB', '#148447', raw.replace('MG400: ', '', 1))
        if ('ไม่เชื่อมต่อ' in raw or 'ผิดพลาด' in raw or 'ตรวจสอบไม่ได้' in raw
                or 'timed out' in raw or 'ไม่ถูกต้อง' in raw):
            return ('ไม่เชื่อมต่อ', '#FFE9E9', '#CC2B35', raw.split('—', 1)[-1].strip() if '—' in raw else raw.replace('MG400: ', '', 1))
        return 'ยังไม่ตรวจสอบ', '#EAF2FB', '#54769A', 'ตรวจสอบ MG400 ก่อนเริ่มเล่น'

    @staticmethod
    def _rounded(draw, rect, radius, fill, outline=None, width=1):
        draw.rounded_rectangle(rect, radius=radius, fill=fill, outline=outline, width=width)

    def _surface(self, selection, hover, ready, status):
        key = (selection, hover, ready, status)
        if key in self._image_cache:
            return self._image_cache[key]
        mul = DPR
        img = Image.new('RGBA', (W * mul, H * mul), (230, 242, 255, 255))
        def R(box):
            return tuple(round(v * mul) for v in box)
        def rounded(box, rad, color, outline=None, line=1):
            d = ImageDraw.Draw(img)
            d.rounded_rectangle(R(box), radius=round(rad*mul), fill=color,
                                outline=outline, width=line*mul)
        def softshadow(box, rad, blur=13, alpha=33):
            lay = Image.new('RGBA', img.size, (0,0,0,0))
            d = ImageDraw.Draw(lay)
            b=(box[0]+3, box[1]+5, box[2]+3, box[3]+5)
            d.rounded_rectangle(R(b), radius=rad*mul, fill=(19, 74, 135, alpha))
            img.alpha_composite(lay.filter(ImageFilter.GaussianBlur(blur*mul)))
        def gradrounded(box, rad, top, bottom, outline=None, line=1):
            bbox = R(box)
            width = bbox[2]-bbox[0]
            height = bbox[3]-bbox[1]
            layer=Image.new('RGBA',(width,height))
            d=ImageDraw.Draw(layer)
            a=tuple(int(top[i:i+2],16) for i in (1,3,5))
            b=tuple(int(bottom[i:i+2],16) for i in (1,3,5))
            for yy in range(height):
                t=yy/max(1,height-1)
                c=tuple(int(a[j]*(1-t)+b[j]*t) for j in range(3))+(255,)
                d.line((0,yy,width,yy), fill=c)
            mask=Image.new('L',(width,height),0)
            ImageDraw.Draw(mask).rounded_rectangle((0,0,width-1,height-1),rad*mul,fill=255)
            img.paste(layer,(bbox[0],bbox[1]),mask)
            if outline:
                ImageDraw.Draw(img).rounded_rectangle(bbox,radius=rad*mul,
                    outline=outline,width=line*mul)

        softshadow((5, 5, W-5, H-5), 20, 13, 26)
        gradrounded((4,4,W-4,H-4), 19, '#FFFFFF', '#F7FBFF', '#DFEAF7')
        # header underline and icon disc
        rounded((100, 70, W-28, 72), 1, '#DFE9F5')
        rounded((100,70,167,73), 2, BLUE)
        rounded((28, 16, 92, 80), 33, '#E5F1FF')
        gradrounded((39,27,81,69), 22, '#389AFE', '#0058CD')
        # options
        cells=[(26,100,240,251),(253,100,467,251),(480,100,694,251)]
        for i,box in enumerate(cells):
            name=self.LEVELS[i]
            sel=(selection==name)
            is_hover=hover==f'level:{name}'
            if sel:
                softshadow(box,15,9,35)
                gradrounded(box,15,'#F4FAFF','#E2F1FF','#1482FF',2)
            elif is_hover:
                gradrounded(box,15,'#FFFFFF','#E8F5FF','#8BBEF3')
            else:
                gradrounded(box,15,'#FFFFFF','#F8FBFF','#DCE7F3')
        # status card
        softshadow((22,271,698,405),17,10,21)
        gradrounded((22,271,698,405),17,'#FFFFFF','#FBFDFF','#E1EAF5')
        rounded((47,296,111,360),34,'#E5F1FF')
        status_color = {
            'ไม่เชื่อมต่อ': '#FFE6E8',
            'ติดต่อได้': '#E3F8EA',
            'กำลังตรวจสอบ': '#E4F0FF',
            'ยังไม่ตรวจสอบ': '#EDF3F9',
        }.get(status, '#EDF3F9')
        rounded((128, 324, 282, 357), 16, status_color)
        rounded((358,294,360,381),1,'#DFE8F4')
        if hover=='check':
            gradrounded((382,304,679,372),13,'#ECF7FF','#D9EBFF','#71B4FF')
        else:
            gradrounded((382,304,679,372),13,'#FFFFFF','#E9F3FF','#A5CDFF')
        # big start button. Disabled remains visible but muted.
        softshadow((26,425,694,501),13,9,27 if ready else 14)
        if ready:
            gradrounded((26,425,694,501),13,'#2698FF' if hover=='start' else '#238DFF',
                        '#0053C5' if hover=='start' else '#0758C8','#519FFC')
        else:
            gradrounded((26,425,694,501),13,'#BCD8EF','#8EB6DA','#B5D0E7')
        # utility buttons
        for action, box in [('rules',(26,523,352,595)),('settings',(368,523,694,595))]:
            softshadow(box,13,7,20)
            if hover == action:
                gradrounded(box,13,'#FFFFFF','#D5EAFF','#80B9FF')
            else:
                gradrounded(box,13,'#FFFFFF','#EBF5FF','#A5CEFF')
        # bottom line
        rounded((28,661,242,663),1,'#C8DBF3')
        rounded((477,661,692,663),1,'#C8DBF3')
        self._image_cache[key] = img.convert('RGB')
        if len(self._image_cache)>8:
            self._image_cache.pop(next(iter(self._image_cache)))
        return self._image_cache[key]

    def _font(self, size, bold=False):
        size = max(8, round(size * self._scale))
        return (self.font, size, 'bold' if bold else 'normal')

    def _at(self,x,y):
        ox,oy=self._offset
        return ox+x*self._scale,oy+y*self._scale

    def _text(self,x,y,string,size=14,color=NAVY,bold=False,anchor='center',width=None):
        kw={}
        if width is not None:
            kw['width']=width*self._scale
        self.create_text(*self._at(x,y),text=string,fill=color,anchor=anchor,
                         font=self._font(size,bold),justify='center' if anchor=='center' else 'left',**kw)

    def _circle(self,x,y,r,fill,outline='',thick=1):
        a=self._at(x-r,y-r);b=self._at(x+r,y+r)
        self.create_oval(*a,*b,fill=fill,outline=outline,width=thick*self._scale)

    def _line(self,x1,y1,x2,y2,color,width):
        self.create_line(*self._at(x1,y1),*self._at(x2,y2),fill=color,width=max(1,self._scale*width),capstyle=tk.ROUND)


    def _diamond(self, cx, cy, r, fill, outline='', thick=1):
        pts=[]
        for dx,dy in ((0,-r),(r,0),(0,r),(-r,0)):
            x,y=self._at(cx+dx,cy+dy)
            pts.extend((x,y))
        self.create_polygon(pts, fill=fill, outline=outline, width=max(1, self._scale*thick))

    def _star(self, cx, cy, r_outer, r_inner, fill, outline='', thick=1):
        import math
        pts=[]
        for i in range(10):
            ang = math.radians(-90 + i*36)
            r = r_outer if i % 2 == 0 else r_inner
            x = cx + math.cos(ang) * r
            y = cy + math.sin(ang) * r
            xx, yy = self._at(x, y)
            pts.extend((xx, yy))
        self.create_polygon(pts, fill=fill, outline=outline, width=max(1, self._scale*thick), joinstyle=tk.ROUND)

    def _checker_piece(self, cx, cy, size, base_color, ring_color='#FFFFFF', crown=False, accent_color=None):
        accent_color = accent_color or '#2D8DFF'
        # shadow
        self._circle(cx, cy + size*0.12, size*0.98, '#DCE7F3')
        # body
        self._circle(cx, cy, size, base_color, '#D7E5F6', 1)
        self._circle(cx, cy - size*0.05, size*0.72, self._mix(base_color, '#FFFFFF', 0.12), self._mix(base_color, '#FFFFFF', 0.20), 1)
        self._circle(cx - size*0.25, cy - size*0.28, size*0.15, '#FFFFFF')
        # edge highlight
        self._line(cx - size*0.46, cy - size*0.46, cx + size*0.22, cy - size*0.62, '#FFFFFF', 1.5)
        if crown:
            base_y = cy - size*0.95
            pts = [
                (cx - size*0.52, base_y + size*0.56),
                (cx - size*0.34, base_y + size*0.06),
                (cx - size*0.08, base_y + size*0.40),
                (cx + 0.00,   base_y + 0.00),
                (cx + size*0.10, base_y + size*0.42),
                (cx + size*0.34, base_y + size*0.08),
                (cx + size*0.52, base_y + size*0.56),
            ]
            coords=[]
            for x,y in pts:
                xx,yy=self._at(x,y)
                coords.extend((xx,yy))
            self.create_polygon(coords, fill='#F6B117', outline='#D28B00', width=max(1, self._scale*1.2), joinstyle=tk.ROUND)
            self._line(cx - size*0.54, base_y + size*0.56, cx + size*0.54, base_y + size*0.56, '#D28B00', 1.2)
            for bx,by,rr in ((cx - size*0.34, base_y + size*0.06, size*0.06), (cx + 0.00, base_y + 0.00, size*0.06), (cx + size*0.34, base_y + size*0.08, size*0.06)):
                self._circle(bx, by, rr, '#FFD766', '#D28B00', 1)
        else:
            # small accent spark for easy/medium for a gamey feel
            self._line(cx - size*1.25, cy - size*0.15, cx - size*1.00, cy - size*0.42, accent_color, 2)
            self._line(cx - size*1.14, cy - size*0.58, cx - size*0.96, cy - size*0.18, accent_color, 2)

    @staticmethod
    def _mix(c1, c2, t):
        def rgb(h):
            h=h.lstrip('#')
            return tuple(int(h[i:i+2],16) for i in (0,2,4))
        a=rgb(c1); b=rgb(c2)
        c=tuple(round(a[i]*(1-t)+b[i]*t) for i in range(3))
        return '#%02X%02X%02X' % c

    def _status_detail(self, detail):
        s=detail.strip().replace('MG400: ','')
        if len(s)>50:
            return s[:47] + '…'
        return s

    def _redraw(self):
        self._draw_after=None
        if not self.winfo_exists():
            return
        w=max(2,self.winfo_width());h=max(2,self.winfo_height())
        self.delete('all')
        scale=min(w/W,h/H)
        self._scale=scale
        ox=(w-W*scale)/2
        oy=(h-H*scale)/2
        self._offset=ox,oy
        bg=Image.new('RGB',(w,h),'#E6F2FF')
        flag,pillbg,pillfg,detail=self._colors()
        panel=self._surface(self.selected_level.get(),self._hover,
                            self.selected_level.get() in self.LEVELS, flag)
        rendered=panel.resize((max(1,int(W*scale)),max(1,int(H*scale))),Image.Resampling.LANCZOS)
        bg.paste(rendered,(round(ox),round(oy)))
        self._photo=ImageTk.PhotoImage(bg,master=self)
        self.create_image(0,0,image=self._photo,anchor='nw')
        # Header
        self._text(60,48,'⚙',24,'white',True)
        self._text(106,48,'กรุณาเลือกระดับความยาก',20,NAVY,True,'w')
        # Level cards with game-themed difficulty icons (piece + stars)
        piece_styles = {
            'ง่าย': ('#1D2230', False, 1),
            'ปานกลาง': ('#F18822', False, 2),
            'ยาก': ('#1D2230', True, 3),
        }
        for i,level in enumerate(self.LEVELS):
            left=26+i*227
            sel=level==self.selected_level.get()
            self._circle(left+31,135,12,'#FFFFFF','#1681F7' if sel else '#607F9F',2)
            if sel:
                self._circle(left+31,135,6,'#127AF0')
            # subtle game-board diamonds behind the main icon
            for dx,dy,fill in [(-24,4,'#EAF3FF'), (0,-6,'#DCEEFF'), (24,4,'#EAF3FF'), (-12,18,'#D4E8FF'), (12,18,'#DDEEFF')]:
                self._diamond(left+107+dx, 156+dy, 11, fill)
            base_color, crown, stars = piece_styles[level]
            self._checker_piece(left+107, 154, 24, base_color, crown=crown,
                                accent_color='#2C85F6' if level != 'ปานกลาง' else '#F28C28')
            # star rating
            star_y = 191
            start_x = left + 75
            for s in range(3):
                fill = '#157DFA' if s < stars else '#BFD0E6'
                self._star(start_x + s*32, star_y, 9, 4.2, fill)
            self._line(left+44, 205, left+170, 205, '#D4E3F5', 1)
            self._text(left+107,218,level,19,NAVY,True)
            self._text(left+107,238,self.CAPTIONS[i],10.5,MUTED)
        # Real robot probe status. TCP preflight is not safety clearance.
        self._text(79,328,'⚙',27,'#0755A9',True)
        self._text(129,306,'สถานะ MG400',18,NAVY,True,'w')
        self._circle(148,339,5,pillfg)
        self._text(161,339,flag,14,pillfg,True,'w')
        self._text(129,374,self._status_detail(detail),12,MUTED,False,'w',210)
        self._text(407,337,'⟳',26,'#0750A9',True)
        self._text(435,337,'ตรวจสอบการเชื่อมต่อ MG400',13,'#064395',True,'w')
        # Large start button
        self._text(305,463,'▶',26,'white',True)
        self._text(403,464,'เริ่มเกม',29,'white',True)
        self._text(72,559,'▤',25,'#084394',True)
        self._text(202,560,'ดูกติกา / ผังหมายเลข',16,'#073B86',True)
        self._text(411,559,'⚙',25,'#084394',True)
        self._text(563,560,'ตั้งค่ากล้อง / MG400',16,'#073B86',True)
        # Keep status guidance *inside* the right card, not beneath both panels.
        # Bound its length and width to prevent clipping at smaller window sizes.
        hint = self.status_text.get().strip()
        if hint:
            if len(hint) > 85:
                hint = hint[:82].rstrip() + '…'
            self._text(360,621,hint,11,MUTED,False,'center',640)
        self._text(294,663,'⇥',21,'#1755A4',True)
        self._text(398,663,'ออกจากโปรแกรม',15,'#315B8D',False)
        # Interactive hit areas (logical x1,y1,x2,y2)
        self._areas=[
            ('level:ง่าย',(26,100,240,251)),
            ('level:ปานกลาง',(253,100,467,251)),
            ('level:ยาก',(480,100,694,251)),
            ('check',(382,304,679,372)),
            ('start',(26,425,694,501)),
            ('rules',(26,523,352,595)),
            ('settings',(368,523,694,595)),
            ('exit',(251,636,471,690)),
        ]

    def _hit(self,event):
        if self._scale<=0: return None
        x=(event.x-self._offset[0])/self._scale
        y=(event.y-self._offset[1])/self._scale
        for name,(a,b,c,d) in self._areas:
            if a<=x<=c and b<=y<=d:
                return name
        return None

    def _motion(self,event):
        action=self._hit(event)
        if action != self._hover:
            self._hover=action
            self.config(cursor='hand2' if action else 'arrow')
            self._schedule_redraw()

    def _leave(self,_event):
        if self._hover:
            self._hover=None
            self.config(cursor='arrow')
            self._schedule_redraw()

    def _click(self,event):
        action=self._hit(event)
        if action=='start' and self.selected_level.get() not in self.LEVELS:
            return
        if action:
            self.actions[action]()

    def _keyboard_start(self,_event):
        if self.selected_level.get() in self.LEVELS:
            self.actions['start']()
