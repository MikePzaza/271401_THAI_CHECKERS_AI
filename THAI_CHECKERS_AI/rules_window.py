"""Soft Sky rules window preview. Uses Tkinter only; no game/device imports."""
import tkinter as tk
from tkinter import font as tkfont
from tkinter import ttk

BG = '#E5F2FD'
WHITE = '#FFFFFF'
BLUE = '#126BCB'
INK = '#193D61'
MUTED = '#637F99'
LINE = '#D7E7F5'


class RulesWindow(tk.Toplevel):
    def __init__(self, parent, on_close=None):
        super().__init__(parent)
        self.title('กติกาหมากฮอส 8 × 8')
        self.configure(bg=BG)
        self.close = on_close or self.destroy
        self.protocol('WM_DELETE_WINDOW', self.close)
        families = tkfont.families(self)
        self.face = 'Tahoma' if 'Tahoma' in families else ('Garuda' if 'Garuda' in families else 'TkDefaultFont')
        width, height = min(920, self.winfo_screenwidth()-60), min(800, self.winfo_screenheight()-100)
        self.geometry(f'{width}x{height}')
        self.minsize(min(800,width), min(600,height))
        self.pages = {}
        self.tab_buttons = {}
        self.active_tab = None
        self._header()
        self._footer()
        self._tabs()
        self._body()
        self._play_page()
        self._result_page()
        self.select_tab('play')
        self.update_idletasks()
        self.geometry(f'+{max(0,(self.winfo_screenwidth()-width)//2)}+{max(0,(self.winfo_screenheight()-height)//2)}')

    def text(self, parent, value, size=11, color=INK, bold=False, bg=WHITE, **kwargs):
        return tk.Label(parent, text=value, bg=bg, fg=color,
                        font=(self.face,size,'bold' if bold else 'normal'), **kwargs)

    def _header(self):
        header = tk.Frame(self, bg=BG)
        header.pack(fill='x', padx=26, pady=(20,12))
        board = tk.Canvas(header,width=72,height=72,bg=BG,highlightthickness=0)
        board.pack(side='left',padx=(0,16))
        for row in range(8):
            for col in range(8):
                x,y=4+col*8,4+row*8
                dark=(row+col)%2==1
                board.create_rectangle(x,y,x+8,y+8,fill='#89BEEA' if dark else WHITE,outline='')
                if dark and row in (0,1,6,7):
                    board.create_oval(x+1,y+1,x+7,y+7,fill='#F7A13C' if row<2 else '#364451',outline='')
        copy = tk.Frame(header,bg=BG)
        copy.pack(side='left')
        self.text(copy,'กติกาหมากฮอส 8 × 8',22,bold=True,bg=BG).pack(anchor='w')
        self.text(copy,'วิธีเดินหมากและเงื่อนไขจบเกมตามโปรแกรมนี้',11,color=MUTED,bg=BG).pack(anchor='w')
        badges = tk.Frame(self,bg=BG)
        badges.pack(fill='x',padx=26,pady=(0,12))
        for col,(name,detail,color) in enumerate([
                ('คุณ','หมากดำ 8 ตัว','#364451'),
                ('AI ADOM','หมากส้ม 8 ตัว','#F7A13C')]):
            badges.columnconfigure(col,weight=1,uniform='players')
            cell=tk.Frame(badges,bg=WHITE,padx=14,pady=6)
            cell.grid(row=0,column=col,sticky='ew',padx=(0,6) if col==0 else (6,0))
            emblem=tk.Canvas(cell,width=30,height=30,bg=WHITE,highlightthickness=0)
            emblem.pack(side='left',padx=(0,10))
            emblem.create_oval(3,3,27,27,fill=color,outline='')
            self.text(cell,name,12,bold=True).pack(side='left')
            self.text(cell,detail,10,color=MUTED).pack(side='right')

    def _footer(self):
        footer=tk.Frame(self,bg=WHITE,highlightthickness=1,highlightbackground=LINE)
        footer.pack(side='bottom',fill='x')
        self.text(footer,'กระดาน 8 × 8  •  เล่นบนช่องสีฟ้า',10,color=MUTED).pack(side='left',padx=26)
        self.return_button=tk.Button(footer,text='กลับไปหน้าก่อนหน้า',command=self.close,
            font=(self.face,11,'bold'),bg=BLUE,fg=WHITE,activebackground='#0758AE',
            activeforeground=WHITE,relief='flat',bd=0,padx=24,pady=10,cursor='hand2',
            highlightthickness=2,highlightbackground=WHITE,highlightcolor='#83BDF5')
        self.return_button.pack(side='right',padx=24,pady=12)

    def _tabs(self):
        bar=tk.Frame(self,bg=BG)
        bar.pack(fill='x',padx=26,pady=(0,12))
        for key,title in [('play','วิธีเล่น'),('result','แพ้–ชนะ / เสมอ')]:
            button=tk.Button(bar,text=title,command=lambda k=key:self.select_tab(k),
                font=(self.face,12,'bold'),relief='flat',bd=0,padx=24,pady=8,
                cursor='hand2',highlightthickness=0)
            button.pack(side='left',padx=(0,8))
            self.tab_buttons[key]=button

    def _body(self):
        frame=tk.Frame(self,bg=BG)
        frame.pack(fill='both',expand=True,padx=26,pady=(0,14))
        self.canvas=tk.Canvas(frame,bg=BG,highlightthickness=0)
        self.scrollbar=ttk.Scrollbar(frame,orient='vertical',command=self.canvas.yview)
        self.scrollbar.pack(side='right',fill='y')
        self.canvas.pack(side='left',fill='both',expand=True)
        self.canvas.configure(yscrollcommand=self.scrollbar.set)
        self.content=tk.Frame(self.canvas,bg=BG)
        self.content_id=self.canvas.create_window(0,0,window=self.content,anchor='nw')
        self.canvas.bind('<Configure>',self._resize)
        self.content.bind('<Configure>',lambda _event:self._scrollregion())

    def _resize(self,event):
        self.canvas.itemconfigure(self.content_id,width=event.width)
        self._scrollregion()

    def _scrollregion(self):
        self.canvas.configure(scrollregion=self.canvas.bbox('all'))

    def _wheel(self,event):
        if self.content.winfo_reqheight()>self.canvas.winfo_height():
            amount=-1 if getattr(event,'num',None)==4 else 1 if getattr(event,'num',None)==5 else -int(event.delta/120)
            self.canvas.yview_scroll(amount,'units')
        return 'break'

    def _bind_wheel(self,widget):
        widget.bind('<MouseWheel>',self._wheel)
        widget.bind('<Button-4>',self._wheel)
        widget.bind('<Button-5>',self._wheel)
        for child in widget.winfo_children():
            self._bind_wheel(child)

    def _card(self,parent,number,title,lines,row,col):
        parent.columnconfigure(col,weight=1,uniform='cards')
        card=tk.Frame(parent,bg=WHITE,padx=16,pady=12,
                      highlightthickness=1,highlightbackground=LINE)
        card.grid(row=row,column=col,sticky='nsew',padx=(0,6) if col==0 else (6,0),pady=(0,12))
        heading=tk.Frame(card,bg=WHITE)
        heading.pack(fill='x',pady=(0,8))
        self.text(heading,number,11,color=BLUE,bold=True).pack(side='left',padx=(0,10))
        self.text(heading,title,13,bold=True).pack(side='left')
        body=[]
        for line in lines:
            label=self.text(card,'•  '+line,11,wraplength=350,justify='left',anchor='w')
            label.pack(fill='x',pady=(0,3))
            body.append(label)
        card.bind('<Configure>',lambda event:[label.configure(wraplength=max(230,event.width-36)) for label in body])
        return card

    def _play_page(self):
        page=tk.Frame(self.content,bg=BG)
        self.pages['play']=page
        self._card(page,'01','หมากธรรมดา',[
            'เดินเฉียงไปข้างหน้า ครั้งละ 1 ช่อง',
            'กินได้เฉพาะทิศไปข้างหน้า',
            'เดินและวางหมากบนช่องสีฟ้า'],0,0)
        self._card(page,'02','การกินหมาก',[
            'มีตากินต้องกิน และต้องกินต่อจนจบ',
            'เลือกเส้นทางกินที่ถูกกติกาได้',
            'ไม่บังคับเลือกทางที่กินมากที่สุด'],0,1)
        self._card(page,'03','การเดินของฮอส',[
            'เดินตามแนวทแยงได้หลายช่อง',
            'กินแล้วลงช่องว่างแรกหลังหมากคู่ต่อสู้',
            'นำหมากที่ถูกกินออกทันที'],1,0)
        self._card(page,'04','การเลื่อนเป็นฮอส',[
            'ถึงแถวสุดท้ายของอีกฝ่าย จะเป็นฮอส',
            'หากกินแล้วเข้าฮอส ให้จบตานั้น',
            'ฮอสใช้จุดสีเขียวเพื่อให้กล้องตรวจ'],1,1)
        note=tk.Frame(page,bg='#D8EAFA',padx=14,pady=10)
        note.grid(row=2,column=0,columnspan=2,sticky='ew')
        message=self.text(note,
            'กระดานจริง: คุณหยิบหมากที่ถูกกินออกและเปลี่ยนฮอสของคุณด้วยมือ ส่วน AI ใช้ MG400\n'
            'กล้องใช้ยืนยันตาเดินเท่านั้น การใช้ MG400 ต้องผ่านการตรวจระบบป้องกันที่หน้างาน',
            10,color=INK,bg='#D8EAFA',justify='left',wraplength=800)
        message.pack(fill='x')
        note.bind('<Configure>',lambda event:message.configure(wraplength=max(300,event.width-30)))

    def _result_page(self):
        page=tk.Frame(self.content,bg=BG)
        self.pages['result']=page
        hero=tk.Frame(page,bg=WHITE,padx=22,pady=16,highlightthickness=1,highlightbackground='#91BDEA')
        hero.pack(fill='x',pady=(0,12))
        self.text(hero,'เสมอทันที',19,color=BLUE,bold=True).pack(anchor='w')
        self.text(hero,'เหลือฮอสฝ่ายละ 1 ตัว และไม่มีหมากอื่นเหลือ',14,bold=True,
                  justify='left',wraplength=780).pack(anchor='w',pady=(2,4))
        self.text(hero,'เกมจบหลังกล้องยืนยันตาเดิน โดยไม่ต้องเดินตาถัดไป',11,color=MUTED,
                  wraplength=780,justify='left').pack(anchor='w')
        grid=tk.Frame(page,bg=BG)
        grid.pack(fill='x')
        self._card(grid,'01','เงื่อนไขชนะ',[
            'อีกฝ่ายไม่มีหมากเหลือบนกระดาน',
            'หรืออีกฝ่ายไม่มีตาเดินที่ถูกกติกา'],0,0)
        self._card(grid,'02','เสมอกรณีอื่น',[
            'ตำแหน่งหมากเดิมและฝ่ายที่เดิน ซ้ำ 3 ครั้ง',
            'ครบ 80 ครึ่งตา โดยไม่กินหรือเลื่อนฮอส'],0,1)
        note=tk.Frame(page,bg='#D8EAFA',padx=16,pady=12)
        note.pack(fill='x')
        self.text(note,'นับครึ่งตาอย่างไร?',12,bold=True,bg='#D8EAFA').pack(anchor='w')
        explanation=self.text(note,
            'คุณเดิน 1 ครั้ง = 1 ครึ่งตา  •  AI เดิน 1 ครั้ง = 1 ครึ่งตา\n'
            'ครบ 80 ครึ่งตา เท่ากับฝ่ายละ 40 ครั้ง ตัวนับเริ่มใหม่เมื่อกินหมากหรือเลื่อนเป็นฮอส',
            10,bg='#D8EAFA',wraplength=800,justify='left')
        explanation.pack(fill='x',pady=(3,0))
        note.bind('<Configure>',lambda event:explanation.configure(wraplength=max(300,event.width-36)))

    def select_tab(self,key):
        for page in self.pages.values():
            page.pack_forget()
        self.pages[key].pack(fill='x')
        self.active_tab=key
        for tab,button in self.tab_buttons.items():
            selected=tab==key
            button.configure(bg=BLUE if selected else WHITE,fg=WHITE if selected else INK,
                             activebackground='#0758AE' if selected else '#D1E8FA',
                             activeforeground=WHITE if selected else INK)
        self.update_idletasks()
        self._scrollregion()
        self.canvas.yview_moveto(0)
        self._bind_wheel(self.pages[key])


def main():
    root=tk.Tk()
    root.withdraw()
    RulesWindow(root,on_close=root.destroy)
    root.mainloop()


if __name__=='__main__':
    main()
