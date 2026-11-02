"""
MG400 + Webcam (Eye-to-hand) : จับหมากสีส้มและสีดำ

ใช้งาน:
  python mg400_vision.py calibrate   # สอบเทียบกล้อง -> หุ่น (ทำครั้งเดียว / ทำใหม่เมื่อกล้องขยับ)
  python mg400_vision.py camfocus    # ปรับโฟกัสกล้องด้วยแถบเลื่อน
  python mg400_vision.py setclear    # บันทึกจุดหลีกกล้อง (ทำครั้งเดียว)
  python mg400_vision.py setdiscard  # บันทึกจุดทิ้งหมาก (ทำครั้งเดียว ใช้กับคำสั่ง x ในโหมด play)
  python mg400_vision.py pumptest    # ทดสอบปั๊ม (ไม่ขยับแขน)
  python mg400_vision.py pose        # อ่านตำแหน่งปัจจุบัน (ใช้วัด PICK_Z)
  python mg400_vision.py run         # ตรวจจับวัตถุ -> ไปเหนือวัตถุ -> ค้าง -> ถอยไปจุดหลีกกล้อง
  python mg400_vision.py boardcal    # สอบเทียบกระดาน 8x8 (วางหมากตำแหน่งเริ่มเกมก่อน)
  python mg400_vision.py play        # สั่งย้ายหมากตามเลขช่อง 1-64 เช่น พิมพ์ 10 19
  python mg400_vision.py setroi      # ลากกรอบสี่เหลี่ยมเฉพาะบริเวณกระดาน+ที่เก็บฮอส (กล้องจะไม่สนใจมือ/ของนอกกรอบ)
  python mg400_vision.py setstock    # หาตำแหน่งจุดเก็บตัวฮอสจากกล้อง (วางฮอสบนจุดสีเขียวทั้ง 8 จุดก่อน) ทำครั้งเดียว
  python mg400_vision.py teachstock  # (แนะนำ) สอนตำแหน่งจุดเก็บฮอสด้วยมือ ลากหัวดูดไปกลางฮอสแต่ละตัว -> แม่นกว่า
  (ในโหมด play พิมพ์ k 64 = เลื่อนขั้นหมากที่ช่อง 64 เป็นฮอส)
"""
import math
import os
import queue
import re
import socket
import sys
import threading
import time

import cv2
import numpy as np

# ==============================
# CONFIG
# ==============================
ROBOT_IP = "192.168.1.6"
DASHBOARD_PORT = 29999
MOVE_PORT = 30003
CAM_INDEX = 0
ROTATE_180 = True           # หมุนภาพกล้อง 180 องศา (ถ้าเปลี่ยนค่านี้ ต้องสอบเทียบใหม่)
CAM_EXPOSURE = None         # None = ออโต้ / ตัวเลข = ล็อกค่า exposure (DSHOW ใช้ค่าลบ เช่น -6, -5, -4 ยิ่งน้อยยิ่งมืด) กันแสงแกว่งตอนแขน/มือผ่านทำให้สีเพี้ยน
CAM_FOCUS = None            # None = autofocus / 0-255 = ล็อกโฟกัสด้วยมือ (หาค่าด้วยโหมด camfocus)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CALIB_FILE = os.path.join(BASE_DIR, "calib.npz")
CLEAR_FILE = os.path.join(BASE_DIR, "clear_pos.npz")   # จุดหลีกกล้อง (ตั้งด้วยโหมด setclear)
BOARD_FILE = os.path.join(BASE_DIR, "board.npz")       # แผนที่ช่องกระดาน (ตั้งด้วยโหมด boardcal)
DISCARD_FILE = os.path.join(BASE_DIR, "discard_pos.npz")  # จุดทิ้งหมาก (ตั้งด้วยโหมด setdiscard)
ROI_FILE = os.path.join(BASE_DIR, "roi.npz")              # กรอบพื้นที่ตรวจจับ (ตั้งด้วยโหมด setroi)
STOCK_FILE = os.path.join(BASE_DIR, "stock_pos.npz")      # จุดเก็บตัวฮอส (ตั้งด้วยโหมด setstock)

ALLOW_MOVE = True           # True = หุ่นขยับจริง / False = ดูอย่างเดียวไม่สั่งหุ่น
TOOL_R = 0.0                # องศา: Rotation ของ Tool ล็อกที่ 0 ทุกการเคลื่อนที่ (ค่า r ที่บันทึกไว้ในจุดหลีก/จุดทิ้งถูกเมินทั้งหมด)
SPEED = 85                  # % ความเร็ว
HOVER_ABOVE = 30.0          # mm เหนือระนาบที่สอบเทียบ
HOLD_SECONDS = 0.5          # ค้างหลังยกวัตถุขึ้น ก่อนถอยไปจุดหลีกกล้อง

# ---- ปั๊มสุญญากาศ ----
USE_PUMP = True             # True = ลงดูดวัตถุด้วยปั๊ม / False = แค่ไปเหนือวัตถุเหมือนเดิม
VAC_DO = 2                  # DO2 = ดูดสุญญากาศ (vacuum)
BLOW_DO = 1                 # DO1 = เป่าลม (pressure) ใช้ปล่อยวัตถุ
BLOW_SECONDS = 0.5          # เวลาเป่าลมตอนปล่อยวัตถุ
PUMP_SETTLE = 1.0           # วินาทีที่รอให้แรงดูดจับวัตถุหลังเปิดปั๊ม
PICK_Z = -151.8               # Z (mm) ที่หัวดูดแตะผิวบนของวัตถุ -> ต้องวัดเอง (ใช้โหมด pose)
# ขอบเขตพื้นที่ทำงานที่ปลอดภัย (mm) -> แก้ให้ตรงกับโต๊ะของคุณ
X_RANGE = (185.0, 415.0)   # 8x8: กระดานใหญ่ขึ้น ต้องรัน calibrate แล้วแก้ตามค่าที่พิมพ์ออกมา
Y_RANGE = (-125.0, 125.0)
Z_RANGE = (-157.0, 0.0)

# ช่วงสี HSV ของวัตถุ -> ปรับตามวัตถุจริง (กด 'm' ตอนรัน เพื่อดูภาพ mask ช่วยปรับค่า)
#   H: 0-179, S: 0-255, V: 0-255
#   "circ" = ความกลมขั้นต่ำ (0 = ไม่กรอง) ใช้กรองเงา/เส้นขอบสีเข้มที่ไม่ใช่หมาก
COLORS = {
    "orange": {"low": (5, 120, 120), "high": (22, 255, 255), "circ": 0.0},
    # ดำ: หมากจริงวัดได้ H~100-107 (ออกฟ้านิด ๆ) S~35-70 V~55-100 -> เผื่อแสงเปลี่ยน/แสงสะท้อนด้วย V<=150, S<=170 (ถ้าจับเกิน เช่น เงา ให้ลด V; ถ้ายังหลุด ให้เพิ่ม V ทีละ 10)
    #   "close" = ขนาดตัวปิดรู (px) กันแสงสะท้อนบนหน้าหมาก/จุดเขียวทำให้ mask ขาดเป็นหลายชิ้น
    "black":  {"low": (0, 0, 0),     "high": (179, 170, 150), "circ": 0.5, "close": 5},
}

TAG = {"orange": "O", "black": "B"}                       # ตัวอักษรต่อท้ายหมายเลขบนภาพ
DRAW_COLOR = {"orange": (0, 255, 0), "black": (255, 0, 255)}  # สีวงกลม (BGR) ที่ใช้วาด
SNAP_DARK_TOL = 0.7            # หมากเยื้องกลางช่อง/คร่อมขอบ: ปัดเข้าช่องเดินได้ที่ใกล้สุดถ้าห่างไม่เกินเท่านี้ (หน่วย = ช่อง; 0 = ปิด)
OCC_VOTE_FRAMES = 5           # โหวตการเห็นหมากในช่องจากกี่เฟรมล่าสุด (เห็นอย่างน้อยครึ่งหนึ่งถึงนับว่ามี) แก้หมากริมกระดานที่เห็นๆ หายๆ; ตั้ง 1 = ปิด
STABLE_STREAK = 5              # ต้องเห็นกระดานเหมือนกันติดกันกี่เฟรมถึงนับว่านิ่ง
STABLE_MAX_FRAMES = 40          # รอให้นิ่งได้นานสุดกี่เฟรม (เช่น รอมือพ้น)
MIN_AREA = 400
MAX_AREA = 15000            # ตัดวัตถุที่ใหญ่เกินหมาก (เช่น เงาพื้นที่กว้าง, ตัวหุ่น)
SHOW_OUT_MARGIN = 40.0     # แสดงวัตถุนอกพื้นที่ปลอดภัยเฉพาะที่อยู่ห่างขอบไม่เกินกี่ mm (ซ่อนมือ/ชิ้นส่วนหุ่นที่อยู่ไกล)
MAX_OBJECTS = 24            # จำนวนหมากสูงสุดที่แสดงหมายเลข (กดเลขได้ 1-9, ที่เหลือใช้คลิกเมาส์)

# ---- กระดานหมากฮอส 8x8 (ช่อง 1-64 เริ่มมุมซ้ายล่าง นับทีละแถวขึ้นไปทางบน) ----
BOARD_N = 8
PIECES_PER_SIDE = BOARD_N          # หมากฮอสไทย 8x8 ฝั่งละ 8 ตัว (2 แถวแรก แถวละ 4)
KING_PICK_Z_OFFSET = 0.0     # mm: หยิบฮอสบนกระดานที่ PICK_Z + ค่านี้ (บวก = สูงขึ้น/กดเบาลง) ใช้เฉพาะฮอส
MIN_REACH_X = 238.0          # mm: X ต่ำสุดที่หัวดูดลงไปหยิบ/วางได้ (ใกล้ฐานกว่านี้แขนไปไม่ถึง/ค้าง) ช่องที่ X น้อยกว่านี้จะหยิบที่ X นี้ แล้วชดเชยตอนวาง (None = ปิด)
VIA_DIP_MM = 60.0            # mm: กินหลายตัว -> ที่จุดลงจอดกลางทางแต่ละจุด ย่อแขนลงเท่านี้แล้วยกขึ้น ก่อนไปจุดถัดไป
MOVJ_APPROACH_MM = 0.0      # mm: MovL ทุกท่า -> ใช้ MovJ เข้าไปหยุดห่างเป้า 10 mm ก่อน แล้วค่อย MovL ช่วงสุดท้าย (กัน MovL ติด joint limit; ตั้ง 0 = ปิด ใช้ MovL ตลอดทาง)
MOVL_FALLBACK_MOVJ = True    # True = MovL ตัวไหนก็ตามที่ค้าง (หุ่นไม่ขยับ ไม่มี error) ให้ล้างคิวแล้วลอง MovJ ไปจุดเดิมแทน
MOVL_WAIT = 30.0             # วินาที: รอ MovL นานสุดเท่านี้ก่อนสลับเป็น MovJ (อย่าตั้งสั้นไป: ท่าเดินทางไกลใช้เวลานาน)
ARRIVE_TOL = 2.0             # mm: ถ้าหมดเวลาแต่ตำแหน่งจริงอยู่ห่างเป้าไม่เกินเท่านี้ ถือว่าถึงแล้ว (Sync ไม่ตอบ) ไม่ต้องสั่งซ้ำ
KING_LIFT_METHOD = "movl_high"   # วิธียกฮอสหลังดูดติด: "movl_high" = MovL ตรงขึ้นไป KING_CARRY_Z ทีเดียว (เหมือนตอนยกหมากไปทิ้งที่ยกได้ปกติ) / "movl_movj" = MovL KING_LIFT_MOVL แล้ว MovJ / "movj" = MovJ ทีเดียว
KING_LIFT_FALLBACK_MOVJ = True   # True = ถ้ายกฮอสด้วย MovL แล้วหุ่นค้าง (ไม่ตอบใน KING_LIFT_WAIT วินาที) ให้หยุด/ล้างคิวแล้วลองยกด้วย MovJ แทนทันที
KING_LIFT_WAIT = 5.0         # วินาที: รอ MovL ยกฮอสนานสุดเท่านี้ก่อนสลับไป MovJ
KING_LIFT_MOVL = 20.0        # mm: ดูดฮอสติดแล้วยกตรงด้วย MovL เท่านี้ก่อน แล้วที่เหลือขึ้นด้วย MovJ (ตั้ง 0 = ยกด้วย MovJ ทั้งช่วง)
KING_CARRY_Z = -150.0         # mm: ความสูงที่ยกฮอสขึ้นไปก่อนเดินทาง (หมากธรรมดายังยกแค่ hover เหมือนเดิม) ตอนลงวางใช้ MovL
DISCARD_TRAVEL_RAISE = 40.0  # mm: ยกเพิ่มอีกเท่านี้ (40 = 4 ซม.) จากความสูงเดินทางไปจุดทิ้ง ก่อนเคลื่อนไปเหนือจุดทิ้ง
DISCARD_EXTRA_LIFT = 60.0   # mm: ตอนหยิบหมากไปทิ้ง ยกสูงกว่าความสูง hover อีกเท่านี้ก่อนเดินทาง (กันชนที่ทิ้ง)
# ---- ตัวฮอส (หมากสีเดิม + จุดสีเขียวตรงกลาง) ----
GREEN_LOW = (35, 60, 120)        # HSV ของจุดเขียว (วัดจริง H~41-51, S~103-137, V~168-183) ปรับได้ถ้าจับไม่เจอ
GREEN_HIGH = (70, 255, 255)
GREEN_MIN_AREA = 60              # พื้นที่ (px) ของจุดเขียวขั้นต่ำ
GREEN_MAX_AREA = 3000
GREEN_MIN_CIRC = 0.5             # ความกลมขั้นต่ำของจุด
KING_DOT_DIST = 32               # px: จุดเขียวต้องอยู่ห่างจากกึ่งกลางหมากไม่เกินเท่านี้ ถึงนับว่าเป็นฮอส
STOCK_ORANGE_SIDE = "left"       # ที่เก็บฮอสของหุ่น (ส้ม) อยู่ด้านใดของกระดานในภาพ ("left"/"right") อีกด้านเป็นของคน (ดำ)
STOCK_DYNAMIC = True             # True = หยิบฮอสตามตำแหน่งที่กล้องเห็นจริงตอนนั้น (วางเยื้องจุดได้) / False = หยิบตามจุดที่สอนไว้เป๊ะ ๆ
STOCK_SEARCH_PX = 60             # px: โหมด dynamic หาฮอสในรัศมีนี้รอบจุดเก็บแต่ละจุด
STOCK_MATCH_PX = 30              # px: ฮอสต้องอยู่ห่างจุดเก็บ (ในภาพ) ไม่เกินเท่านี้ ถึงนับว่ายังอยู่ที่จุดนั้น
STOCK_REACH = (180.0, 425.0)     # mm: ระยะจากฐานหุ่นที่แขนเอื้อมถึง (MG400 สูงสุดราว 440) ใช้เช็คจุดเก็บฮอสซึ่งอยู่นอก Y_RANGE
AUTO_PROMOTE = False             # True = พอหุ่นเดินหมากถึงแถวสุดท้าย เปลี่ยนเป็นฮอสให้เองอัตโนมัติ
PLACE_Z_OFFSET = 3.5        # mm: วางหมากที่ระดับ PICK_Z + ค่านี้ (กันหัวดูดกดหมากแรงเกิน)
SQUARE_TOL = 0.46           # หมากต้องอยู่ห่างกึ่งกลางช่องไม่เกินกี่ "ช่อง" ถึงนับว่าอยู่ในช่องนั้น (เผื่อวางมือเยื้อง/มุมกล้องขอบกระดาน)
# ค่าเริ่มต้นของแผนที่ช่อง -> พิกัดหุ่น: X = a0 + a1*col + a2*row , Y = b0 + b1*col + b2*row
# (ประมาณจากภาพจริงของคุณ ควรรัน boardcal เพื่อคำนวณใหม่จากกล้องและกระดานจริง)
BOARD_AFFINE_DEFAULT = [[406.14, -0.81, -30.41],
                        [-74.31, 31.51, -0.82]]
START_BLACK = tuple(n for n in range(1, 2 * BOARD_N + 1) if ((n - 1) // BOARD_N + (n - 1) % BOARD_N) % 2 == 0)  # ฝั่งคน (สีดำ) 2 แถวล่าง = 1,3,5,7,10,12,14,16
START_ORANGE = tuple(n for n in range(BOARD_N * (BOARD_N - 2) + 1, BOARD_N * BOARD_N + 1) if ((n - 1) // BOARD_N + (n - 1) % BOARD_N) % 2 == 0)  # ฝั่งหุ่น (สีส้ม) 2 แถวบน


# ==============================
# ROBOT
# ==============================
class MG400:
    def __init__(self, ip):
        self.dash = self._open(ip, DASHBOARD_PORT, 5)
        self.move = self._open(ip, MOVE_PORT, 30)  # Sync() อาจรอนาน (30 วิ: ค้างกว่านี้ถือว่าหุ่นไม่ไปต่อ)

    @staticmethod
    def _open(ip, port, timeout):
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.settimeout(timeout)
        s.connect((ip, port))
        return s

    @staticmethod
    def _txrx(sock, cmd):
        sock.sendall(cmd.encode())
        buf = b""
        while not buf.endswith(b";"):
            chunk = sock.recv(1024)
            if not chunk:
                raise ConnectionError("Robot closed connection")
            buf += chunk
        resp = buf.decode().strip()
        print(f">> {cmd}   << {resp}")
        err = int(resp.split(",")[0])
        if err != 0:
            raise RuntimeError(f"Robot error {err}: {resp}")
        return resp

    def d(self, cmd):
        return self._txrx(self.dash, cmd)

    def m(self, cmd):
        return self._txrx(self.move, cmd)

    def set_motion_timeout(self, seconds):
        self.move.settimeout(seconds)

    def recover_motion(self):
        """หลังคำสั่งเคลื่อนที่ค้าง: สั่งหยุด/ล้างคิวคำสั่ง (ถ้าเฟิร์มแวร์รองรับ) แล้วเทข้อมูลค้างในซ็อกเก็ตทิ้ง เพื่อไม่ให้คำตอบเก่าปนกับคำสั่งถัดไป"""
        for cmd in ("ResetRobot()", "ClearError()"):
            try:
                self.d(cmd)
            except Exception as e:
                print(f"   recover: {cmd} -> {e!r}")
        try:
            self.move.settimeout(0.3)
            while self.move.recv(4096):
                pass
        except Exception:
            pass
        finally:
            self.move.settimeout(30)
        time.sleep(0.5)

    def diagnose(self):
        """ถามสถานะหุ่นหลังเกิดปัญหา (ไม่ raise) เพื่อบอกสาเหตุที่หุ่นไม่ยอมไปต่อ"""
        out = []
        for name in ("RobotMode()", "GetErrorID()", "GetPose()"):
            try:
                self.dash.settimeout(3)
                self.dash.sendall(name.encode())
                buf = b""
                while not buf.endswith(b";"):
                    chunk = self.dash.recv(1024)
                    if not chunk:
                        break
                    buf += chunk
                out.append(buf.decode().strip())
            except Exception as e:
                out.append(f"{name}: {e!r}")
        return " | ".join(out)

    def pose(self):
        resp = self.d("GetPose()")
        vals = re.search(r"\{(.*?)\}", resp).group(1).split(",")
        return [float(v) for v in vals[:4]]  # x, y, z, r

    def clear_error_soft(self):
        """ClearError แบบไม่ล้ม: ถ้าหุ่นปฏิเสธ (-1 = ยังมี Alarm/E-Stop ค้าง) แค่เตือน แล้วทำต่อ (setclear/setdiscard ใช้ปุ่ม drag ลากหุ่นออกเองได้)"""
        try:
            self.d("ClearError()")
            return True
        except Exception as e:
            print(f"!! ClearError ไม่สำเร็จ ({e}) — หุ่นยังมี Alarm/E-Stop ค้าง: ปลด E-Stop, เคลียร์ Alarm ใน DobotStudio,"
                  " ถ้าเป็น joint limit ให้กดปุ่ม drag ที่แขนลากออกจากจุดนั้นก่อน แล้ว Enable")
            try:
                print("   สถานะ:", self.diagnose())
            except Exception:
                pass
            self._offer_enable()
            return False

    def _offer_enable(self):
        """RobotMode 4 = ยังไม่ Enable (มอเตอร์ปล่อย) -> ถามผู้ควบคุมก่อน ถ้าพิมพ์ y ถึงสั่ง EnableRobot() (ไม่สั่งเองโดยอัตโนมัติ และไม่ขยับหุ่น)"""
        try:
            mode = self.d("RobotMode()")
        except Exception:
            return
        if "{4}" not in mode:
            if "{9}" in mode:
                print("   RobotMode=9 (Error): ปลด E-Stop/เคลียร์ Alarm ใน DobotStudio ก่อน")
            return
        print("   RobotMode=4: หุ่นยังไม่ถูก Enable (อ่านตำแหน่ง/เคลียร์ error ไม่ได้ ยังไม่ได้เกิดจาก Alarm)")
        try:
            ans = input("   ให้โปรแกรมสั่ง EnableRobot() ตอนนี้ไหม? (หุ่นจะล็อกมอเตอร์ ไม่เคลื่อนที่) พิมพ์ y แล้ว Enter [เว้นว่าง = ไม่]: ")
        except EOFError:
            return
        if ans.strip().lower() != "y":
            print("   ข้าม — กด Enable เองใน DobotStudio แล้วรันใหม่")
            return
        try:
            self.d("EnableRobot()")
            for _ in range(30):
                if "{5}" in self.d("RobotMode()"):
                    print("   Enable สำเร็จ (RobotMode=5)")
                    return
                time.sleep(0.5)
            print("   !! Enable แล้วแต่ยังไม่เข้าโหมด 5 ตรวจที่ DobotStudio")
        except Exception as e:
            print(f"   !! EnableRobot ไม่สำเร็จ: {e}")

    def enable(self):
        self.d("ClearError()")
        self.d("EnableRobot()")
        for _ in range(30):
            if "{5}" in self.d("RobotMode()"):
                break
            time.sleep(0.5)
        else:
            raise RuntimeError("Robot did not reach enabled state")
        self.d(f"SpeedFactor({SPEED})")

    # --- ปั๊ม: ไม่เปิดดูดกับเป่าพร้อมกันเด็ดขาด ---
    def vacuum_on(self):
        self.d(f"DO({BLOW_DO},0)")
        self.d(f"DO({VAC_DO},1)")

    def pump_off(self):
        """ปิดทั้งดูดและเป่า (ใช้ตอนเริ่ม / เกิด error / ออกโปรแกรม)"""
        self.d(f"DO({VAC_DO},0)")
        self.d(f"DO({BLOW_DO},0)")

    def release(self):
        """ปล่อยวัตถุ: ปิดดูด -> เป่าลมสั้น ๆ -> ปิดเป่า"""
        self.d(f"DO({VAC_DO},0)")
        self.d(f"DO({BLOW_DO},1)")
        time.sleep(BLOW_SECONDS)
        self.d(f"DO({BLOW_DO},0)")

    def close(self):
        for s in (self.dash, self.move):
            try:
                s.close()
            except Exception:
                pass


# ==============================
# CAMERA
# ==============================
def read_frame(cap):
    """อ่านภาพจากกล้อง และหมุน 180 องศาถ้าตั้ง ROTATE_180"""
    ok, frame = cap.read()
    if ok and ROTATE_180:
        frame = cv2.rotate(frame, cv2.ROTATE_180)
    return ok, frame


def apply_focus(cap):
    """ตั้งโฟกัส: CAM_FOCUS=None -> autofocus, ไม่งั้นปิด autofocus แล้วล็อกค่าโฟกัส (และตั้ง exposure ถ้ากำหนด CAM_EXPOSURE)"""
    if CAM_FOCUS is None:
        cap.set(cv2.CAP_PROP_AUTOFOCUS, 1)
    else:
        cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)
        cap.set(cv2.CAP_PROP_FOCUS, float(CAM_FOCUS))
    if CAM_EXPOSURE is not None:
        cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)      # 0.25 = manual (DirectShow)
        cap.set(cv2.CAP_PROP_EXPOSURE, float(CAM_EXPOSURE))


def open_camera():
    t0 = time.time()
    print("กำลังเปิดกล้อง ...")
    # Windows: DirectShow เปิดกล้อง USB เร็วกว่า backend เริ่มต้น (MSMF) มาก
    backend = cv2.CAP_DSHOW if sys.platform.startswith("win") else cv2.CAP_ANY
    cap = cv2.VideoCapture(CAM_INDEX, backend)
    if not cap.isOpened():
        raise RuntimeError("Cannot open camera")
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))  # BRIO ส่งภาพเร็วขึ้นเมื่อใช้ MJPG
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    apply_focus(cap)
    for _ in range(10):  # ทิ้งเฟรมแรก ๆ ให้ exposure นิ่ง
        cap.read()
    apply_focus(cap)     # ตั้งซ้ำหลังกล้องเริ่มส่งภาพ (บางรุ่นไม่รับค่าตอนยังไม่มีเฟรม)
    for _ in range(5):
        cap.read()
    print(f"เปิดกล้องเสร็จ ({time.time() - t0:.1f} วินาที)")
    return cap


def camfocus():
    """
    ปรับโฟกัสด้วยแถบเลื่อน: วางวัตถุ/กระดาษที่มีตัวหนังสือบนโต๊ะ (ระดับเดียวกับที่วัตถุจะวาง)
    เลื่อนจนภาพคมที่สุด (ตัวเลข Sharpness สูงสุด) แล้วกด q จะแสดงค่าให้ไปใส่ CAM_FOCUS
    """
    cap = open_camera()
    cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)
    start = int(CAM_FOCUS) if CAM_FOCUS is not None else 100
    win = "camfocus"
    cv2.namedWindow(win)
    state = {"focus": start}

    def on_change(v):
        state["focus"] = v
        cap.set(cv2.CAP_PROP_FOCUS, float(v))

    cv2.createTrackbar("Focus", win, start, 255, on_change)
    cap.set(cv2.CAP_PROP_FOCUS, float(start))
    print("เลื่อนแถบ Focus จนภาพคมที่สุด แล้วกด 'q'")
    try:
        while True:
            ok, frame = read_frame(cap)
            if not ok:
                break
            h, w = frame.shape[:2]
            roi = frame[h // 4: 3 * h // 4, w // 4: 3 * w // 4]  # วัดความคมที่กลางภาพ
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            sharp = cv2.Laplacian(gray, cv2.CV_64F).var()
            cv2.rectangle(frame, (w // 4, h // 4), (3 * w // 4, 3 * h // 4), (0, 255, 255), 1)
            cv2.putText(frame, f"Focus={state['focus']}  Sharpness={sharp:.0f}  (Q = done)",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            cv2.imshow(win, frame)
            if cv2.waitKey(30) & 0xFF in (ord("q"), ord("Q")):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()
    print(f"\nใส่ค่านี้ใน CONFIG:  CAM_FOCUS = {state['focus']}")
    print("หลังเปลี่ยนโฟกัส ควรสอบเทียบ (calibrate) ใหม่")


_ROI_CACHE = {}


def get_roi():
    """คืน (x1, y1, x2, y2) ของกรอบตรวจจับ หรือ None ถ้ายังไม่ได้ตั้ง (ตรวจทั้งภาพ)"""
    if "roi" not in _ROI_CACHE:
        _ROI_CACHE["roi"] = tuple(int(v) for v in np.load(ROI_FILE)["roi"]) if os.path.exists(ROI_FILE) else None
    return _ROI_CACHE["roi"]


def apply_roi(mask):
    """ล้าง mask นอกกรอบตรวจจับให้เป็น 0 (พิกัดพิกเซลยังเป็นของภาพเต็ม จึงไม่ต้อง calibrate ใหม่)"""
    roi = get_roi()
    if roi is None:
        return mask
    x1, y1, x2, y2 = roi
    out = np.zeros_like(mask)
    out[y1:y2, x1:x2] = mask[y1:y2, x1:x2]
    return out


def in_roi(u, v):
    """True ถ้าจุดพิกเซล (u, v) อยู่ในกรอบตรวจจับ (หรือยังไม่ได้ตั้งกรอบ)"""
    roi = get_roi()
    if roi is None:
        return True
    x1, y1, x2, y2 = roi
    return x1 <= u < x2 and y1 <= v < y2


def setroi():
    """ลากเมาส์ครอบเฉพาะกระดาน + ที่เก็บฮอสทั้ง 2 ข้าง (ไม่เอาถาดทิ้งหมาก/มือ/คีย์บอร์ด) แล้วกด Enter/Space"""
    cap = open_camera()
    try:
        for _ in range(8):
            cap.read()
        ok, frame = read_frame(cap)
    finally:
        cap.release()
    if not ok:
        raise RuntimeError("Cannot read frame")
    print("ลากกรอบครอบกระดาน + จุดเก็บฮอสซ้าย/ขวา แล้วกด Enter (กด C เพื่อยกเลิก)")
    r = cv2.selectROI("setroi", frame, showCrosshair=False)
    cv2.destroyAllWindows()
    x, y, w, h = [int(v) for v in r]
    if w < 50 or h < 50:
        print("ไม่ได้เลือกกรอบ ไม่บันทึก")
        return
    np.savez(ROI_FILE, roi=np.array([x, y, x + w, y + h]))
    print(f"บันทึกกรอบ: x {x}-{x + w}, y {y}-{y + h}   ไฟล์: {ROI_FILE}")
    print("(ลบไฟล์ roi.npz ถ้าอยากกลับไปตรวจทั้งภาพ)")


def build_masks(frame):
    """สร้าง mask แยกตามสีที่กำหนดใน COLORS พร้อมอุดรูสติกเกอร์จุดเขียวที่อยู่กลางหมาก"""
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    kernel = np.ones((5, 5), np.uint8)
    green_raw = cv2.inRange(hsv, GREEN_LOW, GREEN_HIGH)
    masks = {}
    for name, cfg in COLORS.items():
        m = cv2.inRange(hsv, cfg["low"], cfg["high"])
        # ถ้ารูจุดเขียวอยู่ติดกับหรืออยู่ข้างในเนื้อหมาก ให้รวมเข้าใน mask หมากเพื่อไม่ให้หมากแหว่งเป็นรูปตัว C
        touching_green = cv2.bitwise_and(green_raw, cv2.dilate(m, kernel))
        if np.any(touching_green):
            m = cv2.bitwise_or(m, touching_green)
        k = cfg.get("close", 0)
        if k:
            m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
        masks[name] = apply_roi(cv2.morphologyEx(m, cv2.MORPH_OPEN, kernel))
    return masks


def _extract_piece_centers(mask, min_circ):
    """สกัดจุดกึ่งกลางหมากจาก mask โดยใช้ Convex Hull ช่วยกู้คืนหมากที่มีแสงสะท้อนแหว่งที่ขอบ"""
    centers = []
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for c in cnts:
        area = cv2.contourArea(c)
        per = cv2.arcLength(c, True)
        circ = (4 * np.pi * area / (per * per)) if per > 0 else 0.0
        hull = cv2.convexHull(c)
        h_area = cv2.contourArea(hull)
        h_per = cv2.arcLength(hull, True)
        h_circ = (4 * np.pi * h_area / (h_per * h_per)) if h_per > 0 else 0.0
        (_, _), (rw, rh), _ = cv2.minAreaRect(hull)
        aspect = min(rw, rh) / max(rw, rh, 1e-5)

        if MIN_AREA <= area <= MAX_AREA and circ >= min_circ:
            # หากหมากมีรอยแหว่งจากแสงสะท้อนหรือจุดเขียว ให้คำนวณจุดศูนย์กลางจาก hull เพื่อไม่ให้พิกัดเบี้ยวหนีศูนย์กลางช่อง
            use_c = hull if (area < 0.88 * h_area and h_area <= 2600 and h_circ >= 0.70 and aspect >= 0.65) else c
            M = cv2.moments(use_c)
            if M["m00"] > 0:
                centers.append((M["m10"] / M["m00"], M["m01"] / M["m00"]))
        elif MIN_AREA <= h_area <= 2600 and area >= max(200.0, 0.42 * h_area) and h_circ >= max(0.65, min_circ) and aspect >= 0.65:
            # กู้คืนหมากทรงกลมที่มีแสงสะท้อนแหว่งจนค่า circ ดิบตกเกณฑ์
            M = cv2.moments(hull)
            if M["m00"] > 0:
                centers.append((M["m10"] / M["m00"], M["m01"] / M["m00"]))
    return centers


def find_objects(frame):
    """คืนรายการ (u, v, ชื่อสี) ของหมากทุกชิ้นที่ใหญ่พอและรูปร่างผ่านเกณฑ์ (มี Multi-Pass กู้คืนหมากดำในเงา/แสงสะท้อน)"""
    pts = []
    for name, mask in build_masks(frame).items():
        for u, v_px in _extract_piece_centers(mask, COLORS[name]["circ"]):
            pts.append((u, v_px, name))

    # Multi-pass fallback สำหรับหมากสีดำ:
    # 1) Core-dark pass (V<=115, S<=130): แยกหมากดำออกจากเงาเชื่อม (V=120..150) หรือช่องสีฟ้าที่อยู่ในเงามืด
    # 2) Glare-tolerant pass (V<=175, S<=100): จับหมากดำในช่องที่โดนไฟส่องสะท้อนสว่างจ้า
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    green_raw = cv2.inRange(hsv, GREEN_LOW, GREEN_HIGH)
    k_open = np.ones((5, 5), np.uint8)
    k_close7 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    for low, high, do_close, min_c in (
        ((0, 0, 0), (179, 130, 115), False, 0.45),
        ((0, 0, 0), (179, 100, 175), True, 0.55),
    ):
        m = cv2.inRange(hsv, low, high)
        touching_green = cv2.bitwise_and(green_raw, cv2.dilate(m, k_open))
        if np.any(touching_green):
            m = cv2.bitwise_or(m, touching_green)
        if do_close:
            m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, k_close7)
        m = apply_roi(cv2.morphologyEx(m, cv2.MORPH_OPEN, k_open))
        for u, v_px in _extract_piece_centers(m, min_c):
            if not any(np.hypot(u - pu, v_px - pv) <= 28.0 for (pu, pv, _) in pts):
                pts.append((u, v_px, "black"))
    return pts


def find_green_dots(frame):
    """คืนรายการ (u, v) ของจุดสีเขียวกลม ๆ ที่ขนาดพอดี (จุดกลางตัวฮอส / จุดที่เก็บฮอส)"""
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    m = cv2.inRange(hsv, GREEN_LOW, GREEN_HIGH)
    m = apply_roi(cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8)))
    cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    dots = []
    for c in cnts:
        area = cv2.contourArea(c)
        hull = cv2.convexHull(c)
        h_area = cv2.contourArea(hull)
        if h_area < 40 or area > GREEN_MAX_AREA:
            continue
        per = cv2.arcLength(c, True)
        circ = (4 * np.pi * area / (per * per)) if per > 0 else 0.0
        h_per = cv2.arcLength(hull, True)
        h_circ = (4 * np.pi * h_area / (h_per * h_per)) if h_per > 0 else 0.0
        if area >= GREEN_MIN_AREA and circ >= GREEN_MIN_CIRC:
            M = cv2.moments(c)
        elif h_area >= GREEN_MIN_AREA and area >= 0.5 * h_area and h_circ >= GREEN_MIN_CIRC:
            M = cv2.moments(hull)
        else:
            continue
        if M["m00"] > 0:
            dots.append((M["m10"] / M["m00"], M["m01"] / M["m00"]))
    return dots


def is_king_at(u, v, dots):
    return any(np.hypot(u - du, v - dv) <= KING_DOT_DIST for (du, dv) in dots)


# ==============================
# CALIBRATE
# ==============================
def calibrate():
    """
    1) วางสติกเกอร์ 6-9 จุดบนโต๊ะ กระจายทั่วพื้นที่ทำงาน (ที่ระนาบเดียวกับวัตถุ)
    2) คลิกแต่ละจุดในภาพ แล้วกด 's'
    3) สคริปต์จะบอกให้ย้ายปลายหุ่นไปแตะจุดนั้นทีละจุด (ใช้ปุ่ม drag/jog) แล้วกด Enter
    """
    cap = open_camera()
    ok, frame = read_frame(cap)
    cap.release()
    if not ok:
        raise RuntimeError("Cannot read frame")

    clicks = []
    view = frame.copy()

    def on_mouse(event, x, y, *_):
        if event == cv2.EVENT_LBUTTONDOWN:
            clicks.append((x, y))
            cv2.circle(view, (x, y), 6, (0, 0, 255), -1)
            cv2.putText(view, str(len(clicks)), (x + 8, y - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    cv2.namedWindow("calibrate")
    cv2.setMouseCallback("calibrate", on_mouse)
    print("คลิกจุดอ้างอิงตามลำดับ แล้วกด 's' (ต้องมีอย่างน้อย 4 จุด, 'q' ยกเลิก)")
    while True:
        cv2.imshow("calibrate", view)
        k = cv2.waitKey(30) & 0xFF
        if k == ord("q"):
            return
        if k == ord("s") and len(clicks) >= 4:
            break
    cv2.destroyAllWindows()

    print(f"กำลังเชื่อมต่อหุ่น {ROBOT_IP} ...")
    robot = MG400(ROBOT_IP)
    try:
        robot.clear_error_soft()
        robot_pts = []
        for i, _ in enumerate(clicks, 1):
            input(f"ย้ายปลายหุ่นไปแตะจุดที่ {i} แล้วกด Enter ...")
            x, y, z, _r = robot.pose()
            robot_pts.append((x, y, z))
            print(f"  จุด {i}: x={x:.1f} y={y:.1f} z={z:.1f}")
    finally:
        robot.close()

    px = np.array(clicks, dtype=np.float32)
    rb = np.array(robot_pts, dtype=np.float32)
    H, _ = cv2.findHomography(px, rb[:, :2], cv2.RANSAC, 5.0)

    # ตรวจความคลาดเคลื่อน
    proj = cv2.perspectiveTransform(px.reshape(-1, 1, 2), H).reshape(-1, 2)
    err = np.linalg.norm(proj - rb[:, :2], axis=1)
    print(f"\nReprojection error: mean={err.mean():.2f} mm, max={err.max():.2f} mm")
    if err.max() > 5:
        print("คำเตือน: error สูง ลองสอบเทียบใหม่ / เพิ่มจำนวนจุด")

    np.savez(CALIB_FILE, H=H, z_plane=float(rb[:, 2].mean()), rot180=ROTATE_180)
    print(f"บันทึกแล้วที่: {CALIB_FILE}")
    print(f"z ระนาบ = {rb[:, 2].mean():.1f} mm")
    mn, mx = rb[:, :2].min(axis=0), rb[:, :2].max(axis=0)
    print("\nจุดสอบเทียบครอบคลุม:")
    print(f"  X {mn[0]:.1f} ถึง {mx[0]:.1f}   Y {mn[1]:.1f} ถึง {mx[1]:.1f}")
    print("ค่าแนะนำสำหรับ CONFIG (เผื่อ 10 mm ตรวจเช็กว่าหุ่นเอื้อมถึงและปลอดภัยก่อนใช้):")
    print(f"  X_RANGE = ({mn[0] - 10:.0f}, {mx[0] + 10:.0f})")
    print(f"  Y_RANGE = ({mn[1] - 10:.0f}, {mx[1] + 10:.0f})")


# ==============================
# SET CLEAR POSITION (จุดหลีกกล้อง)
# ==============================
def setclear():
    """
    ย้ายปลายหุ่นไปจุดที่แขนไม่บังกล้อง (และไม่ชนอะไร) แล้วกด Enter เพื่อบันทึก
    แนะนำให้อยู่สูงกว่าความสูง hover ของวัตถุ เพื่อไม่ให้ชนวัตถุตอนถอย
    """
    robot = MG400(ROBOT_IP)
    try:
        robot.clear_error_soft()
        input("ย้ายปลายหุ่นไปจุดหลีกกล้อง แล้วกด Enter ...")
        x, y, z, r = robot.pose()
    finally:
        robot.close()
    np.savez(CLEAR_FILE, pos=np.array([x, y, z, TOOL_R]))
    print(f"บันทึกจุดหลีกกล้อง: x={x:.1f} y={y:.1f} z={z:.1f} r={r:.1f}")
    print(f"ไฟล์: {CLEAR_FILE}")


# ==============================
# SET DISCARD POSITION (จุดทิ้งหมาก)
# ==============================
def setdiscard():
    """
    ใช้ปุ่ม drag ลากหัวดูดไปเหนือจุดที่ต้องการทิ้งหมาก (เช่น ถาดหรือกล่องข้างกระดาน)
    ที่ความสูงที่ปล่อยแล้วหมากตกลงได้อย่างปลอดภัย แล้วกด Enter เพื่อบันทึก (x, y, z)
    หมากที่ถูกกินจะถูกนำมาปล่อยที่จุดนี้ ด้วยคำสั่ง  x <ช่อง>  ในโหมด play
    """
    robot = MG400(ROBOT_IP)
    try:
        robot.clear_error_soft()
        input("ลากหัวดูดไปเหนือจุดทิ้งหมาก (ที่ความสูงที่จะปล่อย) แล้วกด Enter ...")
        x, y, z, r = robot.pose()
    finally:
        robot.close()
    if not (Z_RANGE[0] <= z <= Z_RANGE[1]):
        print(f"คำเตือน: z={z:.1f} อยู่นอก Z_RANGE {Z_RANGE} ไม่บันทึก")
        return
    if in_range(x, y):
        print("คำเตือน: จุดนี้อยู่ในพื้นที่ทำงาน (X_RANGE/Y_RANGE) อาจตรงกับช่องกระดานหรือบังกล้อง "
              "ถ้าไม่ได้ตั้งใจ ให้เลือกจุดนอกกระดาน")
    np.savez(DISCARD_FILE, pos=np.array([x, y, z, TOOL_R]))
    print(f"บันทึกจุดทิ้งหมาก: x={x:.1f} y={y:.1f} z={z:.1f} r={r:.1f}")
    print(f"ไฟล์: {DISCARD_FILE}")


def load_discard():
    """คืนจุดทิ้งหมาก [x, y, z, r] หรือ None ถ้ายังไม่ได้ตั้ง"""
    if os.path.exists(DISCARD_FILE):
        return [float(v) for v in np.load(DISCARD_FILE)["pos"]]
    return None


# ==============================
# PUMP TEST / POSE
# ==============================
def pumptest():
    """ทดสอบปั๊มโดยไม่ขยับแขน: ดูด 3 วินาที แล้วปล่อย (เป่าลม)"""
    robot = MG400(ROBOT_IP)
    try:
        robot.enable()
        input(f"จะเปิดดูด (DO{VAC_DO}) 3 วินาที แล้วเป่าปล่อย (DO{BLOW_DO}) กด Enter เพื่อเริ่ม ...")
        robot.vacuum_on()
        time.sleep(3)
        robot.release()
        print("ปล่อยแล้ว")
    finally:
        try:
            robot.pump_off()
            print("ปิดปั๊มทั้งหมดแล้ว")
        finally:
            robot.close()


def show_pose():
    """อ่านตำแหน่งปัจจุบัน ใช้วัดค่า PICK_Z (เลื่อนหัวดูดไปแตะผิวบนวัตถุ แล้วอ่านค่า z)"""
    robot = MG400(ROBOT_IP)
    try:
        x, y, z, r = robot.pose()
        print(f"x={x:.1f} y={y:.1f} z={z:.1f} r={r:.1f}")
    finally:
        robot.close()


# ==============================
# RUN
# ==============================
def in_range(x, y):
    return X_RANGE[0] <= x <= X_RANGE[1] and Y_RANGE[0] <= y <= Y_RANGE[1]


def margin_outside(x, y):
    """ระยะ (mm) ที่อยู่นอกพื้นที่ปลอดภัย (0 = อยู่ข้างใน)"""
    dx = max(X_RANGE[0] - x, 0.0, x - X_RANGE[1])
    dy = max(Y_RANGE[0] - y, 0.0, y - Y_RANGE[1])
    return float(np.hypot(dx, dy))


def annotate(frame, objs):
    """
    วาดหมายเลขบนภาพ และรายการพิกัดเป็นแผงข้อความมุมซ้ายบน (ไม่ให้ตัวหนังสือทับกัน)
    คืนรายการ (x, y, u, v) ของหมากที่สั่งได้ เรียงตามหมายเลข (x,y = พิกัดหุ่น, u,v = พิกเซล)
    """
    numbered = []
    lines = []
    for (x, y, u, v, name) in objs:
        pt = (int(u), int(v))
        if in_range(x, y) and len(numbered) < MAX_OBJECTS:
            numbered.append((x, y, u, v))
            n = len(numbered)
            cv2.circle(frame, pt, 9, (255, 255, 255), 2)  # ขอบขาวให้เห็นชัดบนหมากสีเข้ม
            cv2.circle(frame, pt, 7, DRAW_COLOR[name], -1)
            cv2.putText(frame, f"{n}{TAG[name]}", (pt[0] + 12, pt[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 4)
            cv2.putText(frame, f"{n}{TAG[name]}", (pt[0] + 12, pt[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, DRAW_COLOR[name], 2)
            lines.append((f"{n}{TAG[name]}  X={x:.0f} Y={y:.0f}", DRAW_COLOR[name]))
        elif margin_outside(x, y) <= SHOW_OUT_MARGIN:
            cv2.circle(frame, pt, 8, (0, 0, 255), 2)
            cv2.putText(frame, "OUT", (pt[0] + 12, pt[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            lines.append((f"OUT{TAG[name]} X={x:.0f} Y={y:.0f} (+{margin_outside(x, y):.0f}mm)", (0, 0, 255)))
        # ไกลจากพื้นที่ทำงานมาก (มือ/ตัวหุ่น/อื่น ๆ) ไม่วาดอะไร

    lines = lines[:24]
    if lines:
        panel_h = 14 + 22 * len(lines)
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 40), (300, 40 + panel_h), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)
        for i, (text, color) in enumerate(lines):
            cv2.putText(frame, text, (8, 62 + 22 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
    return numbered


def confirm_target(cap, to_robot, tx, ty):
    """อ่านภาพใหม่ 5 เฟรม ยืนยันว่าวัตถุที่เลือกยังอยู่ที่เดิมและนิ่ง"""
    samples = []
    for _ in range(5):
        ok, f = read_frame(cap)
        if not ok:
            return None
        cands = [to_robot(p[0], p[1]) for p in find_objects(f)]
        near = [c for c in cands if np.hypot(c[0] - tx, c[1] - ty) <= 5.0]
        if not near:
            return None
        samples.append(min(near, key=lambda c: np.hypot(c[0] - tx, c[1] - ty)))
    arr = np.array(samples)
    if np.ptp(arr, axis=0).max() > 3.0:
        return None
    return arr.mean(axis=0)


def pick_sequence(robot, x, y, z_hover, clear_pos):
    """
    เหนือวัตถุ -> ลงแตะ -> เปิดดูด -> ยกขึ้น -> ค้าง -> ถอยไปจุดหลีกกล้อง -> ปล่อยวัตถุ
    คืน True ถ้ายังถือวัตถุอยู่ (ปกติจะเป็น False เพราะปล่อยที่จุดหลีกกล้อง)
    """
    def go(px, py, pz, pr=0.0):
        try:
            robot.m(f"MovL({px:.1f},{py:.1f},{pz:.1f},{TOOL_R:.1f})")
            robot.m("Sync()")
        except Exception as e:
            info = robot.diagnose() if hasattr(robot, "diagnose") else ""
            print(f"!! หุ่นไปต่อไม่ได้ที่ MovL({px:.1f},{py:.1f},{pz:.1f}): {e!r}\n!! สถานะ: {info}")
            raise RuntimeError(f"หุ่นหยุดที่ MovL({px:.1f},{py:.1f},{pz:.1f}) : {e} | {info}") from e

    holding = False
    try:
        go(x, y, z_hover)
        if USE_PUMP:
            go(x, y, PICK_Z)
            robot.vacuum_on()
            holding = True
            time.sleep(PUMP_SETTLE)
            go(x, y, z_hover)
        print(f"ค้างไว้ {HOLD_SECONDS:.1f} วินาที ...")
        time.sleep(HOLD_SECONDS)
        cx, cy, cz, cr = clear_pos
        print("ถอยไปจุดหลีกกล้อง")
        go(cx, cy, cz, cr)
        if USE_PUMP:
            print("ปล่อยวัตถุที่จุดหลีกกล้อง")
            robot.release()
            holding = False
    except Exception:
        # เกิดปัญหากลางทาง: ปิดปั๊มเพื่อความปลอดภัย แล้วส่งต่อ error
        if USE_PUMP:
            try:
                robot.pump_off()
            except Exception:
                pass
        raise
    return holding


# ==============================
# BOARD (กระดาน 8x8 ช่อง 1-64)
# ==============================
def square_rc(n):
    """เลขช่อง 1-64 -> (row, col) โดย row 0 = แถวล่างสุด, col 0 = ซ้ายสุด"""
    return (n - 1) // BOARD_N, (n - 1) % BOARD_N


def is_dark(n):
    """ช่องสีเข้มที่ใช้เดินหมาก (ช่อง 1,3,5,7,10,12,14,16,...) = ช่องสีฟ้าบนกระดานจริง"""
    r, c = square_rc(n)
    return (r + c) % 2 == 0


def load_board():
    if os.path.exists(BOARD_FILE):
        return np.load(BOARD_FILE)["A"]
    print("คำเตือน: ยังไม่มี board.npz ใช้ค่าเริ่มต้น (แนะนำให้รันโหมด boardcal)")
    return np.array(BOARD_AFFINE_DEFAULT, dtype=float)


def square_xy(A, n):
    r, c = square_rc(n)
    return A @ np.array([1.0, c, r])


def xy_to_square(A, x, y):
    """พิกัดหุ่น -> เลขช่อง (None ถ้าไม่ได้อยู่ใกล้กึ่งกลางช่องใดช่องหนึ่ง)"""
    cr = np.linalg.solve(A[:, 1:], np.array([x, y]) - A[:, 0])  # (col, row)
    c, r = cr
    ci, ri = int(round(c)), int(round(r))
    if not (0 <= ci < BOARD_N and 0 <= ri < BOARD_N):
        return None
    if abs(c - ci) > SQUARE_TOL or abs(r - ri) > SQUARE_TOL:
        return None
    return ri * BOARD_N + ci + 1


def occupancy(A, objs):
    """objs = [(x, y, u, v, name)] -> {เลขช่อง: 'black'/'orange'}"""
    occ = {}
    for (x, y, _u, _v, name, _k) in objs:
        n = xy_to_square(A, x, y)
        if n is not None:
            occ[n] = name
    return occ


def kings_on_board(A, objs):
    """objs -> เซ็ตเลขช่องที่เป็นตัวฮอส"""
    ks = set()
    for (x, y, _u, _v, _name, king) in objs:
        n = xy_to_square(A, x, y)
        if n is not None and king:
            ks.add(n)
    return ks


def _layout_group(group, first_row, second_row, v_desc):
    """จัดหมาก PIECES_PER_SIDE ตัวของสีเดียวลงช่องเริ่มเกม 2 แถว: แถวที่ v อยู่ไกลสุดในภาพ = first_row"""
    h = PIECES_PER_SIDE // 2
    ordered = sorted(group, key=lambda o: -o[3] if v_desc else o[3])
    out = []
    for part, row in ((ordered[:h], first_row), (ordered[h:], second_row)):
        cols = [c for c in range(BOARD_N) if (row + c) % 2 == 0]
        for k, piece in enumerate(sorted(part, key=lambda o: o[2])):
            out.append((piece, row, cols[k]))
    return out


def _layout_error(pairs):
    M = np.array([[1.0, c, r] for (_p, r, c) in pairs])
    XY = np.array([[p[0], p[1]] for (p, _r, _c) in pairs])
    sol, *_ = np.linalg.lstsq(M, XY, rcond=None)
    return float(np.linalg.norm(M @ sol - XY, axis=1).max())


def _best_subset(group, first_row, second_row, v_desc, label, max_err=8.0):
    """ถ้าเห็นหมากสีนี้เกิน (เช่น ฮอสสำรองที่ไม่เห็นจุดเขียว / วัตถุข้างกระดาน) เลือก PIECES_PER_SIDE ตัวที่เรียงเป็นกริดตรงที่สุด"""
    import itertools, math
    n = PIECES_PER_SIDE
    if len(group) == n:
        return _layout_group(group, first_row, second_row, v_desc)
    if len(group) < n:
        raise RuntimeError(f"เจอหมาก{label}แค่ {len(group)} ตัว ต้องการ {n} ตัว")
    if math.comb(len(group), n) > 200000:
        raise RuntimeError(f"เจอหมาก{label} {len(group)} ตัว มากเกินไป (ต้องการ {n}) ตรวจกรอบ setroi / HSV / สิ่งแปลกปลอมในภาพ")
    best = None
    for comb in itertools.combinations(group, n):
        pairs = _layout_group(comb, first_row, second_row, v_desc)
        err = _layout_error(pairs)
        if best is None or err < best[0]:
            best = (err, pairs)
    if best[0] > max_err:
        raise RuntimeError(f"เจอหมาก{label} {len(group)} ตัว (ต้องการ {n}) และหาชุดที่เรียงเป็นกริดตำแหน่งเริ่มเกมไม่ได้ "
                           f"(คลาดเคลื่อนต่ำสุด {best[0]:.1f} mm) ตรวจว่าวางหมากครบตามผัง")
    extra = len(group) - n
    print(f"   หมาก{label}เกิน {extra} ตัว (ไม่ใช่หมากบนกระดาน) ตัดออกอัตโนมัติ โดยเลือก {n} ตัวที่เรียงเป็นกริด (คลาดเคลื่อน {best[0]:.1f} mm)")
    return best[1]


def assign_start_squares(objs, use_range=True):
    """
    จับคู่หมากทั้งหมดในภาพกับช่องเริ่มเกม (ภาพต้องเห็นคนอยู่ล่าง-หุ่นอยู่บน เหมือนผัง)
    ใช้ตำแหน่งพิกเซล: ดำ = 2 แถวล่าง, ส้ม = 2 แถวบน
    ถ้าเห็นหมากสีใดเกินจำนวน จะเลือกชุดที่เรียงเป็นกริดตำแหน่งเริ่มเกมได้ตรงที่สุด
    คืนรายการ (หมาก, row, col)
    """
    ok_pos = (lambda o: in_range(o[0], o[1])) if use_range else (lambda o: True)
    black = [o for o in objs if o[4] == "black" and ok_pos(o)]
    orange = [o for o in objs if o[4] == "orange" and ok_pos(o)]
    if len(black) < PIECES_PER_SIDE or len(orange) < PIECES_PER_SIDE:
        raise RuntimeError(f"ต้องเห็นหมากดำ {PIECES_PER_SIDE} ตัว + ส้ม {PIECES_PER_SIDE} ตัวในตำแหน่งเริ่มเกม (เจอดำ {len(black)}, ส้ม {len(orange)})")
    # ดำ: แถว 0 อยู่ล่างสุดของภาพ (v มาก) / ส้ม: แถวบนสุด (v น้อย) = BOARD_N-1
    return (_best_subset(black, 0, 1, True, "ดำ") +
            _best_subset(orange, BOARD_N - 1, BOARD_N - 2, False, "ส้ม"))


def load_calib():
    if not os.path.exists(CALIB_FILE):
        raise RuntimeError("ไม่พบ calib.npz — ต้องรันโหมด calibrate (สอบเทียบกล้อง-หุ่น) ก่อนโหมดอื่น ๆ")
    data = np.load(CALIB_FILE)
    calib_rot = bool(data["rot180"]) if "rot180" in data.files else False
    if calib_rot != ROTATE_180:
        raise RuntimeError(
            f"ค่า ROTATE_180 ตอนสอบเทียบ ({calib_rot}) ไม่ตรงกับตอนนี้ ({ROTATE_180}) "
            "ให้ตั้งให้ตรงกัน หรือสอบเทียบ (calibrate) ใหม่")
    H, z_plane = data["H"], float(data["z_plane"])

    def to_robot(u, v):
        p = H @ np.array([u, v, 1.0])
        return p[0] / p[2], p[1] / p[2]

    return H, z_plane, to_robot


def detect_robot_objs(frame, to_robot):
    """คืน [(x, y, u, v, สี, king)] king=True ถ้ามีจุดเขียวอยู่กลางหมาก (จับคู่เฉพาะหมากที่อยู่ใกล้จุดเขียวที่สุด)"""
    objs = []
    dots = find_green_dots(frame)
    found = find_objects(frame)
    king_indices = set()
    for (du, dv) in dots:
        best_idx, best_dist = None, float("inf")
        for idx, (u, v_px, _name) in enumerate(found):
            d = float(np.hypot(u - du, v_px - dv))
            if d <= KING_DOT_DIST and d < best_dist:
                best_idx, best_dist = idx, d
        if best_idx is not None:
            king_indices.add(best_idx)
    for idx, (u, v_px, name) in enumerate(found):
        if not in_roi(u, v_px):  # วัตถุนอกกรอบตรวจจับ (setroi) ไม่นับเด็ดขาด
            continue
        x, y = to_robot(u, v_px)
        objs.append((x, y, u, v_px, name, idx in king_indices))
    return objs


def _boardcal_report(frame, objs):
    """พิมพ์สิ่งที่กล้องเห็นทั้งหมด + บันทึกภาพ boardcal_debug.png ให้ดูว่าตัวไหนถูก/ไม่ถูกจับ"""
    nb = sum(o[4] == "black" and not o[5] for o in objs)
    no = sum(o[4] == "orange" and not o[5] for o in objs)
    nk = sum(bool(o[5]) for o in objs)
    print(f"กล้องเห็นวัตถุ {len(objs)} ชิ้น: ดำ {nb}, ส้ม {no}, ฮอส(จุดเขียว) {nk}  (ต้องการ ดำ {PIECES_PER_SIDE} + ส้ม {PIECES_PER_SIDE} ไม่นับฮอส)")
    for o in sorted(objs, key=lambda o: (o[4], o[3], o[2])):
        flag = "" if in_range(o[0], o[1]) else "  (นอก X_RANGE/Y_RANGE — โหมดนี้ไม่กรอง)"
        print(f"   {o[4]:6s}{' K' if o[5] else '  '} พิกเซล=({o[2]:.0f},{o[3]:.0f}) หุ่น=({o[0]:.1f},{o[1]:.1f}){flag}")
    img = frame.copy()
    roi = get_roi()
    if roi is None:
        print("คำเตือน: ยังไม่ได้ตั้งกรอบตรวจจับ (setroi) — ตรวจทั้งภาพ จึงอาจเจอของนอกกระดานปน")
    else:
        print(f"กรอบตรวจจับ (setroi) = x {roi[0]}-{roi[2]}, y {roi[1]}-{roi[3]} (นับเฉพาะวัตถุในกรอบนี้)")
        cv2.rectangle(img, (roi[0], roi[1]), (roi[2], roi[3]), (0, 255, 0), 2)
    for o in objs:
        col = (0, 255, 255) if o[5] else ((40, 40, 40) if o[4] == "black" else (0, 140, 255))
        cv2.circle(img, (int(o[2]), int(o[3])), 22, (255, 255, 255), 3)
        cv2.circle(img, (int(o[2]), int(o[3])), 18, col, 3)
    path = os.path.join(BASE_DIR, "boardcal_debug.png")
    try:
        cv2.imwrite(path, img)
        print(f"บันทึกภาพที่ระบบจับได้ไว้ที่: {path} (วงกลม = ที่จับได้; ถ้าหมากไม่มีวง แปลว่าสี/แสง/กรอบ setroi ทำให้จับไม่เจอ)")
    except Exception as e:
        print("บันทึกภาพ debug ไม่ได้:", e)


def boardcal():
    """
    วางหมากตามตำแหน่งเริ่มเกม (ดำ 8 ตัวล่าง ส้ม 8 ตัวบน) แล้วรันโหมดนี้
    สคริปต์จะคำนวณว่าช่อง 1-64 อยู่ที่พิกัดหุ่นเท่าไร แล้วบันทึกลง board.npz
    """
    _H, _z, to_robot = load_calib()
    cap = open_camera()
    try:
        objs = []
        for _ in range(5):  # ใช้เฟรมสุดท้าย (ผ่านการหน่วงให้ภาพนิ่ง)
            ok, frame = read_frame(cap)
            if not ok:
                raise RuntimeError("Cannot read frame")
            objs = detect_robot_objs(frame, to_robot)
    finally:
        cap.release()
    _boardcal_report(frame, objs)
    # ตอนสอบเทียบกระดาน ยังไม่รู้ว่ากระดานอยู่ตรงไหน จึงไม่กรองด้วย X_RANGE/Y_RANGE และตัดตัวฮอส (มีจุดเขียว เช่นที่เก็บข้างกระดาน) ออก
    plain = [o for o in objs if not o[5]]
    pairs = assign_start_squares(plain, use_range=False)
    M = np.array([[1.0, c, r] for (_p, r, c) in pairs])
    XY = np.array([[p[0], p[1]] for (p, _r, _c) in pairs])
    sol, *_ = np.linalg.lstsq(M, XY, rcond=None)
    A = sol.T
    err = np.linalg.norm(M @ sol - XY, axis=1)
    print(f"ความคลาดเคลื่อนเทียบกับกริดเรียบ: mean={err.mean():.1f} mm, max={err.max():.1f} mm")
    if err.max() > 6:
        print("คำเตือน: คลาดเคลื่อนสูง ตรวจว่าหมากอยู่กลางช่อง ไม่มีหมากเกิน/ขาด และสอบเทียบกล้องดีหรือยัง")
    np.savez(BOARD_FILE, A=A)
    print(f"บันทึกแล้วที่: {BOARD_FILE}")
    for n in (1, BOARD_N, BOARD_N * (BOARD_N - 1) + 1, BOARD_N * BOARD_N):
        x, y = square_xy(A, n)
        print(f"  ช่อง {n:2d}: X={x:.1f} Y={y:.1f}")


def draw_board(frame, A, Hinv, occ, kings=()):
    """วาดเลขช่องครบทุกช่องบนภาพ (ช่องเดินได้เป็นตัวหนา สีตามหมากที่อยู่ในช่อง)"""
    for n in range(1, BOARD_N * BOARD_N + 1):
        x, y = square_xy(A, n)
        p = Hinv @ np.array([x, y, 1.0])
        u, v = p[0] / p[2], p[1] / p[2]
        if not (0 <= u < frame.shape[1] and 0 <= v < frame.shape[0]):
            continue
        playable = is_dark(n)
        color = (200, 200, 200) if playable else (140, 140, 140)  # ช่องขาว (เดินไม่ได้) สีเทาเข้มกว่า
        if n in occ:
            color = DRAW_COLOR[occ[n]]
        label = f"{n}K" if n in kings else str(n)
        pos = (int(u) - 32, int(v) - 20)
        cv2.putText(frame, label, pos, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 4 if playable else 3)
        cv2.putText(frame, label, pos, cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2 if playable else 1)
        if n in kings:
            cv2.circle(frame, (int(u), int(v)), 26, (0, 255, 255), 2)


def stable_occupancy(cap, to_robot, A, frames=None):
    """
    อ่านเฟรมต่อเนื่อง จนเห็นสถานะกระดานเหมือนกันติดกัน STABLE_STREAK เฟรม (รอได้สูงสุด STABLE_MAX_FRAMES)
    คืน (occ, kings) หรือ None ถ้าไม่นิ่ง   occ = {ช่อง: สี}, kings = เซ็ตช่องที่เป็นฮอส
    """
    need = frames or STABLE_STREAK
    snap, streak = None, 0
    for _ in range(STABLE_MAX_FRAMES):
        ok, f = read_frame(cap)
        if not ok:
            return None
        objs = detect_robot_objs(f, to_robot)
        cur = (occupancy(A, objs), kings_on_board(A, objs))
        streak = streak + 1 if cur == snap else 1
        snap = cur
        if streak >= need:
            return snap
    return None


def stable_objs(cap, to_robot, frames=None):
    """อ่านเฟรมต่อเนื่องจนจำนวนวัตถุเท่ากันติดกันหลายเฟรม คืนรายการวัตถุ ไม่งั้นคืน None"""
    need = frames or STABLE_STREAK
    n_prev, streak, objs = None, 0, None
    for _ in range(STABLE_MAX_FRAMES):
        ok, f = read_frame(cap)
        if not ok:
            return None
        objs = detect_robot_objs(f, to_robot)
        streak = streak + 1 if len(objs) == n_prev else 1
        n_prev = len(objs)
        if streak >= need:
            return objs
    return None


# ==============================
# KING STOCK (จุดเก็บตัวฮอส)
# ==============================
def stock_reach_ok(x, y):
    r = float(np.hypot(x, y))
    return STOCK_REACH[0] <= r <= STOCK_REACH[1]


def setstock():
    """
    วางตัวฮอสบนจุดสีเขียวทั้ง 8 จุด (ส้ม 4 ตัวด้านของหุ่น, ดำ 4 ตัวด้านของคน) แล้วรันโหมดนี้
    สคริปต์จะหาตำแหน่งจุดเก็บจากกล้อง แล้วบันทึกลง stock_pos.npz (ทั้งตำแหน่งในภาพและพิกัดหุ่น)
    พิกัดหุ่นที่ได้ตรงนี้เป็นค่าประมาณ (จุดเก็บอยู่นอกบริเวณที่ calibrate) ถ้าหยิบไม่แม่น ให้รัน teachstock ต่อ
    """
    H, _z, to_robot = load_calib()
    Hinv = np.linalg.inv(H)
    A = load_board()
    cap = open_camera()
    try:
        for _ in range(8):
            cap.read()
        ok, frame = read_frame(cap)
        if not ok:
            raise RuntimeError("Cannot read frame")
        dots = find_green_dots(frame)
    finally:
        cap.release()

    def col_u(c):
        pt = A @ np.array([1.0, c, (BOARD_N - 1) / 2])
        q = Hinv @ np.array([pt[0], pt[1], 1.0])
        return q[0] / q[2]
    u_lo, u_hi = sorted((col_u(-0.5), col_u(BOARD_N - 0.5)))
    left = sorted([d for d in dots if d[0] < u_lo], key=lambda d: d[1])
    right = sorted([d for d in dots if d[0] > u_hi], key=lambda d: d[1])
    print(f"เจอจุดเขียวนอกกระดาน: ซ้าย {len(left)} จุด, ขวา {len(right)} จุด (ต้องการ 4 + 4)")
    if len(left) != 4 or len(right) != 4:
        raise RuntimeError("จุดเขียวไม่ครบ 4+4 ตรวจกรอบ setroi ต้องครอบที่เก็บทั้ง 2 ข้าง / แสง / GREEN_LOW-GREEN_HIGH")
    orange_pix, black_pix = (left, right) if STOCK_ORANGE_SIDE == "left" else (right, left)
    save = {}
    for key, pix in (("orange", orange_pix), ("black", black_pix)):
        xy = np.array([to_robot(u, v) for (u, v) in pix])
        save[key + "_xy"] = xy
        save[key + "_vxy"] = xy.copy()      # ค่าที่กล้องคำนวณ (ใช้หาค่าชดเชยเมื่อสอนด้วย teachstock)
        save[key + "_uv"] = np.array(pix)
        print(f"ที่เก็บฮอส {key} (บนลงล่างตามภาพ):")
        for i, (x, y) in enumerate(xy, 1):
            flag = "" if stock_reach_ok(x, y) else "   <-- แขนอาจเอื้อมไม่ถึง!"
            print(f"  #{i}: X={x:.1f} Y={y:.1f}{flag}")
    np.savez(STOCK_FILE, **save)
    print(f"บันทึกแล้วที่: {STOCK_FILE}")
    print("แนะนำ: รันโหมด teachstock เพื่อสอนตำแหน่งหยิบด้วยมือให้แม่นขึ้น")


def teachstock():
    """
    สอนตำแหน่งหยิบฮอสทั้ง 8 จุดด้วยมือ (แม่นกว่าใช้กล้อง เพราะจุดเก็บอยู่นอกบริเวณ calibrate)
    ต้องรัน setstock ก่อน 1 ครั้ง แล้วรันโหมดนี้: ลากหัวดูดไปกลางตัวฮอสแต่ละตัว (ให้หัวดูดแตะผิวฮอส) แล้วกด Enter
    ลำดับ: ส้ม #1-#4 (บนลงล่างตามภาพ) แล้วดำ #1-#4  ถ้าเว้นช่องไหน พิมพ์ s แล้ว Enter เพื่อข้าม (ใช้ค่าเดิม)
    """
    if not os.path.exists(STOCK_FILE):
        raise RuntimeError("ยังไม่มี stock_pos.npz ให้รันโหมด setstock ก่อน")
    d = dict(np.load(STOCK_FILE))
    if "orange_uv" not in d:
        raise RuntimeError("stock_pos.npz เป็นรุ่นเก่า ให้รันโหมด setstock ใหม่ก่อน")
    robot = MG400(ROBOT_IP)
    try:
        robot.clear_error_soft()
        for key in ("orange", "black"):
            for i in range(len(d[key + "_xy"])):
                old = d[key + "_xy"][i]
                ans = input(f"[{key} #{i + 1}] ลากหัวดูดไปกลางฮอส (เดิม X={old[0]:.1f} Y={old[1]:.1f}) แล้วกด Enter (s = ข้าม) ...").strip().lower()
                if ans == "s":
                    continue
                x, y, z, _r = robot.pose()
                d[key + "_xy"][i] = (x, y)
                print(f"   บันทึก X={x:.1f} Y={y:.1f}  (Z={z:.1f} ไม่ใช้ ตอนหยิบจะใช้ PICK_Z)")
    finally:
        robot.close()
    np.savez(STOCK_FILE, **d)
    print(f"บันทึกแล้วที่: {STOCK_FILE}")


STOCK_OFFSET = {"orange": (0.0, 0.0), "black": (0.0, 0.0)}


def load_stock():
    """คืน {'orange': [(x, y, u, v) x4], 'black': [...]} หรือ None ถ้ายังไม่ได้ตั้ง (x,y = พิกัดหุ่น, u,v = พิกเซล)"""
    if not os.path.exists(STOCK_FILE):
        return None
    d = np.load(STOCK_FILE)
    if "orange_uv" not in d.files:
        print("คำเตือน: stock_pos.npz เป็นรุ่นเก่า ให้รันโหมด setstock ใหม่")
        return None
    for k in ("orange", "black"):      # ค่าชดเชย = ที่สอนด้วยมือ - ที่กล้องคำนวณ (แก้ความคลาดของกล้องแถวที่เก็บ)
        if k + "_vxy" in d.files:
            STOCK_OFFSET[k] = tuple(float(v) for v in (d[k + "_xy"] - d[k + "_vxy"]).mean(axis=0))
        else:
            STOCK_OFFSET[k] = (0.0, 0.0)
    return {k: [(float(a), float(b), float(c), float(e)) for (a, b), (c, e) in zip(d[k + "_xy"], d[k + "_uv"])]
            for k in ("orange", "black")}


def _stock_has_king(objs, color, su, sv):
    return any(k and n == color and np.hypot(u - su, v - sv) <= STOCK_MATCH_PX
               for (_x, _y, u, v, n, k) in objs)


def pick_stock_king(objs, stock, color):
    """หาจุดเก็บของสี color ที่ยังมีฮอสวางอยู่ (ตามกล้อง) คืน (x, y) พิกัดหุ่น หรือ None"""
    if STOCK_DYNAMIC:
        # ใช้ตำแหน่งที่กล้องเห็นจริง + ค่าชดเชยจาก teachstock (ถ้ามี) เลือกตัวที่อยู่ใกล้จุดเก็บมากสุดก่อน
        ox, oy = STOCK_OFFSET[color]
        best = None
        for (x, y, u, v, n, k) in objs:
            if not (k and n == color):
                continue
            d = min(np.hypot(u - su, v - sv) for (_sx, _sy, su, sv) in stock[color])
            if d <= STOCK_SEARCH_PX and (best is None or d < best[0]):
                best = (d, x + ox, y + oy)
        return None if best is None else (best[1], best[2])
    for (sx, sy, su, sv) in stock[color]:
        if _stock_has_king(objs, color, su, sv):
            return (sx, sy)
    return None


def draw_stock(frame, stock, objs):
    """วาดจุดเก็บฮอสบนภาพ: เหลือง = มีฮอสอยู่, แดง = ว่าง"""
    for color, pts in stock.items():
        for (_sx, _sy, su, sv) in pts:
            has = _stock_has_king(objs, color, su, sv)
            cv2.circle(frame, (int(su), int(sv)), 34, (0, 255, 255) if has else (0, 0, 255), 2)


def move_piece(robot, src, dst, clear_pos, z_hover, discard=False, discard_pos=None, to_clear=True, src_is_king=False, via=()):
    """
    หยิบหมากที่ src -> ไปวางที่ dst
    ถ้า discard=True: ไปปล่อยที่ discard_pos (จุดทิ้งที่ตั้งด้วย setdiscard) ถ้าไม่มี ใช้จุดหลีกกล้องแทน
    จบด้วยถอยไปจุดหลีกกล้องไม่ให้บังกล้อง (to_clear=True)
    to_clear=False: ไม่ไปจุดหลีกกล้อง ยกแขนค้างไว้ที่ความสูงปลอดภัยเหนือจุดที่เพิ่งวาง/ทิ้ง เพื่อไปหยิบชิ้นถัดไปต่อเลย (ใช้กับขั้นที่ไม่ใช่ขั้นสุดท้ายของตา)
    src, dst = (x, y) พิกัดหุ่น, discard_pos = [x, y, z, r]
    via = รายการ (x, y) จุดลงจอดกลางทางของการกินหลายตัว: ถือหมากผ่านแต่ละจุด ย่อแขนลง VIA_DIP_MM แล้วยกขึ้น ก่อนไปจุดถัดไป
    """
    # ช่องที่ใกล้ฐานเกิน MIN_REACH_X: หยิบที่ X ต่ำสุดที่ไปได้ แล้วเลื่อนจุดวางตามส่วนต่างที่หมากเยื้องใต้หัวดูด
    off_x = 0.0
    if MIN_REACH_X is not None:
        if src[0] < MIN_REACH_X:
            off_x = MIN_REACH_X - src[0]
            print(f"   ช่องต้นทาง X={src[0]:.1f} ใกล้ฐานเกินไป -> หยิบที่ X={MIN_REACH_X:.1f} (หมากเยื้องใต้หัว {off_x:.1f} mm) แล้วชดเชยตอนวาง")
            src = (MIN_REACH_X, src[1])
        if dst is not None:
            want = dst[0] + off_x
            if want < MIN_REACH_X:
                print(f"   ปลายทาง X={dst[0]:.1f} ใกล้ฐานเกินไป -> วางที่ X={MIN_REACH_X:.1f} (หมากจะเยื้อง {MIN_REACH_X - want:.1f} mm)")
                want = MIN_REACH_X
            dst = (want, dst[1])
    via = [(max(MIN_REACH_X, vx + off_x) if MIN_REACH_X is not None else vx + off_x, vy) for (vx, vy) in (via or ())]
    def go(px, py, pz, pr=0.0, joint=False):
        if not joint and MOVJ_APPROACH_MM > 0:
            try:
                cx0, cy0, cz0, _r0 = robot.pose()
            except Exception:
                cx0 = None
            if cx0 is not None:
                d = math.sqrt((px - cx0) ** 2 + (py - cy0) ** 2 + (pz - cz0) ** 2)
                if d > MOVJ_APPROACH_MM + 1.0:
                    k = MOVJ_APPROACH_MM / d   # จุดที่ห่างเป้า MOVJ_APPROACH_MM บนเส้นตรงจากตำแหน่งปัจจุบัน
                    _go_one(px - (px - cx0) * k, py - (py - cy0) * k, pz - (pz - cz0) * k, True)
        _go_one(px, py, pz, joint)

    def _go_one(px, py, pz, joint):
        cmd = "MovJ" if joint else "MovL"

        def arrived():
            try:
                x, y, z, _r = robot.pose()
                return max(abs(x - px), abs(y - py), abs(z - pz)) <= ARRIVE_TOL
            except Exception:
                return False

        try:
            try:
                if cmd == "MovL" and MOVL_FALLBACK_MOVJ and hasattr(robot, "set_motion_timeout"):
                    robot.set_motion_timeout(MOVL_WAIT)
                robot.m(f"{cmd}({px:.1f},{py:.1f},{pz:.1f},{TOOL_R:.1f})")
                robot.m("Sync()")
            except Exception as e1:
                if arrived():
                    print(f"!! {cmd} ไม่ตอบกลับ ({e1}) แต่หุ่นอยู่ที่เป้าหมายแล้ว -> ถือว่าสำเร็จ")
                    if hasattr(robot, "recover_motion"):
                        robot.recover_motion()   # เทคำตอบค้างทิ้ง กันปนกับคำสั่งถัดไป
                elif cmd != "MovL" or not MOVL_FALLBACK_MOVJ:
                    raise
                else:
                    print(f"!! MovL({px:.1f},{py:.1f},{pz:.1f}) ค้าง ({e1}) -> ล้างคิวแล้วลอง MovJ แทน")
                    if hasattr(robot, "recover_motion"):
                        robot.recover_motion()
                    cmd = "MovJ"
                    try:
                        robot.m(f"MovJ({px:.1f},{py:.1f},{pz:.1f},{TOOL_R:.1f})")
                        robot.m("Sync()")
                    except Exception:
                        if not arrived():
                            raise
                        print("!! MovJ ไม่ตอบกลับ แต่หุ่นอยู่ที่เป้าหมายแล้ว -> ถือว่าสำเร็จ")
                        if hasattr(robot, "recover_motion"):
                            robot.recover_motion()
            finally:
                if hasattr(robot, "set_motion_timeout"):
                    try:
                        robot.set_motion_timeout(30)
                    except Exception:
                        pass
        except Exception as e:
            info = robot.diagnose() if hasattr(robot, "diagnose") else ""
            print(f"!! หุ่นไปต่อไม่ได้ที่ {cmd}({px:.1f},{py:.1f},{pz:.1f}): {e!r}\n!! สถานะ: {info}")
            raise RuntimeError(f"หุ่นหยุดที่ {cmd}({px:.1f},{py:.1f},{pz:.1f}) : {e} | {info}") from e

    z_place = PICK_Z + PLACE_Z_OFFSET
    cx, cy, cz, cr = clear_pos
    try:
        pick_z = PICK_Z + (KING_PICK_Z_OFFSET if (src_is_king and not discard) else 0.0)
        go(src[0], src[1], z_hover)
        go(src[0], src[1], pick_z)
        robot.vacuum_on()
        time.sleep(PUMP_SETTLE)
        if discard:
            if discard_pos is None:
                go(src[0], src[1], z_hover)
                go(cx, cy, cz, cr)          # ยังไม่ได้ตั้งจุดทิ้ง: ปล่อยที่จุดหลีกกล้องเหมือนเดิม
                robot.release()
                return
            dx, dy, dz, dr = discard_pos
            # ยกสูงขึ้นอีก (ไม่เกินเพดาน Z_RANGE) แล้วเดินทางที่ความสูงนี้ ไม่ให้ชนที่ทิ้ง
            lift_z = min(Z_RANGE[1], max(z_hover + DISCARD_EXTRA_LIFT, dz) + DISCARD_TRAVEL_RAISE)
            go(src[0], src[1], lift_z)
            go(dx, dy, lift_z, dr)
            if dz < lift_z:
                go(dx, dy, dz, dr)
            robot.release()
            if dz < lift_z:
                go(dx, dy, lift_z, dr)      # ยกกลับก่อนค่อยออก ไม่ลากผ่านหมากที่เพิ่งปล่อย
            if to_clear:
                go(cx, cy, cz, cr)
            return
        if src_is_king:
            carry_z = min(Z_RANGE[1], max(KING_CARRY_Z, z_hover))
            if KING_LIFT_METHOD == "movl_movj":
                go(src[0], src[1], min(carry_z, pick_z + KING_LIFT_MOVL))   # ช่วงแรก MovL ตรงขึ้น
                go(src[0], src[1], carry_z, joint=True)                      # ที่เหลือ MovJ
            elif KING_LIFT_METHOD == "movj":
                go(src[0], src[1], carry_z, joint=True)
            else:
                try:
                    if KING_LIFT_FALLBACK_MOVJ and hasattr(robot, "set_motion_timeout"):
                        robot.set_motion_timeout(KING_LIFT_WAIT)
                    go(src[0], src[1], carry_z)                              # movl_high: MovL ทีเดียวขึ้นสูง
                except Exception as exc:
                    if not KING_LIFT_FALLBACK_MOVJ:
                        raise
                    print(f"!! ยกฮอสด้วย MovL ไม่สำเร็จ ({exc}) -> หยุด/ล้างคิว แล้วลองยกด้วย MovJ")
                    if hasattr(robot, "recover_motion"):
                        robot.recover_motion()
                    go(src[0], src[1], carry_z, joint=True)                  # ถ้า MovJ ก็ค้าง จะ raise ตามเดิม
                finally:
                    if hasattr(robot, "set_motion_timeout"):
                        try:
                            robot.set_motion_timeout(30)
                        except Exception:
                            pass
        else:
            carry_z = z_hover
            go(src[0], src[1], z_hover)
        for (vx, vy) in via:                                                  # กินหลายตัว: จอดที่จุดลงจอดกลางทางแล้วย่อลง-ยกขึ้น
            go(vx, vy, carry_z)
            go(vx, vy, max(Z_RANGE[0], z_place, carry_z - VIA_DIP_MM))   # ย่อลงไม่ต่ำกว่าความสูงวางหมาก (กันกดหมากลงกระดาน)
            go(vx, vy, carry_z)
        go(dst[0], dst[1], carry_z)
        go(dst[0], dst[1], z_place)                                          # ลงด้วย MovL
        robot.release()
        go(dst[0], dst[1], z_hover)
        if to_clear:
            print(f"ค้างไว้ {HOLD_SECONDS:.1f} วินาที ...")
            time.sleep(HOLD_SECONDS)
            go(cx, cy, cz, cr)
    except Exception:
        try:
            robot.pump_off()   # เกิดปัญหากลางทาง: ปิดปั๊ม (หมากที่ถืออยู่จะหล่น)
        except Exception:
            pass
        raise


def run():
    data = np.load(CALIB_FILE)
    H, z_plane = data["H"], float(data["z_plane"])
    z_target = z_plane + HOVER_ABOVE
    calib_rot = bool(data["rot180"]) if "rot180" in data.files else False
    if calib_rot != ROTATE_180:
        raise RuntimeError(
            f"ค่า ROTATE_180 ตอนสอบเทียบ ({calib_rot}) ไม่ตรงกับตอนนี้ ({ROTATE_180}) "
            "ให้ตั้งให้ตรงกัน หรือสอบเทียบ (calibrate) ใหม่")

    def to_robot(u, v):
        p = H @ np.array([u, v, 1.0])
        return p[0] / p[2], p[1] / p[2]

    if not (Z_RANGE[0] <= z_target <= Z_RANGE[1]):
        raise RuntimeError(f"Z เป้าหมาย {z_target:.1f} อยู่นอก Z_RANGE {Z_RANGE}")

    clear_pos = None
    if os.path.exists(CLEAR_FILE):
        clear_pos = [float(v) for v in np.load(CLEAR_FILE)["pos"]]
        print(f"จุดหลีกกล้อง: {clear_pos[:3]}")
        if clear_pos[2] < z_target:
            print("คำเตือน: จุดหลีกกล้องต่ำกว่าความสูง hover อาจชนวัตถุตอนถอย")
    elif ALLOW_MOVE:
        raise RuntimeError("ยังไม่ได้ตั้งจุดหลีกกล้อง ให้รันโหมด setclear ก่อน")

    if ALLOW_MOVE and USE_PUMP:
        if PICK_Z is None:
            raise RuntimeError("ยังไม่ได้ตั้ง PICK_Z (ใช้โหมด pose วัด z ตอนหัวดูดแตะผิวบนวัตถุ)")
        if not (Z_RANGE[0] <= PICK_Z <= z_target):
            raise RuntimeError(f"PICK_Z={PICK_Z} ต้องอยู่ระหว่าง {Z_RANGE[0]} ถึง {z_target:.1f}")

    robot = MG400(ROBOT_IP) if ALLOW_MOVE else None
    if robot:
        robot.enable()
        if USE_PUMP:
            robot.pump_off()  # เริ่มต้นด้วยปั๊มปิด

    cap = open_camera()
    holding = False
    show_mask = False
    click = {"pt": None}

    def on_mouse(event, mx, my, *_):
        if event == cv2.EVENT_LBUTTONDOWN:
            click["pt"] = (mx, my)

    cv2.namedWindow("run")
    cv2.setMouseCallback("run", on_mouse)
    print("กดเลข 1-9 หรือ คลิกที่หมาก เพื่อให้หุ่นหยิบหมากหมายเลขนั้นไปปล่อยที่จุดหลีกกล้อง, "
          "'o' ปล่อยด้วยมือ, 'm' ดู mask, 'q' ออก")
    try:
        while True:
            ok, frame = read_frame(cap)
            if not ok:
                break
            frame_raw = frame.copy()

            # แปลงทุกชิ้นเป็นพิกัดหุ่น แล้วเรียงตาม X, Y ให้เลขคงที่
            objs = []
            for (u, v, name) in find_objects(frame):
                x, y = to_robot(u, v)
                objs.append((x, y, u, v, name))
            objs.sort(key=lambda o: (round(o[0] / 15), o[1]))

            numbered = annotate(frame, objs)

            cv2.putText(frame, f"Objects: {len(numbered)}  press 1-9 or CLICK a piece, Q quit",
                        (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            cv2.imshow("run", frame)
            if show_mask:
                masks = build_masks(frame_raw)
                combined = np.zeros(frame_raw.shape[:2], np.uint8)
                for m in masks.values():
                    combined = cv2.bitwise_or(combined, m)
                cv2.imshow("mask", combined)

            k = cv2.waitKey(30) & 0xFF
            if k in (ord("q"), ord("Q")):
                break
            if k in (ord("m"), ord("M")):
                show_mask = not show_mask
                if not show_mask:
                    cv2.destroyWindow("mask")
                continue
            if k in (ord("o"), ord("O")):
                if robot and USE_PUMP:
                    robot.release()
                    holding = False
                    print("ปล่อยวัตถุแล้ว (ปิดดูด + เป่าลม)")
                continue
            idx = None
            if ord("1") <= k <= ord("9"):
                idx = k - ord("1")
            elif click["pt"] is not None:
                cx_, cy_ = click["pt"]
                click["pt"] = None
                if numbered:
                    d = [np.hypot(p[2] - cx_, p[3] - cy_) for p in numbered]
                    j = int(np.argmin(d))
                    if d[j] <= 45:  # คลิกใกล้หมากไม่เกิน 45 พิกเซล
                        idx = j
            if idx is not None:
                if holding:
                    print("กำลังถือวัตถุอยู่ กด 'o' เพื่อปล่อยก่อน")
                    continue
                if idx >= len(numbered):
                    print(f"ไม่มีวัตถุหมายเลข {idx + 1}")
                    continue
                tx, ty = numbered[idx][:2]
                pos = confirm_target(cap, to_robot, tx, ty)
                if pos is None:
                    print("วัตถุไม่นิ่ง / หายไป ลองใหม่อีกครั้ง")
                    continue
                x, y = pos
                print(f"เป้าหมาย #{idx + 1}: X={x:.1f} Y={y:.1f} Z={z_target:.1f}")
                if robot:
                    holding = pick_sequence(robot, x, y, z_target, clear_pos)
                    for _ in range(5):  # ทิ้งเฟรมเก่าในบัฟเฟอร์
                        cap.read()
                else:
                    print("(ALLOW_MOVE=False จึงไม่สั่งหุ่น)")
    finally:
        cap.release()
        cv2.destroyAllWindows()
        if robot:
            if USE_PUMP:
                try:
                    robot.pump_off()  # ออกโปรแกรมแล้วปิดปั๊มเสมอ (ของที่ถืออยู่จะหล่น)
                except Exception:
                    pass
            robot.close()



def promote(arg, robot, cap, to_robot, A, stock, clear_pos, z_hover, discard_pos):
    """เลื่อนขั้นหมากที่ช่อง arg เป็นฮอส: ทิ้งหมากธรรมดา -> หยิบฮอสสีเดียวกันจากที่เก็บ -> วางที่ช่องนั้น"""
    n = int(arg)
    if not (1 <= n <= BOARD_N * BOARD_N and is_dark(n)):
        print("เลขช่องต้องเป็นช่องสีฟ้า (ช่องเดินได้) ในช่วง 1-64")
        return
    if stock is None:
        print("ยังไม่ได้ตั้งจุดเก็บฮอส รันโหมด setstock ก่อน")
        return
    if robot is None:
        print("(ALLOW_MOVE=False จึงไม่สั่งหุ่น)")
        return
    snap = stable_occupancy(cap, to_robot, A)
    if snap is None:
        print("ภาพยังไม่นิ่ง ลองใหม่")
        return
    occ, kings = snap
    if n not in occ:
        print(f"ช่อง {n} ไม่มีหมาก (กล้องไม่เห็น) ไม่สั่งหุ่น")
        return
    if n in kings:
        print(f"ช่อง {n} เป็นฮอสอยู่แล้ว")
        return
    color = occ[n]
    objs = stable_objs(cap, to_robot)
    if objs is None:
        print("ภาพยังไม่นิ่ง ลองใหม่")
        return
    src = pick_stock_king(objs, stock, color)
    if src is None:
        print(f"ไม่มีฮอสสี {color} เหลือในที่เก็บ (กล้องไม่เห็นจุดเขียวบนหมากที่จุดเก็บ) ไม่สั่งหุ่น")
        return
    if not stock_reach_ok(*src):
        print(f"จุดเก็บ (X={src[0]:.0f}, Y={src[1]:.0f}) อยู่นอกระยะเอื้อมของแขน ไม่สั่งหุ่น")
        return
    pos = square_xy(A, n)
    if not in_range(*pos):
        print(f"ช่อง {n} อยู่นอกพื้นที่ปลอดภัย ไม่สั่งหุ่น")
        return
    print(f"เลื่อนขั้น {color} ช่อง {n}: ทิ้งหมากธรรมดา แล้วหยิบฮอสจาก X={src[0]:.1f} Y={src[1]:.1f} มาวาง")
    try:
        move_piece(robot, pos, None, clear_pos, z_hover, discard=True, discard_pos=discard_pos, to_clear=False)
        for _ in range(8):
            cap.read()
        time.sleep(0.5)
        mid = stable_occupancy(cap, to_robot, A)
        if mid is None or n in mid[0]:
            print("ช่องยังไม่ว่างหลังทิ้งหมาก (ภาพไม่นิ่งหรือหมากไม่ถูกหยิบ) หยุดเพื่อความปลอดภัย ตรวจด้วยคำสั่ง s")
            return
        move_piece(robot, src, pos, clear_pos, z_hover)
    except Exception as e:
        print("ERROR ระหว่างเคลื่อนที่:", e)
        return
    for _ in range(8):
        cap.read()
    time.sleep(0.5)
    after = stable_occupancy(cap, to_robot, A)
    if after is None:
        print("เสร็จแล้ว แต่ภาพยังไม่นิ่ง ตรวจด้วยคำสั่ง s")
    elif after[0].get(n) == color and n in after[1]:
        print(f"OK: ช่อง {n} เป็นฮอส {color} แล้ว")
    else:
        print(f"คำเตือน: ผลลัพธ์ไม่ตรงที่คาด (ช่อง {n}: {after[0].get(n)}, ฮอส={n in after[1]}) ตรวจหมากบนกระดานก่อนสั่งต่อ")


HELP_TEXT = """
คำสั่ง (พิมพ์ใน Terminal แล้วกด Enter):
  8 13      ย้ายหมากจากช่อง 8 ไปช่อง 13  (รับ 8-13 ได้เหมือนกัน)
  x 20      เก็บหมากที่ช่อง 20 ไปทิ้งที่จุดทิ้งหมาก (ตั้งด้วยโหมด setdiscard) ใช้กับหมากที่ถูกกิน
  k 64      เลื่อนขั้นหมากที่ช่อง 64 เป็นฮอส: ทิ้งหมากธรรมดาก่อน แล้วหยิบฮอสสีเดียวกันจากที่เก็บมาวางแทน
  s         แสดงสถานะกระดานที่กล้องเห็น (K = ฮอส)
  h         แสดงคำสั่ง
  q         ออก
"""


def play():
    """สั่งหุ่นย้ายหมากตามเลขช่อง 1-64 โดยตรวจด้วยกล้องก่อนและหลังทุกครั้ง"""
    H, z_plane, to_robot = load_calib()
    Hinv = np.linalg.inv(H)
    A = load_board()
    z_hover = z_plane + HOVER_ABOVE
    z_place = PICK_Z + PLACE_Z_OFFSET
    if not (Z_RANGE[0] <= PICK_Z <= z_place <= z_hover <= Z_RANGE[1]):
        raise RuntimeError(f"ลำดับ Z ผิด: PICK_Z={PICK_Z}, วาง={z_place}, hover={z_hover:.1f}, Z_RANGE={Z_RANGE}")
    if not os.path.exists(CLEAR_FILE):
        raise RuntimeError("ยังไม่ได้ตั้งจุดหลีกกล้อง ให้รันโหมด setclear ก่อน")
    clear_pos = [float(v) for v in np.load(CLEAR_FILE)["pos"]]
    if clear_pos[2] < z_hover:
        print("คำเตือน: จุดหลีกกล้องต่ำกว่าความสูง hover อาจชนหมากตอนถอย")
    for n in range(1, BOARD_N * BOARD_N + 1):
        if is_dark(n) and not in_range(*square_xy(A, n)):
            print(f"คำเตือน: ช่อง {n} อยู่นอก X_RANGE/Y_RANGE จะสั่งไปช่องนี้ไม่ได้")
    stock = load_stock()
    if stock is None:
        print("คำเตือน: ยังไม่ได้ตั้งจุดเก็บฮอส (รันโหมด setstock) คำสั่ง k จะใช้ไม่ได้")
    else:
        for color, pts in stock.items():
            for i, (sx, sy, _su, _sv) in enumerate(pts, 1):
                if not stock_reach_ok(sx, sy):
                    print(f"คำเตือน: จุดเก็บฮอส {color} #{i} (X={sx:.0f}, Y={sy:.0f}) อยู่นอกระยะเอื้อมของแขน")
    discard_pos = load_discard()
    if discard_pos is None:
        print("คำเตือน: ยังไม่ได้ตั้งจุดทิ้งหมาก (รันโหมด setdiscard) คำสั่ง x จะปล่อยที่จุดหลีกกล้องแทน")
    else:
        print(f"จุดทิ้งหมาก: X={discard_pos[0]:.1f} Y={discard_pos[1]:.1f} Z={discard_pos[2]:.1f}")

    robot = MG400(ROBOT_IP) if ALLOW_MOVE else None
    if robot:
        robot.enable()
        robot.pump_off()
    else:
        print("ALLOW_MOVE=False: จำลองการสั่งเท่านั้น หุ่นไม่ขยับ")

    cap = open_camera()
    cmds = queue.Queue()

    def reader():
        while True:
            try:
                cmds.put(input().strip())
            except EOFError:
                break

    threading.Thread(target=reader, daemon=True).start()
    print(HELP_TEXT)
    occ = {}
    try:
        while True:
            ok, frame = read_frame(cap)
            if not ok:
                break
            objs = detect_robot_objs(frame, to_robot)
            occ = occupancy(A, objs)
            kings = kings_on_board(A, objs)
            for (x, y, u, v, name, _k) in objs:   # หมากที่ไม่ตรงช่องใด ๆ ใกล้กระดาน = เตือน
                if in_range(x, y) and xy_to_square(A, x, y) is None:
                    cv2.circle(frame, (int(u), int(v)), 14, (0, 0, 255), 2)
            draw_board(frame, A, Hinv, occ, kings)
            if stock:
                draw_stock(frame, stock, objs)
            roi = get_roi()
            if roi:
                cv2.rectangle(frame, (roi[0], roi[1]), (roi[2], roi[3]), (255, 255, 0), 1)
            cv2.putText(frame, f"play: pieces on board={len(occ)}   type '8 13' in terminal, Q = quit",
                        (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            cv2.imshow("play", frame)
            if cv2.waitKey(30) & 0xFF in (ord("q"), ord("Q")):
                break

            try:
                line = cmds.get_nowait().lower()
            except queue.Empty:
                continue
            if line in ("q", "quit", "exit"):
                break
            if line in ("h", "help", "?"):
                print(HELP_TEXT)
                continue
            if line == "s":
                snap = stable_occupancy(cap, to_robot, A)
                if snap is None:
                    print("ภาพยังไม่นิ่ง ลองใหม่")
                else:
                    so, sk = snap
                    fmt = lambda col: [(f"{n}K" if n in sk else str(n)) for n in sorted(n for n, c in so.items() if c == col)]
                    print("ดำ(คน): %s | ส้ม(หุ่น): %s" % (fmt("black"), fmt("orange")))
                    if stock:
                        st_objs = stable_objs(cap, to_robot) or []
                        for color in ("orange", "black"):
                            cnt = sum(1 for (_sx, _sy, su, sv) in stock[color] if _stock_has_king(st_objs, color, su, sv))
                            print(f"  ฮอส{color} เหลือในที่เก็บ: {cnt}/4")
                    for n, c in so.items():
                        if n not in sk:
                            r, _c = square_rc(n)
                            if (c == "orange" and r == 0) or (c == "black" and r == BOARD_N - 1):
                                print(f"  *** หมาก {c} ช่อง {n} ถึงแถวสุดท้ายแล้ว พิมพ์ k {n} เพื่อเปลี่ยนเป็นฮอส")
                continue

            m_move = re.fullmatch(r"(\d+)\s*[-,>\s]\s*(\d+)", line)
            m_rm = re.fullmatch(r"x\s*(\d+)", line)
            m_king = re.fullmatch(r"k\s*(\d+)", line)
            if m_king:
                promote(m_king.group(1), robot, cap, to_robot, A, stock, clear_pos, z_hover, discard_pos)
                continue
            if not (m_move or m_rm):
                print("ไม่เข้าใจคำสั่ง พิมพ์ h เพื่อดูวิธีใช้")
                continue
            if m_move:
                a_, b_ = int(m_move.group(1)), int(m_move.group(2))
                squares = (a_, b_)
            else:
                a_, b_ = int(m_rm.group(1)), None
                squares = (a_,)
            if any(not (1 <= q <= BOARD_N * BOARD_N and is_dark(q)) for q in squares):
                print("เลขช่องต้องเป็นช่องสีฟ้า (ช่องเดินได้) ในช่วง 1-64")
                continue

            snap = stable_occupancy(cap, to_robot, A)
            if snap is None:
                print("ภาพยังไม่นิ่ง (มีอะไรขยับในภาพ?) ลองใหม่")
                continue
            snap, snap_kings = snap
            if a_ not in snap:
                print(f"ช่อง {a_} ไม่มีหมาก (กล้องไม่เห็น) ไม่สั่งหุ่น")
                continue
            if b_ is not None and b_ in snap:
                print(f"ช่อง {b_} มีหมากอยู่แล้ว ({snap[b_]}) ไม่สั่งหุ่น")
                continue
            src = square_xy(A, a_)
            dst = square_xy(A, b_) if b_ is not None else None
            for q, p in ((a_, src), (b_, dst)):
                if p is not None and not in_range(*p):
                    print(f"ช่อง {q} อยู่นอกพื้นที่ปลอดภัย (X={p[0]:.0f}, Y={p[1]:.0f}) ไม่สั่งหุ่น")
                    break
            else:
                color = snap[a_]
                what = f"ย้าย {color} ช่อง {a_} -> {b_}" if b_ else f"เก็บหมาก {color} ที่ช่อง {a_} ไปทิ้ง"
                print(f"{what}   (src X={src[0]:.1f} Y={src[1]:.1f}" +
                      (f", dst X={dst[0]:.1f} Y={dst[1]:.1f})" if dst is not None else ")"))
                if not robot:
                    print("(ALLOW_MOVE=False จึงไม่สั่งหุ่น)")
                    continue
                try:
                    move_piece(robot, src, dst, clear_pos, z_hover, discard=(b_ is None), discard_pos=discard_pos)
                except Exception as e:
                    print("ERROR ระหว่างเคลื่อนที่:", e)
                    continue
                for _ in range(8):      # ทิ้งเฟรมเก่า รอแขนพ้นภาพและภาพนิ่ง
                    cap.read()
                time.sleep(0.5)
                after = stable_occupancy(cap, to_robot, A)
                if after is None:
                    print("ย้ายเสร็จ แต่ภาพยังไม่นิ่ง ตรวจสอบด้วยคำสั่ง s")
                elif b_ is not None and after[0].get(b_) == color and a_ not in after[0]:
                    kn = " (ฮอส)" if b_ in after[1] else ""
                    if (a_ in snap_kings) != (b_ in after[1]):
                        print("คำเตือน: สถานะฮอสก่อน/หลังย้ายไม่ตรงกัน (จุดเขียวอาจถูกบัง/หมากพลิก) ตรวจด้วยคำสั่ง s")
                    print(f"OK: กล้องเห็นหมาก {color}{kn} อยู่ช่อง {b_} แล้ว")
                    r_b, _c = square_rc(b_)
                    if b_ not in after[1] and ((color == "orange" and r_b == 0) or (color == "black" and r_b == BOARD_N - 1)):
                        if AUTO_PROMOTE and color == "orange":
                            promote(str(b_), robot, cap, to_robot, A, stock, clear_pos, z_hover, discard_pos)
                        else:
                            print(f"*** หมากถึงแถวสุดท้ายแล้ว พิมพ์ k {b_} เพื่อเปลี่ยนเป็นฮอส")
                elif b_ is None and a_ not in after[0]:
                    print(f"OK: ช่อง {a_} ว่างแล้ว")
                else:
                    print(f"คำเตือน: ผลลัพธ์ไม่ตรงที่คาด (ช่อง {a_}: {after[0].get(a_)}, ช่อง {b_}: {after[0].get(b_) if b_ else '-'}) "
                          "ตรวจหมากบนกระดานก่อนสั่งต่อ")
    finally:
        cap.release()
        cv2.destroyAllWindows()
        if robot:
            try:
                robot.pump_off()
            except Exception:
                pass
            robot.close()


# ถ้ากด Run เฉย ๆ จะใช้โหมดนี้ ("calibrate", "camfocus", "setclear", "setdiscard", "setroi", "setstock", "teachstock", "pumptest", "pose", "boardcal", "play" หรือ "run")
DEFAULT_MODE = "camfocus"

def _dispatch(mode):
    if mode == "calibrate":
        calibrate()
    elif mode == "setclear":
        setclear()
    elif mode == "setdiscard":
        setdiscard()
    elif mode == "camfocus":
        camfocus()
    elif mode == "pumptest":
        pumptest()
    elif mode == "pose":
        show_pose()
    elif mode == "boardcal":
        boardcal()
    elif mode == "setroi":
        setroi()
    elif mode == "setstock":
        setstock()
    elif mode == "teachstock":
        teachstock()
    elif mode == "play":
        play()
    elif mode == "run":
        run()
    else:
        print(__doc__)


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_MODE
    try:
        _dispatch(mode)
    except (SystemExit, KeyboardInterrupt):
        raise
    except Exception:
        import traceback
        traceback.print_exc()
        print(f"\n*** โหมด {mode} จบด้วย error ตามข้อความด้านบน ***")
        if os.environ.get("MG400_PAUSE_ON_ERROR"):
            try:
                input("กด Enter เพื่อปิดหน้าต่างนี้ ...")
            except EOFError:
                pass
        sys.exit(1)
