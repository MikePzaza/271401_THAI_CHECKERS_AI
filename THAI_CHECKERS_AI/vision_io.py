"""Use the user's TESTED V4 HSV detection and camera-to-MG400 calibration.

board.npz is required by the real-board app; affine fallback is for offline unit tests only.
Stable readings reject duplicates and unsnapped pieces instead of guessing.
"""
import time
from collections import deque
from pathlib import Path
import cv2
import numpy as np
import mg400_visionV4 as user

class VisionFailure(RuntimeError):
    pass

class BoardCamera:
    def __init__(self, *, require_board_file=False):
        self.root = Path(__file__).resolve().parent
        if require_board_file and not user.os.path.exists(user.BOARD_FILE):
            raise VisionFailure('ยังไม่มี board.npz — ต้องรัน boardcal จากโค้ด V4 ก่อนใช้ REAL')
        if not Path(user.CALIB_FILE).is_file():
            raise VisionFailure('หา calib.npz ไม่พบ')
        try:
            self.H, self.z_plane, self.to_robot = user.load_calib()
            self.A = user.load_board()
        except Exception as e:
            raise VisionFailure('อ่านไฟล์สอบเทียบไม่สำเร็จ: ' + str(e)) from e
        self.camera = None
        self.last_kings = set()
        self._obs_hist = deque(maxlen=getattr(user, 'OCC_VOTE_FRAMES', 5))   # ประวัติการเห็นหมากรายเฟรม: กันหมากบางตัวเห็นๆ หายๆ (เช่น หมากริมกระดานที่เงา/วัตถุข้างเคียงรบกวน)
        self._obs_time = 0.0
        self._king_hist = deque(maxlen=5)   # ประวัติจุดเขียว 5 เฟรมล่าสุด: กันจุดเขียวกะพริบทำให้ฮอสหาย/โผล่เป็นช่วง ๆ
        self.last_robot_objects = []
        self.Hinv = np.linalg.inv(self.H)

    def _snap_dark(self, x, y):
        tol = getattr(user, 'SNAP_DARK_TOL', 0.7)
        if tol <= 0:
            return None
        col, row = np.linalg.solve(self.A[:, 1:], np.array([x, y]) - self.A[:, 0])
        best = None
        for ci in (int(np.floor(col)), int(np.ceil(col))):
            for ri in (int(np.floor(row)), int(np.ceil(row))):
                if 0 <= ci < user.BOARD_N and 0 <= ri < user.BOARD_N and (ri + ci) % 2 == 0:
                    d = float(np.hypot(col - ci, row - ri))
                    if d <= tol and (best is None or d < best[0]):
                        best = (d, ri * user.BOARD_N + ci + 1)
        return None if best is None else best[1]

    @property
    def verified_board(self):
        return Path(user.BOARD_FILE).exists()

    def open(self):
        if self.camera is None:
            self.camera = user.open_camera()

    def close(self):
        if self.camera is not None:
            self.camera.release()
            self.camera = None

    def read_frame(self):
        if self.camera is None:
            raise VisionFailure('กรุณากดเปิดกล้องก่อน')
        ok, frame = user.read_frame(self.camera)
        if not ok:
            raise VisionFailure('อ่านเฟรมกล้องไม่สำเร็จ')
        return frame

    def observe(self, frame):
        """Use V4 color + green-dot king detection. Keep occupancy API intact."""
        obs = {}
        kings = set()
        detected = user.detect_robot_objs(frame, self.to_robot)
        for x, y, u, v, color, king in detected:
            n = user.xy_to_square(self.A, x, y)
            if n is None or not user.is_dark(n):
                # หมากวางเยื้องจากกลางช่อง (เช่น คร่อมขอบ/เกือบเลยขอบกระดาน): ถ้าอยู่ใกล้กลางช่องเดินได้ช่องใดช่องหนึ่งพอ ให้ปัดเข้าช่องนั้น
                snapped = self._snap_dark(x, y)
                if snapped is not None:
                    n = snapped
            if n is None:
                cr = np.linalg.solve(self.A[:, 1:], np.array([x, y]) - self.A[:, 0])
                col, row = cr
                # ขอบกระดานจริงอยู่ที่ -0.5 ถึง 5.5 (ส่วนจุดเก็บฮอสอยู่ราว col -1.4 และ 6.2)
                # ขยายเป็น -0.65 ถึง BOARD_N-0.35 เพื่อไม่ให้หมากที่วางชิดขอบนอกถูกตัดทิ้งเงียบ ๆ จนกลายเป็นหมากหาย
                if -0.65 <= col <= user.BOARD_N - 0.35 and -0.65 <= row <= user.BOARD_N - 0.35:
                    raise VisionFailure('พบหมากใกล้กระดานแต่ไม่อยู่กลางช่อง (กรุณาจัดให้นิ่ง)')
                continue
            if not user.is_dark(n):
                raise VisionFailure(f'ตรวจพบหมากบนช่องที่เดินไม่ได้: {n}')
            if n in obs:
                raise VisionFailure(f'กล้องตรวจพบวัตถุซ้อนกันในช่อง {n}')
            obs[n] = color
            if king:
                kings.add(n)
        if sum(color == 'black' for color in obs.values()) > user.PIECES_PER_SIDE or sum(color == 'orange' for color in obs.values()) > user.PIECES_PER_SIDE:
            raise VisionFailure(f'ตรวจพบหมากมากกว่า {user.PIECES_PER_SIDE} ตัวต่อฝ่าย ตรวจ HSV / เงา')
        # โหวตการครอบครองช่องแบบเสียงข้างมากของเฟรมล่าสุด (ล้างประวัติถ้าเว้นช่วงนาน เช่น ตอนหุ่นทำงาน)
        now = time.monotonic()
        if now - self._obs_time > 0.8:
            self._obs_hist.clear()
        self._obs_time = now
        self._obs_hist.append(dict(obs))
        need_occ = len(self._obs_hist) // 2 + 1
        voted_obs = {}
        for n in {n for h in self._obs_hist for n in h}:
            for color in ('black', 'orange'):
                if sum(h.get(n) == color for h in self._obs_hist) >= need_occ:
                    voted_obs[n] = color
        obs = voted_obs
        kings = {n for n in kings if n in obs}
        self._king_hist.append(frozenset(kings))
        need = len(self._king_hist) // 2 + 1                     # เสียงข้างมากของเฟรมล่าสุด
        voted = {n for n in obs if sum(n in h for h in self._king_hist) >= need}
        self.last_kings = voted
        self.last_robot_objects = detected   # includes king reserves outside board for V4 dynamic stock picking
        return obs

    def snapshot(self, frames=5):
        """
        อ่านต่อเนื่องจนเห็นกระดาน+ฮอสเหมือนกันติดกัน `frames` เฟรม (รอได้สูงสุด user.STABLE_MAX_FRAMES เฟรม
        เช่น รอมือพ้นภาพ) ถ้าไม่นิ่งถึงเวลา แจ้ง VisionFailure ตามเดิม
        """
        if self.camera is None:
            raise VisionFailure('ยังไม่ได้เปิดกล้อง')
        prev, streak, last_error = None, 0, None
        for _ in range(max(frames, getattr(user, 'STABLE_MAX_FRAMES', frames))):
            try:
                seen = self.observe(self.read_frame())
            except VisionFailure as exc:
                if 'เปิดกล้อง' in str(exc) or 'อ่านเฟรม' in str(exc):
                    raise
                last_error, prev, streak = exc, None, 0
                continue
            state = (seen, frozenset(self.last_kings))
            streak = streak + 1 if state == prev else 1
            prev = state
            if streak >= frames:
                return state[0]
        if last_error is not None:
            raise last_error
        raise VisionFailure('ภาพ/ตำแหน่งฮอสยังไม่นิ่งหรือมีมือบัง รอแล้วลองอีกครั้ง')

    def annotate(self, frame, obs=None):
        display = frame.copy()
        try:
            seen = obs if obs is not None else self.observe(frame)
            user.draw_board(display,self.A,self.Hinv,seen,self.last_kings)
        except (VisionFailure,ValueError,np.linalg.LinAlgError):
            pass
        return display

    def xy(self, cell):
        return tuple(map(float,user.square_xy(self.A,cell)))
