"""ค่าสอบเทียบจำลองสำหรับเทสต์ 8x8 (ไม่ใช่ค่าจริงหน้างาน): ไฟล์ npz ของจริงถูกลบออกแล้ว ต้องสอบเทียบใหม่"""
import os, sys, tempfile
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import mg400_visionV4 as v

FIX = Path(__file__).resolve().parent / "fixtures"
A = np.array([[406.14, -0.81, -30.41], [-110.0, 31.51, -0.82]])
_dir = tempfile.mkdtemp()
BOARD = os.path.join(_dir, "board.npz")
np.savez(BOARD, A=A)
v.BOARD_FILE = BOARD
v.CALIB_FILE = str(FIX / "calib.npz")
v.CLEAR_FILE = str(FIX / "clear_pos.npz")
v.DISCARD_FILE = str(FIX / "discard_pos.npz")
v.STOCK_FILE = str(FIX / "stock_pos.npz")
v.ROI_FILE = os.path.join(_dir, "none_roi.npz")
v._ROI_CACHE["roi"] = None
