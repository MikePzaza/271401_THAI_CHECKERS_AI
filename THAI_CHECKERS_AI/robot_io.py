"""MG400 integration using the operator-tested mg400_visionV4 profile.

Implements V4 color/king reserve logic, dedicated discard, and raised travel
heights, under the dashboard's independent site safety approval gate.
Importing this module does not connect to hardware or command movement.
"""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import json
import math
import re
import time
import numpy as np
from game_logic import move_numbers, number_of
from engine import move_promotes, crown_skipped
import mg400_visionV4 as v


class RobotFailure(RuntimeError):
    pass


@dataclass(frozen=True)
class Transfer:
    source: int | None
    dest: int | None
    kind: str  # piece, captured, king_remove, king_place
    king: bool = False  # หมากที่หยิบเป็นฮอสอยู่แล้ว
    via: tuple = ()     # ช่องลงจอดกลางทางของการกินหลายตัว (ถือหมากผ่าน ย่อลง-ยกขึ้นที่แต่ละช่อง)


def moves_to_transfers(move, board=None, capture_last=True, direct=True):
    """
    capture_last=True (ค่าเริ่มต้นใหม่): เดินหมากตัวที่กินก่อน แล้วค่อยเก็บหมากที่ถูกกินไปทิ้ง
    capture_last=False: แบบเดิม เก็บหมากที่ถูกกินก่อนแล้วค่อยเดิน
    direct=True (ค่าเริ่มต้นใหม่): หมากที่กินหลายตัวหยิบครั้งเดียวไปวางที่ช่องปลายทางเลย (ข้ามช่องกลางที่ยกผ่านด้วยความสูง hover)
    direct=False: แบบเดิม หยิบ-วางทีละช่วง (25->13 แล้ว 13->1)
    ตามด้วยเลื่อนขั้น (V4 discard + ฮอสจากที่เก็บ) ถ้าหมากเดินถึงแถวสุดท้าย
    """
    captured = [Transfer(number_of(c), None, 'captured') for c in move.captures]
    path = move_numbers(move)
    is_king = False
    if board is not None:
        r0, c0 = move.start
        is_king = abs(board[r0][c0]) == 2
    if direct:
        hops = [Transfer(path[0], path[-1], 'piece', is_king, tuple(path[1:-1]))]
    else:
        hops = [Transfer(a, b, 'piece', is_king) for a, b in zip(path, path[1:])]
    steps = (hops + captured) if capture_last else (captured + hops)
    # เดินถึงแถวสุดท้ายแล้วชนะเลย -> ไม่ต้องเปลี่ยนเป็นฮอส จบเกมได้ทันที
    if board is not None and move_promotes(board, move) and not crown_skipped(board, move):
        steps.extend((Transfer(path[-1], None, 'king_remove'),
                      Transfer(None, path[-1], 'king_place')))
    return steps


def _board_target(p, label):
    x, y = map(float, p)
    if not all(map(math.isfinite, (x, y))) or not v.in_range(x, y):
        raise RobotFailure(f'{label}: พิกัดนอกพื้นที่ช่องกระดาน {p}')
    return x, y


def _external_target(p, label):
    """V4 stock/discard can be outside the rectangular BOARD limits."""
    if p is None or len(p) < 2:
        raise RobotFailure(f'{label}: ไม่มีพิกัด')
    x, y = map(float, p[:2])
    if not all(map(math.isfinite, (x, y))) or not v.stock_reach_ok(x, y):
        raise RobotFailure(f'{label}: นอกขอบเขตแขนที่กำหนดใน V4 ({x:.1f},{y:.1f})')
    return x, y


def _z(z, label):
    z = float(z)
    if not math.isfinite(z) or not v.Z_RANGE[0] <= z <= v.Z_RANGE[1]:
        raise RobotFailure(f'{label}: Z={z} นอก Z_RANGE={v.Z_RANGE}')
    return z


class RobotPlanner:
    def __init__(self, camera):
        self.camera = camera
        self.z_hover = float(camera.z_plane) + v.HOVER_ABOVE
        self.z_place = v.PICK_Z + v.PLACE_Z_OFFSET
        if not (v.Z_RANGE[0] <= v.PICK_Z <= self.z_place <= self.z_hover <= v.Z_RANGE[1]):
            raise RobotFailure('PICK_Z / PLACE_Z / HOVER_Z ไม่อยู่ในลำดับที่ปลอดภัย')
        self.clear = self._load_clear()
        self.stock = v.load_stock()
        self.king_source = None
        self.has_kings = False

    @staticmethod
    def _load_clear():
        if not Path(v.CLEAR_FILE).is_file():
            raise RobotFailure('ไม่พบ clear_pos.npz (จุดหลบกล้อง V4)')
        pos = np.load(v.CLEAR_FILE, allow_pickle=False)['pos']
        if pos.shape != (4,) or not np.isfinite(pos).all():
            raise RobotFailure('clear_pos.npz ไม่ถูกต้อง')
        _external_target(pos, 'จุดหลบกล้อง')
        _z(pos[2], 'จุดหลบกล้อง')
        return tuple(map(float, pos))

    def plan(self, move, board=None):
        # Planning is pure math (supports offline test cameras); hardware run()
        # enforces board calibration before opening any MG400 socket.
        self.has_kings = any(abs(piece) == 2 for row in board for piece in row) if board is not None else False
        steps = moves_to_transfers(move, board)
        for step in steps:
            if step.source is not None:
                _board_target(self.camera.xy(step.source), f'ช่อง {step.source}')
            if step.dest is not None:
                _board_target(self.camera.xy(step.dest), f'ช่อง {step.dest}')
        self.king_source = None
        if any(s.kind == 'king_place' for s in steps):
            if self.stock is None:
                raise RobotFailure('ยังไม่มี stock_pos.npz ของ V4 สำหรับหยิบฮอสจริง')
            objs = getattr(self.camera, 'last_robot_objects', None)
            if objs is None:
                raise RobotFailure('กล้องยังไม่มีภาพจุดเก็บฮอสจาก V4')
            # ฮอสและหมากธรรมดาสูงเท่ากัน: คง PICK_Z/เส้นทาง V4 ทุกอย่างไว้
            # ที่ V17 เดิม STOCK_DYNAMIC ใช้ XY ที่คำนวณจากกล้องนอกกระดาน
            # และยอมรับวัตถุห่างจุดเก็บได้ถึง 60 px แม้ความคลาดเคลื่อน
            # นี้อาจทำให้หัวดูดไม่อยู่กึ่งกลางฮอสจริงเมื่อหุ่นลงไปหยิบ
            # เลือกตำแหน่ง XY ที่บันทึกใน stock_pos.npz *เฉพาะ* เมื่อ
            # กล้องยืนยันว่ามีฮอสสีส้มจริงภายใน STOCK_MATCH_PX ของจุดนั้น
            candidates = []
            for sx, sy, su, sv in self.stock['orange']:
                for _x, _y, u, w, color, is_king in objs:
                    if color != 'orange' or not is_king:
                        continue
                    distance = math.hypot(float(u) - su, float(w) - sv)
                    if distance <= v.STOCK_MATCH_PX:
                        candidates.append((distance, sx, sy))
            if not candidates:
                raise RobotFailure(
                    'กล้องไม่ยืนยันฮอสสีส้มใกล้ตำแหน่งที่บันทึกไว้ใน stock_pos.npz '
                    '(ต้องเห็นจุดเขียวและวางตัวฮอสตรงจุดเก็บ; ไม่สั่งแขนไปเดาพิกัด)')
            _distance, sx, sy = min(candidates)
            self.king_source = _external_target((sx, sy), 'จุดหยิบฮอสสีส้ม')
        return steps


class LiveRunner:
    """No MG400 connection until all site-verified interlocks are explicitly approved."""
    def __init__(self, gate_path):
        self.path = Path(gate_path)
        if not self.path.is_file():
            raise RobotFailure('ไม่พบ live_safety_config.json')
        self.cfg = json.loads(self.path.read_text(encoding='utf-8'))
        # motion_engine: "sensor" (ของเดิม: ต้องมีเซนเซอร์สุญญากาศ DI) | "v4" (ใช้ move_piece ของ mg400_visionV4 ที่ทดสอบกับหุ่นจริง ไม่ต้องมีเซนเซอร์)
        self.engine = self.cfg.get('motion_engine', 'sensor')
        if self.engine not in ('sensor', 'v4'):
            raise RobotFailure("motion_engine ต้องเป็น 'sensor' หรือ 'v4'")
        keys = ['approved_for_this_setup', 'emergency_stop_hardware_tested',
                'human_exclusion_guard_tested',
                'board_alignment_physically_verified', 'tool_clearance_tested']
        if self.engine == 'sensor':
            keys.append('vacuum_sensor_tested')
        for key in keys:
            if self.cfg.get(key) is not True:
                raise RobotFailure(f'ยังไม่ผ่านการตรวจหน้างาน: {key}')
        self.pin = self.active = None
        if self.engine == 'sensor':
            pin = self.cfg.get('vacuum_sensor_di')
            if not isinstance(pin, int) or isinstance(pin, bool) or pin < 1:
                raise RobotFailure('ต้องระบุ Digital Input ของเซนเซอร์สุญญากาศที่ตรวจจริง')
            if self.cfg.get('sensor_active_value') not in (0, 1):
                raise RobotFailure('sensor_active_value ต้องเป็น 0/1 ที่ทดสอบแล้ว')
            self.pin = pin
            self.active = self.cfg['sensor_active_value']
        # False = ยืนยันครั้งเดียวตอนกดเริ่มเกม แล้วหุ่นเดินเองทุกตา (ยังหยุดได้ด้วยปุ่มหยุด/E-stop)
        self.per_move_confirmation = self.cfg.get('per_move_confirmation', True) is not False
        self.king_verified = self.cfg.get('physical_king_identification_verified') is True
        self.discard = v.load_discard()
        configured = self.cfg.get('capture_discard_pose')
        if self.discard is not None:
            # The user-tested .npz is the physical position of record; never silently override.
            if configured is not None and (len(configured) != 4 or not np.allclose(configured, self.discard, atol=0.1)):
                raise RobotFailure('จุดทิ้งหมากใน JSON ไม่ตรงกับ discard_pos.npz ที่ทดสอบ')
        elif configured is not None:
            self.discard = configured
        if self.discard is not None:
            if len(self.discard) != 4 or not all(math.isfinite(float(x)) for x in self.discard):
                raise RobotFailure('พิกัดจุดทิ้งหมากไม่ถูกต้อง')
            _external_target(self.discard, 'จุดทิ้งหมาก')
            _z(self.discard[2], 'จุดทิ้งหมาก')

    def _read_sensor(self, robot):
        reply = robot.d(f'DI({self.pin})')
        found = re.search(r'\{\s*([01])\s*\}', reply)
        if found is None:
            raise RobotFailure('อ่านสถานะเซนเซอร์ดูดไม่ได้: ' + reply)
        return int(found.group(1)) == self.active

    def _go(self, robot, pose, stop):
        if stop():
            raise RobotFailure('ได้รับคำขอหยุด; หากอันตรายต้องใช้ E-stop ฮาร์ดแวร์')
        x, y, z, _r = pose
        r = v.TOOL_R   # Rotation ของ Tool ล็อก 0 ทุกการเคลื่อนที่
        _z(z, 'เส้นทางหุ่นยนต์')
        robot.m(f'MovL({x:.1f},{y:.1f},{z:.1f},{r:.1f})')
        robot.m('Sync()')

    def _transfer(self, robot, source, dest, planner, *, stop, report, discard=False):
        sx, sy = source
        dx, dy, dz, dr = dest
        _z(dz, 'ระดับปล่อยหมาก')
        _external_target(source, 'จุดหยิบ')
        _external_target(dest, 'จุดปล่อย')
        r = v.TOOL_R
        # V4: special raised path over the external discard tray.
        travel_z = min(v.Z_RANGE[1], max(planner.z_hover + v.DISCARD_EXTRA_LIFT, dz) + v.DISCARD_TRAVEL_RAISE) if discard else planner.z_hover
        _z(travel_z, 'ระดับเดินทาง')
        if travel_z < v.PICK_Z or travel_z < dz:
            raise RobotFailure('ทางเดินแนวนอนต่ำกว่าจุดหยิบหรือจุดวาง')
        self._go(robot, (sx, sy, planner.z_hover, r), stop)
        self._go(robot, (sx, sy, v.PICK_Z, r), stop)
        robot.vacuum_on()
        time.sleep(v.PUMP_SETTLE)
        if not self._read_sensor(robot):
            raise RobotFailure('เซนเซอร์ไม่ยืนยันว่าดูดหมากติด')
        self._go(robot, (sx, sy, travel_z, r), stop)
        self._go(robot, (dx, dy, travel_z, dr), stop)
        if dz < travel_z:
            self._go(robot, (dx, dy, dz, dr), stop)
        robot.release()
        time.sleep(0.2)
        if self._read_sensor(robot):
            raise RobotFailure('เซนเซอร์ยังตรวจพบหมากหลังปล่อย')
        if dz < travel_z:
            self._go(robot, (dx, dy, travel_z, dr), stop)
        self._go(robot, planner.clear, stop)
        report('ทิ้งหมากแล้ว' if discard else 'วางหมากแล้ว')

    def _transfer_v4(self, robot, step, planner, *, stop, report, last=True):
        """ใช้ v.move_piece ของ mg400_visionV4 (เส้นทาง/ปั๊ม/ยกสูงตอนทิ้ง ตามที่ทดสอบกับหุ่นจริง)"""
        if stop():
            raise RobotFailure('ได้รับคำขอหยุด; หากอันตรายต้องใช้ E-stop ฮาร์ดแวร์')
        cam = planner.camera
        clear = list(planner.clear)
        if step.kind in ('captured', 'king_remove'):
            src = _board_target(cam.xy(step.source), f'ช่อง {step.source}')
            report(f'MG400 ทิ้งหมากช่อง {step.source} ลงจุดทิ้ง')
            v.move_piece(robot, src, None, clear, planner.z_hover, discard=True, discard_pos=self.discard, to_clear=last)
            report('ทิ้งหมากแล้ว')
        elif step.kind == 'king_place':
            dst = _board_target(cam.xy(step.dest), f'ช่อง {step.dest}')
            report(f'MG400 หยิบฮอสสีส้มจากที่เก็บ → ช่อง {step.dest}')
            v.move_piece(robot, planner.king_source, dst, clear, planner.z_hover, to_clear=last)
            report('วางฮอสแล้ว')
        else:
            src = _board_target(cam.xy(step.source), f'ช่อง {step.source}')
            dst = _board_target(cam.xy(step.dest), f'ช่อง {step.dest}')
            report(f'MG400 เดินหมาก {step.source} → {step.dest}')
            via = [_board_target(cam.xy(n), f'ช่อง {n}') for n in (step.via or ())]
            v.move_piece(robot, src, dst, clear, planner.z_hover, to_clear=last, src_is_king=step.king, via=via)
            report('วางหมากแล้ว')

    def run(self, steps, planner, *, report, stop, robot_ip=None):
        if not planner.camera.verified_board:
            raise RobotFailure('ต้องมี board.npz จริงก่อนเชื่อมต่อ MG400')
        if any(s.kind in ('captured', 'king_remove') for s in steps) and self.discard is None:
            raise RobotFailure('ยังไม่มีจุดทิ้งหมาก ไม่ปล่อยหมากที่จุดหลบกล้องแทน')
        crowning = any(s.kind in ('king_remove', 'king_place') for s in steps)
        if crowning and not self.king_verified:
            raise RobotFailure('การระบุฮอส V4 ยังไม่ได้รับการตรวจยืนยันที่หน้างาน')
        if crowning and planner.king_source is None:
            raise RobotFailure('ไม่ได้ระบุจุดหยิบฮอสจากภาพ V4')
        # Preflight all positions BEFORE connecting.
        for step in steps:
            if step.source is not None:
                _board_target(planner.camera.xy(step.source), f'ช่อง {step.source}')
            if step.dest is not None:
                _board_target(planner.camera.xy(step.dest), f'ช่อง {step.dest}')
        if crowning:
            _external_target(planner.king_source, 'ฮอสสำรอง')
        if stop():
            raise RobotFailure('มีคำขอหยุดก่อนเชื่อมต่อ MG400')
        robot = None
        try:
            robot = v.MG400(robot_ip or v.ROBOT_IP)
            # Never clear hardware alarms automatically.
            robot.d('EnableRobot()')
            for _ in range(30):
                if '{5}' in robot.d('RobotMode()'):
                    break
                time.sleep(0.5)
            else:
                raise RobotFailure('MG400 ยังไม่พร้อม/ไม่ Enabled ตรวจ pendant และ alarm')
            robot.d(f'SpeedFactor({v.SPEED})')
            robot.pump_off()
            for idx, step in enumerate(steps):
                if stop():
                    raise RobotFailure('ได้รับคำขอหยุดระหว่างตา')
                if self.engine == 'v4':
                    # ไปจุดหลีกกล้องเฉพาะหลังขั้นสุดท้ายของตา ขั้นก่อนหน้ายกแขนแล้วไปหยิบชิ้นถัดไปต่อเลย
                    self._transfer_v4(robot, step, planner, stop=stop, report=report, last=(idx == len(steps) - 1))
                    continue
                if step.kind in ('captured', 'king_remove'):
                    source = _board_target(planner.camera.xy(step.source), f'ช่อง {step.source}')
                    report(f'MG400 ทิ้งหมากช่อง {step.source} ลงถาด V4')
                    self._transfer(robot, source, self.discard, planner, stop=stop, report=report, discard=True)
                elif step.kind == 'king_place':
                    source = planner.king_source
                    dest = _board_target(planner.camera.xy(step.dest), f'ช่อง {step.dest}')
                    report(f'MG400 หยิบฮอสสีส้มจากที่เก็บ → ช่อง {step.dest}')
                    self._transfer(robot, source, (*dest, planner.z_place, 0.0), planner, stop=stop, report=report)
                else:
                    source = _board_target(planner.camera.xy(step.source), f'ช่อง {step.source}')
                    dest = _board_target(planner.camera.xy(step.dest), f'ช่อง {step.dest}')
                    report(f'MG400 เดินหมาก {step.source} → {step.dest}')
                    self._transfer(robot, source, (*dest, planner.z_place, 0.0), planner, stop=stop, report=report)
            return 'ROBOT REPORTED DONE (AWAIT CAMERA KING + OCCUPANCY VERIFICATION)'
        finally:
            if robot is not None:
                # Do not drop a held piece by disabling vacuum on interrupted motion.
                # Physical emergency stop and recovery are operator responsibilities.
                try:
                    robot.close()
                except Exception:
                    pass
