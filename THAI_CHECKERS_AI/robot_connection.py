"""READ-ONLY MG400 TCP/IP connectivity diagnostic (no EnableRobot or motion).

The preflight only checks whether MG400 answers RobotMode() via dashboard port
and accepts a separate TCP connection on the motion port. TCP success is NOT
safety approval, a protective interlock, or proof that any move will succeed.
"""
from __future__ import annotations

from dataclasses import dataclass
import ipaddress
import json
from pathlib import Path
import re
import socket
import time

from mg400_visionV4 import ROBOT_IP, DASHBOARD_PORT, MOVE_PORT

CONFIG_PATH = Path(__file__).resolve().parent / 'robot_connection.json'
MODE_DESCRIPTIONS = {
    1: 'กำลังเริ่มระบบ', 2: 'เบรกถูกปลด', 3: 'ยังไม่จ่ายไฟ',
    4: 'ยังไม่ Enable', 5: 'Enable / ว่าง', 6: 'โหมดลากแขน',
    7: 'กำลังเคลื่อนหรือทำงาน', 8: 'กำลัง Jog/บันทึก',
    9: 'Alarm — ต้องให้ผู้ควบคุมตรวจ', 10: 'Paused',
    11: 'Collision / Jog — ต้องตรวจ',
}


def validate_ip(value: str) -> str:
    """Require IPv4; refuse hostnames and malformed entries rather than guessing."""
    try:
        return str(ipaddress.IPv4Address(value.strip()))
    except (ValueError, AttributeError) as exc:
        raise ValueError('IP ของ MG400 ต้องเป็น IPv4 เช่น 192.168.1.6') from exc


def load_robot_ip(path: Path = CONFIG_PATH) -> str:
    path = Path(path)
    if not path.is_file():
        return validate_ip(ROBOT_IP)
    try:
        data = json.loads(path.read_text(encoding='utf-8'))
        if not isinstance(data, dict):
            raise ValueError('เนื้อหา JSON ต้องเป็น object')
        return validate_ip(data['ip'])
    except (ValueError, KeyError, TypeError) as exc:
        raise ValueError(f'ไฟล์ตั้งค่า MG400 ผิดพลาด ({path.name}): {exc}') from exc


def save_robot_ip(ip: str, path: Path = CONFIG_PATH) -> str:
    validated = validate_ip(ip)
    Path(path).write_text(json.dumps({'ip': validated}, ensure_ascii=False, indent=2) + '\n',
                          encoding='utf-8')
    return validated


@dataclass(frozen=True)
class RobotProbe:
    ip: str
    dashboard_ok: bool
    motion_ok: bool
    mode: int | None
    detail: str
    checked_at: float

    @property
    def connected(self):
        return self.dashboard_ok and self.motion_ok and self.mode is not None

    @property
    def available_for_game(self):
        # Other operating modes may already be moving, alarmed, or unsafe.
        # This is only a software preflight, NOT a hardware safety clearance.
        return self.connected and self.mode in (4, 5)

    def display(self):
        if not self.connected:
            return 'ไม่เชื่อมต่อ — ' + self.detail
        return f'ติดต่อได้ {self.ip} (TCP 29999/30003) — {MODE_DESCRIPTIONS.get(self.mode, "โหมด " + str(self.mode))}'


def _robot_mode_reply(reply: str) -> int:
    # Expected: 0,{5},RobotMode();   (may contain whitespace/newlines)
    match = re.fullmatch(r'\s*(-?\d+)\s*,\s*\{\s*(\d+)\s*\}\s*,.*;\s*', reply, re.S)
    if not match:
        raise ValueError(f'RobotMode() ตอบกลับในรูปแบบที่ไม่รู้จัก: {reply[:90]!r}')
    if int(match.group(1)) != 0:
        raise ValueError(f'MG400 ปฏิเสธคำสั่งอ่านสถานะ: {reply[:90]}')
    return int(match.group(2))


def probe_mg400(ip: str, *, timeout: float = 2.0) -> RobotProbe:
    """Passive check: only read RobotMode and TCP-connect port 30003; no movement."""
    checked_at = time.monotonic()
    ip = validate_ip(ip)
    try:
        with socket.create_connection((ip, DASHBOARD_PORT), timeout=timeout) as sock:
            sock.settimeout(timeout)
            sock.sendall(b'RobotMode()')
            data = bytearray()
            while b';' not in data:
                block = sock.recv(1024)
                if not block:
                    raise ConnectionError('MG400 ปิดการเชื่อมต่อก่อนส่งคำตอบ')
                data.extend(block)
                if len(data) > 4096:
                    raise ValueError('คำตอบ RobotMode() ยาวผิดปกติ')
            mode = _robot_mode_reply(bytes(data).split(b';', 1)[0].decode('ascii', errors='replace') + ';')
    except (OSError, ValueError, ConnectionError) as exc:
        return RobotProbe(ip, False, False, None, f'พอร์ต {DASHBOARD_PORT}: {exc}', checked_at)

    try:
        # Checking the port does not send commands to the movement queue.
        with socket.create_connection((ip, MOVE_PORT), timeout=timeout):
            pass
    except OSError as exc:
        return RobotProbe(ip, True, False, mode, f'พอร์ต {MOVE_PORT}: {exc}', checked_at)

    return RobotProbe(ip, True, True, mode, MODE_DESCRIPTIONS.get(mode, f'โหมด {mode}'), checked_at)
