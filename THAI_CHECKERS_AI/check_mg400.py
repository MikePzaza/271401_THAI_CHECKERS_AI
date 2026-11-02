"""Read-only Windows CLI for MG400 connectivity."""
from robot_connection import load_robot_ip, probe_mg400


def main():
    try:
        ip = load_robot_ip()
    except ValueError as exc:
        print('IP ตั้งค่าไม่ถูกต้อง:', exc)
        return 2
    print(f'กำลังอ่านสถานะ MG400 ที่ {ip} (ไม่สั่งแขนเคลื่อนที่)')
    result = probe_mg400(ip, timeout=3.0)
    print(result.display())
    if not result.connected:
        print('ยังไม่สามารถติดต่อทั้งสองพอร์ต: ตรวจสาย, IP, Subnet และสถานะเครื่อง')
        return 1
    if not result.available_for_game:
        print('ติดต่อ TCP ได้ แต่ RobotMode ไม่อยู่ในสถานะ 4/5 — ต้องตรวจเครื่องจริง')
        return 1
    print('ตรวจ TCP สำเร็จ แต่ยังไม่ได้ตรวจ E-stop/Guard/Calibration/Camera และไม่ได้เริ่มเกม')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
