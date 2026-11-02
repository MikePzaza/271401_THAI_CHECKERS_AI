"""MG400 on-site hardware PRE-FLIGHT: read-only TCP queries and optional webcam check.

No EnableRobot, ClearError, MovL/MovJ, DO or suction commands are issued.
Use on the Windows PC physically connected to the MG400, with a trained operator.
This program does not certify E-stop/guard/physical calibration.
"""
from __future__ import annotations

import argparse
from datetime import datetime
import json
import math
from pathlib import Path
import re
import socket
import sys
import traceback

import numpy as np

import mg400_visionV4 as v
from robot_connection import load_robot_ip, probe_mg400, MODE_DESCRIPTIONS

ROOT = Path(__file__).resolve().parent
SHAPES = {
    'calib.npz': {'H': (3, 3), 'z_plane': (), 'rot180': ()},
    'board.npz': {'A': (2, 3)},
    'roi.npz': {'roi': (4,)},
    'clear_pos.npz': {'pos': (4,)},
    'discard_pos.npz': {'pos': (4,)},
    'stock_pos.npz': {'orange_xy': (4, 2), 'black_xy': (4, 2), 'orange_uv': (4, 2), 'black_uv': (4, 2)},
}


def audit_files():
    statuses = {}
    for name, keys in SHAPES.items():
        try:
            with np.load(ROOT/name, allow_pickle=False) as data:
                for key, shape in keys.items():
                    if key not in data.files or data[key].shape != shape or not np.isfinite(data[key]).all():
                        raise ValueError(f'{key}: key/shape/finite mismatch (expected {shape})')
            statuses[name] = {'ok': True, 'detail': 'schema OK'}
        except Exception as ex:
            statuses[name] = {'ok': False, 'detail': str(ex)}
    try:
        with np.load(ROOT/'calib.npz',allow_pickle=False) as cal:
            if bool(cal['rot180']) != v.ROTATE_180:
                raise ValueError('ROTATE_180 not equal to camera calibration setting')
        with np.load(ROOT/'board.npz',allow_pickle=False) as board:
            A = board['A']
            if abs(np.linalg.det(A[:, 1:])) < 1e-3:
                raise ValueError('board affine matrix is singular')
            off_board = [n for n in range(1,37) if v.is_dark(n) and not v.in_range(*v.square_xy(A,n))]
            if off_board:
                raise ValueError('playable squares outside V4 rectangular limits: '+str(off_board))
        statuses['consistency'] = {'ok': True, 'detail': 'rotation + board bounds OK (not physically measured)'}
    except Exception as ex:
        statuses['consistency'] = {'ok': False, 'detail': str(ex)}
    return statuses


def dashboard_readonly_get_pose(ip, timeout=3):
    """Only GetPose() read query on dashboard port. Does not command movement."""
    with socket.create_connection((ip, v.DASHBOARD_PORT), timeout=timeout) as s:
        s.settimeout(timeout)
        s.sendall(b'GetPose()')
        reply = bytearray()
        while b';' not in reply:
            chunk = s.recv(512)
            if not chunk:
                raise ConnectionError('MG400 disconnected before GetPose reply')
            reply.extend(chunk)
            if len(reply) > 4096:
                raise ValueError('GetPose reply too long')
    message = bytes(reply).split(b';',1)[0].decode('ascii',errors='replace')+';'
    m = re.fullmatch(r'\s*(-?\d+)\s*,\s*\{([^{}]+)\}\s*,.*;', message, re.S)
    if not m or int(m.group(1)) != 0:
        raise ValueError('Unexpected GetPose reply: '+message[:150])
    parts = [float(x.strip()) for x in m.group(2).split(',')]
    if len(parts) < 4 or not all(map(math.isfinite, parts[:4])):
        raise ValueError('Pose must contain finite X,Y,Z,R')
    return {'x': round(parts[0],3), 'y': round(parts[1],3), 'z': round(parts[2],3), 'r': round(parts[3],3)}


def camera_snapshot(report_folder, tag):
    """Optional webcam capture. NO robot connection or robot movement command."""
    info = {'ok': False, 'image': None, 'board_detected': None, 'kings': None}
    cap = None
    try:
        H, zp, to_robot = v.load_calib()
        A = v.load_board()
        cap = v.open_camera()
        ok = False
        frame = None
        for _ in range(5):
            ok, frame = v.read_frame(cap)
        if not ok:
            raise RuntimeError('Cannot read webcam frame')
        if frame.shape[1] < 850 or frame.shape[0] < 580:
            raise RuntimeError('Camera resolution smaller than saved ROI (expected 1280x720 setup)')
        objects = v.detect_robot_objs(frame, to_robot)
        occ = v.occupancy(A, objects)
        kings = v.kings_on_board(A, objects)
        canvas = frame.copy()
        v.draw_board(canvas, A, np.linalg.inv(H), occ, kings)
        roi = v.get_roi()
        if roi:
            x1,y1,x2,y2 = roi
            import cv2
            cv2.rectangle(canvas, (x1,y1), (x2,y2), (255,255,0), 2)
        image_path = report_folder/f'mg400_camera_board_{tag}.png'
        import cv2
        if not cv2.imwrite(str(image_path), canvas):
            raise RuntimeError('Cannot save camera preview')
        initial_black = {n for n, color in occ.items() if color == 'black'}
        initial_orange = {n for n, color in occ.items() if color == 'orange'}
        initial_setup_ok = initial_black == set(v.START_BLACK) and initial_orange == set(v.START_ORANGE)
        stock = v.load_stock()
        reserve_counts = None
        if stock:
            reserve_counts = {c: sum(1 for (_x,_y,u,vv) in stock[c]
                                      if v._stock_has_king(objects,c,u,vv)) for c in ('orange','black')}
        info.update(ok=True, image=str(image_path), board_detected={str(k): val for k,val in sorted(occ.items())},
                    kings=sorted(kings), count_black=len(initial_black),
                    count_orange=len(initial_orange), initial_setup_matches_v4=initial_setup_ok,
                    king_reserves_detected=reserve_counts)
        info['warning'] = ('This image is only a visual overlay; board.npz was inherited from Dashboard V7. '
                           'Operator must compare physical square centers before any motion.')
    except Exception as ex:
        info['error'] = f'{type(ex).__name__}: {ex}'
    finally:
        if cap is not None:
            cap.release()
    return info


def main(argv=None):
    parser = argparse.ArgumentParser(description='MG400 actual on-site read-only preflight (NO MOTOR MOVEMENT)')
    parser.add_argument('--camera', action='store_true', help='open webcam and save an annotated 8x8 calibration preview')
    parser.add_argument('--timeout', type=float, default=3.0)
    args = parser.parse_args(argv)
    if not 0.2 <= args.timeout <= 20:
        parser.error('--timeout must be between 0.2 and 20 seconds')
    now = datetime.now().astimezone()
    tag = now.strftime('%Y%m%d_%H%M%S')
    folder = ROOT/'hardware_test_reports'
    folder.mkdir(exist_ok=True)
    result = {'test_type':'READ_ONLY_ON_SITE_PREFLIGHT','timestamp':now.isoformat(),
              'commands_sent':['RobotMode()','GetPose()'],
              'no_motion_or_outputs_commanded':True,
              'physical_robot_motion_verified':False,
              'safety_interlocks_verified':False,
              'physical_board_alignment_verified':False}
    print('\n=== MG400 REAL-HARDWARE READ-ONLY TEST ===')
    print('This test does NOT move robot, switch pump, enable robot or clear errors.')
    result['local_profile'] = audit_files()
    for name, chk in result['local_profile'].items():
        print(f"[{'PASS' if chk['ok'] else 'FAIL'}] FILE {name}: {chk['detail']}")
    try:
        ip=load_robot_ip()
        result['ip']=ip
        print(f'\nConnecting to MG400 at {ip} ...')
        probe=probe_mg400(ip,timeout=args.timeout)
        result['tcp']={'dashboard_29999':probe.dashboard_ok,'motion_30003_open_only':probe.motion_ok,
                       'robot_mode':probe.mode,'mode_description':MODE_DESCRIPTIONS.get(probe.mode,str(probe.mode)),
                       'connected':probe.connected,'details':probe.detail}
        print('[%s] NETWORK: %s' % ('PASS' if probe.connected else 'FAIL', probe.display()))
        if probe.dashboard_ok:
            try:
                pose=dashboard_readonly_get_pose(ip,args.timeout)
                result['pose']=pose
                print('[PASS] GetPose(): X={x}, Y={y}, Z={z}, R={r} (read only)'.format(**pose))
            except Exception as ex:
                result['pose_error']=f'{type(ex).__name__}: {ex}'
                print('[WARN] GetPose():',result['pose_error'])
    except Exception as ex:
        result['connection_error']=f'{type(ex).__name__}: {ex}'
        print('[FAIL] Connection config:', result['connection_error'])
    if args.camera:
        print('\nChecking camera and calibration overlay (NO robot movement) ...')
        result['camera']=camera_snapshot(folder,tag)
        cam=result['camera']
        print('[%s] CAMERA: %s' % ('PASS' if cam['ok'] else 'FAIL', cam.get('image',cam.get('error',''))))
        if cam['ok']:
            print('Detected pieces: black=%s, orange=%s; kings=%s' %
                  (cam['count_black'],cam['count_orange'],cam['kings']))
            print('Start layout matches V4:',cam['initial_setup_matches_v4'],
                  '| king stock seen:',cam['king_reserves_detected'])
    result['next_required_checks'] = [
        'Compare all projected board centers to physical squares; original board.npz is from Dashboard V7',
        'Trained operator must physically test E-stop and guarding',
        'Verify vacuum DI sensor pin, trigger polarity, grasp and release using safeguarded manual procedure',
        'Inspect pick/drop/stock paths and tool clearance with teach pendant at low speed',
        'Do NOT set live_safety_config.json flags until each corresponding on-site check passed',
    ]
    report_path = folder/f'mg400_readonly_{tag}.json'
    report_path.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print('\nSaved report:',report_path)
    network_ok = result.get('tcp',{}).get('connected',False)
    files_ok=all(x['ok'] for x in result['local_profile'].values())
    cam_ok=not args.camera or result.get('camera',{}).get('ok',False)
    print('RESULT:', ('PASS: TCP + local files; physical move NOT tested' if network_ok and files_ok and cam_ok
                       else 'INCOMPLETE/FAIL: see report; NO robot movement tested'))
    print('IMPORTANT: This does not clear safety locks or prove the game can start.')
    return 0 if network_ok and files_ok and cam_ok else 1


if __name__=='__main__':
    raise SystemExit(main())
