"""Read-only local calibration audit. Never opens MG400 socket/camera."""
from pathlib import Path
import hashlib
import numpy as np
import mg400_visionV4 as v
ROOT = Path(__file__).resolve().parent
SCHEMA = {'calib.npz': {'H': (3,3), 'z_plane': (), 'rot180': ()},
          'board.npz': {'A': (2,3)},
          'clear_pos.npz': {'pos': (4,)},
          'discard_pos.npz': {'pos': (4,)},
          'roi.npz': {'roi': (4,)},
          'stock_pos.npz': {'orange_xy': (4,2), 'black_xy': (4,2),
                            'orange_uv': (4,2), 'black_uv': (4,2)}}

def main():
    print('MG400 V4 - Offline profile audit (NO robot commands)')
    fail=[]
    for name, fields in SCHEMA.items():
        file=ROOT/name
        if not file.is_file():
            print('[MISSING]',name);fail.append(name);continue
        try:
            with np.load(file,allow_pickle=False) as saved:
                for field, shape in fields.items():
                    if field not in saved or saved[field].shape != shape or not np.isfinite(saved[field]).all():
                        raise ValueError(f'{field} invalid/shape != {shape}')
            print('[OK]',name, 'sha256',hashlib.sha256(file.read_bytes()).hexdigest()[:12])
        except Exception as exc:
            print('[ERROR]',name,exc);fail.append(name)
    if not fail:
        cal=np.load(ROOT/'calib.npz',allow_pickle=False)
        rot=bool(cal['rot180'])
        if rot != v.ROTATE_180:
            fail.append('ROTATE_180');print('[ERROR] rotation mismatch')
        d=v.load_discard()
        print('Board calibration: existing board.npz from V7; verify physically before REAL')
        print('Discard outside board bounds:',not v.in_range(*d[:2]),
              '| inside V4 approved reach:',v.stock_reach_ok(*d[:2]))
        if not v.stock_reach_ok(*d[:2]):fail.append('discard reach')
    print('RESULT:', 'CALIBRATION FILES PASS (NOT HARDWARE APPROVAL)' if not fail else 'FAILED: '+', '.join(fail))
    return not fail

if __name__=='__main__':
    raise SystemExit(0 if main() else 1)
