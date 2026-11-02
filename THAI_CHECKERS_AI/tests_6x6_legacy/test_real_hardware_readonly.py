"""Unit tests for the *read-only* actual-MG400 preflight utility."""
import unittest
import socket
from unittest.mock import patch
import TEST_REAL_MG400_READONLY as real


class FakeSocket:
    def __init__(self, reply=b'0,{290.1,10.2,-101.9,0.0},GetPose();'):
        self.reply=reply
        self.calls=[]
    def __enter__(self): return self
    def __exit__(self, *args): return False
    def settimeout(self, value): pass
    def sendall(self, value): self.calls.append(value)
    def recv(self, size):
        data,self.reply=self.reply,b''
        return data


class RealHardwareReadonly(unittest.TestCase):
    def test_calibration_schema(self):
        self.assertTrue(all(v['ok'] for v in real.audit_files().values()))

    def test_readonly_pose_command(self):
        s=FakeSocket()
        with patch.object(real.socket,'create_connection',return_value=s):
            p=real.dashboard_readonly_get_pose('127.0.0.1')
        self.assertEqual(s.calls,[b'GetPose()'])
        self.assertEqual(p['x'],290.1)
        self.assertEqual(p['z'],-101.9)

    def test_reject_bad_pose(self):
        s=FakeSocket(b'0,{nan,1,2,3},GetPose();')
        with patch.object(real.socket,'create_connection',return_value=s):
            with self.assertRaises(ValueError):
                real.dashboard_readonly_get_pose('127.0.0.1')


if __name__=='__main__': unittest.main()
