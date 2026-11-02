"""Offline tests: the connection diagnostic must NEVER drive the MG400."""
import sys
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from robot_connection import (probe_mg400, load_robot_ip, save_robot_ip,
                              validate_ip, _robot_mode_reply)


class FakeSocket:
    def __init__(self, incoming=b'0,{5},RobotMode();'):
        self.incoming = incoming
        self.sent = []
        self.timeout = None
        self.closed = False

    def __enter__(self): return self
    def __exit__(self, *args): self.closed = True
    def settimeout(self, value): self.timeout = value
    def sendall(self, data): self.sent.append(data)
    def recv(self, size):
        data, self.incoming = self.incoming, b''
        return data


class RobotPreflightTests(unittest.TestCase):
    def test_read_only_both_tcp_ports_and_mode(self):
        dashboard, motion = FakeSocket(), FakeSocket()
        with patch('robot_connection.socket.create_connection', side_effect=[dashboard, motion]) as connect:
            result = probe_mg400('192.168.1.6')
        self.assertTrue(result.connected)
        self.assertTrue(result.available_for_game)
        self.assertEqual(result.mode, 5)
        self.assertEqual(dashboard.sent, [b'RobotMode()'])
        self.assertEqual(motion.sent, [])
        self.assertTrue(dashboard.closed and motion.closed)
        self.assertEqual([x.args[0][1] for x in connect.call_args_list], [29999, 30003])

    def test_connected_alarm_still_not_ready(self):
        with patch('robot_connection.socket.create_connection', side_effect=[FakeSocket(b'0,{9},RobotMode();'), FakeSocket()]):
            probe = probe_mg400('192.168.1.6')
        self.assertTrue(probe.connected)
        self.assertFalse(probe.available_for_game)
        self.assertIn('Alarm', probe.display())

    def test_connected_disabled_explicitly_distinguished(self):
        with patch('robot_connection.socket.create_connection', side_effect=[FakeSocket(b'0,{4},RobotMode();'), FakeSocket()]):
            probe = probe_mg400('192.168.1.6')
        self.assertTrue(probe.connected)
        self.assertTrue(probe.available_for_game)  # runner's later EnableRobot() requires safety approval
        self.assertIn('ยังไม่ Enable', probe.display())

    def test_port_does_not_respond(self):
        with patch('robot_connection.socket.create_connection', side_effect=TimeoutError('timeout')):
            probe = probe_mg400('192.168.1.6')
        self.assertFalse(probe.connected)
        self.assertFalse(probe.available_for_game)
        self.assertIn('29999', probe.display())

    def test_motion_port_inaccessible(self):
        with patch('robot_connection.socket.create_connection', side_effect=[FakeSocket(), OSError('refused')]):
            probe = probe_mg400('192.168.1.6')
        self.assertTrue(probe.dashboard_ok)
        self.assertFalse(probe.motion_ok)
        self.assertFalse(probe.connected)
        self.assertIn('30003', probe.display())

    def test_malformed_response_rejected(self):
        with self.assertRaises(ValueError):
            _robot_mode_reply('0,{},RobotMode();')
        with patch('robot_connection.socket.create_connection', return_value=FakeSocket(b'1,{5},RobotMode();')) as connect:
            probe = probe_mg400('192.168.1.6')
        self.assertFalse(probe.connected)
        self.assertEqual(connect.call_count, 1)

    def test_config_roundtrip_and_invalid_ip(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'connection.json'
            self.assertEqual(save_robot_ip('192.168.2.6', path), '192.168.2.6')
            self.assertEqual(load_robot_ip(path), '192.168.2.6')
            with self.assertRaises(ValueError):
                save_robot_ip('not-an-ip', path)
            self.assertEqual(load_robot_ip(path), '192.168.2.6')
            path.write_text('{"ip":"bad"}', encoding='utf-8')
            with self.assertRaises(ValueError):
                load_robot_ip(path)
            with self.assertRaises(ValueError):
                validate_ip('example.com')

if __name__ == '__main__': unittest.main()
