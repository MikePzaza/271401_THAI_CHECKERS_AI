"""Mock-only test proving a configured MG400 IP reaches the real runner path."""
import json
import sys
from pathlib import Path
import tempfile
import unittest
from unittest.mock import MagicMock, patch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from robot_io import LiveRunner, Transfer


class FakeCamera:
    verified_board = True
    def xy(self, cell):
        return (300.0 + cell * 0.1, 0.0)


class FakePlanner:
    camera = FakeCamera()
    has_kings = False
    z_hover = -80.0
    z_place = -145.0
    clear = (330.0, 0.0, -80.0, 0.0)


class RunnerConfiguredIPTests(unittest.TestCase):
    def test_motion_session_uses_chosen_ip_after_review(self):
        # There is no actual MG400 involved: all robot functions are mocked.
        approval = {
            'approved_for_this_setup': True,
            'emergency_stop_hardware_tested': True,
            'human_exclusion_guard_tested': True,
            'vacuum_sensor_tested': True,
            'board_alignment_physically_verified': True,
            'tool_clearance_tested': True,
            'vacuum_sensor_di': 1,
            'sensor_active_value': 1,
            'tool_r': 0.0,
        }
        with tempfile.TemporaryDirectory() as td:
            config = Path(td)/'test_only_approved.json'
            config.write_text(json.dumps(approval), encoding='utf-8')
            runner = LiveRunner(config)
            robot = MagicMock()
            sensor = iter(['0,{1},DI(1);', '0,{0},DI(1);'])
            def answer(cmd):
                if cmd == 'RobotMode()':
                    return '0,{5},RobotMode();'
                if cmd == 'DI(1)':
                    return next(sensor)
                return '0,{},Other();'
            robot.d.side_effect = answer
            with patch('robot_io.v.MG400', return_value=robot) as socket_connect, \
                 patch('robot_io.time.sleep', return_value=None):
                status = runner.run([Transfer(25, 20, 'piece')], FakePlanner(),
                                    stop=lambda: False, report=lambda msg: None,
                                    robot_ip='192.168.2.6')
        socket_connect.assert_called_once_with('192.168.2.6')
        self.assertIn('AWAIT CAMERA', status)
        self.assertTrue(robot.m.called)
        robot.close.assert_called_once()

if __name__=='__main__': unittest.main()
