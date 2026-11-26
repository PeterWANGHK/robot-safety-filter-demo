"""Local-QP behavior and independent all-pair acceptance checks."""
from types import SimpleNamespace
from unittest.mock import patch
import unittest

import numpy as np

from cbf_demo.distance_controller import RobotDistanceCBFFilter
from cbf_demo.robot import RebotModel
from cbf_demo.robot_field import pair_clearance_constraints
from cbf_demo.scenes import START, clutter_scene


def _affine_geometry(clearance=.06):
    gradients = np.zeros((50, 6)); gradients[0, 0] = 1.
    distances = np.full(50, 2.); distances[0] = clearance
    return {'valid': True, 'distances_m': distances, 'gradients_q': gradients}


class DistanceFilterQPTests(unittest.TestCase):
    def setUp(self):
        self.robot = SimpleNamespace(lower=np.full(6, -2.), upper=np.full(6, 2.))

    def test_near_pair_blocks_approach_but_preserves_away_command(self):
        safety = RobotDistanceCBFFilter(self.robot)
        with patch('cbf_demo.distance_controller.pair_clearance_constraints', return_value=_affine_geometry()):
            nominal = np.zeros(6); nominal[0] = -.4
            result = safety.filter(np.zeros(6), nominal)
            self.assertTrue(result.success)
            self.assertAlmostEqual(result.velocity[0], -.02, places=5)
            self.assertEqual(result.total_constraints, 50)
            self.assertLess(result.qp_constraints, 50)
            self.assertTrue(safety.last_diagnostics['all_pair_rows_postchecked'])
            nominal[0] = .3
            away = safety.filter(np.zeros(6), nominal)
            np.testing.assert_allclose(away.velocity, nominal, atol=2e-6)
            self.assertEqual(safety.barrier_units, 'm')

    def test_hard_velocity_and_joint_margin_bounds(self):
        safety = RobotDistanceCBFFilter(self.robot, joint_margin=.05)
        geometry = _affine_geometry(clearance=2.)
        with patch('cbf_demo.distance_controller.pair_clearance_constraints', return_value=geometry):
            q = np.zeros(6); q[1] = 1.94
            result = safety.filter(q, np.full(6, 5.))
            self.assertTrue(result.success)
            self.assertLessEqual(np.max(np.abs(result.velocity)), .450002)
            self.assertLessEqual(result.velocity[1], .020002)

    def test_contact_nonfinite_and_invalid_normals_fail_closed(self):
        safety = RobotDistanceCBFFilter(self.robot)
        self.assertFalse(safety.filter(np.zeros(6), np.full(6, np.nan)).success)
        geometry = _affine_geometry(); geometry['valid'] = False
        with patch('cbf_demo.distance_controller.pair_clearance_constraints', return_value=geometry):
            result = safety.filter(np.zeros(6), np.zeros(6))
            self.assertFalse(result.success)
            np.testing.assert_array_equal(result.velocity, np.zeros(6))
        for kwargs in ({'margin': np.nan}, {'speed': 0.}, {'alpha': -1.}):
            with self.assertRaises(ValueError):
                RobotDistanceCBFFilter(self.robot, **kwargs)

    def test_unsafe_initial_margin_is_reported_even_if_recovery_exists(self):
        safety = RobotDistanceCBFFilter(self.robot)
        with patch('cbf_demo.distance_controller.pair_clearance_constraints', return_value=_affine_geometry(.04)):
            result = safety.filter(np.zeros(6), np.zeros(6))
            self.assertTrue(result.success)
            self.assertGreaterEqual(result.velocity[0], .019998)
            self.assertFalse(safety.last_diagnostics['initial_margin_feasible'])
            self.assertFalse(safety.formal_safety_certificate)


class ActualRobotDistanceFilterTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.robot = RebotModel(obstacles=clutter_scene())

    def test_all_fifty_actual_pair_rows_and_gradient_acceptance(self):
        robot = self.robot
        safety = RobotDistanceCBFFilter(robot)
        nominal = np.array([.45, .1, -.1, .05, 0., 0.])
        result = safety.filter(START, nominal)
        self.assertTrue(result.success, result.status)
        self.assertEqual(result.total_constraints, 50)
        self.assertLessEqual(np.max(np.abs(result.velocity)), .450002)
        geometry = pair_clearance_constraints(robot, START)
        original_residual = geometry['gradients_q']@result.velocity + 2*(geometry['distances_m']-.05)
        self.assertGreaterEqual(float(original_residual.min()), -2e-6)
        # Independent centred difference checks the geometry that generated
        # the actual 50 hard rows, not a duplicate QP implementation.
        for joint in range(6):
            delta = np.zeros(6); delta[joint] = 1e-6
            plus = pair_clearance_constraints(robot, START+delta)['distances_m']
            minus = pair_clearance_constraints(robot, START-delta)['distances_m']
            np.testing.assert_allclose((plus-minus)/2e-6, geometry['gradients_q'][:, joint],
                                       atol=2e-5, rtol=2e-4)


if __name__ == '__main__':
    unittest.main()
