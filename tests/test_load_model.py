"""Independent physics/sign tests for the static payload calculation."""
import json
import unittest

import mujoco
import numpy as np

from cbf_demo.load_model import analyze_static_load, feasible_payload_interval
from cbf_demo.robot import RebotModel


class PayloadIntervalTests(unittest.TestCase):
    def test_signed_intersection_and_limiting_joint(self):
        result = feasible_payload_interval([.2, -.3], [1., -2.], [1.2, 1.7])
        self.assertTrue(result['feasible'])
        self.assertEqual(result['minimum_kg'], 0.)
        self.assertAlmostEqual(result['maximum_kg'], .7)
        self.assertEqual(result['limiting_upper_joint'], 'joint2')

    def test_counterweight_interval_is_not_claimed_as_zero_to_maximum(self):
        result = feasible_payload_interval([2.], [-1.], [1.])
        self.assertTrue(result['feasible'])
        self.assertEqual(result['minimum_kg'], 1.)
        self.assertEqual(result['maximum_kg'], 3.)
        self.assertFalse(result['zero_payload_feasible'])
        self.assertIn('counterweight_only', result['status'])
        self.assertFalse(result['hardware_payload_rating'])

    def test_constant_slope_infeasible_and_unbounded_cases(self):
        impossible = feasible_payload_interval([2.], [0.], [1.])
        self.assertFalse(impossible['feasible'])
        self.assertIsNone(impossible['maximum_kg'])
        self.assertFalse(impossible['maximum_unbounded'])
        self.assertEqual(impossible['constant_over_cap_joints'], ['joint1'])
        unbounded = feasible_payload_interval([.2], [0.], [1.])
        self.assertTrue(unbounded['feasible'])
        self.assertTrue(unbounded['maximum_unbounded'])
        self.assertIsNone(unbounded['maximum_kg'])
        json.dumps([impossible, unbounded], allow_nan=False)

    def test_disjoint_positive_intervals_and_negative_only_interval(self):
        self.assertFalse(feasible_payload_interval([2., 5.], [-1., -1.], [1., 1.])['feasible'])
        self.assertFalse(feasible_payload_interval([2.], [1.], [1.])['feasible'])

    def test_boundary_zero_cap_and_input_validation(self):
        result = feasible_payload_interval([0.], [1.], [0.])
        self.assertTrue(result['feasible'])
        self.assertEqual(result['minimum_kg'], result['maximum_kg'])
        for args in (([0.], [np.nan], [1.]), ([0.], [1.], [-1.]), ([0., 1.], [1.], [1.])):
            with self.assertRaises(ValueError):
                feasible_payload_interval(*args)


class StaticLoadPhysicsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.robot = RebotModel()

    def test_payload_torque_matches_potential_energy_derivative(self):
        robot = self.robot
        q = np.array([.4, -1.1, -1.4, -.4, .3, -.2])
        offset = np.array([.08, -.02, .04])
        mass, epsilon = .7, 1e-6
        report = analyze_static_load(robot, q, payload_kg=mass, com_offset=offset)
        expected = np.asarray(report['tau_payload_nm'])
        gravity = robot.model.opt.gravity.copy()
        site = mujoco.mj_name2id(robot.model, mujoco.mjtObj.mjOBJ_SITE, 'tcp')
        old = robot.get_q()
        def potential(configuration):
            robot.set_q(configuration)
            p = robot.data.site_xpos[site] + robot.data.site_xmat[site].reshape(3, 3) @ offset
            return -mass*np.dot(gravity, p)
        try:
            derivative = []
            for joint in range(6):
                delta = np.zeros(6); delta[joint] = epsilon
                derivative.append((potential(q+delta)-potential(q-delta))/(2*epsilon))
            np.testing.assert_allclose(expected, derivative, atol=2e-8, rtol=2e-7)
        finally:
            robot.set_q(old)

    def test_robot_gravity_torque_matches_original_inertial_energy(self):
        robot = self.robot
        q = np.array([-.6, -1.4, -1.1, -.5, .2, .3])
        report = analyze_static_load(robot, q)
        gravity = robot.model.opt.gravity.copy()
        old = robot.get_q()
        def potential(configuration):
            robot.set_q(configuration)
            return -np.sum(robot.model.body_mass * (robot.data.xipos @ gravity))
        try:
            derivative = []
            for joint in range(6):
                delta = np.zeros(6); delta[joint] = 1e-6
                derivative.append((potential(q+delta)-potential(q-delta))/2e-6)
            np.testing.assert_allclose(report['tau_robot_nm'], derivative, atol=2e-8, rtol=2e-7)
        finally:
            robot.set_q(old)

    def test_full_integration_state_and_gravity_preserved(self):
        robot = self.robot
        robot.set_q([-.3, -1.2, -1.5, -.2, .3, -.1])
        robot.data.qvel[:] = [.3, -.4, .2, .1, .5, -.2]
        robot.data.qfrc_applied[:] = [1., 2., 3., 4., 5., 6.]
        robot.data.xfrc_applied[-1] = [1., 2., 3., .1, .2, .3]
        robot.data.qacc_warmstart[:] = [.5, .6, .7, .8, .9, 1.]
        robot.data.time = 7.25
        mujoco.mj_forward(robot.model, robot.data)
        specification = mujoco.mjtState.mjSTATE_INTEGRATION
        before = np.empty(mujoco.mj_stateSize(robot.model, specification))
        after = np.empty_like(before)
        mujoco.mj_getState(robot.model, robot.data, before, specification)
        gravity = robot.model.opt.gravity.copy()
        report = analyze_static_load(robot, [.7, -1.4, -1.2, -.5, .4, .1], payload_kg=.4)
        mujoco.mj_getState(robot.model, robot.data, after, specification)
        np.testing.assert_array_equal(before, after)
        np.testing.assert_array_equal(gravity, robot.model.opt.gravity)
        self.assertFalse(report['physics_advanced'])
        json.dumps(report, allow_nan=False)

    def test_payload_scaling_headroom_and_real_joint_axes(self):
        q = [0., -1.1, -1.4, -.4, .3, 0.]
        first = analyze_static_load(self.robot, q, payload_kg=.2)
        second = analyze_static_load(self.robot, q, payload_kg=.4, headroom_fraction=.5)
        np.testing.assert_allclose(np.asarray(second['tau_payload_nm']), 2*np.asarray(first['tau_payload_nm']))
        np.testing.assert_allclose(second['caps_nm'], .5*self.robot.torque_limits)
        np.testing.assert_allclose(np.linalg.norm(second['joint_axes_world'], axis=1), 1., atol=1e-12)
        self.assertEqual(len(second['joint_origins_world_m']), 6)
        self.assertFalse(second['hardware_payload_rating'])
        self.assertEqual(second['joint_group_indices']['wrist'], [3, 4, 5])


if __name__ == '__main__':
    unittest.main()
