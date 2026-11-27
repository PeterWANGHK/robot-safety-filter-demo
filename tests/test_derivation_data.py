"""Independent local derivatives and state/provenance checks for field math."""
from copy import deepcopy
import json
import unittest

import mujoco
import numpy as np

from cbf_demo.command_comparison import build_command_comparison
from cbf_demo.derivation_data import build_derivation_data
from cbf_demo.load_model import _preserve_integration_state
from cbf_demo.robot import RebotModel
from cbf_demo.robot_field import pair_clearance_constraints
from cbf_demo.scenes import SURFACE_EPSILON, build_field, clutter_scene


class DerivationDataTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.robot = RebotModel(clutter_scene(), epsilon=SURFACE_EPSILON)
        cls.field = build_field(clutter_scene())
        cls.q = np.array([-1.17, -1.13, -1.37, -.43, .31, .17])
        cls.frames = [(1.25, cls.q.copy())]
        cls.commands = build_command_comparison(cls.robot, cls.field, cls.frames)
        cls.robot.set_q(cls.q)
        cls.robot.data.qvel[:] = [.2, -.1, .3, .1, .2, -.1]
        cls.robot.data.qfrc_applied[:] = [1., 2., 3., 4., 5., 6.]
        cls.robot.data.qacc_warmstart[:] = np.arange(6)/10
        cls.robot.data.time = 7.25
        mujoco.mj_forward(cls.robot.model, cls.robot.data)
        cls.state_spec = mujoco.mjtState.mjSTATE_INTEGRATION
        cls.before = np.empty(mujoco.mj_stateSize(cls.robot.model, cls.state_spec))
        mujoco.mj_getState(cls.robot.model, cls.robot.data, cls.before, cls.state_spec)
        cls.h_before = cls.field.h.copy()
        cls.data = build_derivation_data(cls.robot, cls.field, cls.frames, command_data=cls.commands)
        cls.after = np.empty_like(cls.before)
        mujoco.mj_getState(cls.robot.model, cls.robot.data, cls.after, cls.state_spec)

    def test_selected_poisson_chain_rule_matches_fixed_point_finite_difference(self):
        robot, field = self.robot, self.field
        velocity = np.array([.11, -.17, .13, .07, -.09, .05])
        with _preserve_integration_state(robot.model, robot.data):
            for name, region in self.data['frames'][0]['regions'].items():
                row = region['poisson']
                self.assertTrue(row['field_valid'], name)
                self.assertGreater(row['h_m2'], 0)
                self.assertNotEqual(row['body_name'], 'base_link')
                jacobian, gradient = np.asarray(row['jacobian_m_per_rad']), np.asarray(row['gradient_world_m'])
                a = np.asarray(row['a_m2_per_rad'])
                np.testing.assert_allclose(a, gradient@jacobian, atol=1e-15)
                index, epsilon = row['sample_index'], 1e-6
                plus = robot.points(self.q+epsilon*velocity)[index]
                minus = robot.points(self.q-epsilon*velocity)[index]
                np.testing.assert_allclose((plus-minus)/(2*epsilon), jacobian@velocity, atol=1e-8)
                hplus = field.sample(plus)[0][0]
                hminus = field.sample(minus)[0][0]
                self.assertAlmostEqual((hplus-hminus)/(2*epsilon), float(a@velocity), places=8)

    def test_all_pair_witnesses_gradients_and_region_choices_are_actual(self):
        frame = self.data['frames'][0]
        pairs = frame['pairs']
        self.assertEqual(len(pairs['distances_m']), 50)
        self.assertTrue(pairs['valid'])
        reference = pair_clearance_constraints(self.robot, self.q)
        np.testing.assert_allclose(pairs['distances_m'], reference['distances_m'], atol=1e-14)
        np.testing.assert_allclose(pairs['gradients_q'], reference['gradients_q'], atol=1e-14)
        for key, region in frame['regions'].items():
            candidates = np.asarray(region['pair_indices'])
            selected = region['pair_index']
            self.assertEqual(selected, int(candidates[np.argmin(np.asarray(pairs['distances_m'])[candidates])]))
            allowed = self.data['regions'][key]['bodies']
            if allowed is not None:
                self.assertIn(pairs['robot_bodies'][selected], allowed)
        np.testing.assert_allclose(np.linalg.norm(pairs['normal_obstacle_to_robot_world'], axis=1), 1., atol=1e-8)
        self.assertEqual(len(frame['boxes']), 10)
        self.assertEqual(len(self.data['obstacles']), 5)
        for original, packed in zip(self.robot.obstacles, self.data['obstacles']):
            self.assertEqual(packed['name'], original['name'])
            np.testing.assert_array_equal(packed['center_m'], original['center'])
            np.testing.assert_array_equal(packed['half_size_m'], original['half_size'])

    def test_raw_minima_use_every_original_geometry_row(self):
        robot = self.robot
        with _preserve_integration_state(robot.model, robot.data):
            points, jacobians = robot.points_and_jacobians(self.q)
            h, gradients = self.field.sample(points)
            a = np.einsum('ni,nij->nj', gradients, jacobians)
        frame = self.data['frames'][0]
        pair_a, distance = np.asarray(frame['pairs']['gradients_q']), np.asarray(frame['pairs']['distances_m'])
        alpha = self.data['controller']['alpha_per_s']
        reserve = self.data['controller']['poisson_h_reserve_m2']
        self.assertEqual(len(a), 798)
        self.assertEqual(len(frame['command_residuals']), 13)
        for command, diagnostics in zip(self.commands['frames'][0]['commands'], frame['command_residuals']):
            nominal = np.asarray(command['nominal_rad_s'])
            self.assertAlmostEqual(diagnostics['nominal_poisson_min_m2_s'], float(np.min(a@nominal+alpha*(h-reserve))), places=14)
            if command['poisson']['success']:
                projected = np.asarray(command['poisson']['velocity_rad_s'])
                self.assertAlmostEqual(diagnostics['poisson_min_m2_s'], float(np.min(a@projected+alpha*(h-reserve))), places=14)
            for index, margin in enumerate(self.data['margins_m']):
                self.assertAlmostEqual(diagnostics['nominal_pair_min_m_s'][index],
                                       float(np.min(pair_a@nominal+alpha*(distance-margin))), places=14)
                if command['distance'][index]['success']:
                    projected = np.asarray(command['distance'][index]['velocity_rad_s'])
                    self.assertAlmostEqual(diagnostics['distance_min_m_s'][index],
                                           float(np.min(pair_a@projected+alpha*(distance-margin))), places=14)

    def test_full_integration_and_field_state_preserved_and_json_finite(self):
        np.testing.assert_array_equal(self.after, self.before)
        np.testing.assert_array_equal(self.field.h, self.h_before)
        json.dumps(self.data, allow_nan=False)
        self.assertFalse(self.data['assumptions']['physics_advanced'])
        self.assertFalse(self.data['assumptions']['controllers_solved'])
        self.assertEqual(self.data['assumptions']['all_poisson_controller_rows_retained'], 798)
        self.assertTrue(self.data['raw_residuals_from_precomputed_commands'])

    def test_rejected_commands_get_no_successful_projection_residual(self):
        commands = deepcopy(self.commands)
        commands['frames'][0]['commands'][0]['poisson']['success'] = False
        commands['frames'][0]['commands'][0]['distance'][1]['success'] = False
        data = build_derivation_data(self.robot, self.field, self.frames, command_data=commands)
        diagnostics = data['frames'][0]['command_residuals'][0]
        self.assertIsNone(diagnostics['poisson_min_m2_s'])
        self.assertIsNone(diagnostics['distance_min_m_s'][1])
        self.assertIsNotNone(diagnostics['nominal_poisson_min_m2_s'])

    def test_mismatched_precomputed_pose_margin_and_time_rejected(self):
        for key, value in [('time_s', 2.), ('q_rad', [0., -1., -1., 0., 0., 0.])]:
            commands = deepcopy(self.commands)
            commands['frames'][0][key] = value
            with self.assertRaises(ValueError):
                build_derivation_data(self.robot, self.field, self.frames, command_data=commands)
        with self.assertRaises(ValueError):
            build_derivation_data(self.robot, self.field, self.frames, margins=(.05,), command_data=self.commands)
        with self.assertRaises(ValueError):
            build_derivation_data(self.robot, self.field, [])


if __name__ == '__main__':
    unittest.main()
