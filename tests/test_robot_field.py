"""Analytic distance, articulated-frame and closest-pair derivative checks."""
import unittest

import mujoco
import numpy as np

from cbf_demo.robot import RebotModel
from cbf_demo.robot_field import (world_proxy_boxes, sample_robot_distance,
                                   build_robot_distance_grid, pair_clearance_constraints)


class RobotDistanceTests(unittest.TestCase):
    def test_rotated_box_far_surface_and_interior(self):
        rotation = np.array([[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]])
        box = dict(body_name='rotated', center=np.array([.3, -.2, .4]),
                   half_size=np.array([.1, .2, .3]), rotation=rotation)
        local = np.array([[.3, .4, 0.], [.1, .04, .02], [.03, 0., 0.]])
        points = local @ rotation.T + box['center']
        distance, body, gradient = sample_robot_distance(points, [box])
        np.testing.assert_allclose(distance, [np.sqrt(.08), 0., -.07], atol=1e-12)
        self.assertEqual(body.tolist(), ['rotated']*3)
        np.testing.assert_allclose(gradient[0], np.array([1., 1., 0.])/np.sqrt(2) @ rotation.T)

    def test_world_gradient_matches_finite_differences_away_from_ties(self):
        angle = .43
        rotation = np.array([[np.cos(angle), -np.sin(angle), 0.],
                             [np.sin(angle), np.cos(angle), 0.], [0., 0., 1.]])
        boxes = [dict(body_name='a', center=[.2, -.1, .3], half_size=[.1, .2, .15], rotation=rotation)]
        points = np.array([[.5, .6, .7], [.23, -.11, .31]])
        gradient = sample_robot_distance(points, boxes)[2]
        for axis in range(3):
            delta = np.zeros(3); delta[axis] = 1e-6
            derivative = (sample_robot_distance(points+delta, boxes)[0]
                          -sample_robot_distance(points-delta, boxes)[0])/2e-6
            np.testing.assert_allclose(derivative, gradient[:, axis], atol=1e-9)

    def test_union_minimum_and_overlap_interior_semantics(self):
        boxes = [dict(body_name='a', center=[-.05, 0., 0.], half_size=[.1, .2, .3]),
                 dict(body_name='b', center=[.05, 0., 0.], half_size=[.1, .2, .3])]
        distance, body, _ = sample_robot_distance([[.3, 0., 0.], [0., 0., 0.]], boxes)
        np.testing.assert_allclose(distance, [.15, -.05])
        self.assertEqual(body[0], 'b')
        # True union boundary distance at the interior point is .15, whereas
        # minimum individual signed distance is -.05: deliberately documented.
        self.assertNotAlmostEqual(distance[1], -.15)
        grid = build_robot_distance_grid(boxes, spacing=.05, padding=.1)
        self.assertFalse(grid['inside_overlap_is_true_union_distance'])
        touching = [dict(body_name='left', center=[-.1, 0., 0.], half_size=[.1, .2, .3]),
                    dict(body_name='right', center=[.1, 0., 0.], half_size=[.1, .2, .3])]
        # Their shared face is interior to the union but min signed-box
        # distance is zero there. Positive exterior envelopes remain valid.
        self.assertEqual(sample_robot_distance([0., 0., 0.], touching)[0][0], 0.)


class ArticulatedRobotDistanceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.robot = RebotModel(obstacles=[dict(name='post', center=[.42, .19, .42], half_size=[.025, .04, .09])])

    def test_robot_queries_preserve_full_integration_state(self):
        robot = self.robot
        robot.set_q([-.8, -1.1, -1.4, -.4, .3, 0.])
        robot.data.qvel[:] = [.2, -.1, .3, .2, .1, -.2]
        robot.data.qfrc_applied[:] = [1., 2., 3., 4., 5., 6.]
        robot.data.time = 5.5
        mujoco.mj_forward(robot.model, robot.data)
        spec = mujoco.mjtState.mjSTATE_INTEGRATION
        before = np.empty(mujoco.mj_stateSize(robot.model, spec)); after = np.empty_like(before)
        mujoco.mj_getState(robot.model, robot.data, before, spec)
        q = [.7, -1.2, -1.3, -.3, .2, .1]
        boxes = world_proxy_boxes(robot, q)
        self.assertEqual(len(boxes), 10)
        pair_clearance_constraints(robot, q)
        mujoco.mj_getState(robot.model, robot.data, after, spec)
        np.testing.assert_array_equal(after, before)

    def test_positive_pair_witness_sign_and_joint_gradient(self):
        robot = self.robot
        q = np.array([-.8, -1.1, -1.4, -.4, .3, .1])
        result = pair_clearance_constraints(robot, q)
        self.assertTrue(result['valid'])
        difference = result['robot_witness_world_m']-result['obstacle_witness_world_m']
        np.testing.assert_allclose(np.linalg.norm(difference, axis=1), result['distances_m'], atol=1e-8)
        np.testing.assert_allclose(difference/result['distances_m'][:, None],
                                   result['normal_obstacle_to_robot_world'], atol=1e-8)
        for joint in range(6):
            delta = np.zeros(6); delta[joint] = 1e-6
            plus = pair_clearance_constraints(robot, q+delta)['distances_m']
            minus = pair_clearance_constraints(robot, q-delta)['distances_m']
            np.testing.assert_allclose((plus-minus)/2e-6, result['gradients_q'][:, joint],
                                       atol=2e-5, rtol=2e-4)

    def test_near_contact_pair_is_rejected(self):
        robot = RebotModel(obstacles=[dict(name='base_overlap', center=[0., 0., .03], half_size=[.02, .02, .02])])
        result = pair_clearance_constraints(robot)
        self.assertFalse(result['valid'])
        self.assertTrue(np.any(result['distances_m'] < 0))
        self.assertFalse(result['formal_safety_certificate'])


if __name__ == '__main__':
    unittest.main()
