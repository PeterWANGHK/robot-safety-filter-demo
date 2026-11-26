"""Geometry checks deliberately independent of the CBF field/controller."""

import unittest

import mujoco
import numpy as np

from cbf_demo.robot import RebotModel, _box_surface


class RobotTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.robot = RebotModel(epsilon=0.035)

    def test_proxies_enclose_every_original_visual_vertex(self):
        robot = self.robot
        self.assertEqual({box.body_name for box in robot.proxy_boxes},
                         {"base_link", "link1", "link2", "link3", "link4", "link5", "link6", "end_link", "finger_left_link", "finger_right_link"})
        for box in robot.proxy_boxes:
            vertices = robot.visual_vertices[box.body_name]
            self.assertTrue(np.all(vertices >= box.center - box.half_size - 1e-12), box.body_name)
            self.assertTrue(np.all(vertices <= box.center + box.half_size + 1e-12), box.body_name)
        self.assertLessEqual(robot.coverage_radius, robot.epsilon + 1e-12)

    def test_compiled_meshes_are_inside_same_body_proxies(self):
        # Reconstruct compiled mesh coordinates using each geom's compiled pose.
        # This independently detects STL/body-frame and mesh canonicalization errors.
        model = self.robot.model
        for box in self.robot.proxy_boxes:
            body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, box.body_name)
            for geom_id in range(model.ngeom):
                name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom_id) or ""
                if model.geom_bodyid[geom_id] != body_id or not name.endswith("_visual"):
                    continue
                mesh_id = model.geom_dataid[geom_id]
                begin = model.mesh_vertadr[mesh_id]
                vertices = model.mesh_vert[begin: begin + model.mesh_vertnum[mesh_id]]
                matrix = np.empty(9)
                mujoco.mju_quat2Mat(matrix, model.geom_quat[geom_id])
                local = vertices @ matrix.reshape(3, 3).T + model.geom_pos[geom_id]
                self.assertTrue(np.all(np.abs(local - box.center) <= box.half_size + 1e-6), name)

    def test_surface_cover_empirically(self):
        random = np.random.default_rng(134)
        for box in self.robot.proxy_boxes:
            samples, radius = _box_surface(box.center, box.half_size, self.robot.epsilon)
            for axis in range(3):
                points = random.uniform(-1.0, 1.0, (100, 3)) * box.half_size
                points[:, axis] = box.half_size[axis]
                points += box.center
                minimum = np.linalg.norm(points[:, None, :] - samples[None, :, :], axis=2).min(axis=1)
                self.assertLessEqual(float(minimum.max()), radius + 1e-9)

    def test_surface_and_tcp_jacobians_against_finite_difference(self):
        robot = self.robot
        q = np.array([0.2, -1.3, -1.1, -0.3, 0.4, -0.2])
        points, jacobian = robot.points_and_jacobians(q)
        tcp_jacobian = robot.tcp_jacobian()
        epsilon = 1e-6
        for joint in range(6):
            offset = np.zeros(6)
            offset[joint] = epsilon
            plus, _ = robot.points_and_jacobians(q + offset)
            tcp_plus = robot.tcp()
            minus, _ = robot.points_and_jacobians(q - offset)
            tcp_minus = robot.tcp()
            np.testing.assert_allclose((plus - minus) / (2 * epsilon), jacobian[:, :, joint], atol=2e-7, rtol=2e-6)
            np.testing.assert_allclose((tcp_plus - tcp_minus) / (2 * epsilon), tcp_jacobian[:, joint], atol=2e-7, rtol=2e-6)
        robot.set_q(q)
        self.assertTrue(np.all(np.isfinite(points)))

    def test_exact_box_clearance_detects_overlap(self):
        robot = RebotModel(obstacles=[{"name": "base_overlap", "center": [0.0, 0.0, 0.03], "half_size": [0.02, 0.02, 0.02]}])
        self.assertLess(robot.clearance(), 0.0)
        # Static base and static world geoms have the same weld, so MuJoCo does
        # not make a dynamic contact for that pair.  Signed distance still must
        # detect the overlap (including configurations rejected before motion).

    def test_pd_torque_caps_and_finite_dynamics(self):
        robot = self.robot
        q = np.array([0.2, -1.3, -1.1, -0.3, 0.4, -0.2])
        robot.set_q(q)
        robot.data.qvel[:] = 0.0
        torque = robot.step(q + 3.0, np.zeros(6), substeps=5)
        self.assertTrue(np.all(np.abs(torque) <= robot.torque_limits + 1e-12))
        self.assertTrue(np.all(np.isfinite(robot.get_q())))
        self.assertEqual(robot.model.nq, 6)
        self.assertEqual(robot.model.nv, 6)

    def test_physics_substep_aggregates_match_independent_ticks(self):
        # A slower controller must not lose a short torque peak or a minimum
        # distance between its ticks. Compare its batch to separate physics
        # ticks, which expose every state independently to the test.
        obstacles = [{"name": "front", "center": [0.34, 0.0, 0.45], "half_size": [0.02, 0.02, 0.08]}]
        batch, reference = RebotModel(obstacles), RebotModel(obstacles)
        q = np.array([-0.8, -1.1, -1.4, -0.4, 0.3, 0.0])
        target, velocity = q + np.array([0.05, 0.01, 0.0, 0.0, 0.0, 0.0]), np.array([0.2, 0.1, 0.0, 0.0, 0.0, 0.0])
        batch.set_q(q); reference.set_q(q)
        batch.step(target, velocity, substeps=20)
        initial = reference.clearance()
        distances, contacts, torques = [initial], [], []
        for _ in range(20):
            torques.append(reference.step(target, velocity))
            distances.append(reference.clearance())
            contacts.append(reference.environment_contacts())
        torques = np.asarray(torques)
        np.testing.assert_allclose(batch.get_q(), reference.get_q(), atol=1e-13)
        np.testing.assert_allclose(batch.last_step_peak_torque, np.max(np.abs(torques), axis=0), atol=1e-12)
        np.testing.assert_allclose(batch.last_step_torque_sumsq, np.sum(torques**2, axis=0), atol=1e-12)
        self.assertAlmostEqual(batch.last_step_min_clearance, min(distances), places=12)
        self.assertEqual(batch.last_step_max_contacts, max(contacts))
        self.assertEqual(batch.last_step_substeps, 20)


if __name__ == "__main__":
    unittest.main()
