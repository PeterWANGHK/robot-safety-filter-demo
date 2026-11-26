"""Semantic checks for world geometry and non-executing visualization helpers."""
import unittest

import mujoco
import numpy as np

from cbf_demo.field_plots import box_mesh, robot_mesh, _tcp_route
from cbf_demo.robot import RebotModel


class FieldPlotGeometryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.robot = RebotModel()

    def test_oriented_box_corners_and_outward_triangle_winding(self):
        center = np.array([.35, -.21, .47])
        half_size = np.array([.1, .2, .3])
        rotation = np.array([[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]])
        vertices, faces = box_mesh(center, half_size, rotation)
        np.testing.assert_allclose(vertices.min(axis=0), center-[.2, .1, .3])
        np.testing.assert_allclose(vertices.max(axis=0), center+[.2, .1, .3])
        self.assertEqual(vertices.shape, (8, 3))
        self.assertEqual(faces.shape, (12, 3))
        triangles = vertices[faces]
        normals = np.cross(triangles[:, 1]-triangles[:, 0], triangles[:, 2]-triangles[:, 0])
        self.assertTrue(np.all(np.einsum('ij,ij->i', normals, triangles.mean(axis=1)-center) > 0))

    def test_arm_mesh_matches_compiled_collision_geometry_in_world(self):
        # MuJoCo's compiled geom pose provides an independent transform chain;
        # asymmetric q catches transposition, link-frame and joint-sign errors.
        robot = self.robot
        q = np.array([.63, -1.17, -1.42, -.23, .41, -.37])
        vertices, faces = robot_mesh(robot, q)
        previous = robot.get_q()
        try:
            robot.set_q(q)
            signs = np.array([[x, y, z] for x in (-1, 1)
                              for y in (-1, 1) for z in (-1, 1)])
            for index, geom_id in enumerate(robot.proxy_geom_ids):
                center = robot.data.geom_xpos[geom_id]
                rotation = robot.data.geom_xmat[geom_id].reshape(3, 3)
                expected = (signs*robot.model.geom_size[geom_id]) @ rotation.T + center
                np.testing.assert_allclose(vertices[index*8:(index+1)*8], expected, atol=1e-12)
            self.assertTrue(np.all(faces >= 0))
            self.assertLess(int(faces.max()), len(vertices))
        finally:
            robot.set_q(previous)

    def test_pose_and_route_queries_preserve_integration_state(self):
        robot = self.robot
        robot.set_q([-.7, -1.2, -1.4, -.3, .2, .1])
        robot.data.qvel[:] = [.1, -.2, .1, 0., .2, .1]
        robot.data.qfrc_applied[:] = [1., 2., 3., 4., 5., 6.]
        robot.data.time = 4.25
        mujoco.mj_forward(robot.model, robot.data)
        spec = mujoco.mjtState.mjSTATE_INTEGRATION
        before = np.empty(mujoco.mj_stateSize(robot.model, spec))
        after = np.empty_like(before)
        mujoco.mj_getState(robot.model, robot.data, before, spec)
        poses = np.array([[.6, -1.1, -1.3, -.2, .3, .1],
                          [-.5, -1.4, -1.2, -.4, .1, -.2]])
        robot_mesh(robot, poses[0])
        route = _tcp_route(robot, poses)
        mujoco.mj_getState(robot.model, robot.data, after, spec)
        np.testing.assert_array_equal(after, before)
        self.assertEqual(route.shape, (2, 3))
        self.assertTrue(np.all(np.isfinite(route)))
        self.assertGreater(np.linalg.norm(route[1]-route[0]), .01)


if __name__ == '__main__':
    unittest.main()
