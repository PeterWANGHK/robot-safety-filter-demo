"""Official appearance provenance, independent FK and geometry invariance."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

from cbf_demo.load_model import _preserve_integration_state
from cbf_demo.robot import RebotModel
from cbf_demo.scenes import START, GOAL
from cbf_demo.visual_model import (SEEED_COMMIT, URDF_RELATIVE, URDFKinematics,
                                   build_visual_model_data, read_stl_mesh,
                                   rpy_matrix, transform_visual_vertices)


class VisualTransformTests(unittest.TestCase):
    def test_visual_scale_precedes_origin_rotation_and_translation(self):
        source = np.array([[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]])
        rendered = transform_visual_vertices(source, scale=(2., 3., 4.), xyz=(.1, .2, .3), rpy=(0., 0., np.pi/2))
        expected = np.array([[.1, 2.2, .3], [-2.9, .2, .3], [.1, .2, 4.3]])
        np.testing.assert_allclose(rendered, expected, atol=1e-15)

    def test_urdf_fixed_axis_rpy_and_prismatic_mimic(self):
        text = '''<robot name="test"><link name="base"/><link name="hinge"/><link name="left"/><link name="right"/>
        <joint name="turn" type="revolute"><parent link="base"/><child link="hinge"/>
        <origin xyz=".1 .2 .3" rpy=".2 .3 .4"/><axis xyz="0 0 1"/></joint>
        <joint name="slide" type="prismatic"><parent link="hinge"/><child link="left"/><axis xyz="0 1 0"/></joint>
        <joint name="mirror" type="prismatic"><parent link="hinge"/><child link="right"/><axis xyz="0 1 0"/>
        <mimic joint="slide" multiplier="-1" offset="0"/></joint></robot>'''
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary)/'test.urdf'; path.write_text(text)
            poses = URDFKinematics(path).forward({'turn': .6, 'slide': .02})
        expected_rotation = rpy_matrix((.2, .3, .4))@rpy_matrix((0., 0., .6))
        np.testing.assert_allclose(poses['hinge'][:3, :3], expected_rotation, atol=1e-15)
        np.testing.assert_allclose(poses['left'][:3, 3]-poses['hinge'][:3, 3], expected_rotation@np.array([0., .02, 0.]), atol=1e-15)
        np.testing.assert_allclose(poses['right'][:3, 3]-poses['hinge'][:3, 3], expected_rotation@np.array([0., -.02, 0.]), atol=1e-15)


class OfficialVisualModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.robot = RebotModel()
        cls.repo = Path(__file__).resolve().parents[3]/'rebotarm_ros2'
        if not (cls.repo/URDF_RELATIVE).is_file():
            # Public source checkouts can set the official source location.
            import os
            cls.repo = Path(os.environ.get('SEEED_REBOT_ROOT', cls.repo))
        if not (cls.repo/URDF_RELATIVE).is_file():
            raise unittest.SkipTest('Pinned official Seeed DM description checkout is required')
        cls.frames = [(0., START), (7.3, np.array([.43, -1.23, -1.41, -.27, .32, -.19])), (16.6, GOAL)]
        cls.robot.set_q(START)
        cls.robot.data.qvel[:] = [.2, -.1, .3, .1, .2, -.1]
        cls.robot.data.qfrc_applied[:] = np.arange(6)+1
        cls.robot.data.time = 12.5
        mujoco.mj_forward(cls.robot.model, cls.robot.data)
        cls.spec = mujoco.mjtState.mjSTATE_INTEGRATION
        cls.before = np.empty(mujoco.mj_stateSize(cls.robot.model, cls.spec))
        mujoco.mj_getState(cls.robot.model, cls.robot.data, cls.before, cls.spec)
        cls.geometry_before = cls.robot.model.geom_size.copy()
        cls.data = build_visual_model_data(cls.robot, cls.frames, official_repo=cls.repo)
        cls.after = np.empty_like(cls.before)
        mujoco.mj_getState(cls.robot.model, cls.robot.data, cls.after, cls.spec)

    def test_pinned_sources_materials_and_all_visual_parts_retained(self):
        data = self.data
        self.assertEqual(data['provenance']['commit'], SEEED_COMMIT)
        self.assertEqual(len(data['meshes']), 34)
        self.assertEqual(data['provenance']['original_triangles'], 430912)
        self.assertEqual(data['provenance']['display_mesh_derivation_license'], 'CERN-OHL-W-2.0')
        self.assertEqual(data['provenance']['package_license_tag'], 'Apache-2.0')
        urdf = ET.parse(self.repo/URDF_RELATIVE).getroot()
        materials = {m.get('name'): np.fromstring(m.find('color').get('rgba'), sep=' ')
                     for m in urdf.findall('material')}
        for mesh in data['meshes']:
            raw = (self.repo/mesh['source_mesh']).read_bytes()
            self.assertEqual(hashlib.sha256(raw).hexdigest(), mesh['source_sha256'])
            self.assertTrue(mesh['matches_recorded_visual_source_sha256'])
            np.testing.assert_array_equal(mesh['rgba'], materials[mesh['material_name']])
        self.assertTrue(data['compatibility']['compatible'])

    def test_independent_official_fk_matches_recorded_body_poses(self):
        with _preserve_integration_state(self.robot.model, self.robot.data):
            for frame, (_, q) in zip(self.data['frames'], self.frames):
                self.robot.set_q(q)
                for mesh, transform in zip(self.data['meshes'], frame['transforms']):
                    body = mujoco.mj_name2id(self.robot.model, mujoco.mjtObj.mjOBJ_BODY, mesh['body_name'])
                    np.testing.assert_allclose(transform['translation_world_m'], self.robot.data.xpos[body], atol=1e-9)
                    np.testing.assert_allclose(transform['rotation_world_from_link'], self.robot.data.xmat[body].reshape(3, 3), atol=1e-9)
        self.assertLess(self.data['compatibility']['max_link_position_error_m'], 1e-9)

    def test_mesh_bounds_and_simplification_remain_finite_and_disclosed(self):
        self.assertLess(self.data['provenance']['display_triangles'], 130000)
        for mesh in self.data['meshes']:
            vertices, faces = np.asarray(mesh['vertices']), np.asarray(mesh['faces'])
            self.assertTrue(np.all(np.isfinite(vertices)))
            self.assertEqual(faces.shape[1], 3)
            self.assertTrue(np.all(faces >= 0))
            self.assertLess(int(faces.max()), len(vertices))
            source, _ = read_stl_mesh(self.repo/mesh['source_mesh'])
            # All official DM visual origins/scales are identity in this pin.
            self.assertTrue(np.all(vertices >= source.min(axis=0)-1e-8))
            self.assertTrue(np.all(vertices <= source.max(axis=0)+1e-8))
            self.assertLessEqual(mesh['simplification']['nearest_vertex_deviation_max_m'], .003+1e-9)
        self.assertFalse(self.data['provenance']['geometry_used_for_collision_or_control'])

    def test_full_integration_and_collision_geometry_preserved(self):
        np.testing.assert_array_equal(self.before, self.after)
        np.testing.assert_array_equal(self.geometry_before, self.robot.model.geom_size)
        self.assertFalse(self.data['physics_advanced'])
        self.assertFalse(self.data['controls_or_collision_geometry_changed'])
        json.dumps(self.data, allow_nan=False)
        self.assertNotIn('/home/', json.dumps(self.data))


if __name__ == '__main__':
    unittest.main()
