"""Geometry/provenance and offline-payload checks for parallel 3-D scenes."""
import base64
import gzip
import json
import re
import unittest

import mujoco
import numpy as np

from cbf_demo.comparison_view import (
    build_comparison_data, comparison_fragment, comparison_script,
)
from cbf_demo.scenes import build_field, clutter_scene, START, SURFACE_EPSILON
from cbf_demo.robot import RebotModel


class ComparisonViewTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.robot=RebotModel(clutter_scene(),epsilon=SURFACE_EPSILON)
        cls.field=build_field(clutter_scene())
        cls.data=build_comparison_data(cls.robot,cls.field,[(0.,START)])

    def test_full_oriented_geometry_and_margin_shells(self):
        data=self.data
        self.assertEqual(len(data['frames'][0]['arm_vertices']),80)
        self.assertEqual(len(data['arm_faces']),120)
        self.assertEqual(len(data['arm_face_colors']),120)
        self.assertEqual(len(data['obstacles']),5)
        self.assertEqual(data['margins_m'],[.025,.05,.1])
        self.assertEqual(data['default_margin_m'],.05)
        for mesh in data['frames'][0]['shells']:
            self.assertGreater(len(mesh['vertices']),0)
            self.assertGreater(len(mesh['faces']),0)
            self.assertTrue(np.all(np.isfinite(mesh['vertices'])))
        self.assertFalse(data['scope']['physics_advanced'])
        self.assertFalse(data['scope']['right_geometry_used_for_control'])

    def test_nearest_pair_matches_independent_whole_box_distance(self):
        previous=self.robot.get_q()
        try:
            self.robot.set_q(START)
            self.assertAlmostEqual(self.data['frames'][0]['nearest']['clearance_m'],
                                   self.robot.clearance(),places=12)
            points=np.asarray(self.data['frames'][0]['nearest']['points_m'])
            self.assertEqual(points.shape,(2,3))
            self.assertTrue(np.all(np.isfinite(points)))
        finally:
            self.robot.set_q(previous)

    def test_data_preserves_full_simulator_state_and_field(self):
        spec=mujoco.mjtState.mjSTATE_INTEGRATION
        before=np.empty(mujoco.mj_stateSize(self.robot.model,spec))
        self.robot.data.time=12.3
        self.robot.data.qvel[:]=np.arange(6)/10
        mujoco.mj_getState(self.robot.model,self.robot.data,before,spec)
        field_before=self.field.h.copy()
        build_comparison_data(self.robot,self.field,[(1.2,START)],margins=(.05,))
        after=np.empty_like(before)
        mujoco.mj_getState(self.robot.model,self.robot.data,after,spec)
        np.testing.assert_array_equal(after,before)
        np.testing.assert_array_equal(self.field.h,field_before)

    def test_compressed_payload_round_trips_without_another_plotly_bundle(self):
        fragment=comparison_fragment(self.data)
        script=comparison_script(self.data)
        packed=re.search(r"const packed='([^']+)';",script).group(1)
        decoded=json.loads(gzip.decompress(base64.b64decode(packed)))
        self.assertEqual(decoded,self.data)
        self.assertIn('comparison-pose',fragment)
        self.assertIn('DecompressionStream',script)
        self.assertIn('rebot-comparison-change',script)
        self.assertNotIn('cdn.plot.ly',fragment+script)
        self.assertNotIn('Plotly.js v',fragment+script)
        self.assertNotIn('/home/',json.dumps(self.data))

    def test_invalid_pose_and_margin_are_rejected(self):
        with self.assertRaises(ValueError):
            build_comparison_data(self.robot,self.field,[])
        with self.assertRaises(ValueError):
            build_comparison_data(self.robot,self.field,[(0,START)],margins=(-1,))


if __name__=='__main__':
    unittest.main()
