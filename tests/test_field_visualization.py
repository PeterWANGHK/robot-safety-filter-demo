"""Physical-coordinate and display-only checks for 3-D field extraction."""

from types import SimpleNamespace
import unittest

import numpy as np

from cbf_demo.field_visualization import (
    build_field_meshes, field_rgb, FIELD_LEVELS, FIELD_COLORS,
)


class FieldVisualizationTests(unittest.TestCase):
    def setUp(self):
        # An affine field has an exactly known planar isosurface; asymmetric
        # coordinates detect axis swaps and missing world-origin translation.
        origin = np.array([-0.3, 0.2, -0.1])
        spacing = 0.02
        shape = (17, 13, 11)
        x = origin[0] + spacing*np.arange(shape[0])
        values = np.broadcast_to(x[:, None, None], shape).copy()
        self.field = SimpleNamespace(origin=origin, spacing=spacing, h=values)

    def test_affine_isosurface_uses_world_metres_and_valid_triangles(self):
        before = self.field.h.copy()
        meshes = build_field_meshes(self.field, levels=(-0.19,), step_size=1)
        self.assertEqual(len(meshes), 1)
        mesh = meshes[0]
        np.testing.assert_allclose(mesh['vertices'][:, 0], -0.19, atol=3e-8)
        np.testing.assert_allclose(mesh['vertices'][:, 1].min(), 0.2, atol=3e-8)
        np.testing.assert_allclose(mesh['vertices'][:, 2].max(), 0.1, atol=3e-8)
        self.assertEqual(mesh['faces'].shape[1], 3)
        self.assertGreater(len(mesh['faces']), 0)
        self.assertTrue(np.all(mesh['faces'] >= 0))
        self.assertLess(mesh['faces'].max(), len(mesh['vertices']))
        np.testing.assert_array_equal(self.field.h, before)

    def test_crop_keeps_only_inside_triangles_without_changing_field(self):
        before = self.field.h.copy()
        bounds = ((-0.20, 0.25, -0.05), (-0.18, 0.39, 0.07))
        mesh = build_field_meshes(self.field, levels=(-0.19,), bounds=bounds, step_size=1)[0]
        self.assertTrue(np.all(mesh['vertices'] >= np.asarray(bounds[0])))
        self.assertTrue(np.all(mesh['vertices'] <= np.asarray(bounds[1])))
        self.assertEqual(len(np.unique(mesh['faces'])), len(mesh['vertices']))
        np.testing.assert_array_equal(self.field.h, before)
        self.assertEqual(build_field_meshes(self.field, levels=(-0.19,),
                                            bounds=((1, 1, 1), (2, 2, 2))), [])

    def test_out_of_range_duplicate_and_constant_levels(self):
        meshes = build_field_meshes(self.field, levels=(4, -4, -0.19, -0.19), step_size=1)
        self.assertEqual([mesh['level'] for mesh in meshes], [-0.19])
        constant = SimpleNamespace(origin=np.zeros(3), spacing=.1, h=np.ones((4, 4, 4)))
        self.assertEqual(build_field_meshes(constant, levels=(0, 1, 2)), [])

    def test_palette_matches_fixed_level_anchors(self):
        expected = np.array([[int(c[i:i+2], 16)/255 for i in (1, 3, 5)] for c in FIELD_COLORS])
        np.testing.assert_allclose(field_rgb(FIELD_LEVELS), expected)
        np.testing.assert_allclose(field_rgb([-1, 100]), expected[[0, -1]])

    def test_invalid_extraction_parameters_are_rejected(self):
        for step in (0, -1, 1.2, True):
            with self.assertRaises(ValueError):
                build_field_meshes(self.field, step_size=step)
        with self.assertRaises(ValueError):
            build_field_meshes(self.field, bounds=((1, 0, 0), (0, 1, 1)))
        with self.assertRaises(ValueError):
            build_field_meshes(self.field, levels=(np.nan,))


if __name__ == '__main__':
    unittest.main()
