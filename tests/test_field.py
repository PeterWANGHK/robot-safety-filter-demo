"""Numerical and conservative-geometry checks for the static Poisson field."""

import unittest

import numpy as np

from cbf_demo.field import PoissonSafetyField


class TestPoissonSafetyField(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.empty = PoissonSafetyField(
            origin=(-0.25, -0.25, -0.25), shape=(21, 21, 21), spacing=0.025,
            occupancy=np.zeros((21, 21, 21), dtype=bool),
        )
        cls.blocked = PoissonSafetyField.from_boxes(
            origin=(-0.25, -0.25, -0.25), shape=(21, 21, 21), spacing=0.025,
            boxes=[((-0.025, -0.025, -0.025), (0.025, 0.025, 0.025))], inflation=0.025,
        )

    def test_discrete_poisson_equation_and_boundary(self):
        field = self.blocked
        h = field.h
        laplacian = (
            h[2:, 1:-1, 1:-1] + h[:-2, 1:-1, 1:-1]
            + h[1:-1, 2:, 1:-1] + h[1:-1, :-2, 1:-1]
            + h[1:-1, 1:-1, 2:] + h[1:-1, 1:-1, :-2]
            - 6 * h[1:-1, 1:-1, 1:-1]
        ) / field.spacing**2
        free = field.free_mask[1:-1, 1:-1, 1:-1]
        self.assertLess(np.max(np.abs(laplacian[free] + 1)), 1e-7)
        self.assertTrue(field.solve_info["converged"])
        self.assertLess(field.solve_info["relative_residual_l2"], 2e-10)
        self.assertTrue(np.all(h[field.buffered_occupancy] == 0))
        self.assertTrue(np.all(h[field.solver_occupancy] == 0))
        self.assertTrue(np.all(h[0] == 0))
        self.assertTrue(np.all(h[-1] == 0))
        self.assertTrue(np.all(h[:, 0] == 0))
        self.assertTrue(np.all(h[:, -1] == 0))
        self.assertTrue(np.all(h[:, :, 0] == 0))
        self.assertTrue(np.all(h[:, :, -1] == 0))
        self.assertGreater(np.min(h[field.free_mask]), 0)

    def test_empty_scene_symmetry_and_positive_interior(self):
        field = self.empty
        np.testing.assert_allclose(field.h, field.h[::-1, :, :], atol=1e-13)
        np.testing.assert_allclose(field.h, np.swapaxes(field.h, 0, 2), atol=1e-13)
        self.assertFalse(np.any(field.buffered_occupancy))
        self.assertGreater(field.sample([[0, 0, 0]])[0][0], 0)

    def test_interpolated_gradient_is_derivative_of_sampled_value(self):
        # Avoid voxel faces: the trilinear interpolant is differentiable
        # within each cell but its gradient need not be continuous at faces.
        points = np.array([[0.112, 0.058, -0.042], [-0.131, 0.087, 0.063]])
        field = self.blocked
        self.assertTrue(np.all(field.is_valid(points)))
        _, gradients = field.sample(points)
        delta = 1e-7
        numerical = np.zeros_like(gradients)
        for axis in range(3):
            shift = np.zeros(3)
            shift[axis] = delta
            plus = field.sample(points + shift)[0]
            minus = field.sample(points - shift)[0]
            numerical[:, axis] = (plus - minus) / (2 * delta)
        np.testing.assert_allclose(gradients, numerical, rtol=2e-8, atol=2e-10)

    def test_invalid_queries_fail_negative_with_no_gradient(self):
        points = [[0, 0, 0], [1, 0, 0], [-0.25, 0, 0], [np.nan, 0, 0], [np.inf, 0, 0]]
        values, gradients = self.blocked.sample(points)
        self.assertTrue(np.all(values < 0))
        self.assertFalse(np.any(self.blocked.is_valid(points)))
        np.testing.assert_array_equal(gradients, np.zeros((5, 3)))

    def test_inflation_is_conservative_and_keeps_original_mask(self):
        field = self.blocked
        self.assertGreater(field.padding, field.inflation)
        self.assertGreater(np.count_nonzero(field.buffered_occupancy), np.count_nonzero(field.occupancy))
        self.assertTrue(np.all(field.buffered_occupancy[field.occupancy]))
        self.assertTrue(np.all(field.solver_occupancy[field.buffered_occupancy]))
        self.assertGreater(np.count_nonzero(field.solver_occupancy), np.count_nonzero(field.buffered_occupancy))
        original_count = np.count_nonzero(field.occupancy)
        self.assertGreater(original_count, 0)
        field.sample([[0, 0, 0]])
        self.assertEqual(np.count_nonzero(field.occupancy), original_count)

    def test_field_tends_to_zero_before_invalid_cell_boundary(self):
        # A blocked plane makes the old failure obvious: rejecting any cell
        # touching the solve mask used to jump from a positive h to invalid.
        # The extra solve collar now forces all corners of that cell to zero.
        occupied = np.zeros((21, 21, 21), dtype=bool)
        occupied[10, :, :] = True
        field = PoissonSafetyField([0, 0, 0], occupied.shape, 0.025, occupied)
        boundary_x = 9 * field.spacing
        points = np.array([
            [boundary_x - 1e-4, 0.251, 0.249],
            [boundary_x - 1e-7, 0.251, 0.249],
        ])
        values, _ = field.sample(points)
        self.assertTrue(np.all(field.is_valid(points)))
        self.assertTrue(np.all(values > 0))
        self.assertLess(values[1], 1e-8)
        np.testing.assert_allclose(values[0] / values[1], 1000, rtol=1e-7)
        self.assertFalse(field.is_valid([[boundary_x, 0.251, 0.249]])[0])
        self.assertLess(field.sample([[boundary_x, 0.251, 0.249]])[0][0], 0)
        self.assertGreater(field.solve_info['solver_occupied_sites'], field.solve_info['buffered_occupied_sites'])

    def test_every_buffered_cell_has_zero_interpolation_corners(self):
        # Exact containment relation needed by the validity semantics, checked
        # throughout a 3-D box scene rather than only on an axial approach.
        field = self.blocked
        cell_shape = tuple(size - 1 for size in field.shape)
        touches_required = np.zeros(cell_shape, dtype=bool)
        every_corner_zero = np.ones(cell_shape, dtype=bool)
        for i in range(2):
            for j in range(2):
                for k in range(2):
                    slices = tuple(slice(offset, offset + size) for offset, size in zip((i, j, k), cell_shape))
                    touches_required |= field.buffered_occupancy[slices]
                    every_corner_zero &= field.h[slices] == 0
        self.assertTrue(np.all(every_corner_zero[touches_required]))

    def test_field_is_not_distance_and_scales_as_length_squared(self):
        original = self.empty
        scaled = PoissonSafetyField(
            2 * original.origin, original.shape, 2 * original.spacing,
            np.zeros(original.shape, dtype=bool),
        )
        np.testing.assert_allclose(scaled.h, 4 * original.h, rtol=1e-10, atol=1e-13)

    def test_empty_queries_and_fully_occupied_scene(self):
        values, gradients = self.empty.sample(np.empty((0, 3)))
        self.assertEqual(values.shape, (0,))
        self.assertEqual(gradients.shape, (0, 3))
        full = PoissonSafetyField([0, 0, 0], (3, 3, 3), 0.1, np.ones((3, 3, 3), dtype=bool))
        self.assertEqual(full.solve_info["unknowns"], 0)
        self.assertLess(full.sample([[0.1, 0.1, 0.1]])[0][0], 0)

    def test_invalid_input_is_rejected(self):
        for spacing in [0, -1, np.nan]:
            with self.assertRaises(ValueError):
                PoissonSafetyField([0, 0, 0], (3, 3, 3), spacing, np.zeros((3, 3, 3), bool))
        with self.assertRaises(ValueError):
            PoissonSafetyField([0, 0, 0], (3, 3, 3), 0.1, np.zeros((3, 3, 3)))
        with self.assertRaises(ValueError):
            PoissonSafetyField.from_boxes([0, 0, 0], (3, 3, 3), 0.1, [([1, 0, 0], [0, 1, 1])])


if __name__ == "__main__":
    unittest.main()
