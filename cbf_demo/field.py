"""CPU Poisson safety field for a static, voxelized 3-D scene.

This preliminary paper-inspired implementation solves ``Laplacian(h) = -1``
in free space, with homogeneous Dirichlet values on inflated occupied grid
sites and on the rectangular outer boundary.  A constant source is an explicit
demo choice, not a claim to reproduce the paper's source construction exactly.
In particular, h has units m^2; it is NOT a Euclidean signed distance field.

Grid sites are at ``origin + spacing * (i, j, k)``.  ``occupancy`` is the
original physical mask; ``buffered_occupancy`` also includes the requested
inflation and a half-voxel-diagonal allowance. ``solver_occupancy`` adds one
full 3x3x3 corner neighborhood around that mask. This extra numerical collar
makes every cell touching buffered occupancy identically zero, so a trajectory
approaching an invalid cell sees h tend to zero first. It avoids a positive
jump at the validity boundary. Invalid-cell queries return negative h and zero
gradient and must be rejected by the safety controller. Zero-valued collar
points are also invalid, although sampling them retains their raw zero value.

The returned gradient is the exact derivative of the trilinear interpolant.
It is piecewise continuous and can jump at voxel faces; this is a numerical
approximation, not a continuously differentiable safety certificate.
"""

from __future__ import annotations

import inspect
import time
from typing import Iterable

import numpy as np
from scipy.fft import dstn, idstn
from scipy.ndimage import binary_dilation, distance_transform_edt
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import LinearOperator, cg


class PoissonSafetyField:
    """Solve and sample a static 3-D Poisson field using metric coordinates.

    Parameters
    ----------
    origin : array-like, shape (3,)
        Position in metres of grid site [0, 0, 0].
    shape : tuple of three integers >= 3
        Number of grid sites per axis, including the Dirichlet boundary.
    spacing : positive float
        Isotropic grid spacing, in metres.
    occupancy : bool array, shape ``shape``
        Original physical occupied grid sites. It is copied, never modified.
    inflation : nonnegative float
        Requested configuration-space obstacle padding in metres. An
        additional half-voxel diagonal is included conservatively.

    ``solve_info`` records the discrete residual in physical PDE units,
    convergence status, iteration count and timing.  A failed solve raises
    RuntimeError instead of publishing a partially converged field.
    """

    def __init__(
        self,
        origin: Iterable[float],
        shape: tuple[int, int, int],
        spacing: float,
        occupancy: np.ndarray,
        inflation: float = 0.0,
    ) -> None:
        self.origin = np.asarray(origin, dtype=float).copy()
        if self.origin.shape != (3,) or not np.all(np.isfinite(self.origin)):
            raise ValueError("origin must contain three finite coordinates")
        supplied_shape = tuple(shape)
        if len(supplied_shape) != 3 or any(
            not isinstance(n, (int, np.integer)) or isinstance(n, (bool, np.bool_)) or n < 3
            for n in supplied_shape
        ):
            raise ValueError("shape must contain three integers >= 3")
        self.shape = tuple(int(n) for n in supplied_shape)
        self.spacing = float(spacing)
        self.inflation = float(inflation)
        if not np.isfinite(self.spacing) or self.spacing <= 0:
            raise ValueError("spacing must be finite and positive")
        if not np.isfinite(self.inflation) or self.inflation < 0:
            raise ValueError("inflation must be finite and nonnegative")
        physical = np.asarray(occupancy)
        if physical.shape != self.shape or physical.dtype != np.bool_:
            raise ValueError("occupancy must be a boolean array matching shape")
        self.occupancy = physical.copy()
        self.axes = tuple(
            self.origin[a] + self.spacing * np.arange(self.shape[a]) for a in range(3)
        )
        self.upper = self.origin + self.spacing * (np.asarray(self.shape) - 1)
        self.padding = self.inflation + np.sqrt(3.0) * self.spacing / 2.0
        self.buffered_occupancy = self.occupancy.copy()
        if np.any(self.occupancy):
            distances = distance_transform_edt(~self.occupancy, sampling=self.spacing)
            self.buffered_occupancy = distances <= self.padding
        # Any cell with one required buffered corner is rejected by _cells.
        # Its other seven corners differ by at most one index on each axis.
        # This exact dilation therefore makes h identically zero throughout
        # every rejected cell, unlike rejecting cells around the solve mask
        # itself (which creates a positive h jump invisible to a CBF).
        self.solver_occupancy = binary_dilation(
            self.buffered_occupancy, structure=np.ones((3, 3, 3), dtype=bool)
        )
        self.numerical_collar_max_m = float(np.sqrt(3.0) * self.spacing)
        self.h = np.zeros(self.shape, dtype=float)
        self.invalid_value = -self.spacing**2
        self._solve()

    @classmethod
    def from_boxes(
        cls,
        origin: Iterable[float],
        shape: tuple[int, int, int],
        spacing: float,
        boxes: Iterable[tuple[Iterable[float], Iterable[float]]],
        inflation: float = 0.0,
    ) -> "PoissonSafetyField":
        """Construct from world-aligned boxes ``[(lower_xyz, upper_xyz), ...]``.

        A physical grid voxel is marked if its closed, spacing-wide cube
        intersects a box.  Original and buffered masks remain available
        separately.  Boxes outside the rectangular field are simply clipped.
        Rotated geometry needs conservative AABBs or a supplied occupancy mask.
        """
        origin_array = np.asarray(origin, dtype=float)
        supplied_shape = tuple(shape)
        if origin_array.shape != (3,) or not np.all(np.isfinite(origin_array)):
            raise ValueError("origin must contain three finite coordinates")
        if len(supplied_shape) != 3 or any(
            not isinstance(n, (int, np.integer)) or isinstance(n, (bool, np.bool_)) or n < 3
            for n in supplied_shape
        ):
            raise ValueError("shape must contain three integers >= 3")
        step = float(spacing)
        if not np.isfinite(step) or step <= 0:
            raise ValueError("spacing must be finite and positive")
        axes = [origin_array[a] + step * np.arange(supplied_shape[a]) for a in range(3)]
        occupancy = np.zeros(supplied_shape, dtype=bool)
        for lower, upper in boxes:
            lower = np.asarray(lower, dtype=float)
            upper = np.asarray(upper, dtype=float)
            if (
                lower.shape != (3,)
                or upper.shape != (3,)
                or not np.all(np.isfinite(lower))
                or not np.all(np.isfinite(upper))
                or np.any(lower > upper)
            ):
                raise ValueError("each box needs finite lower <= upper xyz corners")
            indices = [
                np.flatnonzero((axes[a] >= lower[a] - step / 2) & (axes[a] <= upper[a] + step / 2))
                for a in range(3)
            ]
            if all(len(i) for i in indices):
                occupancy[np.ix_(*indices)] = True
        return cls(origin_array, supplied_shape, step, occupancy, inflation)

    def _solve(self) -> None:
        started = time.perf_counter()
        free = ~self.solver_occupancy.copy()
        for axis in range(3):
            boundary = [slice(None)] * 3
            boundary[axis] = 0
            free[tuple(boundary)] = False
            boundary[axis] = -1
            free[tuple(boundary)] = False
        self.free_mask = free
        count = int(np.count_nonzero(free))
        if count == 0:
            self.solve_info = {
                "converged": True, "unknowns": 0, "iterations": 0,
                "residual_inf": 0.0, "relative_residual_l2": 0.0,
                "seconds": time.perf_counter() - started,
                "equation": "laplacian(h)=-1", "h_units": "m^2",
                "buffered_occupied_sites": int(np.count_nonzero(self.buffered_occupancy)),
                "solver_occupied_sites": int(np.count_nonzero(self.solver_occupancy)),
                "numerical_collar_max_m": self.numerical_collar_max_m,
            }
            return

        # A is the positive, dimensionless 7-point negative Laplacian.
        number = np.full(self.shape, -1, dtype=np.int64)
        number[free] = np.arange(count)
        coordinates = np.argwhere(free)
        rows = [np.arange(count)]
        columns = [np.arange(count)]
        data = [np.full(count, 6.0)]
        for axis in range(3):
            for offset in (-1, 1):
                neighbor = coordinates.copy()
                neighbor[:, axis] += offset
                neighbor_numbers = number[tuple(neighbor.T)]
                present = neighbor_numbers >= 0
                rows.append(np.flatnonzero(present))
                columns.append(neighbor_numbers[present])
                data.append(np.full(np.count_nonzero(present), -1.0))
        matrix = coo_matrix(
            (np.concatenate(data), (np.concatenate(rows), np.concatenate(columns))),
            shape=(count, count),
        ).tocsr()
        rhs = np.full(count, self.spacing**2)

        # DST inversion of the unperforated rectangular Laplacian is an SPD
        # preconditioner after injection/restriction to the free sites. It
        # keeps a roughly 60^3 static scene affordable on ARM CPUs, without
        # a dense factorization or a GPU dependency.
        interior_shape = tuple(n - 2 for n in self.shape)
        eigenvalues = np.zeros(interior_shape)
        for axis, n in enumerate(interior_shape):
            lam = 2 - 2 * np.cos(np.pi * np.arange(1, n + 1) / (n + 1))
            reshape = [1, 1, 1]
            reshape[axis] = n
            eigenvalues += lam.reshape(reshape)
        inside_free = free[1:-1, 1:-1, 1:-1]

        def precondition(vector: np.ndarray) -> np.ndarray:
            padded = np.zeros(interior_shape)
            padded[inside_free] = vector
            solution = idstn(dstn(padded, type=1, norm="ortho") / eigenvalues, type=1, norm="ortho")
            return solution[inside_free]

        preconditioner = LinearOperator((count, count), matvec=precondition, dtype=float)
        iterations = 0

        def on_iteration(_: np.ndarray) -> None:
            nonlocal iterations
            iterations += 1

        # scipy >=1.12 renamed tol to rtol; support the Jetson system version
        # as well as newer installations without changing the numerical goal.
        kwargs = {"rtol" if "rtol" in inspect.signature(cg).parameters else "tol": 1e-10}
        solution, status = cg(
            matrix, rhs, M=preconditioner, atol=0.0, maxiter=2000,
            callback=on_iteration, **kwargs,
        )
        residual = matrix @ solution - rhs
        self.solve_info = {
            "converged": status == 0,
            "unknowns": count,
            "iterations": iterations,
            "residual_inf": float(np.max(np.abs(residual)) / self.spacing**2),
            "relative_residual_l2": float(np.linalg.norm(residual) / np.linalg.norm(rhs)),
            "seconds": time.perf_counter() - started,
            "equation": "laplacian(h)=-1", "h_units": "m^2",
            "buffered_occupied_sites": int(np.count_nonzero(self.buffered_occupancy)),
            "solver_occupied_sites": int(np.count_nonzero(self.solver_occupancy)),
            "numerical_collar_max_m": self.numerical_collar_max_m,
        }
        if status != 0 or not np.all(np.isfinite(solution)):
            raise RuntimeError(f"Poisson solve failed: {self.solve_info}")
        if np.min(solution) < -1e-10 * self.spacing**2:
            raise RuntimeError("Poisson solve violated the discrete maximum principle")
        self.h[free] = np.maximum(solution, 0.0)

    def _cells(self, points: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        points = np.asarray(points, dtype=float)
        if points.shape == (3,):
            points = points[None, :]
        if points.ndim != 2 or points.shape[1] != 3:
            raise ValueError("points must have shape (N, 3), or (3,)")
        finite = np.all(np.isfinite(points), axis=1)
        inside = finite & np.all(points > self.origin, axis=1) & np.all(points < self.upper, axis=1)
        grid = np.zeros_like(points)
        grid[finite] = (points[finite] - self.origin) / self.spacing
        grid = np.clip(grid, 0, np.asarray(self.shape) - 1)
        cells = np.minimum(np.floor(grid).astype(np.int64), np.asarray(self.shape) - 2)
        fractions = grid - cells
        for i in range(2):
            for j in range(2):
                for k in range(2):
                    inside &= ~self.buffered_occupancy[cells[:, 0] + i, cells[:, 1] + j, cells[:, 2] + k]
        return cells, fractions, inside

    def is_valid(self, points: np.ndarray) -> np.ndarray:
        """True only inside the buffered free domain with strictly positive h.

        A planner/controller should additionally require its configured
        positive h reserve; is_valid alone imposes no tracking-error reserve.
        """
        cells, fractions, valid = self._cells(points)
        values = np.zeros(len(cells))
        for i in range(2):
            wx = fractions[:, 0] if i else 1 - fractions[:, 0]
            for j in range(2):
                wy = fractions[:, 1] if j else 1 - fractions[:, 1]
                for k in range(2):
                    wz = fractions[:, 2] if k else 1 - fractions[:, 2]
                    corner = self.h[cells[:, 0] + i, cells[:, 1] + j, cells[:, 2] + k]
                    values += corner * wx * wy * wz
        return valid & (values > 0)

    def sample(self, points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return h[N] and exact trilinear gradients[N,3] at world points.

        Outside, boundary, nonfinite and buffered-cell queries yield
        ``(-spacing**2, [0,0,0])``.  Controllers must treat these as invalid
        geometry and fail closed, rather than assuming zero gradient means
        the robot is safe. Empty point arrays are supported.
        """
        cells, fractions, valid = self._cells(points)
        values = np.zeros(len(cells))
        gradients = np.zeros((len(cells), 3))
        for i in range(2):
            wx = fractions[:, 0] if i else 1 - fractions[:, 0]
            for j in range(2):
                wy = fractions[:, 1] if j else 1 - fractions[:, 1]
                for k in range(2):
                    wz = fractions[:, 2] if k else 1 - fractions[:, 2]
                    corner = self.h[cells[:, 0] + i, cells[:, 1] + j, cells[:, 2] + k]
                    values += corner * wx * wy * wz
                    gradients[:, 0] += corner * (1 if i else -1) * wy * wz / self.spacing
                    gradients[:, 1] += corner * wx * (1 if j else -1) * wz / self.spacing
                    gradients[:, 2] += corner * wx * wy * (1 if k else -1) / self.spacing
        values[~valid] = self.invalid_value
        gradients[~valid] = 0.0
        return values, gradients
