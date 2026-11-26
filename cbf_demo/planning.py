"""Sampled position reachability and collision-aware joint-space planning.

This is a preliminary planning layer, not a certification tool. Reachability is
an FK sample cache: it cannot establish orientation, payload, continuous-path,
or inverse-kinematics feasibility. Collision checking uses the robot model's
conservative geometry and, optionally, the buffered Poisson-field domain.
Edges are checked at finite joint intervals; they are not continuous-collision
certificates. No real robot command is published by this module.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import json
from pathlib import Path
import time
from typing import Any, Iterator

import numpy as np


@contextmanager
def _preserve_configuration(robot: Any) -> Iterator[None]:
    previous = np.asarray(robot.get_q(), dtype=float).copy()
    try:
        yield
    finally:
        robot.set_q(previous)


def _limits(robot: Any) -> tuple[np.ndarray, np.ndarray]:
    lower = np.asarray(robot.lower, dtype=float).reshape(-1)
    upper = np.asarray(robot.upper, dtype=float).reshape(-1)
    if lower.shape != upper.shape or not lower.size:
        raise ValueError("Robot joint limits must be matching nonempty vectors")
    if not np.all(np.isfinite(lower)) or not np.all(np.isfinite(upper)):
        raise ValueError("Finite joint limits are required")
    if np.any(lower >= upper):
        raise ValueError("Each lower joint limit must be below its upper limit")
    return lower, upper


class CollisionChecker:
    """Read-only configuration checks against model and optional field geometry.

    ``count`` includes every configuration check, including malformed inputs.
    ``self_collision_checked`` states whether the model supplies a usable
    self-clearance query. A model without that query does not gain self-collision
    protection from the environment field. Planning must not run concurrently
    with another user of the same mutable robot model.
    """

    def __init__(self, robot: Any, field: Any = None,
                 clearance_margin: float = 0.015):
        self.robot = robot
        self.field = field
        self.lower, self.upper = _limits(robot)
        self.clearance_margin = float(clearance_margin)
        if not np.isfinite(self.clearance_margin) or self.clearance_margin < 0:
            raise ValueError("clearance_margin must be finite and nonnegative")
        self.count = 0
        self.self_collision_checked = False
        self.last_reason = "not_checked"

    @property
    def collision_checks(self) -> int:
        return self.count

    def __call__(self, q: np.ndarray) -> bool:
        self.count += 1
        q = np.asarray(q, dtype=float)
        if q.shape != self.lower.shape or not np.all(np.isfinite(q)):
            self.last_reason = "invalid_configuration"
            return False
        if np.any(q < self.lower) or np.any(q > self.upper):
            self.last_reason = "joint_limit"
            return False
        with _preserve_configuration(self.robot):
            self.robot.set_q(q)
            clearance = float(self.robot.clearance())
            if np.isnan(clearance) or clearance <= self.clearance_margin:
                self.last_reason = "environment_clearance"
                return False
            self_query = getattr(self.robot, "self_clearance", None)
            if callable(self_query):
                self_distance = self_query()
                if self_distance is not None:
                    self.self_collision_checked = True
                    self_distance = float(self_distance)
                    if np.isnan(self_distance) or self_distance <= self.clearance_margin:
                        self.last_reason = "self_clearance"
                        return False
            if self.field is not None:
                point_query = getattr(self.robot, "points", None)
                if callable(point_query):
                    points = np.asarray(point_query(), dtype=float)
                else:
                    points = np.asarray(self.robot.points_and_jacobians()[0], dtype=float)
                if not np.all(np.isfinite(points)) or not np.all(self.field.is_valid(points)):
                    self.last_reason = "outside_buffered_field"
                    return False
                h, gradient = self.field.sample(points)
                if (not np.all(np.isfinite(h)) or not np.all(np.isfinite(gradient))
                        or np.any(np.asarray(h) < 0.0)):
                    self.last_reason = "invalid_field_value"
                    return False
        self.last_reason = "free"
        return True

    def edge_free(self, a: np.ndarray, b: np.ndarray,
                  max_joint_step: float = 0.05) -> bool:
        """Check endpoints and interpolation with bounded maximum joint step."""
        step = float(max_joint_step)
        if not np.isfinite(step) or step <= 0:
            raise ValueError("max_joint_step must be positive and finite")
        a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
        if (a.shape != self.lower.shape or b.shape != self.lower.shape
                or not np.all(np.isfinite(a)) or not np.all(np.isfinite(b))):
            return False
        intervals = max(1, int(np.ceil(np.max(np.abs(b - a)) / step)))
        return all(self(a + fraction * (b - a))
                   for fraction in np.linspace(0.0, 1.0, intervals + 1))


@dataclass
class PlanResult:
    path: np.ndarray | None
    success: bool
    reason: str
    elapsed_s: float
    collision_checks: int


class _Tree:
    def __init__(self, root: np.ndarray, starts_at_start: bool):
        self.nodes = [root.copy()]
        self.parents = [-1]
        self.starts_at_start = starts_at_start

    def branch(self, index: int) -> list[np.ndarray]:
        branch = []
        while index >= 0:
            branch.append(self.nodes[index])
            index = self.parents[index]
        return list(reversed(branch))


class RRTConnect:
    """Seeded bidirectional RRT-Connect with finite-resolution edge checks."""

    def __init__(self, max_iterations: int = 2500, step_size: float = 0.25,
                 edge_resolution: float = 0.04, seed: int = 7):
        if isinstance(max_iterations, bool) or int(max_iterations) != max_iterations or max_iterations < 1:
            raise ValueError("max_iterations must be a positive integer")
        self.max_iterations = int(max_iterations)
        self.step_size = float(step_size)
        self.edge_resolution = float(edge_resolution)
        if (not np.isfinite(self.step_size) or self.step_size <= 0
                or not np.isfinite(self.edge_resolution) or self.edge_resolution <= 0):
            raise ValueError("Planner step sizes must be positive and finite")
        self.seed = int(seed)

    def _extend(self, tree: _Tree, target: np.ndarray,
                checker: CollisionChecker) -> tuple[str, int]:
        distances = np.linalg.norm(np.asarray(tree.nodes) - target, axis=1)
        nearest = int(np.argmin(distances))
        distance = float(distances[nearest])
        if distance < 1e-12:
            return "reached", nearest
        candidate = tree.nodes[nearest] + min(1.0, self.step_size / distance) * (target - tree.nodes[nearest])
        if not checker.edge_free(tree.nodes[nearest], candidate, self.edge_resolution):
            return "trapped", nearest
        tree.nodes.append(candidate)
        tree.parents.append(nearest)
        return ("reached" if distance <= self.step_size else "advanced"), len(tree.nodes) - 1

    def plan(self, start: np.ndarray, goal: np.ndarray,
             checker: CollisionChecker) -> PlanResult:
        began = time.perf_counter()
        checks_before = checker.count

        def result(path: np.ndarray | None, reason: str) -> PlanResult:
            return PlanResult(path, path is not None, reason,
                              time.perf_counter() - began, checker.count - checks_before)

        start, goal = np.asarray(start, dtype=float), np.asarray(goal, dtype=float)
        with _preserve_configuration(checker.robot):
            if not checker(start):
                return result(None, "invalid_start:" + checker.last_reason)
            if not checker(goal):
                return result(None, "invalid_goal:" + checker.last_reason)
            if np.array_equal(start, goal):
                return result(start[None, :].copy(), "already_at_goal")
            if checker.edge_free(start, goal, self.edge_resolution):
                return result(np.vstack((start, goal)), "direct_path")
            rng = np.random.default_rng(self.seed)
            a, b = _Tree(start, True), _Tree(goal, False)
            for _ in range(self.max_iterations):
                target = b.nodes[0] if rng.random() < 0.15 else rng.uniform(checker.lower, checker.upper)
                status_a, index_a = self._extend(a, target, checker)
                if status_a != "trapped":
                    while True:
                        status_b, index_b = self._extend(b, a.nodes[index_a], checker)
                        if status_b != "advanced":
                            break
                    if status_b == "reached":
                        branch_a, branch_b = a.branch(index_a), b.branch(index_b)
                        if a.starts_at_start:
                            path = branch_a + list(reversed(branch_b))[1:]
                        else:
                            path = branch_b + list(reversed(branch_a))[1:]
                        return result(np.asarray(path), "rrt_connect")
                a, b = b, a
        return result(None, "iteration_limit")


def shortcut_path(path: np.ndarray, checker: CollisionChecker,
                  edge_resolution: float = 0.04, attempts: int = 80,
                  seed: int = 7) -> np.ndarray:
    """Deterministically remove detours only after checking replacement edges."""
    path = np.asarray(path, dtype=float).copy()
    if path.ndim != 2 or path.shape[1:] != checker.lower.shape:
        raise ValueError("path must have one configuration per row")
    rng = np.random.default_rng(seed)
    for _ in range(max(0, int(attempts))):
        if len(path) < 3:
            break
        a, b = sorted(rng.choice(len(path), size=2, replace=False))
        if b > a + 1 and checker.edge_free(path[a], path[b], edge_resolution):
            path = np.vstack((path[:a + 1], path[b:]))
    return path


@dataclass
class ReachabilityMap:
    """Position-only sampled FK map retaining exact joint seeds, not an IK map."""

    xyz: np.ndarray
    joint_samples: np.ndarray
    lower: np.ndarray
    upper: np.ndarray
    voxel_size: float = 0.04
    attempted_samples: int = 0

    def __post_init__(self) -> None:
        self.xyz = np.asarray(self.xyz, dtype=float).reshape(-1, 3)
        self.lower = np.asarray(self.lower, dtype=float).reshape(-1)
        self.upper = np.asarray(self.upper, dtype=float).reshape(-1)
        self.joint_samples = np.asarray(self.joint_samples, dtype=float).reshape(-1, len(self.lower))
        if len(self.xyz) != len(self.joint_samples):
            raise ValueError("Position and joint sample counts differ")
        if not np.all(np.isfinite(self.xyz)) or not np.all(np.isfinite(self.joint_samples)):
            raise ValueError("Reachability samples must be finite")
        self.voxel_size = float(self.voxel_size)
        if not np.isfinite(self.voxel_size) or self.voxel_size <= 0:
            raise ValueError("voxel_size must be positive and finite")

    @property
    def sample_count(self) -> int:
        return len(self.xyz)

    @property
    def voxel_keys(self) -> np.ndarray:
        return np.unique(np.floor(self.xyz / self.voxel_size).astype(np.int64), axis=0)

    @classmethod
    def from_joint_samples(cls, robot: Any, qs: np.ndarray,
                           checker: CollisionChecker | None = None,
                           voxel_size: float = 0.04) -> "ReachabilityMap":
        lower, upper = _limits(robot)
        qs = np.asarray(qs, dtype=float)
        if qs.ndim != 2 or qs.shape[1] != len(lower):
            raise ValueError("qs must have one valid-size joint vector per row")
        positions, configurations = [], []
        with _preserve_configuration(robot):
            for q in qs:
                if (not np.all(np.isfinite(q)) or np.any(q < lower)
                        or np.any(q > upper) or (checker is not None and not checker(q))):
                    continue
                robot.set_q(q)
                position = np.asarray(robot.tcp(), dtype=float)
                if position.shape != (3,) or not np.all(np.isfinite(position)):
                    raise ValueError("robot.tcp() must return a finite XYZ vector")
                positions.append(position.copy())
                configurations.append(q.copy())
        return cls(np.asarray(positions).reshape(-1, 3),
                   np.asarray(configurations).reshape(-1, len(lower)),
                   lower.copy(), upper.copy(), voxel_size, len(qs))

    @classmethod
    def build(cls, robot: Any, checker: CollisionChecker | None = None,
              samples: int = 500, voxel_size: float = 0.04,
              seed: int = 7) -> "ReachabilityMap":
        if isinstance(samples, bool) or int(samples) != samples or samples < 1:
            raise ValueError("samples must be a positive integer")
        lower, upper = _limits(robot)
        rng = np.random.default_rng(seed)
        qs = rng.uniform(lower, upper, size=(int(samples), len(lower)))
        qs[0] = np.asarray(robot.get_q(), dtype=float)
        return cls.from_joint_samples(robot, qs, checker, voxel_size)

    def select(self, target_xyz: np.ndarray,
               tolerance: float = 0.04) -> np.ndarray | None:
        """Return a seed whose sampled TCP is within the exact XYZ tolerance.

        No IK refinement is performed. A missing seed means 'not demonstrated by
        this sample set', not 'unreachable'. The seed's TCP orientation is free.
        """
        target = np.asarray(target_xyz, dtype=float)
        if target.shape != (3,) or not np.all(np.isfinite(target)):
            raise ValueError("target_xyz must contain three finite coordinates")
        tolerance = float(tolerance)
        if not np.isfinite(tolerance) or tolerance < 0:
            raise ValueError("tolerance must be finite and nonnegative")
        if not self.sample_count:
            return None
        distances = np.linalg.norm(self.xyz - target, axis=1)
        closest = int(np.argmin(distances))
        return self.joint_samples[closest].copy() if distances[closest] <= tolerance else None

    def save(self, path: str | Path) -> None:
        metadata = json.dumps({
            "kind": "position_only_sampled_fk_not_ik",
            "orientation_certified": False,
            "payload_certified": False,
            "continuous_path_certified": False,
            "attempted_samples": int(self.attempted_samples),
        })
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, xyz=self.xyz, joint_samples=self.joint_samples,
                            lower=self.lower, upper=self.upper,
                            voxel_size=self.voxel_size, voxel_keys=self.voxel_keys,
                            metadata=np.array(metadata))

    @classmethod
    def load(cls, path: str | Path) -> "ReachabilityMap":
        with np.load(path, allow_pickle=False) as data:
            metadata = json.loads(str(data["metadata"]))
            if metadata.get("kind") != "position_only_sampled_fk_not_ik":
                raise ValueError("Unsupported reachability map format")
            return cls(data["xyz"].copy(), data["joint_samples"].copy(),
                       data["lower"].copy(), data["upper"].copy(),
                       float(data["voxel_size"]), int(metadata["attempted_samples"]))
