"""Planning tests use a tiny deterministic robot, with no ROS or hardware."""

import tempfile
from pathlib import Path
import unittest

import numpy as np

from cbf_demo.planning import CollisionChecker, ReachabilityMap, RRTConnect, shortcut_path


class FakeRobot:
    def __init__(self, obstacle="box"):
        self.lower = np.full(6, -1.0)
        self.upper = np.full(6, 1.0)
        self.q = np.array([-0.8, 0.0, 0.0, 0.0, 0.0, 0.0])
        self.obstacle = obstacle
        self.self_distance = None

    def set_q(self, q):
        self.q = np.asarray(q, dtype=float).copy()

    def get_q(self):
        return self.q.copy()

    def tcp(self):
        return self.q[:3].copy()

    def points(self):
        return self.q[:3][None, :].copy()

    def clearance(self):
        if self.obstacle is None:
            return float("inf")
        if self.obstacle == "wall":
            return abs(self.q[0]) - 0.25
        delta = np.abs(self.q[:2]) - np.array([0.25, 0.25])
        return float(np.linalg.norm(np.maximum(delta, 0)) + min(np.max(delta), 0))

    def self_clearance(self):
        return self.self_distance


class FakeField:
    def is_valid(self, points):
        return points[:, 1] >= 0.0

    def sample(self, points):
        return np.ones(len(points)), np.tile([0.0, 1.0, 0.0], (len(points), 1))


class PlanningTests(unittest.TestCase):
    def test_checker_preserves_q_and_catches_edge_collision(self):
        robot = FakeRobot()
        original = robot.get_q()
        checker = CollisionChecker(robot)
        goal = original.copy()
        goal[0] = 0.8
        self.assertTrue(checker(original))
        self.assertTrue(checker(goal))
        self.assertFalse(checker.edge_free(original, goal, max_joint_step=0.04))
        np.testing.assert_array_equal(robot.get_q(), original)
        self.assertGreater(checker.count, 2)

    def test_checker_rejects_limits_nonfinite_and_optional_self_collision(self):
        robot = FakeRobot(None)
        checker = CollisionChecker(robot)
        original = robot.get_q()
        bad = original.copy()
        bad[4] = 2
        self.assertFalse(checker(bad))
        bad[4] = np.nan
        self.assertFalse(checker(bad))
        self.assertFalse(checker(np.zeros(3)))
        robot.self_distance = 0.01
        self.assertFalse(checker(original))
        self.assertTrue(checker.self_collision_checked)
        np.testing.assert_array_equal(robot.get_q(), original)

    def test_buffered_field_is_checked_independently(self):
        robot = FakeRobot(None)
        checker = CollisionChecker(robot, FakeField())
        q = robot.get_q()
        q[1] = -0.1
        self.assertFalse(checker(q))
        self.assertEqual(checker.last_reason, "outside_buffered_field")
        q[1] = 0.1
        self.assertTrue(checker(q))

    def test_rrt_connect_routes_around_obstacle_deterministically(self):
        robot = FakeRobot()
        start = robot.get_q()
        goal = start.copy()
        goal[0] = 0.8
        checker = CollisionChecker(robot)
        planner = RRTConnect(max_iterations=2000, seed=7)
        first = planner.plan(start, goal, checker)
        second = planner.plan(start, goal, checker)
        self.assertTrue(first.success, first.reason)
        self.assertEqual(first.reason, "rrt_connect")
        self.assertGreater(first.collision_checks, 0)
        np.testing.assert_array_equal(first.path, second.path)
        np.testing.assert_allclose(first.path[0], start)
        np.testing.assert_allclose(first.path[-1], goal)
        self.assertTrue(all(checker.edge_free(a, b, 0.02)
                            for a, b in zip(first.path[:-1], first.path[1:])))
        reduced = shortcut_path(first.path, checker)
        self.assertLessEqual(len(reduced), len(first.path))
        self.assertTrue(all(checker.edge_free(a, b, 0.02)
                            for a, b in zip(reduced[:-1], reduced[1:])))
        np.testing.assert_array_equal(robot.get_q(), start)

    def test_blocked_or_invalid_endpoints_do_not_change_robot(self):
        robot = FakeRobot("wall")
        start = robot.get_q()
        goal = start.copy()
        goal[0] = 0.8
        checker = CollisionChecker(robot)
        planner = RRTConnect(max_iterations=75, seed=7)
        blocked = planner.plan(start, goal, checker)
        self.assertFalse(blocked.success)
        self.assertEqual(blocked.reason, "iteration_limit")
        goal[0] = 0
        self.assertTrue(planner.plan(start, goal, checker).reason.startswith("invalid_goal:"))
        start_bad = start.copy()
        start_bad[0] = np.nan
        self.assertTrue(planner.plan(start_bad, goal, checker).reason.startswith("invalid_start:"))
        np.testing.assert_array_equal(robot.get_q(), start)

    def test_direct_and_already_at_goal(self):
        robot = FakeRobot(None)
        checker = CollisionChecker(robot)
        q = robot.get_q()
        goal = q.copy()
        goal[0] += 0.1
        planner = RRTConnect()
        self.assertEqual(planner.plan(q, goal, checker).reason, "direct_path")
        same = planner.plan(q, q, checker)
        self.assertTrue(same.success)
        self.assertEqual(same.path.shape, (1, 6))

    def test_position_sample_map_exact_tolerance_and_serialization(self):
        robot = FakeRobot(None)
        original = robot.get_q()
        other = original.copy()
        other[:3] = [0.25, 0.4, 0.2]
        invalid = original.copy()
        invalid[3] = 10
        reach = ReachabilityMap.from_joint_samples(robot, np.vstack((original, other, invalid)))
        self.assertEqual(reach.sample_count, 2)
        self.assertEqual(reach.attempted_samples, 3)
        np.testing.assert_array_equal(reach.select(other[:3], tolerance=0), other)
        self.assertIsNone(reach.select([0.25, 0.4, 0.3], tolerance=0.04))
        self.assertEqual(reach.voxel_keys.shape, (2, 3))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "reach.npz"
            reach.save(path)
            loaded = ReachabilityMap.load(path)
            np.testing.assert_array_equal(loaded.xyz, reach.xyz)
            np.testing.assert_array_equal(loaded.joint_samples, reach.joint_samples)
        np.testing.assert_array_equal(robot.get_q(), original)

    def test_sample_map_filters_collisions_and_is_reproducible(self):
        robot = FakeRobot()
        original = robot.get_q()
        checker = CollisionChecker(robot)
        a = ReachabilityMap.build(robot, checker, samples=40, seed=7)
        b = ReachabilityMap.build(robot, checker, samples=40, seed=7)
        self.assertGreater(a.sample_count, 0)
        self.assertLess(a.sample_count, 40)
        np.testing.assert_array_equal(a.joint_samples, b.joint_samples)
        np.testing.assert_array_equal(robot.get_q(), original)
        self.assertTrue(all(checker(q) for q in a.joint_samples))

    def test_empty_map_does_not_establish_unreachability(self):
        robot = FakeRobot()
        checker = CollisionChecker(robot)
        reach = ReachabilityMap.from_joint_samples(robot, np.zeros((2, 6)), checker)
        self.assertEqual(reach.sample_count, 0)
        self.assertEqual(reach.voxel_keys.shape, (0, 3))
        self.assertIsNone(reach.select([0.0, 0.0, 0.0]))


if __name__ == "__main__":
    unittest.main()
