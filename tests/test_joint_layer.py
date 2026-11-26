"""Cross-runtime physics checks for the explanatory browser layer, not a controller."""

import json
from pathlib import Path
import shutil
import subprocess

import mujoco
import numpy as np
import pytest

from cbf_demo.joint_layer import REGIONS, build_joint_layer_data
from cbf_demo.load_model import analyze_static_load, feasible_payload_interval, _preserve_integration_state
from cbf_demo.robot import RebotModel
from cbf_demo.scenes import START, GOAL, SURFACE_EPSILON, build_field, clutter_scene


@pytest.fixture(scope="module")
def layer():
    obstacles = clutter_scene()
    robot = RebotModel(obstacles=obstacles, epsilon=SURFACE_EPSILON)
    field = build_field(obstacles)
    data = build_joint_layer_data(robot, field, [(0.0, START), (1.0, GOAL)])
    return robot, field, data


def _node_results(payload, script):
    executable = shutil.which("node")
    if executable is None:
        pytest.skip("Node is required for the browser/Python cross-runtime check")
    asset = Path(__file__).resolve().parents[1] / "assets" / "load_math.js"
    program = "const fs=require('fs');const math=require(process.argv[1]);" + script
    completed = subprocess.run([executable, "-e", program, str(asset)],
                               input=json.dumps(payload), text=True,
                               capture_output=True, check=True)
    return json.loads(completed.stdout)


def test_regions_partition_all_798_surface_samples_and_use_full_minima(layer):
    robot, field, data = layer
    names = np.asarray(robot.point_body_names)
    assert len(names) == 798
    assert data["assumptions"]["all_cbf_constraints_retained"] == len(names)
    grouped = [body for key, (_, bodies) in REGIONS.items() if key != "whole" for body in bodies]
    assert len(grouped) == len(set(grouped))
    assert set(grouped) == set(names)
    with _preserve_integration_state(robot.model, robot.data):
        for frame in data["frames"]:
            robot.set_q(frame["q_rad"])
            points, jacobians = robot.points_and_jacobians()
            values, gradients = field.sample(points)
            valid = field.is_valid(points)
            regions = frame["regions"]
            assert regions["whole"]["sample_count"] == 798
            assert sum(region["sample_count"] for key, region in regions.items() if key != "whole") == 798
            for key, (_, bodies) in REGIONS.items():
                indices = np.arange(len(names)) if bodies is None else np.flatnonzero(np.isin(names, bodies))
                region = regions[key]
                worst = int(indices[np.argmin(values[indices])])
                assert region["sample_count"] == len(indices)
                assert region["valid_count"] == int(valid[indices].sum())
                assert region["valid_count"] == region["sample_count"]
                assert region["critical_sample"] == worst
                assert region["critical_body"] == names[worst]
                assert region["h_min_m2"] == values[worst]
                np.testing.assert_allclose(region["critical_point_m"], points[worst], atol=1e-12)
                np.testing.assert_allclose(region["barrier_joint_sensitivity_m2_per_rad"],
                                           gradients[worst] @ jacobians[worst], atol=1e-12)
                # Display decimation must not reduce the analysis/constraint set.
                assert len(region["displayed_points_m"]) == len(indices[::4])


def test_build_layer_preserves_complete_integration_state(layer):
    robot, field, _ = layer
    with _preserve_integration_state(robot.model, robot.data):
        robot.set_q(START)
        robot.data.qvel[:] = np.linspace(-0.3, 0.4, robot.model.nv)
        robot.data.qfrc_applied[:] = np.linspace(1.0, 2.0, robot.model.nv)
        robot.data.xfrc_applied[-1] = [1.0, 2.0, 3.0, -0.1, -0.2, -0.3]
        robot.data.qacc_warmstart[:] = np.linspace(0.2, 0.8, robot.model.nv)
        robot.data.time = 8.75
        mujoco.mj_forward(robot.model, robot.data)
        specification = mujoco.mjtState.mjSTATE_INTEGRATION
        before = np.empty(mujoco.mj_stateSize(robot.model, specification))
        after = np.empty_like(before)
        mujoco.mj_getState(robot.model, robot.data, before, specification)
        gravity = robot.model.opt.gravity.copy()
        build_joint_layer_data(robot, field, [(1.2, GOAL), (3.0, START)])
        mujoco.mj_getState(robot.model, robot.data, after, specification)
        np.testing.assert_array_equal(after, before)
        np.testing.assert_array_equal(robot.model.opt.gravity, gravity)


def test_browser_load_evaluation_matches_exact_mujoco_com_offsets(layer):
    robot, _, data = layer
    cases, expected = [], []
    for frame in data["frames"]:
        for axis in range(3):
            for offset in (-0.12, 0.12):
                for mass, fraction in ((0.35, 0.7), (2.3, 0.5)):
                    com = np.zeros(3)
                    com[axis] = offset
                    expected.append(analyze_static_load(robot, frame["q_rad"],
                                                        payload_kg=mass, com_offset=com,
                                                        headroom_fraction=fraction))
                    cases.append(dict(frame=frame, caps=data["model_caps_nm"], mass=mass,
                                      fraction=fraction, offset=offset, axis=axis))
    results = _node_results(cases, """
      const cases=JSON.parse(fs.readFileSync(0,'utf8'));
      console.log(JSON.stringify(cases.map(c=>math.evaluate(c.frame,c.caps,c.mass,c.fraction,c.offset,c.axis))));
    """)
    assert len(results) == 24
    for actual, exact in zip(results, expected):
        for js_key, py_key in (("perKg", "tau_per_kg_nm"), ("payload", "tau_payload_nm"),
                               ("total", "tau_total_nm"), ("caps", "caps_nm"),
                               ("ratio", "utilization"), ("com", "payload_com_world_m")):
            np.testing.assert_allclose(actual[js_key], exact[py_key], atol=2e-11, rtol=2e-11)
        assert actual["bottleneck"] == exact["joint_names"].index(exact["limiting_joint"])
        assert actual["feasible"] == exact["static_load_feasible"]
        interval = exact["payload_interval"]
        assert actual["capacity"]["feasible"] == interval["feasible"]
        assert actual["capacity"]["zeroFeasible"] == interval["zero_payload_feasible"]
        for js_key, py_key in (("minimumKg", "minimum_kg"), ("maximumKg", "maximum_kg")):
            if interval[py_key] is None:
                assert actual["capacity"][js_key] is None
            else:
                assert actual["capacity"][js_key] == pytest.approx(interval[py_key], abs=2e-11)


def test_browser_interval_handles_counterweight_unbounded_and_impossible_cases():
    cases = [
        dict(gravity=[2.0], perKg=[-1.0], caps=[1.0]),
        dict(gravity=[0.2], perKg=[0.0], caps=[1.0]),
        dict(gravity=[2.0], perKg=[0.0], caps=[1.0]),
        dict(gravity=[2.0, 5.0], perKg=[-1.0, -1.0], caps=[1.0, 1.0]),
        dict(gravity=[2.0], perKg=[1.0], caps=[1.0]),
        dict(gravity=[0.0], perKg=[1.0], caps=[0.0]),
        dict(gravity=[0.2, -0.3], perKg=[1.0, -2.0], caps=[1.2, 1.7]),
    ]
    results = _node_results(cases, """
      const cases=JSON.parse(fs.readFileSync(0,'utf8'));
      console.log(JSON.stringify(cases.map(c=>math.interval(c.gravity,c.perKg,c.caps))));
    """)
    for case, actual in zip(cases, results):
        expected = feasible_payload_interval(case["gravity"], case["perKg"], case["caps"])
        assert actual["feasible"] == expected["feasible"]
        assert actual["zeroFeasible"] == expected["zero_payload_feasible"]
        assert actual["unbounded"] == expected["maximum_unbounded"]
        assert actual["minimumKg"] == expected["minimum_kg"]
        assert actual["maximumKg"] == expected["maximum_kg"]
        for js_key, py_key in (("upperJoint", "limiting_upper_joint"),
                               ("lowerJoint", "limiting_lower_joint")):
            joint = expected[py_key]
            assert actual[js_key] == (None if joint is None else int(joint.replace("joint", ""))-1)
