"""Independent inverse-statics payload estimates for the fixed-base B601.

For an attached point mass with world COM p(q), gravitational force is m*g.
The *holding* torque is

    tau_hold(q, m) = qfrc_bias(q, qdot=0) - m * J_COM(q).T @ g_world.

This estimates static gravity demand using the actual vendor inertials and
kinematics. It is not a hardware payload rating, a grasp test, a dynamic torque
constraint, or a change to the collision filter/controller. In particular an
object's collision geometry and inertial dynamics are not added to the plant.

MuJoCo references:
https://mujoco.readthedocs.io/en/stable/APIreference/APIfunctions.html#mj-jac
https://mujoco.readthedocs.io/en/stable/programming/simulation.html#integration-state
"""
from __future__ import annotations

from contextlib import contextmanager
import math

import mujoco
import numpy as np


JOINT_NAMES = tuple(f'joint{i}' for i in range(1, 7))
JOINT_GROUPS = {'base': (0,), 'shoulder': (1,), 'elbow': (2,), 'wrist': (3, 4, 5)}
JOINT_ROLE_LABELS = (
    'Base rotation', 'Shoulder-like proximal hinge', 'Elbow-like middle hinge',
    'Wrist-like distal hinge 1', 'Wrist-like distal hinge 2', 'Wrist-like distal hinge 3',
)


@contextmanager
def _preserve_integration_state(model, data):
    """Preserve every forward-dynamics input, not only q and qdot.

    mjSTATE_INTEGRATION includes time, applied forces, controls and warmstarts.
    Derived quantities are recomputed after restoring the inputs. The final
    setState also preserves exact integration inputs if a future solver version
    changes a warmstart during that recomputation. No physics step occurs.
    """
    specification = mujoco.mjtState.mjSTATE_INTEGRATION
    saved = np.empty(mujoco.mj_stateSize(model, specification))
    mujoco.mj_getState(model, data, saved, specification)
    try:
        yield
    finally:
        mujoco.mj_setState(model, data, saved, specification)
        mujoco.mj_forward(model, data)
        mujoco.mj_setState(model, data, saved, specification)


def feasible_payload_interval(tau0, per_kg, caps, *, joint_names=None):
    """Intersect ``abs(tau0 + mass*per_kg) <= caps`` with ``mass >= 0``.

    The returned interval is exact for the supplied affine static model. Loads
    can counteract robot gravity: m=0 may be infeasible even when a positive
    interval exists. Such a case is explicitly identified and must not be
    described as an ordinary zero-to-maximum supported payload.

    ``maximum_kg=None`` denotes an unbounded interval only when feasible=True;
    both bounds are None when no nonnegative mass is feasible. No infinity or
    NaN is put in the returned JSON-compatible dictionary.
    """
    tau0, per_kg, caps = (np.asarray(value, dtype=float) for value in (tau0, per_kg, caps))
    if (tau0.ndim != 1 or not len(tau0) or per_kg.shape != tau0.shape or caps.shape != tau0.shape
            or not all(np.all(np.isfinite(value)) for value in (tau0, per_kg, caps))
            or np.any(caps < 0)):
        raise ValueError('tau0, per_kg and caps must be matching finite vectors with nonnegative caps')
    names = tuple(joint_names) if joint_names is not None else tuple(f'joint{i+1}' for i in range(len(tau0)))
    if len(names) != len(tau0):
        raise ValueError('joint_names must match the torque vectors')
    minimum, maximum = 0.0, math.inf
    lower_joint, upper_joint = None, None
    zero_feasible = bool(np.all(np.abs(tau0) <= caps))
    impossible_constant_joints = []
    for name, offset, slope, cap in zip(names, tau0, per_kg, caps):
        if slope == 0.0:
            if abs(offset) > cap:
                impossible_constant_joints.append(name)
            continue
        a, b = (-cap-offset)/slope, (cap-offset)/slope
        lower, upper = min(a, b), max(a, b)
        if lower > minimum:
            minimum, lower_joint = float(lower), name
        if upper < maximum:
            maximum, upper_joint = float(upper), name
    feasible = not impossible_constant_joints and maximum >= minimum
    if not feasible:
        status = 'no_nonnegative_payload_interval'
    elif not zero_feasible:
        status = 'counterweight_only_interval_zero_payload_infeasible'
    elif math.isinf(maximum):
        status = 'unbounded_in_this_static_model'
    else:
        status = 'zero_to_maximum_static_model_interval'
    return {
        'feasible': bool(feasible),
        'minimum_kg': float(minimum) if feasible else None,
        'maximum_kg': float(maximum) if feasible and math.isfinite(maximum) else None,
        'maximum_unbounded': bool(feasible and math.isinf(maximum)),
        'zero_payload_feasible': zero_feasible,
        'limiting_lower_joint': lower_joint,
        'limiting_upper_joint': upper_joint,
        'constant_over_cap_joints': impossible_constant_joints,
        'status': status,
        'hardware_payload_rating': False,
    }


def analyze_static_load(robot, q=None, *, payload_kg=0.0, com_offset=(0.0, 0.0, 0.0),
                        headroom_fraction=0.7):
    """Return a JSON-compatible static loading report without advancing physics.

    ``com_offset`` is the payload COM in metres in the *current TCP frame*.
    The output contains actual MuJoCo joint origins/axes and the translational
    COM Jacobian; no biological shoulder/elbow/wrist geometry is substituted.
    Headroom scales URDF model effort limits; it is an illustrative reserve,
    not a validated continuous-duty or safety factor.
    """
    payload_kg, headroom_fraction = float(payload_kg), float(headroom_fraction)
    offset = np.asarray(com_offset, dtype=float)
    if not math.isfinite(payload_kg) or payload_kg < 0:
        raise ValueError('payload_kg must be finite and nonnegative')
    if not math.isfinite(headroom_fraction) or not 0 < headroom_fraction <= 1:
        raise ValueError('headroom_fraction must lie in (0, 1]')
    if offset.shape != (3,) or not np.all(np.isfinite(offset)):
        raise ValueError('com_offset must contain three finite TCP-frame coordinates')
    model, data = robot.model, robot.data
    q = np.asarray(robot.get_q() if q is None else q, dtype=float)
    if q.shape != (6,) or not np.all(np.isfinite(q)):
        raise ValueError('q must contain six finite B601 joint angles')
    if np.any(q < robot.lower) or np.any(q > robot.upper):
        raise ValueError('q is outside the model joint limits')
    limits = np.asarray(robot.torque_limits, dtype=float)
    if limits.shape != (6,) or not np.all(np.isfinite(limits)) or np.any(limits <= 0):
        raise ValueError('robot torque_limits must be six finite positive model caps')
    caps = headroom_fraction*limits
    gravity = np.asarray(model.opt.gravity, dtype=float).copy()
    tcp_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, 'tcp')
    if tcp_id < 0:
        raise ValueError('robot model has no tcp site')
    tcp_body = int(model.site_bodyid[tcp_id])
    joint_ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name) for name in JOINT_NAMES]
    if any(joint < 0 for joint in joint_ids):
        raise ValueError('robot model is missing an arm joint')
    with _preserve_integration_state(model, data):
        data.qpos[robot.qpos_indices] = q
        data.qvel[:] = 0.0
        mujoco.mj_forward(model, data)
        tau_robot = data.qfrc_bias[robot.dof_indices].copy()
        tcp_position = data.site_xpos[tcp_id].copy()
        tcp_rotation = data.site_xmat[tcp_id].reshape(3, 3).copy()
        world_com = tcp_position + tcp_rotation @ offset
        jacp, jacr = np.zeros((3, model.nv)), np.zeros((3, model.nv))
        mujoco.mj_jac(model, data, jacp, jacr, world_com, tcp_body)
        jacobian = jacp[:, robot.dof_indices].copy()
        origins, axes = data.xanchor[joint_ids].copy(), data.xaxis[joint_ids].copy()
    per_kg = -jacobian.T @ gravity
    tau_payload = payload_kg*per_kg
    total = tau_robot + tau_payload
    utilization = np.abs(total)/caps
    radius = world_com[None, :] - origins
    perpendicular_radius = np.linalg.norm(np.cross(axes, radius), axis=1)
    gravity_norm = float(np.linalg.norm(gravity))
    signed_moment_arm = per_kg/gravity_norm if gravity_norm > 0 else np.zeros(6)
    interval = feasible_payload_interval(tau_robot, per_kg, caps, joint_names=JOINT_NAMES)
    feasible = bool(np.all(np.abs(total) <= caps))
    limiting = int(np.argmax(utilization))
    return {
        'analysis_kind': 'fixed_pose_static_gravity_only_model',
        'q_rad': q.tolist(), 'joint_names': list(JOINT_NAMES),
        'payload_kg': payload_kg, 'com_offset_tcp_m': offset.tolist(),
        'gravity_world_m_s2': gravity.tolist(),
        'weight_force_world_n': (payload_kg*gravity).tolist(),
        'support_force_world_n': (-payload_kg*gravity).tolist(),
        'tcp_world_m': tcp_position.tolist(),
        'tcp_rotation_world_from_tcp': tcp_rotation.tolist(),
        'payload_com_world_m': world_com.tolist(),
        'point_jacobian_m_per_rad': jacobian.tolist(),
        'tau_robot_nm': tau_robot.tolist(),
        'tau_per_kg_nm': per_kg.tolist(),
        'tau_payload_nm': tau_payload.tolist(),
        'tau_total_nm': total.tolist(),
        'model_torque_limits_nm': limits.tolist(),
        'headroom_fraction': headroom_fraction,
        'caps_nm': caps.tolist(),
        'utilization': utilization.tolist(),
        'torque_reserve_nm': (caps-np.abs(total)).tolist(),
        'limiting_joint': JOINT_NAMES[limiting],
        'static_load_feasible': feasible,
        'payload_interval': interval,
        'joint_origins_world_m': origins.tolist(),
        'joint_axes_world': axes.tolist(),
        'signed_gravity_moment_arm_m': signed_moment_arm.tolist(),
        'axis_to_com_perpendicular_distance_m': perpendicular_radius.tolist(),
        'joint_role_labels': list(JOINT_ROLE_LABELS),
        'joint_group_indices': {name: list(indices) for name, indices in JOINT_GROUPS.items()},
        'group_max_utilization': {name: float(np.max(utilization[list(indices)]))
                                  for name, indices in JOINT_GROUPS.items()},
        'anatomical_names_are_analogies': True,
        'hardware_payload_rating': False,
        'robot_model_mass_total_kg': float(np.sum(model.body_mass)),
        'physics_advanced': False,
        'limitations': [
            'URDF effort limits and link inertials are model inputs, not validated hardware ratings.',
            'Headroom is an illustrative fraction, not a certified safety factor.',
            'No acceleration, impact, contact wrench, friction, thermal or voltage limits are modeled.',
            'No gripper friction, finger-force, grasp stability or object collision check is performed.',
            'An offset COM is an attached point mass; object rotational inertia and geometry are not added.',
            'Mobile-base stability, wheel-ground contact, tipping and base motion are not modeled.',
            'A counterweight-only interval does not establish a zero-to-maximum payload capacity.',
        ],
    }
