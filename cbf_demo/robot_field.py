"""Robot-centred distance envelopes, distinct from the environment Poisson PDE.

For world point x and articulated proxy boxes B_b(q),

    d_R(x,q) = min_b sdf_OBB(x; B_b(q)),     b_R = d_R - margin.

Outside the union this is its Euclidean distance. Inside overlapping boxes the
minimum of their signed distances is NOT generally the distance to the union's
exterior boundary. It is nonpositive on the union; an internal seam between
merely touching boxes can be zero even inside that union. Box faces, edges,
corners and nearest-body ties make this field piecewise differentiable. Display
meshes do not turn it into a smooth CBF or certify collision-free motion.

All lengths and distance offsets here are in metres, while the environment's
Poisson h has units m^2. No control, ROS or physical hardware is invoked.
"""
from __future__ import annotations

import math

import mujoco
import numpy as np

from .load_model import _preserve_integration_state


ENVELOPE_COLORS = ('#E69F00', '#56B4E9', '#0072B2')


def world_proxy_boxes(robot, q=None):
    """Return copied world OBBs, preserving complete integration state."""
    boxes = []
    with _preserve_integration_state(robot.model, robot.data):
        if q is not None:
            robot.set_q(q)
        else:
            mujoco.mj_forward(robot.model, robot.data)
        for proxy, geom_id in zip(robot.proxy_boxes, robot.proxy_geom_ids):
            body_id = mujoco.mj_name2id(robot.model, mujoco.mjtObj.mjOBJ_BODY, proxy.body_name)
            boxes.append({
                'body_name': proxy.body_name, 'body_id': int(body_id), 'geom_id': int(geom_id),
                'center': robot.data.geom_xpos[geom_id].copy(),
                'half_size': robot.model.geom_size[geom_id].copy(),
                'rotation': robot.data.geom_xmat[geom_id].reshape(3, 3).copy(),
            })
    return boxes


def _validated_boxes(boxes):
    result = []
    for index, box in enumerate(boxes):
        center = np.asarray(box['center'], dtype=float)
        half_size = np.asarray(box['half_size'], dtype=float)
        rotation = np.asarray(box.get('rotation', np.eye(3)), dtype=float)
        if (center.shape != (3,) or half_size.shape != (3,) or rotation.shape != (3, 3)
                or not all(np.all(np.isfinite(a)) for a in (center, half_size, rotation))
                or np.any(half_size <= 0)
                or not np.allclose(rotation.T@rotation, np.eye(3), atol=1e-7)
                or not np.isclose(np.linalg.det(rotation), 1., atol=1e-7)):
            raise ValueError('Each OBB needs a finite center, positive half_size and proper orthogonal rotation')
        result.append((str(box.get('body_name', f'box_{index}')), center, half_size, rotation))
    if not result:
        raise ValueError('At least one robot proxy box is required')
    return result


def sample_robot_distance(worldpoints, boxes):
    """Return ``(distance[N], closest_body[N], gradient_world[N,3])``.

    Gradients are analytic at differentiable points. At an interior face tie or
    nearest-body tie a deterministic first candidate is selected; that chosen
    one-sided gradient is not a unique derivative. A single (3,) point is
    accepted and still returns arrays of length one.
    """
    points = np.asarray(worldpoints, dtype=float)
    if points.shape == (3,):
        points = points[None, :]
    if points.ndim != 2 or points.shape[1] != 3 or not np.all(np.isfinite(points)):
        raise ValueError('worldpoints must be finite with shape (N,3) or (3,)')
    validated = _validated_boxes(boxes)
    distance = np.full(len(points), math.inf)
    closest = np.empty(len(points), dtype=object)
    gradient = np.zeros((len(points), 3))
    for name, center, half_size, rotation in validated:
        local = (points-center) @ rotation
        excess = np.abs(local)-half_size
        outside_vector = np.maximum(excess, 0.)
        outside_norm = np.linalg.norm(outside_vector, axis=1)
        values = outside_norm + np.minimum(np.max(excess, axis=1), 0.)
        signs = np.where(local < 0, -1., 1.)
        local_gradient = np.divide(outside_vector*signs, outside_norm[:, None],
                                   out=np.zeros_like(local), where=outside_norm[:, None] > 0)
        interior = outside_norm == 0
        if np.any(interior):
            rows = np.flatnonzero(interior)
            face_axes = np.argmax(excess[interior], axis=1)
            local_gradient[rows, face_axes] = signs[rows, face_axes]
        choose = values < distance
        distance[choose] = values[choose]
        closest[choose] = name
        gradient[choose] = local_gradient[choose] @ rotation.T
    return distance, closest, gradient


def build_robot_distance_grid(boxes, *, bounds=None, spacing=.025, padding=.16,
                              max_nodes=2_000_000):
    """Sample d_R on a world XYZ grid, without changing robot/PDE geometry."""
    validated = _validated_boxes(boxes)
    spacing, padding = float(spacing), float(padding)
    if not math.isfinite(spacing) or spacing <= 0 or not math.isfinite(padding) or padding < 0:
        raise ValueError('spacing must be positive and padding nonnegative')
    if bounds is None:
        lower, upper = [], []
        for _, center, half_size, rotation in validated:
            extent = np.abs(rotation) @ half_size
            lower.append(center-extent); upper.append(center+extent)
        bounds = np.array([np.min(lower, axis=0)-padding, np.max(upper, axis=0)+padding])
    bounds = np.asarray(bounds, dtype=float)
    if bounds.shape != (2, 3) or not np.all(np.isfinite(bounds)) or np.any(bounds[1] <= bounds[0]):
        raise ValueError('bounds must contain finite increasing lower/upper XYZ corners')
    shape = np.ceil((bounds[1]-bounds[0])/spacing).astype(int)+1
    count = int(np.prod(shape))
    if count > max_nodes:
        raise ValueError(f'Display grid has {count} nodes, above max_nodes={max_nodes}')
    axes = tuple(bounds[0, axis]+spacing*np.arange(shape[axis]) for axis in range(3))
    xx, yy, zz = np.meshgrid(*axes, indexing='ij')
    points = np.column_stack([xx.ravel(), yy.ravel(), zz.ravel()])
    values = np.empty(count)
    for begin in range(0, count, 32768):
        end = min(begin+32768, count)
        values[begin:end] = sample_robot_distance(points[begin:end], boxes)[0]
    return {'distance': values.reshape(tuple(shape)), 'origin': bounds[0].copy(),
            'spacing': spacing, 'axes': axes,
            'bounds': np.array([bounds[0], [axis[-1] for axis in axes]]),
            'units': 'm', 'inside_overlap_is_true_union_distance': False,
            'display_only': True}


def build_robot_shells(grid, offsets=(.02, .05, .10), *, step_size=2):
    """Extract d_R=offset display shells; these grids never define a filter."""
    from .field_visualization import build_field_meshes
    from types import SimpleNamespace
    offsets = tuple(float(offset) for offset in offsets)
    if any(not math.isfinite(offset) or offset <= 0 for offset in offsets):
        raise ValueError('Envelope offsets must be positive finite metres')
    field = SimpleNamespace(h=np.asarray(grid['distance']), origin=grid['origin'], spacing=grid['spacing'])
    meshes = build_field_meshes(field, levels=offsets, step_size=step_size)
    result = []
    ordered = sorted(set(offsets))
    for mesh in meshes:
        index = ordered.index(mesh['level'])
        result.append({'offset': mesh['level'], 'margin': mesh['level'],
                       'vertices': mesh['vertices'], 'faces': mesh['faces'],
                       'color': ENVELOPE_COLORS[min(index, len(ENVELOPE_COLORS)-1)],
                       'units': 'm', 'display_only': True})
    return result


def robot_distance_meshes(robot, q=None, margins=(.02, .05, .10), *, spacing=.025, step_size=2):
    """Convenience API for parallel articulated-envelope visualizations."""
    boxes = world_proxy_boxes(robot, q)
    grid = build_robot_distance_grid(boxes, spacing=spacing, padding=max(margins)+2*spacing)
    return build_robot_shells(grid, offsets=margins, step_size=step_size)


def pair_clearance_constraints(robot, q=None, *, minimum_distance=1e-5):
    """Independent box-pair distances and first-order joint gradients.

    For positive separation d and MuJoCo witnesses p_R,p_O, the normal points
    from obstacle to robot: n=(p_R-p_O)/d. With a static obstacle,
    d_dot=n.T J_R(p_R) qdot. A relative-degree-one distance CBF would impose
    this gradient times qdot >= -alpha*(d-margin). No QP is solved here.

    Queries at/inside contact are rejected: their witness-normal derivative is
    not certified by this routine. ``valid`` concerns the local pair query,
    never continuous-time, actuator-tracking, or whole-controller safety.
    Closest features can switch, so gradients are piecewise differentiable.
    """
    minimum_distance = float(minimum_distance)
    if not math.isfinite(minimum_distance) or minimum_distance <= 0:
        raise ValueError('minimum_distance must be positive and finite')
    distances, gradients, robot_points, obstacle_points = [], [], [], []
    normals, bodies, obstacle_names, valid_pairs = [], [], [], []
    with _preserve_integration_state(robot.model, robot.data):
        if q is not None:
            robot.set_q(q)
        else:
            mujoco.mj_forward(robot.model, robot.data)
        for proxy, robot_geom in zip(robot.proxy_boxes, robot.proxy_geom_ids):
            body_id = int(robot.model.geom_bodyid[robot_geom])
            for obstacle, obstacle_geom in zip(robot.obstacles, robot.obstacle_geom_ids):
                witness = np.zeros(6)
                distance = float(mujoco.mj_geomDistance(robot.model, robot.data, int(robot_geom),
                                                       int(obstacle_geom), 100., witness))
                normal = np.zeros(3)
                valid = math.isfinite(distance) and minimum_distance < distance < 100.
                if valid:
                    normal = (witness[:3]-witness[3:])/distance
                    valid = bool(np.all(np.isfinite(normal)) and abs(np.linalg.norm(normal)-1.) < 1e-5)
                jacp, jacr = np.zeros((3, robot.model.nv)), np.zeros((3, robot.model.nv))
                if valid:
                    mujoco.mj_jac(robot.model, robot.data, jacp, jacr, witness[:3], body_id)
                gradients.append(normal @ jacp[:, robot.dof_indices] if valid else np.zeros(6))
                distances.append(distance); robot_points.append(witness[:3].copy())
                obstacle_points.append(witness[3:].copy()); normals.append(normal)
                bodies.append(proxy.body_name); obstacle_names.append(obstacle['name']); valid_pairs.append(valid)
    return {'valid': bool(all(valid_pairs)), 'pair_valid': np.asarray(valid_pairs, dtype=bool),
            'distances_m': np.asarray(distances), 'gradients_q': np.asarray(gradients).reshape(-1, 6),
            'robot_witness_world_m': np.asarray(robot_points).reshape(-1, 3),
            'obstacle_witness_world_m': np.asarray(obstacle_points).reshape(-1, 3),
            'normal_obstacle_to_robot_world': np.asarray(normals).reshape(-1, 3),
            'robot_bodies': np.asarray(bodies), 'obstacle_names': np.asarray(obstacle_names),
            'minimum_valid_distance_m': minimum_distance,
            'self_collision_checked': False, 'obstacle_motion_modeled': False,
            'formal_safety_certificate': False}
