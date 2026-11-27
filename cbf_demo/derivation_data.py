"""Pose-specific, read-only mathematical diagnostics for the field view.

Selected points explain one inequality. They never replace the complete
Poisson-surface or proxy-pair constraint sets. Optional precomputed commands
allow exact raw minimum geometry residuals over all rows without rerunning a
QP or transmitting the complete surface-row matrix to the browser.
"""
from __future__ import annotations

import numpy as np

from .controller import PoissonCBFFilter
from .joint_layer import REGIONS
from .load_model import _preserve_integration_state
from .robot_field import pair_clearance_constraints, world_proxy_boxes


def _point_record(index, points, values, gradients, jacobians, names, valid, *, selection):
    index = int(index)
    return dict(sample_index=index, body_name=str(names[index]),
                point_world_m=points[index].tolist(), h_m2=float(values[index]),
                gradient_world_m=gradients[index].tolist(),
                jacobian_m_per_rad=jacobians[index].tolist(),
                a_m2_per_rad=(gradients[index]@jacobians[index]).tolist(),
                field_valid=bool(valid[index]), selection=selection)


def _select_diagnostic(indices, values, jacobians, valid):
    """Prefer a positive valid moving sample, disclosing this display choice."""
    moving = np.linalg.norm(jacobians[indices], axis=(1, 2)) > 1e-12
    moving_indices = indices[moving]
    candidates = moving_indices if len(moving_indices) else indices
    eligible = candidates[valid[candidates] & (values[candidates] > 0)]
    if len(eligible):
        candidates = eligible
        selection = ('minimum_valid_h_over_moving_samples' if len(moving_indices)
                     else 'minimum_valid_h_over_static_samples')
    else:
        selection = ('no_positive_valid_moving_sample' if len(moving_indices)
                     else 'no_positive_valid_static_sample')
    return int(candidates[np.argmin(values[candidates])]), selection, len(moving_indices)


def _minimum_residual(a, h, velocity, alpha, reserve=0., valid=True):
    if not valid or not len(h):
        return None
    velocity = np.asarray(velocity, dtype=float)
    if velocity.shape != (a.shape[1],) or not np.all(np.isfinite(velocity)):
        raise ValueError('Precomputed command velocities must be finite and match the joint dimension')
    return float(np.min(a@velocity+alpha*(h-reserve)))


def _command_residuals(command_frame, a, h, pair_a, distances, margins, controller,
                       poisson_valid, pair_valid):
    result = []
    for command in command_frame['commands']:
        if len(command['distance']) != len(margins):
            raise ValueError('Precomputed distance-command margins do not match derivation margins')
        nominal = command['nominal_rad_s']
        poisson = command['poisson']
        poisson_velocity = poisson['velocity_rad_s']
        nominal_poisson = _minimum_residual(a, h, nominal, controller.alpha,
                                           controller.h_reserve, poisson_valid)
        projected_poisson = _minimum_residual(a, h, poisson_velocity, controller.alpha,
                                             controller.h_reserve, poisson_valid and poisson['success'])
        nominal_pair, projected_pair, distance_poisson, poisson_pair = [], [], [], []
        for margin, distance_result in zip(margins, command['distance']):
            pair_h = distances-margin
            nominal_pair.append(_minimum_residual(pair_a, pair_h, nominal,
                                                 controller.alpha, valid=pair_valid))
            projected_pair.append(_minimum_residual(pair_a, pair_h, distance_result['velocity_rad_s'],
                                                   controller.alpha,
                                                   valid=pair_valid and distance_result['success']))
            distance_poisson.append(_minimum_residual(a, h, distance_result['velocity_rad_s'],
                                                     controller.alpha, controller.h_reserve,
                                                     poisson_valid and distance_result['success']))
            poisson_pair.append(_minimum_residual(pair_a, pair_h, poisson_velocity,
                                                 controller.alpha, valid=pair_valid and poisson['success']))
        result.append(dict(nominal_poisson_min_m2_s=nominal_poisson,
                           poisson_min_m2_s=projected_poisson,
                           nominal_pair_min_m_s=nominal_pair,
                           distance_min_m_s=projected_pair,
                           distance_output_poisson_min_m2_s=distance_poisson,
                           poisson_output_pair_min_m_s=poisson_pair,
                           poisson_command_accepted=bool(poisson['success']),
                           distance_commands_accepted=[bool(item['success']) for item in command['distance']]))
    return result


def build_derivation_data(robot, field, frames, *, margins=(.025, .05, .10), command_data=None):
    """Build compact JSON diagnostics, preserving full integration state.

    ``frames`` is an ordered sequence of ``(time_s, q6)`` pairs. The optional
    ``command_data`` is the matching output of build_command_comparison: frame
    counts, timestamps, joint states, margins and gain are verified before use.
    Only arithmetic residual evaluation is performed; no solver/step runs.
    """
    frames = list(frames)
    margins = tuple(float(value) for value in margins)
    if not frames:
        raise ValueError('At least one recorded pose is required')
    if not margins or len(set(margins)) != len(margins) or not np.all(np.isfinite(margins)) or min(margins) <= 0:
        raise ValueError('margins must be distinct finite positive distances')
    controller = PoissonCBFFilter(field, robot.lower, robot.upper)
    if command_data is not None:
        if (len(command_data['frames']) != len(frames)
                or not np.array_equal(np.asarray(command_data['margins_m']), np.asarray(margins))
                or not np.isclose(command_data['alpha_per_s'], controller.alpha, rtol=0, atol=1e-12)
                or not np.allclose(command_data['joint_speed_rad_s'], controller.speed, rtol=0, atol=1e-12)):
            raise ValueError('Command frame count, margins, gain or speed do not match derivation data')
    names = np.asarray(robot.point_body_names)
    records = []
    with _preserve_integration_state(robot.model, robot.data):
        for frame_index, (time_s, q) in enumerate(frames):
            q = np.asarray(q, dtype=float)
            if q.shape != (6,) or not np.all(np.isfinite(q)) or not np.isfinite(time_s):
                raise ValueError('Frames require finite timestamps and six-element q vectors')
            command_frame = None if command_data is None else command_data['frames'][frame_index]
            if command_frame is not None:
                if (not np.isclose(command_frame['time_s'], time_s, rtol=0, atol=1e-10)
                        or not np.allclose(command_frame['q_rad'], q, rtol=0, atol=1e-10)):
                    raise ValueError('Precomputed commands and derivation pose/timestamp differ')
            robot.set_q(q)
            points, jacobians = robot.points_and_jacobians()
            if not np.all(np.isfinite(points)) or not np.all(np.isfinite(jacobians)):
                raise ValueError('Robot point/Jacobian diagnostics must be finite')
            values, gradients = field.sample(points)
            values, gradients = np.asarray(values), np.asarray(gradients)
            valid = np.asarray(field.is_valid(points), dtype=bool)
            if not np.all(np.isfinite(values)) or not np.all(np.isfinite(gradients)):
                raise ValueError('Field diagnostics must be finite')
            a = np.einsum('ni,nij->nj', gradients, jacobians)
            pairs = pair_clearance_constraints(robot)
            if not all(np.all(np.isfinite(pairs[key])) for key in
                       ('distances_m', 'gradients_q', 'robot_witness_world_m',
                        'obstacle_witness_world_m', 'normal_obstacle_to_robot_world')):
                raise ValueError('Pair diagnostic geometry must be finite')
            regions = {}
            pair_names = np.asarray(pairs['robot_bodies'])
            for key, (_, bodies) in REGIONS.items():
                indices = np.flatnonzero(np.ones(len(names), dtype=bool) if bodies is None else np.isin(names, bodies))
                if not len(indices):
                    raise ValueError(f'No surface samples in display region {key}')
                selected, selection, moving_count = _select_diagnostic(indices, values, jacobians, valid)
                pair_indices = np.flatnonzero(np.ones(len(pair_names), dtype=bool) if bodies is None else np.isin(pair_names, bodies))
                pair_index = (int(pair_indices[np.argmin(pairs['distances_m'][pair_indices])])
                              if len(pair_indices) else None)
                regions[key] = dict(poisson=_point_record(selected, points, values, gradients, jacobians,
                                                         names, valid, selection=selection),
                                    pair_index=pair_index, pair_indices=pair_indices.tolist(),
                                    sample_count=len(indices), moving_sample_count=moving_count,
                                    invalid_sample_count=int(np.count_nonzero(~valid[indices])),
                                    region_min_h_m2=float(values[indices].min()))
            by_body = {}
            for body in dict.fromkeys(names.tolist()):
                indices = np.flatnonzero(names == body)
                selected, selection, _ = _select_diagnostic(indices, values, jacobians, valid)
                by_body[str(body)] = _point_record(selected, points, values, gradients, jacobians,
                                                  names, valid, selection=selection)
            packed_pairs = {key: np.asarray(pairs[key]).tolist() for key in
                            ('distances_m', 'gradients_q', 'robot_witness_world_m',
                             'obstacle_witness_world_m', 'normal_obstacle_to_robot_world',
                             'robot_bodies', 'obstacle_names', 'pair_valid')}
            packed_pairs.update(valid=bool(pairs['valid']), minimum_valid_distance_m=pairs['minimum_valid_distance_m'],
                                gradient_units='m/rad', witness_frame='MuJoCo world')
            boxes = [dict(body_name=box['body_name'], center_m=box['center'].tolist(),
                          half_size_m=box['half_size'].tolist(), rotation_world_from_body=box['rotation'].tolist())
                     for box in world_proxy_boxes(robot)]
            record = dict(time_s=float(time_s), q_rad=q.tolist(), pairs=packed_pairs,
                          regions=regions, poisson_by_body=by_body, boxes=boxes,
                          poisson_all_samples_valid=bool(np.all(valid)),
                          poisson_controller_row_count=len(points), pair_row_count=len(pair_names),
                          poisson_global_min_h_m2=float(values.min()),
                          poisson_controller_critical_sample=int(np.argmin(values)),
                          numeric_tolerance=dict(normalized_residual_rad_s=2e-6,
                                                 raw_bounds_are_conservative_global_conversions=True,
                                                 poisson_raw_min_m2_s=2e-6*float(np.maximum(np.linalg.norm(a, axis=1), 1e-3).max()),
                                                 pair_raw_min_m_s=(2e-6*float(np.maximum(np.linalg.norm(pairs['gradients_q'], axis=1), 1e-3).max())
                                                                   if len(pair_names) else None)),
                          geometry_residual_scope='all original geometry inequalities; joint bounds are separate')
            if command_frame is not None:
                record['command_residuals'] = _command_residuals(command_frame, a, values,
                                                               np.asarray(pairs['gradients_q']),
                                                               np.asarray(pairs['distances_m']), margins,
                                                               controller, bool(np.all(valid)), bool(pairs['valid']))
            records.append(record)
    return dict(schema='rebot-pose-derivation-v1', frames=records, margins_m=list(margins),
                controller=dict(alpha_per_s=controller.alpha, poisson_h_reserve_m2=controller.h_reserve,
                                speed_rad_s=controller.speed.tolist(), joint_margin_rad=controller.joint_margin,
                                tracking_speed_bound_m_s=controller.tracking_speed_bound,
                                lower_rad=controller.lower.tolist(), upper_rad=controller.upper.tolist()),
                regions={key: dict(label=label, bodies=None if bodies is None else list(bodies))
                         for key, (label, bodies) in REGIONS.items()},
                obstacles=[dict(name=str(obstacle['name']),
                                center_m=np.asarray(obstacle['center'], dtype=float).tolist(),
                                half_size_m=np.asarray(obstacle['half_size'], dtype=float).tolist())
                           for obstacle in robot.obstacles],
                field=dict(shape=list(field.shape), origin_m=field.origin.tolist(), spacing_m=float(field.spacing),
                           requested_inflation_m=float(field.inflation),
                           inflation_plus_half_voxel_diagonal_m=float(field.padding),
                           numerical_collar_max_m=float(field.numerical_collar_max_m)),
                defaults=dict(region='elbow', margin_m=min(margins, key=lambda value: abs(value-.05))),
                assumptions=dict(physics_advanced=False, controllers_solved=False,
                                 all_poisson_controller_rows_retained=len(names),
                                 selected_points_are_diagnostic_only=True,
                                 obstacle_motion_modeled=False, self_collision_checked=False,
                                 pair_derivative_requires_positive_separation=True,
                                 globally_smooth_barrier=False),
                raw_residuals_from_precomputed_commands=command_data is not None)
