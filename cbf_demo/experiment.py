"""Reachability -> sampled collision planning -> velocity CBF -> MuJoCo PD.

Independent full-box clearance is measured on the actual integrated state.
Episodes abort on contact, invalid field state, or failed QP; an abort is a
failure/replanning request, never counted as successful safe manipulation.
"""
from dataclasses import asdict
import csv
import json
from pathlib import Path
from time import perf_counter, sleep

import mujoco
import numpy as np

from .controller import PoissonCBFFilter
from .planning import CollisionChecker, ReachabilityMap, RRTConnect, shortcut_path
from .robot import RebotModel
from .scenes import START, GOAL, SURFACE_EPSILON, clutter_scene, build_field


def prepare(seed=7, map_samples=250, spacing=.025):
    obstacles = clutter_scene()
    robot = RebotModel(obstacles=obstacles, epsilon=SURFACE_EPSILON)
    field = build_field(obstacles, spacing=spacing)
    robot.set_q(START)
    checker = CollisionChecker(robot, field=field, clearance_margin=.015)
    if not checker(START) or not checker(GOAL):
        raise RuntimeError('Demonstration endpoint rejected: ' + checker.last_reason)
    rng = np.random.default_rng(seed)
    samples = rng.uniform(robot.lower+.02, robot.upper-.02, (map_samples, 6))
    # Known task seeds are disclosed, not presented as independent random IK.
    samples = np.vstack([samples, START, GOAL])
    reach = ReachabilityMap.from_joint_samples(robot, samples, checker)
    target = robot.tcp(GOAL)
    selected = reach.select(target, tolerance=1e-8)
    if selected is None:
        raise RuntimeError('Target has no valid sampled reachability seed')
    robot.set_q(START)
    return robot, field, checker, reach, selected, target


def plan_path(robot, checker, goal, seed):
    result = RRTConnect(max_iterations=3500, step_size=.3,
                        edge_resolution=.035, seed=seed).plan(START, goal, checker)
    if result.success:
        result.path = shortcut_path(result.path, checker, edge_resolution=.025,
                                    attempts=100, seed=seed)
    return result


def run_episode(robot, field, goal, target, *, mode='hierarchy', path=None,
                plant='dynamics', seconds=35., dt=.02, viewer=False, ros=False,
                hold=False, gain=1.6):
    if mode not in ('unfiltered', 'filter-only', 'hierarchy'):
        raise ValueError('unknown experiment mode')
    if plant not in ('dynamics', 'kinematic') or seconds <= 0 or dt <= 0:
        raise ValueError('invalid plant/time configuration')
    robot.set_q(START)
    robot.data.qvel[:] = 0
    robot.data.time = 0
    mujoco.mj_forward(robot.model, robot.data)
    path = np.vstack([START, goal]) if path is None else np.array(path)
    if mode != 'hierarchy':
        path = np.vstack([START, goal])
    route_xyz = np.array([robot.tcp(q) for q in path])
    robot.set_q(START)
    safety = PoissonCBFFilter(field, robot.lower, robot.upper,
                             alpha=2., speed=.45, h_reserve=2e-5)
    bridge = None
    if ros:
        from .ros_bridge import ROSBridge
        bridge = ROSBridge(robot.obstacles, path=route_xyz, field=field)
    window = None
    if viewer:
        from mujoco import viewer as mjviewer
        window = mjviewer.launch_passive(robot.model, robot.data)
        window.cam.lookat[:] = [0., 0., .38]
        window.cam.distance = 1.9
        window.cam.azimuth, window.cam.elevation = 120, -25
        window.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = True
        window.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTFORCE] = True
        print('MuJoCo viewer ready; controller uses actual simulated joint feedback.',flush=True)
    rows, status = [], 'running'
    episode_peak_torque = np.zeros(6)
    episode_torque_sumsq = np.zeros(6)
    torque_sample_count = 0
    waypoint = 1 if len(path) > 1 else 0
    steps = int(round(seconds/dt))
    substeps = max(1, int(round(dt/robot.model.opt.timestep)))
    if not np.isclose(substeps*robot.model.opt.timestep, dt):
        raise ValueError('control dt must be an integer multiple of physics timestep')
    wall_begin = perf_counter()
    try:
        for k in range(steps):
            cycle_begin = perf_counter()
            if window is not None and not window.is_running():
                status = 'viewer_closed'
                break
            q, qd = robot.get_q(), robot.get_qd()
            if np.max(np.abs(path[waypoint]-q)) < .035 and waypoint < len(path)-1:
                waypoint += 1
            nominal = np.clip(gain*(path[waypoint]-q), -.45, .45)
            points, jacobians = robot.points_and_jacobians()
            h, _ = field.sample(points)
            if mode == 'unfiltered':
                v, qp_ms, correction, residual, qp_ok = nominal, 0., 0., float('nan'), True
                qp_rows = 0
                fallback_used = False
            else:
                filtered = safety.filter(q, nominal, points, jacobians)
                v, qp_ms = filtered.velocity, filtered.solve_ms
                correction, residual, qp_ok = filtered.correction_norm, filtered.minimum_residual, filtered.success
                qp_rows = filtered.qp_constraints
                fallback_used = filtered.fallback_used
                if not qp_ok:
                    status = filtered.status
            tau = np.zeros(6)
            if qp_ok:
                if plant == 'kinematic':
                    robot.set_q(q + dt*v)
                    robot.data.qvel[:] = v
                    robot.data.time += dt
                    mujoco.mj_forward(robot.model, robot.data)
                    tau = robot.data.qfrc_bias[robot.dof_indices].copy()
                else:
                    # Use actual physics q as the next integrated target origin;
                    # never trust commanded/fake joint state as feedback.
                    tau = robot.step(q + dt*v, v, substeps=substeps)
            actual_q, actual_qd = robot.get_q(), robot.get_qd()
            clearance = robot.clearance()
            contacts = robot.environment_contacts()
            if qp_ok and plant == 'dynamics':
                clearance = min(clearance, robot.last_step_min_clearance)
                contacts = max(contacts, robot.last_step_max_contacts)
                episode_peak_torque = np.maximum(episode_peak_torque, robot.last_step_peak_torque)
                episode_torque_sumsq += robot.last_step_torque_sumsq
                torque_sample_count += robot.last_step_substeps
            else:
                episode_peak_torque = np.maximum(episode_peak_torque, np.abs(tau))
                episode_torque_sumsq += tau**2
                torque_sample_count += 1
            actual_points = robot.points()
            actual_h, _ = field.sample(actual_points)
            tcp = robot.tcp()
            goal_error = float(np.linalg.norm(tcp-target))
            joint_error = float(np.max(np.abs(actual_q-goal)))
            point_speed_error = float(np.linalg.norm(
                np.einsum('nij,j->ni', jacobians, actual_qd-v), axis=1).max())
            row = dict(t=float(robot.data.time), clearance_m=clearance, min_h_m2=float(actual_h.min()),
                       qp_ms=qp_ms, qp_residual=residual, correction=correction,
                       qp_rows=qp_rows, all_surface_constraints=len(points),
                       qp_fallback=int(fallback_used),
                       goal_error_m=goal_error, contacts=contacts,
                       waypoint=waypoint, point_velocity_error_m_s=point_speed_error,
                       cycle_ms=(perf_counter()-cycle_begin)*1000)
            for i in range(6):
                row[f'q{i+1}'] = float(actual_q[i])
                row[f'qd{i+1}'] = float(actual_qd[i])
                row[f'tau{i+1}'] = float(tau[i])
                row[f'v{i+1}'] = float(v[i])
            row.update(tcp_x=float(tcp[0]), tcp_y=float(tcp[1]), tcp_z=float(tcp[2]))
            rows.append(row)
            if bridge is not None and k % 2 == 0:
                bridge.publish(actual_q, actual_qd, tau, tcp, actual_points, actual_h,
                               dict(mode=mode, plant=plant, status=status,
                                    **{key: val for key, val in row.items() if not key.startswith(('q', 'tau', 'v'))}))
            if window is not None:
                window.sync()
                sleep(max(0., dt-(perf_counter()-cycle_begin)))
            if clearance <= 0 or contacts:
                status = 'collision'
                break
            if not qp_ok:
                break
            if mode != 'unfiltered' and not np.all(field.is_valid(actual_points)):
                status = 'tracking_left_buffered_domain'
                break
            if (waypoint == len(path)-1 and goal_error < .012
                    and joint_error < .035 and np.linalg.norm(actual_qd) < .08):
                status = 'goal_reached'
                break
            # Operational trigger for the no-planner comparison, not a claim
            # that the CBF problem is infeasible or the geometry is deadlocked.
            if mode == 'filter-only' and len(rows) >= 101 and goal_error > .02:
                prior = np.array([rows[-101][f'q{i+1}'] for i in range(6)])
                if np.linalg.norm(actual_q-prior) < .004:
                    status = 'progress_stalled_replan_required'
                    break
        if status == 'running':
            status = 'timeout'
        if hold and window is not None:
            print(f'Episode {status}; physics is frozen for inspection. Close the window or press Ctrl+C to save and exit.', flush=True)
            while window.is_running():
                window.sync()
                if bridge is not None:
                    p = robot.points(); hv, _ = field.sample(p)
                    bridge.publish(robot.get_q(), robot.get_qd(), robot.last_torques,
                                   robot.tcp(), p, hv, dict(status=status))
                sleep(dt)
    except KeyboardInterrupt:
        if status == 'running':
            status = 'interrupted'
        print('Saving collected simulation telemetry before exit.',flush=True)
    finally:
        if window is not None:
            window.close()
        if bridge is not None:
            bridge.close()
    if not rows:
        raise RuntimeError('No simulation samples collected')
    summary = dict(mode=mode, plant=plant, status=status, success=status=='goal_reached',
                   steps=len(rows), simulation_seconds=float(robot.data.time),
                   wall_seconds=perf_counter()-wall_begin,
                   minimum_clearance_m=min(r['clearance_m'] for r in rows),
                   minimum_h_m2=min(r['min_h_m2'] for r in rows),
                   maximum_contacts=max(r['contacts'] for r in rows),
                   final_goal_error_m=rows[-1]['goal_error_m'],
                   filter_interventions=sum(r['correction']>1e-4 for r in rows),
                   qp_fallback_count=sum(r['qp_fallback'] for r in rows),
                   qp_mean_ms=float(np.mean([r['qp_ms'] for r in rows])),
                   qp_p95_ms=float(np.percentile([r['qp_ms'] for r in rows],95)),
                   cycle_p95_ms=float(np.percentile([r['cycle_ms'] for r in rows],95)),
                   peak_torque_nm=episode_peak_torque.tolist(),
                   rms_torque_nm=np.sqrt(episode_torque_sumsq/max(1,torque_sample_count)).tolist(),
                   torque_statistics_hz=1/robot.model.opt.timestep if plant=='dynamics' else 1/dt,
                   maximum_point_velocity_error_m_s=max(r['point_velocity_error_m_s'] for r in rows),
                   torque_limits_nm=robot.torque_limits.tolist(),
                   contact_force_control=False, self_collision_enforced=False,
                   orientation_task=False, grasp_task=False)
    summary['stall_policy'] = {'joint_displacement_norm_rad':.004,
                              'window_seconds':100*dt,
                              'applies_to':'filter-only'}
    return rows, summary


def save_run(directory, rows, summary, path=None):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    with (directory/'telemetry.csv').open('w',newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    (directory/'summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n')
    if path is not None:
        np.save(directory/'planned_joint_path.npy', path)


# Exported figures share the reusable SciencePlots policy.
from .plotting import plot_runs, plot_preparation  # noqa: E402,F401
