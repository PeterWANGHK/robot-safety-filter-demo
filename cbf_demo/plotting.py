"""Scientific exports, separate from simulation and real-time visualization.

All figures use the shared SciencePlots context and export PNG plus vector PDF.
The plotted telemetry does not certify physical safety, reachability, or motor
ratings. Simulation substep summaries and control-tick traces remain distinct.
"""
from collections import Counter
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use('Agg')
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import MaxNLocator
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from .plot_style import (BUFFER_COLOR, MODE_COLORS, MODE_LABELS, MODE_LINESTYLES,
                         OBSTACLE_COLOR, plt, save_figure, scientific_style)
from .scenes import clutter_scene


def _style(summary):
    mode = summary.get('mode', '')
    return dict(color=MODE_COLORS.get(mode, '#4477AA'),
                linestyle=MODE_LINESTYLES.get(mode, '-'))


def _label(summary, duplicates=None, status=False):
    mode = summary.get('mode', '')
    label = MODE_LABELS.get(mode, mode)
    if duplicates and duplicates[mode] > 1:
        label += f" (seed {summary.get('seed', '?')})"
    if status:
        label += ': ' + summary.get('status', '').replace('_', ' ')
    return label


def _array(rows, key):
    return np.asarray([row.get(key, np.nan) for row in rows], dtype=float)


def _grid(axis):
    axis.grid(True, which='major', alpha=.25)
    axis.xaxis.set_major_locator(MaxNLocator(5))


def _boxes(axis, obstacles, alpha=.13):
    for obstacle in obstacles:
        center, size = np.asarray(obstacle['center']), np.asarray(obstacle['half_size'])
        vertices = np.asarray([center + size*np.array([a, b, c])
                               for a in (-1, 1) for b in (-1, 1) for c in (-1, 1)])
        faces = [[vertices[i] for i in indices] for indices in
                 ([0, 1, 3, 2], [4, 5, 7, 6], [0, 1, 5, 4],
                  [2, 3, 7, 6], [0, 2, 6, 4], [1, 3, 7, 5])]
        axis.add_collection3d(Poly3DCollection(faces, facecolor=OBSTACLE_COLOR,
                                              edgecolor=OBSTACLE_COLOR,
                                              linewidth=.35, alpha=alpha))


def _set_equal_3d(axis, points, pad=.06):
    points = np.asarray(points, dtype=float)
    points = points[np.all(np.isfinite(points), axis=1)]
    if not len(points):
        return
    lower, upper = points.min(axis=0), points.max(axis=0)
    middle = (lower + upper)/2
    half = max(float((upper-lower).max())/2 + pad, .1)
    axis.set_xlim(middle[0]-half, middle[0]+half)
    axis.set_ylim(middle[1]-half, middle[1]+half)
    axis.set_zlim(middle[2]-half, middle[2]+half)
    axis.set_box_aspect((1, 1, 1))
    axis.view_init(elev=23, azim=-58)


def plot_runs(results, destination):
    """Export comparative traces plus a six-channel effort figure.

    Each input is ``(rows, summary)`` from one episode. Per-joint curves show
    control-tick actuator efforts; annotations use the physics-substep peak/RMS
    summaries when supplied. A failed/aborted episode is identified in the key.
    """
    results = [(rows, summary) for rows, summary in results if len(rows)]
    if not results:
        raise ValueError('Cannot plot an empty experiment result')
    destination = Path(destination)
    duplicates = Counter(summary.get('mode', '') for _, summary in results)
    with scientific_style():
        figure = plt.figure(figsize=(14.2, 8.6), layout='constrained')
        axes = [figure.add_subplot(231, projection='3d')]
        axes.extend(figure.add_subplot(232+i) for i in range(5))
        route, clearance, error, correction, timing, effort = axes
        _boxes(route, clutter_scene())
        all_positions = []
        for rows, summary in results:
            t = _array(rows, 't')
            label, style = _label(summary, duplicates), _style(summary)
            xyz = np.column_stack([_array(rows, key) for key in ('tcp_x', 'tcp_y', 'tcp_z')])
            all_positions.extend(xyz)
            route.plot(*xyz.T, label=label, **style)
            route.scatter(*xyz[0], c=style['color'], s=15, marker='o', depthshade=False)
            route.scatter(*xyz[-1], c=style['color'], s=28, marker='X', depthshade=False)
            clearance.plot(t, 100*_array(rows, 'clearance_m'), label=label, **style)
            error.plot(t, 100*_array(rows, 'goal_error_m'), label=label, **style)
            correction.plot(t, _array(rows, 'correction'), label=label, **style)
            timing.plot(t, _array(rows, 'qp_ms'), label=label, **style)
            caps = np.asarray(summary.get('torque_limits_nm', [27, 27, 27, 7, 7, 7]), dtype=float)
            torques = np.column_stack([_array(rows, f'tau{i}') for i in range(1, 7)])
            effort.plot(t, np.max(np.abs(torques)/caps, axis=1), label=label, **style)
        obstacle_extents = [np.asarray(o['center']) + np.array([s, t, u])*np.asarray(o['half_size'])
                            for o in clutter_scene() for s in (-1, 1)
                            for t in (-1, 1) for u in (-1, 1)]
        _set_equal_3d(route, np.vstack([all_positions, obstacle_extents]), pad=.025)
        route.set(xlabel='$x$ (m)', ylabel='$y$ (m)', zlabel='$z$ (m)',
                  title='(a) Actual tool path in static clutter')
        route.tick_params(labelsize=8)
        clearance.axhline(0, color='#333333', lw=1, ls=':', label='Collision boundary')
        clearance.set(xlabel='Simulation time (s)', ylabel='Minimum separation (cm)',
                      title='(b) Whole-arm box clearance')
        error.set(xlabel='Simulation time (s)', ylabel='Tool position error (cm)',
                  title='(c) Goal convergence')
        correction.set(xlabel='Simulation time (s)', ylabel=r'$\|v-v_{\mathrm{nom}}\|_2$ (rad/s)',
                       title='(d) CBF command intervention')
        timing.set(xlabel='Simulation time (s)', ylabel='QP duration (ms)',
                   title='(e) Measured QP computation')
        effort.axhline(1, color='#333333', lw=1, ls=':')
        effort.set(xlabel='Simulation time (s)', ylabel=r'$\max_j |\tau_j|/\tau_{j,\mathrm{cap}}$',
                   title='(f) Effort at control ticks')
        for axis in axes[1:]:
            _grid(axis)
            axis.set_xlim(left=0)
        handles = [Line2D([], [], label=_label(summary, duplicates, status=True), **_style(summary))
                   for _, summary in results]
        figure.legend(handles=handles, loc='lower center', bbox_to_anchor=(.5, -.11), ncol=1,
                      title='Episode outcomes (one synthetic scene; no statistical claim)',
                      title_fontsize=9)
        plant = ', '.join(sorted({summary.get('plant', 'unknown') for _, summary in results}))
        figure.suptitle(f'B601 Poisson-CBF preliminary reproduction — {plant}\n'
                       'Conservative geometry; environment collision checks exclude self-collision', fontsize=13)
        save_figure(figure, destination)
        plt.close(figure)
    _plot_joint_torques(results, destination, duplicates)


def _plot_joint_torques(results, destination, duplicates):
    with scientific_style():
        figure, axes = plt.subplots(3, 2, figsize=(12.6, 9.2), sharex=True,
                                    layout='constrained')
        for joint, axis in enumerate(axes.flat):
            caps = sorted({float(summary.get('torque_limits_nm', [27, 27, 27, 7, 7, 7])[joint])
                           for _, summary in results})
            annotation = []
            for rows, summary in results:
                axis.plot(_array(rows, 't'), _array(rows, f'tau{joint+1}'),
                          label=_label(summary, duplicates), **_style(summary))
                if 'peak_torque_nm' in summary and 'rms_torque_nm' in summary:
                    annotation.append((_label(summary, duplicates),
                                       summary['peak_torque_nm'][joint],
                                       summary['rms_torque_nm'][joint],
                                       _style(summary)['color']))
            for cap in caps:
                axis.axhline(cap, color='#333333', ls=':', lw=1.0)
                axis.axhline(-cap, color='#333333', ls=':', lw=1.0)
            axis.axhline(0, color='#666666', lw=.6, alpha=.6)
            axis.set_title(f'Joint {joint+1}: model cap $\\pm${caps[0]:g} N m')
            axis.set_ylabel(r'$\tau$ (N m)')
            if joint >= 4:
                axis.set_xlabel('Simulation time (s)')
            _grid(axis)
            axis.set_ylim(-max(caps)*1.22, max(caps)*1.22)
            axis.set_xlim(left=0)
            for line, (label, peak, rms, color) in enumerate(annotation):
                axis.text(.015, .965-line*.085,
                          f'{label}: peak {peak:.2f}, RMS {rms:.2f} N m',
                          transform=axis.transAxes, va='top', fontsize=7.2,
                          color=color, bbox=dict(facecolor='white', edgecolor='none', alpha=.75, pad=.3))
        handles = [Line2D([], [], label=_label(summary, duplicates), **_style(summary))
                   for _, summary in results]
        handles.append(Line2D([], [], color='#333333', ls=':', label='URDF model effort cap'))
        figure.legend(handles=handles, loc='lower center', bbox_to_anchor=(.5, -.045), ncol=min(4, len(handles)))
        rates = sorted({summary.get('torque_statistics_hz') for _, summary in results
                        if summary.get('torque_statistics_hz') is not None})
        rate_label = '/'.join(f'{rate:g}' for rate in rates)
        dynamics_only = all(summary.get('plant') == 'dynamics' for _, summary in results)
        sample_type = 'physics' if dynamics_only else 'reported model'
        statistics = f'Peak/RMS annotations use {rate_label} Hz {sample_type} samples. ' if rate_label else ''
        title = ('Applied actuator torques: six independent arm joints' if dynamics_only
                 else 'Model joint efforts: kinematic bias estimates included')
        figure.suptitle(title+'\n'
                       'Curves show control ticks; caps are model limits, not certified motor ratings', fontsize=13)
        footer = ('Gripper force and contact-force regulation are outside this experiment.' if dynamics_only
                  else 'Kinematic effort estimates do not include actuator tracking dynamics.')
        figure.supxlabel(statistics + footer,
                           fontsize=8, y=-.075)
        basename = destination.with_suffix('') if destination.suffix.lower() in {'.png', '.pdf'} else destination
        save_figure(figure, basename.parent/(basename.name+'_joint_torques'))
        plt.close(figure)


def plot_preparation(field, reach, target, destination):
    """Export sampled position reachability and an annotated Poisson slice."""
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    target = np.asarray(target, dtype=float)
    with scientific_style():
        figure = plt.figure(figsize=(9.0, 7.6), layout='constrained')
        axis = figure.add_subplot(111, projection='3d')
        if len(reach.xyz):
            points = axis.scatter(*reach.xyz.T, c=reach.xyz[:, 2], s=18,
                                  cmap='viridis', alpha=.8, edgecolors='none', rasterized=True)
            figure.colorbar(points, ax=axis, label='Sampled tool height (m)',
                            shrink=.65, pad=.08, aspect=25)
        axis.scatter(*target, color='#EE6677', s=140, marker='*',
                     edgecolor='#333333', linewidth=.55, label='Known goal seed', depthshade=False)
        _set_equal_3d(axis, np.vstack([reach.xyz, target]), pad=.045)
        axis.set(xlabel='$x$ (m)', ylabel='$y$ (m)', zlabel='$z$ (m)')
        axis.legend(loc='upper left')
        figure.suptitle('B601 sampled position reachability\n'
                       f'{reach.sample_count} accepted / {reach.attempted_samples} attempted joint seeds', fontsize=13)
        figure.supxlabel('Known task seeds included. This FK sample cache does not certify orientation, payload or a connecting path.', fontsize=8)
        save_figure(figure, destination/'reachability')
        plt.close(figure)
    _plot_field_slice(field, destination/'field_slice')


def _plot_field_slice(field, destination, z=.48):
    index = int(np.argmin(np.abs(field.axes[2]-z)))
    x, y = field.axes[:2]
    physical = np.asarray(field.occupancy[:, :, index]).T
    buffered = np.asarray(field.buffered_occupancy[:, :, index]).T
    values = np.asarray(field.h[:, :, index]).T
    with scientific_style():
        figure, axis = plt.subplots(figsize=(8.4, 7.0), layout='constrained')
        image = axis.pcolormesh(x, y, np.ma.masked_where(buffered, values*1000),
                               shading='nearest', cmap='viridis', rasterized=True)
        figure.colorbar(image, ax=axis, label=r'Poisson $h$ ($10^{-3}$ m$^2$)',
                        shrink=.88, pad=.035)
        if np.any(buffered):
            axis.contourf(x, y, buffered.astype(float), levels=[.5, 1.5],
                          colors=['#F0E4BB'], alpha=.95)
            if not np.all(buffered):
                axis.contour(x, y, buffered.astype(float), levels=[.5],
                             colors=[BUFFER_COLOR], linestyles='--', linewidths=1.0)
        if np.any(physical):
            axis.contourf(x, y, physical.astype(float), levels=[.5, 1.5],
                          colors=[OBSTACLE_COLOR])
            if not np.all(physical):
                axis.contour(x, y, physical.astype(float), levels=[.5],
                             colors=['#333333'], linewidths=.8)
        stride = max(2, int(round(.12/field.spacing)))
        gx, gy = np.meshgrid(x[1:-1:stride], y[1:-1:stride], indexing='xy')
        queries = np.column_stack([gx.ravel(), gy.ravel(), np.full(gx.size, field.axes[2][index])])
        _, gradients = field.sample(queries)
        valid = field.is_valid(queries)
        norm = np.linalg.norm(gradients[:, :2], axis=1)
        valid &= norm > 1e-10
        if np.any(valid):
            direction = gradients[valid, :2]/norm[valid, None]
            axis.quiver(queries[valid, 0], queries[valid, 1], direction[:, 0], direction[:, 1],
                        color='white', alpha=.75, angles='xy', scale_units='xy', scale=24,
                        width=.0035, headwidth=3.5)
        axis.set(xlabel='$x$ (m)', ylabel='$y$ (m)', aspect='equal',
                 xlim=(x[0], x[-1]), ylim=(y[0], y[-1]),
                 title=f'Poisson safety-field slice at $z={field.axes[2][index]:.3f}$ m')
        axis.legend(handles=[Patch(facecolor=OBSTACLE_COLOR, label='Physical occupancy'),
                             Patch(facecolor='#F0E4BB', edgecolor=BUFFER_COLOR,
                                   linestyle='--', label='Conservative obstacle buffer'),
                             Line2D([], [], color='#555555', marker=r'$\rightarrow$',
                                    ls='none', label=r'Normalized $\nabla h$ direction')],
                    loc='upper center',
                    bbox_to_anchor=(.5, -.12), ncol=1, fontsize=8.5)
        figure.supxlabel('$h$ is a Poisson safety value, not distance or force. '
                         'The 2D slice omits obstacles at other heights.', fontsize=8, y=-.08)
        save_figure(figure, destination)
        plt.close(figure)
