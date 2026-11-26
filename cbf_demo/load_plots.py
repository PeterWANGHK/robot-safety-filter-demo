"""SciencePlots exports for independent, frozen-posture payload estimates.

No loaded dynamics or controller is executed.  Ordered poses can come from
recorded simulation, but these plots recompute zero-velocity inverse statics at
each pose.  Affine torque bounds are model-feasibility estimates, not payload
ratings or evidence that a trajectory/grasp is safe with the illustrated mass.
"""
from pathlib import Path
import json

import numpy as np
import matplotlib

matplotlib.use('Agg')
from matplotlib.lines import Line2D

from .load_model import analyze_static_load
from .plot_style import plt, save_figure, scientific_style


_POSE_COLORS = ('#4477AA', '#AA3377', '#228833')
_POSE_STYLES = ('-', '--', ':')


def _normalize_frames(frame_qs):
    frames = list(frame_qs)
    if not frames:
        raise ValueError('At least one recorded pose is required')
    qs, times = [], []
    for item in frames:
        if (isinstance(item, (tuple, list)) and len(item) == 2
                and np.isscalar(item[0]) and np.asarray(item[1]).shape == (6,)):
            timestamp, q = float(item[0]), np.asarray(item[1], dtype=float)
            if not np.isfinite(timestamp):
                raise ValueError('Pose timestamps must be finite')
        else:
            timestamp, q = None, np.asarray(item, dtype=float)
        if q.shape != (6,) or not np.all(np.isfinite(q)):
            raise ValueError('Every pose must have six finite joint angles')
        qs.append(q.copy()); times.append(timestamp)
    return qs, times


def plot_load_assessment(robot, frame_qs, destination, *, payload_kg=.5,
                         headroom_fraction=.7):
    """Write two PNG/PDF figures and JSON; return the complete load metrics.

    ``frame_qs`` accepts either six-angle q vectors or ordered ``(time, q)``
    pairs. The horizontal axis remains recorded-pose index, because every
    estimate is a frozen static calculation, not a loaded dynamic rollout.
    The local COM is the TCP origin; a different COM needs a separate explicit
    assessment through ``load_model.analyze_static_load``.
    """
    qs, times = _normalize_frames(frame_qs)
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    reports = [analyze_static_load(robot, q, payload_kg=payload_kg,
                                   headroom_fraction=headroom_fraction) for q in qs]
    representative = sorted(set((0, len(qs)//2, len(qs)-1)))
    names = {0: 'Initial snapshot', len(qs)//2: 'Middle snapshot', len(qs)-1: 'Final snapshot'}
    if len(qs) == 1:
        names = {0: 'Only snapshot'}
    mass = np.linspace(0., 4., 161)
    _plot_demand(reports, representative, names, mass, destination/'load_demand')
    _plot_capacity(reports, representative, names, payload_kg, destination/'load_capacity')
    finite = [(index, report['payload_interval']['maximum_kg'])
              for index, report in enumerate(reports)
              if report['payload_interval']['feasible']
              and report['payload_interval']['maximum_kg'] is not None]
    bottleneck = min(finite, key=lambda pair: pair[1]) if finite else None
    zero_feasible = all(report['payload_interval']['zero_payload_feasible'] for report in reports)
    metrics = {
        'analysis_kind': 'independent_zero_velocity_static_snapshots',
        'pose_count': len(qs),
        'pose_times_seconds_if_supplied': times,
        'representative_pose_indices': representative,
        'mass_sweep_kg': [float(mass[0]), float(mass[-1])],
        'illustrated_payload_kg': float(payload_kg),
        'com_offset_tcp_m': [0., 0., 0.],
        'headroom_fraction': float(headroom_fraction),
        'all_zero_payload_snapshots_feasible': zero_feasible,
        'all_illustrated_payload_snapshots_feasible': all(report['static_load_feasible'] for report in reports),
        'minimum_finite_interval_upper_kg': float(bottleneck[1]) if bottleneck else None,
        'bottleneck_snapshot_index': bottleneck[0] if bottleneck else None,
        'bottleneck_upper_joint': reports[bottleneck[0]]['payload_interval']['limiting_upper_joint'] if bottleneck else None,
        'counterweight_only_snapshot_indices': [index for index, report in enumerate(reports)
                                                if report['payload_interval']['feasible']
                                                and not report['payload_interval']['zero_payload_feasible']],
        'infeasible_snapshot_indices': [index for index, report in enumerate(reports)
                                        if not report['payload_interval']['feasible']],
        'unbounded_model_snapshot_indices': [index for index, report in enumerate(reports)
                                             if report['payload_interval']['maximum_unbounded']],
        'physics_or_controller_advanced': False,
        'moving_payload_simulated': False,
        'hardware_payload_rating': False,
        'interpretation': ('Every sampled posture supports m=0 under the chosen static model caps; '
                           'the minimum upper bound is only a sampled-posture model bottleneck.'
                           if zero_feasible else
                           'Some m=0 postures violate model caps. Positive counterweight intervals '
                           'must not be read as a zero-to-maximum supported payload.'),
        'limitations': reports[0]['limitations'] + [
            'Only supplied snapshots are checked; intermediate postures are not certified.',
            'The mass sweep uses a point mass centered at the TCP, not a modeled grasped object.',
            'Capacity upper bounds are mathematical torque intersections, not manufacturer payload ratings.',
        ],
        'reports': reports,
        'figures': ['load_demand.png', 'load_demand.pdf', 'load_capacity.png', 'load_capacity.pdf'],
    }
    (destination/'load_assessment.json').write_text(json.dumps(metrics, indent=2, allow_nan=False)+'\n')
    return metrics


def _plot_demand(reports, representative, names, mass, destination):
    fraction = reports[0]['headroom_fraction']
    caps = np.asarray(reports[0]['caps_nm'])
    with scientific_style():
        figure, axes = plt.subplots(3, 2, figsize=(12.2, 9.0), sharex=True, layout='constrained')
        for joint, axis in enumerate(axes.flat):
            for choice, index in enumerate(representative):
                report = reports[index]
                demand = report['tau_robot_nm'][joint] + mass*report['tau_per_kg_nm'][joint]
                axis.plot(mass, demand, color=_POSE_COLORS[choice], ls=_POSE_STYLES[choice],
                          label=f'{names[index]} (index {index})')
            axis.axhline(caps[joint], color='#333333', ls='--', lw=.9)
            axis.axhline(-caps[joint], color='#333333', ls='--', lw=.9)
            axis.axhline(0, color='#777777', lw=.55)
            axis.set_title(f'Joint {joint+1}: adjusted model cap $\\pm${caps[joint]:.1f} N m')
            axis.set_ylabel(r'Static holding $\tau$ (N m)')
            axis.set_xlim(0, 4)
            axis.grid(alpha=.25)
            if joint >= 4:
                axis.set_xlabel('Point payload mass at TCP (kg)')
        handles = [Line2D([], [], color=_POSE_COLORS[choice], ls=_POSE_STYLES[choice],
                          label=f'{names[index]} (index {index})')
                   for choice, index in enumerate(representative)]
        handles.append(Line2D([], [], color='#333333', ls='--',
                              label=f'{100*fraction:g}% of URDF model caps'))
        figure.legend(handles=handles, loc='lower center', bbox_to_anchor=(.5, -.045),
                      ncol=2, fontsize=8.5)
        figure.suptitle('Gravity-only joint demand versus point payload\n'
                       'Initial, middle and final frozen postures; no loaded motion simulation', fontsize=13)
        figure.supxlabel('Adjusted effort caps are illustrative model limits, not certified continuous motor or payload ratings.',
                           fontsize=8, y=-.075)
        save_figure(figure, destination)
        plt.close(figure)


def _plot_capacity(reports, representative, names, payload_kg, destination):
    upper = np.array([report['payload_interval']['maximum_kg']
                      if report['payload_interval']['feasible']
                      and report['payload_interval']['maximum_kg'] is not None else np.nan
                      for report in reports], dtype=float)
    lower = np.array([report['payload_interval']['minimum_kg']
                      if report['payload_interval']['feasible'] else np.nan
                      for report in reports], dtype=float)
    indices = np.arange(len(reports))
    middle = representative[len(representative)//2]
    report = reports[middle]
    with scientific_style():
        figure, axes = plt.subplots(2, 1, figsize=(11.7, 8.4), layout='constrained',
                                    gridspec_kw={'height_ratios': [1, 1.1]})
        axis, bars = axes
        axis.plot(indices, upper, color='#4477AA', marker='o', markersize=3,
                  label='Feasible interval upper bound (finite only)')
        axis.plot(indices, lower, color='#AA3377', ls=':', lw=1.2,
                  label='Feasible interval lower bound')
        axis.axhline(payload_kg, color='#228833', ls='--', lw=1.1,
                     label=f'Illustrated point payload: {payload_kg:g} kg')
        counterweight = [i for i, r in enumerate(reports)
                         if r['payload_interval']['feasible'] and not r['payload_interval']['zero_payload_feasible']]
        failed = [i for i, r in enumerate(reports) if not r['payload_interval']['feasible']]
        if counterweight:
            axis.scatter(counterweight, upper[counterweight], color='#CC9900', marker='s',
                         zorder=4, label='Counterweight-only: m=0 infeasible')
        if failed:
            axis.scatter(failed, np.zeros(len(failed)), color='#EE6677', marker='x',
                         zorder=4, label='No feasible mass (marker, not 0 kg capacity)')
        unbounded = [i for i, r in enumerate(reports) if r['payload_interval']['maximum_unbounded']]
        if unbounded:
            axis.text(.01, .97, 'Unbounded in static model at indices: '+', '.join(map(str, unbounded)),
                      transform=axis.transAxes, va='top', fontsize=8)
        axis.set(xlabel='Recorded pose index (independent frozen-posture calculations)',
                 ylabel='Static feasible mass interval (kg)',
                 title='(a) Sampled posture bottlenecks; no interpolation or trajectory certificate')
        axis.set_xlim(-.5, max(.5, len(reports)-.5))
        axis.set_ylim(bottom=0)
        axis.grid(alpha=.25)
        axis.legend(loc='best', fontsize=8)

        joint = np.arange(6)
        width = .24
        bars.bar(joint-width, report['tau_robot_nm'], width=width, color='#4477AA', label='Robot self-weight')
        bars.bar(joint, report['tau_payload_nm'], width=width, color='#CCBB44', label=f'{payload_kg:g} kg payload increment')
        bars.bar(joint+width, report['tau_total_nm'], width=width, color='#228833', label='Total static holding demand')
        cap = np.asarray(report['caps_nm'])
        bars.hlines(cap, joint-.43, joint+.43, colors='#333333', linestyles='--', linewidths=.9)
        bars.hlines(-cap, joint-.43, joint+.43, colors='#333333', linestyles='--', linewidths=.9)
        bars.axhline(0, color='#777777', lw=.6)
        bars.set_xticks(joint, [f'J{i+1}' for i in range(6)])
        bars.set(xlabel='Arm joint', ylabel='Signed holding torque (N m)',
                 title=f'(b) Static load decomposition: {names[middle].lower()} (index {middle})')
        bars.grid(axis='y', alpha=.25)
        handles, labels = bars.get_legend_handles_labels()
        handles.append(Line2D([], [], color='#333333', ls='--', label='Adjusted model effort caps'))
        figure.legend(handles=handles, loc='lower center', bbox_to_anchor=(.5, -.045), ncol=2, fontsize=8.5)
        figure.suptitle(f'B601 static load assessment — {100*report["headroom_fraction"]:g}% model-cap allowance\n'
                       'TCP-centered point payload; acceleration, grasp and mobile-base stability omitted', fontsize=13)
        figure.supxlabel('Static feasibility is not a hardware payload rating. A positive lower mass bound can represent a counterweight requirement.',
                           fontsize=8, y=-.075)
        save_figure(figure, destination)
        plt.close(figure)
