"""Matched, frozen-state projections; these are not new closed-loop rollouts."""
from pathlib import Path
import json
import gzip
import base64
import html

import numpy as np

from .controller import PoissonCBFFilter
from .load_model import _preserve_integration_state
from .scenes import GOAL


def build_command_comparison(robot, field, frames, *, margins=(.025, .05, .10)):
    """Solve both velocity QPs at exactly the same q and nominal commands.

    Margin applies only to the pair-distance filter. Poisson numerical
    inflation/collar is unchanged, so this is not a matched conservatism trial.
    Each command is an independent local query, never integrated or sent to ROS.
    """
    from .distance_controller import RobotDistanceCBFFilter
    poisson = PoissonCBFFilter(field, robot.lower, robot.upper)
    distance = [RobotDistanceCBFFilter(robot, margin=m) for m in margins]
    names = ['Goal seeking'] + [f'J{j+1} {sign}' for j in range(6) for sign in ('+', '−')]
    cases = []
    with _preserve_integration_state(robot.model, robot.data):
        for time_s, q in frames:
            robot.set_q(q)
            points, jac = robot.points_and_jacobians()
            tcp, tcp_jac = robot.tcp().copy(), robot.tcp_jacobian().copy()
            nominal = [np.clip(1.2*(GOAL-np.asarray(q)), -.45, .45)]
            nominal += [sign*.45*np.eye(6)[j] for j in range(6) for sign in (1, -1)]
            commands = []
            for v in nominal:
                p = poisson.filter(q, v, points, jac)
                ps = _result(p, tcp_jac)
                ds = []
                for filt in distance:
                    d = filt.filter(q, v)
                    item = _result(d, tcp_jac)
                    item['initial_margin_feasible'] = bool(d.minimum_h >= -1e-9)
                    ds.append(item)
                commands.append(dict(nominal_rad_s=v.tolist(),
                                     nominal_tcp_m_s=(tcp_jac@v).tolist(),
                                     poisson=ps, distance=ds))
            cases.append(dict(time_s=float(time_s), q_rad=np.asarray(q).tolist(),
                              tcp_m=tcp.tolist(), clearance_m=float(robot.clearance()),
                              commands=commands))
    return dict(schema='rebot-frozen-velocity-projections-v1',
                margins_m=list(margins), command_names=names, frames=cases,
                alpha_per_s=2., joint_speed_rad_s=.45,
                poisson_barrier_units='m^2', distance_barrier_units='m',
                simulation_advanced=False, physical_commands_sent=False,
                scope='independent one-step projections at recorded hierarchy states; no closed-loop comparison',
                matched_effective_clearance=False)


def _result(result, tcp_jac):
    return dict(success=bool(result.success), status=result.status,
                velocity_rad_s=result.velocity.tolist(),
                tcp_velocity_m_s=(tcp_jac@result.velocity).tolist(),
                minimum_barrier=float(result.minimum_h) if np.isfinite(result.minimum_h) else None,
                normalized_residual=float(result.minimum_residual) if np.isfinite(result.minimum_residual) else None,
                correction_norm_rad_s=float(result.correction_norm) if result.success else None,
                solve_ms=float(result.solve_ms), all_rows=int(result.total_constraints),
                qp_rows=int(result.qp_constraints), fallback_used=bool(result.fallback_used))


def command_comparison_fragment():
    return '''<section id="command-comparison" class="note">
<h2>What does each filter do to a joint command?</h2>
<p>Select a nominal command at the displayed pose. Both QPs use the same joint state,
velocity bounds (±0.45 rad/s), gain (2 s⁻¹), and objective. The distance filter uses
the selected arm-envelope margin. The Poisson buffer and numerical collar retain
their original values, so equal conservatism is not assumed.</p>
<label for="comparison-command">Nominal motion </label><select id="comparison-command"></select>
<button id="comparison-example" type="button" style="margin-left:12px">Show an example correction</button>
<p id="command-comparison-summary" role="status">Loading the precomputed command projections…</p>
<div style="overflow-x:auto"><table id="comparison-command-table" style="border-collapse:collapse;width:100%;max-width:950px">
<thead><tr><th>Joint velocity (rad/s)</th><th>Nominal</th><th>Environment Poisson CBF</th><th>Arm distance CBF</th></tr></thead><tbody></tbody></table></div>
<p id="comparison-command-diagnostics"></p>
<p><strong>Local queries, not motion playback:</strong> these commands were solved independently
at frozen recorded states; selecting one does not simulate its subsequent trajectory.
The pose replay above remains the original planner + Poisson CBF experiment.
The distance filter checks all 10 arm-box × 5 obstacle-box pairs, using positive-separation witnesses.
It does not check self-collision, grasped-object geometry, moving obstacles or braking distance.</p>
</section>'''


def command_comparison_script(data):
    payload = base64.b64encode(gzip.compress(json.dumps(data, separators=(',', ':'), allow_nan=False).encode(), mtime=0)).decode()
    source = (Path(__file__).resolve().parents[1]/'assets'/'command_comparison.js').read_text()
    return '<script>\n'+source.replace('__COMMAND_COMPARISON_GZIP__', payload)+'\n</script>'


def plot_command_comparison(data, destination):
    """SciencePlots export with physically comparable command/clearance units."""
    import matplotlib.pyplot as plt
    from .plot_style import scientific_style, save_figure
    frames = data['frames']
    t = [f['time_s'] for f in frames]
    m = int(np.argmin(np.abs(np.asarray(data['margins_m'])-.05)))
    margin = data['margins_m'][m]
    with scientific_style():
        fig, axes = plt.subplots(2, 1, figsize=(8, 6), sharex=True, constrained_layout=True)
        for name, color, key in [('Environment Poisson', '#4477AA', 'poisson'),
                                 (f'Arm distance (margin {margin:g} m)', '#EE6677', 'distance')]:
            results = [(f['commands'][0][key][m] if key == 'distance' else f['commands'][0][key]) for f in frames]
            axes[0].plot(t, [r['correction_norm_rad_s'] for r in results], label=name, color=color)
        axes[0].set_ylabel(r'$\|v^*-v_{\rm nom}\|_2$ (rad/s)')
        axes[0].legend()
        axes[0].set_title(f'Goal-seeking command projected at the same {len(frames)} frozen arm poses')
        axes[1].plot(t, [f['clearance_m'] for f in frames], color='#228833', label='Actual proxy-to-clutter distance')
        axes[1].axhline(margin, color='#EE6677', ls='--', label='Distance-CBF margin')
        axes[1].set_ylabel('Clearance (m)')
        axes[1].set_xlabel('Source replay time (s); initial pose at t = 0')
        axes[1].legend()
        for ax in axes:
            ax.grid(True)
        save_figure(fig, destination)
        plt.close(fig)


def comparison_math_fragment():
    equations = [
        (r'd_R(x,q)=\min_b\left(\|\max(|R_b^T(x-c_b)|-a_b,0)\|_2+\min\{\max_i(|R_b^T(x-c_b)|-a_b)_i,0\}\right)',
         'Robot field: exterior Euclidean distance to the articulated box union.'),
        (r'\mathcal E_\delta(q)=\{x:d_R(x,q)\leq\delta\},\qquad b_{bo}(q)=d(B_b(q),O_o)-\delta',
         'The safety envelope surrounds every body; all body–obstacle pair barriers must remain nonnegative.'),
        (r'n_{bo}=\frac{p_R-p_O}{d_{bo}},\qquad \dot b_{bo}=n_{bo}^{T}J_b(p_R,q)v\geq-\alpha b_{bo}',
         'At positive separation and a differentiable closest pair, the normal points from obstacle to robot.'),
        (r'v^*=\mathop{\rm argmin}_v\tfrac12\|v-v_{\rm nom}\|_2^2\quad\text{subject to all 50 pair inequalities and joint bounds}',
         'The filter projects the nominal joint velocity; a drawn envelope alone does not change a command.'),
    ]
    output = '<details class="note"><summary><strong>Mathematics: the field attached to the arm</strong></summary>'
    for tex, caption in equations:
        output += '<p>'+html.escape(caption)+'</p><div class="math-equation" data-tex="'+html.escape(tex, quote=True)+'">'+html.escape(tex)+'</div>'
    return output+'''<p>For disjoint sets, the minimum robot-to-environment Euclidean distance is symmetric.
The Poisson field solves a different boundary-value problem and is not a distance transform;
its gradient directions and filtered commands can differ. Inside overlapping arm boxes,
the minimum signed-box function is only an inside indicator, not the exact signed distance
to the exterior boundary of their union. Positive offset shells remain exterior envelopes.</p>
<p>At box edges or changes in the closest pair, derivatives can be non-unique. Display grids
and meshes are approximations; the distance QP queries the analytic MuJoCo box pairs.
Continuous-time invariance also requires a feasible initial state, valid dynamics, sufficiently
regular constraints, feasible QPs and exact tracking. These offline demonstrations do not establish those conditions.</p></details>'''
