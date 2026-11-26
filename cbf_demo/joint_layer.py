"""Read-only body-barrier and static payload diagnostics for the HTML view.

Display groups select complete link surfaces, never just joint centers. All
collision-filter constraints remain in the controller. No payload is inserted
into the recorded dynamics by this explanatory layer.
"""
from pathlib import Path
import html
import json

import numpy as np

from .load_model import analyze_static_load, _preserve_integration_state

REGIONS = {
    'whole': ('Whole arm — all body surfaces', None),
    'shoulder': ('Base / shoulder / upper arm — base, links 1–2', ('base_link', 'link1', 'link2')),
    'elbow': ('Elbow / forearm — link 3', ('link3',)),
    'wrist': ('Wrist complex — links 4–6', ('link4', 'link5', 'link6')),
    'gripper': ('Gripper and fingers', ('end_link', 'finger_left_link', 'finger_right_link')),
}


def build_joint_layer_data(robot, field, frames):
    """Evaluate every body sample and exact affine holding torque at each pose."""
    records = []
    names = np.asarray(robot.point_body_names)
    with _preserve_integration_state(robot.model, robot.data):
        for time_s, q in frames:
            robot.set_q(q)
            points, jacobians = robot.points_and_jacobians()
            values, gradients = field.sample(points)
            valid = field.is_valid(points)
            regions = {}
            for key, (_, bodies) in REGIONS.items():
                indices = np.flatnonzero(np.ones(len(names), dtype=bool) if bodies is None
                                          else np.isin(names, bodies))
                if not len(indices):
                    raise ValueError(f'No robot geometry in display region {key}')
                worst = int(indices[np.argmin(values[indices])])
                shown = indices[::4]  # Only rendering is decimated; minima use all points.
                regions[key] = dict(sample_count=len(indices), valid_count=int(valid[indices].sum()),
                                    h_min_m2=float(values[worst]), critical_body=str(names[worst]),
                                    critical_sample=worst,
                                    critical_point_m=points[worst].tolist(),
                                    critical_gradient_world_m=gradients[worst].tolist(),
                                    barrier_joint_sensitivity_m2_per_rad=(gradients[worst] @ jacobians[worst]).tolist(),
                                    displayed_points_m=np.round(points[shown], 7).tolist(),
                                    displayed_h_m2=np.round(values[shown], 9).tolist())
            load = analyze_static_load(robot, q, payload_kg=.5)
            slopes = []
            for axis in range(3):
                offset = np.zeros(3); offset[axis] = .1
                report = analyze_static_load(robot, q, com_offset=offset)
                slopes.append(((np.array(report['tau_per_kg_nm'])-load['tau_per_kg_nm'])/.1).tolist())
            records.append(dict(time_s=float(time_s), q_rad=np.asarray(q).tolist(), regions=regions,
                                tau_robot_nm=load['tau_robot_nm'], per_kg_tcp_nm=load['tau_per_kg_nm'],
                                per_kg_per_m_tool_xyz_nm=slopes,
                                tcp_m=load['tcp_world_m'],
                                tool_axes_world=np.array(load['tcp_rotation_world_from_tcp']).T.tolist(),
                                joint_origins_m=load['joint_origins_world_m'],
                                joint_axes_world=load['joint_axes_world'],
                                whole_arm_clearance_m=float(robot.clearance())))
    return dict(frames=records, model_caps_nm=np.asarray(robot.torque_limits).tolist(),
                proxy_bodies=[box.body_name for box in robot.proxy_boxes],
                regions={key: label for key, (label, _) in REGIONS.items()},
                region_bodies={key: None if bodies is None else list(bodies)
                               for key, (_, bodies) in REGIONS.items()},
                joint_labels=['J1 base yaw', 'J2 shoulder pitch', 'J3 elbow pitch',
                              'J4 distal pitch', 'J5 wrist bend', 'J6 tool roll'],
                defaults=dict(payload_kg=.5, available_fraction=.7, tool_z_offset_m=0.),
                assumptions=dict(static_only=True, payload_geometry_added=False,
                                 torque_constraints_added_to_controller=False,
                                 all_cbf_constraints_retained=len(names),
                                 anatomical_labels_are_analogies=True))


def joint_layer_controls(data):
    options = ''.join(f'<option value="{key}">{html.escape(label)}</option>'
                      for key, label in data['regions'].items())
    return '''<section id="joint-load-layer" aria-labelledby="joint-layer-heading">
<h2 id="joint-layer-heading">Body geometry and payload support</h2>
<p class="layer-scope">Collision diagnostics and static holding estimates at the displayed pose.
The payload controls do not alter the recorded motion or add an object to the simulator.</p>
<div class="load-controls">
<label>Focus on body region<select id="body-region">''' + options + '''</select></label>
<label>Payload mass: <output id="payload-value">0.50 kg</output>
<input id="payload-mass" type="range" min="0" max="4" step="0.05" value="0.5"></label>
<label>Usable model torque: <output id="budget-value">70%</output>
<input id="torque-budget" type="range" min="20" max="100" step="5" value="70"></label>
<label>Payload COM offset: <output id="com-value">0 cm</output>
<select id="payload-com-axis" aria-label="Tool-frame COM offset axis"><option value="0">Tool x axis</option>
<option value="1">Tool y axis</option><option value="2" selected>Tool z axis</option></select>
<input id="payload-com" type="range" min="-20" max="20" step="1" value="0"></label>
</div><p id="load-summary" aria-live="polite"></p>
<p id="region-summary"></p></section>'''


def joint_layer_details():
    return '''<section aria-labelledby="load-table-heading"><h2 id="load-table-heading">What each joint must provide</h2>
<p>Signed torques below hold the arm and point-mass payload still. The budget is the selected fraction
of URDF model caps; it is not a certified continuous motor rating. J1 can have zero gravity torque
while still carrying support forces and requiring dynamic torque.</p>
<div class="table-scroll"><table class="joint-table"><thead><tr>
<th>Joint</th><th>Angle<br>deg</th><th>Arm gravity<br>N·m</th><th>Payload<br>N·m</th>
<th>Total hold<br>N·m</th><th>Budget<br>N·m</th><th>Budget used</th>
</tr></thead><tbody id="joint-load-rows"></tbody></table></div>
<h3>How joint motion changes the most exposed body point</h3>
<p id="critical-summary"></p><div class="table-scroll"><table class="joint-table"><thead><tr>
<th>Local sensitivity</th><th>J1</th><th>J2</th><th>J3</th><th>J4</th><th>J5</th><th>J6</th>
</tr></thead><tbody><tr id="sensitivity-row"></tr></tbody></table></div>
<p>These are ∂H/∂q values in m²/rad, not torques. A positive entry means a small positive rotation
in that joint increases this point's barrier value with the other joints fixed. The critical sample
can change between poses. Focusing the display does not remove any of the 798 CBF constraints.</p>
<p>Human arms follow the same leverage and virtual-work principles, but muscle forces require muscle
moment arms, co-contraction, posture and fatigue data. Torque-budget usage here is a deterministic
model ratio; it cannot give a probability of physical failure or a person's lifting capacity.</p>
</section>'''


def joint_layer_style():
    return '''<style>
.load-controls{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:12px 28px;max-width:1000px}
.load-controls label{display:block;font-size:16px}.load-controls input,.load-controls select{display:block;width:100%;margin-top:5px}
.load-controls select{font:inherit;padding:6px;max-width:100%;box-sizing:border-box}
.layer-scope{color:#526170}.table-scroll{overflow-x:auto}.joint-table{border-collapse:collapse;width:100%;max-width:1100px;font-size:15px;font-variant-numeric:tabular-nums}
.joint-table th,.joint-table td{padding:9px 12px;text-align:right;border-bottom:1px solid #dfe3e8}
.joint-table th:first-child,.joint-table td:first-child{text-align:left}.joint-table th{font-weight:normal;background:#f4f6f8}
.over-budget{color:#ad321d;font-weight:bold}.math-notes{max-width:1100px}.math-notes summary{font-size:19px;padding:12px 0;cursor:pointer}
.math-notes details{border-top:1px solid #dfe3e8}.math-equation{overflow-x:auto;padding:8px 0;max-width:100%}
.math-notes table{border-collapse:collapse}.math-notes td,.math-notes th{padding:8px;text-align:left;border-bottom:1px solid #dfe3e8}
.math-notes p,.math-notes li{max-width:1000px}.katex{font-size:1.08em}
@media(max-width:650px){.load-controls{grid-template-columns:1fr}.joint-table th,.joint-table td{padding:7px;font-size:14px}}
</style>'''


def joint_layer_script(data, trace_indices):
    payload = json.dumps(dict(data=data, traces=trace_indices), separators=(',', ':'), allow_nan=False).replace('<', '\\u003c')
    math_js = (Path(__file__).resolve().parent.parent/'assets/load_math.js').read_text()
    ui_js = (Path(__file__).resolve().parent.parent/'assets/joint_layer.js').read_text()
    return ('<script>'+math_js+'</script><script id="joint-layer-data" type="application/json">'+payload+
            '</script><script>window.addEventListener("load",function(){'+ui_js+'});</script>')
