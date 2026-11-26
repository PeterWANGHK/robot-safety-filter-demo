"""Offline parallel 3-D views: environment Poisson versus robot offset field.

The same recorded robot configurations and physical obstacle boxes appear in
both scenes. The left field is the unchanged environment Poisson h (m^2); the
right surfaces are positive exterior-distance offsets of the articulated union
of robot proxy boxes (m). They illustrate different representations, not
equivalent barrier functions or physical force fields. No physics is stepped.

Public integration API:
``data = build_comparison_data(robot, field, frames)``
``comparison_fragment(data)`` supplies HTML/controls and scoped CSS.
``comparison_script(data)`` mounts using the page's existing Plotly bundle.
``build_comparison_view(...)`` writes an optional standalone review artifact.
"""

from __future__ import annotations

import base64
import gzip
import html
import json
from pathlib import Path

import mujoco
import numpy as np

from .field_visualization import build_field_meshes, DEFAULT_DISPLAY_BOUNDS
from .load_model import _preserve_integration_state


_ASSETS = Path(__file__).resolve().parent.parent / 'assets'
_SIGNS = np.array([[x, y, z] for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)])
_QUADS = np.array([[0, 1, 3, 2], [4, 6, 7, 5], [0, 4, 5, 1],
                   [2, 3, 7, 6], [0, 2, 6, 4], [1, 5, 7, 3]])
_FACES = np.concatenate([_QUADS[:, [0, 1, 2]], _QUADS[:, [0, 2, 3]]])
_EDGES = np.array([[i, j] for i in range(8) for j in range(i+1, 8)
                   if np.count_nonzero(_SIGNS[i] != _SIGNS[j]) == 1])
BODY_GROUPS = {
    'shoulder': dict(label='Base / shoulder', bodies=('base_link', 'link1', 'link2'), color='#4477AA'),
    'elbow': dict(label='Elbow / forearm', bodies=('link3',), color='#EE7733'),
    'wrist': dict(label='Wrist', bodies=('link4', 'link5', 'link6'), color='#228833'),
    'gripper': dict(label='Gripper', bodies=('end_link', 'finger_left_link', 'finger_right_link'), color='#AA3377'),
}


def _mesh(vertices, faces, **attributes):
    result = dict(vertices=np.round(np.asarray(vertices, dtype=float), 5).tolist(),
                  faces=np.asarray(faces, dtype=int).tolist())
    result.update(attributes)
    return result


def _box(center, half_size, rotation=None):
    vertices = _SIGNS*np.asarray(half_size)
    if rotation is not None:
        vertices = vertices@np.asarray(rotation).T
    return vertices+np.asarray(center), _FACES.copy()


def _body_group(name):
    return next((key for key, group in BODY_GROUPS.items() if name in group['bodies']), 'shoulder')


def _robot_vertices(robot):
    arrays = []
    for box in robot.proxy_boxes:
        body = mujoco.mj_name2id(robot.model, mujoco.mjtObj.mjOBJ_BODY, box.body_name)
        rotation = robot.data.xmat[body].reshape(3, 3)
        center = robot.data.xpos[body]+rotation@box.center
        vertices, _ = _box(center, box.half_size, rotation)
        arrays.append(vertices)
    return np.concatenate(arrays)


def _closest_pair(robot):
    distance, endpoints, names = float('inf'), None, None
    for proxy_index, robot_geom in enumerate(robot.proxy_geom_ids):
        for obstacle_index, obstacle_geom in enumerate(robot.obstacle_geom_ids):
            points = np.zeros(6)
            value = mujoco.mj_geomDistance(robot.model, robot.data, int(robot_geom),
                                           int(obstacle_geom), 100.0, points)
            if value < distance:
                distance = float(value)
                endpoints = points.reshape(2, 3).copy()
                names = (robot.proxy_boxes[proxy_index].body_name,
                         robot.obstacles[obstacle_index]['name'])
    return dict(clearance_m=distance if np.isfinite(distance) else None,
                points_m=np.round(endpoints, 6).tolist() if endpoints is not None else [],
                robot_body=names[0] if names else None, obstacle=names[1] if names else None)


def build_comparison_data(robot, field, frames, *, margins=(0.025, 0.05, 0.10),
                          spacing=0.025, step_size=2):
    """Cache both representations at recorded ``[(time_s,q6), ...]`` poses.

    Every pose is used; callers can subsample a long log before calling. Shell
    geometry is display-only and coarse by default. The returned plain JSON
    dictionary contains no paths, ROS messages, or executable data. Simulator
    integration inputs are preserved, and no time step or controller runs.
    """
    from .robot_field import robot_distance_meshes

    frames = list(frames)
    if not frames:
        raise ValueError('At least one recorded frame is required')
    margins = tuple(sorted(set(float(value) for value in margins)))
    if not margins or not np.all(np.isfinite(margins)) or min(margins) <= 0:
        raise ValueError('margins must be finite positive exterior distances')
    if not np.isfinite(spacing) or spacing <= 0:
        raise ValueError('spacing must be finite and positive')
    if not isinstance(step_size, (int, np.integer)) or isinstance(step_size, bool) or step_size < 1:
        raise ValueError('step_size must be a positive integer')
    environment = [_mesh(mesh['vertices'], mesh['faces'], level_m2=mesh['level'], color=mesh['color'])
                   for mesh in build_field_meshes(field, bounds=DEFAULT_DISPLAY_BOUNDS, step_size=2)]
    obstacles = []
    for obstacle in robot.obstacles:
        vertices, faces = _box(obstacle['center'], obstacle['half_size'])
        obstacles.append(_mesh(vertices, faces, name=str(obstacle['name'])))
    body_faces = np.concatenate([_FACES+8*index for index in range(len(robot.proxy_boxes))])
    body_edges = np.concatenate([_EDGES+8*index for index in range(len(robot.proxy_boxes))])
    groups = [_body_group(box.body_name) for box in robot.proxy_boxes]
    face_colors = [BODY_GROUPS[group]['color'] for group in groups for _ in _FACES]
    records = []
    with _preserve_integration_state(robot.model, robot.data):
        for time_s, q in frames:
            q = np.asarray(q, dtype=float)
            if q.shape != (6,) or not np.all(np.isfinite(q)) or not np.isfinite(time_s):
                raise ValueError('Frames require finite timestamps and six-element q vectors')
            robot.set_q(q)
            arm_vertices = _robot_vertices(robot)
            points = robot.points()
            values, _ = field.sample(points)
            valid = field.is_valid(points)
            nearest = _closest_pair(robot)
            shells = robot_distance_meshes(robot, q=q, margins=margins,
                                           spacing=float(spacing), step_size=int(step_size))
            by_margin = {round(float(mesh['offset']), 8): mesh for mesh in shells}
            packed_shells = []
            for margin in margins:
                mesh = by_margin.get(round(margin, 8))
                if mesh is None:
                    packed_shells.append(dict(margin_m=margin, vertices=[], faces=[], color='#44AA99'))
                else:
                    packed_shells.append(_mesh(mesh['vertices'], mesh['faces'], margin_m=margin,
                                               color=str(mesh['color'])))
            records.append(dict(time_s=float(time_s), q_rad=np.round(q, 9).tolist(),
                                arm_vertices=np.round(arm_vertices, 5).tolist(),
                                tcp_m=np.round(robot.tcp(), 6).tolist(),
                                shells=packed_shells, nearest=nearest,
                                min_h_m2=float(np.min(values)),
                                all_poisson_samples_valid=bool(np.all(valid))))
    return dict(version=1, frames=records, environment=environment, obstacles=obstacles,
                margins_m=list(margins), default_margin_m=min(margins, key=lambda value: abs(value-.05)),
                arm_faces=body_faces.tolist(), arm_edges=body_edges.tolist(),
                arm_face_colors=face_colors, body_groups=groups,
                body_group_definitions={key: dict(label=value['label'], color=value['color'])
                                        for key, value in BODY_GROUPS.items()},
                display_bounds_m=np.asarray(DEFAULT_DISPLAY_BOUNDS).tolist(),
                distance_grid_spacing_m=float(spacing), marching_cubes_stride=int(step_size),
                scope=dict(physics_advanced=False, robot_geometry='oriented conservative body boxes',
                           left_units='m^2', right_units='m',
                           right_geometry_used_for_control=False,
                           collision_controller='environment Poisson joint-velocity CBF',
                           negative_robot_field_not_exact_union_interior_distance=True))


def comparison_fragment(data):
    """HTML controls/plot container; no Plotly bundle or executable payload."""
    environment_controls = ''.join(
        '<label><input type="checkbox" class="comparison-poisson-toggle" data-index="'
        + str(index) + '" checked> h = ' + f'{mesh["level_m2"]:g}' + ' m²</label>'
        for index, mesh in enumerate(data['environment']))
    margin_controls = ''.join(
        '<label><input type="checkbox" class="comparison-margin-toggle" data-index="'
        + str(index) + '"' + (' checked' if margin == data['default_margin_m'] else '')
        + '> ' + f'{margin*100:g}' + ' cm offset</label>'
        for index, margin in enumerate(data['margins_m']))
    groups = ''.join('<span><i style="background:' + definition['color'] + '"></i>'
                     + html.escape(definition['label']) + '</span>'
                     for definition in data['body_group_definitions'].values())
    style = (_ASSETS/'comparison_view.css').read_text(encoding='utf-8')
    return '<style>'+style+'</style>'+'''<section id="field-comparison" aria-labelledby="field-comparison-heading">
<h2 id="field-comparison-heading">Environment field and robot-centred safety envelope</h2>
<p>The two views share the same arm pose, obstacles, metric axes and camera.
Drag either scene to rotate both. The left Poisson shells stay fixed in the room;
the right distance envelope follows every articulated link, including the elbow and wrist.</p>
<div class="comparison-control-grid"><fieldset><legend>Environment Poisson h (m²)</legend>'''+environment_controls+'''
</fieldset><fieldset><legend>Robot exterior distance d<sub>R</sub> (metres)</legend>'''+margin_controls+'''
</fieldset></div><p class="comparison-active-margin">Active command-comparison margin:
<output id="comparison-active-margin">'''+f'{100*data["default_margin_m"]:g}'+''' cm</output>.
The latest checked shell selects this margin; other checked shells are visual overlays.</p>
<div class="comparison-body-key" aria-label="Matching body groups">'''+groups+'''
</div><div class="comparison-pose-control"><label for="comparison-pose">Shared recorded pose:
<output id="comparison-pose-label">Initial pose</output></label>
<input id="comparison-pose" type="range" min="0" max="'''+str(len(data['frames'])-1)+'''" step="1" value="0"
aria-label="Shared arm pose in both field representations"></div>
<div id="field-comparison-3d" role="img" aria-label="Synchronized environment Poisson and robot distance field scenes"></div>
<p id="comparison-pose-metrics" aria-live="polite"></p>
<p class="comparison-velocity-key">When a command is selected, the blue arrow on the left and teal arrow on the right
show the corresponding filtered <strong>instantaneous TCP velocity × 0.25 s</strong>.
They are local command illustrations, not planned paths or predicted trajectories.</p>
<p class="comparison-scope">Right-hand shells are the positive-distance offset of the union of full oriented robot boxes,
not balls around joint centres. The dashed segment joins the nearest robot/obstacle pair.
An obstacle entering an offset envelope is within that chosen margin. This is a coarse display
of a distance representation, not the same CBF certificate as the left Poisson field.
The current recorded controller remains the environment-Poisson velocity filter; selecting a shell
does not change its constraints, add payload geometry, or apply a force.</p>
<p class="comparison-status" id="comparison-status" role="status">Preparing cached 3D comparison…</p></section>'''


def comparison_script(data, *, compress=True):
    """Mount with existing global Plotly; optionally compress cached geometry.

    Compression uses standard browser DecompressionStream('gzip') without a
    network request. Set compress=False for a browser without that API.
    """
    payload = json.dumps(data, separators=(',', ':'), allow_nan=False)
    if compress:
        encoded = base64.b64encode(gzip.compress(payload.encode(), mtime=0)).decode('ascii')
        initializer = "if(!window.DecompressionStream)throw new Error('This offline view needs a modern Chromium, Firefox or Safari browser with gzip DecompressionStream support.'); const packed='"+encoded+"'; const bytes=Uint8Array.from(atob(packed),c=>c.charCodeAt(0)); const data=JSON.parse(await new Response(new Blob([bytes]).stream().pipeThrough(new DecompressionStream('gzip'))).text());"
    else:
        initializer = 'const data='+payload.replace('<', '\\u003c')+';'
    script = (_ASSETS/'comparison_view.js').read_text(encoding='utf-8')
    return '<script>(async()=>{try{'+initializer+script+'''
}catch(error){const status=document.getElementById('comparison-status');
if(status)status.textContent='Comparison could not initialize: '+error.message;
console.error('reBot comparison',error);}})();</script>'''


def build_comparison_view(robot, field, frames, destination, **kwargs):
    """Write a standalone offline review page and inspectable data JSON.

    Main-report integration should use fragment/script and reuse its Plotly
    bundle instead. This convenience wrapper embeds exactly one Plotly bundle.
    """
    from plotly.offline import get_plotlyjs
    data = build_comparison_data(robot, field, frames, **kwargs)
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    document = '''<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>B601 field comparison</title>
<style>body{margin:20px;color:#273340;background:white;font:17px/1.5 Georgia,serif}main{max-width:1500px;margin:auto}</style>
</head><body><main>'''+comparison_fragment(data)+'</main><script>'+get_plotlyjs()+'</script>'+comparison_script(data)+'</body></html>'
    destination.write_text(document, encoding='utf-8')
    destination.with_suffix('.json').write_text(json.dumps(data, separators=(',', ':'), allow_nan=False), encoding='utf-8')
    return data


def plot_field_comparison(data, destination, pose_index=0):
    """Save matching SciencePlots PNG/PDF parallel 3-D views at one pose.

    This reads cached geometry only. Both panes have identical camera, axis
    ranges, OBB robot geometry and physical obstacles. The default 5 cm
    robot-envelope shell is shown on the right; it is display geometry rather
    than a second closed-loop controller result. Returns (png_path,pdf_path).
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    from .plot_style import scientific_style, save_figure

    if (not isinstance(pose_index, (int, np.integer)) or isinstance(pose_index, bool)
            or not 0 <= pose_index < len(data['frames'])):
        raise ValueError('pose_index must identify one cached recorded pose')
    frame = data['frames'][pose_index]
    lower, upper = np.asarray(data['display_bounds_m'])
    selected = min(frame['shells'], key=lambda mesh: abs(mesh['margin_m']-data['default_margin_m']))

    def add_mesh(axis, mesh, color, alpha, edges=False):
        vertices, faces = np.asarray(mesh['vertices']), np.asarray(mesh['faces'], dtype=int)
        if not len(faces):
            return
        surface = Poly3DCollection(vertices[faces], facecolors=color, alpha=alpha,
                                   edgecolors='#263d50' if edges else 'none',
                                   linewidths=.35 if edges else 0,
                                   rasterized=not edges)
        axis.add_collection3d(surface)

    with scientific_style():
        figure = plt.figure(figsize=(13.5, 7.3))
        figure.subplots_adjust(left=.025, right=.98, top=.79, bottom=.17, wspace=.025)
        axes = [figure.add_subplot(1, 2, index+1, projection='3d') for index in range(2)]
        for index, axis in enumerate(axes):
            if index == 0:
                for mesh in data['environment']:
                    add_mesh(axis, mesh, mesh['color'], .13)
            else:
                add_mesh(axis, selected, selected['color'], .25)
            for obstacle in data['obstacles']:
                add_mesh(axis, obstacle, '#B85042', .45)
            add_mesh(axis, dict(vertices=frame['arm_vertices'], faces=data['arm_faces']),
                     data['arm_face_colors'], .97, edges=True)
            nearest = np.asarray(frame['nearest']['points_m'])
            if nearest.shape == (2, 3):
                axis.plot(*nearest.T, color='#AA3377', ls='--', lw=1.7, marker='o', ms=3)
            axis.set(xlim=(lower[0], upper[0]), ylim=(lower[1], upper[1]),
                     zlim=(lower[2], upper[2]), xlabel='$x$ (m)', ylabel='$y$ (m)', zlabel='$z$ (m)')
            axis.set_box_aspect(upper-lower)
            axis.view_init(elev=25, azim=-57)
            axis.tick_params(labelsize=8)
        axes[0].set_title('(a) Environment Poisson field $h$ (m$^2$)\nFixed in the room', fontsize=11)
        axes[1].set_title('(b) Robot exterior distance $d_R$ (m)\nArticulated envelope, '+f'{100*selected["margin_m"]:g} cm offset', fontsize=11)
        left_legend = [Patch(facecolor=mesh['color'], alpha=.65,
                             label='$h='+f'{mesh["level_m2"]:g}'+'$ m$^2$')
                       for mesh in data['environment']]
        axes[0].legend(handles=left_legend, loc='upper left', fontsize=8.5,
                       frameon=True, framealpha=.9)
        axes[1].legend(handles=[Patch(facecolor=selected['color'], alpha=.5,
                                      label='$d_R='+f'{selected["margin_m"]:g}'+'$ m'),
                               Patch(facecolor='#B85042', alpha=.6, label='Physical clutter'),
                               Line2D([], [], color='#AA3377', ls='--', marker='o', ms=3,
                                      label='Nearest whole-box pair')],
                       loc='upper left', fontsize=8.5, frameon=True, framealpha=.9)
        body_handles = [Patch(facecolor=group['color'], label=group['label'])
                        for group in data['body_group_definitions'].values()]
        figure.legend(handles=body_handles, loc='upper center', ncol=4,
                      bbox_to_anchor=(.5, .925), fontsize=9)
        figure.suptitle('B601: environment and robot-centred field representations', fontsize=15, y=.98)
        nearest = frame['nearest']['clearance_m']
        clearance = 'undefined' if nearest is None else f'{100*nearest:.2f} cm'
        caption = ('Same recorded arm pose, physical boxes, axis ranges and camera in both panes. '
                   'Barrier values are not distance or force.\n'
                   f'Snapshot {pose_index+1}/{len(data["frames"])} at {frame["time_s"]:.2f} s; '
                   f'whole-box clearance {clearance}. Coarse shells are display geometry; '
                   'they do not change the recorded controller.')
        figure.text(.5, .055, caption, ha='center', va='center', fontsize=9, linespacing=1.6)
        paths = save_figure(figure, destination)
        plt.close(figure)
    return paths
