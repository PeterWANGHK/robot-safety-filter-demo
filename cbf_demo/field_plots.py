"""Three-dimensional views of the actual Poisson grid, never a force field.

Scientific PNG/PDF exports use SciencePlots through plot_style. The standalone
Plotly view uses matching fonts/colors; Matplotlib styles do not style WebGL.
Robot poses are measured simulation samples, not an interpolated trajectory.
"""
from __future__ import annotations

import csv
import html
import json
from pathlib import Path

import mujoco
import numpy as np

from .field_visualization import (build_field_meshes, FIELD_COLORS, FIELD_LEVELS,
                                  DEFAULT_DISPLAY_BOUNDS)
from .scenes import START, GOAL, clutter_scene, build_field

_SIGNS = np.array([[x, y, z] for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)])
_QUADS = np.array([[0, 1, 3, 2], [4, 6, 7, 5], [0, 4, 5, 1],
                   [2, 3, 7, 6], [0, 2, 6, 4], [1, 5, 7, 3]])
_FACES = np.concatenate([_QUADS[:, [0, 1, 2]], _QUADS[:, [0, 2, 3]]])


def box_mesh(center, half_size, rotation=None):
    """Return explicit triangles, avoiding an accidental convex-hull remake."""
    vertices = _SIGNS * np.asarray(half_size)
    if rotation is not None:
        vertices = vertices @ np.asarray(rotation).T
    return vertices + np.asarray(center), _FACES.copy()


def robot_mesh(robot, q):
    """World-frame conservative arm boxes, preserving simulator state."""
    previous = robot.get_q()
    try:
        robot.set_q(q)
        vertices, faces = [], []
        for index, proxy in enumerate(robot.proxy_boxes):
            bid = mujoco.mj_name2id(robot.model, mujoco.mjtObj.mjOBJ_BODY, proxy.body_name)
            rotation = robot.data.xmat[bid].reshape(3, 3)
            center = robot.data.xpos[bid] + rotation @ proxy.center
            v, f = box_mesh(center, proxy.half_size, rotation)
            vertices.append(v)
            faces.append(f + index * 8)
        return np.concatenate(vertices), np.concatenate(faces)
    finally:
        robot.set_q(previous)


def _tcp_route(robot, path):
    previous = robot.get_q()
    try:
        return np.array([robot.tcp(q) for q in path])
    finally:
        robot.set_q(previous)


def _colorscale():
    # Fixed anchors match RViz, so values retain their meaning across runs.
    return [[0, '#f5dfae'], [.1, FIELD_COLORS[0]], [.4, FIELD_COLORS[1]],
            [1, FIELD_COLORS[2]]]


def recorded_frames(results):
    """Initial pose plus at most forty actual logged hierarchy configurations."""
    hierarchy = next((rows for rows, summary in results if summary['mode']=='hierarchy'), None)
    frames = [(0., START.copy())]
    if hierarchy:
        indices = np.unique(np.linspace(0, len(hierarchy)-1, min(40, len(hierarchy)), dtype=int))
        frames += [(hierarchy[i]['t'], np.array([hierarchy[i][f'q{j}'] for j in range(1, 7)]))
                   for i in indices]
    return frames


def plot_field_3d(field, robot, destination, *, results=(), path=None, target=None):
    """Save 3D scientific figure, offline interactive HTML, and grid/provenance."""
    from .plot_style import scientific_style, save_figure
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap, Normalize
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    meshes = build_field_meshes(field, bounds=DEFAULT_DISPLAY_BOUNDS)
    q = START
    route = _tcp_route(robot, path) if path is not None else None
    lower, upper = np.asarray(DEFAULT_DISPLAY_BOUNDS)
    cmap = LinearSegmentedColormap.from_list('poisson_barrier', _colorscale())
    norm = Normalize(0, .02, clip=True)
    with scientific_style():
        fig = plt.figure(figsize=(13.0, 7.2), constrained_layout=True)
        grid = fig.add_gridspec(2, 3, width_ratios=(1, 1, 1.2))
        ax = fig.add_subplot(grid[:, :2], projection='3d')
        for mesh in meshes:
            surface = Poly3DCollection(mesh['vertices'][mesh['faces']],
                                       facecolor=mesh['color'], edgecolor='none',
                                       alpha=.12 if mesh['level'] < .005 else .065,
                                       rasterized=True)
            ax.add_collection3d(surface)
        for obstacle in robot.obstacles:
            v, f = box_mesh(obstacle['center'], obstacle['half_size'])
            ax.add_collection3d(Poly3DCollection(v[f], facecolor='#b85042',
                                                edgecolor='#753a32', linewidths=.25,
                                                alpha=.6))
        v, f = robot_mesh(robot, q)
        ax.add_collection3d(Poly3DCollection(v[f], facecolor='#374553',
                                            edgecolor='#192b38', linewidths=.25,
                                            alpha=.95))
        if route is not None:
            ax.plot(*route.T, color='#009E73', lw=1.5, ls='--', label='Planned TCP guide')
        from .plot_style import MODE_COLORS, MODE_LABELS
        for rows, summary in results:
            xyz = np.array([[r[f'tcp_{k}'] for k in 'xyz'] for r in rows])
            name = summary['mode']
            ax.plot(*xyz.T, color=MODE_COLORS[name], lw=2,
                    label=MODE_LABELS[name])
        if target is not None:
            ax.scatter(*target, marker='*', s=100, color='#CC79A7',
                       edgecolor='white', linewidth=.5, label='Task goal')
        ax.set(xlim=(lower[0], upper[0]), ylim=(lower[1], upper[1]),
               zlim=(lower[2], upper[2]), xlabel='$x$ (m)', ylabel='$y$ (m)',
               zlabel='$z$ (m)', title='(a) 3D isosurfaces, arm proxies and tool paths')
        ax.set_box_aspect(upper-lower)
        ax.view_init(elev=25, azim=-57)
        ax.tick_params(labelsize=8)
        legend = [Patch(facecolor=m['color'], alpha=.6,
                        label=f'$h={m["level"]:.3f}$ m$^2$') for m in meshes]
        legend += [Patch(facecolor='#b85042', label='Physical clutter'),
                   Patch(facecolor='#374553', label='Arm collision proxies (start)')]
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles=legend + handles, loc='upper left', fontsize=7.6,
                  frameon=True, framealpha=.88)

        slices = [(.48, '(b) Horizontal slice and increasing-$h$ directions'),
                  (.20, '(c) Lower slice: buffers separate from physical boxes')]
        for panel, (height, title) in enumerate(slices):
            axis = fig.add_subplot(grid[panel, 2])
            iz = int(np.argmin(abs(field.axes[2]-height)))
            xx, yy = np.meshgrid(field.axes[0], field.axes[1], indexing='ij')
            im = axis.pcolormesh(xx, yy, field.h[:, :, iz], cmap=cmap, norm=norm,
                                 shading='auto', rasterized=True)
            for mask, color, ls in [(field.occupancy, '#b85042', '-'),
                                    (field.solver_occupancy, '#555555', '--')]:
                cut = mask[:, :, iz]
                if np.any(cut) and not np.all(cut):
                    axis.contour(xx, yy, cut, levels=[.5], colors=color,
                                 linewidths=.9, linestyles=ls)
            if panel == 0:
                points = np.column_stack([xx[::6, ::6].ravel(), yy[::6, ::6].ravel(),
                                           np.full(xx[::6, ::6].size, field.axes[2][iz])])
                h, gradient = field.sample(points)
                keep = field.is_valid(points) & (h > 1e-5)
                points, gradient = points[keep], gradient[keep, :2]
                length = np.linalg.norm(gradient, axis=1)
                direction = np.divide(gradient, length[:, None],
                                      out=np.zeros_like(gradient), where=length[:, None]>1e-10)
                axis.quiver(points[:, 0], points[:, 1], direction[:, 0], direction[:, 1],
                            color='#293a4a', scale=18, width=.003, alpha=.7)
            axis.set(xlim=(lower[0], upper[0]), ylim=(lower[1], upper[1]),
                     xlabel='$x$ (m)', ylabel='$y$ (m)', aspect='equal',
                     title=title + f'\n$z={field.axes[2][iz]:.3f}$ m')
            axis.tick_params(labelsize=8)
            fig.colorbar(im, ax=axis, shrink=.9, extend='max',
                         label='$h$ (m$^2$); color saturates at 0.02')
            if panel == 1:
                axis.legend(handles=[Line2D([], [], color='#b85042', label='Physical occupancy'),
                                     Line2D([], [], color='#555555', ls='--',
                                            label='Solver collar boundary')], fontsize=7, loc='lower left')
        fig.suptitle('B601: three-dimensional Poisson safety field', fontsize=15)
        fig.supxlabel('Barrier value is not distance or force. Shells are display-cropped; the full PDE grid is unchanged.\n'
                      'Arrows show normalized spatial gradients, not commanded forces. Arm boxes are conservative.', fontsize=9)
        save_figure(fig, destination/'field_3d.png')
        plt.close(fig)
    metadata = write_interactive_field(field, robot, destination/'field_3d.html',
                                       meshes=meshes, results=results, path=path, target=target)
    from .load_plots import plot_load_assessment
    assessment = plot_load_assessment(robot, recorded_frames(results), destination)
    metadata['static_load_assessment'] = {
        key: assessment[key] for key in ('minimum_finite_interval_upper_kg',
                                         'bottleneck_snapshot_index', 'bottleneck_upper_joint')
        if key in assessment}
    np.savez_compressed(destination/'field_grid.npz', h=field.h, origin=field.origin,
                        spacing=field.spacing, occupancy=field.occupancy,
                        buffered_occupancy=field.buffered_occupancy,
                        solver_occupancy=field.solver_occupancy)
    metadata.update(grid_shape=list(field.shape), spacing_m=float(field.spacing),
                    domain_lower_m=field.origin.tolist(), domain_upper_m=field.upper.tolist(),
                    display_bounds_m=np.asarray(DEFAULT_DISPLAY_BOUNDS).tolist(),
                    grid_export='field_grid.npz', matplotlib_style=['science', 'no-latex', 'bright'],
                    control_or_physics_advanced=False)
    (destination/'visualization.json').write_text(json.dumps(metadata, indent=2)+'\n')
    return metadata


def write_interactive_field(field, robot, destination, *, meshes=None, results=(), path=None, target=None):
    """Self-contained HTML: legend toggles, z slices, gradient arrows, pose slider."""
    import plotly.graph_objects as go
    if meshes is None:
        meshes = build_field_meshes(field, bounds=DEFAULT_DISPLAY_BOUNDS)
    fig = go.Figure()
    for mesh in meshes:
        v, f = mesh['vertices'], mesh['faces']
        fig.add_trace(go.Mesh3d(x=v[:, 0], y=v[:, 1], z=v[:, 2],
                               i=f[:, 0], j=f[:, 1], k=f[:, 2], color=mesh['color'],
                               opacity=.15, name=f'h = {mesh["level"]:.3f} m²',
                               showlegend=True, hovertemplate=f'h = {mesh["level"]:.3f} m²<extra>constant barrier surface</extra>',
                               lighting=dict(ambient=.8, diffuse=.5, specular=.05)))
    for obstacle_index, obstacle in enumerate(robot.obstacles):
        v, f = box_mesh(obstacle['center'], obstacle['half_size'])
        fig.add_trace(go.Mesh3d(x=v[:, 0], y=v[:, 1], z=v[:, 2],
                               i=f[:, 0], j=f[:, 1], k=f[:, 2], color='#b85042',
                               opacity=.8, name=obstacle['name'], legendgroup='clutter',
                               showlegend=obstacle_index==0,
                               legendgrouptitle_text='Physical obstacles',
                               hovertemplate=obstacle['name']+'<extra>physical box</extra>'))
    from .plot_style import MODE_COLORS, MODE_LABELS
    for rows, summary in results:
        xyz = np.array([[r[f'tcp_{k}'] for k in 'xyz'] for r in rows])
        fig.add_trace(go.Scatter3d(x=xyz[:, 0], y=xyz[:, 1], z=xyz[:, 2], mode='lines',
                                   line=dict(color=MODE_COLORS[summary['mode']], width=6),
                                   name=MODE_LABELS[summary['mode']]+' TCP trace'))
    if path is not None:
        route = _tcp_route(robot, path)
        fig.add_trace(go.Scatter3d(x=route[:, 0], y=route[:, 1], z=route[:, 2],
                                   mode='lines+markers', line=dict(color='#009E73', dash='dash', width=3),
                                   marker=dict(size=3), name='Planned TCP waypoint guide'))
    if target is not None:
        fig.add_trace(go.Scatter3d(x=[target[0]], y=[target[1]], z=[target[2]], mode='markers',
                                   marker=dict(size=7, color='#CC79A7', symbol='diamond'), name='Task goal'))

    # Optional interior slices are hidden initially to preserve 3D depth cues.
    for height in (.20, .48, .70):
        iz = int(np.argmin(abs(field.axes[2]-height)))
        ix = np.flatnonzero((field.axes[0]>=DEFAULT_DISPLAY_BOUNDS[0][0]) &
                            (field.axes[0]<=DEFAULT_DISPLAY_BOUNDS[1][0]))
        iy = np.flatnonzero((field.axes[1]>=DEFAULT_DISPLAY_BOUNDS[0][1]) &
                            (field.axes[1]<=DEFAULT_DISPLAY_BOUNDS[1][1]))
        xx, yy = np.meshgrid(field.axes[0][ix], field.axes[1][iy], indexing='ij')
        fig.add_trace(go.Surface(x=xx, y=yy, z=np.full(xx.shape, field.axes[2][iz]),
                                 surfacecolor=field.h[np.ix_(ix, iy, [iz])][:, :, 0],
                                 colorscale=_colorscale(), cmin=0, cmax=.02, opacity=.8,
                                 showscale=False, showlegend=True, visible='legendonly',
                                 name=f'XY slice z={field.axes[2][iz]:.2f} m',
                                 hovertemplate='h=%{surfacecolor:.5f} m²<extra>slice (color saturates at 0.02)</extra>'))
    xs = np.linspace(-.5, .6, 9)
    ys = np.linspace(-.5, .5, 9)
    xx, yy = np.meshgrid(xs, ys)
    points = np.column_stack([xx.ravel(), yy.ravel(), np.full(xx.size, .48)])
    values, gradient = field.sample(points)
    keep = field.is_valid(points) & (values > 1e-5)
    points, gradient = points[keep], gradient[keep]
    norm = np.linalg.norm(gradient, axis=1, keepdims=True)
    direction = np.divide(gradient, norm, out=np.zeros_like(gradient), where=norm>1e-10)
    fig.add_trace(go.Cone(x=points[:, 0], y=points[:, 1], z=points[:, 2],
                          u=direction[:, 0], v=direction[:, 1], w=direction[:, 2],
                          sizemode='absolute', sizeref=.055, anchor='tail',
                          colorscale=[[0, '#374553'], [1, '#374553']], showscale=False,
                          name='Increasing-h directions (not forces)', showlegend=True,
                          visible='legendonly', hoverinfo='name'))

    frames_data = recorded_frames(results)
    previous = robot.get_q()
    animation, robot_trace_index = [], len(fig.data)
    try:
        for index, (time_s, q) in enumerate(frames_data):
            v, f = robot_mesh(robot, q)
            robot.set_q(q)
            p = robot.points()[::4]
            h, _ = field.sample(p)
            traces = [go.Mesh3d(x=v[:, 0], y=v[:, 1], z=v[:, 2], i=f[:, 0], j=f[:, 1], k=f[:, 2],
                                color='#374553', opacity=.92, showlegend=True,
                                name='Arm collision proxies', hoverinfo='name'),
                      go.Scatter3d(x=p[:, 0], y=p[:, 1], z=p[:, 2], mode='markers',
                                    marker=dict(size=2.4, color=h, colorscale=_colorscale(),
                                                cmin=0, cmax=.02, showscale=True,
                                                colorbar=dict(title='h (m²)', len=.23, y=.8, x=.01,
                                                              thickness=14, tickvals=[0,.01,.02],
                                                              bgcolor='rgba(255,255,255,0.9)')),
                                    showlegend=True, name='Body samples colored by h',
                                    customdata=h, hovertemplate='h=%{customdata:.5f} m²<extra>body sample</extra>')]
            if index == 0:
                fig.add_traces(traces)
            animation.append(go.Frame(name=str(index), data=traces,
                                       traces=[robot_trace_index, robot_trace_index+1]))
    finally:
        robot.set_q(previous)
    fig.frames = animation
    from .joint_layer import (build_joint_layer_data, joint_layer_controls,
                              joint_layer_details, joint_layer_style, joint_layer_script)
    from .math_explanation import math_explanation_html
    from .math_rendering import offline_math_assets, math_render_script
    from .comparison_view import build_comparison_data, comparison_fragment, comparison_script
    from .derivation_data import build_derivation_data
    from .derivation_tour import derivation_tour_script
    from .visual_model import build_visual_model_data
    from .command_comparison import (build_command_comparison, command_comparison_fragment,
                                     command_comparison_script, comparison_math_fragment,
                                     plot_command_comparison)
    layer_data = build_joint_layer_data(robot, field, frames_data)
    comparison_data = build_comparison_data(robot, field, frames_data)
    comparison_data['visual_model'] = build_visual_model_data(robot, frames_data)
    command_data = build_command_comparison(robot, field, frames_data)
    derivation_data = build_derivation_data(robot, field, frames_data, command_data=command_data)
    trace_indices = dict(arm=robot_trace_index, body=robot_trace_index+1)
    first = layer_data['frames'][0]
    origins = np.asarray(first['joint_origins_m'])
    trace_indices['joints'] = len(fig.data)
    fig.add_trace(go.Scatter3d(x=origins[:, 0], y=origins[:, 1], z=origins[:, 2],
                               mode='markers+text', text=[f'J{i+1}' for i in range(6)],
                               textposition='top center', marker=dict(size=5, color='#228833'),
                               hoverinfo='text', hovertext=layer_data['joint_labels'],
                               name='Joint holding-torque usage', showlegend=True))
    p = first['regions']['whole']['critical_point_m']
    trace_indices['critical'] = len(fig.data)
    fig.add_trace(go.Scatter3d(x=[p[0]], y=[p[1]], z=[p[2]], mode='markers',
                               marker=dict(size=7, color='#CC79A7', symbol='diamond'),
                               name='Critical body sample (selected region)', hoverinfo='name'))
    p = first['tcp_m']
    trace_indices['weight'] = len(fig.data)
    fig.add_trace(go.Scatter3d(x=[p[0], p[0]], y=[p[1], p[1]], z=[p[2], p[2]-.1],
                               mode='lines+markers+text', line=dict(color='#A44322', width=5),
                               marker=dict(size=[5, 2]), text=['Payload COM', ''], textposition='top center',
                               name='Payload weight direction (display length)', hoverinfo='name'))
    lower, upper = np.asarray(DEFAULT_DISPLAY_BOUNDS)
    axis = dict(showbackground=True, backgroundcolor='#f6f7f9', gridcolor='#d9dee5', zeroline=False)
    fig.update_layout(title=dict(text='B601 · 3D Poisson safety field', font=dict(size=24)),
                      font=dict(family='STIX Two Text, Times New Roman, serif', size=15, color='#273340'),
                      paper_bgcolor='white', margin=dict(l=0, r=240, t=70, b=100),
                      height=850, uirevision='keep-camera',
                      legend=dict(x=1.03, y=1, font=dict(size=13), groupclick='togglegroup'),
                      scene=dict(xaxis=dict(**axis, title='x (m)', range=[lower[0], upper[0]]),
                                 yaxis=dict(**axis, title='y (m)', range=[lower[1], upper[1]]),
                                 zaxis=dict(**axis, title='z (m)', range=[lower[2], upper[2]]),
                                 aspectmode='data', camera=dict(eye=dict(x=1.55, y=-1.8, z=1.25))))
    # A native range control is keyboard-accessible and remains correctly
    # aligned when the desktop app resizes a narrow WebGL viewport.
    pose_labels = ['Initial pose'] + [f'{time_s:.2f} s' for time_s, _ in frames_data[1:]]
    pose_control = ''
    if len(animation)>1:
        pose_control = (f'<div class="pose-control"><label for="recorded-pose">Recorded arm pose: '
                        f'<output id="pose-label">Initial pose</output></label><br>'
                        f'<input id="recorded-pose" type="range" min="0" max="{len(animation)-1}" '
                        f'step="1" value="0" aria-label="Recorded arm pose"></div>')
    figure_html = fig.to_html(full_html=False, include_plotlyjs=True, auto_play=False,
                              div_id='safety-field-3d',
                              config=dict(displaylogo=False, responsive=True,
                                          toImageButtonOptions=dict(format='png', scale=2)))
    document = '''<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1"><title>Robot safety filter demo · B601</title>
<style>body{margin:24px;background:#fff;color:#273340;font:17px/1.5 Georgia,serif}
main{max-width:1550px;margin:auto}h1{font-size:28px;margin-bottom:8px}p{max-width:1200px}
.note{padding:12px 18px;background:#f4f6f9;border-left:4px solid #0072B2}
code{font-size:14px}a{color:#0072B2}.pose-control{padding:8px 0}
#recorded-pose{width:min(700px,95%);accent-color:#0072B2}</style></head><body><main>
<h1>Robot safety filter demo · B601</h1>
<p>From geometry to a safe joint-velocity constraint: learn one step at a time on the same six-joint arm.</p>
<nav aria-label="Demo sections"><a href="#field-comparison">Interactive derivation</a> ·
<a href="#command-comparison">Joint command table</a> ·
<a href="#environment-detail">Joint-load explorer</a> ·
<a href="research.html">Wrist-camera research roadmap</a></nav>
COMPARISON_FRAGMENT
COMMAND_FRAGMENT
<details id="environment-detail"><summary><strong>Explore joint loads and the full environment field</strong></summary>
<p>Drag to rotate · Scroll to zoom · Click the legend to toggle shells, slices and arrows.
The pose slider replays recorded arm states.</p>
<p class="note">Each shell has constant barrier value <i>h</i> (m²); <i>h</i>=0 at buffered boundaries.
This is not a force or clearance scale. Arm appearance uses the official visual meshes;
collision checks continue to use the padded body boxes.
Arrows: direction of increasing <i>h</i>.</p>
''' + pose_control + joint_layer_controls(layer_data) + figure_html + joint_layer_details() + '''</details>
<details id="advanced-math-reference"><summary><strong>Detailed model reference (optional)</strong></summary>''' + comparison_math_fragment() + math_explanation_html(
    dict(grid_shape=field.shape, spacing_m=field.spacing, requested_inflation_m=field.inflation,
         model=robot.geometry_metadata)) + '''</details>
</main><script>
// Keep the scene usable in a narrow desktop side panel as well as a full tab.
const fieldPlot=document.getElementById('safety-field-3d');
const poseControl=document.getElementById('recorded-pose');
const poseLabels=POSE_LABELS_JSON;
const detailedPanel=document.getElementById('environment-detail');
function renderDetailedPose(){
  if(!poseControl)return;
  return Plotly.animate(fieldPlot,[String(poseControl.value)],{mode:'immediate',
    frame:{duration:0,redraw:true},transition:{duration:0}}).then(()=>{
      if(window.rebotUpdateJointLayer)window.rebotUpdateJointLayer();
      if(window.rebotUpdateDetailedVisuals)window.rebotUpdateDetailedVisuals();
    });
}
if(poseControl)poseControl.addEventListener('input',()=>{
  document.getElementById('pose-label').textContent=poseLabels[Number(poseControl.value)];
  if(detailedPanel.open)renderDetailedPose();
});
detailedPanel.addEventListener('toggle',()=>{if(detailedPanel.open){Plotly.Plots.resize(fieldPlot);renderDetailedPose();}});
let lastNarrow=null;
function fitFieldView(){
  const narrow=window.innerWidth<1050;
  if(narrow===lastNarrow)return;
  lastNarrow=narrow;
  Plotly.relayout(fieldPlot,narrow ? {
    height:1000,'margin.l':0,'margin.r':25,'margin.t':65,'margin.b':270,
    'legend.orientation':'h','legend.x':0,'legend.y':-.10,
    'legend.font.size':12,'legend.entrywidth':.48,'legend.entrywidthmode':'fraction'
  } : {
    height:850,'margin.l':0,'margin.r':240,'margin.t':70,'margin.b':100,
    'legend.orientation':'v','legend.x':1.03,'legend.y':1,
    'legend.font.size':13,'legend.entrywidth':0,'legend.entrywidthmode':'pixels'
  });
}
fitFieldView();window.addEventListener('resize',fitFieldView);
</script></body></html>'''
    document = document.replace('POSE_LABELS_JSON', json.dumps(pose_labels))
    document = document.replace('COMPARISON_FRAGMENT', comparison_fragment(comparison_data))
    document = document.replace('COMMAND_FRAGMENT', command_comparison_fragment())
    document = document.replace('COMPARISON_MATH', comparison_math_fragment())
    document = document.replace('</head>', joint_layer_style()+'</head>')
    document = document.replace('</body>', offline_math_assets()+joint_layer_script(layer_data,trace_indices)
                                +comparison_script(comparison_data)+command_comparison_script(command_data)
                                +derivation_tour_script(derivation_data)
                                +'<script>'+(Path(__file__).resolve().parents[1]/'assets/detailed_visual.js').read_text()+'</script>'
                                +math_render_script()+'</body>')
    Path(destination).write_text(document)
    Path(destination).with_name('joint_load_layer.json').write_text(json.dumps(layer_data, indent=2)+'\n')
    Path(destination).with_name('field_comparison_data.json').write_text(json.dumps(comparison_data, separators=(',', ':'))+'\n')
    Path(destination).with_name('command_comparison_data.json').write_text(json.dumps(command_data, indent=2)+'\n')
    Path(destination).with_name('derivation_data.json').write_text(json.dumps(derivation_data, separators=(',', ':'), allow_nan=False)+'\n')
    Path(destination).with_name('visual_model_data.json').write_text(json.dumps(comparison_data['visual_model'], separators=(',', ':'), allow_nan=False)+'\n')
    plot_command_comparison(command_data, Path(destination).with_name('filter_command_comparison.png'))
    return dict(isosurface_levels_m2=[m['level'] for m in meshes],
                mesh_triangle_counts=[len(m['faces']) for m in meshes],
                interactive_pose_count=len(frames_data),
                pose_source='saved hierarchy telemetry plus initial pose' if len(frames_data)>1 else 'initial pose only',
                body_sample_display_stride=4, cbf_constraint_sampling_changed=False,
                html_standalone=True, field_units='m^2 (not distance or force)',
                joint_load_layer=dict(static_only=True, regions=list(layer_data['regions']),
                                      payload_geometry_in_physics=False, torque_filter_deployed=False),
                mathematics_rendered_offline=True,
                parallel_comparison=dict(pose_count=len(frames_data), margins_m=comparison_data['margins_m'],
                                         command_cases_per_pose=len(command_data['command_names']),
                                         guided_derivation=True,
                                         closed_loop_distance_comparison=False))


def visualize_saved(run_directory, destination):
    """Replot the supported fixed-clutter scene without advancing dynamics.

    Legacy records predate explicit obstacle/grid exports. They are labeled as
    reconstructed from this fixed scene rather than as an archived field load.
    Refuse mismatched geometry instead of silently overlaying another model.
    """
    from .robot import RebotModel
    from .planning import ReachabilityMap
    from .plotting import plot_preparation, plot_runs
    root = Path(run_directory)
    configuration = json.loads((root/'configuration.json').read_text())
    obstacles = configuration.get('obstacles', clutter_scene())
    if obstacles != clutter_scene() or not np.allclose(configuration['start_q'], START):
        raise ValueError('Saved visualization supports the fixed clutter scene and its documented start pose only')
    field = build_field(obstacles, spacing=configuration['spacing_m'],
                        inflation=configuration['requested_inflation_m'])
    model_config = configuration['model']
    robot = RebotModel(obstacles=obstacles,
                       epsilon=model_config['requested_epsilon_m'],
                       proxy_padding=model_config['padding_m'])
    if list(field.shape) != configuration['grid_shape']:
        raise ValueError('Current field extent differs from the recorded grid')
    if 'grid_origin_m' in configuration and not np.allclose(field.origin, configuration['grid_origin_m']):
        raise ValueError('Current field origin differs from the recorded grid')
    old_bounds, new_bounds = model_config['bounds'], robot.geometry_metadata['bounds']
    if (len(old_bounds) != len(new_bounds) or any(
            a['body'] != b['body'] or not np.allclose(a['center'], b['center'])
            or not np.allclose(a['half_size'], b['half_size'])
            for a, b in zip(old_bounds, new_bounds))):
        raise ValueError('Current robot collision geometry differs from the saved run')
    robot.set_q(START)
    reach = ReachabilityMap.load(root/'reachability.npz')
    results = []
    for mode in ('unfiltered', 'filter-only', 'hierarchy'):
        folder = root/mode
        if not (folder/'telemetry.csv').is_file():
            continue
        with (folder/'telemetry.csv').open() as stream:
            rows = [{k: float(v) for k, v in row.items()} for row in csv.DictReader(stream)]
        results.append((rows, json.loads((folder/'summary.json').read_text())))
    plan = json.loads((root/'plan.json').read_text()) if (root/'plan.json').exists() else {}
    path = np.array(plan['path']) if plan.get('path') is not None else None
    target = np.array(configuration['target_xyz'])
    Path(destination).mkdir(parents=True, exist_ok=True)
    plot_preparation(field, reach, target, destination)
    if results:
        plot_runs(results, Path(destination)/'comparison.png')
    metadata = plot_field_3d(field, robot, destination, results=results, path=path, target=target)
    metadata.update(source_run=str(root.resolve()), replay_only=True,
                    source_configuration=configuration,
                    field_source='reconstructed from current fixed-scene code and recorded grid/inflation; not an archived field load',
                    legacy_record_without_scene_snapshot='obstacles' not in configuration,
                    dynamics_benchmark_reexecuted=False)
    (Path(destination)/'visualization.json').write_text(json.dumps(metadata, indent=2)+'\n')
    return metadata
