"""Official B601-DM URDF appearance, independently checked against replay FK.

This is a display adapter. It never replaces the existing collision proxies,
inertials, controls or recorded experiment. The Seeed-hosted DM assets are
pinned and checked against their Git blobs; their inherited licensing still
requires the upstream provenance notices. No hardware or network is accessed.
"""
from __future__ import annotations

import hashlib
import math
import os
from pathlib import Path
import struct
import subprocess
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

from .load_model import _preserve_integration_state


SEEED_REPOSITORY = 'https://github.com/Seeed-Projects/reBotArmController_ROS2'
SEEED_COMMIT = '4a2b31b569d7856ea310b90f24f7de82374dbc75'
URDF_RELATIVE = 'src/rebotarm_bringup/description/DM/urdf/ReBot_Arm_DM.urdf'
RECORDED_REPOSITORY = 'https://github.com/Yang-Ci/ReBot_Arm_DigitalTwin_DM'
RECORDED_COMMIT = '914c763c4adffb5f5744c43ecffbf6199c282bd6'


def _vec(value, default):
    return np.asarray(default if value is None else [float(x) for x in value.split()], dtype=float)


def rpy_matrix(rpy):
    """URDF fixed-axis roll/pitch/yaw: Rz(yaw) Ry(pitch) Rx(roll)."""
    r, p, y = np.asarray(rpy, dtype=float)
    cr, sr, cp, sp, cy, sy = math.cos(r), math.sin(r), math.cos(p), math.sin(p), math.cos(y), math.sin(y)
    return (np.array([[cy, -sy, 0.], [sy, cy, 0.], [0., 0., 1.]])
            @np.array([[cp, 0., sp], [0., 1., 0.], [-sp, 0., cp]])
            @np.array([[1., 0., 0.], [0., cr, -sr], [0., sr, cr]]))


def _origin(element):
    result = np.eye(4)
    if element is not None:
        result[:3, 3] = _vec(element.get('xyz'), (0., 0., 0.))
        result[:3, :3] = rpy_matrix(_vec(element.get('rpy'), (0., 0., 0.)))
    return result


def transform_visual_vertices(vertices, *, scale=(1., 1., 1.), xyz=(0., 0., 0.), rpy=(0., 0., 0.)):
    """URDF mesh scale, then visual-origin rotation and translation."""
    vertices, scale, xyz, rpy = (np.asarray(value, dtype=float) for value in (vertices, scale, xyz, rpy))
    if (vertices.ndim != 2 or vertices.shape[1] != 3 or scale.shape != (3,)
            or xyz.shape != (3,) or rpy.shape != (3,)
            or not all(np.all(np.isfinite(value)) for value in (vertices, scale, xyz, rpy))
            or np.any(scale <= 0)):
        raise ValueError('Visual geometry needs finite Nx3 vertices and finite positive XYZ scale')
    return (vertices*scale)@rpy_matrix(rpy).T+xyz


class URDFKinematics:
    """Minimal independent FK for fixed, revolute/continuous and slide joints."""
    def __init__(self, urdf_path):
        self.path = Path(urdf_path)
        self.root = ET.parse(self.path).getroot()
        self.links = tuple(link.get('name') for link in self.root.findall('link'))
        self.joints = self.root.findall('joint')
        children = {joint.find('child').get('link') for joint in self.joints}
        roots = set(self.links)-children
        if len(roots) != 1:
            raise ValueError('URDF must have exactly one root link')
        self.root_link = roots.pop()

    def forward(self, positions):
        positions = dict(positions)
        transforms = {self.root_link: np.eye(4)}
        pending = list(self.joints)
        while pending:
            progressed = False
            for joint in list(pending):
                parent, child = joint.find('parent').get('link'), joint.find('child').get('link')
                if parent not in transforms:
                    continue
                name, kind = joint.get('name'), joint.get('type')
                mimic = joint.find('mimic')
                if mimic is not None:
                    value = (positions.get(mimic.get('joint'), 0.)*float(mimic.get('multiplier', 1.))
                             +float(mimic.get('offset', 0.)))
                else:
                    value = float(positions.get(name, 0.))
                motion = np.eye(4)
                if kind in ('revolute', 'continuous', 'prismatic'):
                    axis = _vec(joint.find('axis').get('xyz') if joint.find('axis') is not None else None, (1., 0., 0.))
                    if not np.all(np.isfinite(axis)) or np.linalg.norm(axis) == 0 or not np.isfinite(value):
                        raise ValueError('Invalid URDF joint axis or position')
                    axis = axis/np.linalg.norm(axis)
                    if kind == 'prismatic':
                        motion[:3, 3] = axis*value
                    else:
                        x, y, z = axis
                        cross = np.array([[0., -z, y], [z, 0., -x], [-y, x, 0.]])
                        motion[:3, :3] = np.eye(3)+math.sin(value)*cross+(1-math.cos(value))*(cross@cross)
                elif kind != 'fixed':
                    raise ValueError(f'Unsupported URDF joint type: {kind}')
                transforms[child] = transforms[parent]@_origin(joint.find('origin'))@motion
                positions[name] = value
                pending.remove(joint); progressed = True
            if not progressed:
                raise ValueError('URDF joint tree is cyclic or disconnected')
        return transforms


def read_stl_mesh(path):
    """Read indexed native STL triangles, retaining their original winding."""
    raw = Path(path).read_bytes()
    count = struct.unpack_from('<I', raw, 80)[0] if len(raw) >= 84 else 0
    if len(raw) != 84+50*count or not count:
        raise ValueError('Expected a nonempty binary STL from the pinned model')
    dtype = np.dtype([('normal', '<f4', (3,)), ('vertices', '<f4', (3, 3)), ('attribute', '<u2')])
    triangles = np.frombuffer(raw, dtype=dtype, count=count, offset=84)['vertices'].reshape(-1, 3).astype(float)
    if not np.all(np.isfinite(triangles)):
        raise ValueError('STL coordinates must be finite')
    vertices, inverse = np.unique(triangles, axis=0, return_inverse=True)
    return vertices, inverse.reshape(-1, 3)


def _cluster_mesh(vertices, faces, spacing):
    grid = np.floor((vertices-vertices.min(axis=0))/spacing).astype(np.int64)
    _, labels = np.unique(grid, axis=0, return_inverse=True)
    counts = np.bincount(labels)
    reduced = np.column_stack([np.bincount(labels, weights=vertices[:, axis])/counts for axis in range(3)])
    triangles = labels[faces]
    keep = ((triangles[:, 0] != triangles[:, 1]) & (triangles[:, 1] != triangles[:, 2])
            & (triangles[:, 0] != triangles[:, 2]))
    triangles = triangles[keep]
    if len(triangles):
        points = reduced[triangles]
        triangles = triangles[np.linalg.norm(np.cross(points[:, 1]-points[:, 0], points[:, 2]-points[:, 0]), axis=1) > 1e-16]
        triangles = np.unique(triangles, axis=0)
    if not len(triangles):
        return vertices, faces
    used, indices = np.unique(triangles, return_inverse=True)
    return reduced[used], indices.reshape(-1, 3)


def simplify_visual_mesh(vertices, faces, target_faces, *, max_spacing=.003):
    """Deterministic display-only vertex clustering; never triangle striding.

    Material parts remain separate. Clustering may remove submillimetre detail
    and alter topology. Reported nearest-vertex deviations are diagnostics, not
    a surface Hausdorff bound or a conservative collision approximation.
    """
    if target_faces < 1 or max_spacing <= 0:
        raise ValueError('Visual mesh budgets must be positive')
    if len(faces) <= target_faces:
        return vertices, faces, dict(method='none', cluster_spacing_m=0., original_faces=len(faces), output_faces=len(faces), nearest_vertex_deviation_max_m=0.)
    from scipy.spatial import cKDTree
    low, spacing = 0., .0003
    best, best_spacing = (vertices, faces), 0.
    while True:
        reduced = _cluster_mesh(vertices, faces, spacing)
        best, best_spacing = reduced, spacing
        if len(reduced[1]) <= target_faces or spacing >= max_spacing:
            break
        low, spacing = spacing, min(max_spacing, spacing*1.5)
    if len(best[1]) <= target_faces:
        high = best_spacing
        for _ in range(3):
            middle = (low+high)/2
            reduced = _cluster_mesh(vertices, faces, middle)
            if len(reduced[1]) <= target_faces:
                best, best_spacing, high = reduced, middle, middle
            else:
                low = middle
    deviation = cKDTree(best[0]).query(vertices, workers=1)[0]
    # Face count is a soft target: retain fine disconnected trim instead of
    # accepting a collapsed feature merely to meet a browser budget.
    for _ in range(5):
        if deviation.max() <= .003 or best_spacing == 0:
            break
        best_spacing *= .7
        best = _cluster_mesh(vertices, faces, best_spacing)
        deviation = cKDTree(best[0]).query(vertices, workers=1)[0]
    return best[0], best[1], dict(method='deterministic voxel vertex clustering', cluster_spacing_m=float(best_spacing),
                                  original_faces=len(faces), output_faces=len(best[1]),
                                  nearest_vertex_deviation_max_m=float(deviation.max()),
                                  nearest_vertex_deviation_p95_m=float(np.percentile(deviation, 95)),
                                  nearest_vertex_quality_target_m=.003,
                                  surface_hausdorff_bound=False, collision_geometry=False)


def _git(repo, *arguments):
    return subprocess.run(['git', '-C', str(repo), *arguments], check=True, capture_output=True).stdout


def _source_mesh_path(uri, repo):
    if not uri.startswith('package://rebotarm_bringup/'):
        raise ValueError('Pinned official visuals must use the rebotarm_bringup package URI')
    path = (repo/'src'/uri.removeprefix('package://')).resolve()
    path.relative_to(repo.resolve())
    return path


def build_visual_model_data(robot, frames, *, official_repo=None, target_faces=80_000):
    """Export official local geometry and independently verified replay transforms.

    ``frames`` contains (time_s,q6) pairs. A kinematic discrepancy larger than
    1 micrometre or 1 microradian rejects replay rather than relabeling old
    recorded q as a different official robot. Finger positions remain the
    recorded fixed opening, +0.02 and -0.02 m.
    """
    root = Path(os.environ.get('REBOT_ROOT', Path(__file__).resolve().parents[3]))
    repo = Path(official_repo or os.environ.get('SEEED_REBOT_ROOT', root/'rebotarm_ros2')).resolve()
    path = repo/URDF_RELATIVE
    commit = _git(repo, 'rev-parse', 'HEAD').decode().strip()
    if commit != SEEED_COMMIT:
        raise ValueError('Official model checkout is not the pinned, verified Seeed revision')
    frames = list(frames)
    if not frames or not isinstance(target_faces, int) or target_faces < 1000:
        raise ValueError('Recorded poses and a visual face budget >= 1000 are required')
    model = URDFKinematics(path)
    global_colors = {material.get('name'): _vec(material.find('color').get('rgba'), (0.7, 0.7, 0.7, 1.))
                     for material in model.root.findall('material') if material.find('color') is not None}
    visuals = []
    recorded_mesh_directory = robot.source_path.parents[2]/'rebotarm_bringup/description/meshes'
    sources = {URDF_RELATIVE: hashlib.sha256(path.read_bytes()).hexdigest()}
    for link in model.root.findall('link'):
        for visual in link.findall('visual'):
            mesh = visual.find('geometry/mesh')
            if mesh is None:
                raise ValueError('Expected mesh-only pinned official visual model')
            source = _source_mesh_path(mesh.get('filename'), repo)
            rel = str(source.relative_to(repo))
            raw = source.read_bytes()
            # Prove the working file is exactly the pinned Git blob.
            if raw != _git(repo, 'show', f'{commit}:{rel}'):
                raise ValueError(f'Official visual source has local changes: {rel}')
            sources[rel] = hashlib.sha256(raw).hexdigest()
            licensed_copy = recorded_mesh_directory/source.name
            if not licensed_copy.is_file() or hashlib.sha256(licensed_copy.read_bytes()).hexdigest() != sources[rel]:
                raise ValueError('The official visual asset differs from the licensed recorded-model copy')
            # Derive redistributed simplified geometry from the existing
            # CERN-OHL-W-2.0 copy, after proving the official STL is identical.
            vertices, faces = read_stl_mesh(licensed_copy)
            scale = _vec(mesh.get('scale'), (1., 1., 1.))
            if scale.shape != (3,) or not np.all(np.isfinite(scale)) or np.any(scale <= 0):
                raise ValueError('URDF mesh scale must contain three finite positive values')
            origin = _origin(visual.find('origin'))
            visual_origin = visual.find('origin')
            visual_rpy = _vec(visual_origin.get('rpy') if visual_origin is not None else None, (0., 0., 0.))
            vertices = transform_visual_vertices(vertices, scale=scale, xyz=origin[:3, 3], rpy=visual_rpy)
            material = visual.find('material')
            material_name = material.get('name', '') if material is not None else ''
            rgba = global_colors.get(material_name, np.array([.7, .7, .7, 1.]))
            if material is not None and material.find('color') is not None:
                rgba = _vec(material.find('color').get('rgba'), (.7, .7, .7, 1.))
            if rgba.shape != (4,) or not np.all(np.isfinite(rgba)) or np.any(rgba < 0) or np.any(rgba > 1):
                raise ValueError('URDF RGBA must contain four values in [0,1]')
            visuals.append(dict(name=visual.get('name', source.stem), body_name=link.get('name'),
                                vertices=vertices, faces=faces, material_name=material_name,
                                rgba=rgba.tolist(), color='#'+''.join(f'{int(round(x*255)):02x}' for x in rgba[:3]),
                                opacity=float(rgba[3]), source_mesh=rel, source_sha256=sources[rel],
                                visual_origin_link_m=origin[:3, 3].tolist(),
                                visual_rotation_link=origin[:3, :3].tolist(), scale=scale.tolist()))
    if path.read_bytes() != _git(repo, 'show', f'{commit}:{URDF_RELATIVE}'):
        raise ValueError('Official URDF has uncommitted local modifications')
    total_source_faces = sum(len(visual['faces']) for visual in visuals)
    packed_meshes = []
    mesh_matches = []
    for visual in visuals:
        budget = max(min(500, len(visual['faces'])), int(target_faces*len(visual['faces'])/total_source_faces))
        vertices, faces, simplification = simplify_visual_mesh(visual['vertices'], visual['faces'], budget)
        old = recorded_mesh_directory/Path(visual['source_mesh']).name
        same = old.is_file() and hashlib.sha256(old.read_bytes()).hexdigest() == visual['source_sha256']
        mesh_matches.append(same)
        packed = {key: value for key, value in visual.items() if key not in ('vertices', 'faces')}
        packed.update(vertices=np.round(vertices, 8).tolist(), faces=faces.tolist(), simplification=simplification,
                      matches_recorded_visual_source_sha256=bool(same))
        packed_meshes.append(packed)
    records, position_errors, rotation_errors = [], [], []
    joint_map = {f'joint{i}': f'joint{i}' for i in range(1, 7)}
    with _preserve_integration_state(robot.model, robot.data):
        for time_s, q in frames:
            q = np.asarray(q, dtype=float)
            if q.shape != (6,) or not np.all(np.isfinite(q)) or not np.isfinite(time_s):
                raise ValueError('Visual replay requires finite timestamps and q6')
            positions = {f'joint{i+1}': float(value) for i, value in enumerate(q)}
            positions.update(finger_left=.02, finger_right=-.02)
            official = model.forward(positions)
            robot.set_q(q)
            for body_name in model.links:
                body_id = mujoco.mj_name2id(robot.model, mujoco.mjtObj.mjOBJ_BODY, body_name)
                if body_id < 0:
                    raise ValueError(f'Official link has no recorded-model counterpart: {body_name}')
                pose = official[body_name]
                position_errors.append(float(np.linalg.norm(pose[:3, 3]-robot.data.xpos[body_id])))
                relative = pose[:3, :3].T@robot.data.xmat[body_id].reshape(3, 3)
                rotation_errors.append(float(np.arccos(np.clip((np.trace(relative)-1)/2, -1., 1.))))
            records.append(dict(time_s=float(time_s), q_rad=q.tolist(), transforms=[
                dict(rotation_world_from_link=np.round(official[visual['body_name']][:3, :3], 12).tolist(),
                     translation_world_m=np.round(official[visual['body_name']][:3, 3], 12).tolist())
                for visual in visuals]))
    max_position, max_rotation = max(position_errors), max(rotation_errors)
    kinematics_match = max_position <= 1e-6 and max_rotation <= 1e-6
    if not kinematics_match or not all(mesh_matches):
        raise ValueError(f'Official appearance cannot be replayed over unchanged experiment: position={max_position}, rotation={max_rotation}, meshes={all(mesh_matches)}')
    return dict(schema='rebot-official-visual-model-v1', meshes=packed_meshes, frames=records,
                provenance=dict(repository=SEEED_REPOSITORY, commit=commit, urdf=URDF_RELATIVE,
                                source_sha256=sources, package_license_tag='Apache-2.0',
                                repository_root_license_found=(repo/'LICENSE').is_file(),
                                inherited_mesh_provenance='All 34 DM visual STLs are byte-identical to the pinned Yang-Ci model; retain inherited upstream notices.',
                                authoritative_license_review='See demo SOURCE_PROVENANCE; package tags do not override inherited asset terms.',
                                original_triangles=total_source_faces,
                                display_triangles=sum(len(mesh['faces']) for mesh in packed_meshes),
                                requested_display_face_budget=target_faces,
                                display_face_budget_is_soft_for_shape_retention=True,
                                maximum_nearest_vertex_deviation_m=max(mesh['simplification']['nearest_vertex_deviation_max_m'] for mesh in packed_meshes),
                                display_mesh_derivation_repository=RECORDED_REPOSITORY,
                                display_mesh_derivation_commit=RECORDED_COMMIT,
                                display_mesh_derivation_license='CERN-OHL-W-2.0',
                                simplification='Display-only deterministic vertex clustering; original materials and all visual parts retained. Small detail/topology can change.',
                                geometry_used_for_collision_or_control=False),
                compatibility=dict(compatible=True, kinematics_match=kinematics_match,
                                   source_meshes_match=bool(all(mesh_matches)),
                                   max_link_position_error_m=max_position,
                                   max_link_rotation_error_rad=max_rotation,
                                   joint_mapping=joint_map, fixed_finger_positions_m={'finger_left': .02, 'finger_right': -.02},
                                   note='Official DM visual meshes and joint kinematics match the recorded model; source file paths changed. Appearance only; original safety proxies and recorded control remain unchanged.'),
                units='m', coordinate_frame='fixed-base MuJoCo world',
                vertices_include_visual_origin_and_scale=True,
                physics_advanced=False, controls_or_collision_geometry_changed=False)
