"""Fixed-base B601 geometry and dynamics for the preliminary CBF experiment.

The vendor model has detailed visual meshes, but no collision geometry for most
arm links.  This module deliberately adds conservative *body-local boxes*.  The
CBF samples their surfaces and MuJoCo checks those same boxes, so an attractive
rendering cannot silently stand in for a whole-arm collision model.  These boxes
can be substantially larger than the actual mesh; that conservatism is reported
as a limitation rather than tuned away after an experiment.

No ROS nodes, physical drivers, viewer, or network connection are started here.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import math
import os
import re
import struct
import xml.etree.ElementTree as ET

import mujoco
import numpy as np


@dataclass(frozen=True)
class ProxyBox:
    """A box enclosing every visual-mesh vertex in one body's local frame."""

    body_name: str
    center: np.ndarray
    half_size: np.ndarray
    visual_lower: np.ndarray
    visual_upper: np.ndarray


def _numbers(text: str | None, default: tuple[float, ...]) -> np.ndarray:
    return np.asarray(default if text is None else [float(v) for v in text.split()], dtype=float)


def _rotation(element: ET.Element) -> np.ndarray:
    """Read only orientation encodings used/supported by this MJCF adaptation."""
    if "quat" in element.attrib:
        quaternion = _numbers(element.get("quat"), (1.0, 0.0, 0.0, 0.0))
        quaternion /= np.linalg.norm(quaternion)
        matrix = np.empty(9)
        mujoco.mju_quat2Mat(matrix, quaternion)
        return matrix.reshape(3, 3)
    if "euler" in element.attrib:
        # Source compiler uses angle=radian and eulerseq=xyz.  Intrinsic xyz is
        # R_x @ R_y @ R_z in MuJoCo's local-frame orientation convention.
        x, y, z = _numbers(element.get("euler"), (0.0, 0.0, 0.0))
        cx, sx, cy, sy, cz, sz = math.cos(x), math.sin(x), math.cos(y), math.sin(y), math.cos(z), math.sin(z)
        return np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]]) @ np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]]) @ np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
    if any(key in element.attrib for key in ("axisangle", "xyaxes", "zaxis")):
        raise ValueError("Unsupported mesh-geom orientation in vendor MJCF; update geometry conversion explicitly")
    return np.eye(3)


def _stl_vertices(path: Path) -> np.ndarray:
    """Read STL coordinates before MuJoCo recentres/rotates mesh assets.

    MuJoCo's mesh_vert is in a compiled principal-axis frame, so treating it as
    the original body-local coordinates would create incorrectly placed boxes.
    Reading the original STL avoids that common enclosure error.
    """
    raw = path.read_bytes()
    if len(raw) >= 84:
        count = struct.unpack_from("<I", raw, 80)[0]
        if len(raw) == 84 + 50 * count:
            dtype = np.dtype([("normal", "<f4", (3,)), ("vertices", "<f4", (3, 3)), ("attribute", "<u2")])
            result = np.frombuffer(raw, dtype=dtype, count=count, offset=84)["vertices"].reshape(-1, 3).astype(float)
            if not len(result) or not np.all(np.isfinite(result)):
                raise ValueError(f"Empty or nonfinite STL: {path}")
            return result
    vertices = re.findall(rb"\bvertex\s+([-+\deE.]+)\s+([-+\deE.]+)\s+([-+\deE.]+)", raw)
    if not vertices:
        raise ValueError(f"Unsupported or empty STL: {path}")
    return np.asarray(vertices, dtype=float)


def _box_surface(center: np.ndarray, half_size: np.ndarray, epsilon: float) -> tuple[np.ndarray, float]:
    """Sample all six faces with a certified Euclidean surface-cover radius.

    Every rectangular face cell is sampled at all four corners.  Its furthest
    point is at most half its diagonal from a sample.  Spacing <= sqrt(2)*eps
    therefore covers every point on every face within eps, including edges.
    """
    samples = []
    actual_radius = 0.0
    spacing = math.sqrt(2.0) * epsilon
    for normal_axis in range(3):
        a, b = [axis for axis in range(3) if axis != normal_axis]
        na = max(1, math.ceil(2.0 * half_size[a] / spacing))
        nb = max(1, math.ceil(2.0 * half_size[b] / spacing))
        av = np.linspace(-half_size[a], half_size[a], na + 1)
        bv = np.linspace(-half_size[b], half_size[b], nb + 1)
        actual_radius = max(actual_radius, 0.5 * math.hypot(2.0 * half_size[a] / na, 2.0 * half_size[b] / nb))
        aa, bb = np.meshgrid(av, bv, indexing="ij")
        for sign in (-1.0, 1.0):
            face = np.zeros((aa.size, 3))
            face[:, normal_axis] = sign * half_size[normal_axis]
            face[:, a] = aa.ravel()
            face[:, b] = bb.ravel()
            samples.append(face + center)
    return np.unique(np.round(np.concatenate(samples), 12), axis=0), actual_radius


class RebotModel:
    """Six-DOF B601 with fixed open fingers and conservative collision boxes.

    ``obstacles`` contains dictionaries with name, center and half_size (metres).
    Obstacles are static axis-aligned boxes.  The gripper is fixed at 0.04 m total
    slide opening.  Joint control is saturated PD plus MuJoCo bias compensation,
    not a torque-level CBF or a physical-robot controller.
    """

    def __init__(self, obstacles=None, timestep: float = 0.002, epsilon: float = 0.035,
                 model_path: str | Path | None = None, proxy_padding: float = 0.002):
        if not (np.isfinite(epsilon) and epsilon > 0 and np.isfinite(timestep) and timestep > 0):
            raise ValueError("epsilon and timestep must be finite and positive")
        if not (np.isfinite(proxy_padding) and proxy_padding >= 0):
            raise ValueError("proxy_padding must be finite and nonnegative")
        root_path = Path(os.environ.get("REBOT_ROOT", Path(__file__).resolve().parents[3]))
        self.source_path = Path(model_path) if model_path else root_path / "rebotarm_digital_twin_dm/reBotArm_ros2_DM/src/rebotarm_mujoco/models/rebotarm_b601_colored.xml"
        xml = ET.parse(self.source_path).getroot()
        compiler = xml.find("compiler")
        meshdir = (self.source_path.parent / compiler.get("meshdir", "")).resolve()
        if not meshdir.is_dir():
            # The source tree stores meshes in the bringup package; the installed
            # ament package copies them beside its models directory instead.
            meshdir = self.source_path.parents[2] / "rebotarm_bringup/description/meshes"
        if not meshdir.is_dir():
            raise FileNotFoundError(f"Cannot find vendor mesh directory: {meshdir}")
        compiler.set("meshdir", str(meshdir))
        # Restore the canonical palette without importing the ROS-dependent
        # upstream loader into this standalone experiment.
        visual_urdf = self.source_path.parents[2] / "rebotarm_bringup/description/urdf/ReBot_Arm_DM.urdf"
        if visual_urdf.exists():
            palette = {}
            for material in ET.parse(visual_urdf).getroot().findall('material'):
                color = material.find('color')
                if color is not None and color.get('rgba') is not None:
                    palette[material.get('name')] = color.get('rgba')
            for material in xml.find('asset').findall('material'):
                key = material.get('name','').removesuffix('_mat')
                if key in palette:
                    material.set('rgba',palette[key])
        xml.find("option").set("timestep", str(timestep))
        xml.find("option").set("integrator", "implicitfast")
        self.epsilon = float(epsilon)
        self.proxy_padding = float(proxy_padding)
        self.obstacles = []
        world = xml.find("worldbody")
        for body in list(world.findall("body")):
            if body.get("name") != "base_link":
                world.remove(body)
        for node in list(xml):
            if node.tag in {"contact", "actuator", "sensor", "equality", "keyframe"}:
                xml.remove(node)
        # Keep floor as a visual reference.  It is NOT an environment obstacle:
        # callers must include a ground box if floor avoidance is required.
        for geom in xml.iter("geom"):
            geom.set("contype", "0")
            geom.set("conaffinity", "0")
            geom.set("group", "1")
        for body in world.iter("body"):
            name = body.get("name")
            if name in {"finger_left_link", "finger_right_link"}:
                for joint in list(body.findall("joint")):
                    body.remove(joint)
                body.set("pos", "0 0.02 0" if name == "finger_left_link" else "0 -0.02 0")

        assets = {mesh.get("name"): mesh for mesh in xml.find("asset").findall("mesh")}
        self.proxy_boxes: list[ProxyBox] = []
        self.visual_vertices: dict[str, np.ndarray] = {}
        local_point_arrays, local_names = [], []
        self.coverage_radius = 0.0
        for body in world.iter("body"):
            vertices = []
            for geom in body.findall("geom"):
                if geom.get("type") != "mesh" or not geom.get("name", "").endswith("_visual"):
                    continue
                asset = assets[geom.get("mesh")]
                if any(attr in asset.attrib for attr in ("refpos", "refquat")):
                    raise ValueError("Vendor mesh has refpos/refquat; explicit raw-mesh conversion needs updating")
                raw = _stl_vertices(meshdir / asset.get("file"))
                raw *= _numbers(asset.get("scale"), (1.0, 1.0, 1.0))
                vertices.append(raw @ _rotation(geom).T + _numbers(geom.get("pos"), (0.0, 0.0, 0.0)))
            if not vertices:
                continue
            name = body.get("name")
            combined = np.concatenate(vertices)
            lower, upper = combined.min(axis=0), combined.max(axis=0)
            center = (lower + upper) / 2.0
            half_size = (upper - lower) / 2.0 + proxy_padding
            if np.any(half_size <= 0):
                raise ValueError(f"Degenerate collision proxy: {name}")
            self.visual_vertices[name] = combined
            self.proxy_boxes.append(ProxyBox(name, center, half_size, lower, upper))
            ET.SubElement(body, "geom", {
                "name": f"cbf_proxy_{name}", "type": "box",
                "pos": " ".join(map(str, center)), "size": " ".join(map(str, half_size)),
                "mass": "0", "contype": "1", "conaffinity": "2", "group": "3",
                "rgba": "0.2 0.85 0.35 0.09", "friction": "0.8 0.1 0.1"})
            sampled, radius = _box_surface(center, half_size, epsilon)
            local_point_arrays.append(sampled)
            local_names.extend([name] * len(sampled))
            self.coverage_radius = max(self.coverage_radius, radius)
        self.local_points = np.concatenate(local_point_arrays)
        self.point_body_names = np.asarray(local_names)

        for index, obstacle in enumerate(obstacles or []):
            name = str(obstacle.get("name", f"obstacle_{index}"))
            center = np.asarray(obstacle["center"], dtype=float)
            half_size = np.asarray(obstacle["half_size"], dtype=float)
            if center.shape != (3,) or half_size.shape != (3,) or not np.all(np.isfinite(center)) or not np.all(np.isfinite(half_size)) or np.any(half_size <= 0):
                raise ValueError(f"Invalid static box obstacle {name}")
            if any(item["name"] == name for item in self.obstacles):
                raise ValueError(f"Duplicate obstacle name: {name}")
            self.obstacles.append({"name": name, "center": center.copy(), "half_size": half_size.copy()})
            ET.SubElement(world, "geom", {
                "name": f"cbf_obstacle_{name}", "type": "box",
                "pos": " ".join(map(str, center)), "size": " ".join(map(str, half_size)),
                "contype": "2", "conaffinity": "1", "group": "0",
                "rgba": "0.78 0.26 0.19 0.85", "friction": "0.8 0.1 0.1"})
        self.xml_string = ET.tostring(xml, encoding="unicode")
        self.model = mujoco.MjModel.from_xml_string(self.xml_string)
        self.data = mujoco.MjData(self.model)
        joint_ids = [mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, f"joint{i}") for i in range(1, 7)]
        self.qpos_indices = np.asarray([self.model.jnt_qposadr[j] for j in joint_ids], dtype=int)
        self.dof_indices = np.asarray([self.model.jnt_dofadr[j] for j in joint_ids], dtype=int)
        self.lower = self.model.jnt_range[joint_ids, 0].copy()
        self.upper = self.model.jnt_range[joint_ids, 1].copy()
        # URDF effort limits are model limits, not independently verified motor
        # continuous ratings.  The distal joints must not inherit a 30 Nm demo cap.
        self.torque_limits = np.array([27.0, 27.0, 27.0, 7.0, 7.0, 7.0])
        self.velocity_limits = np.array([1.0] * 6)
        self.kp = np.array([110.0, 120.0, 110.0, 55.0, 40.0, 30.0])
        self.kd = np.array([16.0, 16.0, 14.0, 8.0, 6.0, 5.0])
        self.proxy_geom_ids = np.asarray([mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, f"cbf_proxy_{box.body_name}") for box in self.proxy_boxes])
        self.obstacle_geom_ids = np.asarray([mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, f"cbf_obstacle_{item['name']}") for item in self.obstacles], dtype=int)
        self._body_groups = []
        for box in self.proxy_boxes:
            body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, box.body_name)
            self._body_groups.append((body_id, np.flatnonzero(self.point_body_names == box.body_name)))
        self._tcp_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "tcp")
        self.last_torques = np.zeros(6)
        self.last_step_min_clearance = math.inf
        self.last_step_max_contacts = 0
        self.last_step_peak_torque = np.zeros(6)
        self.last_step_torque_sumsq = np.zeros(6)
        self.last_step_substeps = 0
        self.geometry_metadata = {
            "proxy_type": "body-local visual-mesh AABB plus padding",
            "proxy_count": len(self.proxy_boxes),
            "surface_point_count": len(self.local_points),
            "requested_epsilon_m": self.epsilon,
            "coverage_radius_m": self.coverage_radius,
            "padding_m": self.proxy_padding,
            "fixed_gripper_slide_opening_m": 0.04,
            "environment_collision_check": "exact MuJoCo box-box signed distance",
            "self_collision_enforced": False,
            "floor_collision_enforced": False,
            "bounds": [{"body": box.body_name, "center": box.center.tolist(),
                        "half_size": box.half_size.tolist()} for box in self.proxy_boxes],
        }
        mujoco.mj_forward(self.model, self.data)

    def set_q(self, q) -> None:
        q = np.asarray(q, dtype=float)
        if q.shape != (6,) or not np.all(np.isfinite(q)):
            raise ValueError("q must contain six finite joint angles")
        self.data.qpos[self.qpos_indices] = q
        mujoco.mj_forward(self.model, self.data)

    def get_q(self) -> np.ndarray:
        return self.data.qpos[self.qpos_indices].copy()

    def get_qd(self) -> np.ndarray:
        return self.data.qvel[self.dof_indices].copy()

    def points(self, q=None) -> np.ndarray:
        """World-frame proxy-surface samples, without computing Jacobians."""
        if q is not None:
            self.set_q(q)
        points = np.empty_like(self.local_points)
        for body_id, indices in self._body_groups:
            points[indices] = self.data.xpos[body_id] + self.local_points[indices] @ self.data.xmat[body_id].reshape(3, 3).T
        return points

    def points_and_jacobians(self, q=None) -> tuple[np.ndarray, np.ndarray]:
        if q is not None:
            self.set_q(q)
        points = np.empty_like(self.local_points)
        jacobians = np.empty((len(points), 3, 6))
        jacp, jacr = np.zeros((3, self.model.nv)), np.zeros((3, self.model.nv))
        for body_id, indices in self._body_groups:
            offsets = self.local_points[indices] @ self.data.xmat[body_id].reshape(3, 3).T
            points[indices] = self.data.xpos[body_id] + offsets
            mujoco.mj_jacBody(self.model, self.data, jacp, jacr, body_id)
            rotational = np.cross(jacr[:, self.dof_indices].T[None, :, :], offsets[:, None, :]).transpose(0, 2, 1)
            jacobians[indices] = jacp[:, self.dof_indices][None, :, :] + rotational
        return points, jacobians

    def tcp(self, q=None) -> np.ndarray:
        if q is not None:
            self.set_q(q)
        return self.data.site_xpos[self._tcp_id].copy()

    def tcp_jacobian(self, q=None) -> np.ndarray:
        if q is not None:
            self.set_q(q)
        jacp, jacr = np.zeros((3, self.model.nv)), np.zeros((3, self.model.nv))
        mujoco.mj_jacSite(self.model, self.data, jacp, jacr, self._tcp_id)
        return jacp[:, self.dof_indices].copy()

    def clearance(self, q=None) -> float:
        """Minimum signed proxy--obstacle distance in m; negative = penetration.

        This checks the entire box geometry, independently of surface samples.
        It excludes self collision and the visual floor; infinity if no obstacles.
        """
        if q is not None:
            self.set_q(q)
        distance = math.inf
        for robot_geom in self.proxy_geom_ids:
            for obstacle_geom in self.obstacle_geom_ids:
                value = mujoco.mj_geomDistance(self.model, self.data, int(robot_geom), int(obstacle_geom), 100.0, None)
                distance = min(distance, float(value))
        return distance

    def environment_contacts(self) -> int:
        """Number of currently active robot--obstacle contacts."""
        robot, obstacles = set(self.proxy_geom_ids), set(self.obstacle_geom_ids)
        return sum((contact.geom1 in robot and contact.geom2 in obstacles) or (contact.geom2 in robot and contact.geom1 in obstacles) for contact in self.data.contact)

    def step(self, target_q, target_qd=None, substeps: int = 1) -> np.ndarray:
        """Advance saturated PD+bias dynamics and return last applied torques.

        Geometry is monitored after *every* physics substep.  Aggregate
        ``last_step_*`` diagnostics therefore retain contacts, negative
        clearances and torque peaks that could disappear before the next slower
        control tick. ``last_step_torque_sumsq`` is an unweighted sum of squared
        substep torques, for computing RMS with ``last_step_substeps``.  This is
        still discrete-time monitoring, not continuous collision detection.
        """
        target_q = np.asarray(target_q, dtype=float)
        target_qd = np.zeros(6) if target_qd is None else np.asarray(target_qd, dtype=float)
        if target_q.shape != (6,) or target_qd.shape != (6,) or not np.all(np.isfinite(target_q)) or not np.all(np.isfinite(target_qd)):
            raise ValueError("Targets must contain six finite values")
        if not isinstance(substeps, (int, np.integer)) or substeps < 1:
            raise ValueError("substeps must be a positive integer")
        mujoco.mj_forward(self.model, self.data)
        self.last_step_min_clearance = self.clearance()
        self.last_step_max_contacts = self.environment_contacts()
        self.last_step_peak_torque = np.zeros(6)
        self.last_step_torque_sumsq = np.zeros(6)
        self.last_step_substeps = int(substeps)
        for _ in range(substeps):
            torque = self.kp * (target_q - self.get_q()) + self.kd * (target_qd - self.get_qd()) + self.data.qfrc_bias[self.dof_indices]
            self.last_torques = np.clip(torque, -self.torque_limits, self.torque_limits)
            self.last_step_peak_torque = np.maximum(self.last_step_peak_torque, np.abs(self.last_torques))
            self.last_step_torque_sumsq += self.last_torques**2
            self.data.qfrc_applied[:] = 0.0
            self.data.qfrc_applied[self.dof_indices] = self.last_torques
            mujoco.mj_step(self.model, self.data)
            # mj_step leaves position-derived arrays from the start of that
            # integration step. Refresh them before querying its final state.
            mujoco.mj_forward(self.model, self.data)
            self.last_step_min_clearance = min(self.last_step_min_clearance, self.clearance())
            self.last_step_max_contacts = max(self.last_step_max_contacts, self.environment_contacts())
        return self.last_torques.copy()
