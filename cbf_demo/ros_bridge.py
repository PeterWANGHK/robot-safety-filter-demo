"""Optional, one-way ROS 2 visualization of the standalone CBF simulation.

``bridge = ROSBridge(obstacles, path=tcp_xyz_path, field=poisson_field)``
``bridge.publish(q, qd, applied_tau, tcp, surface_points, h_values, diagnostics)``
``bridge.close()``

Obstacles are dictionaries with ``center`` and ``half_size`` XYZ vectors in
metres. ``path`` is optional Nx3 TCP positions, not joint configurations.
All coordinates are in ``base_link``, identical to this fixed-base model's
world. The bridge only publishes actual simulation states, diagnostics and
RViz markers; it has no subscriptions, command publishers, action clients or
hardware interface. Domain 63 is required to isolate its /joint_states topic.
ROS imports are deferred until construction, so importing this module does
not require a ROS environment and offline experiments remain independent.
"""

from __future__ import annotations

from collections import deque
from dataclasses import asdict, is_dataclass
import json
import math
import os
from typing import Any

import numpy as np


def _json_safe(value: Any) -> Any:
    """Serialize diagnostics without emitting nonstandard JSON NaN values."""
    if is_dataclass(value):
        return _json_safe(asdict(value))
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return number if math.isfinite(number) else None
    if value is None or isinstance(value, str):
        return value
    return str(value)


class ROSBridge:
    """Publish-only visualization bridge; call close in a finally block."""

    def __init__(self, obstacles, path=None, field=None):
        if os.environ.get("ROS_DOMAIN_ID") != "63":
            raise RuntimeError("CBF visualization requires isolated ROS_DOMAIN_ID=63; use the CBF wrapper.")
        try:
            import rclpy
            from rclpy.context import Context
            from rclpy.executors import SingleThreadedExecutor
            from rclpy.qos import QoSProfile, DurabilityPolicy, ReliabilityPolicy
            from geometry_msgs.msg import Point
            from sensor_msgs.msg import JointState
            from std_msgs.msg import ColorRGBA, String
            from visualization_msgs.msg import Marker, MarkerArray
        except ImportError as exc:
            raise RuntimeError(
                "ROS visualization unavailable. Source /opt/ros/jazzy/setup.bash "
                "and the dedicated reBot physics workspace, or run without --ros."
            ) from exc

        self._rclpy = rclpy
        self._Point, self._Color = Point, ColorRGBA
        self._JointState, self._String = JointState, String
        self._Marker, self._MarkerArray = Marker, MarkerArray
        self._context = Context()
        rclpy.init(args=[], context=self._context)
        self._node = rclpy.create_node("rebot_poisson_cbf", context=self._context)
        self._executor = SingleThreadedExecutor(context=self._context)
        self._executor.add_node(self._node)
        self._joint_publisher = self._node.create_publisher(JointState, "/joint_states", 5)
        self._diagnostics_publisher = self._node.create_publisher(String, "/rebot_cbf/diagnostics", 5)
        self._markers_publisher = self._node.create_publisher(MarkerArray, "/rebot_cbf/markers", 5)
        static_qos = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL,
                                reliability=ReliabilityPolicy.RELIABLE)
        self._field_publisher = self._node.create_publisher(MarkerArray, "/rebot_cbf/field", static_qos)
        self.obstacles = []
        for obstacle in obstacles:
            center = np.asarray(obstacle["center"], dtype=float)
            half_size = np.asarray(obstacle["half_size"], dtype=float)
            if (center.shape != (3,) or half_size.shape != (3,)
                    or not np.all(np.isfinite(center)) or not np.all(np.isfinite(half_size))
                    or np.any(half_size <= 0)):
                self.close()
                raise ValueError("obstacles must have finite center and positive half_size XYZ vectors")
            self.obstacles.append({"center": center.copy(), "half_size": half_size.copy()})
        self._trace = deque(maxlen=2000)
        self._path = None
        self.set_path(path)
        self._closed = False
        if field is not None:
            try:
                self._publish_static_field(field)
            except Exception:
                self.close()
                raise

    @property
    def path(self):
        return None if self._path is None else self._path.copy()

    @path.setter
    def path(self, value):
        self.set_path(value)

    def set_path(self, path):
        """Set an optional planned TCP path in world/base_link XYZ metres."""
        if path is None:
            self._path = None
            return
        coordinates = np.asarray(path, dtype=float)
        if coordinates.ndim != 2 or coordinates.shape[1] != 3 or not np.all(np.isfinite(coordinates)):
            raise ValueError("path must be a finite (N,3) TCP XYZ array, not a joint-space path")
        self._path = coordinates.copy()

    def _point(self, xyz):
        return self._Point(x=float(xyz[0]), y=float(xyz[1]), z=float(xyz[2]))

    def _marker(self, namespace, marker_id, kind, stamp):
        marker = self._Marker()
        marker.header.frame_id = "base_link"
        marker.header.stamp = stamp
        marker.ns, marker.id, marker.type = namespace, marker_id, kind
        marker.action = self._Marker.ADD
        marker.pose.orientation.w = 1.0
        return marker

    def _publish_static_field(self, field):
        """Extract once and latch translucent 3-D scalar-field shells.

        This topic is separate from frequently updated robot sample markers;
        late RViz subscribers receive its transient-local retained message.
        """
        from .field_visualization import build_field_meshes, DEFAULT_DISPLAY_BOUNDS
        meshes = build_field_meshes(field, bounds=DEFAULT_DISPLAY_BOUNDS, step_size=2)
        stamp = self._node.get_clock().now().to_msg()
        array = self._MarkerArray()
        for index, mesh in enumerate(meshes):
            # Independent namespaces let RViz toggle each scalar level.
            namespace = "poisson_h_" + f"{mesh['level']:g}".replace(".", "_") + "_m2"
            marker = self._marker(namespace, index, self._Marker.TRIANGLE_LIST, stamp)
            marker.scale.x = marker.scale.y = marker.scale.z = 1.0
            rgb = [int(mesh["color"][offset:offset+2], 16)/255 for offset in (1, 3, 5)]
            marker.color = self._Color(r=rgb[0], g=rgb[1], b=rgb[2], a=0.16)
            marker.points = [self._point(point) for point in mesh["vertices"][mesh["faces"]].reshape(-1, 3)]
            array.markers.append(marker)
        legend = self._marker("poisson_h_legend", 0, self._Marker.TEXT_VIEW_FACING, stamp)
        legend.pose.position = self._point([0.05, -0.52, 0.93])
        legend.scale.z = 0.027
        legend.color = self._Color(r=0.95, g=0.95, b=0.95, a=1.0)
        legend.text = "Poisson h (m^2), NOT clearance\nOrange: low h; blue: high h\nLevels: " + ", ".join(f"{mesh['level']:g}" for mesh in meshes) + "\nDisplay crop; control field unchanged"
        array.markers.append(legend)
        # Keep the publisher alive for transient-local delivery. The message
        # is published once, not rebuilt/reserialized at the control rate.
        self._field_publisher.publish(array)
        self._executor.spin_once(timeout_sec=0.0)

    def publish(self, q, qd, tau, tcp, points, values, diagnostics):
        """Publish actual simulated joint states and geometric diagnostics.

        q/qd/tau have six entries (rad, rad/s, N m). The fixed visual gripper
        opening is 0.04 m total. Surface points and h values must correspond;
        callers may subsample them to bound visualization bandwidth.
        """
        if self._closed:
            raise RuntimeError("ROS visualization bridge is closed")
        q, qd, tau = [np.asarray(vector, dtype=float) for vector in (q, qd, tau)]
        tcp, points, values = [np.asarray(vector, dtype=float) for vector in (tcp, points, values)]
        if any(vector.shape != (6,) or not np.all(np.isfinite(vector)) for vector in (q, qd, tau)):
            raise ValueError("q, qd and applied tau must each be finite six-element vectors")
        if tcp.shape != (3,) or not np.all(np.isfinite(tcp)):
            raise ValueError("tcp must be a finite XYZ vector")
        if points.ndim != 2 or points.shape[1] != 3 or values.shape != (len(points),):
            raise ValueError("surface points must have shape (N,3) and values shape (N,)")
        if not np.all(np.isfinite(points)):
            raise ValueError("surface points must be finite")

        stamp = self._node.get_clock().now().to_msg()
        joints = self._JointState()
        joints.header.stamp, joints.header.frame_id = stamp, "base_link"
        joints.name = [f"joint{i}" for i in range(1, 7)] + ["finger_left", "finger_right"]
        joints.position = q.tolist() + [0.02, -0.02]
        joints.velocity = qd.tolist() + [0.0, 0.0]
        joints.effort = tau.tolist() + [0.0, 0.0]
        self._joint_publisher.publish(joints)

        message = self._String()
        message.data = json.dumps(_json_safe(diagnostics), allow_nan=False, separators=(",", ":"))
        self._diagnostics_publisher.publish(message)

        array = self._MarkerArray()
        for index, obstacle in enumerate(self.obstacles):
            marker = self._marker("physical_obstacles", index, self._Marker.CUBE, stamp)
            marker.pose.position = self._point(obstacle["center"])
            marker.scale.x, marker.scale.y, marker.scale.z = (2 * obstacle["half_size"]).tolist()
            marker.color = self._Color(r=0.8, g=0.35, b=0.2, a=0.55)
            array.markers.append(marker)

        samples = self._marker("surface_h_m2", 0, self._Marker.POINTS, stamp)
        samples.scale.x = samples.scale.y = 0.008
        samples.points = [self._point(point) for point in points]
        from .field_visualization import field_rgb
        colors = field_rgb(values)
        for value, rgb in zip(values, colors):
            if not np.isfinite(value) or value <= 0:
                color = self._Color(r=1.0, g=0.0, b=0.0, a=1.0)
            else:
                # Same fixed palette as the 3-D field; h is not distance.
                color = self._Color(r=float(rgb[0]), g=float(rgb[1]), b=float(rgb[2]), a=1.0)
            samples.colors.append(color)
        array.markers.append(samples)

        self._trace.append(tcp.copy())
        trace = self._marker("actual_tcp_trace", 0, self._Marker.LINE_STRIP, stamp)
        trace.scale.x = 0.004
        trace.color = self._Color(r=1.0, g=0.85, b=0.1, a=1.0)
        trace.points = [self._point(point) for point in self._trace]
        array.markers.append(trace)

        planned = self._marker("planned_tcp_path", 0, self._Marker.LINE_STRIP, stamp)
        planned.scale.x = 0.003
        planned.color = self._Color(r=0.15, g=0.7, b=1.0, a=0.9)
        if self._path is None or len(self._path) < 2:
            planned.action = self._Marker.DELETE
        else:
            planned.points = [self._point(point) for point in self._path]
        array.markers.append(planned)
        self._markers_publisher.publish(array)
        self._executor.spin_once(timeout_sec=0.0)

    def close(self):
        """Release only this bridge's private ROS context; idempotent."""
        if getattr(self, "_closed", False):
            return
        self._closed = True
        if hasattr(self, "_executor"):
            self._executor.shutdown(timeout_sec=0.1)
        if hasattr(self, "_node"):
            self._node.destroy_node()
        if hasattr(self, "_context") and self._context.ok():
            self._context.shutdown()
