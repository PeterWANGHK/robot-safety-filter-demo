"""Read-only Qt torque monitor for the isolated CBF simulation, ROS domain 63.

The six signed effort channels are matched by joint name, never array order.
Missing/nonfinite efforts are left blank; this plot does not infer torque or
contact force and has no command publisher, action client or hardware interface.
"""

from __future__ import annotations

import argparse
from collections import deque
import math
import os
from pathlib import Path
import sys
import time


JOINT_NAMES = tuple(f"joint{i}" for i in range(1, 7))
MODEL_CAPS = (27.0, 27.0, 27.0, 7.0, 7.0, 7.0)


class TorqueHistory:
    """Small message-clock cache usable without Qt or a ROS installation."""

    def __init__(self, window_seconds: float = 30.0):
        self.window_seconds = float(window_seconds)
        if not math.isfinite(self.window_seconds) or self.window_seconds <= 0:
            raise ValueError("window_seconds must be positive and finite")
        self.channels = [deque() for _ in JOINT_NAMES]
        self.last_received = [None] * len(JOINT_NAMES)
        self.started = time.monotonic()
        self.latest_time = 0.0
        self.header_origin = None
        self.clock_kind = "waiting"
        self.message_count = 0
        self.clock_resets = 0

    def append(self, names, effort, stamp_seconds: float | None = None,
               received_at: float | None = None) -> bool:
        now = time.monotonic() if received_at is None else float(received_at)
        indices = {name: index for index, name in enumerate(names)}
        values = {}
        for channel, name in enumerate(JOINT_NAMES):
            index = indices.get(name)
            if index is not None and index < len(effort):
                value = float(effort[index])
                if math.isfinite(value):
                    values[channel] = value
        if not values:
            return False
        elapsed = now - self.started
        if stamp_seconds is not None and math.isfinite(stamp_seconds) and stamp_seconds > 0:
            if self.header_origin is None:
                self.header_origin = stamp_seconds - elapsed
            timestamp = stamp_seconds - self.header_origin
            self.clock_kind = "message header"
            if timestamp < self.latest_time - 0.05:
                # ROS/simulation clock restarted. Do not join distinct epochs.
                self.channels = [deque() for _ in JOINT_NAMES]
                self.last_received = [None] * len(JOINT_NAMES)
                self.header_origin = stamp_seconds - elapsed
                timestamp = elapsed
                self.clock_resets += 1
        else:
            timestamp = elapsed
            self.clock_kind = "monotonic fallback"
            if timestamp < self.latest_time - 0.05:
                self.channels = [deque() for _ in JOINT_NAMES]
                self.last_received = [None] * len(JOINT_NAMES)
                self.header_origin = None
                self.clock_resets += 1
        self.latest_time = timestamp
        oldest = timestamp - self.window_seconds
        for channel, history in enumerate(self.channels):
            # A missing value creates a gap, rather than an invented zero or a
            # stale flat line that appears to be a new measurement.
            if channel in values:
                history.append((timestamp, values[channel]))
                self.last_received[channel] = now
            elif history:
                history.append((timestamp, math.nan))
            while history and history[0][0] < oldest:
                history.popleft()
        self.message_count += 1
        return True


class LiveTorquePlot:
    def __init__(self, window_seconds=30.0, output=None, seconds=None):
        if os.environ.get("ROS_DOMAIN_ID") != "63":
            raise RuntimeError("Torque plotting requires ROS_DOMAIN_ID=63; use ./rebot cbf plot.")
        self.history = TorqueHistory(window_seconds)
        self.output = Path(output).expanduser() if output else None
        self.seconds = seconds
        self._closed = False
        self._context = self._node = self._executor = self._timer = None
        self.figure = None
        self.saved_paths = []

        import matplotlib
        matplotlib.use("QtAgg", force=True)
        from matplotlib import pyplot as plt
        from matplotlib.backends.qt_compat import QtCore
        import rclpy
        from rclpy.context import Context
        from rclpy.executors import SingleThreadedExecutor
        from rclpy.qos import qos_profile_sensor_data
        from sensor_msgs.msg import JointState
        from .plot_style import scientific_style, save_figure

        self._plt, self._save_figure = plt, save_figure
        try:
            self._context = Context()
            rclpy.init(args=[], context=self._context)
            self._node = rclpy.create_node(
                "rebot_cbf_signed_torque_plot", context=self._context,
                enable_rosout=False, start_parameter_services=False,
            )
            self._executor = SingleThreadedExecutor(context=self._context)
            self._executor.add_node(self._node)
            self._subscription = self._node.create_subscription(
                JointState, "/joint_states", self._receive, qos_profile_sensor_data,
            )
            with scientific_style():
                self.figure, axes = plt.subplots(3, 2, figsize=(11.4, 8.2), sharex=True,
                                                 layout="constrained")
                self.axes = axes.ravel()
                self.lines, self.readouts = [], []
                colors = plt.rcParams["axes.prop_cycle"].by_key().get("color", ["C0"])
                for channel, (axis, cap) in enumerate(zip(self.axes, MODEL_CAPS)):
                    color = colors[channel % len(colors)]
                    line, = axis.plot([], [], color=color, linewidth=1.6)
                    self.lines.append(line)
                    axis.axhline(0, color="0.55", linewidth=0.65)
                    axis.axhline(cap, color="0.45", linestyle="--", linewidth=0.9)
                    axis.axhline(-cap, color="0.45", linestyle="--", linewidth=0.9)
                    axis.set_title(f"Joint {channel + 1}  |  model cap ±{cap:g} N m", loc="left")
                    axis.set_ylabel("Signed torque (N m)")
                    axis.set_ylim(-1.18 * cap, 1.18 * cap)
                    axis.set_xlim(0, self.history.window_seconds)
                    axis.grid(True, alpha=0.23)
                    self.readouts.append(axis.text(
                        0.98, 0.96, "waiting", transform=axis.transAxes,
                        ha="right", va="top", color="0.4", fontsize=9,
                    ))
                for axis in self.axes[-2:]:
                    axis.set_xlabel("Time (s)")
                self.title = self.figure.suptitle("B601 simulated joint torques — waiting for /joint_states")
                self.figure.supxlabel(
                    "Applied simulated actuator effort; dashed caps are model limits, not certified motor ratings.\n"
                    "ROS domain 63 • read-only monitor • effort is not measured contact force",
                    fontsize=9,
                )
            self.figure.canvas.manager.set_window_title("B601 CBF · six signed joint torques")
            self.figure.canvas.mpl_connect("close_event", self.close)
            self._timer = QtCore.QTimer(self.figure.canvas)
            self._timer.setInterval(50)
            self._timer.timeout.connect(self._refresh)
            self._timer.start()
        except Exception:
            self.close()
            raise

    def _receive(self, message):
        stamp = message.header.stamp
        seconds = float(stamp.sec) + float(stamp.nanosec) * 1e-9
        self.history.append(message.name, message.effort, seconds)

    def _refresh(self):
        if self._closed:
            return
        # Bounded nonblocking batch keeps the Qt event loop responsive while
        # consuming a burst of state messages. No background global ROS spin.
        for _ in range(8):
            self._executor.spin_once(timeout_sec=0.0)
        now = time.monotonic()
        newest_age = None
        for channel, (axis, line, readout, cap) in enumerate(
                zip(self.axes, self.lines, self.readouts, MODEL_CAPS)):
            history = self.history.channels[channel]
            received = self.history.last_received[channel]
            if received is None:
                line.set_data([], [])
                readout.set_text("waiting for effort")
                continue
            age = now - received
            newest_age = age if newest_age is None else min(newest_age, age)
            xs, ys = zip(*history) if history else ([], [])
            line.set_data(xs, ys)
            finite = [value for value in ys if math.isfinite(value)]
            limit = max(cap * 1.18, max((abs(value) * 1.1 for value in finite), default=0.0))
            axis.set_ylim(-limit, limit)
            latest = finite[-1] if finite else None
            label = f"{latest:+.2f} N m" if latest is not None else "no finite effort"
            if age > 1.0:
                label += f"\nstale ({age:.1f} s)"
            elif ys and not math.isfinite(ys[-1]):
                label += "\nmissing in latest message"
            readout.set_text(label)
            readout.set_color("#A44322" if age > 1.0 else "0.25")
        right = max(self.history.window_seconds, self.history.latest_time)
        self.axes[0].set_xlim(right - self.history.window_seconds, right)
        if newest_age is None:
            status = "waiting for /joint_states with named effort channels"
        elif newest_age > 1.0:
            status = f"stale — last effort {newest_age:.1f} s ago"
        else:
            status = f"live • {self.history.clock_kind} • {self.history.message_count} messages"
        if self.history.clock_resets:
            status += f" • clock resets: {self.history.clock_resets}"
        self.title.set_text("B601 simulated joint torques — " + status)
        self.figure.canvas.draw_idle()
        if self.seconds is not None and now - self.history.started >= self.seconds:
            self._plt.close(self.figure)

    def show(self):
        try:
            self._plt.show(block=True)
        finally:
            self.close()

    def close(self, event=None):
        if self._closed:
            return
        self._closed = True
        if self._timer is not None:
            self._timer.stop()
        try:
            if self.output is not None and self.figure is not None:
                self.saved_paths = self._save_figure(self.figure, self.output)
                print(f"Saved signed torque plot: {self.output}", flush=True)
        finally:
            if self._executor is not None:
                self._executor.shutdown(timeout_sec=0.1)
            if self._node is not None:
                self._node.destroy_node()
            if self._context is not None and self._context.ok():
                self._context.shutdown()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--window-seconds", type=float, default=30.0,
                        help="history duration in seconds (default: 30)")
    parser.add_argument("--output", help="optional output stem; save PNG and PDF when the window closes")
    parser.add_argument("--seconds", type=float, help="automatically close after this many wall-clock seconds")
    args = parser.parse_args(argv)
    if (not math.isfinite(args.window_seconds) or args.window_seconds <= 0
            or (args.seconds is not None and (not math.isfinite(args.seconds) or args.seconds <= 0))):
        parser.error("durations must be positive and finite")
    try:
        LiveTorquePlot(args.window_seconds, args.output, args.seconds).show()
    except KeyboardInterrupt:
        return 130
    except (ImportError, RuntimeError) as error:
        print(f"Torque plot unavailable: {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
