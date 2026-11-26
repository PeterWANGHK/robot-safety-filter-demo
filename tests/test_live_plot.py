"""Torque history semantics, without importing Qt or starting ROS nodes."""

import math

import pytest

from cbf_demo.live_plot import TorqueHistory


def test_signed_efforts_are_matched_by_joint_name():
    history = TorqueHistory()
    now = history.started
    assert history.append(["joint6", "joint2", "joint1"], [-3.5, -1.25, 4.0], 100.0, now)
    assert history.channels[0][-1] == (0.0, 4.0)
    assert history.channels[1][-1] == (0.0, -1.25)
    assert history.channels[5][-1] == (0.0, -3.5)
    assert not history.channels[2]
    assert history.last_received[2] is None


def test_missing_or_nonfinite_efforts_produce_gaps_not_zeroes():
    history = TorqueHistory()
    now = history.started
    history.append(["joint1", "joint2", "joint3"], [2.0, -2.0, 1.0], 100.0, now)
    history.append(["joint1", "joint2", "joint3"], [3.0, math.nan, math.inf], 101.0, now + 1)
    assert history.channels[0][-1] == (1.0, 3.0)
    assert math.isnan(history.channels[1][-1][1])
    assert math.isnan(history.channels[2][-1][1])
    assert history.last_received[1] == now
    assert history.last_received[2] == now
    history.append(["joint1"], [4.0], 102.0, now + 2)
    assert math.isnan(history.channels[1][-1][1])
    assert not history.channels[3]


def test_rolling_window_prunes_old_measurements():
    history = TorqueHistory(window_seconds=2)
    now = history.started
    for index in range(5):
        history.append(["joint1"], [float(index)], 100.0 + index, now + index)
    assert list(history.channels[0]) == [(2.0, 2.0), (3.0, 3.0), (4.0, 4.0)]
    assert history.latest_time == 4.0


def test_header_clock_restart_clears_previous_epoch():
    history = TorqueHistory()
    now = history.started
    history.append(["joint1", "joint2"], [1.0, 2.0], 100.0, now)
    history.append(["joint1"], [3.0], 101.0, now + 1)
    history.append(["joint1"], [-4.0], 1.0, now + 2)
    assert history.clock_resets == 1
    assert history.clock_kind == "message header"
    assert list(history.channels[0]) == [(2.0, -4.0)]
    assert not history.channels[1]
    assert history.last_received[1] is None


def test_missing_header_uses_monotonic_fallback():
    history = TorqueHistory(window_seconds=2)
    now = history.started
    history.append(["joint1"], [-1.0], None, now + 0.5)
    history.append(["joint1"], [-2.0], 0.0, now + 1.5)
    assert history.clock_kind == "monotonic fallback"
    assert list(history.channels[0]) == [(0.5, -1.0), (1.5, -2.0)]
    history.append(["joint1"], [-3.0], None, now + 4.0)
    assert list(history.channels[0]) == [(4.0, -3.0)]


def test_unrelated_or_empty_messages_do_not_fabricate_fresh_data():
    history = TorqueHistory()
    now = history.started
    history.append(["joint1"], [1.0], 100.0, now)
    assert not history.append(["finger_left"], [0.0], 101.0, now + 1)
    assert not history.append(["joint1"], [], 102.0, now + 2)
    assert not history.append(["joint1"], [math.nan], 103.0, now + 3)
    assert history.message_count == 1
    assert history.last_received[0] == now
    assert list(history.channels[0]) == [(0.0, 1.0)]


@pytest.mark.parametrize("duration", [0, -1, math.nan, math.inf])
def test_invalid_history_duration_is_rejected(duration):
    with pytest.raises(ValueError):
        TorqueHistory(duration)
