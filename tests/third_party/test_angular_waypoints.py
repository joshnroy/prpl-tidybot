"""Tests for the onboard joint trajectory's timing and wrap guard (pure helpers; the
Kortex call itself runs only on the robot)."""

import pytest

from prpl_tidybot.third_party.kinova import angular_waypoint_durations, crosses_wrap


def test_each_segment_is_timed_within_the_arms_speed_and_acceleration():
    """The arm interpolates between waypoints with a peak speed well above the
    average and accelerates within each segment; the 2026-11-02 lift was rejected
    (35 deg/s against 30, 200 deg/s^2 against 80). Each segment's average speed stays
    under half the speed limit and its change over the squared duration under a sixth
    of the acceleration limit."""
    start = [0.0] * 7
    path = [[2.0] + [0.0] * 6, [2.0, 40.0] + [0.0] * 5]
    durations = angular_waypoint_durations(
        start, path, speed_deg_s=30.0, accel_deg_s2=80.0, minimum_s=0.3
    )
    for change, duration in zip((2.0, 40.0), durations):
        assert change / duration <= 0.5 * 30.0 + 1e-9
        assert change / duration**2 <= 80.0 / 6.0 + 1e-9
    assert durations[0] >= 0.3


def test_the_short_way_round_is_measured():
    """359 -> 1 degrees is a 2-degree change, not 358."""
    start = [359.0] + [0.0] * 6
    durations = angular_waypoint_durations(
        start, [[1.0] + [0.0] * 6], speed_deg_s=4.0, accel_deg_s2=1e9, minimum_s=0.0
    )
    assert durations == pytest.approx([1.0])


def test_a_path_crossing_zero_on_a_joint_is_flagged():
    """Angles go to the arm modulo 360: a joint passing 0/360 between waypoints could be
    driven the long way, so such a path is not sent as one trajectory."""
    assert crosses_wrap([[350.0] + [0.0] * 6, [10.0] + [0.0] * 6])
    assert not crosses_wrap([[10.0] + [0.0] * 6, [30.0] + [0.0] * 6])
