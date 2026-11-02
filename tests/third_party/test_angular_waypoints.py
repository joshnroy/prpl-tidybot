"""Tests for the onboard joint trajectory's timing and wrap guard (pure helpers; the
Kortex call itself runs only on the robot)."""

import pytest

from prpl_tidybot.third_party.kinova import angular_waypoint_durations, crosses_wrap


def test_each_segment_takes_its_largest_joint_change_at_the_speed():
    """A segment's duration is its largest joint change over the speed, at least the
    minimum."""
    start = [0.0] * 7
    path = [[10.0] + [0.0] * 6, [10.0, 40.0] + [0.0] * 5, [10.0, 40.0] + [0.0] * 5]
    durations = angular_waypoint_durations(start, path, speed_deg_s=20.0, minimum_s=0.3)
    assert durations == pytest.approx([0.5, 2.0, 0.3])


def test_the_short_way_round_is_measured():
    """359 -> 1 degrees is a 2-degree change, not 358."""
    start = [359.0] + [0.0] * 6
    durations = angular_waypoint_durations(
        start, [[1.0] + [0.0] * 6], speed_deg_s=2.0, minimum_s=0.0
    )
    assert durations == pytest.approx([1.0])


def test_a_path_crossing_zero_on_a_joint_is_flagged():
    """Angles go to the arm modulo 360: a joint passing 0/360 between waypoints could be
    driven the long way, so such a path is not sent as one trajectory."""
    assert crosses_wrap([[350.0] + [0.0] * 6, [10.0] + [0.0] * 6])
    assert not crosses_wrap([[10.0] + [0.0] * 6, [30.0] + [0.0] * 6])
