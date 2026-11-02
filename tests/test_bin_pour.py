"""Tests for the bin pour: grasp the bin's near rim top-down, lift it, turn the base a
quarter turn so it hangs beside where it stood, tip it over about the wall so whatever
is inside falls out, then put it back and let go (the real counterpart of the
simulator's ``BinPourController``)."""

import math

import numpy as np
import pytest

from prpl_tidybot.bin_localizer import BinFit
from prpl_tidybot.bin_pour import (
    BinPourGeometry,
    bin_points_in_hand,
    joint_limits,
    plan_bin_pour,
    pour_point,
)
from prpl_tidybot.top_down_pick import TopDownPickGeometry, forward_kinematics

GEOM = TopDownPickGeometry()
POUR = BinPourGeometry()
FLOOR = (-1.5, 1.7, -1.5, 1.5)
# The blue Rubbermaid bin on the robot's side of the barrier.
BIN = BinFit(
    0.2,
    -0.4,
    0.0,
    0.365,
    0.26,
    0.384,
    0.005,
    base_length=0.29,
    base_width=0.19,
)
ROBOT = (-0.6, -0.9, 0.0)


@pytest.fixture(scope="module", name="plan")
def _plan():
    """One plan for the bin above, shared by the tests (IK is slow)."""
    return plan_bin_pour(BIN, ROBOT, GEOM, POUR, FLOOR)


def test_the_robot_stands_at_the_standoff_facing_the_bin(plan):
    """The base parks `standoff` from the bin's centre, facing it along a face."""
    x, y, theta = plan.stand
    assert math.hypot(BIN.x - x, BIN.y - y) == pytest.approx(plan.standoff, abs=1e-6)
    assert plan.standoff == POUR.standoff  # the blue bin solves at the preferred one
    facing = math.atan2(BIN.y - y, BIN.x - x)
    assert abs((theta - facing + math.pi) % (2 * math.pi) - math.pi) < 1e-6
    assert abs((theta - BIN.yaw + math.pi / 4) % (math.pi / 2) - math.pi / 4) < 1e-6


def test_the_stages_mirror_the_simulators_pour(plan):
    """Approach, grasp, lift, turn, tip, empty, then the same back and let go."""
    assert [s.name for s in plan.stages] == [
        "hover",
        "rim",
        "close",
        "lift",
        "turn",
        "tip",
        "empty",
        "untip",
        "unturn",
        "lower",
        "open",
        "withdraw",
    ]


def test_the_fingers_straddle_the_near_wall_at_the_rim(plan):
    """At the rim the tool points down, its pinch point just inside the near wall's
    top and the fingers closing across the wall (along the robot's heading)."""
    rim = next(s for s in plan.stages if s.name == "rim")
    pos, approach, closing = forward_kinematics(rim.arm[-1], GEOM)
    span = BIN.width if plan.face % 2 else BIN.length
    near_wall = plan.standoff - span / 2 + POUR.rim_inset
    assert pos[0] + GEOM.mounting_offset == pytest.approx(near_wall, abs=0.005)
    assert pos[1] == pytest.approx(0.0, abs=0.005)
    assert pos[2] == pytest.approx(
        GEOM.floor_z + BIN.height - POUR.below_rim, abs=0.005
    )
    assert approach[2] < -0.999
    assert abs(closing[0]) > 0.999


def test_the_tipped_bin_is_upside_down_and_away_from_the_robot(plan):
    """After the tip the bin's bottom is above its rim and in front of the gripper,
    so the opening faces down and the bin never swings back over the base."""
    tip = next(s for s in plan.stages if s.name == "tip")
    span = BIN.width if plan.face % 2 else BIN.length
    bottom, rim_centre = bin_points_in_hand(tip.arm[-1], GEOM, POUR, span, BIN)
    # 120 degrees of tip: the bottom ~0.18 m above the rim, the opening facing down.
    assert bottom[2] > rim_centre[2] + 0.1
    assert min(bottom[0], rim_centre[0]) > 0.0


def test_the_lift_raises_the_bin_clear_of_the_floor(plan):
    """Lifted, the pinch point is at `tip_height`: the blue bin's base 0.48 m up, the
    simulator's lift."""
    lift = next(s for s in plan.stages if s.name == "lift")
    span = BIN.width if plan.face % 2 else BIN.length
    bottom, _ = bin_points_in_hand(lift.arm[-1], GEOM, POUR, span, BIN)
    assert bottom[2] - GEOM.floor_z == pytest.approx(0.48, abs=0.01)


def test_no_arm_move_jumps_ik_branches(plan):
    """Every joint changes by under 0.5 rad between consecutive waypoints, so the
    streamed moves cannot swing the arm through an unplanned configuration."""
    previous = None
    for stage in plan.stages:
        for q in stage.arm:
            if previous is not None:
                step = np.abs((np.array(q) - previous + np.pi) % (2 * np.pi) - np.pi)
                assert step.max() < 0.5, stage.name
            previous = np.array(q)


def test_the_turn_is_a_quarter_turn_of_the_base_and_back(plan):
    """The base turns a quarter turn with the bin held, then turns back."""
    turn = next(s for s in plan.stages if s.name == "turn")
    unturn = next(s for s in plan.stages if s.name == "unturn")
    assert abs(abs(turn.base[2] - plan.stand[2]) - math.pi / 2) < 1e-6
    assert unturn.base == pytest.approx(plan.stand)


def test_the_gripper_is_closed_from_close_to_open(plan):
    """Closed while the bin is held, open on the way in and out."""
    held = [s.gripper for s in plan.stages]
    names = [s.name for s in plan.stages]
    assert held[: names.index("close")] == [0.0] * names.index("close")
    assert all(g == 1.0 for g in held[names.index("close") : names.index("open")])
    assert held[names.index("open") :] == [0.0] * (len(names) - names.index("open"))


def test_a_bin_with_no_drivable_standoff_raises():
    """A bin whose every standoff leaves the drivable box (across the barrier, in a
    corner) has no pour."""
    corner = BinFit(1.68, 1.48, 0.0, 0.365, 0.26, 0.384)
    with pytest.raises(ValueError):
        plan_bin_pour(corner, ROBOT, GEOM, POUR, (-1.5, 1.7, -1.5, 0.3))


def test_every_waypoint_is_within_the_arms_joint_limits(plan):
    """The IK solver ignores the joint limits; the plan must not (2026-11-02: the
    first plan sent joint 6 to 156 degrees against a 119.7 limit)."""
    lo, hi = joint_limits(GEOM)
    for stage in plan.stages:
        for q in stage.arm:
            assert np.all(np.array(q) >= lo) and np.all(np.array(q) <= hi), stage.name


def test_the_tip_is_far_enough_to_empty_the_bin(plan):
    """Past 90 degrees the opening faces down; the default tips 120."""
    assert POUR.tilt >= math.radians(110)
    assert len(next(s for s in plan.stages if s.name == "tip").arm) >= 12


def test_a_stand_the_workspace_rejects_is_not_chosen():
    """With a validity check (the barrier, the bin), only stands it accepts are used,
    for the stand and the turned stand alike."""
    plan = plan_bin_pour(
        BIN, ROBOT, GEOM, POUR, FLOOR, is_valid=lambda pose: pose[0] < BIN.x
    )
    assert plan.stand[0] < BIN.x
    with pytest.raises(ValueError, match="workspace"):
        plan_bin_pour(BIN, ROBOT, GEOM, POUR, FLOOR, is_valid=lambda pose: False)


def test_the_pour_point_is_beside_the_bin_and_off_the_base(plan):
    """The contents fall under the tipped rim: a quarter turn round the base from the
    bin's spot, clear of the base's own footprint."""
    x, y = pour_point(plan, BIN, GEOM, POUR)
    sx, sy, _ = plan.stand
    assert math.hypot(x - BIN.x, y - BIN.y) > 0.5
    assert math.hypot(x - sx, y - sy) > 0.35
