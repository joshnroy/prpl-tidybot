"""The pick's drive to its standoff through one waypoint, and its other standoffs, when
the straight drive is refused (2026-11-08 15:32: a practice toss landed on open floor
past the close bin, and the straight drive to the one standoff tried clipped the bin's
corner)."""

# pylint: disable=protected-access
import math

import pytest
from spatialmath import SE2

from prpl_tidybot import robot_skills
from prpl_tidybot.aimed_toss import (
    Barrier,
    Workspace,
    _bin_corners,
    _rectangles_overlap,
    footprint,
)
from prpl_tidybot.bin_localizer import BinFit
from prpl_tidybot.robot_skills import pick_problem, pick_stand
from prpl_tidybot.top_down_pick import (
    TopDownPickGeometry,
    standoff_pose,
    standoff_poses,
)

FLOOR = (-1.5, 1.7, -1.5, 1.5)
GEOM = TopDownPickGeometry()
# The 15:32 observe reply (skill_calls.jsonl record 180), verbatim.
ROBOT_1532 = (0.96866306779177, -0.845300596361343, 3.1130531207942447)
CUBE_1532 = {
    "x": -0.8861164958402752,
    "y": -0.9794895846377999,
    "yaw": -0.7068583470577035,
}
BIN_1532 = BinFit(
    x=-0.2299999999999987,
    y=-0.8462499999999993,
    yaw=1.4529866022852795,
    length=0.365,
    width=0.26,
    height=0.384,
)
WS_1532 = Workspace(
    floor_bounds=FLOOR,
    barrier=Barrier(
        x=0.0,
        y=0.374,
        length=6.0,
        thickness=0.05,
        yaw=0.026694,
        robot_side=(0.026689993556555915, -0.9996437586680323),
    ),
    clearance=0.05,
)


def _poses(start, legs, steps=20):
    poses, here = [], start
    for goal in legs:
        d = math.atan2(math.sin(goal[2] - here[2]), math.cos(goal[2] - here[2]))
        for k in range(1, steps + 1):
            f = k / steps
            poses.append(
                (
                    here[0] + f * (goal[0] - here[0]),
                    here[1] + f * (goal[1] - here[1]),
                    here[2] + f * d,
                )
            )
        here = goal
    return poses


def test_the_1532_straight_drive_clips_the_bin():
    """The refusal as logged: the one standoff tried, and its straight drive."""
    stand = standoff_pose(
        ROBOT_1532,
        (CUBE_1532["x"], CUBE_1532["y"]),
        GEOM,
        FLOOR,
        WS_1532.valid(BIN_1532),
    )
    problem = WS_1532.straight_problem(ROBOT_1532, stand, BIN_1532)
    assert problem is not None and "overlaps the bin" in problem
    assert WS_1532.plan_path(ROBOT_1532, stand, BIN_1532)[1] is not None


def test_the_1532_pick_drives_there_through_a_waypoint():
    """The pick now starts: the standoff it chose is reached by legs that each pass the
    straight check, and no pose on them overlaps the bin or crosses the barrier."""
    assert pick_problem(ROBOT_1532, CUBE_1532, GEOM, FLOOR, WS_1532, BIN_1532) is None
    stand, legs, problem = pick_stand(
        ROBOT_1532, CUBE_1532, GEOM, FLOOR, WS_1532, BIN_1532
    )
    assert problem is None and stand is not None
    assert legs[-1] == stand and 1 < len(legs) <= 2
    here = ROBOT_1532
    for leg in legs:
        assert WS_1532.straight_problem(here, leg, BIN_1532) is None
        here = leg
    bin_box = _bin_corners(BIN_1532, WS_1532.clearance)
    for pose in _poses(ROBOT_1532, legs):
        assert not _rectangles_overlap(footprint(pose), bin_box)
        assert WS_1532.barrier_clearance(pose) >= WS_1532.clearance
    assert math.hypot(stand[0] - CUBE_1532["x"], stand[1] - CUBE_1532["y"]) == (
        pytest.approx(GEOM.standoff)
    )
    # The waypoint drive is the pick's only: a plain plan_path still refuses.
    assert WS_1532.plan_path(ROBOT_1532, stand, BIN_1532)[1] is not None
    assert WS_1532.plan_path(ROBOT_1532, stand, BIN_1532, detour=True) == (legs, None)


def test_the_picks_drive_follows_the_waypoint_legs_in_order(monkeypatch):
    """``_drive_to(..., detour=True)`` (the pick's drive) plans from where the base is
    and drives each returned leg straight, waypoint first; without ``detour`` the same
    drive is refused before anything moves."""

    class _Env:
        def reset(self):  # pylint: disable=missing-function-docstring
            return type("Obs", (), {"map_base_pose": SE2(*ROBOT_1532)})(), {}

    driven = []
    monkeypatch.setattr(
        robot_skills,
        "_drive_straight",
        lambda real_env, pose, stop: driven.append(tuple(pose)) or tuple(pose),
    )
    stand, legs, _ = pick_stand(ROBOT_1532, CUBE_1532, GEOM, FLOOR, WS_1532, BIN_1532)
    with pytest.raises(ValueError, match="overlaps the bin"):
        robot_skills._drive_to(_Env(), list(stand), WS_1532, None, BIN_1532)
    assert not driven
    reached = robot_skills._drive_to(
        _Env(), list(stand), WS_1532, None, BIN_1532, detour=True
    )
    assert driven == legs and reached == stand


def test_a_pick_solvable_straight_keeps_its_one_leg_plan():
    """No bin in the way: the same standoff as before, one straight leg."""
    robot = (0.9, -0.8, math.pi)
    cube = {"x": 0.2, "y": -0.8, "yaw": 0.0}
    old = standoff_pose(robot, (0.2, -0.8), GEOM, FLOOR, WS_1532.valid(None))
    assert WS_1532.plan_path(robot, old, None) == ([old], None)
    stand, legs, problem = pick_stand(robot, cube, GEOM, FLOOR, WS_1532, None)
    assert problem is None and stand == old and legs == [old]
    assert WS_1532.plan_path(robot, old, None, detour=True) == ([old], None)


def test_an_unreachable_cube_is_still_refused():
    """The robot shut in a corridor between the barrier and the floor's edge, a bin
    across it: the standoffs beyond the bin exist, but no drive (straight, backing off
    or through a waypoint) reaches any, so the pick is refused with the first's
    reason."""
    corridor = Workspace(
        floor_bounds=FLOOR,
        barrier=Barrier(
            x=0.0, y=-1.0, length=6.0, thickness=0.05, yaw=0.0, robot_side=(0.0, -1.0)
        ),
        clearance=0.05,
    )
    across = BinFit(x=1.0, y=-1.3, yaw=0.0, length=0.3, width=0.8, height=0.3)
    robot = (1.5, -1.42, math.pi)
    cube = {"x": -0.5, "y": -1.42, "yaw": 0.0}
    assert corridor.pose_problem(robot, across) is None
    stands = list(
        standoff_poses(robot, (-0.5, -1.42), GEOM, FLOOR, corridor.valid(across))
    )
    assert stands  # standoffs exist; none can be driven to
    stand, legs, why = pick_stand(robot, cube, GEOM, FLOOR, corridor, across)
    assert stand is None and legs == []
    assert why is not None and why.startswith("pick standoff unreachable")
    assert "no single waypoint" in why and "overlaps the bin" in why
    assert pick_problem(robot, cube, GEOM, FLOOR, corridor, across) == why
