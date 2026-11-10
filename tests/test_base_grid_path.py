"""Tests for the base's grid route (:meth:`Workspace.plan_path` when no straight drive
or back-off works)."""

import dataclasses
import math
import time

import numpy as np
import pytest

from prpl_tidybot.aimed_toss import Barrier, Workspace
from prpl_tidybot.bin_localizer import BinFit
from prpl_tidybot.top_down_pick import TopDownPickGeometry, standoff_pose

FLOOR = (-1.5, 1.7, -1.5, 1.5)
# The room of 2026-11-08 (inbin-audit s2, skill_calls.jsonl record 180: the observe at
# 15:32 whose can_pick was refused with "base footprint overlaps the bin").
TAPE = Barrier(
    x=0.0,
    y=0.374,
    length=6.0,
    thickness=0.05,
    yaw=0.026694,
    robot_side=(0.026689993556555915, -0.9996437586680323),
)
WS = Workspace(floor_bounds=FLOOR, barrier=TAPE, clearance=0.05)
CLOSE_BIN = BinFit(
    x=-0.2299999999999987,
    y=-0.8462499999999993,
    yaw=1.4529866022852795,
    length=0.365,
    width=0.26,
    height=0.384,
)
ROBOT_1532 = (0.96866306779177, -0.845300596361343, 3.1130531207942447)
CUBE_1532 = {
    "x": -0.8861164958402752,
    "y": -0.9794895846377999,
    "yaw": -0.7068583470577035,
}
GEOM = TopDownPickGeometry()


def _stand_1532():
    return standoff_pose(
        ROBOT_1532,
        (CUBE_1532["x"], CUBE_1532["y"]),
        GEOM,
        FLOOR,
        WS.valid(CLOSE_BIN),
    )


def _assert_every_leg_passes(ws, start, legs, bin_fit):
    here = start
    for leg in legs:
        assert ws.straight_problem(here, leg, bin_fit) is None, (here, leg)
        here = leg


def test_the_1532_stand_behind_the_close_bin_is_reached_by_a_bent_route():
    """15:32: the preferred stand, behind the bin from the robot, has no straight
    drive (it crosses the bin) and no back-off; the grid route goes round, and every
    leg of it passes the straight-drive check."""
    stand = _stand_1532()
    assert stand == pytest.approx((-0.594, -0.377, -2.022), abs=1e-3)
    assert "overlaps the bin" in WS.straight_problem(ROBOT_1532, stand, CLOSE_BIN)
    legs, problem = WS.plan_path(ROBOT_1532, stand, CLOSE_BIN)
    assert problem is None
    assert 2 <= len(legs) <= 4
    assert legs[-1] == pytest.approx(stand)
    _assert_every_leg_passes(WS, ROBOT_1532, legs, CLOSE_BIN)
    assert WS.path_problem(ROBOT_1532, stand, CLOSE_BIN) is None


def test_the_1532_pick_is_no_longer_refused():
    """The pick's own check (``can_pick``) at 15:32 now passes."""
    from prpl_tidybot.robot_skills import (  # pylint: disable=import-outside-toplevel
        pick_problem,
    )

    assert pick_problem(ROBOT_1532, CUBE_1532, GEOM, FLOOR, WS, CLOSE_BIN) is None


def test_a_straight_drive_is_returned_exactly_as_before():
    """When the direct drive is valid the planner returns [goal] untouched: the grid
    is never consulted."""
    goal = (0.5, -0.3, 0.7)
    assert WS.plan_path(ROBOT_1532, goal, CLOSE_BIN) == ([goal], None)
    goal = (1.2, -1.2, math.pi / 2)
    legs, problem = WS.plan_path(ROBOT_1532, goal, CLOSE_BIN)
    assert problem is None and legs == [goal]


def test_a_back_off_drive_is_returned_exactly_as_before():
    """The back-off case (run9's far stand) still takes the back-off, not a grid
    route."""
    far_stand = (0.22318, -0.01625, 1.56916)
    standoff = (0.512, -0.060, math.atan2(-0.161 + 0.060, 1.174 - 0.512))
    legs, problem = WS.plan_path(far_stand, standoff)
    assert problem is None
    assert legs == [WS.back_off_pose(far_stand), standoff]


@pytest.mark.parametrize(
    "goal, word",
    [
        ((1.9, -0.8, 0.0), "drivable box"),  # outside the floor bounds
        ((0.0, 0.9, 0.0), "barrier"),  # across the tape
        ((-0.23, -0.85, 0.0), "bin"),  # on the bin
    ],
)
def test_an_invalid_goal_is_still_refused_with_its_reason(goal, word):
    """A goal the base may not stand at is refused, naming why."""
    legs, problem = WS.plan_path(ROBOT_1532, goal, CLOSE_BIN)
    assert legs == [] and problem is not None and word in problem


# A long bin across the floor's -x, -y corner: the corner pose itself is valid (the
# base fits between the bin and the corner) but no route leads in or out of it.
CORNER_BIN = BinFit(
    x=-1.05, y=-1.05, yaw=-math.pi / 4, length=1.2, width=0.3, height=0.2
)
CORNER = (-1.5, -1.5, math.pi / 4)


def test_an_enclosed_goal_is_refused_with_a_clear_reason():
    """The corner behind the long bin: a valid pose, no route to it."""
    ws = Workspace(floor_bounds=FLOOR, clearance=0.05)
    assert ws.pose_problem(CORNER, CORNER_BIN) is None
    start = (0.5, 0.5, 0.0)
    assert ws.pose_problem(start, CORNER_BIN) is None
    legs, problem = ws.plan_path(start, CORNER, CORNER_BIN)
    assert legs == [] and problem is not None
    assert "no route" in problem and "overlaps the bin" in problem


def test_a_goal_walled_off_by_the_bin_is_not_reached_across_the_barrier():
    """A bin spanning the whole floor leaves a strip between it and the tape: a goal
    there is a valid pose, but the only ways in would cross the tape or the bin, so it
    stays refused."""
    wall_bin = BinFit(x=0.1, y=-0.6, yaw=0.0, length=3.5, width=0.2, height=0.2)
    start = (0.1, -1.2, 0.0)
    goal = (0.1, -0.05, 0.0)
    assert WS.pose_problem(start, wall_bin) is None
    assert WS.pose_problem(goal, wall_bin) is None
    legs, problem = WS.plan_path(start, goal, wall_bin)
    assert legs == [] and problem is not None and "no route" in problem


def test_the_route_is_deterministic():
    """Same input, same legs, from a fresh workspace too."""
    stand = _stand_1532()
    first = WS.plan_path(ROBOT_1532, stand, CLOSE_BIN)
    again = WS.plan_path(ROBOT_1532, stand, CLOSE_BIN)
    fresh = Workspace(floor_bounds=FLOOR, barrier=TAPE, clearance=0.05).plan_path(
        ROBOT_1532, stand, CLOSE_BIN
    )
    assert first == again == fresh


def test_a_route_is_planned_in_under_a_fifth_of_a_second():
    """The 15:32 route and the enclosed refusal (the whole grid searched) both plan
    in under 0.2 s."""
    stand = _stand_1532()
    ws = Workspace(floor_bounds=FLOOR, clearance=0.05)
    for start, goal, ws_, fit in [
        (ROBOT_1532, stand, WS, CLOSE_BIN),
        ((0.5, 0.5, 0.0), CORNER, ws, CORNER_BIN),
    ]:
        t0 = time.perf_counter()
        ws_.plan_path(start, goal, fit)
        assert time.perf_counter() - t0 < 0.2


@dataclasses.dataclass(frozen=True)
class _Refusing(Workspace):
    """The 15:32 workspace, but with no drive to the stands in ``refused``."""

    refused: tuple = ()

    def path_problem(self, start, goal, bin_fit=None):
        if any(goal == pytest.approx(r) for r in self.refused):
            return "test: no route"
        return super().path_problem(start, goal, bin_fit)


def _candidates(robot, cube_xy, ws, fit, count):
    """standoff_pose's valid candidates, in its order."""
    found = []
    while len(found) < count:
        found.append(
            standoff_pose(
                robot,
                cube_xy,
                GEOM,
                FLOOR,
                lambda p: ws.pose_problem(p, fit) is None
                and not any(p == pytest.approx(f) for f in found),
            )
        )
    return found


def test_the_pick_takes_the_next_reachable_stand_in_order():
    """No drive to the preferred stand: the pick takes standoff_pose's next valid
    candidate it can drive to, in standoff_pose's own order; with none at all, the
    preferred stand's refusal."""
    from prpl_tidybot.robot_skills import (  # pylint: disable=import-outside-toplevel
        pick_stand,
    )

    xy = (CUBE_1532["x"], CUBE_1532["y"])
    candidates = _candidates(ROBOT_1532, xy, WS, CLOSE_BIN, 3)
    first, second, third = candidates[0], candidates[1], candidates[2]
    assert first == pytest.approx(_stand_1532())
    one = _Refusing(floor_bounds=FLOOR, barrier=TAPE, refused=(first,))
    assert pick_stand(ROBOT_1532, xy, GEOM, FLOOR, one, CLOSE_BIN) == (
        pytest.approx(second),
        None,
    )
    two = _Refusing(floor_bounds=FLOOR, barrier=TAPE, refused=(first, second))
    stand, why = pick_stand(ROBOT_1532, xy, GEOM, FLOOR, two, CLOSE_BIN)
    assert why is None and stand == pytest.approx(third)

    class _None(Workspace):
        def path_problem(self, start, goal, bin_fit=None):
            return "test: no route"

    nothing = _None(floor_bounds=FLOOR, barrier=TAPE)
    assert pick_stand(ROBOT_1532, xy, GEOM, FLOOR, nothing, CLOSE_BIN) == (
        None,
        "pick standoff unreachable: test: no route",
    )


def test_the_pick_keeps_the_preferred_stand_when_it_is_reachable():
    """15:32: the preferred stand now has a route, so the pick keeps it."""
    from prpl_tidybot.robot_skills import (  # pylint: disable=import-outside-toplevel
        pick_stand,
    )

    xy = (CUBE_1532["x"], CUBE_1532["y"])
    stand, why = pick_stand(ROBOT_1532, xy, GEOM, FLOOR, WS, CLOSE_BIN)
    assert why is None and stand == pytest.approx(_stand_1532())


def test_a_clear_sweep_passes_the_sampled_check_at_any_resolution():
    """The swept-footprint test the route's legs in and out rest on: a translation it
    calls clear passes straight_problem sampled every 2 mm (it is exact, not
    sampled), over a seeded spread of legs through the 15:32 room."""
    rng = np.random.default_rng(0)
    clear = 0
    for _ in range(300):
        a = (rng.uniform(-1.5, 1.7), rng.uniform(-1.5, 0.3))
        b = (a[0] + rng.uniform(-0.8, 0.8), a[1] + rng.uniform(-0.8, 0.8))
        heading = rng.uniform(-math.pi, math.pi)
        if WS.pose_problem((*a, heading), CLOSE_BIN) is not None:
            continue
        if not WS._sweep_clear(  # pylint: disable=protected-access
            a, b, heading, CLOSE_BIN, WS.clearance
        ):
            continue
        clear += 1
        steps = max(20, math.ceil(math.hypot(b[0] - a[0], b[1] - a[1]) / 0.002))
        assert (
            WS.straight_problem((*a, heading), (*b, heading), CLOSE_BIN, steps) is None
        )
    assert clear > 50


def test_every_route_found_in_the_room_passes_the_straight_check():
    """A seeded spread of starts and goals in the 15:32 room: whatever plan_path
    returns, every leg passes straight_problem, and the last leg ends at the goal."""
    rng = np.random.default_rng(1)

    def pose():
        while True:
            p = (
                rng.uniform(-1.5, 1.7),
                rng.uniform(-1.5, 0.3),
                rng.uniform(-math.pi, math.pi),
            )
            if WS.pose_problem(p, CLOSE_BIN) is None:
                return p

    routed = 0
    for _ in range(60):
        start, goal = pose(), pose()
        legs, problem = WS.plan_path(start, goal, CLOSE_BIN)
        if problem is not None:
            assert legs == []
            continue
        routed += len(legs) > 1
        assert legs[-1] == pytest.approx(goal)
        _assert_every_leg_passes(WS, start, legs, CLOSE_BIN)
    assert routed > 0
