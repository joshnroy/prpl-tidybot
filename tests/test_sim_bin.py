"""Tests for the sim's bin: grasped at its rim, it rides with the gripper; rolled over,
whatever is in it falls out under the rim; released, it stands upright where it was
set down. The arm is moved through the real dump plan's joint targets."""

import math

import numpy as np
import pytest

from prpl_tidybot.bin_localizer import BinFit
from prpl_tidybot.bin_pour import BinPourGeometry, plan_bin_pour, pour_point
from prpl_tidybot.sim_robot import RIM_READING, SimBinSource, SimWorld
from prpl_tidybot.top_down_pick import TopDownPickGeometry

GEOM = TopDownPickGeometry()
POUR = BinPourGeometry()
FLOOR = (-1.5, 1.7, -1.5, 1.5)
BIN_XY_YAW = (0.2, -0.4, 0.0)
# The blue bin: rim length, width, height, base length, width, wall.
BIN_MODEL = (0.365, 0.26, 0.384, 0.29, 0.19, 0.005)
BOTTLE_HALF = 0.033


def _fit():
    x, y, yaw = BIN_XY_YAW
    return BinFit(x, y, yaw, *BIN_MODEL[:3], BIN_MODEL[5])


@pytest.fixture(scope="module", name="plan")
def _plan():
    """The dump the skill would make for the bin."""
    return plan_bin_pour(_fit(), (-0.6, -0.9, 0.0), GEOM, POUR, FLOOR)


def _world(plan, cube=None):
    """A world with the bin placed and the robot at the plan's standoff."""
    world = SimWorld(
        base_pose=plan.stand,
        cube=cube or (-1.0, -1.0, 0.0),
        object_half=BOTTLE_HALF,
        bin_pose=BIN_XY_YAW,
        bin_model=BIN_MODEL,
    )
    return world


def _go(world, plan, name, base=None):
    """Put the arm (and base) at a stage's last waypoint, as the streamed moves
    would, and let the controllers settle."""
    stage = next(s for s in plan.stages if s.name == name)
    for q in stage.arm:
        world.q = np.array(q, float)
        world.base = np.array(base if base is not None else stage.base, float)
        world.advance(0.02)


def _close(world):
    world.gripper_target = 1.0
    world.advance(0.8)


def _open(world):
    world.gripper_target = 0.0
    world.advance(0.8)


def test_closing_on_the_rim_grasps_the_bin_at_the_rim_reading(plan):
    """Closed over the side wall's rim, the fingers stop at the rim's reading and
    the bin is held."""
    world = _world(plan)
    _go(world, plan, "rim")
    _close(world)
    assert world.bin_held
    assert world.gripper == pytest.approx(RIM_READING)


def test_closing_on_nothing_does_not_grasp_the_bin(plan):
    """Closed in the air above the rim (the hover), the fingers close fully."""
    world = _world(plan)
    _go(world, plan, "hover")
    _close(world)
    assert not world.bin_held
    assert world.gripper == pytest.approx(1.0)


def test_the_held_bin_lifts_with_the_gripper(plan):
    """Lifted, the bin's base leaves the floor by the plan's lift."""
    world = _world(plan)
    _go(world, plan, "rim")
    _close(world)
    _go(world, plan, "lift")
    lift = POUR.tip_height - (BIN_MODEL[2] - POUR.below_rim)
    assert world.bin_transform()[2, 3] == pytest.approx(lift, abs=0.01)


def test_a_bottle_in_the_bin_falls_out_at_the_pour_point(plan):
    """A bottle inside when the bin is grasped rides with it, and falls out under the
    rim once the bin rolls over: where the plan says it will."""
    world = _world(plan, cube=(BIN_XY_YAW[0], BIN_XY_YAW[1], 0.3))
    _go(world, plan, "rim")
    _close(world)
    _go(world, plan, "lift")
    assert world.cube_position()[2] > 0.3  # riding in the lifted bin
    turn = next(s for s in plan.stages if s.name == "turn")
    world.base = np.array(turn.base, float)
    _go(world, plan, "tip")
    world.advance(1.0)
    rest = world.cube_at_rest()
    assert rest[2] == pytest.approx(BOTTLE_HALF)
    assert rest[:2] == pytest.approx(pour_point(plan, _fit(), GEOM, POUR), abs=0.03)


def test_released_the_bin_stands_upright_where_it_was(plan):
    """The whole dump and back: the bin is set down upright on its spot and let go."""
    world = _world(plan)
    for name in ("rim", "close", "lift", "turn", "tip", "untip", "unturn", "lower"):
        if name == "close":
            _close(world)
        else:
            _go(world, plan, name)
    _open(world)
    assert not world.bin_held
    t = world.bin_transform()
    assert t[2, 3] == pytest.approx(0.0, abs=1e-9)
    assert t[2, 2] == pytest.approx(1.0)
    assert t[:2, 3] == pytest.approx(BIN_XY_YAW[:2], abs=0.01)


def test_the_sim_bin_source_reports_the_live_bin(plan):
    """The skill server's bin source in sim: wherever the bin stands now."""
    world = _world(plan)
    source = SimBinSource(world)
    fit = source.localize()
    assert (fit.x, fit.y) == pytest.approx(BIN_XY_YAW[:2])
    assert (fit.length, fit.height) == pytest.approx((BIN_MODEL[0], BIN_MODEL[2]))
    world.place_bin(-0.5, -0.8, math.pi / 2)
    fit = source.localize()
    assert (fit.x, fit.y) == pytest.approx((-0.5, -0.8))
    assert abs(math.sin(fit.yaw - math.pi / 2)) < 1e-9


def test_a_rim_grasp_with_the_pads_along_the_wall_does_not_hold(plan):
    """The pads must straddle the wall: turned a quarter turn about the tool (pads
    along the wall), closing at the same point grasps nothing."""
    world = _world(plan)
    _go(world, plan, "rim")
    world.q[6] += math.pi / 2  # the wrist turns the pads about the tool axis
    world.advance(0.02)
    _close(world)
    assert not world.bin_held


def test_a_bottle_grasp_needs_the_pads_across_its_axis():
    """A lying bottle is held only when the pads close across its axis: along it,
    a finger lands on the bottle (2026-11-02, the first real bottle pick)."""
    from prpl_tidybot.bin_pour import (  # pylint: disable=import-outside-toplevel
        _down_quat,
        tool_joints,
    )
    from prpl_tidybot.third_party.constants import (  # pylint: disable=import-outside-toplevel
        RETRACT_ARM_CONF,
    )

    for pads_yaw, held in ((math.pi / 2, True), (0.0, False)):
        world = SimWorld(
            base_pose=(0.0, 0.0, 0.0),
            cube=(0.67, 0.0, 0.0),  # lying along the map's x
            object_half=BOTTLE_HALF,
            object_kind="bottle",
        )
        # _down_quat(c) lays the pinch frame's y along c and so its x, the pads,
        # along c + 90 degrees.
        quat = _down_quat(pads_yaw - math.pi / 2)
        rot = np.empty(9)
        import mujoco  # pylint: disable=import-outside-toplevel

        mujoco.mju_quat2Mat(  # pylint: disable=no-member
            rot, np.array([quat[3], quat[0], quat[1], quat[2]])
        )
        q = tool_joints(
            (0.55, 0.0, GEOM.floor_z + GEOM.grasp_height),
            rot.reshape(3, 3),
            GEOM,
            list(RETRACT_ARM_CONF),
        )
        world.q = np.array(q, float)
        world.advance(0.02)
        _close(world)
        assert world.held == held, pads_yaw
