"""Pour the bin out: the real counterpart of the simulator's ``BinPourController``
(hitl-pmp, ``environments/tossing3d/kinder_backend.py``).

The same stages: park ``standoff`` from the bin facing one of its faces, grasp the
near wall's rim top-down, lift the bin clear of the floor, turn the base a quarter
turn so the bin hangs beside where it stood, tip it over, hold it while whatever is
inside falls out, then reverse every move, set the bin down where it was and let go.

One deliberate difference: the tip is about the gripped wall's own direction (the
wall tangent, horizontal), not about the robot's heading. The fingers close across
the wall, so a rotation about the heading would be about the line through both pads,
which a pinch grip resists only by friction about the contact normal; about the wall
tangent the pads' normal forces hold it. The bin tips forward, away from the robot,
so it never swings back over the base.

Everything here is planning (poses and joint targets); the skill server streams the
stages. Frames: the map frame for the base, the arm's base frame for the arm (as in
:mod:`prpl_tidybot.top_down_pick`).
"""

from __future__ import annotations

# mujoco is a compiled extension; pylint cannot see its members.
# pylint: disable=no-member

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

import mujoco  # type: ignore[import-untyped]
import numpy as np
from numpy.typing import NDArray

from prpl_tidybot.bin_localizer import BinFit
from prpl_tidybot.third_party.constants import RETRACT_ARM_CONF
from prpl_tidybot.top_down_pick import (  # pylint: disable=protected-access
    _SOLVER_LOCK,
    TopDownPickGeometry,
    _down_quat,
    _solver,
    forward_kinematics,
)

# A joint may change by at most this (rad) between consecutive waypoints.
_MAX_STEP = 0.5


@dataclass(frozen=True)
class BinPourGeometry:  # pylint: disable=too-many-instance-attributes
    """Distances (m) and angles (rad) of the pour; the defaults follow the simulator's
    controller (0.48 m lift, a quarter turn) except where the real arm's joint limits
    differ: at the simulator's 0.76 m standoff the elbow is folded so far that the
    wrist reaches only ~90 degrees of tip within the Gen3's limits (joint 6 +-119.7,
    joint 4 +-147.3), the opening still sideways; from 0.84 m it reaches 120."""

    standoff: float = 0.84
    # Other standoffs tried, in order, when the preferred one does not solve.
    standoff_options: tuple[float, ...] = (0.80, 0.88, 0.92, 0.76)
    # The pinch point sits this far inside the wall's outer face (the rolled lip and
    # the wall between the pads) and this far below the rim.
    rim_inset: float = 0.012
    below_rim: float = 0.03
    hover: float = 0.15
    # The pinch point's height above the floor while the bin is up and tipped: what
    # the arm's reach depends on, whatever the bin's height (the blue bin, rim at
    # 0.384 m, is then lifted 0.48 m as in the simulator).
    tip_height: float = 0.834
    # The bin's base must clear the floor by at least this once lifted.
    min_lift: float = 0.2
    # How far the bin tips over: past 90 degrees the opening faces down.
    tilt: float = math.radians(120.0)
    # Waypoint spacing for the streamed moves.
    lift_step: float = 0.08
    tilt_step: float = math.radians(10.0)
    # Seconds held upside down while the contents fall out.
    empty_s: float = 2.0
    # The bin, lifted, may swing this far outside the drivable box (nothing at its
    # height is assumed there).
    swing_margin: float = 0.3


@dataclass(frozen=True)
class PourStage:
    """One stage: the base pose (map), the arm's waypoints (joints, streamed in order)
    and the gripper command held throughout (0 open, 1 closed)."""

    name: str
    base: tuple[float, float, float]
    arm: tuple[tuple[float, ...], ...]
    gripper: float
    hold_s: float = 0.0


@dataclass(frozen=True)
class BinPourPlan:
    """The chosen approach and its stages."""

    stand: tuple[float, float, float]
    face: int
    turn: float
    stages: tuple[PourStage, ...] = field(default_factory=tuple)
    standoff: float = 0.0


def _tool_frame(
    q: Sequence[float], geom: TopDownPickGeometry
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Pinch point and tool rotation (columns x, closing y, approach z), arm frame."""
    pos, approach, closing = forward_kinematics(q, geom)
    return pos, np.column_stack([np.cross(closing, approach), closing, approach])


def _down_rotation() -> NDArray[np.float64]:
    """The tool pointing down with the fingers closing along the arm's +x (across a
    wall that faces the robot)."""
    quat = _down_quat(0.0)  # (x, y, z, w)
    mat = np.empty(9)
    mujoco.mju_quat2Mat(mat, np.array([quat[3], quat[0], quat[1], quat[2]]))
    return mat.reshape(3, 3)


def _rot_y(angle: float) -> NDArray[np.float64]:
    c, s = math.cos(angle), math.sin(angle)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])


def joint_limits(geom: TopDownPickGeometry) -> tuple[NDArray[np.float64], ...]:
    """The arm's joint limits from the IK model (unlimited joints as +-inf): the
    solver itself does not enforce them."""
    model = _solver(geom).model
    lo = np.array(
        [model.jnt_range[j, 0] if model.jnt_limited[j] else -np.inf for j in range(7)]
    )
    hi = np.array(
        [model.jnt_range[j, 1] if model.jnt_limited[j] else np.inf for j in range(7)]
    )
    return lo, hi


def tool_joints(
    position: Sequence[float],
    rotation: NDArray[np.float64],
    geom: TopDownPickGeometry,
    seed: Sequence[float],
) -> list[float]:
    """Joints putting the pinch point at ``position`` with tool rotation ``rotation``
    (arm frame). Raises ``ValueError`` if the IK does not get within 5 mm and 2
    degrees, or only outside the joint limits."""
    wxyz = np.empty(4)
    mujoco.mju_mat2Quat(wxyz, np.asarray(rotation, float).flatten())
    quat = np.array([wxyz[1], wxyz[2], wxyz[3], wxyz[0]])
    goal = np.asarray(position, float)
    q = np.asarray(seed, float)
    with _SOLVER_LOCK:
        for _ in range(10):
            q = _solver(geom).solve(goal, quat, q)  # type: ignore[no-untyped-call]
            pos, rot = _tool_frame(q, geom)
            err = rot @ np.asarray(rotation).T
            angle = math.acos(max(-1.0, min(1.0, (np.trace(err) - 1) / 2)))
            if np.linalg.norm(pos - goal) < 0.005 and angle < math.radians(2.0):
                lo, hi = joint_limits(geom)
                if np.any(q < lo) or np.any(q > hi):
                    raise ValueError(
                        f"IK reached {np.round(goal, 3).tolist()} only outside the "
                        f"joint limits ({np.round(np.degrees(q), 0).tolist()} deg)"
                    )
                return [float(v) for v in q]
    raise ValueError(
        f"IK did not reach {np.round(goal, 3).tolist()} "
        f"(got {np.round(pos, 3).tolist()})"
    )


def bin_points_in_hand(
    q: Sequence[float],
    geom: TopDownPickGeometry,
    pour: BinPourGeometry,
    span: float,
    bin_fit: BinFit,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """The held bin's bottom centre and rim centre (arm frame) at joints ``q``, the
    bin rigid in the hand as it was grasped."""
    reach = span / 2 - pour.rim_inset
    down = _down_rotation()
    rim = down.T @ np.array([reach, 0.0, pour.below_rim])
    bottom = down.T @ np.array([reach, 0.0, pour.below_rim - bin_fit.height])
    pos, rot = _tool_frame(q, geom)
    return pos + rot @ bottom, pos + rot @ rim


def _steps(start: float, end: float, step: float) -> list[float]:
    n = max(1, math.ceil(abs(end - start) / step))
    return [start + (end - start) * k / n for k in range(1, n + 1)]


def _swing_clear(
    centre: Sequence[float],
    stand: Sequence[float],
    turn: float,
    radius: float,
    bounds: Sequence[float],
) -> bool:
    """Whether the bin, carried round the base by ``turn``, stays in ``bounds``."""
    for a in np.linspace(0.0, turn, 9):
        c, s = math.cos(a), math.sin(a)
        dx, dy = centre[0] - stand[0], centre[1] - stand[1]
        x, y = stand[0] + c * dx - s * dy, stand[1] + s * dx + c * dy
        if not (
            bounds[0] + radius <= x <= bounds[1] - radius
            and bounds[2] + radius <= y <= bounds[3] - radius
        ):
            return False
    return True


def _arm_stages(
    geom: TopDownPickGeometry,
    pour: BinPourGeometry,
    standoff: float,
    span: float,
    bin_fit: BinFit,
) -> dict[str, list[list[float]]]:
    """The arm's waypoints per stage, in the arm frame (the same for any stand)."""
    x = standoff - span / 2 + pour.rim_inset - geom.mounting_offset
    z_rim = geom.floor_z + bin_fit.height - pour.below_rim
    z_up = geom.floor_z + pour.tip_height
    if z_up - z_rim < pour.min_lift:
        raise ValueError(
            f"a {bin_fit.height:.2f} m bin held at {pour.tip_height:.2f} m clears the "
            f"floor by under {pour.min_lift:.2f} m"
        )
    down = _down_rotation()
    seed: list[float] = list(RETRACT_ARM_CONF)
    hover = tool_joints((x, 0.0, z_rim + pour.hover), down, geom, seed)
    rim = [tool_joints((x, 0.0, z_rim), down, geom, hover)]
    lift: list[list[float]] = []
    for z in _steps(z_rim, z_up, pour.lift_step):
        lift.append(tool_joints((x, 0.0, z), down, geom, (lift or rim)[-1]))
    tip: list[list[float]] = []
    for a in _steps(0.0, -pour.tilt, pour.tilt_step):
        tip.append(
            tool_joints((x, 0.0, z_up), _rot_y(a) @ down, geom, (tip or lift)[-1])
        )
    untip = tip[-2::-1] + [lift[-1]]
    lower = lift[-2::-1] + rim
    return {
        "hover": [hover],
        "rim": rim,
        "lift": lift,
        "tip": tip,
        "untip": untip,
        "lower": lower,
        "withdraw": [hover],
    }


def plan_bin_pour(  # pylint: disable=too-many-locals
    bin_fit: BinFit,
    robot: Sequence[float],
    geom: TopDownPickGeometry,
    pour: BinPourGeometry,
    floor_bounds: Sequence[float],
    is_valid: Callable[[tuple[float, float, float]], bool] | None = None,
) -> BinPourPlan:
    """The pour for the bin at ``bin_fit``: of the four faces and two turn directions,
    the one with the standoff nearest the robot whose stand lies in the drivable box,
    whose turn keeps the lifted bin within ``swing_margin`` of it, and whose arm moves
    all solve. ``is_valid`` (``Workspace.valid``) also vets the stand and the turned
    stand: on the robot's side of the barrier, clear of it and of the bin. Raises
    ``ValueError`` (with each candidate's reason) if none does."""
    radius = 0.5 * math.hypot(bin_fit.length, bin_fit.width)
    grown = (
        floor_bounds[0] - pour.swing_margin,
        floor_bounds[1] + pour.swing_margin,
        floor_bounds[2] - pour.swing_margin,
        floor_bounds[3] + pour.swing_margin,
    )
    reasons: list[str] = []
    arms: dict[tuple[float, int], dict[str, list[list[float]]] | str] = {}
    for standoff in (pour.standoff, *pour.standoff_options):
        plan = _plan_at(
            bin_fit, robot, geom, pour, standoff, grown, radius, arms, is_valid
        )
        if isinstance(plan, BinPourPlan):
            return plan
        reasons.extend(f"{standoff:.2f} m, {r}" for r in plan)
    raise ValueError(f"No bin pour: {'; '.join(reasons)}")


def _plan_at(  # pylint: disable=too-many-arguments,too-many-positional-arguments,too-many-locals
    bin_fit: BinFit,
    robot: Sequence[float],
    geom: TopDownPickGeometry,
    pour: BinPourGeometry,
    standoff: float,
    grown: Sequence[float],
    radius: float,
    arms: dict[tuple[float, int], dict[str, list[list[float]]] | str],
    is_valid: Callable[[tuple[float, float, float]], bool] | None = None,
) -> BinPourPlan | list[str]:
    """The pour from ``standoff``, or each candidate's reason for failing."""
    floor_bounds = (
        grown[0] + pour.swing_margin,
        grown[1] - pour.swing_margin,
        grown[2] + pour.swing_margin,
        grown[3] - pour.swing_margin,
    )
    candidates = []
    for face in range(4):
        heading = bin_fit.yaw + face * math.pi / 2
        heading = (heading + math.pi) % (2 * math.pi) - math.pi
        stand = (
            bin_fit.x - standoff * math.cos(heading),
            bin_fit.y - standoff * math.sin(heading),
            heading,
        )
        for turn in (math.pi / 2, -math.pi / 2):
            travel = math.hypot(stand[0] - robot[0], stand[1] - robot[1])
            candidates.append((travel, face, turn, stand))
    reasons = []
    for _, face, turn, stand in sorted(candidates):
        if not (
            floor_bounds[0] <= stand[0] <= floor_bounds[1]
            and floor_bounds[2] <= stand[1] <= floor_bounds[3]
        ):
            reasons.append(f"face {face}: stand {stand[:2]} outside the drivable box")
            continue
        turned = (stand[0], stand[1], stand[2] + turn)
        if is_valid is not None and not (is_valid(stand) and is_valid(turned)):
            reasons.append(f"face {face}, turn {math.degrees(turn):+.0f}: workspace")
            continue
        if not _swing_clear((bin_fit.x, bin_fit.y), stand, turn, radius, grown):
            reasons.append(f"face {face}, turn {math.degrees(turn):+.0f}: swing")
            continue
        span = bin_fit.length if face % 2 == 0 else bin_fit.width
        key = (standoff, face % 2)
        if key not in arms:
            try:
                arms[key] = _arm_stages(geom, pour, standoff, span, bin_fit)
            except ValueError as e:
                arms[key] = str(e)
        arm = arms[key]
        if isinstance(arm, str):
            reasons.append(f"face {face}: {arm}")
            continue

        def stage(
            name: str,
            waypoints: list[list[float]],
            gripper: float,
            base: tuple[float, float, float] = stand,
            hold: float = 0.0,
        ) -> PourStage:
            return PourStage(
                name, base, tuple(tuple(q) for q in waypoints), gripper, hold
            )

        held = [arm["lift"][-1]]
        stages = (
            stage("hover", arm["hover"], 0.0),
            stage("rim", arm["rim"], 0.0),
            stage("close", arm["rim"], 1.0),
            stage("lift", arm["lift"], 1.0),
            stage("turn", held, 1.0, turned),
            stage("tip", arm["tip"], 1.0, turned),
            stage("empty", [arm["tip"][-1]], 1.0, turned, pour.empty_s),
            stage("untip", arm["untip"], 1.0, turned),
            stage("unturn", held, 1.0),
            stage("lower", arm["lower"], 1.0),
            stage("open", arm["rim"], 0.0),
            stage("withdraw", arm["withdraw"], 0.0),
        )
        return BinPourPlan(stand, face, turn, stages, standoff)
    return reasons


def pour_point(
    plan: BinPourPlan,
    bin_fit: BinFit,
    geom: TopDownPickGeometry,
    pour: BinPourGeometry,
) -> tuple[float, float]:
    """Where the contents fall (map frame): under the tipped bin's rim centre, the
    base at its turned pose."""
    tip = next(s for s in plan.stages if s.name == "tip")
    span = bin_fit.length if plan.face % 2 == 0 else bin_fit.width
    _, rim = bin_points_in_hand(tip.arm[-1], geom, pour, span, bin_fit)
    x, y, theta = tip.base
    ax, ay = rim[0] + geom.mounting_offset, rim[1]
    c, s = math.cos(theta), math.sin(theta)
    return (x + c * ax - s * ay, y + s * ax + c * ay)
