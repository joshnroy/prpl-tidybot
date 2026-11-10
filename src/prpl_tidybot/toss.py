"""Trajectory math for the arm toss, ported from Jimmy Wu's real-robot toss.

The toss (``KinovaArm.toss`` and ``MoveJController`` in Jimmy's tidybot code) moves the
arm to a wind-up pose under the Kinova's own high-level controller, then swings it to a
release pose with a trapezoidal joint profile streamed at 1 kHz in low-level position
servoing, opening the gripper at cycle ``gripper_release_ms`` of the swing. Everything
here is pure so the profile can be checked without the arm;
``third_party/arm_server.py`` runs it.

All angles are Kortex degrees (the units of the actuator feedback and of ``Base_pb2``
joint-angle actions). Joint 1 carries the toss heading; the other six joints of both
poses are Jimmy's constants.
"""

from __future__ import annotations

import math
from typing import Any, Sequence

import numpy as np
from numpy.typing import NDArray

CYCLIC_STEP_SIZE = 0.001  # s; one cycle of the 1 kHz low-level loop.

# Jimmy's toss.
TOSS_MAX_VEL = 140.0  # deg/s, of the joint that moves furthest.
TOSS_MAX_ACCEL = 300.0  # deg/s^2
TOSS_MAX_DECEL = 200.0  # deg/s^2
TOSS_GRIPPER_RELEASE_MS = 600  # cycle of the swing at which the gripper opens.
_TOSS_WINDUP_JOINTS_2_TO_7 = (50.0, 180.0, 250.0, 0.0, 260.0, 90.0)
_TOSS_RELEASE_JOINTS_2_TO_7 = (20.0, 180.0, 325.0, 0.0, 25.0, 90.0)

# Jimmy's arrival check before the swing: every joint within 1 deg of the
# wind-up pose, or the swing is not started.
TOSS_WINDUP_TOLERANCE_DEG = 1.0


def toss_windup_pose(heading: float) -> list[float]:
    """The pose the arm is moved to before the swing, with joint 1 at ``heading``."""
    return [float(heading), *_TOSS_WINDUP_JOINTS_2_TO_7]


def toss_release_pose(heading: float) -> list[float]:
    """The pose the swing ends at, with joint 1 at ``heading``."""
    return [float(heading), *_TOSS_RELEASE_JOINTS_2_TO_7]


def trapezoidal_motion_profile(
    pos_diff: float,
    max_vel: float,
    max_accel: float,
    max_decel: float,
    step_size: float = CYCLIC_STEP_SIZE,
) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
    """Times and positions (from 0 to ``pos_diff``) of a trapezoidal profile.

    A port of ``MoveJController.trapezoidal_motion_profile``: one sample per
    ``step_size``, float32 like the original. When ``pos_diff`` is too short to reach
    ``max_vel`` the profile accelerates straight into deceleration (with the original's
    half-distance bug in that branch fixed; see below).
    """
    if pos_diff < 0:
        raise ValueError(f"pos_diff must be non-negative, got {pos_diff}.")

    if pos_diff < 0.5 * max_vel**2 / max_accel + 0.5 * max_vel**2 / max_decel:
        # Jimmy's original drops the factor of 2 here, so this branch covered
        # only half of pos_diff. Every toss swing is long enough to reach
        # max_vel and never takes it; any other caller would stop short.
        accel_duration = (2 * pos_diff / (max_accel + max_accel**2 / max_decel)) ** 0.5
        decel_duration = (max_accel / max_decel) * accel_duration
        const_duration = 0.0
    else:
        accel_duration = max_vel / max_accel
        decel_duration = max_vel / max_decel
        const_duration = (
            pos_diff / max_vel - 0.5 * max_vel / max_accel - 0.5 * max_vel / max_decel
        )
    total_duration = accel_duration + const_duration + decel_duration

    t = np.arange(0, total_duration + step_size, step_size, dtype=np.float32)
    pos = np.zeros_like(t)

    accel_idx = math.ceil(accel_duration / step_size)
    pos[:accel_idx] = 0.5 * max_accel * t[:accel_idx] ** 2

    decel_idx = math.ceil((accel_duration + const_duration) / step_size)
    pos[accel_idx:decel_idx] = 0.5 * max_accel * accel_duration**2 + max_vel * (
        t[accel_idx:decel_idx] - accel_duration
    )

    tmp = t[decel_idx:] - (accel_duration + const_duration)
    pos[decel_idx:] = 0.5 * max_accel * accel_duration**2 + max_vel * const_duration
    pos[decel_idx:] += (max_decel * decel_duration) * tmp - 0.5 * max_decel * tmp**2

    return t, pos


def wrapped_joint_differences(
    start: Sequence[float], goal: Sequence[float]
) -> NDArray[np.float64]:
    """Per-joint ``goal - start`` wrapped into [-180, 180) deg: the short way."""
    diff = np.asarray(goal, dtype=np.float64) - np.asarray(start, dtype=np.float64)
    return (diff + 180.0) % 360.0 - 180.0


def max_joint_error_deg(a: Sequence[float], b: Sequence[float]) -> float:
    """The largest per-joint distance between ``a`` and ``b``, modulo 360 deg."""
    diffs = np.abs(np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64))
    return float(np.minimum(diffs % 360.0, 360.0 - diffs % 360.0).max())


def joint_trajectory(
    start: Sequence[float],
    goal: Sequence[float],
    max_vel: float,
    max_accel: float,
    max_decel: float,
) -> NDArray[np.float32]:
    """A synchronized ``(cycles, joints)`` trajectory from ``start`` to ``goal``.

    As in ``MoveJController.execute``: the joint with the largest wrapped move follows
    the trapezoidal profile, and every other joint follows the same profile scaled by
    its share of that move, so all joints start and stop together. Positions are ``start
    + share * profile`` and are not re-wrapped, so a joint crossing 360 deg is commanded
    past it, as Jimmy's code does.
    """
    diffs = wrapped_joint_differences(start, goal)
    max_diff = float(np.abs(diffs).max())
    if max_diff == 0.0:
        return np.asarray([start], dtype=np.float32)
    _, pos = trapezoidal_motion_profile(
        max_diff, max_vel=max_vel, max_accel=max_accel, max_decel=max_decel
    )
    trajectory = np.zeros((pos.shape[0], len(diffs)), dtype=np.float32)
    for i, diff in enumerate(diffs):
        trajectory[:, i] = start[i] + (diff / max_diff) * pos
    return trajectory


def plan_toss_swing(
    start: Sequence[float],
    heading: float,
    max_vel: float = TOSS_MAX_VEL,
    gripper_release_ms: int = TOSS_GRIPPER_RELEASE_MS,
    max_accel: float = TOSS_MAX_ACCEL,
    max_decel: float = TOSS_MAX_DECEL,
) -> NDArray[np.float32]:
    """The swing from ``start`` (the measured wind-up pose) to the release pose.

    Raises ``ValueError`` for a speed outside ``(0, TOSS_MAX_VEL]`` (nothing faster than
    Jimmy's toss has been run) or a release cycle the swing never reaches, which would
    leave the gripper closed.
    """
    if not 0.0 < max_vel <= TOSS_MAX_VEL:
        raise ValueError(
            f"Toss max_vel must be in (0, {TOSS_MAX_VEL}] deg/s, got {max_vel}."
        )
    trajectory = joint_trajectory(
        start,
        toss_release_pose(heading),
        max_vel=max_vel,
        max_accel=max_accel,
        max_decel=max_decel,
    )
    if not 0 <= gripper_release_ms < len(trajectory):
        raise ValueError(
            f"gripper_release_ms={gripper_release_ms} is outside the "
            f"{len(trajectory)}-cycle swing; the gripper would never open."
        )
    return trajectory


def describe_toss(
    heading: float,
    max_vel: float = TOSS_MAX_VEL,
    gripper_release_ms: int = TOSS_GRIPPER_RELEASE_MS,
) -> dict[str, Any]:
    """Timing of the nominal swing (from the exact wind-up pose), for the operator.

    ``release_fraction`` is how far through the swing the gripper opens. The
    release is a fixed cycle count, so changing the speed moves the release
    point along the swing: Jimmy's 600 ms at 140 deg/s is 41% of the way, at
    60 deg/s it is 24%.
    """
    start = toss_windup_pose(heading)
    trajectory = plan_toss_swing(
        start, heading, max_vel=max_vel, gripper_release_ms=gripper_release_ms
    )
    travel = np.abs(trajectory - trajectory[0]).max(axis=1)
    return {
        "heading_deg": float(heading),
        "max_vel_deg_s": float(max_vel),
        "gripper_release_ms": int(gripper_release_ms),
        "duration_s": (len(trajectory) - 1) * CYCLIC_STEP_SIZE,
        "release_time_s": gripper_release_ms * CYCLIC_STEP_SIZE,
        "release_fraction": float(travel[gripper_release_ms] / travel[-1]),
        "release_joints_deg": [float(q) for q in trajectory[gripper_release_ms]],
    }


def summarize_swing(trace: dict[str, Any]) -> dict[str, Any]:
    """What the arm measurably did during the swing, from the 1 kHz loop's trace
    (``TorqueControlledArm.run_position_trajectory``): peak measured joint speed, the
    measured joints and joint velocities at the cycle the gripper-open command was
    first sent, and the largest per-joint gap between measured and commanded position.

    Measured and commanded are compared on the same cycle, so the tracking error
    includes the one-frame feedback delay. Cycles whose feedback failed are NaN and
    skipped; at release the first cycle with feedback from there on is used.
    """
    t = np.asarray(trace["t"], dtype=np.float64)
    q = np.asarray(trace["q_deg"], dtype=np.float64)
    dq = np.asarray(trace["dq_deg_s"], dtype=np.float64)
    commanded = np.asarray(trace["commanded_deg"], dtype=np.float64)
    valid = np.flatnonzero(~np.isnan(t))
    release = int(trace["release_cycle"])
    summary: dict[str, Any] = {
        "measured_release_cycle": release,
        "failed_cycles": int(trace["failed_cycles"]),
        "measured_cycles": int(valid.size),
    }
    if valid.size == 0:
        return summary
    error = (q[valid] - commanded[valid] + 180.0) % 360.0 - 180.0
    summary["measured_max_vel_deg_s"] = float(np.abs(dq[valid]).max())
    summary["max_tracking_error_deg"] = float(np.abs(error).max())
    after = valid[valid >= release] if release >= 0 else valid[:0]
    if after.size:
        k = int(after[0])
        summary["measured_release_feedback_cycle"] = k
        summary["measured_release_time_s"] = float(t[k] - t[valid[0]])
        summary["measured_release_joints_deg"] = [float(v) for v in q[k]]
        summary["measured_release_joint_vel_deg_s"] = [float(v) for v in dq[k]]
    if "gripper_pos" in trace:
        summary |= _summarize_gripper_and_timing(trace, t, release)
    return summary


# Gripper motor position is 0 (open) to 100 (closed); 2 of 100 is 2% of travel.
GRIPPER_MOVED_THRESHOLD = 2.0
STALL_CYCLE_MS = 1.5


def _summarize_gripper_and_timing(
    trace: dict[str, Any], t: NDArray[np.float64], release: int
) -> dict[str, Any]:
    """When the fingers first moved after the open command, and how long each
    Refresh took / how often a cycle started over ``STALL_CYCLE_MS`` after the last."""
    gripper = np.asarray(trace["gripper_pos"], dtype=np.float64)
    refresh = np.asarray(trace["refresh_ms"], dtype=np.float64)
    gap = np.asarray(trace["loop_gap_ms"], dtype=np.float64)
    summary: dict[str, Any] = {
        "refresh_ms_max": float(np.nanmax(refresh)),
        "refresh_ms_p99": float(np.nanpercentile(refresh, 99)),
        # A cycle's period is the last cycle's Refresh plus the gap before this one.
        "stall_cycles_over_1p5ms": int(
            np.count_nonzero(refresh[:-1] + gap[1:] > STALL_CYCLE_MS)
        ),
    }
    if 0 <= release < len(gripper) and not np.isnan(gripper[release]):
        moved = np.flatnonzero(
            np.abs(gripper[release:] - gripper[release]) > GRIPPER_MOVED_THRESHOLD
        )
        if moved.size:
            k = release + int(moved[0])
            summary["measured_gripper_open_cycle"] = k
            summary["measured_gripper_open_delay_ms"] = float(
                (t[k] - t[release]) * 1000.0
            )
    return summary
