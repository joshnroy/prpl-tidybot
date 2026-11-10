"""Toss at a bin wherever it stands: the real robot's toss parameters, their bounds, and
the base pose a toss is thrown from.

Tossing3D's toss skill takes ``(standoff_m, speed_deg_s, release_ms)``: how far the base
centre stands from the bin centre, the swing's peak joint speed and the cycle at which
the gripper opens. On this robot the swing is :func:`prpl_tidybot.toss.plan_toss_swing`
(a trapezoidal profile to a fixed release pose, at most 140 deg/s) with joint 1 at
heading 0, so the cube flies straight ahead along the base's heading. The real bounds
(:class:`RealTossBounds`) come from that swing and a ballistic model of the cube from
the release state (validated within ~0.1 m against a real throw):

* ``speed`` 100-140 deg/s: 140 is the most this arm has thrown (``TOSS_MAX_VEL``);
  below ~100 the cube does not reach a bin outside the base's own footprint.
* ``release_ms`` 500-850: across this window, at every speed above, the pinch point
  moves forward *and* up at release (forward speed >= 0.03 m/s, upward >= 0.17 m/s),
  so the cube is thrown ahead and never down at the floor or back at the robot. At
  140 deg/s and 850 ms it is almost straight up (it lands ~0.1 m ahead of release).
* ``standoff`` 0.9-1.45 m, base centre to bin centre: the span of rim-height landing
  distances that box produces.

Anything outside the box is rejected *before* the arm or base moves.

The base pose (:func:`aimed_toss_pose`) faces the bin centre from ``standoff`` along an
azimuth ``direction`` (map frame, from the bin centre to the robot). It must keep the
base's centre in the drivable box, the base's footprint (0.548 x 0.508 m, centred on
the base) clear of the bin, and the whole footprint on the robot's side of the
barrier: like Tossing3D's room-spanning barrier, the robot never crosses it.
"""

from __future__ import annotations

import heapq
import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from prpl_tidybot.bin_localizer import BinFit

# The TidyBot++ base, centre-referenced (m): along its heading, and across.
BASE_LENGTH = 0.548
BASE_WIDTH = 0.508
# The base's footprint turns within this circle about its centre.
_TURN_RADIUS = 0.5 * math.hypot(BASE_LENGTH, BASE_WIDTH)
# How much further a back-off goes, for the drive's tracking error.
BACK_OFF_SLACK = 0.02
# The grid route's cell (m), and how far from the start or the goal a straight leg into
# or out of the grid's free cells is tried.
GRID_CELL = 0.05
_ATTACH_RADIUS = 0.8
_START, _GOAL = -1, -2
_NEIGHBOURS = tuple(
    (da, db, math.hypot(da, db))
    for da in (-1, 0, 1)
    for db in (-1, 0, 1)
    if (da, db) != (0, 0)
)

Pose = tuple[float, float, float]


@dataclass(frozen=True)
class RealTossBounds:
    """The box of real toss parameters every one of which is a safe, forward, upward
    throw (see the module docstring for the derivation)."""

    standoff: tuple[float, float] = (0.9, 1.45)
    speed: tuple[float, float] = (100.0, 140.0)
    release_ms: tuple[float, float] = (500.0, 850.0)

    def validate(self, standoff: float | None, speed: float, release_ms: float) -> None:
        """Raise ``ValueError`` naming every parameter outside the box (or not a finite
        number). ``standoff`` None checks the swing alone (a toss from a fixed pose)."""
        bad = []
        checks = [
            ("standoff", standoff, self.standoff),
            ("speed", speed, self.speed),
            ("release_ms", release_ms, self.release_ms),
        ]
        for name, value, (lo, hi) in checks[1:] if standoff is None else checks:
            if not (isinstance(value, (int, float)) and math.isfinite(value)):
                bad.append(f"{name}={value!r} is not a number")
            elif not lo <= value <= hi:
                bad.append(f"{name}={value} outside [{lo}, {hi}]")
        if bad:
            raise ValueError("Toss rejected: " + "; ".join(bad) + ".")

    @classmethod
    def from_config(cls, cfg: Any) -> RealTossBounds:
        """From a ``{standoff: [lo, hi], speed: [...], release_ms: [...]}`` block."""
        return cls(
            standoff=(float(cfg["standoff"][0]), float(cfg["standoff"][1])),
            speed=(float(cfg["speed"][0]), float(cfg["speed"][1])),
            release_ms=(float(cfg["release_ms"][0]), float(cfg["release_ms"][1])),
        )


@dataclass(frozen=True)
class Barrier:
    """A rectangle on the floor (map frame) the robot never crosses: its long side
    (``length``) runs along ``yaw``.

    ``robot_side`` is the map-frame direction from the barrier into the robot's side
    (normalized): the robot works beyond the rectangle's face in that direction. The
    default, +x, is the original placeholder layout; the real room's tape runs along x
    with the robot on its -y side.
    """

    x: float = 0.0
    y: float = 0.0
    length: float = 1.8
    thickness: float = 0.05
    yaw: float = math.pi / 2
    robot_side: tuple[float, float] = (1.0, 0.0)

    def __post_init__(self) -> None:
        nx, ny = (float(v) for v in self.robot_side)
        norm = math.hypot(nx, ny)
        if not math.isfinite(norm) or norm < 1e-9:
            raise ValueError(
                f"Barrier robot_side {self.robot_side} is not a direction."
            )
        object.__setattr__(self, "robot_side", (nx / norm, ny / norm))

    def depth(self, point: Sequence[float]) -> float:
        """How far ``point`` (map x, y) lies past the barrier's robot-side face into
        the robot's side; negative on the face or across it."""
        nx, ny = self.robot_side
        c, s = math.cos(self.yaw), math.sin(self.yaw)
        # The rectangle's half-extent along the robot-side direction.
        half = 0.5 * (
            self.length * abs(nx * c + ny * s) + self.thickness * abs(-nx * s + ny * c)
        )
        return nx * (point[0] - self.x) + ny * (point[1] - self.y) - half

    def depths(
        self, x: NDArray[np.float64], y: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        """:meth:`depth` of every point (``x[i]``, ``y[i]``) at once."""
        return np.asarray(self.depth((0.0, 0.0)) + self.robot_side[0] * x) + (
            self.robot_side[1] * y
        )

    def as_dict(self) -> dict[str, Any]:
        """Plain floats: the skill server's ``observe()["barrier"]``."""
        return {
            "x": float(self.x),
            "y": float(self.y),
            "length": float(self.length),
            "thickness": float(self.thickness),
            "yaw": float(self.yaw),
            "robot_side": [float(v) for v in self.robot_side],
        }

    @classmethod
    def from_config(cls, cfg: Any) -> Barrier:
        """From ``{center: [x, y], length, thickness, yaw, robot_side: [x, y]}``
        (``robot_side`` defaults to +x)."""
        side = cfg.get("robot_side", None) if hasattr(cfg, "get") else None
        return cls(
            x=float(cfg["center"][0]),
            y=float(cfg["center"][1]),
            length=float(cfg["length"]),
            thickness=float(cfg["thickness"]),
            yaw=float(cfg["yaw"]),
            robot_side=(
                (1.0, 0.0) if side is None else (float(side[0]), float(side[1]))
            ),
        )


@dataclass(frozen=True)
class Workspace:
    """Where the base may be: its centre inside ``floor_bounds`` (x_min, x_max, y_min,
    y_max) and, when there is a barrier, its whole footprint at least ``clearance``
    beyond the barrier's robot-side face.

    ``cube_half`` is the cube's half side: a cube whose footprint reaches the barrier
    is not picked (0 checks only its centre).
    """

    floor_bounds: tuple[float, float, float, float] = (-1.5, 1.7, -1.5, 1.5)
    barrier: Barrier | None = None
    clearance: float = 0.05
    cube_half: float = 0.0

    def cube_problem(self, xy: Sequence[float], yaw: float | None) -> str | None:
        """Why the cube at ``xy`` (yaw ``yaw`` rad, None if unknown: any) may not be
        picked because of the barrier, or None: its centre past the robot-side face,
        or its footprint reaching it."""
        if self.barrier is None:
            return None
        depth = self.barrier.depth(xy)
        if depth < 0.0:
            return "cube past the barrier"
        nx, ny = self.barrier.robot_side
        if yaw is None:
            reach = self.cube_half * math.sqrt(2.0)
        else:
            c, s = math.cos(yaw), math.sin(yaw)
            reach = self.cube_half * (abs(nx * c + ny * s) + abs(-nx * s + ny * c))
        if depth < reach:
            return (
                f"cube overlapping the barrier (its footprint reaches "
                f"{reach - depth:.3f} m past the robot-side edge)"
            )
        return None

    def barrier_clearance(self, pose: Sequence[float]) -> float | None:
        """How far the base's footprint at ``pose`` lies past the barrier's robot-side
        face (negative: across it), or None without a barrier."""
        if self.barrier is None:
            return None
        return min(self.barrier.depth(c) for c in footprint(pose))

    def pose_problem(
        self,
        pose: Sequence[float],
        bin_fit: BinFit | None = None,
        required: float | None = None,
    ) -> str | None:
        """Why the base cannot stand at ``pose``, or None if it can. ``required``: the
        barrier clearance to keep, if not ``clearance`` (a drive that starts inside
        the margin keeps its start's)."""
        x, y = pose[0], pose[1]
        fb = self.floor_bounds
        if not (fb[0] <= x <= fb[1] and fb[2] <= y <= fb[3]):
            return f"base centre ({x:+.3f}, {y:+.3f}) outside the drivable box {fb}"
        corners = footprint(pose)
        depth = self.barrier_clearance(pose)
        need = self.clearance if required is None else required
        if depth is not None and depth < need:
            return (
                f"base footprint {depth:+.3f} m from the barrier's robot-side "
                f"face, across the barrier (must stay {need:.3f} m clear)"
            )
        if bin_fit is not None and _rectangles_overlap(
            corners, _bin_corners(bin_fit, self.clearance)
        ):
            return "base footprint overlaps the bin"
        return None

    def path_problem(
        self,
        start: Sequence[float],
        goal: Sequence[float],
        bin_fit: BinFit | None = None,
    ) -> str | None:
        """Why the base cannot drive from ``start`` to ``goal``
        (:meth:`plan_path`), or None."""
        return self.plan_path(start, goal, bin_fit)[1]

    def plan_path(
        self,
        start: Sequence[float],
        goal: Sequence[float],
        bin_fit: BinFit | None = None,
    ) -> tuple[list[tuple[float, float, float]], str | None]:
        """The straight legs from ``start`` to ``goal`` ([goal], [back-off, goal], or a
        grid route's [waypoint, ..., goal]) and None, or ([], why the drive is
        refused).

        Straight to the goal when :meth:`straight_problem` allows it. Otherwise,
        when the base is too near the barrier to turn freely (a turn towards it swings
        a corner closer: run9 at the far toss stand), first back straight away from
        the barrier along its normal with no turn (:meth:`back_off_pose`), then drive
        on; both legs must pass the full check. Only when neither works (the straight
        line crosses the bin: 2026-11-08 15:32) is a bent route searched for
        (:meth:`grid_route`); every one of its legs passes :meth:`straight_problem`.
        A goal the base may not stand at is refused without a search.
        """
        goal_pose = (float(goal[0]), float(goal[1]), float(goal[2]))
        direct = self.straight_problem(start, goal_pose, bin_fit)
        if direct is None:
            return [goal_pose], None
        refusal = direct
        back = self.back_off_pose(start)
        if back is not None:
            where = f"backing straight off the barrier first, to ({back[0]:+.3f}, "
            where += f"{back[1]:+.3f})"
            problem = self.straight_problem(start, back, bin_fit)
            if problem is None:
                problem = self.straight_problem(back, goal_pose, bin_fit)
                if problem is None:
                    return [back, goal_pose], None
                where += ", then"
            else:
                where += ", is refused too"
            refusal = f"{direct}; {where}: {problem}"
        if self.pose_problem(goal_pose, bin_fit) is not None:
            return [], refusal  # no route ends at a pose the base may not stand at
        legs, why = self.grid_route(start, goal_pose, bin_fit)
        if legs:
            return legs, None
        return [], f"{refusal}; {why}"

    def grid_route(  # pylint: disable=too-many-locals
        self,
        start: Sequence[float],
        goal: Sequence[float],
        bin_fit: BinFit | None = None,
    ) -> tuple[list[tuple[float, float, float]], str]:
        """A bent drive from ``start`` to ``goal`` in as few straight legs as the
        straight-drive check allows, or ([], why there is none).

        The base is holonomic, so the search is in (x, y) only: A* over a
        ``GRID_CELL`` grid of the drivable box, 8-connected, Euclidean heuristic, ties
        broken by cell index. A cell is free when the circle the footprint turns within
        (``_TURN_RADIUS``) keeps ``clearance`` from the barrier and stays off the bin
        grown by ``clearance``: the footprint is then valid there at *every* heading, so
        the base may turn anywhere along the route. That is conservative (the base fits
        through narrower gaps at the right heading) but it never misleads the search
        about turning.

        The start and the goal need not be free cells (both are often near the barrier
        or the bin). The route enters the free cells by a straight leg holding the
        start heading, and leaves them by one already at the goal heading: pure
        translations, each checked exactly (:meth:`_sweep_clear`, the whole swept
        footprint) for every free cell within ``_ATTACH_RADIUS``. Between the first
        cell and the last the heading turns evenly with distance.

        The cell path is then cut into legs greedily (the farthest waypoint first).
        Every leg returned passes :meth:`straight_problem` at its default resolution
        *and* sampled at least every 1.25 cm and 0.025 rad (:meth:`_leg_problem`),
        turning as it goes exactly as a drive turns.
        """
        cell = GRID_CELL
        fb = self.floor_bounds
        xs = fb[0] + cell * np.arange(math.floor((fb[1] - fb[0]) / cell + 1e-9) + 1)
        ys = fb[2] + cell * np.arange(math.floor((fb[3] - fb[2]) / cell + 1e-9) + 1)
        free = self._turn_free(xs, ys, bin_fit)
        rows = len(ys)
        gx, gy = float(goal[0]), float(goal[1])

        def xy(k: int) -> tuple[float, float]:
            return float(xs[k // rows]), float(ys[k % rows])

        def h(k: int) -> float:
            x, y = xy(k)
            return math.hypot(gx - x, gy - y)

        def attach(
            here: Sequence[float], heading: float, into: bool
        ) -> dict[int, float]:
            """Free cells within reach of ``here`` by a clear translation at
            ``heading`` (into the grid from ``here``, or out of it to ``here``), and
            their distances."""
            need = self.clearance
            depth = self.barrier_clearance(here)
            if into and depth is not None and 0.0 <= depth < self.clearance:
                need = depth - 1e-9  # as straight_problem: a start inside the margin
            found = {}
            a_s = (np.abs(xs - here[0]) <= _ATTACH_RADIUS).nonzero()[0]
            b_s = (np.abs(ys - here[1]) <= _ATTACH_RADIUS).nonzero()[0]
            for a in a_s:
                for b in b_s:
                    d = math.hypot(xs[a] - here[0], ys[b] - here[1])
                    if not free[a, b] or d > _ATTACH_RADIUS:
                        continue
                    point = (float(xs[a]), float(ys[b]))
                    if self._sweep_clear(here, point, heading, bin_fit, need):
                        found[int(a) * rows + int(b)] = d
            return found

        entries = attach(start, float(start[2]), True)
        exits = attach(goal, float(goal[2]), False)
        no_route = f"no route round on the {cell:.2f} m grid"
        if not entries or not exits:
            end = "from the start into" if not entries else "out of them to the goal"
            return [], (
                f"{no_route} (no clear straight leg {end} the cells where the base "
                f"can turn)"
            )
        # (f, h, node, parent, g): node _GOAL is the goal, parent _START the start.
        heap = [(d + h(k), h(k), k, _START, d) for k, d in entries.items()]
        heapq.heapify(heap)
        parents: dict[int, int] = {}
        best = dict(entries)
        while heap:
            _, _, k, parent, g = heapq.heappop(heap)
            if k == _GOAL:
                return self._shortcut(start, goal, parents, parent, xy, bin_fit)
            if k in parents:
                continue
            parents[k] = parent
            if k in exits:
                heapq.heappush(heap, (g + exits[k], 0.0, _GOAL, k, g + exits[k]))
            a, b = divmod(k, rows)
            for da, db, step in _NEIGHBOURS:
                na, nb = a + da, b + db
                if not (0 <= na < len(xs) and 0 <= nb < rows and free[na, nb]):
                    continue
                n = na * rows + nb
                ng = g + step * cell
                if n in parents or ng >= best.get(n, math.inf) - 1e-12:
                    continue
                best[n] = ng
                heapq.heappush(heap, (ng + h(n), h(n), n, k, ng))
        return [], (
            f"{no_route} ({len(parents)} cells where the base can turn reached from "
            f"the start, none with a clear straight leg to the goal)"
        )

    def _turn_free(
        self,
        xs: NDArray[np.float64],
        ys: NDArray[np.float64],
        bin_fit: BinFit | None,
    ) -> NDArray[np.bool_]:
        """Which grid centres the base may stand at, and turn on the spot at: the
        circle its footprint turns within keeps ``clearance`` off the barrier's
        robot-side face and the bin (:meth:`grid_route`)."""
        x, y = np.meshgrid(xs, ys, indexing="ij")
        free = np.ones(x.shape, dtype=bool)
        if self.barrier is not None:
            free &= self.barrier.depths(x, y) - _TURN_RADIUS >= self.clearance + 1e-6
        if bin_fit is not None:
            c, s = math.cos(bin_fit.yaw), math.sin(bin_fit.yaw)
            dx, dy = x - bin_fit.x, y - bin_fit.y
            u = np.abs(c * dx + s * dy) - (0.5 * bin_fit.length + self.clearance)
            v = np.abs(-s * dx + c * dy) - (0.5 * bin_fit.width + self.clearance)
            gap = np.hypot(np.maximum(u, 0.0), np.maximum(v, 0.0))
            free &= gap > _TURN_RADIUS + 1e-6
        return free

    def _sweep_clear(  # pylint: disable=too-many-arguments,too-many-positional-arguments
        self,
        a: Sequence[float],
        b: Sequence[float],
        heading: float,
        bin_fit: BinFit | None,
        need: float,
    ) -> bool:
        """Whether the base translating from ``a`` to ``b`` (x, y) at a fixed
        ``heading`` stays valid the whole way, not just at samples: its centre in the
        drivable box (convex, so the ends suffice), every point of the swept footprint
        ``need`` past the barrier's face (depth is linear, so the two footprints'
        corners suffice), and the swept footprint (the convex hull of the two) off
        the bin grown by ``clearance`` (separating axes: both rectangles' and the
        normal to the motion)."""
        fb = self.floor_bounds
        for p in (a, b):
            if not (fb[0] <= p[0] <= fb[1] and fb[2] <= p[1] <= fb[3]):
                return False
        corners = np.vstack(
            [footprint((a[0], a[1], heading)), footprint((b[0], b[1], heading))]
        )
        if self.barrier is not None:
            if float(self.barrier.depths(corners[:, 0], corners[:, 1]).min()) < need:
                return False
        if bin_fit is None:
            return True
        box = _bin_corners(bin_fit, self.clearance)
        c, s = math.cos(heading), math.sin(heading)
        cb, sb = math.cos(bin_fit.yaw), math.sin(bin_fit.yaw)
        axes = [(c, s), (-s, c), (cb, sb), (-sb, cb)]
        if math.hypot(b[0] - a[0], b[1] - a[1]) > 1e-12:
            axes.append((-(b[1] - a[1]), b[0] - a[0]))
        for axis in axes:
            pa, pb = corners @ axis, box @ axis
            if pa.max() < pb.min() or pb.max() < pa.min():
                return True
        return False

    def _shortcut(  # pylint: disable=too-many-arguments,too-many-positional-arguments
        self,
        start: Sequence[float],
        goal: Sequence[float],
        parents: dict[int, int],
        last: int,
        xy: Callable[[int], tuple[float, float]],
        bin_fit: BinFit | None,
    ) -> tuple[list[tuple[float, float, float]], str]:
        """The A* cell path (``parents`` back from ``last``) as waypoints, then as
        few legs as :meth:`_leg_problem` passes (:meth:`grid_route`)."""
        cells = [last]
        while parents[cells[-1]] != _START:
            cells.append(parents[cells[-1]])
        cells.reverse()
        points = [xy(k) for k in cells]
        # The heading turns from the start's to the goal's evenly with distance
        # between the first cell (still the start heading) and the last (the goal's).
        run = [0.0]
        for p, q in zip(points, points[1:]):
            run.append(run[-1] + math.hypot(q[0] - p[0], q[1] - p[1]))
        turn = math.atan2(math.sin(goal[2] - start[2]), math.cos(goal[2] - start[2]))
        waypoints: list[tuple[float, float, float]] = [
            (float(start[0]), float(start[1]), float(start[2]))
        ]
        if run[-1] < 1e-9:  # one cell: turn on the spot there
            waypoints += [
                (points[0][0], points[0][1], float(start[2])),
                (points[0][0], points[0][1], float(start[2]) + turn),
            ]
        else:
            waypoints += [
                (p[0], p[1], float(start[2]) + turn * r / run[-1])
                for p, r in zip(points, run)
            ]
        waypoints.append((float(goal[0]), float(goal[1]), float(goal[2])))
        legs: list[tuple[float, float, float]] = []
        i = 0
        while i < len(waypoints) - 1:
            for j in range(len(waypoints) - 1, i, -1):
                if self._leg_problem(waypoints[i], waypoints[j], bin_fit) is None:
                    break
            else:
                problem = self._leg_problem(waypoints[i], waypoints[i + 1], bin_fit)
                return [], f"the grid route's leg is refused: {problem}"
            legs.append(waypoints[j])
            i = j
        return legs, ""

    def _leg_problem(
        self,
        start: Sequence[float],
        goal: Sequence[float],
        bin_fit: BinFit | None,
    ) -> str | None:
        """:meth:`straight_problem` at its default resolution, then (a multiple of it,
        so every default sample is among them) sampled at least every 1.25 cm along
        the leg and 0.025 rad of its turn."""
        problem = self.straight_problem(start, goal, bin_fit)
        if problem is not None:
            return problem
        dtheta = math.atan2(math.sin(goal[2] - start[2]), math.cos(goal[2] - start[2]))
        length = math.hypot(goal[0] - start[0], goal[1] - start[1])
        k = max(1, math.ceil(length / 0.25), math.ceil(abs(dtheta) / 0.5))
        return None if k == 1 else self.straight_problem(start, goal, bin_fit, 20 * k)

    def straight_problem(
        self,
        start: Sequence[float],
        goal: Sequence[float],
        bin_fit: BinFit | None = None,
        steps: int = 20,
    ) -> str | None:
        """Why the base cannot drive straight from ``start`` to ``goal`` (turning as it
        goes), or None: every pose after the start must be valid (the start is wherever
        the robot already is). A start inside the barrier's margin (but not across)
        may move as long as no pose comes closer to the barrier than the start."""
        required = None
        here = self.barrier_clearance(start)
        if here is not None and 0.0 <= here < self.clearance:
            required = here - 1e-9
        dtheta = math.atan2(math.sin(goal[2] - start[2]), math.cos(goal[2] - start[2]))
        for k in range(1, steps + 1):
            f = k / steps
            pose = (
                start[0] + f * (goal[0] - start[0]),
                start[1] + f * (goal[1] - start[1]),
                start[2] + f * dtheta,
            )
            problem = self.pose_problem(pose, bin_fit, required)
            if problem is not None:
                return f"on the way to ({goal[0]:+.3f}, {goal[1]:+.3f}): {problem}"
        return None

    def back_off_pose(
        self, start: Sequence[float]
    ) -> tuple[float, float, float] | None:
        """``start`` moved straight away from the barrier along its normal, heading
        unchanged, until the base can turn on the spot and keep ``clearance`` (plus
        ``BACK_OFF_SLACK`` for the drive's tracking error); None when there is no
        barrier, the base is across it, or it can already turn freely."""
        here = self.barrier_clearance(start)
        if self.barrier is None or here is None or here < 0.0:
            return None
        need = self.clearance + _TURN_RADIUS + BACK_OFF_SLACK
        distance = need - self.barrier.depth(start[:2])
        if distance <= 1e-3:
            return None
        nx, ny = self.barrier.robot_side
        return (
            float(start[0] + distance * nx),
            float(start[1] + distance * ny),
            float(start[2]),
        )

    def valid(self, bin_fit: BinFit | None = None) -> Callable[[Sequence[float]], bool]:
        """A predicate for base poses (for ``top_down_pick.standoff_pose``)."""
        return lambda pose: self.pose_problem(pose, bin_fit) is None

    @classmethod
    def from_config(cls, cfg: Any, floor_bounds: Sequence[float]) -> Workspace:
        """From a ``barrier`` block (``enabled``, ``center``, ..., ``clearance``)."""
        fb = tuple(float(v) for v in floor_bounds)
        assert len(fb) == 4
        if cfg is None or not cfg.get("enabled", False):
            return cls(floor_bounds=fb)  # type: ignore[arg-type]
        return cls(
            floor_bounds=fb,  # type: ignore[arg-type]
            barrier=Barrier.from_config(cfg),
            clearance=float(cfg.get("clearance", 0.05)),
            cube_half=float(cfg.get("cube_half", 0.0)),
        )


def footprint(pose: Sequence[float]) -> NDArray[np.float64]:
    """The base's four footprint corners (4, 2) at map pose (x, y, heading)."""
    return _rectangle(pose[0], pose[1], pose[2], BASE_LENGTH, BASE_WIDTH)


def _rectangle(
    x: float, y: float, yaw: float, length: float, width: float
) -> NDArray[np.float64]:
    c, s = math.cos(yaw), math.sin(yaw)
    local = np.array([[1, 1], [1, -1], [-1, -1], [-1, 1]]) * [length / 2, width / 2]
    return np.column_stack(
        (x + c * local[:, 0] - s * local[:, 1], y + s * local[:, 0] + c * local[:, 1])
    )


def _bin_corners(bin_fit: BinFit, grow: float) -> NDArray[np.float64]:
    return _rectangle(
        bin_fit.x,
        bin_fit.y,
        bin_fit.yaw,
        bin_fit.length + 2 * grow,
        bin_fit.width + 2 * grow,
    )


def _rectangles_overlap(a: NDArray[np.float64], b: NDArray[np.float64]) -> bool:
    """Separating-axis test for two convex quadrilaterals."""
    for poly in (a, b):
        for i in range(4):
            edge = poly[(i + 1) % 4] - poly[i]
            axis = np.array([-edge[1], edge[0]])
            pa, pb = a @ axis, b @ axis
            if pa.max() < pb.min() or pb.max() < pa.min():
                return False
    return True


def aimed_toss_pose(
    bin_fit: BinFit,
    standoff: float,
    direction: float,
    workspace: Workspace,
) -> Pose:
    """The base pose ``standoff`` from the bin centre along azimuth ``direction`` (map
    frame, bin centre -> robot), facing the bin.

    Raises ``ValueError`` if the base may not stand there.
    """
    x = bin_fit.x + standoff * math.cos(direction)
    y = bin_fit.y + standoff * math.sin(direction)
    heading = math.atan2(bin_fit.y - y, bin_fit.x - x)
    pose = (x, y, heading)
    problem = workspace.pose_problem(pose, bin_fit)
    if problem is not None:
        raise ValueError(
            f"No toss from direction {math.degrees(direction):+.0f} deg at "
            f"{standoff:.2f} m: {problem}."
        )
    return pose


def choose_toss_direction(
    bin_fit: BinFit,
    standoff: float,
    robot: Sequence[float],
    workspace: Workspace,
) -> float:
    """The azimuth (bin -> robot) nearest the robot's current bearing from the bin whose
    toss pose is valid, swinging round in 15-degree steps as
    ``top_down_pick.standoff_pose`` does.

    Raises ``ValueError`` if none is.
    """
    direct = math.atan2(robot[1] - bin_fit.y, robot[0] - bin_fit.x)
    for k in range(13):
        for sign in (1,) if k in (0, 12) else (1, -1):
            direction = direct + sign * math.radians(15 * k)
            try:
                aimed_toss_pose(bin_fit, standoff, direction, workspace)
            except ValueError:
                continue
            return math.atan2(math.sin(direction), math.cos(direction))
    raise ValueError(
        f"No valid toss pose {standoff:.2f} m from the bin at "
        f"({bin_fit.x:+.3f}, {bin_fit.y:+.3f})."
    )


def plan_aimed_toss(  # pylint: disable=too-many-arguments,too-many-positional-arguments
    bin_fit: BinFit,
    standoff: float,
    speed: float,
    release_ms: float,
    direction: float | None,
    robot: Sequence[float],
    workspace: Workspace,
    bounds: RealTossBounds,
) -> Pose:
    """Everything a toss is checked against before anything moves: the parameters
    against ``bounds``, the toss pose (along ``direction``, or the one nearest the
    robot's bearing when None) against ``workspace`` and the bin, and the straight drive
    there from ``robot``.

    Returns the toss pose; raises ``ValueError``.
    """
    bounds.validate(standoff, speed, release_ms)
    if direction is None:
        direction = choose_toss_direction(bin_fit, standoff, robot, workspace)
    pose = aimed_toss_pose(bin_fit, standoff, direction, workspace)
    problem = workspace.path_problem(robot, pose, bin_fit)
    if problem is not None:
        raise ValueError(f"Toss pose unreachable: {problem}.")
    return pose
