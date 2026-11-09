"""Localize a bottle lying on the floor from the ceiling cameras, by its label.

The bottle is clear plastic, which from the ceiling is barely distinguishable from the
floor, but its label is a saturated red band (2026-11-02, a 20 oz vitaminwater: hue
177-179 and 0-2, saturation ~245, against a floor at hue ~10 and saturation ~92). So
the label is what is found and fitted: a cylinder of the bottle's diameter and the
label's length lying on the floor, whose pose has three degrees of freedom (x, y and
the axis's yaw, which repeats every 180 degrees). Its silhouette in a calibrated camera
is approximated by the convex hull of the bounding box's eight corners, as for the
cube and the bin; the fit maximises its IoU with the label mask, averaged over frames.

The fitted point is the label's centre, which is where a grasp across the bottle
should close. A standing bottle is not handled: from above its label is a thin ring.
"""

from __future__ import annotations

import math
import time
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import cv2 as cv
import numpy as np
from numpy.typing import NDArray

from prpl_tidybot.cube_localizer import (
    CeilingCamera,
    in_calibrated_region,
    load_ceiling_cameras,
)
from prpl_tidybot.real_sim.perceivers.target_source import TargetSource

# Unit box corners, centred in x/y, bottom face on z = 0.
_CORNERS = np.array(
    [[sx, sy, sz] for sx in (-0.5, 0.5) for sy in (-0.5, 0.5) for sz in (0.0, 1.0)]
)
_SEAM_PX = 8
# A blob's area relative to the label's silhouette at its spot (either yaw).
_AREA_RATIO = (0.4, 2.0)
# Pixels of floor kept around the label's silhouette when fitting.
_WINDOW_PX = 40
# Two candidates closer than this (m) are the same bottle.
_SAME_M = 0.1


@dataclass(frozen=True)
class BottleModel:
    """The bottle and its label (m), and the label's colour (OpenCV HSV).

    The defaults are a 20 oz vitaminwater: ~0.20 m long and 0.066 m across (retail
    listing), the label measured 0.10 m long in the calibrated ceiling camera. Red
    wraps around hue 0, so ``hue_ranges`` holds one band on each side of it.
    """

    length: float = 0.20
    diameter: float = 0.066
    label_length: float = 0.10
    hue_ranges: tuple[tuple[int, int], ...] = ((0, 8), (170, 180))
    min_saturation: int = 120
    min_value: int = 60


@dataclass(frozen=True)
class BottleFit:
    """A fitted label pose (map frame) and how well its silhouette matched."""

    x: float
    y: float
    # The bottle's axis, radians in [-pi/2, pi/2): a label looks the same end to end.
    yaw: float
    iou: float
    # Per-camera fits (camera index, x, y, yaw, iou) and their spread.
    per_camera: tuple[tuple[int, float, float, float, float], ...] = ()
    camera_disagreement_m: float = 0.0
    # The camera whose fit was kept (-1 when none was chosen).
    camera: int = -1


def bottle_mask(
    bgr: NDArray[np.uint8], model: BottleModel = BottleModel()
) -> NDArray[np.uint8]:
    """1 inside the convex hull of each blob of the label's colour (its print filled)."""
    hsv: Any = cv.cvtColor(bgr, cv.COLOR_BGR2HSV)
    hue = hsv[:, :, 0]
    mask: Any = np.zeros(hue.shape, bool)
    for lo, hi in model.hue_ranges:
        mask |= (hue >= lo) & (hue <= hi)
    mask &= (hsv[:, :, 1] >= model.min_saturation) & (hsv[:, :, 2] >= model.min_value)
    # Bridges the label's black band when it crosses the label and splits the red in
    # two (2026-11-06: a 4 px gap, scoring 0.636 as two hulls).
    mask = cv.morphologyEx(mask.astype(np.uint8), cv.MORPH_CLOSE, np.ones((9, 9)))
    contours, _ = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    filled = np.zeros_like(mask)
    # Each blob's convex hull: the label's dark print and white stripe can reach its
    # edge (2026-11-05: a third of the label), so they are notches, not holes.
    hulls = [cv.convexHull(c) for c in contours]
    cv.drawContours(filled, hulls, -1, (1,), thickness=cv.FILLED)
    return filled.astype(np.uint8)


def label_silhouette(
    camera: CeilingCamera, x: float, y: float, yaw: float, model: BottleModel
) -> NDArray[np.float64]:
    """The image polygon (convex hull) of the label of a bottle lying at pose."""
    c, s = math.cos(yaw), math.sin(yaw)
    local = _CORNERS * [model.label_length, model.diameter, model.diameter]
    world = np.c_[
        x + c * local[:, 0] - s * local[:, 1],
        y + s * local[:, 0] + c * local[:, 1],
        local[:, 2],
    ]
    hull = cv.convexHull(camera.project(world).astype(np.float32))
    return hull.reshape(-1, 2).astype(np.float64)


def _split(frame: NDArray[np.uint8], num_cameras: int) -> list[NDArray[np.uint8]]:
    rows = frame.shape[0] // num_cameras
    return [frame[i * rows : (i + 1) * rows] for i in range(num_cameras)]


def _valid_rows(rows: int, cols: int, index: int, num: int) -> NDArray[np.float32]:
    """1 except at the stitch line(s) of camera ``index``, which are unknown."""
    valid = np.ones((rows, cols), np.float32)
    if index > 0:
        valid[:_SEAM_PX] = 0
    if index < num - 1:
        valid[rows - _SEAM_PX :] = 0
    return valid


# Inside the bin the label is in the walls' shadow: near black (HSV value ~18 on
# 2026-11-05) and mostly hidden, so it is found as dark, saturated red in the bin's
# opening rather than fitted. The rug's pink stays below this saturation.
_IN_BIN_MIN_SATURATION = 150
_IN_BIN_MIN_VALUE = 10
_IN_BIN_MIN_PIXELS = 30
# Inside a bin the label's red stops at hue 4: a cardboard box's shaded inner wall
# reads hue 8 at saturation ~198 (2026-11-08), where the label's band used to end.
_IN_BIN_MAX_LOW_HUE = 4


# A bottle standing in the bin (or leaning cap-up on a wall) hides its whole label
# behind the walls; only its grey cap shows (2026-11-08: saturation 75-150, value
# 28-69, ~155 px), inside an opening whose shadowed blue reads saturation > 210 and
# value < 20. So when no label is seen, anything in the bin's own blue outline that is
# neither blue nor dark counts as the bottle. Its outline is eroded so the rim's
# highlights stay out (empty bins at both spots on 2026-11-08: 0 px over 99 frames).
# A bin almost straight under a camera shows its own lit floor instead, a brighter and
# slightly washed-out blue (17:33: saturation 156-169, value 122-139), so the cap is
# also capped in brightness.
_BIN_HUE = (100, 130)
_BIN_MIN_SATURATION = 150
_OBJECT_MAX_SATURATION = 170
_OBJECT_MIN_VALUE = 25
_OBJECT_MAX_VALUE = 100
_OBJECT_ERODE_PX = 6
_OBJECT_MIN_PIXELS = 60


def _in_bin_masks(
    hsv: Any,
    region: NDArray[np.uint8],
    model: BottleModel,
    min_value: float = _IN_BIN_MIN_VALUE,
) -> tuple[NDArray[np.bool_], NDArray[np.bool_]]:
    """(label-red pixels, non-bin pixels inside the bin's eroded blue outline) of one
    image, within ``region``."""
    red = np.zeros(region.shape, bool)
    for lo, hi in model.hue_ranges:
        top = min(hi, _IN_BIN_MAX_LOW_HUE) if lo == 0 else hi
        red |= (hsv[:, :, 0] >= lo) & (hsv[:, :, 0] <= top)
    red &= (hsv[:, :, 1] >= _IN_BIN_MIN_SATURATION) & (hsv[:, :, 2] >= min_value)
    red &= region > 0
    blue = (
        (hsv[:, :, 0] >= _BIN_HUE[0])
        & (hsv[:, :, 0] <= _BIN_HUE[1])
        & (hsv[:, :, 1] >= _BIN_MIN_SATURATION)
        & (region > 0)
    )
    other = np.zeros(region.shape, bool)
    count, labels, stats, _ = cv.connectedComponentsWithStats(blue.astype(np.uint8))
    if count > 1:
        largest = 1 + int(np.argmax(stats[1:, cv.CC_STAT_AREA]))
        outline: Any = np.zeros(region.shape, np.uint8)
        hull = cv.convexHull(cv.findNonZero((labels == largest).astype(np.uint8)))
        cv.fillConvexPoly(outline, hull, (1,))
        size = 2 * _OBJECT_ERODE_PX + 1
        outline = cv.erode(outline, np.ones((size, size), np.uint8))
        other = (
            (outline > 0)
            & (hsv[:, :, 1] < _OBJECT_MAX_SATURATION)
            & (hsv[:, :, 2] > _OBJECT_MIN_VALUE)
            & (hsv[:, :, 2] < _OBJECT_MAX_VALUE)
        )
    return red, other


def _largest_blob(mask: NDArray[np.bool_]) -> NDArray[np.int64]:
    """The (row, col) pixels of the largest connected blob of ``mask``."""
    count, labels, stats, _ = cv.connectedComponentsWithStats(mask.astype(np.uint8))
    if count < 2:
        return np.zeros((0, 2), np.int64)
    largest = 1 + int(np.argmax(stats[1:, cv.CC_STAT_AREA]))
    return np.argwhere(labels == largest)


def find_bottle_in_box(  # pylint: disable=too-many-locals
    frames: Sequence[NDArray[np.uint8]],
    cameras: Sequence[CeilingCamera],
    model: BottleModel,
    box: Sequence[float],
    rim_height: float,
    min_value: float = _IN_BIN_MIN_VALUE,
) -> tuple[float, float, float] | None:
    """The bottle's (x, y, yaw) inside the floor box ``box`` (x_min, x_max, y_min,
    y_max; the bin), or None: label-red pixels seen through the box's top at
    ``rim_height``, from the camera that sees the most, cast down to the bottle's axis
    height. The yaw is the red pixels' long axis on the floor (0 for a round blob).
    ``min_value`` is the label's darkest red: a cardboard box's creases read label-red
    at value 13-37, the blue bin's shaded label lower still.

    With no label in view, a blob of anything but the bin inside the bin's opening
    (a standing bottle's cap) is the bottle, cast down from the bottle's standing
    height, yaw 0."""
    corners = np.array(
        [[x, y, rim_height] for x in box[:2] for y in box[2:]], dtype=np.float64
    )
    best: tuple[int, int, NDArray[np.int64]] | None = None
    # The camera and blob of the largest object seen, for when no label is.
    other_camera, other_blob = -1, np.zeros((0, 2), np.int64)
    for i, camera in enumerate(cameras):
        imgs = [_split(f, len(cameras))[i] for f in frames]
        rows, cols = imgs[0].shape[:2]
        region = np.zeros((rows, cols), np.uint8)
        hull = cv.convexHull(camera.project(corners).astype(np.float32)).astype(
            np.int32
        )
        cv.fillConvexPoly(region, hull, (1,))
        region &= _valid_rows(rows, cols, i, len(cameras)).astype(np.uint8)
        votes = np.zeros((rows, cols), np.float32)
        other_votes = np.zeros((rows, cols), np.float32)
        for img in imgs:
            red, other = _in_bin_masks(
                cv.cvtColor(img, cv.COLOR_BGR2HSV), region, model, min_value
            )
            votes += red
            other_votes += other
        pixels = np.argwhere(votes >= len(imgs) / 2)
        if len(pixels) >= _IN_BIN_MIN_PIXELS and (
            best is None or len(pixels) > best[1]
        ):
            best = (i, len(pixels), pixels)
        blob = _largest_blob(other_votes >= len(imgs) / 2)
        if len(blob) >= max(_OBJECT_MIN_PIXELS, len(other_blob) + 1):
            other_camera, other_blob = i, blob
    if best is None:
        if other_camera < 0:
            return None
        camera = cameras[other_camera]
        floor = np.array(
            [camera.to_floor(float(u), float(v), model.length) for v, u in other_blob]
        )
        x, y = floor.mean(axis=0)
        if not (box[0] <= x <= box[1] and box[2] <= y <= box[3]):
            return None
        return float(x), float(y), 0.0
    camera = cameras[best[0]]
    floor = np.array(
        [camera.to_floor(float(u), float(v), model.diameter / 2) for v, u in best[2]]
    )
    x, y = floor.mean(axis=0)
    spread = np.cov((floor - [x, y]).T)
    evals, evecs = np.linalg.eigh(spread)
    yaw = 0.0
    if evals[1] > 2 * max(evals[0], 1e-12):
        yaw = math.atan2(evecs[1, 1], evecs[0, 1])
        yaw = (yaw + math.pi / 2) % math.pi - math.pi / 2
    if not (box[0] <= x <= box[1] and box[2] <= y <= box[3]):
        return None
    return float(x), float(y), float(yaw)


def _excluded(x: float, y: float, exclude: Sequence[Sequence[float]]) -> bool:
    return any(abs(x - ex) < half and abs(y - ey) < half for ex, ey, half in exclude)


@dataclass(frozen=True)
class RobotBox:
    """The robot's base as the ceiling cameras see it (m): its footprint, centred on
    the marker pose and turned with its heading, from the floor up to its top.

    The TidyBot++ base is 0.548 m along its heading and 0.508 m across
    (``aimed_toss.BASE_LENGTH`` / ``BASE_WIDTH``); its top, where the markers sit, is
    0.378 m up (``marker_detector.constants.ROBOT_HEIGHT``). Red cabling runs on it up
    to the top, and a camera looking down at an angle sees it up to 0.25 m beyond the
    footprint (2026-11-06), so a blob is the robot's when the camera's ray to it
    passes through this box, not when its floor point lies in the footprint.
    """

    length: float = 0.548
    width: float = 0.508
    height: float = 0.378
    margin: float = 0.03


def _camera_centre(camera: CeilingCamera) -> NDArray[np.float64]:
    rot: Any = cv.Rodrigues(camera.rvec)[0]
    return np.asarray(-rot.T @ camera.tvec, np.float64).reshape(3)


def _behind_robot(  # pylint: disable=too-many-arguments,too-many-positional-arguments
    camera: CeilingCamera,
    x: float,
    y: float,
    z: float,
    robots: Sequence[Sequence[float]],
    box: RobotBox,
) -> bool:
    """Whether the camera's ray to the map point (x, y, z) passes through the base of
    a robot at any of ``robots`` (map x, y, heading): the point is the robot's own,
    or hidden by it."""
    centre = _camera_centre(camera)
    half_l, half_w = box.length / 2 + box.margin, box.width / 2 + box.margin
    # The ray's floor-plane point at heights 0 and box.height (it is a straight line).
    ends = [
        centre[:2] + (np.array([x, y]) - centre[:2]) * (centre[2] - h) / (centre[2] - z)
        for h in (0.0, box.height)
    ]
    for rx, ry, heading in robots:
        c, s = math.cos(heading), math.sin(heading)
        a, b = (
            np.array(
                [c * (p[0] - rx) + s * (p[1] - ry), -s * (p[0] - rx) + c * (p[1] - ry)]
            )
            for p in ends
        )
        # Clip the segment a -> b to the rectangle (Liang-Barsky).
        lo, hi = 0.0, 1.0
        for k, half in ((0, half_l), (1, half_w)):
            d = b[k] - a[k]
            for num, den in ((half - a[k], d), (half + a[k], -d)):
                if den == 0:
                    if num < 0:
                        lo, hi = 1.0, 0.0
                elif den > 0:
                    hi = min(hi, num / den)
                else:
                    lo = max(lo, num / den)
        if lo <= hi:
            return True
    return False


def find_bottle_candidates(
    frame: NDArray[np.uint8],
    cameras: Sequence[CeilingCamera],
    floor_bounds: Sequence[float],
    model: BottleModel,
    exclude: Sequence[Sequence[float]] = (),
    robots: Sequence[Sequence[float]] = (),
    robot_box: RobotBox = RobotBox(),
) -> list[tuple[float, float]]:
    """Map-frame (x, y) of every label-coloured, label-sized blob in ``floor_bounds``,
    largest first; none the robots' bases (``robots``: map x, y, heading) hide."""
    found: list[tuple[float, float, float]] = []
    for i, (cam, img) in enumerate(zip(cameras, _split(frame, len(cameras)))):
        rows, cols = img.shape[:2]
        mask = bottle_mask(img, model)
        mask *= _valid_rows(rows, cols, i, len(cameras)).astype(np.uint8)
        num, _, stats, centroids = cv.connectedComponentsWithStats(mask)
        for k in range(1, num):
            u, v = centroids[k]
            x, y = cam.to_floor(float(u), float(v), model.diameter / 2)
            if not (
                floor_bounds[0] <= x <= floor_bounds[1]
                and floor_bounds[2] <= y <= floor_bounds[3]
            ) or _excluded(x, y, exclude):
                continue
            if _behind_robot(cam, x, y, model.diameter / 2, robots, robot_box):
                continue
            expected = [
                cv.contourArea(
                    label_silhouette(cam, x, y, yaw, model).astype(np.float32)
                )
                for yaw in (0.0, math.pi / 2)
            ]
            area = float(stats[k, cv.CC_STAT_AREA])
            if _AREA_RATIO[0] * min(expected) <= area <= _AREA_RATIO[1] * max(expected):
                found.append((area, float(x), float(y)))
    candidates: list[tuple[float, float]] = []
    for _, x, y in sorted(found, reverse=True):
        if all(math.hypot(x - cx, y - cy) > _SAME_M for cx, cy in candidates):
            candidates.append((x, y))
    return candidates


def fit_bottle_pose(  # pylint: disable=too-many-locals
    frames: Sequence[NDArray[np.uint8]],
    camera: CeilingCamera,
    index: int,
    num_cameras: int,
    guess_xy: tuple[float, float],
    model: BottleModel,
) -> BottleFit | None:
    """3-DoF silhouette fit of the label in one camera around ``guess_xy``; masks are
    averaged over ``frames`` first."""
    imgs = [_split(f, num_cameras)[index] for f in frames]
    rows, cols = imgs[0].shape[:2]
    u, v = camera.project(np.array([[*guess_xy, model.diameter / 2]]))[0]
    if not (0 <= u < cols and 0 <= v < rows):
        return None
    u0, v0 = max(int(u) - _WINDOW_PX, 0), max(int(v) - _WINDOW_PX, 0)
    u1, v1 = min(int(u) + _WINDOW_PX, cols), min(int(v) + _WINDOW_PX, rows)
    prob = np.mean([bottle_mask(img, model)[v0:v1, u0:u1] for img in imgs], axis=0)
    valid = _valid_rows(rows, cols, index, num_cameras)[v0:v1, u0:u1]
    prob = prob * valid
    if prob.sum() == 0:
        return None

    def score(x: float, y: float, yaw: float) -> float:
        sil = np.zeros(prob.shape, np.float32)
        poly = ((label_silhouette(camera, x, y, yaw, model) - [u0, v0]) * 16).astype(
            np.int32
        )
        cv.fillConvexPoly(sil, poly, (1.0,), lineType=cv.LINE_AA, shift=4)
        sil = sil * valid
        inter = float((sil * prob).sum())
        union = float(prob.sum()) + float(sil.sum()) - inter
        return inter / union if union > 0 else 0.0

    best_score, best = -1.0, np.zeros(3)
    for yaw in np.deg2rad(np.arange(-90.0, 90.0, 7.5)):
        for dx in np.arange(-0.03, 0.0301, 0.01):
            for dy in np.arange(-0.03, 0.0301, 0.01):
                cand = np.array([guess_xy[0] + dx, guess_xy[1] + dy, yaw])
                sc = score(*cand)
                if sc > best_score:
                    best_score, best = sc, cand
    step = np.array([0.004, 0.004, math.radians(3.0)])
    while step[0] > 2e-4:  # coordinate descent with shrinking steps
        improved = False
        for k in range(3):
            for sign in (-1.0, 1.0):
                cand = best.copy()
                cand[k] += sign * step[k]
                sc = score(*cand)
                if sc > best_score:
                    best_score, best, improved = sc, cand, True
        if not improved:
            step = step / 2
    yaw = (best[2] + math.pi / 2) % math.pi - math.pi / 2
    return BottleFit(float(best[0]), float(best[1]), float(yaw), best_score)


def _off_nadir(camera: CeilingCamera, x: float, y: float) -> float:
    """Horizontal distance from the point under the camera to (x, y)."""
    rot: Any = cv.Rodrigues(camera.rvec)[0]
    centre = -rot.T @ camera.tvec
    return float(math.hypot(x - centre[0], y - centre[1]))


def localize_bottle(  # pylint: disable=too-many-arguments,too-many-positional-arguments
    frames: Sequence[NDArray[np.uint8]],
    cameras: Sequence[CeilingCamera],
    model: BottleModel,
    floor_bounds: Sequence[float],
    exclude: Sequence[Sequence[float]] = (),
    min_iou: float = 0.6,
    serials: Sequence[str] | None = None,
    robots: Sequence[Sequence[float]] = (),
    robot_box: RobotBox = RobotBox(),
) -> BottleFit | None:
    """The best-fitting bottle label in ``floor_bounds``, or None if nothing fits well
    enough.

    Each camera is fitted on its own (their calibrations disagree by a few cm where
    they overlap) and, as for the bin, a near tie goes to a camera whose calibrated
    region holds the label, then to the camera that sees it more directly from above.
    ``exclude`` holds map-frame squares (x, y, half side); ``robots`` the poses (x, y,
    heading) of robots whose base (``robot_box``) hides what is behind it.
    """
    best: BottleFit | None = None
    for guess in find_bottle_candidates(
        frames[-1], cameras, floor_bounds, model, exclude, robots, robot_box
    ):
        singles = [
            (i, fit)
            for i, cam in enumerate(cameras)
            if (fit := fit_bottle_pose(frames, cam, i, len(cameras), guess, model))
            is not None
            and fit.iou >= min_iou
        ]
        if not singles:
            continue
        pool = singles
        if serials is not None:
            calibrated = [
                (i, f)
                for i, f in singles
                if in_calibrated_region(cameras[i], i, f.x, f.y, serials)
            ]
            pool = calibrated or singles
        top_iou = max(f.iou for _, f in pool)
        near = [(i, f) for i, f in pool if f.iou >= top_iou - 0.03]
        cam_i, chosen = min(
            near, key=lambda t: _off_nadir(cameras[t[0]], t[1].x, t[1].y)
        )
        if not (
            floor_bounds[0] <= chosen.x <= floor_bounds[1]
            and floor_bounds[2] <= chosen.y <= floor_bounds[3]
        ) or _excluded(chosen.x, chosen.y, exclude):
            continue
        if _behind_robot(
            cameras[cam_i], chosen.x, chosen.y, model.diameter / 2, robots, robot_box
        ):
            continue
        spread = max(
            (math.hypot(a.x - b.x, a.y - b.y) for _, a in singles for _, b in singles),
            default=0.0,
        )
        fit = BottleFit(
            chosen.x,
            chosen.y,
            chosen.yaw,
            chosen.iou,
            tuple((i, f.x, f.y, f.yaw, f.iou) for i, f in singles),
            spread,
            cam_i,
        )
        if best is None or fit.iou > best.iou:
            best = fit
    return best


class CeilingBottleSource(TargetSource):
    """The bottle's label from the ceiling cameras, as ``CeilingCubeTargetSource`` is
    the cube.

    Reads ``num_frames`` stitched frames and the robot's map pose (to ignore the robot's
    own footprint), then runs :func:`localize_bottle`; raises if no bottle is found.
    ``get_target`` is the label's centre at the bottle's axis height, ``get_yaw`` the
    bottle's axis.
    """

    def __init__(  # pylint: disable=too-many-arguments,too-many-positional-arguments
        self,
        serials: Sequence[str],
        image_client: Any = None,
        marker_client: Any = None,
        floor_bounds: Sequence[float] = (-1.5, 1.7, -1.5, 1.5),
        length: float = 0.20,
        diameter: float = 0.066,
        label_length: float = 0.10,
        # Beyond the base's own footprint at its heading (see RobotBox).
        robot_margin: float = 0.03,
        num_frames: int = 10,
        frame_period_s: float = 0.12,
        min_iou: float = 0.6,
        host: str | None = None,
        in_bin_rim_height: float = 0.384,
        in_bin_min_value: float = _IN_BIN_MIN_VALUE,
    ) -> None:
        if image_client is None:
            from prpl_tidybot.marker_detector.ceiling_image_client import (  # pylint: disable=import-outside-toplevel
                CeilingImageClient,
            )

            image_client = CeilingImageClient(host) if host else CeilingImageClient()
        if marker_client is None:
            from prpl_tidybot.marker_detector.client import (  # pylint: disable=import-outside-toplevel
                MarkerDetectorClient,
            )

            marker_client = (
                MarkerDetectorClient(host) if host else MarkerDetectorClient()
            )
        self._images = image_client
        self._markers = marker_client
        self._serials = list(serials)
        self._cameras = load_ceiling_cameras(self._serials)
        self._model = BottleModel(float(length), float(diameter), float(label_length))
        self._bounds = tuple(float(v) for v in floor_bounds)
        self._robot_box = RobotBox(margin=float(robot_margin))
        self._num_frames = int(num_frames)
        self._period = float(frame_period_s)
        self._min_iou = float(min_iou)
        self._rim_height = float(in_bin_rim_height)
        self._in_bin_min_value = float(in_bin_min_value)
        self.last_fit: BottleFit | None = None

    def _frames(self) -> list[NDArray[np.uint8]]:
        frames: list[NDArray[np.uint8]] = []
        deadline = time.time() + 10 * max(self._period, 0.1) * self._num_frames
        while len(frames) < self._num_frames and time.time() < deadline:
            img = self._images.get_image()
            if img is not None and (not frames or img is not frames[-1]):
                frames.append(np.array(img, copy=True))
            if self._period:
                time.sleep(self._period)
        if not frames:
            raise RuntimeError("CeilingBottleSource: no ceiling image arrived")
        return frames

    def robot_poses(self) -> list[tuple[float, float, float]]:
        """The robot's map pose (x, y, heading) now, from the marker detector."""
        # The detector answers each request with the pose current when the request
        # arrived, and the client requests at the end of every call, so the first
        # answer after a drive is from before it (2026-11-05: 0.44 m stale, which hid
        # a bottle behind the robot). The second answer is to a request made now.
        self._markers.get_latest()
        poses = self._markers.get_latest().get("poses") or {}
        return [(float(p[0]), float(p[1]), float(p[2])) for p in poses.values()]

    def localize(self) -> BottleFit:
        """Run the localizer now and return the fit (raises if there is no bottle)."""
        fit = localize_bottle(
            self._frames(),
            self._cameras,
            self._model,
            self._bounds,
            min_iou=self._min_iou,
            serials=self._serials,
            robots=self.robot_poses(),
            robot_box=self._robot_box,
        )
        if fit is None:
            raise RuntimeError(
                f"CeilingBottleSource: no bottle found in the floor box {self._bounds}"
            )
        self.last_fit = fit
        return fit

    def find_pose_in_box(
        self, box: Sequence[float]
    ) -> tuple[float, float, float] | None:
        """The bottle's (x, y, yaw) inside the floor box ``box`` (the bin), or None (see
        :func:`find_bottle_in_box`); the skill server looks here before the floor."""
        return find_bottle_in_box(
            self._frames()[-3:],
            self._cameras,
            self._model,
            box,
            self._rim_height,
            self._in_bin_min_value,
        )

    def get_target(self) -> tuple[float, float, float]:
        fit = self.localize()
        return (fit.x, fit.y, self._model.diameter / 2)

    def get_yaw(self) -> float:
        """The bottle's axis from the last fit (localizes first if there is none)."""
        fit = self.last_fit if self.last_fit is not None else self.localize()
        return fit.yaw

    def alternates(self) -> list[tuple[float, float, float]]:
        """The other cameras' estimates of the bottle from the last localization, most
        confident first, where they differ from the chosen one by over 1 cm: what the
        pick retries at after a miss, as for the cube."""
        fit = self.last_fit
        if fit is None:
            return []
        others = sorted(
            (p for p in fit.per_camera if p[0] != fit.camera), key=lambda p: -p[4]
        )
        return [
            (x, y, yaw)
            for _, x, y, yaw, _ in others
            if math.hypot(x - fit.x, y - fit.y) > 0.01
        ]

    def close(self) -> None:
        """Close the image and marker clients."""
        self._images.close()
        self._markers.close()
