"""Tests for the ceiling-camera bottle localizer: a bottle lying on the floor, found by
its red label (the clear plastic barely shows from the ceiling) and fitted as a
label-sized cylinder, so its pose is the label's centre and the bottle's axis."""

import math
from pathlib import Path

import cv2 as cv
import numpy as np
import pytest

from prpl_tidybot.bottle_localizer import (
    BottleFit,
    BottleModel,
    CeilingBottleSource,
    bottle_mask,
    find_bottle_candidates,
    label_silhouette,
    localize_bottle,
)
from prpl_tidybot.cube_localizer import load_ceiling_cameras

SERIALS = ("AD8A29AE", "5A5C29AE")
H, W = 720, 1280
MODEL = BottleModel()
BOUNDS = (-1.5, 1.7, -1.5, 1.5)


def _hsv_to_bgr(hsv: tuple[int, int, int]) -> tuple[int, int, int]:
    """One OpenCV HSV colour as BGR."""
    px = np.array([[hsv]], np.uint8)
    b, g, r = (int(v) for v in cv.cvtColor(px, cv.COLOR_HSV2BGR)[0, 0])
    return b, g, r


# From a real ceiling frame (2026-11-02): floor H ~10 S ~92 V ~177; the vitaminwater
# label H 177-179 and 0-2 (red wraps around 0) S ~245; the blue bin H ~127.
FLOOR_BGR = _hsv_to_bgr((10, 92, 177))
LABEL_BGR = _hsv_to_bgr((178, 245, 200))
LABEL_LOW_HUE_BGR = _hsv_to_bgr((2, 240, 190))
BIN_BGR = _hsv_to_bgr((127, 170, 110))
CARPET_BGR = (38, 38, 40)


def _quad(cam, x0, x1, y0, y1, z=0.0):
    """Image polygon of a floor rectangle."""
    pts = np.array([[x0, y0, z], [x1, y0, z], [x1, y1, z], [x0, y1, z]], float)
    return cam.project(pts).astype(np.int32)


def _render(cameras, bottles=(), patches=(), seed=0, noise=3.0, colour=LABEL_BGR):
    """A stitched frame: floor, coloured floor patches ((x0, x1, y0, y1), bgr) and
    lying bottles' labels at `bottles` (x, y, yaw)."""
    rng = np.random.default_rng(seed)
    halves = []
    for cam in cameras:
        img = np.empty((H, W, 3), np.float32)
        img[:] = FLOOR_BGR
        for box, bgr in patches:
            cv.fillConvexPoly(img, _quad(cam, *box), bgr, lineType=cv.LINE_AA)
        for x, y, yaw in bottles:
            poly = label_silhouette(cam, x, y, yaw, MODEL).astype(np.int32)
            cv.fillConvexPoly(img, poly, colour, lineType=cv.LINE_AA)
        img += rng.normal(0, noise, img.shape)
        halves.append(np.clip(img, 0, 255).astype(np.uint8))
    return np.vstack(halves)


@pytest.fixture(scope="module", name="cameras")
def _cameras():
    """The lab's calibrated ceiling cameras."""
    return load_ceiling_cameras(SERIALS)


def _axis_err(a, b):
    """A label looks the same end to end: compare axes modulo 180 degrees."""
    return abs((a - b + math.pi / 2) % math.pi - math.pi / 2)


def test_the_model_defaults_to_a_20_oz_bottle():
    """The defaults are the vitaminwater bottle's."""
    assert MODEL.diameter == pytest.approx(0.066)
    assert MODEL.label_length == pytest.approx(0.10)


def test_the_mask_keeps_red_on_both_sides_of_hue_zero_and_nothing_else():
    """Red at hue ~2 and ~178 passes; the floor, the blue bin and a carpet do not."""
    img = np.zeros((40, 200, 3), np.uint8)
    colours = (FLOOR_BGR, BIN_BGR, CARPET_BGR, LABEL_BGR, LABEL_LOW_HUE_BGR)
    for k, bgr in enumerate(colours):
        img[:, 40 * k : 40 * (k + 1)] = bgr
    mask = bottle_mask(img)
    assert mask[:, :120].sum() == 0
    assert mask[5:35, 125:195].all()


@pytest.mark.parametrize(
    "pose", [(0.37, -0.67, 1.18), (-0.8, 0.9, 0.0), (1.2, -1.1, -0.7)]
)
def test_a_lying_bottle_is_localized(cameras, pose):
    """A label rendered at a known pose through the real calibration is recovered."""
    frames = [_render(cameras, bottles=[pose], seed=s) for s in range(3)]
    fit = localize_bottle(frames, cameras, MODEL, BOUNDS, serials=SERIALS)
    assert fit is not None
    assert math.hypot(fit.x - pose[0], fit.y - pose[1]) < 0.015
    assert _axis_err(fit.yaw, pose[2]) < math.radians(5)


def test_a_bottle_beside_the_blue_bin_on_the_carpet_is_found(cameras):
    """The blue bin and a dark carpet next to the bottle do not disturb the fit."""
    pose = (0.6, 1.2, 0.4)
    patches = [((-1.2, 1.6, 0.35, 2.0), CARPET_BGR), ((0.1, 0.4, 1.1, 1.4), BIN_BGR)]
    frame = _render(cameras, bottles=[pose], patches=patches)
    fit = localize_bottle([frame], cameras, MODEL, BOUNDS, serials=SERIALS)
    assert fit is not None
    assert math.hypot(fit.x - pose[0], fit.y - pose[1]) < 0.015


def test_dark_print_on_the_label_still_counts_as_label(cameras):
    """2026-11-05, after a toss: the label's black panel and white stripe faced the
    ceiling and reached its edge, so red-only pixels covered about two thirds of the
    label and the fit scored 0.5985 against min_iou 0.6. A real frame (everything
    but the bottle blacked out)."""
    path = Path(__file__).parent / "data" / "bottle_dark_print_up_20261105.png"
    frame = cv.imread(str(path))
    fit = localize_bottle([frame], cameras, MODEL, BOUNDS, serials=SERIALS)
    assert fit is not None
    assert math.hypot(fit.x - 0.100, fit.y - (-0.466)) < 0.03
    assert fit.iou >= 0.8


def test_nothing_red_gives_none(cameras):
    """No label, no bottle."""
    frame = _render(cameras, patches=[((0.1, 0.4, 1.1, 1.4), BIN_BGR)])
    assert localize_bottle([frame], cameras, MODEL, BOUNDS) is None


def test_a_large_red_patch_is_not_a_bottle(cameras):
    """A red region much bigger than a label is rejected."""
    frame = _render(cameras, patches=[((-0.5, 0.1, -0.9, -0.5), LABEL_BGR)])
    assert localize_bottle([frame], cameras, MODEL, BOUNDS) is None


def test_the_robot_footprint_is_excluded(cameras):
    """A label inside the robot's footprint is ignored."""
    frame = _render(cameras, bottles=[(0.5, -0.6, 0.0)])
    fit = localize_bottle([frame], cameras, MODEL, BOUNDS, exclude=[(0.5, -0.6, 0.45)])
    assert fit is None


class _Images:
    def __init__(self, frames):
        self._frames = list(frames)

    def get_image(self):
        """The next frame, then the last one forever."""
        return self._frames.pop(0) if len(self._frames) > 1 else self._frames[0]

    def close(self):
        """Nothing to release."""


class _Markers:
    def get_latest(self):
        """The robot, well away from the bottle."""
        return {"poses": {"robot": (-1.0, -1.0, 0.0)}}

    def close(self):
        """Nothing to release."""


def test_ceiling_bottle_source_gives_target_and_yaw(cameras):
    """The source returns the label's centre at axis height and the bottle's axis."""
    pose = (0.37, -0.67, 1.18)
    frames = [_render(cameras, bottles=[pose], seed=s) for s in range(2)]
    source = CeilingBottleSource(
        SERIALS,
        image_client=_Images(frames),
        marker_client=_Markers(),
        floor_bounds=BOUNDS,
        num_frames=2,
        frame_period_s=0.0,
    )
    x, y, z = source.get_target()
    assert math.hypot(x - pose[0], y - pose[1]) < 0.015
    assert z == pytest.approx(MODEL.diameter / 2)
    assert _axis_err(source.get_yaw(), pose[2]) < math.radians(5)


def test_ceiling_bottle_source_raises_without_a_bottle(cameras):
    """No bottle in view raises rather than inventing a target."""
    source = CeilingBottleSource(
        SERIALS,
        image_client=_Images([_render(cameras)]),
        marker_client=_Markers(),
        floor_bounds=BOUNDS,
        num_frames=1,
        frame_period_s=0.0,
    )
    with pytest.raises(RuntimeError):
        source.get_target()


def test_alternates_are_the_other_cameras_estimates(cameras):
    """After a miss the pick retries at the other camera's estimate, as for the cube,
    when it differs from the chosen one by over 1 cm."""
    del cameras
    source = CeilingBottleSource(
        SERIALS,
        image_client=_Images([np.zeros((2 * H, W, 3), np.uint8)]),
        marker_client=_Markers(),
        floor_bounds=BOUNDS,
    )
    assert not source.alternates()
    source.last_fit = BottleFit(
        0.30,
        0.0,
        0.4,
        0.8,
        ((1, 0.30, 0.0, 0.4, 0.8), (0, 0.33, 0.01, 0.45, 0.7)),
        0.03,
        1,
    )
    assert source.alternates() == [(0.33, 0.01, 0.45)]
    source.last_fit = BottleFit(
        0.30,
        0.0,
        0.4,
        0.8,
        ((1, 0.30, 0.0, 0.4, 0.8), (0, 0.305, 0.0, 0.4, 0.7)),
        0.005,
        1,
    )
    assert not source.alternates()


# The far bin on 2026-11-05 at 11:10 (observe()'s fit), as `_bin_box` gives it.
FAR_BIN = (0.24375, 1.3125, 0.0196, 0.365, 0.26)


def _bin_box_of(x, y, yaw, length, width):
    c, s = math.cos(yaw), math.sin(yaw)
    hx = 0.5 * (abs(length * c) + abs(width * s))
    hy = 0.5 * (abs(length * s) + abs(width * c))
    return [x - hx, x + hx, y - hy, y + hy]


@pytest.mark.parametrize(
    ("name", "inside"), [("in", True), ("empty", False), ("live", True)]
)
def test_a_bottle_in_the_bin_is_found_by_its_dark_label(cameras, name, inside):
    """2026-11-05 11:10: a toss landed in the blue bin and observe() said no bottle.
    In the bin's shadow the label is near black (HSV value ~18, the floor search needs
    60) and its walls hide most of it, so the floor fit cannot score it. The bin's
    opening is searched for dark, saturated red instead. Frames from the ceiling
    video (half resolution, upscaled), everything but the bin blacked out: the bottle
    in the bin, and the same bin empty 31 s earlier; and a full-resolution live frame
    with the bottle lying in the far bin (13:4x), which the floor search missed."""
    path = Path(__file__).parent / "data" / f"bottle_in_far_bin_{name}_20261105.png"
    source = CeilingBottleSource(
        SERIALS,
        image_client=_Images([cv.imread(str(path))]),
        marker_client=_Markers(),
        floor_bounds=BOUNDS,
        num_frames=1,
        frame_period_s=0.0,
    )
    found = source.find_pose_in_box(_bin_box_of(*FAR_BIN))
    if not inside:
        assert found is None
        return
    assert found is not None
    x, y, _yaw = found
    c, s = math.cos(-FAR_BIN[2]), math.sin(-FAR_BIN[2])
    along = c * (x - FAR_BIN[0]) - s * (y - FAR_BIN[1])
    across = s * (x - FAR_BIN[0]) + c * (y - FAR_BIN[1])
    assert abs(along) < FAR_BIN[3] / 2 and abs(across) < FAR_BIN[4] / 2


class _PipelinedMarkers:
    """The marker detector's protocol: each get_latest() returns the answer to the
    request sent at the end of the previous call, computed when it was sent."""

    def __init__(self, pose):
        self.pose = pose
        self._answer = {"poses": {"robot": pose}}

    def get_latest(self):
        """The pending (possibly stale) answer; then request again, now."""
        answer, self._answer = self._answer, {"poses": {"robot": self.pose}}
        return answer

    def close(self):
        """Nothing to release."""


def test_the_robot_is_excluded_where_it_is_now_not_where_it_was(cameras):
    """2026-11-05 14:46: after the dump the bottle lay behind the robot in plain view,
    but the search excluded a square around where the robot had stood for the toss
    before (0.44 m away): the marker client's first answer after a drive is from
    before it. The exclusion uses a fresh answer."""
    old, new = (1.0, -0.6, 0.0), (0.55, -0.6, 0.0)
    bottle = (1.25, -0.55, 0.2)
    markers = _PipelinedMarkers(old)
    source = CeilingBottleSource(
        SERIALS,
        image_client=_Images([_render(cameras, bottles=[bottle])]),
        marker_client=markers,
        floor_bounds=BOUNDS,
        num_frames=1,
        frame_period_s=0.0,
    )
    markers.get_latest()  # an earlier call left a request pending with `old`
    markers.pose = new  # then the robot drove away
    x, y, _ = source.get_target()
    assert math.hypot(x - bottle[0], y - bottle[1]) < 0.02


class _FixedMarkers:
    """The robot standing still at ``pose`` (map x, y, heading)."""

    def __init__(self, pose):
        self.pose = pose

    def get_latest(self):
        """The robot's pose."""
        return {"poses": {0: self.pose}}

    def close(self):
        """Nothing to release."""


FULL_FLOOR = (-1.846, 2.046, -1.846, 1.846)


def _source(frame, robot_pose):
    return CeilingBottleSource(
        SERIALS,
        image_client=_Images([frame]),
        marker_client=_FixedMarkers(robot_pose),
        floor_bounds=FULL_FLOOR,
        min_iou=0.7,
        num_frames=1,
        frame_period_s=0.0,
    )


# observe() #164 and #13 on 2026-11-06 (09-47-01 logs): the robot's marker pose.
ROBOT_AT_1043 = (1.0701, -0.6282, 3.1368)
ROBOT_AT_0952 = (1.0109, -0.6189, -3.0910)


def test_a_bottle_just_beside_the_base_is_found(cameras):
    """2026-11-06 10:43, after a toss: the bottle lay 8 cm beyond the base's front and
    10 cm beyond its side, 0.35 m from its centre on each map axis, so inside the
    0.40 m square the search used to exclude around the robot whatever its heading,
    and observe() said no bottle. The base's own rectangle at its heading leaves it
    in view. A frame from the ceiling video (half resolution, upscaled: its IoU is not
    the live frame's), everything but the robot and the bottle blacked out."""
    del cameras
    path = Path(__file__).parent / "data" / "bottle_beside_the_robot_20261106.png"
    x, y, _ = _source(cv.imread(str(path)), ROBOT_AT_1043).get_target()
    assert math.hypot(x - 0.711, y + 0.981) < 0.03


@pytest.mark.parametrize(
    ("name", "pose"),
    [
        ("bottle_beside_the_robot_20261106.png", ROBOT_AT_1043),
        ("robot_alone_20261106.png", ROBOT_AT_0952),
    ],
)
def test_the_robots_own_red_is_still_ignored(cameras, name, pose):
    """The base carries red cabling up to its top (0.378 m). From the ceiling it
    lands, mapped to the bottle's height, up to 0.25 m beyond the base's footprint
    (the cameras look down at an angle), so the flat footprint does not cover it:
    no red blob near the robot in two real frames is a candidate, except the bottle
    itself (the frames are the test above's and the robot alone at 09:52)."""
    frame = cv.imread(str(Path(__file__).parent / "data" / name))
    robots = _source(frame, pose).robot_poses()
    found = find_bottle_candidates(frame, cameras, FULL_FLOOR, MODEL, robots=robots)
    near = [(x, y) for x, y in found if math.hypot(x - pose[0], y - pose[1]) < 0.8]
    assert all(math.hypot(x - 0.711, y + 0.981) < 0.05 for x, y in near)


@pytest.mark.parametrize("heading", [0.0, 0.5, 1.2, -2.3])
def test_the_base_is_excluded_at_its_heading(cameras, heading):
    """A label on the base is ignored and one 6 cm beyond its side is found, at any
    heading: what is excluded is the base's own rectangle, turned with it. The robot
    stands under the bottom camera, where the base hides the least floor."""
    rx, ry = 0.0, -0.9
    c, s = math.cos(heading), math.sin(heading)

    def at(along, across):
        return rx + c * along - s * across, ry + s * along + c * across

    on = (*at(0.12, 0.0), heading)
    beside = (*at(0.0, 0.254 + 0.06 + MODEL.diameter / 2), heading)
    robots = [(rx, ry, heading)]
    frame = _render(cameras, bottles=[on])
    assert localize_bottle([frame], cameras, MODEL, BOUNDS, robots=robots) is None
    frame = _render(cameras, bottles=[beside])
    fit = localize_bottle([frame], cameras, MODEL, BOUNDS, robots=robots)
    assert fit is not None
    assert math.hypot(fit.x - beside[0], fit.y - beside[1]) < 0.02


def test_a_label_split_in_two_by_its_dark_band_is_filled_as_one(cameras):
    """2026-11-06 09:34, after a toss: the bottle lay with the label's black band
    across the middle, splitting the red into two blobs. Each blob's own hull left the
    band out, so the fit scored 0.636 against min_iou 0.7 and the bottle was lost. A
    real frame (everything but the bottle blacked out)."""
    path = (
        Path(__file__).parent / "data" / "bottle_label_split_by_dark_band_20261106.png"
    )
    frame = cv.imread(str(path))
    fit = localize_bottle([frame], cameras, MODEL, BOUNDS, min_iou=0.7, serials=SERIALS)
    assert fit is not None
    assert math.hypot(fit.x - 0.43, fit.y - 0.25) < 0.03
    assert fit.iou >= 0.75
