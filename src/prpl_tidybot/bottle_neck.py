"""Which end of the lying bottle is toward the top of the wrist camera's frame at the
pick's hover: the neck or the petaloid base.

The ceiling cameras fit the bottle by its red label, which looks the same turned half a
turn, so the pick's closing yaw has two solutions and the bottle was held neck one way
or the other at random (2026-11-05: neck toward the top of the wrist frame in 8/12 hand-
labelled holds). At the hover the frame shows the label in its lower half and the end
toward the top beyond it; the camera and the bottle both turn with the hand, so that end
is also the one toward the top once held.

The 20 oz vitaminwater bottle's clear plastic runs about 9 cm past the label on the neck
side (shoulders, neck and cap) and about 2.5 cm on the base side, against its 6.6 cm
width. Seen from above, clear plastic on the light floor shows as edges: the topmost row
with edges above the label, in the middle half of the label's columns (where the neck
and the cap are), marks the end. Measured in label widths, so it holds at any distance:
neck 1.0 to 1.5 on the 25 hover frames of 2026-11-05, base 0.2 to 0.3.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2 as cv
import numpy as np
from numpy.typing import NDArray

# Every frame is read at this width (the 640 x 480 wrist frame at half size).
_WIDTH = 320
# The label: red (OpenCV hue wraps at 180), saturated, not dark; blobs smaller than
# this (px at _WIDTH) are print or noise.
_RED_HUE = (10, 165)
_MIN_SATURATION = 110
_MIN_VALUE = 70
_MIN_BLOB = 75
# Edge strength (mean Sobel magnitude over the band, rows averaged over 9) that counts
# as bottle rather than floor. On the 25 real frames every one reads right for any
# threshold from 25 to 40 (at 20 the floor's speckle reads as a neck above a base).
_EDGE = 32.0
_SMOOTH_ROWS = 9
# Clear plastic beyond the label, in label widths: the neck runs about 1.4, the base
# about 0.4 (both measured 1.0+ and 0.3- on the real frames).
NECK_MIN_EXTENT = 0.65


@dataclass(frozen=True)
class EndLook:
    """What one hover frame showed: ``end`` ("neck", "base", or None when it could not
    tell, with ``note`` saying why), the clear extent beyond the label in label widths,
    and the label's box (x, y, w, h) in the frame as given."""

    end: str | None
    extent: float | None = None
    label_box: tuple[int, int, int, int] | None = None
    note: str | None = None


def top_end(image_rgb: NDArray[np.uint8]) -> EndLook:
    """Which end of the bottle is toward the top of this RGB wrist frame."""
    image = np.asarray(image_rgb, np.uint8)
    scale = _WIDTH / image.shape[1]
    size = (_WIDTH, round(image.shape[0] * scale))
    small = np.asarray(cv.resize(image, size, interpolation=cv.INTER_AREA), np.uint8)
    box = _label_box(small)
    if box is None:
        return EndLook(None, note="no label in the wrist frame")
    x, y, w, h = box
    full_box = (round(x / scale), round(y / scale), round(w / scale), round(h / scale))
    if y < NECK_MIN_EXTENT * w:
        # A neck the frame cuts off would read as short as a base.
        return EndLook(None, None, full_box, "label too near the top of the frame")
    grey = cv.GaussianBlur(cv.cvtColor(small, cv.COLOR_RGB2GRAY), (5, 5), 0)
    grey = grey.astype(np.float64)
    edges = np.hypot(cv.Sobel(grey, cv.CV_64F, 1, 0), cv.Sobel(grey, cv.CV_64F, 0, 1))
    rows = edges[:y, x + w // 4 : x + 3 * w // 4].mean(axis=1)
    rows = np.convolve(rows, np.ones(_SMOOTH_ROWS) / _SMOOTH_ROWS, mode="same")
    above = np.nonzero(rows > _EDGE)[0]
    top = int(above.min()) if len(above) else y
    extent = (y - top) / w
    end = "neck" if extent >= NECK_MIN_EXTENT else "base"
    return EndLook(end, round(extent, 3), full_box)


def _label_box(image_rgb: NDArray[np.uint8]) -> tuple[int, int, int, int] | None:
    """The box (x, y, w, h) round every red blob big enough to be label."""
    hsv = np.asarray(cv.cvtColor(image_rgb, cv.COLOR_RGB2HSV), np.int16)
    red = (
        ((hsv[..., 0] < _RED_HUE[0]) | (hsv[..., 0] > _RED_HUE[1]))
        & (hsv[..., 1] > _MIN_SATURATION)
        & (hsv[..., 2] > _MIN_VALUE)
    ).astype(np.uint8)
    count, labels, stats, _ = cv.connectedComponentsWithStats(red)
    keep = [k for k in range(1, count) if stats[k, cv.CC_STAT_AREA] >= _MIN_BLOB]
    if not keep:
        return None
    ys, xs = np.nonzero(np.isin(labels, keep))
    x, y, w, h = cv.boundingRect(np.column_stack([xs, ys]).astype(np.int32))
    return int(x), int(y), int(w), int(h)
