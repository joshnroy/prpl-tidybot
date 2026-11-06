"""Is the bottle in the hand: one wrist frame at the end of the pick's carry.

The gripper reading cannot tell. It stays where the fingers stopped, so a bottle that
slips out after the close still reads as held (2026-11-06 pick 20: closed on the
bottle's flank at 0.262, inside the 0.2-0.9 band, and the bottle fell out during the
carry), and real holds read 0.293 to 0.616 over 63 picks, so no band separates them.

The wrist camera can. It moves with the fingers, so a held bottle sits in the same
place in every frame: across the gap between the finger pads, at the bottom of the
frame, a few centimetres from the lens. With nothing in the hand that gap shows the
floor about 0.6 m below the carry pose: plain, light and neutral. So the score is how
much of that gap is *not* plain floor: pixels that are pink to red (the label, and the
clear plastic, which looks lavender), dark (the label's black print) or on an edge
(the bottle's outline and highlights, which clear plastic shows even where it has no
colour). It is the smaller of the two halves of the gap, so a bottle off to one side
(on the floor, near the lens at a lower pose) does not count.

On the frames at the end of the carry (the last frame of each pick's wrist clip,
taken within 0.2 s of the carry arriving) of 2026-11-05/06, labelled by eye: 63/63
holds score 0.33 to 1.00 (the lowest is a hold on the clear shoulder, cap and neck in
view, no label; the next lowest 0.58), the slip scores 0.00 on all 39 frames after it
fell, and the 3 closes on nothing (their clips end at the hover with the gripper open
and the bottle on the floor in view) score at most 0.005.
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass
from typing import Any

import cv2 as cv
import numpy as np
from numpy.typing import NDArray

# Every frame is read at this width (the 640 x 480 wrist frame at half size).
_WIDTH = 320
# The gap between the finger pads at the bottom of the frame, as fractions of the
# frame: rows above the gripper's own body, two column strips either side of the
# centre (the pads' black tips sit at about 0.27 and 0.76 of the width).
_ROWS = (0.70, 0.93)
_STRIPS = ((0.33, 0.47), (0.53, 0.67))
# Pink to red: (R + B) / 2 - G above this (the floor reads about -4).
_MAGENTA = 8.0
# Dark: the brightest channel below this (the label's black print).
_DARK = 70.0
# An edge: Sobel magnitude, smoothed over 9 px, above this (none of the slip's 39
# frames after it fell has one in the gap; nearer, the floor's speckle comes close).
_EDGE = 40.0
# Held at this score or above. Holds score 0.33 or more, everything else 0.005 or
# less (see the module docstring).
HOLD_THRESHOLD = 0.15


@dataclass(frozen=True)
class HoldLook:
    """What one end-of-carry frame showed: ``held`` (None when there was no verdict,
    with ``note`` saying why), the ``score`` and the ``threshold`` it was held to."""

    held: bool | None
    score: float | None = None
    threshold: float = HOLD_THRESHOLD
    note: str | None = None

    def as_dict(self) -> dict[str, Any]:
        """For the pick's attempt record and reply (JSON)."""
        return asdict(self)


def hold_score(image_rgb: NDArray[np.uint8]) -> float:
    """The fraction of the gap between the finger pads that is not plain floor (the
    smaller of its two halves), in [0, 1], for an RGB wrist frame."""
    image = np.asarray(image_rgb, np.uint8)
    size = (_WIDTH, round(image.shape[0] * _WIDTH / image.shape[1]))
    small = cv.resize(image, size, interpolation=cv.INTER_AREA)
    small = cv.GaussianBlur(small, (5, 5), 0)
    rgb = small.astype(np.float32)
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    grey = cv.cvtColor(small, cv.COLOR_RGB2GRAY).astype(np.float32)
    edges = np.hypot(cv.Sobel(grey, cv.CV_32F, 1, 0), cv.Sobel(grey, cv.CV_32F, 0, 1))
    bottle = (
        (((r + b) / 2 - g) > _MAGENTA)
        | (rgb.max(axis=2) < _DARK)
        | (np.asarray(cv.GaussianBlur(edges, (9, 9), 0), np.float32) > _EDGE)
    )
    h, w = bottle.shape
    rows = slice(int(h * _ROWS[0]), int(h * _ROWS[1]))
    return min(
        float(bottle[rows, int(w * c0) : int(w * c1)].mean()) for c0, c1 in _STRIPS
    )


def look_for_hold(
    image_rgb: NDArray[np.uint8], threshold: float = HOLD_THRESHOLD
) -> HoldLook:
    """Whether this RGB end-of-carry wrist frame shows the bottle in the hand."""
    score = round(hold_score(image_rgb), 4)
    return HoldLook(score >= threshold, score, threshold)


class HoldCheck:
    """The check at the end of the carry: one wrist frame, `settle_s` after the arm
    stops, read by :func:`look_for_hold`. Never raises: no frame, or a camera or a
    frame that fails, is no verdict (``held`` None) with the reason."""

    def __init__(
        self, source: Any, settle_s: float = 0.3, threshold: float = HOLD_THRESHOLD
    ) -> None:
        self._source = source
        self._settle_s = settle_s
        self._threshold = threshold

    def look(self) -> HoldLook:
        """Whether the bottle is in the hand now."""
        if self._settle_s > 0:
            time.sleep(self._settle_s)
        try:
            image = self._source.get_image()
            if image is None:
                return HoldLook(None, threshold=self._threshold, note="no wrist frame")
            return look_for_hold(image, self._threshold)
        except Exception as e:  # pylint: disable=broad-except
            return HoldLook(
                None,
                threshold=self._threshold,
                note=f"wrist frame unreadable: {type(e).__name__}: {e}",
            )
