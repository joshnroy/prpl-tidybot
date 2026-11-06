"""Tests for prpl_tidybot.wrist_hold: is the bottle in the hand at the end of the carry.

The frames are real wrist frames at the end of the carry (2026-11-05/06, 320 x 240),
labelled by eye from the pick's wrist clip: ``held`` the bottle fills the gap between
the fingers; ``slip`` pick 20 of 2026-11-06 (gripper 0.262, logged held, the bottle
fell during the carry); ``empty`` a close on nothing, its clip ending at the hover
after the gripper opened, the bottle on the floor in view.
"""

from pathlib import Path

import cv2 as cv
import numpy as np
import pytest

from prpl_tidybot.wrist_hold import (
    HOLD_THRESHOLD,
    HoldCheck,
    HoldLook,
    hold_score,
    look_for_hold,
)

_DATA = Path(__file__).parent / "data" / "wrist_hold"
_FRAMES = sorted(_DATA.glob("*.jpg"))


def _rgb(path):
    return cv.cvtColor(cv.imread(str(path)), cv.COLOR_BGR2RGB)


def _frame(kind):
    return next(p for p in _FRAMES if p.stem.endswith(kind))


def test_there_are_frames_of_every_kind():
    """The data covers the slip, an empty close and the hardest holds."""
    kinds = {p.stem.rsplit("_", 1)[1] for p in _FRAMES}
    assert kinds == {"held", "slip", "empty"}
    for name in (
        "pick_020_slip",  # the slip: gripper 0.262, read held
        "pick_003_held",  # the lowest real held reading, 0.293
        "pick_008_held",  # the highest, 0.616
        "pick_034_held",  # held by the clear shoulder: cap and neck, no label
        "pick_028_empty",  # a close on nothing, the bottle near in view
    ):
        assert any(p.stem.endswith(name) for p in _FRAMES), name


@pytest.mark.parametrize("path", _FRAMES, ids=lambda p: p.stem)
def test_a_real_frame_reads_as_labelled(path):
    """Each real end-of-carry frame reads as its by-eye label."""
    look = look_for_hold(_rgb(path))
    assert look.held is (path.stem.endswith("_held")), look
    assert 0.0 <= look.score <= 1.0


def test_the_threshold_leaves_a_margin_on_both_sides():
    """Every held frame scores at least twice the threshold, every other one under
    half of it."""
    scores = {p.stem: hold_score(_rgb(p)) for p in _FRAMES}
    held = [s for k, s in scores.items() if k.endswith("_held")]
    other = [s for k, s in scores.items() if not k.endswith("_held")]
    assert min(held) >= 2 * HOLD_THRESHOLD, scores
    assert max(other) <= HOLD_THRESHOLD / 2, scores


def test_the_score_does_not_depend_on_the_frame_size():
    """The live frame is 640 x 480; the test frames are stored at half size."""
    for path in (_frame("pick_034_held"), _frame("pick_020_slip")):
        small = _rgb(path)
        big = cv.resize(small, (640, 480), interpolation=cv.INTER_LINEAR)
        assert hold_score(big) == pytest.approx(hold_score(small), abs=0.05)


def test_a_plain_floor_is_not_held():
    """Nothing between the fingers but floor."""
    floor = np.full((480, 640, 3), (196, 190, 178), np.uint8)
    look = look_for_hold(floor)
    assert look.held is False and look.score == pytest.approx(0.0)


class _Camera:
    def __init__(self, frame=None, error=None):
        self.frame, self.error, self.reads = frame, error, 0

    def get_image(self):
        """The frame, or the error."""
        self.reads += 1
        if self.error is not None:
            raise self.error
        return self.frame


def test_the_check_reads_one_live_frame():
    """One frame per look, read by look_for_hold."""
    camera = _Camera(_rgb(_frame("pick_020_slip")))
    look = HoldCheck(camera, settle_s=0.0).look()
    assert camera.reads == 1 and look.held is False and look.note is None


def test_no_frame_gives_no_verdict():
    """The fail-safe is the caller's: no frame (or a camera that raises) is no verdict,
    with the reason, never an exception."""
    for camera in (_Camera(None), _Camera(error=RuntimeError("camera gone"))):
        look = HoldCheck(camera, settle_s=0.0).look()
        assert look.held is None and look.score is None and look.note, look


def test_a_broken_frame_gives_no_verdict():
    """An empty array is no verdict, not an exception."""
    look = HoldCheck(_Camera(np.zeros((0, 0, 3), np.uint8)), settle_s=0.0).look()
    assert look.held is None and look.note


def test_the_look_is_a_plain_record():
    """The record the pick reply carries."""
    look = HoldLook(held=True, score=0.5, threshold=0.15)
    assert look.as_dict() == {
        "held": True,
        "score": 0.5,
        "threshold": 0.15,
        "note": None,
    }
