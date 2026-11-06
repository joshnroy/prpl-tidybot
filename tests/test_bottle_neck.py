"""Tests for the wrist camera's bottle-end check at the pick's hover: which end of the
lying bottle is toward the top of the image, the neck or the petaloid base.

The frames are the real hover frames of every bottle pick filmed on 2026-11-05 (25
picks, sessions 10-46-22 to 13-37-48), downscaled to 320 x 240; the file name ends in
the end that is at the top of the image. Josh labelled 12 of them from the held frame
(11-05-57 #1 #2, 13-03-14 #1 #2 #4 #8 #10 #11 #16 #20, 13-37-48 #1 #2); the other 13
were labelled the same way from their clips' last frames. In every clip the end at the
top at the hover is the end at the top once held: the camera and the bottle both turn
with the hand.
"""

from pathlib import Path

import cv2 as cv
import numpy as np
import pytest

from prpl_tidybot.bottle_neck import top_end

_DATA = Path(__file__).parent / "data" / "bottle_neck_hover"
_FRAMES = sorted(_DATA.glob("*.jpg"))


def _rgb(path):
    return cv.cvtColor(cv.imread(str(path)), cv.COLOR_BGR2RGB)


def test_there_are_frames_of_both_ends():
    """All 25 picks, both ends represented."""
    ends = [p.stem.rsplit("_", 1)[1] for p in _FRAMES]
    assert len(ends) == 25
    assert ends.count("neck") == 15 and ends.count("base") == 10


@pytest.mark.parametrize("path", _FRAMES, ids=lambda p: p.stem)
def test_the_end_at_the_top_of_a_real_hover_frame(path):
    """The neck's shoulders and cap run about 1.4 bottle widths beyond the label, the
    base under 0.4: the clear plastic seen above the label tells them apart."""
    look = top_end(_rgb(path))
    assert look.end == path.stem.rsplit("_", 1)[1], look


def test_the_check_does_not_depend_on_the_frame_size():
    """The full-size wrist frame (640 x 480) reads the same as the stored one."""
    path = _FRAMES[0]
    big = cv.resize(_rgb(path), (640, 480), interpolation=cv.INTER_LINEAR)
    assert top_end(big).end == top_end(_rgb(path)).end


def test_no_label_no_answer():
    """Bare floor: no label, so no end is named (the pick then does not turn)."""
    floor = np.full((240, 320, 3), (200, 196, 188), np.uint8)
    look = top_end(floor)
    assert look.end is None and look.note


def test_a_label_at_the_top_edge_has_nothing_above_it_to_read():
    """A label near the top of the frame leaves too little above it to tell a short
    base from a neck the frame cuts off: no answer."""
    img = _rgb(_FRAMES[0])
    shifted = np.full_like(img, 200)
    shifted[:85] = img[155:240]
    look = top_end(shifted)
    assert look.end is None and look.note
