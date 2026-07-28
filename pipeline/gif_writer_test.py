"""Tests for gif_writer."""

import numpy as np
import pytest
from PIL import Image, ImageSequence

from pipeline.gif_writer import Canvas, Padded, WriteGif


def _frame(height: int, width: int, value: int = 0) -> np.ndarray:
    frame = np.full((height, width, 3), 255, dtype=np.uint8)
    frame[1:-1, 1:-1] = value
    return frame


def _delays(path) -> list[int]:
    """Each frame's delay in milliseconds, decoding the file as a player
    would rather than reading a header count.
    """
    with Image.open(path) as gif:
        return [frame.info["duration"] for frame in ImageSequence.Iterator(gif)]


def test_writes_a_readable_animation(tmp_path):
    path = tmp_path / "out.gif"
    WriteGif(str(path), [_frame(20, 30, value) for value in (0, 60, 120)], milliseconds_per_frame=40)

    with Image.open(path) as gif:
        assert gif.format == "GIF"
        assert gif.size == (30, 20)
    assert _delays(path) == [40, 40, 40]


def test_frames_of_different_sizes_share_one_canvas(tmp_path):
    """The case that forced padding to exist: the search steps up a bin
    size partway through, so the frames genuinely differ.
    """
    path = tmp_path / "out.gif"
    WriteGif(str(path), [_frame(20, 30), _frame(24, 50)], milliseconds_per_frame=40)

    with Image.open(path) as gif:
        assert gif.size == (50, 24)


def test_identical_frames_collapse_into_one_longer_one(tmp_path):
    """Which is what makes holding on a final result nearly free - and the
    reason a file holds fewer frames than were handed over.
    """
    path = tmp_path / "out.gif"
    WriteGif(str(path), [_frame(20, 30)] * 5, milliseconds_per_frame=40)

    assert _delays(path) == [200]


def test_a_delay_below_the_gif_quantum_is_refused(tmp_path):
    with pytest.raises(ValueError, match="less than 10ms"):
        WriteGif(str(tmp_path / "out.gif"), [_frame(20, 30)], milliseconds_per_frame=5)


def test_delays_are_rounded_to_the_quantum(tmp_path):
    """A GIF stores hundredths of a second, so rounding here is what keeps
    the written delay the one that was asked for rather than one a decoder
    invents.
    """
    path = tmp_path / "out.gif"
    WriteGif(str(path), [_frame(20, 30), _frame(20, 30, 90)], milliseconds_per_frame=64)

    assert _delays(path) == [60, 60]


def test_an_empty_animation_is_refused(tmp_path):
    with pytest.raises(ValueError, match="at least one frame"):
        WriteGif(str(tmp_path / "out.gif"), [])


def test_a_non_bgr_frame_is_refused(tmp_path):
    grey = np.zeros((20, 30), dtype=np.uint8)
    with pytest.raises(ValueError, match="frame 0 is not an 8-bit BGR image"):
        WriteGif(str(tmp_path / "out.gif"), [grey])


def test_the_canvas_is_the_largest_of_each_dimension():
    # Neither frame is the canvas: one is taller and the other is wider.
    assert Canvas([_frame(20, 50), _frame(30, 40)]) == (30, 50)


def test_padding_anchors_at_the_minimum_corner():
    """So parts stay still when the bin around them grows."""
    frame = _frame(4, 6)
    padded = Padded(frame, 8, 10)

    assert padded.shape == (8, 10, 3)
    assert np.array_equal(padded[:4, :6], frame)
    assert (padded[4:, :] == 255).all()
    assert (padded[:, 6:] == 255).all()


def test_padding_never_shrinks_a_frame():
    with pytest.raises(ValueError, match="larger than the"):
        Padded(_frame(20, 30), 10, 30)
