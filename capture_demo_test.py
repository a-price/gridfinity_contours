"""Tests for the capture-window animation.

Split by what they cost. The frame machinery - grabbing a widget, settling
it, holding a frame - is exercised on a bare widget in milliseconds, and
that is where the fiddly parts are. Actually recording the flow loads SAM2
and segments a 5184px photograph, so it is marked slow and asserts only
the things that make the animation worth having: that it shows the real
window, and that the calibration it demonstrates genuinely resolved.

Deliberately no comparison against the committed GIF. That would be the
stored-image assertion `render_demo.py` argues against at length, and it
would fail on every intentional change to how the window looks - which is
most of the changes this animation exists to document.
"""

import os

import numpy as np
import pytest
from PyQt5.QtWidgets import QLabel, QVBoxLayout, QWidget

import capture_demo
from capture_demo import HANDLE, PHOTO, PHOTO_COLORS, Grab, Main, Recording


class _Host(QWidget):
    """A stand-in for the capture window: something with a central layout
    to activate and a label whose text can be changed, which is all
    `Recording` touches.
    """

    def __init__(self) -> None:
        super().__init__()
        self.resize(200, 120)
        self.label = QLabel("before")
        layout = QVBoxLayout(self)
        layout.addWidget(self.label)

    def centralWidget(self) -> QWidget:
        return self


# ------------------------------------------------------------------ frames


def test_a_grab_is_an_8_bit_bgr_image(qapp):
    """What `gif_writer.WriteGif` refuses anything else for."""
    host = _Host()
    host.show()

    frame = Grab(host)

    assert frame.ndim == 3 and frame.shape[2] == 3
    assert frame.dtype == np.uint8
    assert frame.shape[:2] == (host.height(), host.width())


def test_a_grab_is_not_a_view_of_a_reused_qt_buffer(qapp):
    """`QImage.bits()` hands back memory Qt owns and will write over. A
    frame that aliased it would leave every held frame showing whatever
    the window looked like last.
    """
    host = _Host()
    host.show()

    first = Grab(host)
    host.label.setText("after")
    qapp.processEvents()
    second = Grab(host)

    assert not np.array_equal(first, second)


def test_a_grab_has_no_scanline_padding_in_it(qapp):
    """Qt pads scanlines to a 4-byte boundary. Reading the buffer as
    `width * 4` on an odd width shears the picture progressively down the
    frame, which looks like a rendering bug rather than an indexing one.
    """
    host = _Host()
    host.resize(201, 101)  # odd both ways
    host.show()

    frame = Grab(host)

    assert frame.shape[:2] == (101, 201)
    # A sheared grab has the label's text smeared diagonally; an unsheared
    # one keeps the widget's flat background in the bottom corner.
    assert len(np.unique(frame[-1].reshape(-1, 3), axis=0)) == 1


def test_holding_repeats_one_frame(qapp):
    """Repeated rather than paused, so the encoder can collapse the run
    into a single long frame.
    """
    host = _Host()
    host.show()
    recording = Recording(host, qapp)

    recording.Hold(4)

    assert len(recording.frames) == 4
    assert all(np.array_equal(recording.frames[0], frame) for frame in recording.frames)


def test_a_held_frame_shows_the_change_that_preceded_it(qapp):
    """The property the whole recording rests on. Qt lays out and repaints
    lazily, so a grab taken without settling shows the state *before* the
    change - and the animation would be one beat behind itself throughout.
    """
    host = _Host()
    host.show()
    recording = Recording(host, qapp)
    recording.Hold(1)

    host.label.setText("a much longer caption than before")
    recording.Hold(1)

    assert not np.array_equal(recording.frames[0], recording.frames[1])


def test_settling_stops_once_the_window_stops_changing(qapp):
    """Bounded, so a widget that never settles cannot hang `make gifs`."""
    host = _Host()
    host.show()
    recording = Recording(host, qapp)

    assert recording._Settled(passes=1) is not None
    assert np.array_equal(recording._Settled(), recording._Settled())


# ------------------------------------------------------------------ inputs


def test_the_photo_it_names_is_there():
    assert os.path.exists(PHOTO), "the animation names one specific photo"


def test_the_photo_carries_a_calibration_sheet():
    """The reason this demo cannot use just any photo. Without markers,
    `silhouette.py` falls back to pixel space by design - the animation
    would look complete while dropping the one thing the window is for.
    """
    import cv2

    from pipeline.calibration import ArucoCalibration

    calibration = ArucoCalibration()
    calibration.Detect(cv2.imread(PHOTO))

    known = calibration.parameters.marker_positions_mm
    matched = [marker for marker in calibration.detected_corners if marker in known]
    assert len(matched) >= 3, f"only {len(matched)} markers matched a known position"
    assert calibration.GetTransform() is not None


def test_the_click_lands_inside_the_photo():
    import cv2

    height, width = cv2.imread(PHOTO).shape[:2]

    assert 0 <= HANDLE[0] < width and 0 <= HANDLE[1] < height
    assert all(isinstance(value, int) for value in HANDLE), "a real click gives ints; the mask is indexed with them"


def test_the_palette_is_sized_for_a_photograph():
    """The layout animations are line drawings on white, where 16 colours
    is generous. At 16 this one came out uniformly brown with the ArUco
    markers illegible - which would make it evidence for nothing.
    """
    from pipeline.gif_writer import DEFAULT_COLORS

    assert PHOTO_COLORS > DEFAULT_COLORS


# ------------------------------------------------------------- the whole run


@pytest.mark.slow
def test_the_animation_records_the_real_window(tmp_path, qapp):
    """Loads SAM2 and segments a 5184px photograph - about ten seconds.

    What is asserted is what makes it worth having: every frame the same
    size (so the encoder pads nothing and the window does not appear to
    jump), and the flow reaching a millimetre-scale contour rather than
    stopping at a mask.
    """
    from PIL import Image, ImageSequence

    path = str(tmp_path / "capture.gif")
    assert Main(["--out", path]) == 0

    frames = [np.array(frame.convert("RGB")) for frame in ImageSequence.Iterator(Image.open(path))]
    assert len(frames) >= 3, "an empty window, a loaded photo, and a measured contour"
    assert len({frame.shape for frame in frames}) == 1, "a window that changes size reads as a jump"


@pytest.mark.slow
def test_the_flow_ends_in_millimetres(qapp):
    """The claim the animation makes. A capture that ended in pixels would
    look identical and mean nothing - which is exactly what this photo
    having a calibration sheet buys.
    """
    window, _ = capture_demo.Record()

    contours = window.rectify.contours
    assert contours, "selection should have rectified something"
    points = np.asarray(next(iter(contours.values())), dtype=float).reshape(-1, 2)
    extent = points.max(axis=0) - points.min(axis=0)
    assert 100.0 < max(extent) < 400.0, f"a hand tool is decimetres, got {extent} mm"
