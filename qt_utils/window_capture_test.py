"""Tests for photographing a live Qt window.

The fiddly parts of both window demos live here, and they are cheap: a
bare widget, no model, no search. What they are guarding is that a grab
is a *faithful* picture - right pixel format, not aliasing a buffer Qt
will reuse, not sheared, and not taken one layout pass too early. Each of
those produced a plausible-looking wrong picture at some point, which is
the failure mode that makes recorded documentation worse than none.
"""

import numpy as np
from PyQt5.QtWidgets import QLabel, QMainWindow, QVBoxLayout, QWidget

from qt_utils.window_capture import Grab, Settled


class _Host(QWidget):
    """A stand-in for a real window: a layout to activate and a label
    whose text can change, which is all the grabber touches.
    """

    def __init__(self, width: int = 200, height: int = 120) -> None:
        super().__init__()
        self.resize(width, height)
        self.label = QLabel("before")
        layout = QVBoxLayout(self)
        layout.addWidget(self.label)


# -------------------------------------------------------------------- grabs


def test_a_grab_is_an_8_bit_bgr_image(qapp):
    """What `cv2.imwrite` and `gif_writer.WriteGif` both refuse anything
    else for.
    """
    host = _Host()
    host.show()

    frame = Grab(host)

    assert frame.ndim == 3 and frame.shape[2] == 3
    assert frame.dtype == np.uint8
    assert frame.shape[:2] == (host.height(), host.width())


def test_a_grab_is_not_a_view_of_a_reused_qt_buffer(qapp):
    """`QImage.bits()` hands back memory Qt owns and will write over. A
    frame that aliased it would leave every previously-taken picture
    showing whatever the window looked like last.
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
    frame, which reads as a rendering bug rather than an indexing one.
    """
    host = _Host(width=201, height=101)  # odd both ways
    host.show()

    frame = Grab(host)

    assert frame.shape[:2] == (101, 201)
    # A sheared grab has the label's text smeared diagonally; an unsheared
    # one keeps the widget's flat background in the bottom corner.
    assert len(np.unique(frame[-1].reshape(-1, 3), axis=0)) == 1


# ------------------------------------------------------------------ settling


def test_settling_shows_the_change_that_preceded_it(qapp):
    """The property every recorded picture rests on. Qt lays out and
    repaints lazily, so a grab taken without settling shows the state
    *before* the change - and an animation would be one beat behind
    itself throughout.
    """
    host = _Host()
    host.show()
    before = Settled(host, qapp)

    host.label.setText("a much longer caption than before")

    assert not np.array_equal(before, Settled(host, qapp))


def test_settling_stops_once_the_window_stops_changing(qapp):
    """Bounded, so a widget that repaints forever cannot hang a build."""
    host = _Host()
    host.show()

    assert Settled(host, qapp, passes=1) is not None
    assert np.array_equal(Settled(host, qapp), Settled(host, qapp))


def test_a_main_windows_panel_is_laid_out_before_it_is_photographed(qapp):
    """Every window these demos photograph is a `QMainWindow` whose panel
    lives in its *central widget*, and a main window's own layout
    activating does not always reach that far in one pass. Missing it
    caught a half-laid-out control panel in the first frame.
    """
    window = QMainWindow()
    window.resize(300, 200)
    central = _Host()
    window.setCentralWidget(central)
    window.show()
    before = Settled(window, qapp)

    central.label.setText("a caption long enough to change the panel's width")

    assert not np.array_equal(before, Settled(window, qapp))
