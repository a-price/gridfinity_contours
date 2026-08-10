"""Tests for the geometry between a displayed photo and the widget showing it.

Two things depend on that mapping and have to agree about it: turning a
click back into image coordinates, and sizing the marks drawn over the
image. The second is what these are mostly about, because it is the one
that was wrong - annotations were sized in image pixels, so a 43px cross
on a 5184px photograph came out four pixels wide once displayed, and the
same constant on a small image would have covered it.
"""

import numpy as np
import pytest
from PyQt5.QtCore import QEvent, QPointF, Qt
from PyQt5.QtGui import QMouseEvent, QPixmap
from PyQt5.QtWidgets import QLabel

from qt_utils.click_recorder import (
    ClickRecorder,
    ImagePixelsPerScreenPixel,
    WidgetToImageCoords,
)

PHOTO = (3456, 5184, 3)  # the shape of a real capture


def _widget(width: int, height: int, pixmap: tuple[int, int] | None = None) -> QLabel:
    widget = QLabel()
    widget.resize(width, height)
    if pixmap is not None:
        widget.setPixmap(QPixmap(*pixmap))
    return widget


# ------------------------------------------------------------------ scaling


def test_a_big_photo_in_a_small_widget_needs_big_marks(qapp):
    """The whole point. A 5184px photo shown 700px wide covers about seven
    image pixels per screen pixel, so a mark meant to read as ten screen
    pixels has to be drawn seventy image pixels long.
    """
    factor = ImagePixelsPerScreenPixel(_widget(700, 700), PHOTO)

    assert factor == pytest.approx(5184 / 700, rel=0.01)


def test_the_fit_is_the_limiting_dimension(qapp):
    """`KeepAspectRatio` fits whichever side runs out first, so the scale
    is the smaller ratio - the same rule `QPixmap.scaled` applies, or the
    marks would be sized against a fit the window never used.
    """
    letterboxed = ImagePixelsPerScreenPixel(_widget(2000, 100), PHOTO)

    assert letterboxed == pytest.approx(3456 / 100, rel=0.01), "height is what ran out"


def test_a_mark_keeps_its_screen_size_across_photo_resolutions(qapp):
    """The property that makes this worth having. The same annotation
    drawn on a phone snap and on a 5184px camera file has to come out the
    same size on screen, or one of the two is unusable.
    """
    widget = _widget(700, 700)

    small = 10 * ImagePixelsPerScreenPixel(widget, (600, 900, 3))
    large = 10 * ImagePixelsPerScreenPixel(widget, PHOTO)

    assert small / (900 / 700) == pytest.approx(large / (5184 / 700), rel=0.01)


def test_a_bigger_window_wants_smaller_marks(qapp):
    """Sized against the window as well as the photo, since the same photo
    in a maximized window is displayed several times larger.
    """
    small_window = ImagePixelsPerScreenPixel(_widget(600, 600), PHOTO)
    big_window = ImagePixelsPerScreenPixel(_widget(1800, 1800), PHOTO)

    assert big_window < small_window


def test_a_widget_with_no_room_yet_does_not_collapse_the_marks(qapp):
    """Before the first layout a widget can report zero size. Returning
    zero here would multiply every annotation to nothing, which reads as a
    drawing bug rather than a timing one.
    """
    assert ImagePixelsPerScreenPixel(_widget(0, 0), PHOTO) == 1.0
    assert ImagePixelsPerScreenPixel(_widget(700, 700), (0, 0, 3)) == 1.0


def test_it_agrees_with_the_pixmap_once_there_is_one(qapp):
    """It predicts the fit from the widget rather than measuring the
    pixmap, so that it is right on the first frame - but the two describe
    the same scaling and must not drift apart.
    """
    widget = _widget(700, 500)
    scaled = QPixmap(PHOTO[1], PHOTO[0]).scaled(
        widget.size(), Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.FastTransformation
    )
    widget.setPixmap(scaled)

    assert ImagePixelsPerScreenPixel(widget, PHOTO) == pytest.approx(PHOTO[1] / scaled.width(), rel=0.01)


# -------------------------------------------------------------- click coords


def test_a_click_maps_back_to_the_pixel_it_landed_on(qapp):
    """1:1 and unletterboxed, so the arithmetic is checkable by eye."""
    widget = _widget(400, 300, pixmap=(400, 300))

    coords = WidgetToImageCoords(widget, (300, 400, 3), _click(120, 90))

    assert coords == (120, 90)


def test_a_click_outside_the_photo_is_not_a_click_on_it(qapp):
    """The pixmap is centered in the widget and rarely fills it, so the
    margins are part of the window rather than of the image.
    """
    widget = _widget(400, 400, pixmap=(400, 200))

    assert WidgetToImageCoords(widget, (200, 400, 3), _click(200, 5)) is None


def test_a_click_before_anything_is_displayed_is_ignored(qapp):
    assert WidgetToImageCoords(_widget(400, 300), (300, 400, 3), _click(10, 10)) is None


def test_a_recorded_click_is_integer_pixels(qapp):
    """The mask is indexed with these, and numpy refuses a float index."""
    widget = _widget(400, 300, pixmap=(400, 300))
    recorder = ClickRecorder(widget, (300, 400, 3))

    recorder.OnClick(_click(41, 33))

    assert recorder.image_points == [[41, 33]]
    assert all(isinstance(value, (int, np.integer)) for value in recorder.image_points[0])


def test_the_button_says_interior_or_exterior(qapp):
    """Left is a point on the object, right is a point that is not - which
    is how a drop shadow gets excluded.
    """
    widget = _widget(400, 300, pixmap=(400, 300))
    recorder = ClickRecorder(widget, (300, 400, 3))

    recorder.OnClick(_click(10, 10, Qt.MouseButton.LeftButton))
    recorder.OnClick(_click(20, 20, Qt.MouseButton.RightButton))

    assert recorder.image_labels == [1, 0]


def _click(x: int, y: int, button=Qt.MouseButton.LeftButton) -> QMouseEvent:
    return QMouseEvent(QEvent.Type.MouseButtonPress, QPointF(x, y), button, button, Qt.KeyboardModifier.NoModifier)
