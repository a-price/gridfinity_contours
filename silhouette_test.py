"""Integration tests that exercise the pipeline stages against a real photo.

Uses IMG_SPOON.JPG (checked into the repo): a spoon on a dark cloth next to a
ruler card. Click coordinates below were sampled from that image, not guessed.
"""

import math
import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import cv2
import pytest
from PyQt5.QtCore import QEvent, QPointF, Qt
from PyQt5.QtGui import QMouseEvent, QPixmap
from PyQt5.QtWidgets import QApplication, QLabel

from click_recorder import ClickRecorder
from morphology import Morphology
from segmenter import Segmenter

IMAGE_PATH = os.path.join(os.path.dirname(__file__), "IMG_SPOON.JPG")

# Points on the spoon body (positive) and off it (negative), in original
# image coordinates. Found via connected-component analysis of the same
# gray > 50 threshold the app itself uses in SVGGui.find_contours.
SPOON_POINT_A = (3550, 1550)  # bowl
SPOON_POINT_B = (1500, 1150)  # handle
BACKGROUND_POINT = (900, 2700)  # dark cloth
RULER_POINT = (4600, 2400)  # ruler card, a different bright object


@pytest.fixture(scope="session")
def qapp():
    app = QApplication.instance() or QApplication([])
    return app


@pytest.fixture(scope="session")
def spoon_image():
    image = cv2.imread(IMAGE_PATH, cv2.IMREAD_COLOR)
    assert image is not None, f"could not load test image at {IMAGE_PATH}"
    return image


def _click(recorder: ClickRecorder, scale_x: float, scale_y: float, x: int, y: int, button):
    ev = QMouseEvent(
        QEvent.MouseButtonPress,
        QPointF(x * scale_x, y * scale_y),
        button,
        button,
        Qt.NoModifier,
    )
    recorder.OnClick(ev)


def _assert_point_close(actual, target, tol=5):
    # Widget coordinates are quantized to an integer pixmap size, so the
    # image-space round trip loses a few pixels of precision even for a
    # real click - same as what a user would see in the app itself.
    assert math.hypot(actual[0] - target[0], actual[1] - target[1]) <= tol, (
        f"{actual} not within {tol}px of {target}"
    )


def test_click_recorder_records_and_erases_points(qapp, spoon_image):
    height, width = spoon_image.shape[:2]
    nominal_scale = 0.2  # mimic the app displaying a scaled-down pixmap

    widget = QLabel()
    pixmap = QPixmap(int(width * nominal_scale), int(height * nominal_scale))
    widget.setPixmap(pixmap)
    widget.resize(pixmap.size())
    scale_x = pixmap.width() / width
    scale_y = pixmap.height() / height

    recorder = ClickRecorder(widget, spoon_image.shape)

    # Positive clicks on the spoon, negative clicks on background/ruler.
    _click(recorder, scale_x, scale_y, *SPOON_POINT_A, Qt.MouseButton.LeftButton)
    _click(recorder, scale_x, scale_y, *SPOON_POINT_B, Qt.MouseButton.LeftButton)
    _click(recorder, scale_x, scale_y, *BACKGROUND_POINT, Qt.MouseButton.RightButton)
    _click(recorder, scale_x, scale_y, *RULER_POINT, Qt.MouseButton.RightButton)

    assert len(recorder.image_points) == 4
    for actual, target in zip(
        recorder.image_points,
        [SPOON_POINT_A, SPOON_POINT_B, BACKGROUND_POINT, RULER_POINT],
    ):
        _assert_point_close(actual, target)
    assert recorder.image_labels == [1, 1, 0, 0]

    # A middle-click on top of the ruler point should erase just that point.
    _click(recorder, scale_x, scale_y, *RULER_POINT, Qt.MouseButton.MiddleButton)

    assert len(recorder.image_points) == 3
    for actual, target in zip(
        recorder.image_points, [SPOON_POINT_A, SPOON_POINT_B, BACKGROUND_POINT]
    ):
        _assert_point_close(actual, target)
    assert recorder.image_labels == [1, 1, 0]


@pytest.fixture(scope="session")
def segmenter():
    return Segmenter()


@pytest.mark.slow
def test_segment_and_clean_up_spoon_mask(spoon_image, segmenter):
    input_points = [[[list(SPOON_POINT_A), list(SPOON_POINT_B), list(BACKGROUND_POINT)]]]
    input_labels = [[[1, 1, 0]]]

    masks = segmenter.Segment(spoon_image, input_points, input_labels)

    # [object, mask_hypothesis, height, width]
    assert masks.ndim == 4
    height, width = spoon_image.shape[:2]
    assert masks.shape[-2:] == (height, width)

    best_mask = masks[0, 0].astype(bool)
    assert best_mask.any(), "segmenter produced an empty mask"

    # The clicked spoon points should end up inside the predicted mask.
    assert best_mask[SPOON_POINT_A[1], SPOON_POINT_A[0]]
    assert best_mask[SPOON_POINT_B[1], SPOON_POINT_B[0]]

    cleaned = Morphology().Apply(best_mask)
    assert cleaned.shape == best_mask.shape
    assert cleaned.any()
