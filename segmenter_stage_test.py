import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import cv2
import pytest
from PyQt5.QtCore import QEvent, QPointF, Qt
from PyQt5.QtGui import QMouseEvent, QPixmap
from PyQt5.QtWidgets import QApplication, QLabel

from segmenter import SegmenterParameters
from segmenter_stage import SegmenterStage

IMAGE_PATH = os.path.join(os.path.dirname(__file__), "IMG_SPOON.JPG")

SPOON_POINT_A = (3550, 1550)  # bowl
BACKGROUND_POINT = (900, 2700)  # dark cloth


@pytest.fixture(scope="session")
def qapp():
    return QApplication.instance() or QApplication([])


@pytest.fixture(scope="session")
def spoon_image():
    image = cv2.imread(IMAGE_PATH, cv2.IMREAD_COLOR)
    assert image is not None, f"could not load test image at {IMAGE_PATH}"
    return image


def _click(x: int, y: int, button):
    return QMouseEvent(
        QEvent.MouseButtonPress, QPointF(x, y), button, button, Qt.NoModifier
    )


def _make_widget(shape) -> QLabel:
    height, width = shape[:2]
    widget = QLabel()
    pixmap = QPixmap(width, height)  # 1:1 scale keeps click math trivial
    widget.setPixmap(pixmap)
    widget.resize(pixmap.size())
    return widget


class FakeSegmenter:
    """Stands in for the real (slow) Segmenter to test the stage's wiring
    without a model. Returns 3 mask hypotheses, each filled with a distinct
    value, so tests can tell which one the stage picked."""

    def __init__(self) -> None:
        self.parameters = SegmenterParameters()
        self.calls: list[tuple] = []

    def Segment(self, image, input_points, input_labels):
        self.calls.append((input_points, input_labels))
        height, width = image.shape[:2]
        import numpy as np

        masks = np.zeros((1, 3, height, width), dtype=bool)
        masks[0, 0] = True
        return masks


def test_attach_and_click_records_points_on_the_underlying_recorder(qapp, spoon_image):
    widget = _make_widget(spoon_image.shape)
    stage = SegmenterStage(FakeSegmenter())
    changes = []
    stage.AttachToImageWidget(widget, spoon_image.shape, on_change=lambda: changes.append(1))

    stage.OnClick(_click(*SPOON_POINT_A, Qt.MouseButton.LeftButton))
    stage.OnClick(_click(*BACKGROUND_POINT, Qt.MouseButton.RightButton))

    assert stage.click_recorder.image_points == [
        list(SPOON_POINT_A),
        list(BACKGROUND_POINT),
    ]
    assert stage.click_recorder.image_labels == [1, 0]


def test_each_click_triggers_on_change_immediately(qapp, spoon_image):
    # Unlike a slider drag, each click is already a discrete, complete
    # action, so it should trigger on_change right away - no settling delay.
    widget = _make_widget(spoon_image.shape)
    stage = SegmenterStage(FakeSegmenter())
    changes = []
    stage.AttachToImageWidget(widget, spoon_image.shape, on_change=lambda: changes.append(1))

    stage.OnClick(_click(*SPOON_POINT_A, Qt.MouseButton.LeftButton))
    assert changes == [1]

    stage.OnClick(_click(*BACKGROUND_POINT, Qt.MouseButton.RightButton))
    stage.OnClick(_click(1500, 1150, Qt.MouseButton.LeftButton))
    assert changes == [1, 1, 1]


def test_run_with_no_points_leaves_mask_none(spoon_image):
    stage = SegmenterStage(FakeSegmenter())
    stage.Run(spoon_image)
    assert stage.mask is None


def test_run_calls_segmenter_with_points_and_labels_from_click_recorder(qapp, spoon_image):
    widget = _make_widget(spoon_image.shape)
    fake = FakeSegmenter()
    stage = SegmenterStage(fake)
    stage.AttachToImageWidget(widget, spoon_image.shape, on_change=lambda: None)

    stage.OnClick(_click(*SPOON_POINT_A, Qt.MouseButton.LeftButton))
    stage.OnClick(_click(*BACKGROUND_POINT, Qt.MouseButton.RightButton))

    stage.Run(spoon_image)

    assert stage.mask is not None
    assert stage.mask.shape == spoon_image.shape[:2]
    (points, labels), = fake.calls
    assert points == [[[list(SPOON_POINT_A), list(BACKGROUND_POINT)]]]
    assert labels == [[[1, 0]]]


def test_run_uses_configured_mask_hypothesis_index(qapp, spoon_image):
    widget = _make_widget(spoon_image.shape)
    fake = FakeSegmenter()
    stage = SegmenterStage(fake)
    stage.AttachToImageWidget(widget, spoon_image.shape, on_change=lambda: None)
    stage.OnClick(_click(*SPOON_POINT_A, Qt.MouseButton.LeftButton))

    fake.parameters.mask_hypothesis_index = 1
    stage.Run(spoon_image)

    # FakeSegmenter only fills hypothesis 0 with True - picking hypothesis 1
    # should come back empty, proving the index was actually used.
    assert not stage.mask.any()


def test_run_clamps_mask_hypothesis_index_to_available_range(qapp, spoon_image):
    widget = _make_widget(spoon_image.shape)
    fake = FakeSegmenter()
    stage = SegmenterStage(fake)
    stage.AttachToImageWidget(widget, spoon_image.shape, on_change=lambda: None)
    stage.OnClick(_click(*SPOON_POINT_A, Qt.MouseButton.LeftButton))

    fake.parameters.mask_hypothesis_index = 99  # beyond FakeSegmenter's 3 hypotheses
    stage.Run(spoon_image)  # should not raise an IndexError

    assert stage.mask is not None


@pytest.mark.slow
def test_run_end_to_end_with_real_segmenter(qapp, spoon_image):
    from segmenter import Segmenter

    widget = _make_widget(spoon_image.shape)
    stage = SegmenterStage(Segmenter())
    stage.AttachToImageWidget(widget, spoon_image.shape, on_change=lambda: None)

    stage.OnClick(_click(*SPOON_POINT_A, Qt.MouseButton.LeftButton))
    stage.OnClick(_click(1500, 1150, Qt.MouseButton.LeftButton))
    stage.OnClick(_click(*BACKGROUND_POINT, Qt.MouseButton.RightButton))

    stage.Run(spoon_image)

    assert stage.mask is not None
    assert stage.mask.any()
    assert stage.mask[SPOON_POINT_A[1], SPOON_POINT_A[0]]
