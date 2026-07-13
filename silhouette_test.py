"""Integration tests that exercise the pipeline stages against a real photo.

Uses IMG_SPOON.JPG (checked into the repo): a spoon on a dark cloth next to a
ruler card. Click coordinates below were sampled from that image, not guessed.
"""

import math
import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import matplotlib

matplotlib.use("Agg")  # headless: plt.show() must not block or open a window

import cv2
import numpy as np
import pytest
from PyQt5.QtCore import QEvent, QPointF, Qt
from PyQt5.QtGui import QMouseEvent, QPixmap
from PyQt5.QtWidgets import QApplication, QDialog, QLabel

from click_recorder import ClickRecorder, ClickRecorderParameters
from morphology import Morphology
from segmenter import Segmenter
from silhouette import _MODE_SELECT_CONTOUR, _MODE_SELECT_FIDUCIAL, SVGGui

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


def _click(
    recorder: ClickRecorder,
    scale_x: float,
    scale_y: float,
    x: int,
    y: int,
    button,
):
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
    assert math.hypot(actual[0] - target[0], actual[1] - target[1]) <= tol, f"{actual} not within {tol}px of {target}"


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
    _click(
        recorder,
        scale_x,
        scale_y,
        *BACKGROUND_POINT,
        Qt.MouseButton.RightButton,
    )
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
    for actual, target in zip(recorder.image_points, [SPOON_POINT_A, SPOON_POINT_B, BACKGROUND_POINT]):
        _assert_point_close(actual, target)
    assert recorder.image_labels == [1, 1, 0]


def test_click_recorder_respects_configured_erase_radius(qapp, spoon_image):
    height, width = spoon_image.shape[:2]
    widget = QLabel()
    pixmap = QPixmap(width, height)  # 1:1 scale keeps click math trivial
    widget.setPixmap(pixmap)
    widget.resize(pixmap.size())

    recorder = ClickRecorder(widget, spoon_image.shape, ClickRecorderParameters(erase_radius=20.0))
    _click(recorder, 1.0, 1.0, *SPOON_POINT_A, Qt.MouseButton.LeftButton)

    # 15px away is within the configured 20px erase radius, but outside the
    # default 5px radius - proves the parameter is actually being used.
    far_point = (SPOON_POINT_A[0] + 15, SPOON_POINT_A[1])
    _click(recorder, 1.0, 1.0, *far_point, Qt.MouseButton.MiddleButton)

    assert recorder.image_points == []


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


@pytest.fixture(scope="session")
def gui(qapp):
    window = SVGGui()
    window.load_image(IMAGE_PATH)
    return window


def _click_gui(window: SVGGui, x: int, y: int, button) -> None:
    """Simulate a real user click on the main image view at image
    coordinates (x, y), going through SVGGui.image_clicked exactly like a
    mouse press on the displayed pixmap would.
    """
    pixmap = window.image_label.pixmap()
    assert pixmap is not None, "an image must be loaded before clicking"
    assert window.original_image is not None
    scale_x = pixmap.width() / window.original_image.shape[1]
    scale_y = pixmap.height() / window.original_image.shape[0]
    # The pixmap is letterboxed within image_label (aspect ratio preserved),
    # so a real click's widget-relative position includes that margin -
    # image_clicked subtracts it back out.
    offset_x = (window.image_label.width() - pixmap.width()) // 2
    offset_y = (window.image_label.height() - pixmap.height()) // 2
    ev = QMouseEvent(
        QEvent.MouseButtonPress,
        QPointF(x * scale_x + offset_x, y * scale_y + offset_y),
        button,
        button,
        Qt.NoModifier,
    )
    window.image_clicked(ev)


@pytest.mark.slow
def test_full_app_click_flow(gui, monkeypatch):
    """Drives the whole app the way a user would: click to segment, click to
    select the object, then export - replacing what used to be a series of
    one-off manual verification scripts.
    """
    # Don't block on the modal export dialog's own event loop.
    monkeypatch.setattr(QDialog, "exec_", lambda self: None)

    # A cleaner morphology threshold than the 1000px default, so the
    # segmented spoon collapses to a single contour instead of dozens of
    # small noise specks - makes the object-selection step below reliable.
    gui.morphology_stage.morphology.parameters.area = 20000

    assert gui.object_contours == [], "no contours until the user clicks segmentation points"

    # Two positive clicks on the spoon, one negative on the background. Each
    # click is a discrete action, so it triggers segmentation immediately -
    # no settling delay to wait out.
    _click_gui(gui, *SPOON_POINT_A, Qt.MouseButton.LeftButton)
    _click_gui(gui, *SPOON_POINT_B, Qt.MouseButton.LeftButton)
    _click_gui(gui, *BACKGROUND_POINT, Qt.MouseButton.RightButton)

    assert len(gui.segmenter_stage.click_recorder.image_points) == 3
    assert gui.segmenter_stage.mask is not None, "segmentation should run after each click"
    assert gui.morphology_stage.mask is not None
    assert gui.object_contours, "contours should be extracted from the segmented mask"

    # Switch to contour-select mode: further clicks toggle objects instead
    # of adding more segmentation points.
    gui.interaction_mode_combo.setCurrentText(_MODE_SELECT_CONTOUR)

    # Click a point known to be on the segmented spoon (one of the
    # segmentation clicks above) to select its contour - a bounding-box
    # center isn't reliable for a concave shape like a spoon.
    target_index = next(
        i for i, contour in enumerate(gui.object_contours) if cv2.pointPolygonTest(contour, SPOON_POINT_A, False) >= 0
    )
    _click_gui(gui, *SPOON_POINT_A, Qt.MouseButton.LeftButton)

    contour_selection = gui.contour_selection_stage.contour_selection
    assert contour_selection.selected == {target_index}
    assert target_index in contour_selection.simplified
    assert target_index in contour_selection.boxes

    # Fiducial-select mode: the active calibration is still the
    # IdentityCalibration stub, which has no fiducials to select - a click
    # in this mode should be a safe no-op, not crash and not touch the
    # segmentation points or contour selection made above.
    gui.interaction_mode_combo.setCurrentText(_MODE_SELECT_FIDUCIAL)
    _click_gui(gui, *BACKGROUND_POINT, Qt.MouseButton.LeftButton)
    assert len(gui.segmenter_stage.click_recorder.image_points) == 3
    assert contour_selection.selected == {target_index}
    gui.interaction_mode_combo.setCurrentText(_MODE_SELECT_CONTOUR)

    # Export: the spoon photo has no ArUco markers, so GetTransform() raises
    # and export_contours() falls back to an identity transform rather than
    # failing - contours come out in pixel space, and the dialog still
    # builds without raising, thanks to the monkeypatch.
    assert gui.calibration_stage.calibration.detected_corners == {}
    gui.pipeline.RunFrom("export")
    assert target_index in gui.rectify.contours
    assert np.allclose(
        gui.rectify.contours[target_index],
        np.squeeze(contour_selection.simplified[target_index]),
    )
