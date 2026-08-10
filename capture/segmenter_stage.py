from typing import Callable

import cv2
import numpy as np
from PyQt5.QtGui import QMouseEvent
from PyQt5.QtWidgets import QLabel, QWidget

from qt_utils.click_recorder import ClickRecorder, ClickRecorderParameters
from qt_utils.widgets import CreateGroupBox, CreateSlider
from capture.pipeline import Stage
from capture.segmenter import Segmenter, SegmenterLike

# SAM2's processor typically ranks 3 mask hypotheses per click set (0 =
# highest predicted IoU); the slider range is fixed to that rather than
# discovered at runtime.
_MAX_MASK_HYPOTHESIS_INDEX = 2


def _KeepClickedComponents(mask: np.ndarray, seed_points: list) -> np.ndarray:
    """Keeps only the connected component(s) of `mask` that contain at
    least one of `seed_points` (image (x, y) coordinates), discarding any
    other disjoint blob a SAM mask hypothesis may have produced elsewhere
    in the image. Falls back to the unmodified mask if no seed point
    landed on foreground.
    """
    height, width = mask.shape[:2]
    _, labels = cv2.connectedComponents(mask.astype(np.uint8), connectivity=8)

    seed_labels = {int(labels[y, x]) for x, y in seed_points if 0 <= x < width and 0 <= y < height and mask[y, x]}
    seed_labels.discard(0)  # label 0 is background
    if not seed_labels:
        return mask

    return np.isin(labels, list(seed_labels))


class SegmenterStage(Stage):
    """Qt wiring for a Segmenter.

    Most of the segmenter's user parameters are the positive/negative click
    points tracked by a ClickRecorder, not a control panel: instead of
    building a standalone widget for those, this stage attaches a click
    handler to the shared image view. Each click is already a discrete,
    complete action (unlike a slider drag), so it triggers `on_change`
    directly - no settling delay needed. The remaining parameters (erase
    radius, which mask hypothesis to use) do get a small control panel
    widget via CreateWidget.
    """

    def __init__(self, segmenter: SegmenterLike | None = None) -> None:
        self.segmenter: SegmenterLike = segmenter or Segmenter()
        self.click_recorder_parameters = ClickRecorderParameters()
        self.click_recorder: ClickRecorder | None = None
        self.mask: np.ndarray | None = None
        self._on_change: Callable[[], None] | None = None

    def AttachToImageWidget(
        self,
        image_widget: QLabel,
        image_shape: tuple,
        on_change: Callable[[], None],
    ) -> None:
        """Start tracking clicks on `image_widget` as segmentation points."""
        self.click_recorder = ClickRecorder(image_widget, image_shape, self.click_recorder_parameters)
        self._on_change = on_change

    def OnClick(self, ev: QMouseEvent | None) -> None:
        if self.click_recorder is None:
            return
        self.click_recorder.OnClick(ev)
        if self._on_change is not None:
            self._on_change()

    def Run(self, image) -> None:
        if self.click_recorder is None or not self.click_recorder.image_points:
            self.mask = None
            return

        input_points = [[self.click_recorder.image_points]]
        input_labels = [[self.click_recorder.image_labels]]
        masks = self.segmenter.Segment(image, input_points, input_labels)
        hypothesis_index = min(self.segmenter.parameters.mask_hypothesis_index, masks.shape[1] - 1)
        mask = masks[0, hypothesis_index].astype(bool)

        positive_points = [
            point
            for point, label in zip(self.click_recorder.image_points, self.click_recorder.image_labels)
            if label == 1
        ]
        self.mask = _KeepClickedComponents(mask, positive_points) if positive_points else mask

    def CreateWidget(self, on_change: Callable[[], None]) -> QWidget:
        widget, layout = CreateGroupBox("Segmentation")

        def apply_erase_radius(value):
            # Only affects future middle-clicks, not the current mask - no
            # need to rerun the pipeline.
            self.click_recorder_parameters.erase_radius = value

        erase_slider = CreateSlider(
            "Click Erase Radius (px):",
            1,
            50,
            int(self.click_recorder_parameters.erase_radius),
            apply_erase_radius,
        )
        layout.addLayout(erase_slider["layout"])

        def apply_mask_hypothesis(value):
            self.segmenter.parameters.mask_hypothesis_index = value
            on_change()

        hypothesis_slider = CreateSlider(
            "SAM Mask Hypothesis:",
            0,
            _MAX_MASK_HYPOTHESIS_INDEX,
            self.segmenter.parameters.mask_hypothesis_index,
            apply_mask_hypothesis,
        )
        layout.addLayout(hypothesis_slider["layout"])

        return widget
