from typing import Callable

from PyQt5.QtGui import QMouseEvent
from PyQt5.QtWidgets import QLabel

from click_recorder import ClickRecorder
from pipeline import Stage
from segmenter import Segmenter


class SegmenterStage(Stage):
    """Qt wiring for a Segmenter.

    The segmenter's user parameters are the positive/negative click points
    tracked by a ClickRecorder, not a control panel: instead of building a
    standalone widget, this stage attaches a click handler to the shared
    image view. Each click is already a discrete, complete action (unlike a
    slider drag), so it triggers `on_change` directly - no settling delay
    needed.
    """

    def __init__(self, segmenter: Segmenter | None = None) -> None:
        self.segmenter = segmenter or Segmenter()
        self.click_recorder: ClickRecorder | None = None
        self.mask = None
        self._on_change: Callable[[], None] | None = None

    def AttachToImageWidget(
        self, image_widget: QLabel, image_shape: tuple, on_change: Callable[[], None]
    ) -> None:
        """Start tracking clicks on `image_widget` as segmentation points."""
        self.click_recorder = ClickRecorder(image_widget, image_shape)
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
        self.mask = masks[0, 0].astype(bool)
