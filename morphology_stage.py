from typing import Callable

from PyQt5.QtWidgets import QVBoxLayout, QWidget

from morphology import Morphology
from pipeline import CreateSlider, Stage


class MorphologyStage(Stage):
    """Qt wiring for Morphology: a single debounced slider controlling the
    minimum surviving area, and the cleaned-up mask it produces from
    whatever mask is fed into it (e.g. a segmentation stage's output).
    """

    def __init__(self, morphology: Morphology | None = None) -> None:
        self.morphology = morphology or Morphology()
        self.mask = None

    def Run(self, mask_image) -> None:
        self.mask = None if mask_image is None else self.morphology.Apply(mask_image)

    def CreateWidget(self, on_change: Callable[[], None]) -> QWidget:
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(0, 0, 0, 0)

        def apply(value):
            self.morphology.parameters.area = value
            on_change()

        slider = CreateSlider(
            "Cleanup Min Area (px):", 1, 20000, self.morphology.parameters.area, apply
        )
        layout.addLayout(slider["layout"])

        return widget
