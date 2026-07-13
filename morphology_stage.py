from typing import Callable

from PyQt5.QtWidgets import QWidget

from morphology import Morphology
from pipeline import CreateGroupBox, CreateSlider, Stage


class MorphologyStage(Stage):
    """Qt wiring for Morphology: debounced sliders controlling the closing
    radius and minimum surviving area, and the cleaned-up mask it produces
    from whatever mask is fed into it (e.g. a segmentation stage's output).
    """

    def __init__(self, morphology: Morphology | None = None) -> None:
        self.morphology = morphology or Morphology()
        self.mask = None

    def Run(self, mask_image) -> None:
        self.mask = None if mask_image is None else self.morphology.Apply(mask_image)

    def CreateWidget(self, on_change: Callable[[], None]) -> QWidget:
        widget = CreateGroupBox("Mask Cleanup")
        layout = widget.layout()

        def apply_closing_radius(value):
            self.morphology.parameters.closing_radius = value
            on_change()

        closing_slider = CreateSlider(
            "Closing Radius (px):",
            0,
            50,
            self.morphology.parameters.closing_radius,
            apply_closing_radius,
        )
        layout.addLayout(closing_slider["layout"])

        def apply_area(value):
            self.morphology.parameters.area = value
            on_change()

        area_slider = CreateSlider(
            "Cleanup Min Area (px):", 1, 20000, self.morphology.parameters.area, apply_area
        )
        layout.addLayout(area_slider["layout"])

        return widget
