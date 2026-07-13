from typing import Callable

from PyQt5.QtWidgets import QWidget

from contour_extraction import ContourSelection
from pipeline import CreateGroupBox, CreateSlider, Stage

_EPSILON_SLIDER_SCALE = 1000  # slider is int-only; epsilon_fraction is a small float


class ContourSelectionStage(Stage):
    """Qt wiring for ContourSelection: a debounced slider for the
    simplification threshold. Selection itself happens by clicking directly
    on the shared image view (see SVGGui.image_clicked), the same way
    HoughCircleCalibration.ToggleSelection works for circle fiducials, when
    that calibration stage is wired in.
    """

    def __init__(self, contour_selection: ContourSelection | None = None) -> None:
        self.contour_selection = contour_selection or ContourSelection()

    def Run(self, contours: list) -> None:
        self.contour_selection.Run(contours)

    def CreateWidget(self, on_change: Callable[[], None]) -> QWidget:
        widget, layout = CreateGroupBox("Contour Selection")

        def apply(value):
            self.contour_selection.parameters.epsilon_fraction = value / _EPSILON_SLIDER_SCALE
            on_change()

        default = round(self.contour_selection.parameters.epsilon_fraction * _EPSILON_SLIDER_SCALE)
        slider = CreateSlider(
            "Contour Simplification (finer → coarser):", 1, 50, default, apply
        )
        layout.addLayout(slider["layout"])

        return widget
