from typing import Callable

from PyQt5.QtWidgets import QCheckBox, QComboBox, QLabel, QWidget

from capture.morphology import Morphology
from qt_utils.widgets import CreateGroupBox, CreateSlider
from pipeline.core import Stage

# The checkbox and combo captions, named here rather than written inline,
# so anything that has to *find* one of these controls - a test, or the
# script that records the capture animation - does it by the same string
# the widget was built from rather than by a copy that can drift.
LATERAL_LABEL = "Lateral symmetry (left/right)"
LONGITUDINAL_LABEL = "Longitudinal symmetry (front/back)"
COMBINE_LABEL = "Symmetry Combine:"


class MorphologyStage(Stage):
    """Qt wiring for Morphology: debounced sliders controlling the closing
    radius and minimum surviving area, checkboxes for optional
    lateral/longitudinal symmetry and how to combine it (AND/OR) with the
    original mask, and the cleaned-up mask it produces from whatever mask
    is fed into it (e.g. a segmentation stage's output).
    """

    def __init__(self, morphology: Morphology | None = None) -> None:
        self.morphology = morphology or Morphology()
        self.mask = None

    def Run(self, mask_image) -> None:
        self.mask = None if mask_image is None else self.morphology.Apply(mask_image)

    def CreateWidget(self, on_change: Callable[[], None]) -> QWidget:
        widget, layout = CreateGroupBox("Mask Cleanup")

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
            "Cleanup Min Area (px):",
            1,
            20000,
            self.morphology.parameters.area,
            apply_area,
        )
        layout.addLayout(area_slider["layout"])

        lateral_checkbox = QCheckBox(LATERAL_LABEL)
        lateral_checkbox.setChecked(self.morphology.parameters.symmetrize_lateral)

        def apply_lateral(checked):
            self.morphology.parameters.symmetrize_lateral = checked
            on_change()

        lateral_checkbox.toggled.connect(apply_lateral)
        layout.addWidget(lateral_checkbox)

        longitudinal_checkbox = QCheckBox(LONGITUDINAL_LABEL)
        longitudinal_checkbox.setChecked(self.morphology.parameters.symmetrize_longitudinal)

        def apply_longitudinal(checked):
            self.morphology.parameters.symmetrize_longitudinal = checked
            on_change()

        longitudinal_checkbox.toggled.connect(apply_longitudinal)
        layout.addWidget(longitudinal_checkbox)

        layout.addWidget(QLabel(COMBINE_LABEL))
        combine_combo = QComboBox()
        combine_combo.addItems(["and", "or"])
        combine_combo.setCurrentText(self.morphology.parameters.symmetry_combine)

        def apply_combine(text):
            self.morphology.parameters.symmetry_combine = text
            on_change()

        combine_combo.currentTextChanged.connect(apply_combine)
        layout.addWidget(combine_combo)

        return widget
