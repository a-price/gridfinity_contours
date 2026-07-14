from typing import Callable

from PyQt5.QtWidgets import QCheckBox, QComboBox, QLabel, QWidget

from pipeline.morphology import Morphology
from pipeline.core import CreateGroupBox, CreateSlider, Stage


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

        lateral_checkbox = QCheckBox("Lateral symmetry (left/right)")
        lateral_checkbox.setChecked(self.morphology.parameters.symmetrize_lateral)

        def apply_lateral(checked):
            self.morphology.parameters.symmetrize_lateral = checked
            on_change()

        lateral_checkbox.toggled.connect(apply_lateral)
        layout.addWidget(lateral_checkbox)

        longitudinal_checkbox = QCheckBox("Longitudinal symmetry (front/back)")
        longitudinal_checkbox.setChecked(self.morphology.parameters.symmetrize_longitudinal)

        def apply_longitudinal(checked):
            self.morphology.parameters.symmetrize_longitudinal = checked
            on_change()

        longitudinal_checkbox.toggled.connect(apply_longitudinal)
        layout.addWidget(longitudinal_checkbox)

        layout.addWidget(QLabel("Symmetry Combine:"))
        combine_combo = QComboBox()
        combine_combo.addItems(["and", "or"])
        combine_combo.setCurrentText(self.morphology.parameters.symmetry_combine)

        def apply_combine(text):
            self.morphology.parameters.symmetry_combine = text
            on_change()

        combine_combo.currentTextChanged.connect(apply_combine)
        layout.addWidget(combine_combo)

        return widget
