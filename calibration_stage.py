from typing import Callable

from PyQt5.QtWidgets import QLabel, QVBoxLayout, QWidget

from calibration import HoughCircleCalibration, IdentityCalibration
from pipeline import CreateSlider, CreateSpinBox, Stage


class IdentityCalibrationStage(Stage):
    """Qt wiring for IdentityCalibration - a stub with no parameters to
    tune, so its widget is just an explanatory label.
    """

    def __init__(self, calibration: IdentityCalibration | None = None) -> None:
        self.calibration = calibration or IdentityCalibration()

    def Run(self, image) -> None:
        self.calibration.Detect(image)

    def CreateWidget(self, on_change: Callable[[], None]) -> QWidget:
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(QLabel("Calibration: identity stub (1px = 1mm)"))
        return widget


class HoughCircleCalibrationStage(Stage):
    """Qt widget and pipeline wiring for a HoughCircleCalibration.

    Wraps the algorithm (which knows nothing about Qt) and exposes the
    sliders that edit its parameters, debounced so a settled edit - not
    every intermediate drag tick - triggers `on_change`.
    """

    def __init__(self, calibration: HoughCircleCalibration | None = None) -> None:
        self.calibration = calibration or HoughCircleCalibration()
        self._sliders: dict[str, dict] = {}

    def Run(self, image) -> None:
        self.calibration.Detect(image)

    def CreateWidget(self, on_change: Callable[[], None]) -> QWidget:
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(0, 0, 0, 0)

        def add_slider(field_name: str, label_text: str, min_val: int, max_val: int):
            def apply(value):
                setattr(self.calibration.parameters, field_name, value)
                on_change()

            slider = CreateSlider(
                label_text,
                min_val,
                max_val,
                getattr(self.calibration.parameters, field_name),
                apply,
            )
            layout.addLayout(slider["layout"])
            self._sliders[field_name] = slider

        add_slider("max_circles", "Top N Circles:", 1, 20)
        add_slider("min_dist", "Min Distance:", 10, 200)
        add_slider("param1", "Param1 (edge detection):", 10, 200)
        add_slider("param2", "Param2 (threshold):", 1, 100)
        add_slider("min_radius", "Min Radius:", 1, 100)
        add_slider("max_radius", "Max Radius:", 10, 200)
        add_slider("threshold_value", "Binary Threshold:", 1, 255)

        def apply_leg_distance(value):
            self.calibration.parameters.leg_distance_mm = value
            on_change()

        leg_distance = CreateSpinBox(
            "Leg Distance:",
            0.1,
            1000.0,
            self.calibration.parameters.leg_distance_mm,
            apply_leg_distance,
            suffix=" mm",
        )
        layout.addLayout(leg_distance["layout"])
        self._leg_distance_spin_box = leg_distance["spin_box"]

        return widget

    def ConfigureForImageShape(self, shape: tuple) -> None:
        """Update parameter defaults for the loaded image size, and mirror
        them onto the min/max radius sliders (range and displayed value).
        """
        self.calibration.ConfigureForImageShape(shape)

        if not self._sliders:
            return

        min_dim = max(1, min(shape[:2]))
        self._sliders["max_radius"]["slider"].setMaximum(min_dim)
        self._sliders["min_radius"]["slider"].setMaximum(min_dim)
        self._sliders["max_radius"]["slider"].setValue(self.calibration.parameters.max_radius)
        self._sliders["min_radius"]["slider"].setValue(self.calibration.parameters.min_radius)
