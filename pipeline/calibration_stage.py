"""Qt wiring for `pipeline.calibration`.

Only `ArucoCalibrationStage` lives here - the one calibration strategy
`silhouette.py` actually wires into the app. `IdentityCalibration` and
`HoughCircleCalibration` (see `pipeline/calibration.py`) have no Qt
wiring of their own for the same reason: nothing in this project
constructs a calibration stage other than this one.
"""

from typing import Callable

from PyQt5.QtWidgets import QLabel, QWidget

from pipeline.calibration import ArucoCalibration
from pipeline.core import CreateGroupBox, Stage


class ArucoCalibrationStage(Stage):
    """Qt wiring for ArucoCalibration - no parameters to tune from the
    control panel (the marker layout lives in
    ArucoCalibration.parameters.marker_positions_mm), so its widget is just
    a status label showing how many detected markers matched a known
    position, updated on every Run().
    """

    def __init__(self, calibration: ArucoCalibration | None = None) -> None:
        self.calibration = calibration or ArucoCalibration()
        self._status_label: QLabel | None = None

    def Run(self, image) -> None:
        self.calibration.Detect(image)
        self._UpdateStatusLabel()

    def _UpdateStatusLabel(self) -> None:
        if self._status_label is None:
            return
        known_positions = self.calibration.parameters.marker_positions_mm
        matched = sum(1 for marker_id in self.calibration.detected_corners if marker_id in known_positions)
        detected = len(self.calibration.detected_corners)
        self._status_label.setText(
            f"Calibration: ArUco markers ({matched}/{detected} detected markers matched a known position)"
        )

    def CreateWidget(self, on_change: Callable[[], None]) -> QWidget:
        widget, layout = CreateGroupBox("Calibration")
        self._status_label = QLabel("Calibration: ArUco markers (none detected yet)")
        layout.addWidget(self._status_label)
        return widget
