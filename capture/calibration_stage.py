"""Qt wiring for `capture.calibration`.

Only `ArucoCalibrationStage` lives here, because `ArucoCalibration` is the
only strategy any application constructs. The other three
(`IdentityCalibration`, `HoughCircleCalibration`, `PaperCalibration`, all
in `capture/calibration.py`) are reached only from tests - though
`generate_aruco_sheet.py` does take its page size from
`PaperCalibration`, which is how the printed sheet and the calibration
stay agreed on how big a page is.
"""

from typing import Callable

from PyQt5.QtWidgets import QLabel, QWidget

from capture.calibration import ArucoCalibration
from qt_utils.widgets import CreateGroupBox
from capture.pipeline import Stage


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
        # Wrapped, because this label gets much longer once markers are
        # found - and an unwrapped one widens the whole control panel to
        # fit, shoving the image view sideways the moment a photo loads.
        self._status_label.setWordWrap(True)
        layout.addWidget(self._status_label)
        return widget
