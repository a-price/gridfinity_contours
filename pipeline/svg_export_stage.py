from dataclasses import dataclass
from typing import Callable

import numpy as np
from PyQt5.QtWidgets import QFileDialog, QHBoxLayout, QLineEdit, QPushButton, QWidget

from pipeline.core import CreateGroupBox, Stage
from pipeline.svg_writer import WriteSvg


@dataclass
class SvgExportParameters:
    """User-configurable inputs for SvgExportStage: the path to write."""

    filename: str = "contours.svg"


class SvgExportStage(Stage):
    """Writes the selected, calibrated contours (e.g. Rectify.contours) out
    to an SVG file at `parameters.filename` - the format tools like Fusion
    360 expect for sketch import, which is the whole point of a tool named
    SVGGui. Like the rest of export, this only runs when explicitly
    triggered, not on every upstream change.
    """

    def __init__(self) -> None:
        self.parameters = SvgExportParameters()

    def Run(self, contours: dict[int, np.ndarray]) -> None:
        if not contours:
            return
        WriteSvg(self.parameters.filename, contours)

    def CreateWidget(self, on_change: Callable[[], None]) -> QWidget:
        widget, layout = CreateGroupBox("SVG Export")

        row = QHBoxLayout()
        filename_edit = QLineEdit(self.parameters.filename)

        def apply_filename():
            self.parameters.filename = filename_edit.text()

        filename_edit.editingFinished.connect(apply_filename)
        row.addWidget(filename_edit)

        def browse():
            path, _ = QFileDialog.getSaveFileName(widget, "Save SVG", self.parameters.filename, "SVG Files (*.svg)")
            if path:
                filename_edit.setText(path)
                self.parameters.filename = path

        browse_btn = QPushButton("Browse...")
        browse_btn.clicked.connect(browse)
        row.addWidget(browse_btn)

        layout.addLayout(row)
        return widget
