from dataclasses import dataclass
from typing import Callable

import numpy as np
from PyQt5.QtWidgets import QFileDialog, QHBoxLayout, QLineEdit, QPushButton, QWidget

from export.contour_io import SaveContours
from qt_utils.widgets import CreateGroupBox
from pipeline.core import Stage
from export.pdf_writer import WritePdf
from export.svg_writer import WriteSvg


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

    Also writes a same-scale PDF alongside it (`<filename>.pdf`): not every
    SVG viewer/print path honors the SVG's embedded physical units, while
    a PDF's page size is unambiguous - print that one if a printed SVG
    comes out the wrong size.

    And a contour dump (`<filename>.json`), which is what layout_cli.py
    reads. The SVG cannot serve that purpose: it PCA-aligns each contour
    into its own frame and rounds to four decimals for drawing, so it is a
    picture of the contours rather than the contours. Exporting the dump
    here rather than behind its own button is what makes the packer usable
    without re-clicking a photo every time.
    """

    def __init__(self) -> None:
        self.parameters = SvgExportParameters()

    def Run(self, contours: dict[int, np.ndarray]) -> None:
        if not contours:
            return
        WriteSvg(self.parameters.filename, contours)
        WritePdf(f"{self.parameters.filename}.pdf", contours)
        SaveContours(f"{self.parameters.filename}.json", contours)

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
