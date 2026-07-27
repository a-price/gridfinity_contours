"""Packing contours into a Gridfinity bin, interactively.

    layout_gui.py test_data/*.svg

Deliberately separate from silhouette.py. That tool captures *one* photo:
one segmentation, one calibration, one set of clicks. Packing wants many
objects, and in practice they arrive from many sessions - the three spoon
fixtures in test_data are three separate captures. A Pack button inside
the capture window could only ever pack what happened to be in the current
frame.

So this window starts where that one leaves off: it loads the contour
dumps and SVGs those sessions wrote, accumulates them, and packs the
collection. The two may eventually meet under a common entry point, but
that workflow is not understood well enough yet to design for.
"""

import argparse
import os
import sys

import numpy as np
from PyQt5.QtCore import QLibraryInfo, Qt
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import (
    QApplication,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from pipeline.core import CreateGroupBox, Pipeline
from pipeline.layout.loading import ReadContours
from pipeline.layout_stage import LayoutStage

# Fix PyQt5 / OpenCV collision
os.environ["QT_QPA_PLATFORM_PLUGIN_PATH"] = QLibraryInfo.location(QLibraryInfo.PluginsPath)

CONTOUR_FILE_FILTER = "Contours (*.json *.svg);;Contour dumps (*.json);;SVG files (*.svg)"


class LayoutGui(QMainWindow):
    """Load contours from several capture sessions, pack them, print the
    result.
    """

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Gridfinity Layout")
        self.setGeometry(100, 100, 1100, 700)

        self.contours: dict[int, np.ndarray] = {}
        self.sources: list[str] = []
        self.layout_stage = LayoutStage()

        self.pipeline = Pipeline()
        self.pipeline.Register("layout", self.pack, downstream=["display"])
        self.pipeline.Register("display", self.update_display)

        self.init_ui()

    def init_ui(self) -> None:
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        layout = QHBoxLayout(main_widget)

        control_panel = QWidget()
        control_layout = QVBoxLayout(control_panel)
        control_layout.setAlignment(Qt.AlignmentFlag.AlignTop)

        control_layout.addWidget(self._CreateSourceWidget())

        # Packing runs on the Pack button inside this group box, never on a
        # parameter edit - see LayoutStage.
        control_layout.addWidget(self.layout_stage.CreateWidget(on_change=lambda: self.pipeline.RunFrom("layout")))

        control_layout.addWidget(self._CreateExportWidget())
        control_layout.addStretch(1)

        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image_label.setStyleSheet("background-color: #2b2b2b;")

        layout.addWidget(control_panel, stretch=1)
        layout.addWidget(self.image_label, stretch=3)

    def _CreateSourceWidget(self) -> QWidget:
        widget, layout = CreateGroupBox("Contours")

        add_button = QPushButton("Add Contour Files...")
        add_button.clicked.connect(self.browse_for_contours)
        layout.addWidget(add_button)

        clear_button = QPushButton("Clear")
        clear_button.clicked.connect(self.clear_contours)
        layout.addWidget(clear_button)

        self.source_label = QLabel()
        self.source_label.setWordWrap(True)
        layout.addWidget(self.source_label)
        self._UpdateSourceLabel()

        return widget

    def _CreateExportWidget(self) -> QWidget:
        widget, layout = CreateGroupBox("Export")

        row = QHBoxLayout()
        self.export_edit = QLineEdit("layout")
        row.addWidget(self.export_edit)

        browse_button = QPushButton("Browse...")
        browse_button.clicked.connect(self.browse_for_export)
        row.addWidget(browse_button)
        layout.addLayout(row)

        self.export_button = QPushButton("Export SVG + PDF")
        self.export_button.clicked.connect(self.export_layout)
        layout.addWidget(self.export_button)

        self.export_label = QLabel("Nothing exported yet.")
        self.export_label.setWordWrap(True)
        layout.addWidget(self.export_label)

        return widget

    # ------------------------------------------------------------- loading

    def load_contours(self, paths: list[str]) -> None:
        """Add every contour in `paths` to what is already loaded.

        Additive rather than replacing, because that is the whole point of
        this window: a bin's worth of objects arrives a session at a time.
        """
        if not paths:
            return

        try:
            loaded = ReadContours(paths)
        except (OSError, ValueError) as error:
            self.source_label.setText(f"Could not load: {error}")
            return

        for points in loaded.values():
            self.contours[len(self.contours)] = points
        self.sources.extend(paths)

        # Anything already packed described the old set.
        self.layout_stage.Clear()
        self._UpdateSourceLabel()
        self.update_display()

    def browse_for_contours(self) -> None:
        paths, _ = QFileDialog.getOpenFileNames(self, "Add Contour Files", "", CONTOUR_FILE_FILTER)
        self.load_contours(list(paths))

    def clear_contours(self) -> None:
        self.contours = {}
        self.sources = []
        self.layout_stage.Clear()
        self._UpdateSourceLabel()
        self.update_display()

    def _UpdateSourceLabel(self) -> None:
        if not self.contours:
            self.source_label.setText("No contours loaded.")
            return
        names = ", ".join(os.path.basename(path) for path in self.sources)
        self.source_label.setText(f"{len(self.contours)} contours from {len(self.sources)} file(s):\n{names}")

    # ------------------------------------------------------------- packing

    def pack(self) -> None:
        self.layout_stage.Run(self.contours)

    # ------------------------------------------------------------- display

    def update_display(self) -> None:
        image = self.layout_stage.Render()
        if image is None:
            self.image_label.setPixmap(QPixmap())
            self.image_label.setText("Load contours, then press Pack.")
            return

        height, width, _ = image.shape
        q_image = QImage(image.tobytes(), width, height, 3 * width, QImage.Format_RGB888).rgbSwapped()
        self.image_label.setPixmap(
            QPixmap.fromImage(q_image).scaled(
                self.image_label.size(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
        )

    # -------------------------------------------------------------- export

    def browse_for_export(self) -> None:
        path, _ = QFileDialog.getSaveFileName(self, "Export Layout", self.export_edit.text(), "SVG Files (*.svg)")
        if path:
            # The stage appends its own extensions, so strip a chosen one
            # rather than writing layout.svg.svg.
            base, extension = os.path.splitext(path)
            self.export_edit.setText(base if extension.lower() in (".svg", ".pdf") else path)

    def export_layout(self) -> None:
        try:
            written = self.layout_stage.Export(self.export_edit.text())
        except (OSError, ValueError) as error:
            self.export_label.setText(f"Could not export: {error}")
            return
        self.export_label.setText("Wrote " + ", ".join(os.path.basename(path) for path in written))


def main() -> None:
    # QApplication strips any Qt-specific flags out of sys.argv in place,
    # so parse our own arguments from what's left.
    app = QApplication(sys.argv)

    parser = argparse.ArgumentParser(description="Pack captured contours into a Gridfinity bin.")
    parser.add_argument("inputs", nargs="*", metavar="FILE", help="contour dumps (.json) or SVGs to load at launch")
    args = parser.parse_args(sys.argv[1:])

    window = LayoutGui()
    window.show()
    if args.inputs:
        window.load_contours([os.path.abspath(os.path.expanduser(path)) for path in args.inputs])
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
