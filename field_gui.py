"""Looking at the signed distance field a contour gets packed by.

    field_gui.py test_data/*.svg

The layout search has two phases and both of them read one thing: the
distance field `BuildPart` rasterizes for each contour. `solver` prices
every candidate arrangement by sampling one part's boundary against
another's field, and `spacing` evens out the resulting gaps against the
same numbers. Neither draws it, so until now the only way to ask whether
a field was what it looked like was to infer it from where the parts
ended up.

This window loads the same contour files `layout_gui.py` does and shows
one field at a time, at whatever resolution and clearance the layout would
have used. It packs nothing and writes nothing - it is a magnifying glass,
not a stage in the pipeline, which is why it neither exports nor needs a
worker thread.
"""

import argparse
import os
import sys

import numpy as np
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QImage, QMouseEvent, QPixmap
from PyQt5.QtWidgets import (
    QApplication,
    QComboBox,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QPushButton,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from qt_utils.click_recorder import WidgetToImageCoords
from qt_utils.widgets import CreateGroupBox, FixQtOpenCvPluginPath
from panels.field_panel import FieldPanel
from layout.loading import ReadContours

FixQtOpenCvPluginPath()

CONTOUR_FILE_FILTER = "Contours (*.json *.svg);;Contour dumps (*.json);;SVG files (*.svg)"


class FieldGui(QMainWindow):
    """Load contours, pick one, look at its distance field."""

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Distance Field")
        self.setGeometry(100, 100, 1100, 700)

        self.contours: dict[int, np.ndarray] = {}
        self.sources: list[str] = []
        self.field_panel = FieldPanel()

        # The rendered image's shape, which the readout needs to undo the
        # pixmap's letterboxing. Held rather than re-derived because the
        # scale it was drawn at may since have moved.
        self._image_shape: tuple = (0, 0, 3)

        self.init_ui()

    def init_ui(self) -> None:
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        layout = QHBoxLayout(main_widget)

        control_panel = QWidget()
        control_layout = QVBoxLayout(control_panel)
        control_layout.setAlignment(Qt.AlignmentFlag.AlignTop)

        control_layout.addWidget(self._CreateSourceWidget())
        control_layout.addWidget(self.field_panel.CreateWidget(on_change=self.update_display))
        control_layout.addStretch(1)

        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image_label.setStyleSheet("background-color: #2b2b2b;")

        # A field viewer whose numbers can only be estimated from a color
        # is half a tool: the readout is what turns "that looks about
        # right" into a millimeter.
        self.image_label.setMouseTracking(True)
        self.image_label.mouseMoveEvent = self.image_hovered

        # Shown at 1:1 and scrolled, never scaled to fit. Every annotation
        # this view draws - the outline, the clearance rings, the contour
        # lines - is one pixel wide, by design, so that a line stays a line
        # where the field goes flat. Scaling a 3400px field down to an
        # 800px panel would drop most of them: nearest-neighbour deletes
        # whole stretches, and smoothing invents a gradient that is not in
        # the raster. The Scale control on the panel is the zoom, and it
        # resamples the field itself rather than a picture of it.
        self.scroll_area = QScrollArea()
        self.scroll_area.setWidget(self.image_label)
        self.scroll_area.setWidgetResizable(True)

        self.readout_label = QLabel()
        self.readout_label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        view_panel = QWidget()
        view_layout = QVBoxLayout(view_panel)
        view_layout.addWidget(self.scroll_area, stretch=1)
        view_layout.addWidget(self.readout_label)

        layout.addWidget(control_panel, stretch=1)
        layout.addWidget(view_panel, stretch=3)

        self.update_display()

    def _CreateSourceWidget(self) -> QWidget:
        widget, layout = CreateGroupBox("Contours")

        add_button = QPushButton("Add Contour Files...")
        add_button.clicked.connect(self.browse_for_contours)
        layout.addWidget(add_button)

        clear_button = QPushButton("Clear")
        clear_button.clicked.connect(self.clear_contours)
        layout.addWidget(clear_button)

        # Which contour is on screen is a property of what is loaded, so
        # it lives here beside the file list rather than in the panel's
        # own controls - the panel decides how a field is built and drawn,
        # not
        # which one.
        self.contour_box = QComboBox()
        self.contour_box.currentIndexChanged.connect(self.select_contour)
        layout.addWidget(self.contour_box)

        self.source_label = QLabel()
        self.source_label.setWordWrap(True)
        layout.addWidget(self.source_label)
        self._UpdateSourceLabel()

        return widget

    # ------------------------------------------------------------- loading

    def load_contours(self, paths: list[str]) -> None:
        """Add every contour in `paths` to what is already loaded.

        Additive, like `layout_gui`: contours arrive a capture session at
        a time, and comparing one session's field against another's is a
        reason this window exists.
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

        self._RepopulateContourBox()
        self._UpdateSourceLabel()
        self.refresh()

    def browse_for_contours(self) -> None:
        paths, _ = QFileDialog.getOpenFileNames(self, "Add Contour Files", "", CONTOUR_FILE_FILTER)
        self.load_contours(list(paths))

    def clear_contours(self) -> None:
        self.contours = {}
        self.sources = []
        self.field_panel.Clear()
        self._RepopulateContourBox()
        self._UpdateSourceLabel()
        self.refresh()

    def _RepopulateContourBox(self) -> None:
        """Rebuild the selector, keeping the selection where it was.

        Signals are blocked across the rebuild because clearing a combo
        box emits a change to index -1, which would otherwise deselect the
        contour being looked at every time another file was added.
        """
        previous = self.field_panel.selected
        self.contour_box.blockSignals(True)
        self.contour_box.clear()
        for contour_id in sorted(self.contours):
            self.contour_box.addItem(f"Contour {contour_id}", contour_id)
        if previous in self.contours:
            self.contour_box.setCurrentIndex(self.contour_box.findData(previous))
        self.contour_box.blockSignals(False)

    def _UpdateSourceLabel(self) -> None:
        if not self.contours:
            self.source_label.setText("No contours loaded.")
            return
        names = ", ".join(os.path.basename(path) for path in self.sources)
        self.source_label.setText(f"{len(self.contours)} contours from {len(self.sources)} file(s):\n{names}")

    def select_contour(self, index: int) -> None:
        self.field_panel.Select(self.contour_box.itemData(index))
        self.update_display()

    # ------------------------------------------------------------- display

    def refresh(self) -> None:
        """Rebuild the field from the current contours, then redraw."""
        self.field_panel.Run(self.contours)
        self.update_display()

    def update_display(self) -> None:
        self.field_panel.RefreshStatus()
        self.readout_label.setText("")

        image = self.field_panel.Render()
        if image is None:
            self.image_label.setPixmap(QPixmap())
            self.image_label.setText("Load contours to see their distance fields.")
            return

        self._image_shape = image.shape
        height, width, _ = image.shape
        q_image = QImage(image.tobytes(), width, height, 3 * width, QImage.Format_RGB888).rgbSwapped()
        self.image_label.setPixmap(QPixmap.fromImage(q_image))

    def image_hovered(self, ev: QMouseEvent | None) -> None:
        """Report what the field reads under the pointer."""
        if ev is None or self.field_panel.part is None:
            return

        coords = WidgetToImageCoords(self.image_label, self._image_shape, ev)
        self.readout_label.setText("" if coords is None else self.field_panel.Probe(coords))


def main() -> None:
    # QApplication strips any Qt-specific flags out of sys.argv in place,
    # so parse our own arguments from what's left.
    app = QApplication(sys.argv)

    parser = argparse.ArgumentParser(description="View the signed distance field of a captured contour.")
    parser.add_argument("inputs", nargs="*", metavar="FILE", help="contour dumps (.json) or SVGs to load at launch")
    args = parser.parse_args(sys.argv[1:])

    window = FieldGui()
    window.show()
    if args.inputs:
        window.load_contours([os.path.abspath(os.path.expanduser(path)) for path in args.inputs])
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
