"""Packing contours into a Gridfinity bin, interactively.

    layout_gui.py test_data/*.svg

Deliberately separate from silhouette_gui.py. That tool captures *one* photo:
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
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtGui import QCloseEvent, QImage, QPixmap
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

from pipeline.core import CreateGroupBox, FixQtOpenCvPluginPath
from pipeline.layout.loading import ReadContours
from pipeline.layout_stage import EXPORT_EXTENSIONS, LayoutStage

FixQtOpenCvPluginPath()

CONTOUR_FILE_FILTER = "Contours (*.json *.svg);;Contour dumps (*.json);;SVG files (*.svg)"


class PackWorker(QThread):
    """Runs a pack off the UI thread.

    Worth the thread rather than pumping the event loop between restarts:
    pumping re-enters every widget mid-computation, so the panel has to be
    disabled to stay safe, and the window still cannot repaint or resize.
    The packer touches no Qt and shares nothing but the stage it was handed,
    which the main thread agrees not to read until `packed` arrives.
    """

    progressed = pyqtSignal(object)  # packer.Progress
    packed = pyqtSignal()

    def __init__(self, stage: LayoutStage, contours: dict[int, np.ndarray]) -> None:
        super().__init__()
        self._stage = stage
        self._contours = contours
        self._cancelled = False

    def Cancel(self) -> None:
        """Ask the search to stop at its next restart.

        A plain flag rather than a lock: one thread only ever writes it and
        one only ever reads it, and a poll that misses by one restart costs
        a fraction of a second.
        """
        self._cancelled = True

    def run(self) -> None:
        self._stage.Run(self._contours, progress=self.progressed.emit, cancelled=lambda: self._cancelled)
        self.packed.emit()


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

        # No `Pipeline` here, unlike the capture window. Its stages run
        # downstream targets as soon as the stage returns, which is exactly
        # wrong for work that finishes on another thread - "display" would
        # fire while the pack was still going. The finished signal is what
        # sequences this instead.
        self._worker: PackWorker | None = None

        self.init_ui()

    def init_ui(self) -> None:
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        layout = QHBoxLayout(main_widget)

        control_panel = QWidget()
        control_layout = QVBoxLayout(control_panel)
        control_layout.setAlignment(Qt.AlignmentFlag.AlignTop)

        # Held so a running pack can freeze the things that would change
        # what it is packing, or export a result that does not exist yet.
        # The Layout box itself stays live - that is where Cancel is.
        self.source_group = self._CreateSourceWidget()
        control_layout.addWidget(self.source_group)

        # Packing runs on the Pack button inside this group box, never on a
        # parameter edit - see LayoutStage.
        control_layout.addWidget(self.layout_stage.CreateWidget(on_change=self.pack, on_cancel=self.cancel_pack))

        self.export_group = self._CreateExportWidget()
        control_layout.addWidget(self.export_group)
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

        # Named from what the stage actually writes, so adding a format
        # cannot leave the button advertising the old set.
        self.export_button = QPushButton("Export " + " + ".join(e.lstrip(".").upper() for e in EXPORT_EXTENSIONS))
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
        """Start a pack on a worker thread. Returns immediately."""
        if self._worker is not None:
            return

        # A snapshot, so loading more contours mid-pack cannot change the
        # set underneath the search.
        self._worker = PackWorker(self.layout_stage, dict(self.contours))
        self._worker.progressed.connect(self._OnProgress)
        self._worker.packed.connect(self._OnPacked)
        self.layout_stage.SetBusy(True)
        self.layout_stage.SetStatus(f"Layout: packing {len(self.contours)} contours...")
        self._SetSourcesEditable(False)
        self._worker.start()

    def cancel_pack(self) -> None:
        worker = self._worker
        if worker is not None:
            worker.Cancel()
            self.layout_stage.SetStatus("Layout: stopping...")

    def _OnProgress(self, progress) -> None:
        self.layout_stage.SetStatus(f"Layout: packing... {progress}")

    def _OnPacked(self) -> None:
        self._worker = None
        self.layout_stage.SetBusy(False)
        self._SetSourcesEditable(True)
        self.layout_stage.RefreshStatus()
        self.update_display()

    def _SetSourcesEditable(self, editable: bool) -> None:
        self.source_group.setEnabled(editable)
        self.export_group.setEnabled(editable)

    def WaitForPack(self, timeout_ms: int = 120000) -> None:
        """Block until any running pack has finished and its signals have
        been delivered.

        Needed on the way out - a QThread still running when its window is
        destroyed terminates the process - and it is how a test drives an
        asynchronous pack without an event loop of its own.
        """
        worker = self._worker
        if worker is None:
            return
        worker.wait(timeout_ms)
        QApplication.processEvents()

    # Parameter named `a0` to match PyQt5's own stub, which declares it
    # that way - anything else reads as an incompatible override.
    def closeEvent(self, a0: QCloseEvent | None) -> None:
        """Stop and join any running pack before the window goes away.

        A QThread still running when its object is destroyed terminates the
        process, so this is not merely tidy.
        """
        self.cancel_pack()
        self.WaitForPack()
        super().closeEvent(a0)

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
        """Choose the *basename* the export writes all its formats under.

        A save dialog naturally offers a filename, so whatever extension
        comes back has to come off again - the stage appends its own, and
        picking `layout.scad` would otherwise write `layout.scad.scad`.
        """
        pattern = " ".join(f"*{extension}" for extension in EXPORT_EXTENSIONS)
        path, _ = QFileDialog.getSaveFileName(
            self, "Export Layout", self.export_edit.text(), f"Layout files ({pattern})"
        )
        if path:
            base, extension = os.path.splitext(path)
            self.export_edit.setText(base if extension.lower() in EXPORT_EXTENSIONS else path)

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
