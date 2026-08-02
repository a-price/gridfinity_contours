"""The floorplan for a whole library of objects: which bin, which drawer.

    floorplan_gui.py test_data/*.svg --drawers drawers.json

The top of the stack, and the first front end that speaks in more than one
bin. `layout_gui.py` packs the objects you give it into *a* bin; this one
decides which objects should share a bin at all, how many bins that takes,
and which drawer each bin goes in - then draws the floorplan you print and
lay in the drawer, which is what it is named for.

That makes it the slowest thing here by a wide margin: grouping is a
discrete search whose cost function is a stochastic packer, and a real
library is minutes rather than seconds. The window is built around that
fact. The search runs on a worker thread, reports the best arrangement it
has found every quarter second, and can be stopped - and stopping hands
back the best grouping it had, because someone who has watched it for two
minutes and seen the answer stop improving should be able to keep it.

Drawers are typed either way - `500x400` is a measurement, `11x9 cells`
is already counted - and always stored in cells. That conversion is the
one thing in this window that loses information: 500mm holds the same 11
cells as 504mm does. See `drawer.DrawerCells`.
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
    QListWidget,
    QMainWindow,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from pipeline.core import CreateGroupBox, FixQtOpenCvPluginPath
from pipeline.layout.container import BASE_GAP_MM, GRID_PITCH_MM
from pipeline.layout.drawer import ParseDrawer
from pipeline.layout.loading import ReadContours
from pipeline.layout.plan import ReadDrawers, SaveDrawers
from pipeline.floorplan_stage import EXPORT_EXTENSIONS, FloorplanStage

FixQtOpenCvPluginPath()

CONTOUR_FILE_FILTER = "Contours (*.json *.svg);;Contour dumps (*.json);;SVG files (*.svg)"
DRAWER_FILE_FILTER = "Drawer lists (*.json)"
SESSION_FILE_FILTER = "Floorplan sessions (*.json)"


class FloorplanWorker(QThread):
    """Runs the whole search off the UI thread.

    The same shape as `layout_gui.PackWorker` and for the same reason,
    but with more riding on it: this search runs for minutes, so the
    window has to stay responsive enough to show progress and accept a
    Stop the entire time. The search touches no Qt and shares nothing but
    the stage it was handed, which the main thread agrees not to read
    until `finished` arrives.
    """

    progressed = pyqtSignal(object)  # plan.Progress
    planned = pyqtSignal()

    def __init__(self, stage: FloorplanStage, contours: dict[int, np.ndarray]) -> None:
        super().__init__()
        self._stage = stage
        self._contours = contours
        self._cancelled = False

    def Cancel(self) -> None:
        """Ask the search to stop at its next report.

        A plain flag rather than a lock: one thread only ever writes it and
        one only ever reads it, and the searches poll often enough that a
        missed read costs a fraction of a second.
        """
        self._cancelled = True

    def run(self) -> None:
        self._stage.Run(self._contours, report=self.progressed.emit, cancelled=lambda: self._cancelled)
        self.planned.emit()


class FloorplanGui(QMainWindow):
    """Load contours, say what drawers you have, and plan where
    everything goes.
    """

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Gridfinity Floorplan")
        self.setGeometry(100, 100, 1300, 800)

        self.contours: dict[int, np.ndarray] = {}
        self.sources: list[str] = []
        self.floorplan_stage = FloorplanStage()

        self._worker: FloorplanWorker | None = None

        self.init_ui()

    def init_ui(self) -> None:
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        layout = QHBoxLayout(main_widget)

        control_panel = QWidget()
        control_layout = QVBoxLayout(control_panel)
        control_layout.setAlignment(Qt.AlignmentFlag.AlignTop)

        # Frozen while a search runs: both change what is being searched.
        self.source_group = self._CreateSourceWidget()
        self.drawer_group = self._CreateDrawerWidget()
        control_layout.addWidget(self.source_group)
        control_layout.addWidget(self.drawer_group)

        control_layout.addWidget(self.floorplan_stage.CreateWidget(on_change=self.plan, on_cancel=self.cancel_plan))

        self.session_group = self._CreateSessionWidget()
        control_layout.addWidget(self.session_group)

        self.export_group = self._CreateExportWidget()
        control_layout.addWidget(self.export_group)
        control_layout.addStretch(1)

        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image_label.setStyleSheet("background-color: #2b2b2b;")

        layout.addWidget(control_panel, stretch=1)
        layout.addWidget(self.image_label, stretch=3)

        self.update_display()

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

    def _CreateDrawerWidget(self) -> QWidget:
        """The drawers, typed in either unit or loaded from a saved list.

        Millimeters are what a tape measure reads and cells are what a
        saved list holds, so the box takes both and the list always shows
        both. That the two are shown together is what makes the lossy step
        visible rather than implied - a 500mm drawer and a 504mm drawer
        are the same eleven cells, and somebody wondering why should be
        able to see it.
        """
        widget, layout = CreateGroupBox("Drawers")

        row = QHBoxLayout()
        self.drawer_edit = QLineEdit()
        self.drawer_edit.setPlaceholderText("500x400  or  11x9 cells")
        self.drawer_edit.returnPressed.connect(self.add_drawer)
        row.addWidget(self.drawer_edit)

        add_button = QPushButton("Add")
        add_button.clicked.connect(self.add_drawer)
        row.addWidget(add_button)
        layout.addLayout(row)

        self.drawer_list = QListWidget()
        self.drawer_list.setMaximumHeight(110)
        layout.addWidget(self.drawer_list)

        buttons = QHBoxLayout()
        for text, slot in (
            ("Remove", self.remove_drawer),
            ("Load...", self.browse_for_drawers),
            ("Save...", self.save_drawers),
        ):
            button = QPushButton(text)
            button.clicked.connect(slot)
            buttons.addWidget(button)
        layout.addLayout(buttons)

        self.drawer_label = QLabel()
        self.drawer_label.setWordWrap(True)
        layout.addWidget(self.drawer_label)
        self._RefreshDrawers()

        return widget

    def _CreateSessionWidget(self) -> QWidget:
        """Save the floorplan, and pick it up again months later.

        The flow this exists for: a new tool arrives, you load the
        floorplan you already printed, add the one contour, and press
        Plan. The search resumes from what you had instead of
        rediscovering it, so the bins already in your drawer stay as they
        are and the panel says which ones have to come off the printer
        again.
        """
        widget, layout = CreateGroupBox("Session")

        row = QHBoxLayout()
        for text, slot in (("Save...", self.save_session), ("Load...", self.browse_for_session)):
            button = QPushButton(text)
            button.clicked.connect(slot)
            row.addWidget(button)
        layout.addLayout(row)

        self.session_label = QLabel("No session loaded.")
        self.session_label.setWordWrap(True)
        layout.addWidget(self.session_label)

        return widget

    def _CreateExportWidget(self) -> QWidget:
        widget, layout = CreateGroupBox("Export")

        row = QHBoxLayout()
        self.export_edit = QLineEdit("floorplan")
        row.addWidget(self.export_edit)

        browse_button = QPushButton("Browse...")
        browse_button.clicked.connect(self.browse_for_export)
        row.addWidget(browse_button)
        layout.addLayout(row)

        self.export_button = QPushButton("Export " + " + ".join(e.lstrip(".").upper() for e in EXPORT_EXTENSIONS))
        self.export_button.clicked.connect(self.export_plan)
        layout.addWidget(self.export_button)

        self.export_label = QLabel("Nothing exported yet.")
        self.export_label.setWordWrap(True)
        layout.addWidget(self.export_label)

        return widget

    # ------------------------------------------------------------- contours

    def load_contours(self, paths: list[str]) -> None:
        """Add every contour in `paths` to what is already loaded.

        Additive, like `layout_gui`, and more so: this window's whole
        subject is a library assembled from many capture sessions.
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

        # Anything already found described the old library.
        self.floorplan_stage.Clear()
        self._UpdateSourceLabel()
        self.update_display()

    def browse_for_contours(self) -> None:
        paths, _ = QFileDialog.getOpenFileNames(self, "Add Contour Files", "", CONTOUR_FILE_FILTER)
        self.load_contours(list(paths))

    def clear_contours(self) -> None:
        self.contours = {}
        self.sources = []
        self.floorplan_stage.Clear()
        self._UpdateSourceLabel()
        self.update_display()

    def _UpdateSourceLabel(self) -> None:
        if not self.contours:
            self.source_label.setText("No contours loaded.")
            return
        names = ", ".join(os.path.basename(path) for path in self.sources)
        self.source_label.setText(f"{len(self.contours)} contours from {len(self.sources)} file(s):\n{names}")

    # -------------------------------------------------------------- drawers

    def add_drawer(self) -> None:
        """Add the drawer in the text box, measured in millimeters."""
        text = self.drawer_edit.text().strip()
        if not text:
            return
        try:
            drawer = ParseDrawer(text)
        except ValueError as error:
            self.drawer_label.setText(str(error))
            return

        self.floorplan_stage.drawers.append(drawer)
        self.drawer_edit.clear()
        self.floorplan_stage.Clear()
        self._RefreshDrawers()
        self.update_display()

    def remove_drawer(self) -> None:
        row = self.drawer_list.currentRow()
        if 0 <= row < len(self.floorplan_stage.drawers):
            del self.floorplan_stage.drawers[row]
            self.floorplan_stage.Clear()
            self._RefreshDrawers()
            self.update_display()

    def load_drawers(self, path: str) -> None:
        try:
            self.floorplan_stage.drawers = ReadDrawers(path)
        except (OSError, ValueError) as error:
            self.drawer_label.setText(f"Could not load: {error}")
            return
        self.floorplan_stage.Clear()
        self._RefreshDrawers()
        self.update_display()

    def browse_for_drawers(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Load Drawers", "", DRAWER_FILE_FILTER)
        if path:
            self.load_drawers(path)

    def save_drawers(self) -> None:
        path, _ = QFileDialog.getSaveFileName(self, "Save Drawers", "drawers.json", DRAWER_FILE_FILTER)
        if not path:
            return
        try:
            SaveDrawers(path, self.floorplan_stage.drawers)
        except (OSError, ValueError) as error:
            self.drawer_label.setText(f"Could not save: {error}")
            return
        self.drawer_label.setText(f"Wrote {os.path.basename(path)}")

    def _RepopulateDrawers(self) -> None:
        """Redraw the drawer panel after something replaced the list
        wholesale - loading a session, which brings its own drawers."""
        self._RefreshDrawers()

    def _RefreshDrawers(self) -> None:
        """Rebuild the list, showing each drawer in cells and in the
        millimeters those cells actually span.

        The span rather than what was typed: `42*n - 0.5` is what the bins
        will occupy, and the difference between that and the drawer you
        measured is the room the floorplan will not reach into.
        """
        self.drawer_list.clear()
        for drawer in self.floorplan_stage.drawers:
            span = tuple(GRID_PITCH_MM * cells - BASE_GAP_MM for cells in (drawer.width, drawer.height))
            self.drawer_list.addItem(f"{drawer.width} x {drawer.height} cells  ({span[0]:.1f} x {span[1]:.1f}mm)")

        drawers = self.floorplan_stage.drawers
        if not drawers:
            self.drawer_label.setText("No drawers yet - type a size in mm, or load a list.")
            return
        cells = sum(drawer.cells for drawer in drawers)
        self.drawer_label.setText(f"{len(drawers)} drawer(s), {cells} cells of space")

    # -------------------------------------------------------------- session

    def load_session(self, path: str) -> None:
        """Adopt a saved floorplan: its contours, drawers, parameters, and
        the arrangement itself.

        Replaces rather than accumulating, unlike loading contours. A
        session is a whole state, and merging two of them would produce a
        grouping describing parts from both with ids that meant different
        things in each.
        """
        try:
            contours = self.floorplan_stage.Load(path)
        except (OSError, ValueError, KeyError, TypeError) as error:
            self.session_label.setText(f"Could not load: {error}")
            return

        self.contours = dict(contours)
        self.sources = [path]
        stage = self.floorplan_stage
        bins = 0 if stage.resume is None else len(stage.resume.bins)
        self.session_label.setText(
            f"Resuming {os.path.basename(path)}: {len(self.contours)} objects in {bins} bins.\n"
            "Add a contour and press Plan to fit it in."
        )
        self._RepopulateDrawers()
        self._UpdateSourceLabel()
        self.update_display()

    def browse_for_session(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Load Session", "", SESSION_FILE_FILTER)
        if path:
            self.load_session(path)

    def save_session(self) -> None:
        path, _ = QFileDialog.getSaveFileName(self, "Save Session", "floorplan.json", SESSION_FILE_FILTER)
        if not path:
            return
        try:
            self.floorplan_stage.Save(path, self.contours)
        except (OSError, ValueError) as error:
            self.session_label.setText(f"Could not save: {error}")
            return
        self.session_label.setText(f"Wrote {os.path.basename(path)}")

    # ------------------------------------------------------------ searching

    def plan(self) -> None:
        """Start planning on a worker thread. Returns immediately."""
        if self._worker is not None:
            return

        # A snapshot, so loading more contours mid-search cannot change the
        # library underneath it.
        self._worker = FloorplanWorker(self.floorplan_stage, dict(self.contours))
        self._worker.progressed.connect(self._OnProgress)
        self._worker.planned.connect(self._OnPlanned)
        self.floorplan_stage.SetBusy(True)
        self.floorplan_stage.SetStatus(f"Floorplan: grouping {len(self.contours)} objects...")
        self._SetInputsEditable(False)
        self._worker.start()

    def cancel_plan(self) -> None:
        worker = self._worker
        if worker is not None:
            worker.Cancel()
            self.floorplan_stage.SetStatus("Floorplan: stopping...")

    def _OnProgress(self, progress) -> None:
        """Show the best arrangement found so far, and that it is alive.

        Redrawing on every report is affordable because `BuildPlan`
        throttles them to four a second - the picture is a few hundred
        kilopixels and the search between two reports is a quarter second
        of work.
        """
        self.floorplan_stage.SetProgress(progress)
        self.floorplan_stage.SetStatus(f"Floorplan: {progress}")
        self._DrawImage()

    def _OnPlanned(self) -> None:
        self._worker = None
        self.floorplan_stage.SetBusy(False)
        self._SetInputsEditable(True)
        self.floorplan_stage.RefreshStatus()
        self.update_display()

    def _SetInputsEditable(self, editable: bool) -> None:
        self.source_group.setEnabled(editable)
        self.drawer_group.setEnabled(editable)
        self.session_group.setEnabled(editable)
        self.export_group.setEnabled(editable)

    def WaitForPlan(self, timeout_ms: int = 600000) -> None:
        """Block until any running search has finished and its signals have
        been delivered.

        Needed on the way out - a QThread still running when its window is
        destroyed terminates the process - and it is how a test drives an
        asynchronous search without an event loop of its own. The timeout
        is generous because this search legitimately takes minutes.
        """
        worker = self._worker
        if worker is None:
            return
        worker.wait(timeout_ms)
        QApplication.processEvents()

    # Parameter named `a0` to match PyQt5's own stub, which declares it
    # that way - anything else reads as an incompatible override.
    def closeEvent(self, a0: QCloseEvent | None) -> None:
        """Stop and join any running search before the window goes away.

        A QThread still running when its object is destroyed terminates the
        process, so this is not merely tidy.
        """
        self.cancel_plan()
        self.WaitForPlan()
        super().closeEvent(a0)

    # -------------------------------------------------------------- display

    def update_display(self) -> None:
        self.floorplan_stage.RefreshStatus()
        self._DrawImage()

    def _DrawImage(self) -> None:
        image = self.floorplan_stage.Render()
        if image is None:
            self.image_label.setPixmap(QPixmap())
            self.image_label.setText("Load contours, add a drawer, then press Plan.")
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

    # --------------------------------------------------------------- export

    def browse_for_export(self) -> None:
        """Choose the *basename* the export writes under, since the stage
        appends its own extension.
        """
        pattern = " ".join(f"*{extension}" for extension in EXPORT_EXTENSIONS)
        path, _ = QFileDialog.getSaveFileName(
            self, "Export Floorplan", self.export_edit.text(), f"Floorplan ({pattern})"
        )
        if path:
            base, extension = os.path.splitext(path)
            self.export_edit.setText(base if extension.lower() in EXPORT_EXTENSIONS else path)

    def export_plan(self) -> None:
        try:
            written = self.floorplan_stage.Export(self.export_edit.text())
        except (OSError, ValueError) as error:
            self.export_label.setText(f"Could not export: {error}")
            return
        self.export_label.setText("Wrote " + ", ".join(os.path.basename(path) for path in written))


def main() -> None:
    # QApplication strips any Qt-specific flags out of sys.argv in place,
    # so parse our own arguments from what's left.
    app = QApplication(sys.argv)

    parser = argparse.ArgumentParser(description="Plan a floorplan for a library of objects across your drawers.")
    parser.add_argument("inputs", nargs="*", metavar="FILE", help="contour dumps (.json) or SVGs to load at launch")
    parser.add_argument("--session", metavar="FILE", help="a saved floorplan to resume at launch")
    parser.add_argument("--drawers", metavar="FILE", help="a saved drawer list to load at launch")
    parser.add_argument(
        "--drawer",
        action="append",
        metavar="WxH",
        help="a drawer's interior, in mm or as '11x9 cells'; repeat for several",
    )
    args = parser.parse_args(sys.argv[1:])

    window = FloorplanGui()
    window.show()
    if args.session:
        window.load_session(os.path.abspath(os.path.expanduser(args.session)))
    if args.drawers:
        window.load_drawers(os.path.abspath(os.path.expanduser(args.drawers)))
    for text in args.drawer or []:
        window.drawer_edit.setText(text)
        window.add_drawer()
    if args.inputs:
        window.load_contours([os.path.abspath(os.path.expanduser(path)) for path in args.inputs])
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
