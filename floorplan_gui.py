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
    QListWidgetItem,
    QMainWindow,
    QPushButton,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from qt_utils.widgets import CreateGroupBox, FixQtOpenCvPluginPath
from layout.container import BASE_GAP_MM, GRID_PITCH_MM
from layout.drawer import ParseDrawer
from layout.loading import ReadContours
from layout.plan import ReadDrawers, SaveDrawers
from panels.floorplan_panel import EXPORT_EXTENSIONS, FloorplanPanel

FixQtOpenCvPluginPath()

# Room for the panel's vertical scroll bar, so appearing it does not
# squeeze the controls it is scrolling.
PANEL_SCROLLBAR_PX = 24

CONTOUR_FILE_FILTER = "Contours (*.json *.svg);;Contour dumps (*.json);;SVG files (*.svg)"
DRAWER_FILE_FILTER = "Drawer lists (*.json)"
SESSION_FILE_FILTER = "Floorplan sessions (*.json)"


class FloorplanWorker(QThread):
    """Runs the whole search off the UI thread.

    The same shape as `layout_gui.PackWorker` and for the same reason,
    but with more riding on it: this search runs for minutes, so the
    window has to stay responsive enough to show progress and accept a
    Stop the entire time. The search touches no Qt and shares nothing but
    the panel it was handed, which the main thread agrees not to read
    until `finished` arrives.
    """

    progressed = pyqtSignal(object)  # plan.Progress
    planned = pyqtSignal()

    def __init__(self, panel: FloorplanPanel, contours: dict[int, np.ndarray]) -> None:
        super().__init__()
        self._panel = panel
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
        self._panel.Run(self._contours, report=self.progressed.emit, cancelled=lambda: self._cancelled)
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
        self.floorplan_panel = FloorplanPanel()

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

        control_layout.addWidget(self.floorplan_panel.CreateWidget(on_change=self.plan, on_cancel=self.cancel_plan))

        self.pin_group = self._CreatePinWidget()
        control_layout.addWidget(self.pin_group)

        self.session_group = self._CreateSessionWidget()
        control_layout.addWidget(self.session_group)

        self.export_group = self._CreateExportWidget()
        control_layout.addWidget(self.export_group)
        control_layout.addStretch(1)

        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image_label.setStyleSheet("background-color: #2b2b2b;")
        # A QLabel holding a pixmap asks for the pixmap's own size, which
        # would make the window's minimum grow with the floorplan it is
        # showing. `_DrawImage` scales to whatever room it is given, so it
        # needs none of its own.
        self.image_label.setMinimumSize(1, 1)

        # The panel scrolls rather than setting the window's minimum
        # height. Six group boxes stacked up want about 1300px, which is
        # taller than a 1080p screen has room for - and a window whose
        # minimum exceeds the display cannot be maximized or made full
        # screen at all. This window in particular is one people leave
        # open and full screen for minutes while the search runs.
        panel_area = QScrollArea()
        panel_area.setWidget(control_panel)
        panel_area.setWidgetResizable(True)
        panel_area.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        panel_area.setMinimumWidth(control_panel.sizeHint().width() + PANEL_SCROLLBAR_PX)

        layout.addWidget(panel_area, stretch=1)
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

    def _CreatePinWidget(self) -> QWidget:
        """Tick the bins to leave alone.

        The bins you have already printed, or a grouping that happens to
        be right for reasons the cell count cannot see - a set of things
        used together, or one bin that has to stay shallow. A pinned bin
        is held out of the search entirely rather than merely preferred,
        so it comes back exactly as it went in.

        A checklist rather than clicking bins in the picture: the bins
        move around between plans, the picture is scaled to fit, and the
        thing being pinned is a *grouping* - which is a list of contents,
        and reads as one.
        """
        widget, layout = CreateGroupBox("Pinned Bins")

        self.pin_list = QListWidget()
        self.pin_list.setMaximumHeight(130)
        self.pin_list.itemChanged.connect(self._OnPinChanged)
        layout.addWidget(self.pin_list)

        buttons = QHBoxLayout()
        for text, slot in (("Pin All", self.pin_all), ("Unpin All", self.unpin_all)):
            button = QPushButton(text)
            button.clicked.connect(slot)
            buttons.addWidget(button)
        layout.addLayout(buttons)

        self.pin_label = QLabel()
        self.pin_label.setWordWrap(True)
        layout.addWidget(self.pin_label)
        self._RefreshPins()

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

        # Named for what comes out rather than for the file types, because
        # what comes out is the point: the drawer map, and a sheet and a
        # solid for every bin on it.
        self.export_button = QPushButton("Export Floorplan + Every Bin")
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
        self.floorplan_panel.Clear()
        self._UpdateSourceLabel()
        self.update_display()

    def browse_for_contours(self) -> None:
        paths, _ = QFileDialog.getOpenFileNames(self, "Add Contour Files", "", CONTOUR_FILE_FILTER)
        self.load_contours(list(paths))

    def clear_contours(self) -> None:
        self.contours = {}
        self.sources = []
        self.floorplan_panel.Clear()
        # The pins go too. They name bins made of parts that no longer
        # exist, and holding them would refuse the next search rather than
        # preserve anything.
        self.floorplan_panel.pinned = []
        self.floorplan_panel.resume = None
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

        self.floorplan_panel.drawers.append(drawer)
        self.drawer_edit.clear()
        self.floorplan_panel.Clear()
        self._RefreshDrawers()
        self.update_display()

    def remove_drawer(self) -> None:
        row = self.drawer_list.currentRow()
        if 0 <= row < len(self.floorplan_panel.drawers):
            del self.floorplan_panel.drawers[row]
            self.floorplan_panel.Clear()
            self._RefreshDrawers()
            self.update_display()

    def load_drawers(self, path: str) -> None:
        try:
            self.floorplan_panel.drawers = ReadDrawers(path)
        except (OSError, ValueError) as error:
            self.drawer_label.setText(f"Could not load: {error}")
            return
        self.floorplan_panel.Clear()
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
            SaveDrawers(path, self.floorplan_panel.drawers)
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
        for drawer in self.floorplan_panel.drawers:
            span = tuple(GRID_PITCH_MM * cells - BASE_GAP_MM for cells in (drawer.width, drawer.height))
            self.drawer_list.addItem(f"{drawer.width} x {drawer.height} cells  ({span[0]:.1f} x {span[1]:.1f}mm)")

        drawers = self.floorplan_panel.drawers
        if not drawers:
            self.drawer_label.setText("No drawers yet - type a size in mm, or load a list.")
            return
        cells = sum(drawer.cells for drawer in drawers)
        self.drawer_label.setText(f"{len(drawers)} drawer(s), {cells} cells of space")

    # ------------------------------------------------------------ pinning

    def _RefreshPins(self) -> None:
        """Rebuild the checklist from the arrangement now on screen.

        Signals are blocked while it is rebuilt, since setting a check
        state fires `itemChanged` and would write the half-built list back
        over the pins it is being built from.
        """
        bins = self.floorplan_panel.Bins()
        pinned = self.floorplan_panel.PinnedIds()

        self.pin_list.blockSignals(True)
        self.pin_list.clear()
        for bin_id in sorted(bins):
            n, m = bins[bin_id].grid
            contents = ", ".join(str(part_id) for part_id in sorted(bins[bin_id].placements))
            item = QListWidgetItem(f"bin {bin_id}  {n}x{m}  holding {contents}")
            item.setData(Qt.ItemDataRole.UserRole, bin_id)
            item.setCheckState(Qt.CheckState.Checked if bin_id in pinned else Qt.CheckState.Unchecked)
            self.pin_list.addItem(item)
        self.pin_list.blockSignals(False)

        if not bins:
            self.pin_label.setText("Plan or load a floorplan, then tick the bins to keep as they are.")
            return
        held = len(self.floorplan_panel.pinned)
        self.pin_label.setText(
            f"{held} of {len(bins)} bin(s) pinned - held out of the next search entirely."
            if held
            else "Nothing pinned; every bin is up for regrouping."
        )

    def _OnPinChanged(self, _item) -> None:
        self._CommitPins()

    def _CommitPins(self) -> None:
        items = [self.pin_list.item(row) for row in range(self.pin_list.count())]
        checked = [
            item.data(Qt.ItemDataRole.UserRole)
            for item in items
            if item is not None and item.checkState() == Qt.CheckState.Checked
        ]
        try:
            self.floorplan_panel.Pin(checked)
        except ValueError as error:
            self.pin_label.setText(str(error))
            return
        self.update_display()

    def pin_all(self) -> None:
        self.floorplan_panel.Pin(sorted(self.floorplan_panel.Bins()))
        self.update_display()

    def unpin_all(self) -> None:
        self.floorplan_panel.Pin([])
        self.update_display()

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
            contours = self.floorplan_panel.Load(path)
        except (OSError, ValueError, KeyError, TypeError) as error:
            self.session_label.setText(f"Could not load: {error}")
            return

        self.contours = dict(contours)
        self.sources = [path]
        panel = self.floorplan_panel
        bins = 0 if panel.resume is None else len(panel.resume.bins)
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
            self.floorplan_panel.Save(path, self.contours)
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
        self._worker = FloorplanWorker(self.floorplan_panel, dict(self.contours))
        self._worker.progressed.connect(self._OnProgress)
        self._worker.planned.connect(self._OnPlanned)
        self.floorplan_panel.SetBusy(True)
        self.floorplan_panel.SetStatus(f"Floorplan: grouping {len(self.contours)} objects...")
        self._SetInputsEditable(False)
        self._worker.start()

    def cancel_plan(self) -> None:
        worker = self._worker
        if worker is not None:
            worker.Cancel()
            self.floorplan_panel.SetStatus("Floorplan: stopping...")

    def _OnProgress(self, progress) -> None:
        """Show the best arrangement found so far, and that it is alive.

        Redrawing on every report is affordable because `BuildPlan`
        throttles them to four a second - the picture is a few hundred
        kilopixels and the search between two reports is a quarter second
        of work.
        """
        self.floorplan_panel.SetProgress(progress)
        self.floorplan_panel.SetStatus(f"Floorplan: {progress}")
        self._DrawImage()

    def _OnPlanned(self) -> None:
        self._worker = None
        self.floorplan_panel.SetBusy(False)
        self._SetInputsEditable(True)
        self.floorplan_panel.RefreshStatus()
        self.update_display()

    def _SetInputsEditable(self, editable: bool) -> None:
        self.source_group.setEnabled(editable)
        self.drawer_group.setEnabled(editable)
        # A pin is an input to the search that is running; the bins it
        # lists are about to be renumbered anyway.
        self.pin_group.setEnabled(editable)
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
        self.floorplan_panel.RefreshStatus()
        self._RefreshPins()
        self._DrawImage()

    def _DrawImage(self) -> None:
        image = self.floorplan_panel.Render()
        if image is None:
            # The only way back is an empty drawer list: a drawer draws
            # itself whether or not anything has been planned into it.
            self.image_label.setPixmap(QPixmap())
            self.image_label.setText("Add a drawer, then load contours and press Plan.")
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
        """Choose the *basename* the export writes under, since the panel
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
        """Write the map and every bin on it.

        A bin whose solid could not be cut is an alert rather than a
        failure: everything else was written, and the message names the
        wall thickness that stopped it so the pocket offset can be
        changed and the export repeated.
        """
        try:
            report = self.floorplan_panel.Export(self.export_edit.text())
        except (OSError, ValueError) as error:
            self.export_label.setText(f"Could not export: {error}")
            return
        self.export_label.setText(str(report))


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
