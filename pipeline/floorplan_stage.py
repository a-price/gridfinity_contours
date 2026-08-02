"""Qt wiring for the whole-library floorplan: parts to bins to drawers.

The third stage in this project and the slowest by a wide margin. Packing
one bin takes seconds, so `LayoutStage` runs on an explicit button;
grouping a real library and then fitting the bins into drawers takes
minutes, which changes what the panel has to do rather than merely how
long it waits. Two things follow:

* **The picture updates while the search runs.** A window that showed
  nothing until the answer arrived would be indistinguishable from a hung
  one for whole minutes. `plan.BuildPlan` reports a throttled best-so-far
  and this holds the most recent, so `Render` always has something honest
  to draw.
* **Stopping has to give you something.** A cancelled search returns the
  best grouping it had found, flagged, rather than nothing - so a person
  who has seen enough can take what is on screen.

The drawers live here rather than in the window because they are an
input to the search in the same way the parameters are, and because the
feedback edge in `BuildPlan` derives the admissible bin footprints from
them - which makes them a parameter of the packing, not a display option.
"""

from typing import Callable

import numpy as np
from PyQt5.QtWidgets import QLabel, QPushButton, QWidget

from pipeline.core import CreateGroupBox, CreateSpinBox, Stage
from pipeline.layout.drawer import Drawer
from pipeline.layout.floorplan import DEFAULT_DRAWER_PIXELS_PER_MM, RenderFloorplan, WriteFloorplanPdf
from pipeline.layout.loading import BuildParts
from pipeline.layout.parameters import LayoutParameters
from pipeline.layout.part import Part
from pipeline.layout.grouping import Grouping
from pipeline.layout.plan import BuildPlan, Progress, StoragePlan
from pipeline.layout.session import Changes, LoadSession, SaveSession, Verify
from pipeline.layout.render import RenderLayouts

# One page per drawer, so a floorplan is a PDF and not an SVG - see
# `pdf_writer.WriteShapesPdf` on why the page size has to be unambiguous.
EXPORT_EXTENSIONS = (".pdf",)


class FloorplanStage(Stage):
    """Groups parts into bins, fits those bins into the drawers, and draws
    whatever the search has found so far.

    Holds the drawers alongside the `LayoutParameters` because both are
    inputs to the same search: `BuildPlan` narrows the grid-size search to
    footprints these drawers can actually hold before grouping starts.
    """

    def __init__(self, parameters: LayoutParameters | None = None, drawers: list[Drawer] | None = None) -> None:
        self.parameters = parameters or LayoutParameters()
        self.drawers: list[Drawer] = list(drawers or [])
        self.plan: StoragePlan | None = None
        self.progress: Progress | None = None
        self.error: str | None = None
        # Held so a mid-search frame can be drawn before a StoragePlan
        # exists. Rebuilding them per repaint would rasterize the whole
        # library to draw one frame.
        self._parts: dict[int, Part] = {}
        # The floorplan a loaded session brought with it, and the thing
        # the next search resumes from rather than rediscovers. Held
        # across Clear, which drops results and not inputs - a reloaded
        # session is an input.
        self.resume: Grouping | None = None
        self.changes: tuple[list[int], list[int]] | None = None
        self._status_label: QLabel | None = None
        self._plan_button: QPushButton | None = None
        self._cancel_button: QPushButton | None = None

    def Run(
        self,
        contours: dict[int, np.ndarray],
        report: Callable[[Progress], None] | None = None,
        cancelled: Callable[[], bool] | None = None,
    ) -> None:
        """Build parts from `contours`, group them, and fit them into the
        drawers.

        Touches no widgets, so it is safe to call from a worker thread -
        which is how the window runs it. Progress leaves through the
        callback rather than being written to the label here, because a
        label written from the wrong thread is undefined behavior in Qt
        rather than merely bad practice.

        A refusal from the search - no contours, no drawers, a part no
        allowed bin can hold - is recorded rather than raised. All three
        arrive from a file dialog or a text box, and none of them should
        take the window down.
        """
        resume = self.resume
        self.Clear()
        if not contours or not self.drawers:
            self.error = "load some contours and add at least one drawer"
            return

        try:
            self._parts = BuildParts(contours, self.parameters)
            self.plan = BuildPlan(
                self._parts, self.drawers, self.parameters, report=report, cancelled=cancelled, start=resume
            )
            if resume is not None and self.plan.grouping is not None:
                self.changes = Changes(resume, self.plan.grouping)
        except ValueError as error:
            self.error = str(error)

    def Clear(self) -> None:
        """Drop the current result. Inputs - the drawers, and the session
        being resumed - survive, since they are what a rerun runs *on*.
        """
        self.plan = None
        self.progress = None
        self.error = None
        self.changes = None
        self._parts = {}

    def SetProgress(self, progress: Progress) -> None:
        """Hold the latest report, so `Render` can draw a search still
        running.
        """
        self.progress = progress

    def Render(self, pixels_per_mm: float = DEFAULT_DRAWER_PIXELS_PER_MM) -> np.ndarray | None:
        """The best answer so far as a BGR image, or None if there is not
        one yet.

        Three pictures for three states, because the search genuinely has
        three different kinds of answer at different times and drawing a
        floorplan before anything has been assigned would be drawing an
        empty drawer as though it were the result. Once the plan exists it
        is the floorplan; while the drawer search runs it is the partial
        floorplan; while grouping runs it is the bins themselves, which
        exist before any drawer has been chosen for them.
        """
        plan = self.plan
        if plan is not None and plan.assignment is not None:
            return RenderFloorplan(plan.drawers, plan.layouts, plan.assignment, plan.parts, pixels_per_mm)

        progress = self.progress
        if progress is None or not progress.bins:
            return None

        # A report can outlive the parts it describes - clearing the stage
        # while a stale report is still held, say. This is a repaint path,
        # so it declines to draw rather than raising up through Qt.
        parts = self._ProgressParts()
        if not set(part_id for layout in progress.bins for part_id in layout.placements) <= set(parts):
            return None

        layouts = dict(enumerate(progress.bins))
        if progress.assignment is not None:
            return RenderFloorplan(tuple(self.drawers), layouts, progress.assignment, parts, pixels_per_mm)
        return RenderLayouts(list(progress.bins), parts, pixels_per_mm=pixels_per_mm)

    def _ProgressParts(self) -> dict[int, Part]:
        """The parts a mid-search drawing needs.

        A running or cancelled search has no finished `StoragePlan`, so
        these come from the set `Run` built on its way in.
        """
        return self.plan.parts if self.plan is not None else self._parts

    def Save(self, path: str, contours: dict[int, np.ndarray]) -> None:
        """Write the current floorplan and its inputs to a session file.

        Takes the contours because the window owns them and a
        `StoragePlan` cannot give them back - `BuildPart` PCA-aligns and
        resamples on the way in, so a part is not a contour.
        """
        plan = self.plan
        if plan is None or not plan.layouts:
            raise ValueError("nothing planned to save")
        SaveSession(path, plan, contours, self.parameters)

    def Load(self, path: str) -> dict[int, np.ndarray]:
        """Read a session back, returning its contours for the window to
        adopt.

        The floorplan is reconstructed as saved rather than re-solved, so
        it can be looked at - and printed - without running anything. It
        also becomes the arrangement the next search resumes from, which
        is the whole point of having saved it.

        Verified on the way in against the parameters it was made with. A
        session outlives its settings, and one whose clearances no longer
        hold should say so rather than look settled.
        """
        session = LoadSession(path)

        self.Clear()
        self.parameters = session.parameters
        self.drawers = list(session.drawers)
        self.resume = session.grouping
        self._parts = BuildParts(session.contours, session.parameters)

        problems = Verify(session, self._parts)
        if problems:
            self.error = f"{len(problems)} clearance problem(s) in the saved floorplan: {problems[0]}"

        layouts = dict(enumerate(session.grouping.bins))
        self.plan = StoragePlan(
            drawers=tuple(session.drawers),
            parts=self._parts,
            layouts=layouts,
            assignment=session.assignment,
            grouping=session.grouping,
            footprints={index: layout.grid for index, layout in layouts.items()},
        )
        return session.contours

    def Export(self, basename: str) -> list[str]:
        """Write the floorplan: one true-scale page per drawer.

        Lives here rather than in the window because the writer needs the
        drawers, the layouts, the assignment and the parts, and this is
        what holds all four.
        """
        plan = self.plan
        if plan is None or plan.assignment is None:
            raise ValueError("nothing planned to export")

        (path,) = (f"{basename}{extension}" for extension in EXPORT_EXTENSIONS)
        WriteFloorplanPdf(path, plan.drawers, plan.layouts, plan.assignment, plan.parts)
        return [path]

    def Summary(self) -> str:
        """One block for the panel: what was found, or why nothing was."""
        if self.error is not None:
            return f"Floorplan: {self.error}"

        plan = self.plan
        if plan is None:
            if self.progress is not None:
                return f"Floorplan: {self.progress}"
            return "Floorplan: nothing planned yet"

        if plan.cancelled:
            if not plan.layouts:
                return "Floorplan: stopped before an arrangement was found"
            return (
                f"Floorplan: stopped with {len(plan.layouts)} bins / {plan.cells} cells - the drawers were not searched"
            )

        assignment = plan.assignment
        summary = f"Floorplan: {len(plan.layouts)} bins / {plan.cells} cells across {len(plan.drawers)} drawer(s)"
        if assignment is None or not assignment.placed:
            reason = "" if assignment is None else f" - {assignment.detail}"
            return f"{summary}\nnot placed{reason}"

        summary = f"{summary}\n{plan.Report().splitlines()[-1]}"
        if self.changes is not None:
            # The question a resumed session is actually asked. A bin the
            # search could not improve is already sitting in the drawer.
            kept, changed = self.changes
            summary += f"\nresumed: {len(kept)} bin(s) unchanged, {len(changed)} to reprint"
            if changed:
                summary += " (" + ", ".join(str(index) for index in changed) + ")"
        return summary

    def SetStatus(self, text: str) -> None:
        """Put arbitrary text in the panel - progress, while a search runs.

        Called from the main thread only; the window marshals the worker's
        reports onto it through a signal.
        """
        if self._status_label is not None:
            self._status_label.setText(text)

    def RefreshStatus(self) -> None:
        self.SetStatus(self.Summary())

    def SetBusy(self, busy: bool) -> None:
        """Swap the panel between "can start a search" and "can stop one"."""
        if self._plan_button is not None:
            self._plan_button.setEnabled(not busy)
        if self._cancel_button is not None:
            self._cancel_button.setEnabled(busy)

    def CreateWidget(self, on_change: Callable[[], None], on_cancel: Callable[[], None] | None = None) -> QWidget:
        widget, layout = CreateGroupBox("Floorplan")

        clearances = QLabel()

        def show_clearances() -> None:
            clearances.setText(
                f"→ {self.parameters.c_pair:.2f}mm between parts, {self.parameters.c_wall:.2f}mm to the wall"
            )

        def apply_offset(value: float) -> None:
            self.parameters.pocket_offset = value
            show_clearances()

        offset = CreateSpinBox(
            "Pocket Offset (how much larger than the object):",
            0.0,
            10.0,
            self.parameters.pocket_offset,
            apply_offset,
            suffix=" mm",
        )
        layout.addLayout(offset["layout"])
        show_clearances()
        layout.addWidget(clearances)

        def apply_max_grid(value: float) -> None:
            self.parameters.max_grid = int(value)

        max_grid = CreateSpinBox(
            "Max Bin Size (cells per side):",
            1,
            12,
            self.parameters.max_grid,
            apply_max_grid,
            decimals=0,
        )
        layout.addLayout(max_grid["layout"])

        def apply_seed(value: float) -> None:
            self.parameters.seed = int(value)

        seed = CreateSpinBox(
            "Seed (a different one is a different attempt):",
            0,
            9999,
            self.parameters.seed,
            apply_seed,
            decimals=0,
        )
        layout.addLayout(seed["layout"])

        # The only thing that starts a search. Parameter edits deliberately
        # do not, as in LayoutStage and for the same reason several times
        # over - this one takes minutes.
        self._plan_button = QPushButton("Plan")
        self._plan_button.clicked.connect(on_change)
        layout.addWidget(self._plan_button)

        self._cancel_button = QPushButton("Stop")
        self._cancel_button.setEnabled(False)
        if on_cancel is None:
            self._cancel_button.setVisible(False)
        else:
            self._cancel_button.clicked.connect(on_cancel)
        layout.addWidget(self._cancel_button)

        self._status_label = QLabel(self.Summary())
        self._status_label.setWordWrap(True)
        layout.addWidget(self._status_label)

        return widget
