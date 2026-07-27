"""Qt wiring for the packer.

Explicit trigger only, like SvgExportStage. Packing takes seconds, and
the panel's controls are exactly the kind a person drags - wiring them to
rerun on change would freeze the window on every tick of the seed box.
`on_change` here therefore means "the user pressed Pack", not "a
parameter moved".
"""

from typing import Callable

import numpy as np
from PyQt5.QtWidgets import QApplication, QLabel, QPushButton, QWidget

from pipeline.core import CreateGroupBox, CreateSpinBox, Stage
from pipeline.layout.energy import LayoutParameters
from pipeline.layout.loading import BuildParts
from pipeline.layout.packer import Pack, PackResult
from pipeline.layout.part import Part
from pipeline.layout.preview import WriteLayoutPdf, WriteLayoutSvg
from pipeline.layout.render import DEFAULT_PIXELS_PER_MM, RenderLayout


class LayoutStage(Stage):
    """Packs the rectified contours into the smallest Gridfinity bin that
    holds them, and renders the result for the image view.

    Holds a `LayoutParameters` directly rather than wrapping an algorithm
    object, since packing is a function rather than a stateful thing to
    configure.
    """

    def __init__(self, parameters: LayoutParameters | None = None) -> None:
        self.parameters = parameters or LayoutParameters()
        self.result: PackResult | None = None
        self.parts: dict[int, Part] = {}
        self._status_label: QLabel | None = None
        self._pack_button: QPushButton | None = None

    def Run(self, contours: dict[int, np.ndarray]) -> None:
        """Pack `contours` (real-world mm, e.g. Rectify.contours).

        Called from the Pack button, never from a parameter edit.
        """
        self.Clear()
        if not contours:
            # Packed nothing, having tried nothing - which Summary tells
            # apart from never having been asked.
            self.result = PackResult(None, [])
            self._SetStatus(self.Summary())
            return

        self._SetStatus(f"Layout: packing {len(contours)} contours...")
        self._SetBusy(True)
        try:
            self.parts = BuildParts(contours, self.parameters)
            self.result = Pack(self.parts, self.parameters)
        finally:
            self._SetBusy(False)
        self._SetStatus(self.Summary())

    def Clear(self) -> None:
        """Drop the current layout, so the image view goes back to showing
        the photo.
        """
        self.result = None
        self.parts = {}

    @property
    def layout(self):
        """The solved layout, or None if nothing has packed."""
        return self.result.layout if self.result else None

    def Render(self, pixels_per_mm: float = DEFAULT_PIXELS_PER_MM) -> np.ndarray | None:
        """The current layout as a BGR image, or None if there is nothing
        to show.
        """
        layout = self.layout
        return None if layout is None else RenderLayout(layout, self.parts, pixels_per_mm)

    def Export(self, basename: str) -> list[str]:
        """Write the layout as `<basename>.svg` and `<basename>.pdf`,
        returning what was written.

        Lives here rather than in the window because the writers need both
        the layout and the parts it placed, and this is what holds them.
        """
        layout = self.layout
        if layout is None:
            raise ValueError("nothing packed to export")

        svg_path, pdf_path = f"{basename}.svg", f"{basename}.pdf"
        WriteLayoutSvg(svg_path, layout, self.parts)
        WriteLayoutPdf(pdf_path, layout, self.parts)
        return [svg_path, pdf_path]

    def Summary(self) -> str:
        """One line for the panel: what was packed, or why nothing was.

        A failure names the last thing tried rather than saying only "no
        layout" - "every size up to 6x6 was too small" and "the search ran
        out of attempts" call for completely different responses from the
        user, and the packer already distinguishes them.

        The single source of truth for the panel's text, including the
        cases where there is no result at all: a status the label held but
        this did not would resurface as the wrong message the moment the
        widget was rebuilt.
        """
        if self.result is None:
            return "Layout: nothing packed yet"

        layout = self.result.layout
        if layout is None:
            attempts = self.result.attempts
            if not attempts:
                return "Layout: no contours selected"
            reason = attempts[-1].detail
            return f"Layout: no fit up to {self.parameters.max_grid}x{self.parameters.max_grid} - {reason}"

        n, m = layout.grid
        summary = f"Layout: {len(self.parts)} parts in {n}x{m} ({layout.cells} cells)"
        if self.result.skipped:
            smaller = ", ".join(f"{a.grid[0]}x{a.grid[1]}" for a in self.result.skipped)
            summary += f"\n{smaller} was not ruled out - a tighter packing may exist"
        return summary

    def _SetStatus(self, text: str) -> None:
        if self._status_label is not None:
            self._status_label.setText(text)
            # Repaint now: the pack that follows blocks the event loop for
            # seconds, and a "packing..." label that only appears once the
            # packing has finished is worse than none.
            QApplication.processEvents()

    def _SetBusy(self, busy: bool) -> None:
        """Disable Pack while a pack is running.

        Necessary because _SetStatus pumps the event loop, which would
        otherwise let a second click re-enter Run in the middle of the
        first.
        """
        if self._pack_button is not None:
            self._pack_button.setEnabled(not busy)
            QApplication.processEvents()

    def CreateWidget(self, on_change: Callable[[], None]) -> QWidget:
        widget, layout = CreateGroupBox("Layout")

        clearances = QLabel()

        def show_clearances() -> None:
            clearances.setText(
                f"→ {self.parameters.c_pair:.2f}mm between parts, {self.parameters.c_wall:.2f}mm to the wall"
            )

        # Pocket offset drives both clearances rather than exposing them
        # separately (D5): two pockets `c_pair` apart leave a divider of
        # `c_pair - 2*offset`, so independent spin boxes would let a
        # perfectly reasonable-looking pair of numbers produce a divider
        # too thin to print. The derived values are shown read-only.
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
            "Max Grid Size (cells per side):",
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

        # The only thing that triggers a pack. Parameter edits deliberately
        # do not call on_change - see the module docstring.
        self._pack_button = QPushButton("Pack")
        self._pack_button.clicked.connect(on_change)
        layout.addWidget(self._pack_button)

        self._status_label = QLabel(self.Summary())
        self._status_label.setWordWrap(True)
        layout.addWidget(self._status_label)

        return widget
