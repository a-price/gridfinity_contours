"""Qt wiring for the distance-field viewer.

The mirror image of `LayoutStage`, and for the opposite reason. Packing
takes seconds, so that stage runs only on an explicit button; rasterizing
one part and colorizing it takes milliseconds, and watching the field
change as the resolution moves is the entire point of this one. So
`on_change` here means what it means everywhere else in `pipeline.core`:
a parameter settled, redraw.

Only the part being looked at is ever built. Every control on the panel
either changes the raster - `resolution` and, through the clearances,
`pad` - or changes only the picture, and rasterizing the other seventeen
contours somebody happened to load on each tick of a spin box would make
the panel feel broken while the answer sat one part away.
"""

from typing import Callable

import numpy as np
from PyQt5.QtWidgets import QCheckBox, QLabel, QWidget

from qt_utils.widgets import CreateGroupBox, CreateSpinBox
from pipeline.core import Stage
from layout.field import FieldView, PixelToLocal, RenderField
from layout.parameters import LayoutParameters
from layout.part import DISTANT_MM, BuildPart, Part

# Named rather than inline so a test can find the toggle it means to
# click, the same way `layout_stage_test` finds a button by its text.
GRADIENT_LABEL = "Gradient magnitude (shows the creases)"
SAMPLES_LABEL = "Boundary samples (zoom in to separate them)"


class FieldStage(Stage):
    """Rasterizes one contour into a Part and renders its distance field.

    Holds a `LayoutParameters` for what the field *is* and a `FieldView`
    for how it is drawn, which is the same split those two types already
    make: the first changes the raster and forces a rebuild, the second
    changes only the picture.
    """

    def __init__(self, parameters: LayoutParameters | None = None, view: FieldView | None = None) -> None:
        self.parameters = parameters or LayoutParameters()
        self.view = view or FieldView()
        self.contours: dict[int, np.ndarray] = {}
        self.selected: int | None = None
        self.part: Part | None = None
        self.error: str | None = None
        self._status_label: QLabel | None = None

    def Run(self, contours: dict[int, np.ndarray]) -> None:
        """Take a set of contours and rasterize the selected one.

        The selection follows the set rather than being reset by it: a
        person watching contour 3 while nudging the resolution should
        still be watching contour 3 afterwards. It falls back to the first
        contour only when the one being watched is no longer there.
        """
        self.contours = dict(contours)
        if self.selected not in self.contours:
            self.selected = min(self.contours) if self.contours else None
        self._Build()

    def Select(self, part_id: int | None) -> None:
        """Look at a different contour."""
        self.selected = part_id
        self._Build()

    def Clear(self) -> None:
        self.contours = {}
        self.selected = None
        self.part = None
        self.error = None

    def _Build(self) -> None:
        """Rasterize the selected contour, reporting a bad one rather than
        raising.

        A contour too degenerate to rasterize - two points from a stray
        `<polygon>`, say - arrives through a file dialog, and a viewer that
        took the window down over one would be worse than one that said so.
        """
        self.part = None
        self.error = None
        if self.selected is None or self.selected not in self.contours:
            return

        try:
            # The pocket, not the object - this view exists to show what the
            # solver reads, and what the solver reads is the pocket's field.
            # The offset spin box below is wired to the same parameter, so
            # nudging it re-rasterizes rather than only relabelling.
            self.part = BuildPart(
                self.contours[self.selected],
                self.parameters.pocket_offset,
                resolution=self.parameters.resolution,
                pad=self.parameters.pad,
                pocket_resolution=self.parameters.pocket_resolution,
                pocket_simplify=self.parameters.pocket_simplify,
            )
        except ValueError as error:
            self.error = str(error)

    def Render(self) -> np.ndarray | None:
        """The selected part's field as a BGR image, or None if there is
        nothing to show.
        """
        return None if self.part is None else RenderField(self.part, self.view, self.parameters)

    def Probe(self, pixel: tuple[float, float]) -> str:
        """What the field reads at one image pixel, as a line for the
        readout.

        Reported in the units the question is actually asked in: how far
        inside, or how much clearance is left - not a signed number the
        reader has to remember the convention for.
        """
        part = self.part
        if part is None:
            return ""

        point = PixelToLocal(part, pixel, self.view.pixels_per_mm)
        position = f"{point[0]:.1f}, {point[1]:.1f}mm"
        distance = float(part.SampleSdf(point)[0])
        if distance >= DISTANT_MM:
            return f"{position}: off the field"

        # The readout describes whichever view is up, so that a number on
        # the panel and a color on the screen are never about different
        # things.
        if self.view.gradient:
            return f"{position}: gradient {float(np.linalg.norm(part.SampleDerivative(point))):.2f}"
        if distance < 0:
            return f"{position}: {-distance:.2f}mm inside the part"
        return f"{position}: {distance:.2f}mm clear"

    def Summary(self) -> str:
        """What is on screen and what its colors mean.

        The key belongs in text beside the picture rather than drawn into
        it: the numbers move with the parameters, and a legend rasterized
        into the image would have to be re-laid-out at every scale.
        """
        if self.error is not None:
            return f"Field: could not rasterize contour {self.selected} - {self.error}"

        part = self.part
        if part is None:
            return "Field: nothing loaded"

        height, width = part.sdf.shape
        size = part.size
        lines = [
            f"Field: contour {self.selected}, {size[0]:.1f} x {size[1]:.1f}mm",
            f"raster {width}x{height} at {part.resolution:g}mm/px, reaching {part.pad:.2f}mm past the outline",
        ]
        if self.view.gradient:
            lines.append("white where the field is a true distance, dark along its creases")
        else:
            lines.append(f"a contour line every {self.view.band_mm:g}mm")
            lines.append(
                f"red {self.parameters.c_pair_enforced:.2f}mm: no other outline may cross. "
                f"green {self.parameters.spacing_pair:.2f}mm: where spacing aims to hold one"
            )
        return "\n".join(lines)

    def SetStatus(self, text: str) -> None:
        if self._status_label is not None:
            self._status_label.setText(text)

    def RefreshStatus(self) -> None:
        self.SetStatus(self.Summary())

    def CreateWidget(self, on_change: Callable[[], None]) -> QWidget:
        widget, layout = CreateGroupBox("Field")

        clearances = QLabel()

        def show_clearances() -> None:
            clearances.setText(
                f"→ pockets cut {self.parameters.pocket_offset:.2f}mm oversize, then "
                f"{self.parameters.c_pair:.2f}mm dividers and {self.parameters.c_wall:.2f}mm walls"
            )

        def rebuild() -> None:
            self._Build()
            on_change()

        # Resolution and pocket offset are the two that change the raster:
        # the first sets the cell, the second sets the shape being
        # rasterized. It used to reach here indirectly, through the
        # clearances and then `pad`; since D5 it is simply in the geometry.
        def apply_resolution(value: float) -> None:
            self.parameters.resolution = value
            rebuild()

        resolution = CreateSpinBox(
            "Resolution (mm per raster cell):",
            0.05,
            2.0,
            self.parameters.resolution,
            apply_resolution,
            suffix=" mm",
        )
        layout.addLayout(resolution["layout"])

        # The clearances beside it stay read-only, same as LayoutStage's
        # panel: they are what a printable divider and wall measure, not
        # something to type over. What the offset changes here is the
        # outline itself - this view rasterizes the pocket, so the whole
        # picture moves rather than a label.
        def apply_offset(value: float) -> None:
            self.parameters.pocket_offset = value
            show_clearances()
            rebuild()

        offset = CreateSpinBox(
            "Pocket Offset (grows the outline being rasterized):",
            0.0,
            10.0,
            self.parameters.pocket_offset,
            apply_offset,
            suffix=" mm",
        )
        layout.addLayout(offset["layout"])
        show_clearances()
        layout.addWidget(clearances)

        # The rest change only the drawing, so they redraw without
        # rebuilding anything.
        def apply_scale(value: float) -> None:
            self.view.pixels_per_mm = value
            on_change()

        scale = CreateSpinBox(
            "Scale (screen pixels per mm):",
            0.5,
            40.0,
            self.view.pixels_per_mm,
            apply_scale,
            decimals=1,
        )
        layout.addLayout(scale["layout"])

        def apply_band(value: float) -> None:
            self.view.band_mm = value
            on_change()

        band = CreateSpinBox(
            "Contour Lines (spacing):",
            0.1,
            10.0,
            self.view.band_mm,
            apply_band,
            decimals=2,
            suffix=" mm",
        )
        layout.addLayout(band["layout"])

        def apply_gradient(checked: bool) -> None:
            self.view.gradient = checked
            on_change()

        gradient_box = QCheckBox(GRADIENT_LABEL)
        gradient_box.setChecked(self.view.gradient)
        gradient_box.toggled.connect(apply_gradient)
        layout.addWidget(gradient_box)

        def apply_samples(checked: bool) -> None:
            self.view.samples = checked
            on_change()

        samples_box = QCheckBox(SAMPLES_LABEL)
        samples_box.setChecked(self.view.samples)
        samples_box.toggled.connect(apply_samples)
        layout.addWidget(samples_box)

        self._status_label = QLabel(self.Summary())
        self._status_label.setWordWrap(True)
        layout.addWidget(self._status_label)

        return widget
