"""Placing several bins' solids in their drawer - the OpenSCAD counterpart
of `layout/floorplan.py`, which places the same bins' 2D drawings on a
drawer page.

Same split that module already follows against `export/`: this computes
*where* things go, using `Layout`, `Slot`, `Drawer`, `AssignmentResult` -
project concepts `export/` never sees - then hands off fully-positioned
`export.scad_writer.ScadPart`s for `WriteScadScene` to serialize. Nothing
here writes a file directly.

The one piece of geometry worth explaining is the coordinate frame. A
`Slot.cell` and everything in `layout/drawer.py`/`layout/floorplan.py` is
page-frame: origin at a drawer's own minimum corner, y increasing
downward - the same frame the printed floorplan and the PDF/SVG writers
use. OpenSCAD is y-up, and `layout.solid._ToOpenScad` already flips once,
per bin, to bridge exactly this gap for a pocket's own coordinates.
Placing a *whole bin* needs the same flip one level up, plus - for a
turned slot - a rotation of the whole bin about its own center, since
`new_bin`/`bin_render` centers a bin on its own local origin (see
`layout.solid._ToOpenScad`'s docstring).

`_WorldPlacement`'s two cases were checked two independent ways - matrix
algebra on `layout.placement.RotatePoints`'s `orientation=1` case, and
explicit corner tracing - and `scene_test.py` checks a third: it transforms
`layout.floorplan.PlacedBinShapes`'s own page-frame output through the same
drawer-level flip and asserts the two agree, rather than trusting the
algebra alone.

**Every drawer lands in one `.scad`, side by side**, the 3D counterpart of
`layout.render.SideBySide` - which lays the same drawers out on screen for
the identical reason: a whole library is one thing to look at, and separate
files could not show it together. `layout.floorplan.WriteFloorplanPdf`
makes the opposite call, one page per drawer, but that is a print-size
argument specific to paper - nothing you can print at 1:1 fits two drawers
on one sheet - and it does not apply to a scene nobody is printing at true
scale. Unlike a bin's position *within* its own drawer, which the drawer
search solved for, one drawer's position relative to another's is not
meaningful - two real drawers share no physical coordinate system - so
`_DrawerOffsets` only has to keep them from overlapping, not represent
anything.
"""

from dataclasses import dataclass, field, replace
from typing import Sequence

from export.scad_writer import ScadPart, WriteScadScene
from layout.container import GRID_PITCH_MM
from layout.drawer import AssignmentResult, Drawer, Slot
from layout.part import Part
from layout.placement import Layout
from layout.plan import StoragePlan
from layout.preview import OuterFootprint
from layout.solid import DEFAULT_HEIGHT_UNITS, LIBRARY_PATH, GenerateBinModule

# Millimetres of daylight between two drawers laid out in the same scene.
# Wide enough to read as a gap rather than a seam; otherwise arbitrary,
# since - unlike the gap between two bins in one drawer - it is not
# reproducing anything about the real drawers.
DRAWER_GAP_MM = 20.0


def _WorldPlacement(drawer: Drawer, footprint: tuple[int, int], slot: Slot) -> tuple[float, float, float]:
    """Where a bin's OpenSCAD module - centered on its own local origin -
    belongs in its drawer's world frame, and the rotation about Z that
    gets it there: `(x, y, degrees)`.

    `footprint` is the bin's own *unturned* grid size (`layout.grid`), not
    `slot.Footprint(footprint)` - the rotation this returns is what
    accounts for a turned slot, not a pre-swapped size.

    An unturned bin's page-frame rectangle, corner at `slot.cell * 42mm`,
    simply gets centered and flipped like anything else crossing this
    frame boundary. A turned one needs a further `-90°` about the bin's own
    center - not "some quarter turn": the sign is fixed by matching
    `layout.placement.RotatePoints`'s `orientation=1` exactly, since that
    is what decided where the *pockets inside* a turned bin actually ended
    up when `layout.floorplan.PlacedBinShapes` drew the 2D floorplan this
    has to agree with.
    """
    n, m = footprint
    w, h = OuterFootprint(n, m)
    drawer_w, drawer_h = OuterFootprint(drawer.width, drawer.height)
    cx, cy = slot.cell

    if slot.turned:
        x = GRID_PITCH_MM * cx + h / 2.0 - drawer_w / 2.0
        y = drawer_h / 2.0 - GRID_PITCH_MM * cy - w / 2.0
        return x, y, -90.0

    x = GRID_PITCH_MM * cx + w / 2.0 - drawer_w / 2.0
    y = drawer_h / 2.0 - GRID_PITCH_MM * cy - h / 2.0
    return x, y, 0.0


def DrawerParts(
    drawer_index: int,
    drawer: Drawer,
    layouts: dict[int, Layout],
    assignment: AssignmentResult,
    parts: dict[int, Part],
    pocket_offset: float | None = None,
    height_units: int = DEFAULT_HEIGHT_UNITS,
) -> tuple[list[ScadPart], dict[int, str]]:
    """Every bin `assignment` puts in `drawer`, as placed OpenSCAD modules,
    plus why any of them could not be cut, by bin id.

    A bin whose pocket offset leaves too little wall or divider to print -
    the same refusal `GenerateScad` makes on its own - is skipped rather
    than sinking every other bin in the drawer; see `SceneReport`. Both are
    empty for a drawer nothing was assigned to.
    """
    placed = []
    problems = {}
    for bin_id, slot in sorted(assignment.slots.items()):
        if slot.drawer != drawer_index:
            continue
        layout = layouts[bin_id]
        name = f"bin_{bin_id}"
        try:
            module = GenerateBinModule(name, layout, parts, pocket_offset, height_units)
        except ValueError as error:
            problems[bin_id] = str(error)
            continue
        x, y, degrees = _WorldPlacement(drawer, layout.grid, slot)
        placed.append(ScadPart(name, module, x, y, degrees))
    return placed, problems


def _DrawerOffsets(drawers: Sequence[Drawer], gap: float = DRAWER_GAP_MM) -> list[float]:
    """Each drawer's X offset for a combined scene, left to right, `gap`mm
    apart - the 3D counterpart of `layout.render.SideBySide`'s image
    layout. Every drawer stays centered on Y=0, since nothing here needs
    the vertical room `SideBySide`'s top-alignment spends on padding a
    shorter image.
    """
    offsets = []
    cursor = 0.0
    for drawer in drawers:
        width, _ = OuterFootprint(drawer.width, drawer.height)
        offsets.append(cursor + width / 2.0)
        cursor += width + gap
    return offsets


def SceneParts(
    plan: StoragePlan,
    assignment: AssignmentResult,
    pocket_offset: float | None = None,
    height_units: int = DEFAULT_HEIGHT_UNITS,
    gap: float = DRAWER_GAP_MM,
) -> tuple[list[ScadPart], dict[int, str]]:
    """Every bin in `plan`, as placed OpenSCAD modules for one combined
    scene holding every drawer `assignment` uses, laid out side by side.

    Each drawer's bins are computed in that drawer's own centered frame
    (`DrawerParts`) and then shifted by `_DrawerOffsets`, so two drawers'
    bins can never collide even though each was placed without any
    knowledge of the other's position in the scene.
    """
    offsets = _DrawerOffsets(plan.drawers, gap)
    combined: list[ScadPart] = []
    problems: dict[int, str] = {}
    for index, drawer in enumerate(plan.drawers):
        here, here_problems = DrawerParts(index, drawer, plan.layouts, assignment, plan.parts, pocket_offset, height_units)
        problems.update(here_problems)
        combined.extend(replace(part, x=part.x + offsets[index]) for part in here)
    return combined, problems


@dataclass(frozen=True)
class SceneReport:
    """What `WriteScene` produced, and what it could not.

    The solid-scene analogue of `panels.floorplan_panel.ExportReport`, and
    for the identical reason: a `.scad` file missing one bin still looks
    complete unless something says which bin and why.
    """

    written: list[str]
    # Why each bin's pocket could not be cut, by bin id - kept apart rather
    # than pre-formatted, so a caller can say which bins without parsing a
    # sentence back apart.
    problems: dict[int, str] = field(default_factory=dict)


def WriteScene(
    path: str,
    plan: StoragePlan,
    assignment: AssignmentResult | None = None,
    library_path: str = LIBRARY_PATH,
    pocket_offset: float | None = None,
    height_units: int = DEFAULT_HEIGHT_UNITS,
    gap: float = DRAWER_GAP_MM,
) -> SceneReport:
    """Write every drawer `plan` places a cuttable bin in to one `.scad`
    scene at `path`, drawers laid out side by side.

    `assignment` defaults to `plan.assignment`, which is `None` for a plan
    whose search stopped before the drawer level ran; pass one explicitly
    in that case, the same way `panels.floorplan_panel.FloorplanPanel.
    _Assigned` computes one on reload rather than leaving a floorplan that
    can be looked at but not exported.

    Writes nothing if every bin turned out to be uncuttable - `problems`
    still reports why, the same as a partial write would.
    """
    if assignment is None:
        assignment = plan.assignment
    if assignment is None:
        raise ValueError("no drawer assignment to place bins with")
    if not plan.layouts:
        raise ValueError("nothing planned to write")

    parts, problems = SceneParts(plan, assignment, pocket_offset, height_units, gap)
    if not parts:
        return SceneReport([], problems)

    WriteScadScene(path, library_path, parts)
    return SceneReport([path], problems)
