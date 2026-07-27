"""Drawing a whole drawer at true scale: the bins in it, and the objects
in those bins.

The floorplan you print, lay in the drawer, and check against - one page
per drawer, because a page is a physical thing at 1:1 and nothing you can
print at actual size fits two drawers side by side.

It draws `preview.LayoutShapes` rather than walking each layout itself,
which is the same discipline `render.py` follows for the screen: the bin
you see on the floorplan and the bin whose own sheet you print have to be
the same drawing, or one of them is lying. All this module adds is where
each bin sits - a quarter turn if the assignment turned it, then a
translation onto the 42mm lattice.

**The page is the cells, not the drawer.** A `Drawer` knows how many whole
grid cells it holds and not the millimeters it was measured from, so the
page is the footprint of a `W x H` block of bins - `42*W - 0.5` by
`42*H - 0.5`. A real drawer is usually a little larger than that, and the
slack does not appear here. Lay the sheet into a corner rather than
expecting it to reach the far wall.
"""

from dataclasses import replace
from typing import Sequence

import numpy as np

from pipeline.layout.container import GRID_PITCH_MM
from pipeline.layout.drawer import AssignmentResult, Drawer, Slot
from pipeline.layout.part import Part
from pipeline.layout.placement import Layout, RotatePoints
from pipeline.layout.preview import CellBoundaries, ClosedRing, LayoutShapes, OuterFootprint
from pipeline.pdf_writer import Page, WriteShapesPdfPages
from pipeline.svg_writer import Shape

# The drawer edge is the outermost context and the one thing on the page
# you align against a physical edge, so it is drawn heavier than the bin
# outlines inside it but still lighter than the objects, which stay the
# subject.
DRAWER_STROKE_MM = 0.4
DRAWER_COLOR = "#404040"


def _Rectangle(width: float, height: float) -> np.ndarray:
    return np.array([[0.0, 0.0], [width, 0.0], [width, height], [0.0, height]], dtype=np.float64)


def PlacedBinShapes(layout: Layout, slot: Slot, parts: dict[int, Part]) -> list[Shape]:
    """One bin's drawing, moved to where the assignment put it.

    A turned slot rotates the whole drawing a quarter turn rather than
    re-deriving it: `RotatePoints` with orientation 1 is the same exact
    operation a part uses to turn inside a bin, so the bin's page rotates
    from `w x h` to `h x w` with its corner still at the origin, ready to
    translate onto the lattice.
    """
    shapes, width, height = LayoutShapes(layout, parts)
    size = np.array([width, height])
    offset = np.array(slot.cell, dtype=np.float64) * GRID_PITCH_MM

    placed = []
    for shape in shapes:
        points = RotatePoints(np.asarray(shape.points, dtype=np.float64), 1, size) if slot.turned else shape.points
        placed.append(replace(shape, points=np.asarray(points, dtype=np.float64) + offset))
    return placed


def DrawerPage(drawer: Drawer, contents: Sequence[tuple[Layout, Slot]], parts: dict[int, Part]) -> Page:
    """Everything to draw for one drawer, as a page in millimeters.

    Drawn back to front: the cell grid first so it sits under everything,
    then the drawer edge, then the bins. The grid covers the whole drawer
    rather than only the occupied part, which is the point - it is what
    shows where a free cell is, and free cells are the question you take to
    a drawer floorplan.
    """
    width, height = OuterFootprint(drawer.width, drawer.height)

    shapes = list(CellBoundaries(drawer.width, drawer.height, (width, height)))
    shapes.append(
        Shape(
            ClosedRing(_Rectangle(width, height)),
            closed=False,
            stroke=DRAWER_COLOR,
            stroke_width=DRAWER_STROKE_MM,
        )
    )
    for layout, slot in contents:
        shapes.extend(PlacedBinShapes(layout, slot, parts))

    return Page(shapes, width, height)


def FloorplanPages(
    drawers: Sequence[Drawer],
    layouts: dict[int, Layout],
    result: AssignmentResult,
    parts: dict[int, Part],
) -> list[Page]:
    """One page per drawer, in the order the drawers were given.

    Every drawer gets a page, including an empty one - a drawer with
    nothing in it is a useful thing for a map to say, and dropping it would
    silently renumber the pages against the drawers they describe.
    """
    missing = sorted(set(result.slots) - set(layouts))
    if missing:
        raise ValueError(f"assignment places bins {missing}, whose layouts were not given")

    contents: list[list[tuple[Layout, Slot]]] = [[] for _ in drawers]
    for bin_id, slot in sorted(result.slots.items()):
        if not 0 <= slot.drawer < len(drawers):
            raise ValueError(f"bin {bin_id} is assigned to drawer {slot.drawer}, which was not given")
        contents[slot.drawer].append((layouts[bin_id], slot))

    return [DrawerPage(drawer, contents[index], parts) for index, drawer in enumerate(drawers)]


def WriteFloorplanPdf(
    path: str,
    drawers: Sequence[Drawer],
    layouts: dict[int, Layout],
    result: AssignmentResult,
    parts: dict[int, Part],
) -> None:
    """Write the drawer floorplan: one true-scale page per drawer.

    A PDF rather than an SVG because the page size is what makes this
    useful, and an SVG's physical size is not honoured by every viewer or
    print path - the same reason the per-bin preview is printed as PDF.
    """
    WriteShapesPdfPages(path, FloorplanPages(drawers, layouts, result, parts))
