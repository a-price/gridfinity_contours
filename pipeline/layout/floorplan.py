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
from pipeline.layout.preview import BIN_COLOR, CellBoundaries, ClosedRing, LayoutShapes, OuterFootprint
from pipeline.layout.render import Bordered, RenderLayout, RenderShapes, SideBySide
from export.pdf_writer import Page, WriteShapesPdfPages
from export.svg_writer import Shape

# The drawer edge is the outermost context and the one thing on the page
# you align against a physical edge, so it is drawn heavier than the bin
# outlines inside it but still lighter than the objects, which stay the
# subject.
DRAWER_STROKE_MM = 0.4
DRAWER_COLOR = "#404040"

# A bin held out of the search: one that already exists, printed, in a
# drawer. Darker and heavier than `preview.BIN_COLOR` so it separates from
# the bins around it at a floorplan's scale, where an ordinary bin outline
# lands on about one pixel and a colour change alone would not read.
PINNED_COLOR = "#1f5c2e"
PINNED_STROKE_MM = 0.5

# Screen pixels per millimeter for a whole floorplan. A drawer is several
# bins across, so a single bin's scale would produce an image thousands of
# pixels wide; this keeps a 500x400mm drawer around 700px.
DEFAULT_DRAWER_PIXELS_PER_MM = 1.4


def _Rectangle(width: float, height: float) -> np.ndarray:
    return np.array([[0.0, 0.0], [width, 0.0], [width, height], [0.0, height]], dtype=np.float64)


def PlacedBinShapes(layout: Layout, slot: Slot, parts: dict[int, Part], pinned: bool = False) -> list[Shape]:
    """One bin's drawing, moved to where the assignment put it.

    A turned slot rotates the drawing rather than re-deriving it, using
    the same exact quarter turn a part uses inside a bin. The page goes
    from `w x h` to `h x w` with its corner still at the origin, ready to
    translate onto the lattice.

    A pinned bin is drawn with a heavier, darker outline. It is the same
    drawing - the pin is about the search, not the geometry - restyled so
    that "this one is already printed, do not make it again" is legible
    from across the room, on screen and on the printed sheet alike. Only
    the bin's own outlines change; the objects inside stay the subject.
    """
    shapes, width, height = LayoutShapes(layout, parts)
    size = np.array([width, height])
    offset = np.array(slot.cell, dtype=np.float64) * GRID_PITCH_MM

    placed = []
    for shape in shapes:
        points = RotatePoints(np.asarray(shape.points, dtype=np.float64), 1, size) if slot.turned else shape.points
        moved = replace(shape, points=np.asarray(points, dtype=np.float64) + offset)
        if pinned and shape.stroke == BIN_COLOR:
            moved = replace(moved, stroke=PINNED_COLOR, stroke_width=PINNED_STROKE_MM)
        placed.append(moved)
    return placed


def DrawerPage(
    drawer: Drawer,
    contents: Sequence[tuple[Layout, Slot]],
    parts: dict[int, Part],
    pinned: frozenset[int] = frozenset(),
) -> Page:
    """Everything to draw for one drawer, as a page in millimeters.

    Drawn back to front: the cell grid first so it sits under everything,
    then the drawer edge, then the bins. The grid covers the whole drawer
    rather than only the occupied part, which is the point - it is what
    shows where a free cell is, and free cells are the question you take to
    a drawer floorplan.

    `pinned` names bin ids rather than positions in `contents`, which the
    slots already carry - so a caller does not have to keep a second list
    in step with this one.
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
        shapes.extend(PlacedBinShapes(layout, slot, parts, slot.bin_id in pinned))

    return Page(shapes, width, height)


def FloorplanPages(
    drawers: Sequence[Drawer],
    layouts: dict[int, Layout] | None = None,
    result: AssignmentResult | None = None,
    parts: dict[int, Part] | None = None,
    pinned: frozenset[int] = frozenset(),
) -> list[Page]:
    """One page per drawer, in the order the drawers were given.

    Every drawer gets a page, including an empty one - a drawer with
    nothing in it is a useful thing for a map to say, and dropping it would
    silently renumber the pages against the drawers they describe.

    With no assignment at all, every page is that empty page. That is the
    drawer as it stands the moment you tell the tools you own it, and it is
    worth drawing: the grid of cells is the thing the whole system is
    measured against, and seeing it before anything has been planned is how
    you find out you typed the drawer in wrong.
    """
    layouts = layouts or {}
    parts = parts or {}
    slots = {} if result is None else result.slots

    missing = sorted(set(slots) - set(layouts))
    if missing:
        raise ValueError(f"assignment places bins {missing}, whose layouts were not given")

    contents: list[list[tuple[Layout, Slot]]] = [[] for _ in drawers]
    for bin_id, slot in sorted(slots.items()):
        if not 0 <= slot.drawer < len(drawers):
            raise ValueError(f"bin {bin_id} is assigned to drawer {slot.drawer}, which was not given")
        contents[slot.drawer].append((layouts[bin_id], slot))

    return [DrawerPage(drawer, contents[index], parts, pinned) for index, drawer in enumerate(drawers)]


def RenderFloorplan(
    drawers: Sequence[Drawer],
    layouts: dict[int, Layout] | None = None,
    result: AssignmentResult | None = None,
    parts: dict[int, Part] | None = None,
    pixels_per_mm: float = DEFAULT_DRAWER_PIXELS_PER_MM,
    gap: int = 12,
    pinned: frozenset[int] = frozenset(),
) -> np.ndarray:
    """The whole floorplan as one BGR image: every drawer side by side, and
    any bin not in one of them alongside.

    The screen counterpart of `WriteFloorplanPdf`, and drawn through the
    same `FloorplanPages`, so what a window shows and what the printer
    produces cannot drift apart - the discipline `render.py` already
    follows one level down.

    Side by side rather than one drawer at a time, because watching a bin
    move from one drawer to another is the whole point of showing a
    multi-drawer search, and two separate images could not show it. That
    is the opposite of the printed version, which is one page per drawer
    precisely because a page is a physical thing at 1:1.

    **A bin with no slot is drawn beside the drawers rather than left
    out.** An assignment can be partial in three quite different ways - the
    search is still working, it finished and proved something does not fit,
    or a provisional `FirstFit` had nowhere to put it - and in all three
    the bin is real and the person is looking for it. Dropping it would
    make the picture disagree with the bin count in the status line for
    reasons nobody could see. Bordered, because a loose bin next to a
    drawer would otherwise read as a very small drawer.
    """
    layouts = layouts or {}
    parts = parts or {}
    slots = {} if result is None else result.slots

    pages = FloorplanPages(drawers, layouts, result, parts, pinned)
    images = [RenderShapes(page.shapes, page.width, page.height, pixels_per_mm) for page in pages]
    images.extend(
        Bordered(RenderLayout(layouts[bin_id], parts, pixels_per_mm)) for bin_id in sorted(set(layouts) - set(slots))
    )
    return SideBySide(images, gap)


def WriteFloorplanPdf(
    path: str,
    drawers: Sequence[Drawer],
    layouts: dict[int, Layout],
    result: AssignmentResult,
    parts: dict[int, Part],
    pinned: frozenset[int] = frozenset(),
) -> None:
    """Write the drawer floorplan: one true-scale page per drawer.

    A PDF because the page size is the whole point here, and an SVG's is
    not reliably honoured - see `pdf_writer.WriteShapesPdf`.

    Pinned bins are marked on the sheet as well as on screen, since the
    sheet is what you carry to the printer - and "which of these do I
    already own" is the question you are carrying it to answer.
    """
    WriteShapesPdfPages(path, FloorplanPages(drawers, layouts, result, parts, pinned))
