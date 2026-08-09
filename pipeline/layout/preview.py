"""Drawing a solved layout at true scale, to print and lay on a bin.

The page is the bin's *outer* footprint rather than its interior, so the
sheet can be checked against a real bin edge-to-edge: print it, set the
bin on it, and the printed outline should disappear under the rim. The
interior outline and cell boundaries are drawn inside that, so a part
sitting suspiciously close to a wall is visible rather than merely
implied.

**Each part is two outlines.** Solid is the pocket that will be cut;
dashed is the object it was cut for, sitting `pocket_offset` inside it.
Drawing only one was a choice between two different questions the sheet
gets asked - *will this print* wants the pocket, and *did the capture
come out right* wants the object laid under a real tool - so it now
answers both. The gap between the two lines is the offset, at true scale,
which is the only place in the pipeline you can actually see it.

There is deliberately no coordinate flip anywhere in here. The layout
frame is whatever frame the input contours arrived in, and svg_writer
writes that frame straight onto the page; adding a flip here would mirror
every part relative to the export that has already been checked against
real objects. A mirrored outline is the one error a printed template
cannot survive - it measures correctly and still will not fit, because a
tool reflected is a tool upside down in its pocket (D1).
"""

import numpy as np

from pipeline.layout.container import (
    BASE_GAP_MM,
    GRID_PITCH_MM,
    OUTER_CORNER_RADIUS_MM,
    Container,
)
from pipeline.layout.part import Part
from pipeline.layout.placement import Layout
from pipeline.pdf_writer import WriteShapesPdf
from pipeline.svg_writer import Shape, WriteShapesSvg

# Parts are the subject and are drawn heaviest; the bin is context. The
# cell grid is faintest because it is the one thing that is not physically
# present on the printed sheet's subject - it marks where the base's feet
# fall, not anything you can see looking into an empty bin.
PART_STROKE_MM = 0.3
BIN_STROKE_MM = 0.2
GRID_STROKE_MM = 0.15

PART_COLOR = "black"
BIN_COLOR = "#808080"
GRID_COLOR = "#c0c0c0"

INTERIOR_DASHES = (2.0, 1.0)
GRID_DASHES = (1.0, 1.0)

# The object drawn inside its pocket. Shorter runs than either annotation
# dash, so it reads as dotted against the interior's long dashes rather
# than as a third kind of boundary.
#
# It has to stay distinct from GRID_DASHES for a second reason: the dash
# pattern is how `preview_test` picks cell boundaries out of the shape
# list, so an object contour sharing that tuple would be counted as one.
OBJECT_DASHES = (0.6, 0.6)


def OuterFootprint(n: int, m: int) -> tuple[float, float]:
    """The bin's outer size in mm - what a caliper across the rim reads.

    Smaller than `42 * cells` by the standard gap that keeps neighbouring
    bins from binding in a drawer.
    """
    return GRID_PITCH_MM * n - BASE_GAP_MM, GRID_PITCH_MM * m - BASE_GAP_MM


def ClosedRing(points: np.ndarray) -> np.ndarray:
    """A ring with its first point repeated, so it draws closed as an open
    polyline. Public because the drawer floorplan draws the same
    annotations one level up.

    Annotations are written as `<polyline>` rather than `<polygon>` on
    purpose: layout.loading.LoadSvgContours reads only `<polygon>`, so a
    preview reads back as exactly the parts in it, with the bin and its
    grid ignored rather than mistaken for two more objects to pack.
    """
    return np.vstack([points, points[:1]])


def CellBoundaries(n: int, m: int, page: tuple[float, float]) -> list[Shape]:
    """The lines between grid cells, in page coordinates.

    A cell boundary sits at a whole multiple of the pitch in grid
    coordinates, and the footprint starts half the gap into the first cell,
    hence the offset - without it the grid would drift a quarter millimeter
    per cell against the bin it is drawn on.
    """
    width, height = page
    offset = BASE_GAP_MM / 2.0

    shapes = []
    for index in range(1, n):
        x = GRID_PITCH_MM * index - offset
        shapes.append(np.array([[x, 0.0], [x, height]]))
    for index in range(1, m):
        y = GRID_PITCH_MM * index - offset
        shapes.append(np.array([[0.0, y], [width, y]]))

    return [
        Shape(points, closed=False, stroke=GRID_COLOR, stroke_width=GRID_STROKE_MM, dashes=GRID_DASHES)
        for points in shapes
    ]


def LayoutShapes(layout: Layout, parts: dict[int, Part]) -> tuple[list[Shape], float, float]:
    """Everything to draw for `layout`, plus the page size in mm.

    Bin-local coordinates (origin at the interior's minimum corner) become
    page coordinates by translating out to the outer footprint's corner -
    the interior is inset from the rim by the same amount on every side, so
    this is a pure translation with no scaling and nothing to get subtly
    wrong.
    """
    missing = sorted(set(layout.placements) - set(parts))
    if missing:
        raise ValueError(f"layout places parts {missing}, which were not given")

    n, m = layout.grid
    width, height = OuterFootprint(n, m)
    inset = np.array([layout.inset, layout.inset])

    rim = Container(width=width, height=height, radius=OUTER_CORNER_RADIUS_MM)
    shapes = [Shape(ClosedRing(rim.Polygon()), closed=False, stroke=BIN_COLOR, stroke_width=BIN_STROKE_MM)]
    shapes.extend(CellBoundaries(n, m, (width, height)))
    shapes.append(
        Shape(
            ClosedRing(layout.Envelope() + inset),
            closed=False,
            stroke=BIN_COLOR,
            stroke_width=BIN_STROKE_MM,
            dashes=INTERIOR_DASHES,
        )
    )

    for part_id in sorted(layout.placements):
        placement = layout.placements[part_id]
        part = parts[part_id]
        shapes.append(Shape(placement.ToWorld(part) + inset, stroke=PART_COLOR, stroke_width=PART_STROKE_MM))

        # Both contours, because a printed sheet answers two questions and
        # they have different answers. The solid line is the pocket - what
        # gets cut, and what has to clear its neighbours. The dashed one is
        # the object, and it is the line you lay the real tool against to
        # check the capture before committing to a print. Same weight and
        # colour: neither is annotation, and which is which is carried by
        # the dash rather than by importance.
        #
        # Drawn open like every other non-part outline, so
        # `loading.LoadSvgContours` still reads a written preview back as
        # exactly the shapes that will be cut, rather than as twice as many
        # objects to pack.
        #
        # Skipped where the offset is zero and the two are the same
        # polygon - a dashed line printed exactly along a solid one reads
        # as a rendering fault, not as information.
        if part.pocket_offset > 0.0:
            shapes.append(
                Shape(
                    ClosedRing(placement.ObjectToWorld(part) + inset),
                    closed=False,
                    stroke=PART_COLOR,
                    stroke_width=PART_STROKE_MM,
                    dashes=OBJECT_DASHES,
                )
            )

    return shapes, width, height


def WriteLayoutSvg(path: str, layout: Layout, parts: dict[int, Part]) -> None:
    """Write a true-scale preview of `layout` as an SVG."""
    shapes, width, height = LayoutShapes(layout, parts)
    WriteShapesSvg(path, shapes, width, height)


def WriteLayoutPdf(path: str, layout: Layout, parts: dict[int, Part]) -> None:
    """Write a true-scale preview of `layout` as a PDF - the one to print,
    since a PDF's page size is unambiguous where an SVG's is not.
    """
    shapes, width, height = LayoutShapes(layout, parts)
    WriteShapesPdf(path, shapes, width, height)
