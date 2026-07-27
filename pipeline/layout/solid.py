"""Turning a solved layout into a printable Gridfinity bin.

Emits OpenSCAD that builds a bin of the layout's grid size and cuts one
pocket per part, each the part's own outline grown by the print
tolerance.

The tolerance is applied *here* rather than in the layout. A layout
reserves enough room for it (D5 derives the clearances from
`pocket_offset`), but the actual dilation is a property of the printer and
the fit you want, so changing your mind about it re-cuts the solid instead
of invalidating the arrangement. `ThinnestWalls` is what keeps that
freedom honest: ask for a tolerance the layout did not budget for and the
dividers between pockets go too thin to print, so this refuses rather
than emitting a bin that fails on the bed.

Two details in here are easy to get wrong and impossible to notice until
the print is finished - see `_PocketPoints` and `_Cutout`.
"""

import os

import numpy as np

from pipeline.layout.container import DIVIDER_WIDTH_MM, MIN_WALL_MM, InteriorSpan
from pipeline.layout.energy import LayoutParameters
from pipeline.layout.part import Part
from pipeline.layout.placement import Layout
from pipeline.layout.spacing import Gaps
from pipeline.layout.verify import DistanceToBoundary

# The vendored gridfinity-rebuilt-openscad submodule, as an absolute path.
# Absolute because a generated .scad is routinely written somewhere other
# than the repository root, and a relative include would resolve against
# wherever it landed.
LIBRARY_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "gridfinity-rebuilt-openscad", "src", "core")
)

# Gridfinity heights come in multiples of the base height, which is the
# unit the spec and every printable profile are quoted in.
HEIGHT_UNIT_MM = 7.0
BASE_HEIGHT_MM = 7.0
DEFAULT_HEIGHT_UNITS = 3


def ThinnestWalls(layout: Layout, parts: dict[int, Part], pocket_offset: float) -> tuple[float, float]:
    """The thinnest divider between two pockets and the thinnest bin wall,
    in mm, that `pocket_offset` would leave.

    Each pocket is its part grown by the offset, so a gap of `g` between
    two parts becomes a divider of `g - 2*offset`, and a part sitting `w`
    from the interior wall leaves `w - offset` of bin wall. Both shrink
    twice as fast as intuition suggests on the divider, which is exactly
    why this is worth computing rather than eyeballing.

    Returns `inf` for whichever does not apply - a single part has no
    dividers.
    """
    params = LayoutParameters()
    gaps = Gaps(parts, layout.placements, params)
    separations = [slack + params.c_pair for slack in gaps.values()]
    divider = min((gap - 2.0 * pocket_offset for gap in separations), default=np.inf)

    envelope = layout.Envelope()
    wall = np.inf
    for part_id, placement in layout.placements.items():
        distance = float(np.min(DistanceToBoundary(placement.ToWorld(parts[part_id]), envelope)))
        wall = min(wall, distance - pocket_offset)

    return float(divider), float(wall)


def _PocketPoints(layout: Layout, parts: dict[int, Part], part_id: int) -> np.ndarray:
    """One part's outline in OpenSCAD's coordinates.

    Two transforms, both load-bearing:

    * The bin is centered on the origin in OpenSCAD, while a layout's
      origin is the interior's minimum corner - hence the half-interior
      shift.
    * **The y axis is flipped.** The layout frame is the one the contours
      arrived in and the one the printed preview draws, which is a page
      frame: y increases downward. OpenSCAD is y-up. Emitting the numbers
      unchanged would mirror every pocket, which measures correctly on
      every axis and still will not hold the tool (D1). With the flip, the
      bin viewed from above matches the sheet viewed on the table.
    """
    interior = np.array(
        [InteriorSpan(layout.grid[0], layout.inset), InteriorSpan(layout.grid[1], layout.inset)],
        dtype=np.float64,
    )
    points = layout.placements[part_id].ToWorld(parts[part_id])
    centered = points - interior / 2.0
    return np.stack([centered[:, 0], -centered[:, 1]], axis=-1)


def _Polygon(points: np.ndarray) -> str:
    return "polygon(points=[" + ", ".join(f"[{x:.4f}, {y:.4f}]" for x, y in points) + "])"


def _Cutout(points: np.ndarray, pocket_offset: float, depth: float) -> str:
    """One pocket, as the cutout geometry `bin_render` expects.

    `bin_render(bin) { ... }` translates its children to the top of the
    infill and subtracts them, so a cutout's own origin is that surface
    and it extends *downward* - which is why this translates by `-depth`
    before extruding.

    Using the library's own cutout mechanism rather than differencing
    against `bin_render(...)` from the outside buys two things. The depth
    is stated: a bare `linear_extrude()` defaults to 100mm and removes
    everything above z=0, so the pocket floor is wherever the base
    happens to start rather than somewhere chosen. And the base is never
    cut in the first place, instead of being cut and then repaired by
    unioning it back on top - which works, but only for as long as the
    caller remembers to do it.
    """
    return (
        f"    translate([0, 0, {-depth:.4f}])\n"
        f"    linear_extrude(height = {depth:.4f})\n"
        f"    offset(r = {pocket_offset:.4f})\n"
        f"    {_Polygon(points)};"
    )


def GenerateScad(
    layout: Layout,
    parts: dict[int, Part],
    pocket_offset: float | None = None,
    height_units: int = DEFAULT_HEIGHT_UNITS,
    pocket_depth: float | None = None,
    library_path: str = LIBRARY_PATH,
) -> str:
    """An OpenSCAD program for the bin this layout describes.

    `pocket_depth` defaults to the full infill, so each object drops to
    the top of the base, and may not exceed it - see below for why that is
    a usability check rather than a safety one.

    Cutouts can only remove infill. `bin_render` unions the base and the
    wall in outside its own difference, so nothing passed as a child can
    reach either, whatever its depth.
    """
    missing = sorted(set(layout.placements) - set(parts))
    if missing:
        raise ValueError(f"layout places parts {missing}, which were not given")
    if height_units < 1:
        raise ValueError(f"height must be at least one 7mm unit, got {height_units}")

    if pocket_offset is None:
        pocket_offset = LayoutParameters().pocket_offset
    if pocket_offset < 0:
        raise ValueError(f"pocket offset must not be negative, got {pocket_offset}")

    height_mm = height_units * HEIGHT_UNIT_MM
    infill = height_mm - BASE_HEIGHT_MM
    if infill <= 0:
        raise ValueError(f"a {height_units}-unit bin is all base, leaving nothing to cut pockets into")

    depth = infill if pocket_depth is None else pocket_depth
    if depth <= 0:
        raise ValueError(f"pocket depth must be positive, got {depth}")
    if depth > infill:
        # Not a safety check - the base and walls are unioned in outside
        # bin_render's own difference, so a cutout can only ever remove
        # infill and an over-deep one is silently clamped. Refused because
        # silently clamping hides the real answer, which is that the bin
        # needs to be taller.
        raise ValueError(
            f"pocket depth {depth}mm exceeds the {infill}mm of infill in a "
            f"{height_units}-unit bin; raise the height rather than the depth"
        )

    divider, wall = ThinnestWalls(layout, parts, pocket_offset)
    if divider < DIVIDER_WIDTH_MM:
        raise ValueError(
            f"a {pocket_offset}mm pocket offset leaves a {divider:.2f}mm divider between pockets, "
            f"under the {DIVIDER_WIDTH_MM}mm minimum - re-pack with a larger pocket_offset "
            "rather than cutting one this layout has no room for"
        )
    if wall < MIN_WALL_MM:
        raise ValueError(
            f"a {pocket_offset}mm pocket offset leaves a {wall:.2f}mm bin wall, " f"under the {MIN_WALL_MM}mm minimum"
        )

    n, m = layout.grid
    cutouts = "\n".join(
        _Cutout(_PocketPoints(layout, parts, part_id), pocket_offset, depth) for part_id in sorted(layout.placements)
    )

    return (
        f"// Generated by gridfinity_contours. {len(layout.placements)} pockets in a {n}x{m} bin.\n"
        f"// Pocket offset {pocket_offset}mm, depth {depth}mm.\n"
        f"include <{library_path}/standard.scad>\n"
        f"use <{library_path}/bin.scad>\n"
        "\n"
        "bin = new_bin(\n"
        f"    grid_size = [{n}, {m}],\n"
        f"    height_mm = {height_mm:.4f},\n"
        "    include_lip = true\n"
        ");\n"
        "\n"
        "// Cutouts are children of bin_render, which places them at the top\n"
        "// of the infill and subtracts them - they extend downward from there.\n"
        "bin_render(bin) {\n"
        f"{cutouts}\n"
        "}\n"
    )


def WriteScad(path: str, layout: Layout, parts: dict[int, Part], **options) -> None:
    """Write the bin for this layout to `path`.

    Generated in full before the file is opened, so a tolerance that gets
    refused leaves no file at all rather than an empty one that looks like
    a bin until you try to render it.
    """
    program = GenerateScad(layout, parts, **options)
    with open(path, "w") as f:
        f.write(program)
