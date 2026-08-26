"""Turning a solved layout into a printable Gridfinity bin.

Emits OpenSCAD that builds a bin of the layout's grid size and cuts one
pocket per part - and by this point a pocket is simply a polygon, so
cutting it is `linear_extrude` around what `pocket.PocketContour` already
produced. The dilation used to happen here, as an OpenSCAD `offset(r =
...)` applied to the object's outline; `_Cutout` says what that cost.

Cutting at a tolerance the layout was *not* packed at still works, which
is what `--solid-offset` is for: `PocketPolygons` re-grows each pocket
from `part.object_contour` in the part's own frame, so the arrangement
survives. `ThinnestWalls` keeps that honest - ask for more than the
layout reserved and the dividers go too thin to print, so this refuses
rather than emitting a bin that fails on the bed.

Two details in here are easy to get wrong and impossible to notice until
the print is finished - see `_ToOpenScad` and `_Cutout`.
"""

import os

import numpy as np

from layout.container import DIVIDER_WIDTH_MM, MIN_WALL_MM, InteriorSpan
from layout.parameters import LayoutParameters
from layout.part import Part
from layout.pocket import PocketContour
from layout.placement import Layout
from layout.verify import DistanceToBoundary, MinimumSeparation

# The vendored gridfinity-rebuilt-openscad submodule, as an absolute path.
# Absolute because a generated .scad is routinely written somewhere other
# than the repository root, and a relative include would resolve against
# wherever it landed.
#
# One `..` because this module sits in `layout/`, directly under the
# repository root. It was two while the package lived at `pipeline/layout/`,
# and the extra level pointed the include at the repository's *parent* -
# which OpenSCAD reports only as a warning before rendering an empty file.
#
# Forward slashes rather than the platform separator: OpenSCAD reads a
# backslash inside `include <>` as an escape, so a native Windows path
# silently includes nothing. `/` is accepted on every platform, and this
# is a no-op where the separator is already `/`.
LIBRARY_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "gridfinity-rebuilt-openscad", "src", "core")
).replace(os.sep, "/")

# Gridfinity heights come in multiples of the base height, which is the
# unit the spec and every printable profile are quoted in.
HEIGHT_UNIT_MM = 7.0
BASE_HEIGHT_MM = 7.0
DEFAULT_HEIGHT_UNITS = 3


def _PackedOffset(layout: Layout, parts: dict[int, Part]) -> float:
    """The offset the placed parts were packed at.

    Refuses a mixed set rather than picking one. Everything that goes
    through `loading.BuildParts` shares an offset by construction, so a
    disagreement here means a caller assembled parts by hand from two
    parameter sets, and guessing which of them the bin should be cut at
    is exactly the sort of quiet wrong answer this refactor exists to
    remove.
    """
    offsets = {parts[part_id].pocket_offset for part_id in layout.placements}
    if len(offsets) > 1:
        raise ValueError(f"parts were packed at differing pocket offsets {sorted(offsets)}; pass one explicitly")
    return offsets.pop() if offsets else LayoutParameters().pocket_offset


def PocketPolygons(
    layout: Layout,
    parts: dict[int, Part],
    pocket_offset: float | None = None,
) -> dict[int, np.ndarray]:
    """Every placed pocket in bin coordinates, cut at `pocket_offset`.

    Defaults to the offset the parts were packed at, which is the whole
    of the usual case and costs nothing: the pocket is already geometry
    on the Part, so placing it is a rigid transform and no dilation runs
    at all.

    A *different* offset re-grows each pocket from `part.object_contour`,
    which is what makes cutting at a tolerance the layout was not packed
    at still mean something. It works in the part's own frame, so the
    placement that was solved for the packed pocket carries the re-cut
    one unchanged - `Placement.LocalToWorld` rotates within the packed
    pocket's bounding box, and a pocket cut smaller simply sits inside
    it. Larger is not free, though, and `ThinnestWalls` is what catches
    it: the layout only reserved room for what it packed.

    The re-cut reuses the raster and simplification the *part* was built
    with rather than `pocket`'s defaults, so that an offset means the
    same polygon however it arrived. Taking the defaults instead made
    `--solid-offset 1.0` and `--pocket-offset 1.0` disagree for any
    session that had tuned `pocket_resolution` - the test suite's own
    fixtures among them.
    """
    polygons = {}
    for part_id, placement in layout.placements.items():
        part = parts[part_id]
        if pocket_offset is None or pocket_offset == part.pocket_offset:
            polygons[part_id] = placement.ToWorld(part)
        else:
            recut = PocketContour(part.object_contour, pocket_offset, part.pocket_resolution, part.pocket_simplify)
            polygons[part_id] = placement.LocalToWorld(part, recut)
    return polygons


def _MeasureWalls(polygons: dict[int, np.ndarray], envelope: np.ndarray) -> tuple[float, float]:
    """The thinnest divider and thinnest bin wall among pockets already
    placed in bin coordinates.

    The measuring half of `ThinnestWalls`, split out so that a caller
    holding the polygons can measure *those* rather than have them
    derived a second time. `GenerateScad` is that caller, and under
    `--solid-offset` the second derivation was not free: it re-traced
    every pocket, at about 90ms apiece on a real spoon, to arrive at
    polygons it already had.
    """
    divider = np.inf
    ordered = sorted(polygons)
    for index, id_a in enumerate(ordered):
        for id_b in ordered[index + 1 :]:
            divider = min(divider, MinimumSeparation(polygons[id_a], polygons[id_b]))

    wall = np.inf
    for polygon in polygons.values():
        wall = min(wall, float(np.min(DistanceToBoundary(polygon, envelope))))

    return float(divider), float(wall)


def ThinnestWalls(layout: Layout, parts: dict[int, Part], pocket_offset: float | None = None) -> tuple[float, float]:
    """The thinnest divider between two pockets and the thinnest bin wall,
    in mm, that these pockets leave.

    Measured pocket to pocket, with nothing subtracted. It used to take
    the gap between two *objects* and deduct `2*pocket_offset` for the
    two pockets that would be cut around them, because a part was an
    object and the pockets did not exist yet to be measured. Now they do,
    so this reports what is actually there - which also means it no
    longer has to assume the cut matches the prediction.

    Both measurements read exact polygon geometry - `verify.MinimumSeparation`
    for the divider, `DistanceToBoundary` for the wall - rather than the
    rasterized distance fields `spacing.Separations` reads for the
    relaxation's own use. This is the number that decides whether a cut
    bin holds together, so it uses the same exact geometry `CheckLayout`
    verifies a layout against, not an approximation that could read a
    divider as printable when the raster's discretization error alone
    would sink it.

    Returns `inf` for whichever does not apply - a single part has no
    dividers.
    """
    return _MeasureWalls(PocketPolygons(layout, parts, pocket_offset), layout.Envelope())


def _ToOpenScad(layout: Layout, points: np.ndarray) -> np.ndarray:
    """One placed pocket in OpenSCAD's coordinates.

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
    centered = points - interior / 2.0
    return np.stack([centered[:, 0], -centered[:, 1]], axis=-1)


def _Polygon(points: np.ndarray) -> str:
    return "polygon(points=[" + ", ".join(f"[{x:.4f}, {y:.4f}]" for x, y in points) + "])"


def _Cutout(points: np.ndarray, depth: float) -> str:
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

    **The polygon emitted is already the pocket.** There is no
    `offset(r = ...)` here any more, and its absence is the point of the
    whole exercise. OpenSCAD's `offset` tessellates its rounded joins
    from `$fn`/`$fs`/`$fa`, and at a 1mm offset with the defaults
    `Calc::get_fragments_from_r` returns five fragments for a full
    circle - so a 90-degree corner became a single straight chamfer
    sitting 0.293mm *inside* the nominal offset. Measured: a 10x10 square
    offset by 1 came out 142.000mm^2 against an exact 143.142. Every
    sharp convex corner was cut to about 0.71mm of clearance where the
    layout had budgeted 1.0mm, silently. `pocket.PocketContour` does the
    dilation instead, at an error that is bounded, one-sided and
    measured.
    """
    return (
        f"    translate([0, 0, {-depth:.4f}])\n"
        f"    linear_extrude(height = {depth:.4f})\n"
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

    `pocket_offset` defaults to `LayoutParameters().pocket_offset` - the
    same tunable that sizes a layout's clearances in the first place. The
    two are meant to track each other: in practice there is one tolerance
    setting, not two, and a caller who packed at a non-default
    `pocket_offset` should pass that same value here rather than rely on
    the default. Pass a different value deliberately to cut at a tolerance
    the layout did not budget for - `ThinnestWalls` is what keeps that
    honest, refusing anything that leaves too little to print.

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
        pocket_offset = _PackedOffset(layout, parts)
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

    polygons = PocketPolygons(layout, parts, pocket_offset)
    divider, wall = _MeasureWalls(polygons, layout.Envelope())
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
    cutouts = "\n".join(_Cutout(_ToOpenScad(layout, polygons[part_id]), depth) for part_id in sorted(layout.placements))

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
