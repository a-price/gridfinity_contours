"""Tests for solid generation (M7).

Most of what can go wrong here is invisible until a print finishes: a
pocket mirrored, a divider too thin to exist, a bin whose height was
never going to work. Those are what this pins.
"""

import os
import re
import shutil
import subprocess

import numpy as np
import pytest

from layout.container import DIVIDER_WIDTH_MM, InteriorSpan
from layout.loading import BuildParts, LoadParts
from layout.packer import Pack
from layout.parameters import LayoutParameters
from layout.placement import Layout, Placement
from layout.solid import (
    BASE_HEIGHT_MM,
    HEIGHT_UNIT_MM,
    GenerateScad,
    ThinnestWalls,
    WriteScad,
)
from conftest import Rectangle as _rectangle, SPOONS

# An L, so a mirror is detectable: its long arm runs one way only.
ELL = np.array([[0.0, 0.0], [30.0, 0.0], [30.0, 8.0], [8.0, 8.0], [8.0, 20.0], [0.0, 20.0]])


def _layout(contours: dict, positions: dict, grid=(2, 1), pocket_offset: float = 1.0):
    params = LayoutParameters(pocket_offset=pocket_offset)
    parts = BuildParts(contours, params)
    placements = {part_id: Placement(part_id, np.array(position)) for part_id, position in positions.items()}
    return Layout(grid=grid, placements=placements), parts


def _polygons(scad: str) -> list[np.ndarray]:
    """Every polygon in the generated program, in order."""
    found = []
    for body in re.findall(r"polygon\(points=\[(.*?)\]\)", scad):
        pairs = re.findall(r"\[(-?[\d.]+), (-?[\d.]+)\]", body)
        found.append(np.array([[float(x), float(y)] for x, y in pairs]))
    return found


# ------------------------------------------------------------------- walls


def test_a_divider_is_what_is_left_between_two_pockets():
    """Twenty-millimetre squares whose *pockets* start 26mm apart. Each
    pocket is 22mm across at a 1mm offset, so 26 - 22 leaves a 4mm
    divider, and the objects inside them sit 6mm apart.

    Both numbers are real and it matters which one this reports: the
    divider is what gets printed. This used to be computed as the objects'
    6mm minus twice the offset, which reached the same 4mm by prediction
    rather than by measurement - and the mistake it existed to prevent was
    deducting the offset once instead of twice.
    """
    layout, parts = _layout(
        {0: _rectangle(20, 20), 1: _rectangle(20, 20)},
        {0: [4.0, 8.0], 1: [30.0, 8.0]},
    )

    divider, _ = ThinnestWalls(layout, parts)

    assert divider == pytest.approx(4.0, abs=0.15)


def test_a_bin_wall_is_measured_to_the_pocket_not_the_object():
    """A placement positions the *pocket's* corner, so a pocket dropped at
    5mm leaves 5mm of wall and the object inside it sits a further
    millimetre in. This used to read the object's 6mm and deduct the
    offset to get there; now there is nothing to deduct.
    """
    layout, parts = _layout({0: _rectangle(20, 20)}, {0: [5.0, 8.0]})

    _, wall = ThinnestWalls(layout, parts)

    assert wall == pytest.approx(5.0, abs=0.01)


def test_the_divider_reads_exact_geometry_not_the_relaxations_raster():
    """`ThinnestWalls` decides whether a cut bin holds together, so it has
    to agree with `verify.CheckLayout` - the same exact polygon geometry,
    not the rasterized field the relaxation uses for its own, looser
    purposes. Tight enough a tolerance that a regression back to the raster
    would fail it: two 20mm squares 6mm apart have an exactly known gap.
    """
    layout, parts = _layout(
        {0: _rectangle(20, 20), 1: _rectangle(20, 20)},
        {0: [4.0, 8.0], 1: [30.0, 8.0]},  # exactly 6mm of gap
        pocket_offset=0.0,  # so the pockets are the squares, to the micron
    )

    divider, _ = ThinnestWalls(layout, parts)

    assert divider == pytest.approx(6.0, abs=1e-6)


def test_a_single_part_has_no_dividers():
    layout, parts = _layout({0: _rectangle(20, 20)}, {0: [5.0, 8.0]})

    divider, _ = ThinnestWalls(layout, parts, pocket_offset=1.0)

    assert divider == np.inf


# -------------------------------------------------------------- refusals


def test_an_offset_the_layout_never_budgeted_for_is_refused():
    """The point of applying tolerance here is that it can be changed
    without re-packing - but only downward. Asking for more room than the
    arrangement reserved has to fail loudly, not print a bin whose
    dividers are too thin to exist.
    """
    layout, parts = _layout(
        {0: _rectangle(20, 20), 1: _rectangle(20, 20)},
        {0: [4.0, 8.0], 1: [30.0, 8.0]},  # 6mm of gap
    )

    with pytest.raises(ValueError, match="divider"):
        GenerateScad(layout, parts, pocket_offset=2.5)  # would leave 1.0mm, under 1.2


def test_a_smaller_offset_than_the_layout_assumed_is_fine():
    layout, parts = _layout(
        {0: _rectangle(20, 20), 1: _rectangle(20, 20)},
        {0: [4.0, 8.0], 1: [30.0, 8.0]},
    )

    scad = GenerateScad(layout, parts, pocket_offset=0.5)

    # Nothing to grep for in the program any more - the pocket *is* the
    # polygon - so the check is that each one came out cut at 0.5 rather
    # than the 1.0 it was packed at.
    assert "offset(" not in scad
    for polygon in _polygons(scad):
        extent = polygon.max(axis=0) - polygon.min(axis=0)
        assert extent == pytest.approx([21.0, 21.0], abs=0.15)


def test_the_same_offset_cuts_the_same_pocket_whichever_way_it_arrives():
    """`--solid-offset 1.5` and packing at 1.5 have to produce the same
    polygon, or "re-cut at a different tolerance" quietly means "re-cut at
    a different *shape*".

    The trap is the raster: a re-cut that took `pocket`'s module defaults
    rather than the part's own settings traced at a different fidelity, so
    the two agreed only for a session that happened to leave
    `pocket_resolution` alone. Deliberately coarse here, since that is the
    case that used to disagree.

    Compared in each polygon's own frame, because the two do *not* land in
    the same place and should not. A placement anchors the pocket's
    minimum corner, so packing at 1.5 seats the object 1.5mm in from that
    corner, while re-cutting grows the pocket around an object that has
    already been placed and does not move. Growing around the object is
    the behaviour worth having - it is what lets a tolerance change re-cut
    a solid instead of invalidating an arrangement - and here it shows up
    as a 1mm translation between two identical shapes.
    """
    coarse = LayoutParameters(pocket_offset=0.5, pocket_resolution=0.2)
    packed = LayoutParameters(pocket_offset=1.5, pocket_resolution=0.2)
    shape = {0: _rectangle(20, 20)}
    placements = {0: Placement(0, np.array([6.0, 6.0]))}

    recut = GenerateScad(Layout(grid=(2, 1), placements=placements), BuildParts(shape, coarse), pocket_offset=1.5)
    direct = GenerateScad(Layout(grid=(2, 1), placements=placements), BuildParts(shape, packed))

    (from_recut,) = _polygons(recut)
    (from_packed,) = _polygons(direct)
    assert from_recut.shape == from_packed.shape
    np.testing.assert_allclose(
        from_recut - from_recut.min(axis=0),
        from_packed - from_packed.min(axis=0),
        atol=1e-3,
    )


def test_an_offset_that_eats_the_bin_wall_is_refused():
    """Packed at 1.0 with its pocket 1mm off the wall, so the object is
    2mm off it. Re-cutting at 2.5 grows the pocket through the wall.
    """
    layout, parts = _layout({0: _rectangle(20, 20)}, {0: [1.0, 8.0]})

    with pytest.raises(ValueError, match="wall"):
        GenerateScad(layout, parts, pocket_offset=2.5)


def test_a_pocket_deeper_than_the_infill_is_refused():
    """Not for safety - a cutout is a child of `bin_render`, which unions
    the base and wall in outside its own difference, so an over-deep one
    is simply clamped and can damage nothing. Refused because clamping
    silently would hide the real answer: the bin needs to be taller.
    """
    layout, parts = _layout({0: _rectangle(20, 20)}, {0: [5.0, 8.0]})

    with pytest.raises(ValueError, match="raise the height"):
        GenerateScad(layout, parts, pocket_offset=1.0, height_units=3, pocket_depth=20.0)


def test_a_bin_with_no_infill_is_refused():
    layout, parts = _layout({0: _rectangle(20, 20)}, {0: [5.0, 8.0]})

    with pytest.raises(ValueError, match="all base"):
        GenerateScad(layout, parts, pocket_offset=1.0, height_units=1)


def test_the_pocket_offset_defaults_to_the_layouts_own_tunable():
    """One tolerance setting, not two: the default tracks
    `LayoutParameters.pocket_offset` rather than a separately maintained
    number, so the two cannot quietly drift apart.
    """
    layout, parts = _layout({0: _rectangle(20, 20)}, {0: [5.0, 8.0]})

    assert GenerateScad(layout, parts) == GenerateScad(layout, parts, pocket_offset=LayoutParameters().pocket_offset)


def test_a_layout_whose_parts_are_missing_is_refused():
    layout, parts = _layout({0: _rectangle(20, 20)}, {0: [5.0, 8.0]})
    layout = Layout(grid=layout.grid, placements={**layout.placements, 9: Placement(9, np.zeros(2))})

    with pytest.raises(ValueError, match=r"\[9\]"):
        GenerateScad(layout, parts, pocket_offset=1.0)


# ------------------------------------------------------------- coordinates


def test_pockets_are_centered_on_the_bin():
    """A layout's origin is the interior's corner; OpenSCAD's bin is
    centered on the origin.
    """
    params = LayoutParameters()
    interior_x = InteriorSpan(2, params.inset)
    layout, parts = _layout({0: _rectangle(20, 20)}, {0: [5.0, 8.0]}, grid=(2, 1))

    (points,) = _polygons(GenerateScad(layout, parts, pocket_offset=1.0))

    # 5mm into the interior, whose own left edge is half a span left of center.
    assert points[:, 0].min() == pytest.approx(5.0 - interior_x / 2.0, abs=0.01)


def test_an_asymmetric_pocket_is_not_mirrored():
    """The failure this catches passes every dimension check and still
    will not hold the tool.

    OpenSCAD is y-up; the layout frame is the printed page's, y-down. The
    emitted outline must therefore be the placed contour with y negated -
    anything else is a reflection.
    """
    params = LayoutParameters()
    layout, parts = _layout({0: ELL}, {0: [10.0, 6.0]}, grid=(2, 1))
    interior = np.array([InteriorSpan(2, params.inset), InteriorSpan(1, params.inset)])

    (points,) = _polygons(GenerateScad(layout, parts, pocket_offset=1.0))

    placed = layout.placements[0].ToWorld(parts[0]) - interior / 2.0
    expected = np.stack([placed[:, 0], -placed[:, 1]], axis=-1)
    np.testing.assert_allclose(points, expected, atol=0.01)


def test_the_y_flip_is_a_reflection_not_a_rotation():
    """A 180-degree rotation would also map top-left to bottom-right, and
    would be wrong. Negating y alone reverses the outline's winding.
    """
    layout, parts = _layout({0: ELL}, {0: [10.0, 6.0]}, grid=(2, 1))

    (points,) = _polygons(GenerateScad(layout, parts, pocket_offset=1.0))

    def signed_area(polygon: np.ndarray) -> float:
        x, y = polygon[:, 0], polygon[:, 1]
        return 0.5 * float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))

    assert np.sign(signed_area(points)) == -np.sign(signed_area(parts[0].pocket_contour))


# ------------------------------------------------------------- the program


def test_the_cutouts_are_children_of_bin_render():
    """Not a top-level difference against it.

    Both produce a sound bin - the older form cuts the base and unions it
    back on - but only the child form gives the pocket a stated depth,
    since a bare `linear_extrude()` cuts 100mm upward from z=0 and leaves
    the floor wherever the base happens to start. The depth limit below
    is meaningless without this.
    """
    layout, parts = _layout({0: _rectangle(20, 20)}, {0: [5.0, 8.0]})

    scad = GenerateScad(layout, parts, pocket_offset=1.0)

    assert "bin_render(bin) {" in scad
    assert "difference()" not in scad


def test_pockets_extend_downward_from_the_infill_surface():
    layout, parts = _layout({0: _rectangle(20, 20)}, {0: [5.0, 8.0]})

    scad = GenerateScad(layout, parts, pocket_offset=1.0, height_units=3)

    depth = 3 * HEIGHT_UNIT_MM - BASE_HEIGHT_MM
    assert f"translate([0, 0, {-depth:.4f}])" in scad
    assert f"linear_extrude(height = {depth:.4f})" in scad


def test_one_pocket_per_placed_part():
    layout, parts = _layout(
        {0: _rectangle(20, 20), 1: _rectangle(18, 18)},
        {0: [4.0, 8.0], 1: [32.0, 8.0]},
    )

    assert len(_polygons(GenerateScad(layout, parts, pocket_offset=1.0))) == 2


def test_the_bin_takes_its_size_from_the_layout():
    layout, parts = _layout({0: _rectangle(20, 20)}, {0: [5.0, 8.0]}, grid=(3, 2))

    scad = GenerateScad(layout, parts, pocket_offset=1.0)

    assert "grid_size = [3, 2]" in scad


def test_the_library_include_is_absolute():
    """A generated .scad is routinely written somewhere other than the
    repository root, and a relative include would resolve against
    wherever it landed.
    """
    layout, parts = _layout({0: _rectangle(20, 20)}, {0: [5.0, 8.0]})

    scad = GenerateScad(layout, parts, pocket_offset=1.0)

    include = re.search(r"include <(.*?)/standard.scad>", scad)
    assert include is not None and os.path.isabs(include.group(1))


def test_writing_puts_the_program_on_disk(tmp_path):
    layout, parts = _layout({0: _rectangle(20, 20)}, {0: [5.0, 8.0]})
    path = tmp_path / "bin.scad"

    WriteScad(str(path), layout, parts, pocket_offset=1.0)

    assert "bin_render(bin)" in path.read_text()


# ------------------------------------------------------------ does it build


@pytest.mark.slow
@pytest.mark.skipif(shutil.which("openscad") is None, reason="openscad is not installed")
def test_the_spoons_render_to_a_solid(tmp_path):
    """M7's done-when, minus the physical print: the generated program has
    to actually build a closed solid, not merely look plausible.
    """
    params = LayoutParameters()
    parts = LoadParts(SPOONS, params)
    layout = Pack(parts, params).layout
    assert layout is not None

    scad, stl = tmp_path / "spoons.scad", tmp_path / "spoons.stl"
    WriteScad(str(scad), layout, parts, pocket_offset=params.pocket_offset)

    finished = subprocess.run(
        ["openscad", "-o", str(stl), str(scad)],
        capture_output=True,
        text=True,
        timeout=600,
    )

    assert finished.returncode == 0, finished.stderr
    assert stl.stat().st_size > 0
    assert "Simple:        yes" in finished.stderr, "the solid should be manifold"


@pytest.mark.slow
def test_the_spoons_leave_printable_walls():
    params = LayoutParameters()
    parts = LoadParts(SPOONS, params)
    layout = Pack(parts, params).layout
    assert layout is not None

    divider, wall = ThinnestWalls(layout, parts, params.pocket_offset)

    assert divider >= DIVIDER_WIDTH_MM
    assert wall > 0.0
