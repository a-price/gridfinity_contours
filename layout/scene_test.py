"""Tests for scene (M3).

`_WorldPlacement`'s two branches are the fragile part - see the module
docstring - so they get two kinds of check: a hand-computed value, and an
independent cross-check against `layout.placement.RotatePoints`, the same
function `layout.floorplan.PlacedBinShapes` uses to turn a bin's 2D
drawing. A corner alone cannot tell a +90 rotation from a -90 one apart (a
rectangle looks the same either way), so the cross-check uses an
asymmetric point.
"""

import numpy as np
import pytest

from conftest import Rectangle as _rectangle
from export.scad_writer import ScadPart
from layout.container import GRID_PITCH_MM
from layout.drawer import AssignmentResult, Drawer, Slot
from layout.loading import BuildParts
from layout.parameters import LayoutParameters
from layout.placement import Layout, Placement, RotatePoints
from layout.plan import StoragePlan
from layout.preview import OuterFootprint
from layout.scene import DrawerParts, SceneParts, WriteScene, _DrawerOffsets, _WorldPlacement

_DRAWER = Drawer(8, 8)


def _layout(grid: tuple[int, int] = (3, 5)) -> tuple[Layout, dict]:
    params = LayoutParameters(pocket_offset=1.0)
    parts = BuildParts({0: _rectangle(20, 20)}, params)
    placements = {0: Placement(0, np.array([5.0, 8.0]))}
    return Layout(grid=grid, placements=placements, inset=params.inset), parts


# --------------------------------------------------------- _WorldPlacement


def test_an_unturned_bin_is_centered_and_flipped_to_the_drawer():
    """Hand-computed: a 3x5 bin at cell (2, 1) in an 8x8 drawer, unturned."""
    w, h = OuterFootprint(3, 5)
    drawer_w, drawer_h = OuterFootprint(_DRAWER.width, _DRAWER.height)
    slot = Slot(bin_id=0, drawer=0, cell=(2, 1), turned=False)

    x, y, degrees = _WorldPlacement(_DRAWER, (3, 5), slot)

    assert degrees == 0.0
    assert x == pytest.approx(GRID_PITCH_MM * 2 + w / 2.0 - drawer_w / 2.0)
    assert y == pytest.approx(drawer_h / 2.0 - GRID_PITCH_MM * 1 - h / 2.0)


def test_a_turned_bin_rotates_negative_ninety():
    """Hand-computed, same fixture, turned. The footprint passed in stays
    the bin's own *unturned* (n, m) - the swap is what the rotation does,
    not a pre-swapped size.
    """
    w, h = OuterFootprint(3, 5)
    drawer_w, drawer_h = OuterFootprint(_DRAWER.width, _DRAWER.height)
    slot = Slot(bin_id=0, drawer=0, cell=(2, 1), turned=True)

    x, y, degrees = _WorldPlacement(_DRAWER, (3, 5), slot)

    assert degrees == -90.0
    assert x == pytest.approx(GRID_PITCH_MM * 2 + h / 2.0 - drawer_w / 2.0)
    assert y == pytest.approx(drawer_h / 2.0 - GRID_PITCH_MM * 1 - w / 2.0)


@pytest.mark.parametrize("turned", [False, True])
def test_placement_matches_rotatepoints_not_just_the_algebra(turned):
    """An asymmetric point, run through two independent pipelines:

    1. The real 2D turning logic - `RotatePoints` at `orientation=1`, the
       same call `layout.floorplan.PlacedBinShapes` makes for a turned
       slot - plus the slot's offset, then the page-to-world flip this
       module's docstring describes.
    2. `_WorldPlacement`'s own rotate-then-translate, applied to the same
       point after converting it to the bin's own OpenSCAD-local frame
       (center-shift plus a y negation - the same relationship
       `layout.solid._ToOpenScad` uses for a pocket's coordinates).

    They have to land on the same point, including for the unturned case
    (`orientation=0`, `degrees=0`), or the two coordinate systems have
    quietly drifted apart.
    """
    n, m = 3, 5
    w, h = OuterFootprint(n, m)
    drawer_w, drawer_h = OuterFootprint(_DRAWER.width, _DRAWER.height)
    slot = Slot(bin_id=0, drawer=0, cell=(2, 1), turned=turned)

    # Off both center lines, so neither a mirror nor a wrong rotation
    # direction could land here by symmetry.
    local_page = np.array([w * 0.2, h * 0.7])

    orientation = 1 if turned else 0
    turned_page = RotatePoints(local_page[None, :], orientation, np.array([w, h]))[0]
    placed_page = turned_page + np.array(slot.cell) * GRID_PITCH_MM
    expected_world = np.array([placed_page[0] - drawer_w / 2.0, drawer_h / 2.0 - placed_page[1]])

    local_openscad = np.array([local_page[0] - w / 2.0, h / 2.0 - local_page[1]])
    x, y, degrees = _WorldPlacement(_DRAWER, (n, m), slot)
    theta = np.radians(degrees)
    rotation = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    actual_world = rotation @ local_openscad + np.array([x, y])

    assert actual_world == pytest.approx(expected_world, abs=1e-9)


# ------------------------------------------------------------- DrawerParts


def test_drawer_parts_only_holds_its_own_drawers_bins():
    layout, parts = _layout()
    assignment = AssignmentResult(
        "placed",
        {
            0: Slot(bin_id=0, drawer=0, cell=(0, 0), turned=False),
            1: Slot(bin_id=1, drawer=1, cell=(0, 0), turned=False),
        },
    )
    layouts = {0: layout, 1: layout}

    here, problems = DrawerParts(0, _DRAWER, layouts, assignment, parts)

    assert [part.name for part in here] == ["bin_0"]
    assert isinstance(here[0], ScadPart)
    assert problems == {}


def test_drawer_parts_is_empty_for_a_drawer_with_nothing_assigned():
    layout, parts = _layout()
    assignment = AssignmentResult("placed", {0: Slot(bin_id=0, drawer=0, cell=(0, 0), turned=False)})

    assert DrawerParts(1, _DRAWER, {0: layout}, assignment, parts) == ([], {})


def test_an_uncuttable_bin_is_skipped_and_reported_not_raised():
    """A pocket offset that leaves too little bin wall must not sink the
    rest of the drawer. Bin 0 is the fixture `layout.solid_test.
    test_an_offset_that_eats_the_bin_wall_is_refused` uses: packed at 1.0
    with its pocket 1mm off the wall, re-cut at 2.5 grows the pocket
    through it. Bin 1 has room to spare at the same offset and must still
    render - two parts sharing one `BuildParts` call, as a real library
    would, rather than two dicts merged together under colliding ids.
    """
    params = LayoutParameters(pocket_offset=1.0)
    parts = BuildParts({0: _rectangle(20, 20), 1: _rectangle(20, 20)}, params)
    thin = Layout(grid=(2, 1), placements={0: Placement(0, np.array([1.0, 8.0]))}, inset=params.inset)
    good = Layout(grid=(3, 5), placements={1: Placement(1, np.array([5.0, 8.0]))}, inset=params.inset)
    layouts = {0: thin, 1: good}
    assignment = AssignmentResult(
        "placed",
        {
            0: Slot(bin_id=0, drawer=0, cell=(0, 0), turned=False),
            1: Slot(bin_id=1, drawer=0, cell=(3, 0), turned=False),
        },
    )

    here, problems = DrawerParts(0, _DRAWER, layouts, assignment, parts, pocket_offset=2.5)

    assert [part.name for part in here] == ["bin_1"]
    assert list(problems) == [0]
    assert "wall" in problems[0]


# ------------------------------------------------------- combining drawers


def test_drawer_offsets_lay_out_left_to_right_with_a_gap():
    drawers = [Drawer(2, 2), Drawer(3, 3)]
    width0, _ = OuterFootprint(2, 2)
    width1, _ = OuterFootprint(3, 3)

    offsets = _DrawerOffsets(drawers, gap=10.0)

    assert offsets[0] == pytest.approx(width0 / 2.0)
    assert offsets[1] == pytest.approx(width0 + 10.0 + width1 / 2.0)


def _two_drawer_plan() -> tuple[StoragePlan, dict, AssignmentResult]:
    """One bin per drawer, so combining actually has two drawers' worth of
    bins to shift apart.
    """
    layout, parts = _layout()
    drawers = (_DRAWER, Drawer(4, 4))
    layouts = {0: layout, 1: layout}
    assignment = AssignmentResult(
        "placed",
        {
            0: Slot(bin_id=0, drawer=0, cell=(0, 0), turned=False),
            1: Slot(bin_id=1, drawer=1, cell=(0, 0), turned=False),
        },
    )
    plan = StoragePlan(drawers=drawers, parts=parts, layouts=layouts, assignment=assignment)
    return plan, parts, assignment


def test_scene_parts_shifts_each_drawers_bins_by_its_own_offset():
    """Composition, checked against the pieces it composes rather than
    re-deriving the placement math: each drawer's bins, computed alone by
    `DrawerParts`, should reappear in `SceneParts`' output shifted by
    exactly that drawer's `_DrawerOffsets` entry on X and untouched on Y.
    """
    plan, parts, assignment = _two_drawer_plan()

    combined, problems = SceneParts(plan, assignment, gap=10.0)

    offsets = _DrawerOffsets(plan.drawers, gap=10.0)
    alone0 = DrawerParts(0, plan.drawers[0], plan.layouts, assignment, parts)[0][0]
    alone1 = DrawerParts(1, plan.drawers[1], plan.layouts, assignment, parts)[0][0]
    by_name = {part.name: part for part in combined}

    assert by_name["bin_0"].x == pytest.approx(alone0.x + offsets[0])
    assert by_name["bin_0"].y == alone0.y
    assert by_name["bin_1"].x == pytest.approx(alone1.x + offsets[1])
    assert by_name["bin_1"].y == alone1.y
    assert problems == {}


# ----------------------------------------------------------------- WriteScene


def _plan(assignment: AssignmentResult | None) -> tuple[StoragePlan, dict]:
    layout, parts = _layout()
    return (
        StoragePlan(
            drawers=(_DRAWER, Drawer(4, 4)),
            parts=parts,
            layouts={0: layout},
            assignment=assignment,
        ),
        parts,
    )


def test_write_scene_combines_every_drawer_into_one_file(tmp_path):
    plan, parts, assignment = _two_drawer_plan()
    path = str(tmp_path / "scene.scad")

    report = WriteScene(path, plan, assignment=assignment)

    assert report.written == [path]
    assert report.problems == {}
    scad = (tmp_path / "scene.scad").read_text()
    assert "module bin_0() {" in scad
    assert "module bin_1() {" in scad
    assert scad.count("include <") == 1


def test_write_scene_needs_an_assignment():
    plan, _ = _plan(None)

    with pytest.raises(ValueError, match="assignment"):
        WriteScene("scene.scad", plan)


def test_write_scene_takes_an_explicit_assignment_over_a_missing_one(tmp_path):
    """The case `panels.floorplan_panel.FloorplanPanel._Assigned` exists
    for: a plan saved with no assignment because its search stopped before
    the drawer level ran.
    """
    plan, _ = _plan(None)
    assignment = AssignmentResult("placed", {0: Slot(bin_id=0, drawer=1, cell=(0, 0), turned=False)})
    path = str(tmp_path / "scene.scad")

    report = WriteScene(path, plan, assignment=assignment)

    assert report.written == [path]
    assert "module bin_0() {" in (tmp_path / "scene.scad").read_text()
