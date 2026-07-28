"""End to end: a drawer's worth of real objects, contours to drawer floorplan.

Every other test module exercises one source module, by the house rule in
docs/layout.md. This one deliberately does not. It runs the whole stack on
the real `test_data/` captures - load, rasterize, group into bins, assign
the bins to drawers, write the files you would print and slice - and
checks that what comes out the far end is *sound*.

**Sound, not good.** Whether these objects packing into so many cells is a
good answer depends on clearances that are still derived rather than
measured against a print, and on a stochastic solver that is allowed to
get unlucky. Asserting a cell count here would pin a number nobody has
justified and would fail on an unrelated retune. What can be asserted
exactly is that every object ends up in exactly one bin, that no two
objects overlap, and that no two bins overlap in a drawer - statements a
regression anywhere in the stack would break, and which hold or do not
regardless of tuning.

Two deliberate departures from the defaults, both measured rather than
guessed:

* `max_grid` is 7, not the default 6. Four of these objects genuinely need
  a seven-cell bin, and the knife misses six by 0.7mm.
* The solver budget is cut, and the local search is skipped on the full
  set - see the budget constants and the scaling note on the full test.
"""

import glob
import os
from dataclasses import replace
from pathlib import Path

import pytest

from pipeline.layout.container import GRID_PITCH_MM
from pipeline.layout.drawer import Assign, DrawerCells, FreeCells, LargestFreeRegion
from pipeline.layout.floorplan import WriteFloorplanPdf
from pipeline.layout.grouping import FirstFit, Group
from pipeline.layout.loading import LoadParts
from pipeline.layout.parameters import LayoutParameters
from pipeline.layout.preview import WriteLayoutPdf, WriteLayoutSvg
from pipeline.layout.solid import WriteScad
from pipeline.layout.verify import CheckLayout

# Every distinct object in test_data/, by glob rather than by list, so a
# new capture is exercised the moment it is dropped in.
#
# `contours.svg` is filtered by name because it is `SvgExportStage`'s
# default filename: a file called that is an unnamed dump, not a
# catalogued object. The guard is inert today - one did sit here, holding
# a second capture of the same spoon as big_spoon.svg, and has been
# removed - but the export default has not changed.
STALE_EXPORT = "contours.svg"
HOUSEHOLD = sorted(path for path in glob.glob("test_data/*.svg") if os.path.basename(path) != STALE_EXPORT)

# A subset small enough for the local search to finish - see the scaling
# note on `test_the_whole_household_reaches_a_drawer`.
HANDFUL = [
    "test_data/small_spoon.svg",
    "test_data/small_fork.svg",
    "test_data/spreader.svg",
    "test_data/camera_remote_rcv.svg",
]

# Four of these objects need a 7-cell bin: huge_server (260mm),
# serving_spoon (274mm), server (251mm) and knife (243mm). The knife is the
# instructive one - a 6-cell interior is 246.3mm and it needs 247.0mm with
# its wall clearance, so it misses by 0.7mm.
MAX_GRID = 7

# Cut well below the tuned defaults (24 restarts, 400 iterations). This
# test is about whether the stack executes and returns something sound, and
# a stricter budget buys tighter bins at a cost the suite cannot afford.
# Anything asserted here has to hold at any budget.
RESTARTS, ITERATIONS, PATIENCE = 3, 120, 12

# Two kitchen drawers, 500 x 400mm of usable interior each: 11 x 9 cells,
# 99 apiece. Chosen so the bins genuinely have to be split across both -
# one object per bin already costs 144 cells.
DRAWER_MM = (500.0, 400.0)

# Set this to a path to keep the drawer floorplan the full run produces:
#     DRAWER_FLOORPLAN=/tmp/floorplan.pdf make check
# Unset, it goes to the test's scratch directory and is discarded.
FLOORPLAN_ENV = "DRAWER_FLOORPLAN"


def _params(**overrides) -> LayoutParameters:
    base = LayoutParameters(max_grid=MAX_GRID, restarts=RESTARTS, iterations=ITERATIONS, patience=PATIENCE)
    return replace(base, **overrides)


def _assert_bins_are_sound(grouping, parts, params) -> None:
    """Every part in exactly one bin, and every bin geometrically valid.

    Checked at the real clearances rather than for bare overlap. Zero
    overlap is the weaker claim and would pass on a layout whose dividers
    come out too thin to print; what the solver actually promises is that
    *exact* polygon geometry clears `c_pair` and `c_wall`, which is the
    thing a raster error would quietly break.
    """
    placed = [part_id for contents in grouping.Contents() for part_id in contents]
    assert sorted(placed) == sorted(parts), "every object exactly once, none dropped or invented"

    for index, layout in enumerate(grouping.bins):
        problems = CheckLayout(layout, parts, pair_clearance=params.c_pair, wall_clearance=params.c_wall)
        assert problems == [], f"bin {index} violates its own clearances: {problems}"


def _assert_drawers_are_sound(result, footprints, drawers) -> None:
    """Every bin inside a drawer, and no two bins sharing a cell."""
    assert result.placed, f"assignment did not place everything: {result.detail}"

    covered: list[set] = [set() for _ in drawers]
    for bin_id, slot in result.slots.items():
        width, height = slot.Footprint(footprints[bin_id])
        x, y = slot.cell
        drawer = drawers[slot.drawer]
        assert x >= 0 and x + width <= drawer.width, f"bin {bin_id} runs off its drawer horizontally"
        assert y >= 0 and y + height <= drawer.height, f"bin {bin_id} runs off its drawer vertically"

        for dx in range(width):
            for dy in range(height):
                cell = (x + dx, y + dy)
                assert cell not in covered[slot.drawer], f"bin {bin_id} overlaps another at {cell}"
                covered[slot.drawer].add(cell)


# ------------------------------------------------------------- the fixtures


def test_the_fixture_set_is_what_the_tests_think_it_is():
    """Guards the directory itself. Dropping a capture in or out silently
    changes what every measurement below means.
    """
    assert len(HOUSEHOLD) == 18
    assert not any(path.endswith(STALE_EXPORT) for path in HOUSEHOLD)


def test_every_capture_loads_at_a_plausible_size():
    """A contour that loaded 3.78x wrong would still pack, still render,
    and still be wrong - the scale bug this guards is silent everywhere
    downstream.
    """
    parts = LoadParts(HOUSEHOLD, _params())

    assert len(parts) == len(HOUSEHOLD)
    for part_id, part in sorted(parts.items()):
        longest = max(part.size)
        assert 50.0 < longest < 300.0, f"part {part_id} is {longest:.0f}mm long, which is not a household object"
        assert part.area > 100.0


def test_the_defaults_cannot_hold_this_set():
    """Recorded as a fact about the fixtures rather than left as a
    surprise: the default six-cell cap is too small for real objects, and
    `max_grid` has to be raised for any of this to run.
    """
    parts = LoadParts(["test_data/huge_server.svg"], _params(max_grid=6))

    with pytest.raises(ValueError, match="does not fit"):
        Group(parts, _params(max_grid=6))


# --------------------------------------------------------------- end to end


@pytest.mark.slow
def test_a_handful_goes_from_contours_to_a_drawer_floorplan(tmp_path):
    """The full entry point - `Group`, local search included - on a set
    small enough for it, and then all the way out to files.

    The artifacts matter as much as the assignment: a layout that packs
    but cannot be written is not a result anyone can use, and the writers
    are the only part of the stack a soundness check cannot reach.
    """
    params = _params()
    parts = LoadParts(HANDFUL, params)

    grouping = Group(parts, params)
    _assert_bins_are_sound(grouping, parts, params)

    footprints = {index: layout.grid for index, layout in enumerate(grouping.bins)}
    drawers = [DrawerCells(*DRAWER_MM)]
    result = Assign(footprints, drawers)
    _assert_drawers_are_sound(result, footprints, drawers)

    for index, layout in enumerate(grouping.bins):
        WriteLayoutSvg(str(tmp_path / f"bin{index}.svg"), layout, parts)
        WriteLayoutPdf(str(tmp_path / f"bin{index}.pdf"), layout, parts)
        WriteScad(str(tmp_path / f"bin{index}.scad"), layout, parts)

    for index in range(len(grouping.bins)):
        for suffix in ("svg", "pdf", "scad"):
            written = tmp_path / f"bin{index}.{suffix}"
            assert written.stat().st_size > 0, f"{written.name} was written empty"


@pytest.mark.slow
def test_the_whole_household_reaches_a_drawer(tmp_path):
    """All eighteen objects, at the scale a person actually has.

    **The local search is skipped here, and that is a finding rather than
    a shortcut.** `Improve` prices every move and swap between bins, which
    is quadratic in bins and needs a pack per surviving candidate. On four
    objects it finishes in seconds; on eighteen it did not finish in ten
    minutes even before the local search began. `FirstFit` alone is what
    scales today, so it is what this test runs, and `Improve` is covered
    on small fixtures in `grouping_test.py`.
    """
    params = _params()
    parts = LoadParts(HOUSEHOLD, params)

    grouping = FirstFit(parts, params)
    _assert_bins_are_sound(grouping, parts, params)

    layouts = dict(enumerate(grouping.bins))
    footprints = {index: layout.grid for index, layout in layouts.items()}
    drawers = [DrawerCells(*DRAWER_MM), DrawerCells(*DRAWER_MM)]
    result = Assign(footprints, drawers)
    _assert_drawers_are_sound(result, footprints, drawers)

    # The drawer floorplan: one true-scale page per drawer, showing the
    # bins and the objects inside them. Written to wherever
    # `DRAWER_FLOORPLAN` points if it is set, so the floorplan can actually
    # be looked at, and to a scratch file otherwise - a test that leaves
    # files in the tree is a test nobody runs twice.
    floorplan_path = os.environ.get(FLOORPLAN_ENV) or str(tmp_path / "drawer_floorplan.pdf")
    WriteFloorplanPdf(floorplan_path, drawers, layouts, result, parts)

    written = Path(floorplan_path)
    assert written.read_bytes().startswith(b"%PDF")
    assert written.read_bytes().count(b"/MediaBox") == len(drawers), "one page per drawer"

    # Not a quality assertion - just that grouping did something, since a
    # first-fit that merged nothing would pass every soundness check above
    # while having done no work at all.
    assert len(grouping.bins) < len(parts), "first-fit should have merged at least one pair"

    free = FreeCells(drawers, footprints, result)
    usable = LargestFreeRegion(drawers, footprints, result)
    assert all(0 <= region <= total for region, total in zip(usable, free))

    # The cross-level invariant, asserted here rather than in a test of its
    # own so that the fifty seconds of first-fit above are spent once: a
    # footprint the grid size search chose has to be one a drawer can
    # physically take. This is the only place the two levels meet on real
    # data, and what `admissible_grids` exists to guarantee up front.
    for bin_id, slot in result.slots.items():
        drawer = drawers[slot.drawer]
        assert drawer.Holds(footprints[bin_id]), f"bin {bin_id} was placed in a drawer that cannot hold it"
        # And a cell converts back to millimeters on the same 42mm lattice
        # everything else in the project measures in.
        width, height = slot.Footprint(footprints[bin_id])
        assert (slot.cell[0] + width) * GRID_PITCH_MM <= drawer.width * GRID_PITCH_MM
        assert (slot.cell[1] + height) * GRID_PITCH_MM <= drawer.height * GRID_PITCH_MM
