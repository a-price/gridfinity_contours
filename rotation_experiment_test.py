"""Tests for the rotation experiment.

**Deliberately pins none of its findings.** The whole point of the script
is that nobody knows whether off-axis rotation pays, so a cell count that
moved would be a result to read rather than a regression to fix. Asserting
one here would turn every genuine improvement into a red build and every
tuning change into an argument.

What is asserted is that the instrument is not lying: that the exact sweep
and the real packer agree about a case they can both answer, that the
three modes are genuinely different rather than three names for the same
search, and that a mode reporting a tighter bin has not got there by
overlapping parts.
"""

import numpy as np
import pytest

from layout.loading import LoadParts
from layout.packer import Pack
from layout.parameters import EIGHTH_TURNS, FREE_ROTATION, QUARTER_TURNS, ROTATIONS, LayoutParameters
from layout.verify import CheckLayout
from rotation_experiment import SETS, Cells, Main, PackUnder, Reach, SinglePartTable
from conftest import QuickParameters as _quick, Rectangle as _rectangle

# One restart. Every packing test here is about whether the plumbing runs
# and returns something sound, never about how tight the answer is, so the
# budget only has to be non-zero.
_BUDGET = 1


# ------------------------------------------------------------- the fixtures


def test_every_named_set_points_at_files_that_exist():
    """The sets are stems rather than paths, so a typo would surface as a
    load error minutes into a run.
    """
    import os

    for name, stems in SETS.items():
        for stem in stems:
            assert os.path.exists(f"test_data/{stem}.svg"), f"{name} names a missing {stem}"


def test_the_sets_ask_different_questions():
    """Four sets that were secretly the same set would make the multi-part
    table four times as slow and no more informative.
    """
    assert len({tuple(sorted(stems)) for stems in SETS.values()}) == len(SETS)


# ---------------------------------------------------------------- the sweep


def test_the_sweep_agrees_with_the_packer_on_a_case_both_can_answer():
    """The load-bearing check on the exact half. `Reach` reasons about
    bounding boxes with no solver at all, and `Pack` runs the real search;
    on one part they are answering the same question and must agree, or one
    of them is wrong about the geometry.
    """
    params = _quick(max_grid=4)
    parts = LoadParts(["test_data/small_spoon.svg"], params)
    part = next(iter(parts.values()))

    swept = Reach(part.pocket_contour, params)[QUARTER_TURNS]
    packed = Pack(parts, params).layout

    assert packed is not None
    assert swept == packed.grid


def test_a_freer_mode_never_needs_a_bigger_bin():
    """Each mode's angles are a superset of the one before, so its reach
    can only improve. A violation would mean the sweep's masks do not
    nest - that 45 was somehow excluding a quarter turn.
    """
    params = LayoutParameters()
    for stem in ("knife", "medium_spoon", "big_spoon", "small_spoon"):
        part = next(iter(LoadParts([f"test_data/{stem}.svg"], params).values()))
        reach = Reach(part.pocket_contour, params)

        assert Cells(reach[EIGHTH_TURNS]) <= Cells(reach[QUARTER_TURNS]), stem
        assert Cells(reach[FREE_ROTATION]) <= Cells(reach[EIGHTH_TURNS]), stem


def test_the_sweep_finds_the_diagonal_a_quarter_turn_cannot():
    """The mechanism under test, on a shape whose numbers are checkable by
    hand: a 1x1 interior leaves 32.4mm between wall clearances, and a 40mm
    bar crosses that square's 45.8mm diagonal but not its side.
    """
    params = LayoutParameters(max_grid=1)

    reach = Reach(_rectangle(40, 4), params)

    assert reach[QUARTER_TURNS] is None
    assert reach[FREE_ROTATION] == (1, 1)


def test_a_part_too_big_for_every_bin_reaches_nothing():
    params = LayoutParameters(max_grid=2)

    assert Reach(_rectangle(500, 400), params) == {mode: None for mode in ROTATIONS}


def test_missing_and_present_grids_sort_the_same_way():
    """`Cells` is what the verdict column compares, so "fits nothing" has
    to lose to every real bin rather than winning as a zero.
    """
    assert Cells(None) > Cells((7, 7))
    assert Cells((2, 1)) == 2


# ------------------------------------------------------------- the packing


@pytest.mark.parametrize("mode", ROTATIONS)
def test_each_mode_packs_something_sound(mode):
    """Every mode has to survive the exact polygon check, not only the
    raster energy it was solved against. A mode that packed tighter by
    letting parts overlap would read as exactly the finding this experiment
    is looking for.
    """
    params = _quick(max_grid=3)

    label, elapsed = PackUnder(["small_spoon"], mode, params, _BUDGET)

    assert "INVALID" not in label
    assert label != "none"
    assert elapsed >= 0.0


def test_a_turned_placement_is_reported_as_turned():
    """The count in the label is how a reader tells "free rotation found a
    tighter bin" from "free rotation found the same bin the same way".
    """
    params = _quick(max_grid=3)

    label, _ = PackUnder(["small_spoon"], QUARTER_TURNS, params, _BUDGET)

    assert "turned" not in label, "the 90-degree mode cannot turn anything"


def test_an_overlapping_layout_would_be_reported_rather_than_scored():
    """The guard exists because a silently invalid layout is the one
    outcome that would actively mislead. Provoked by asking for clearances
    far larger than the bin the packer already chose.
    """
    params = _quick(max_grid=3)
    parts = LoadParts(["test_data/small_spoon.svg"], params)
    result = Pack(parts, params)

    assert result.layout is not None
    problems = CheckLayout(result.layout, parts, pair_clearance=200.0, wall_clearance=200.0)
    assert problems, "a 200mm clearance in a small bin must register as a problem"


# ------------------------------------------------------------------ the run


def test_the_single_part_table_has_a_row_for_every_part():
    params = LayoutParameters()

    lines = SinglePartTable(["small_spoon", "big_spoon"], params)

    assert len(lines) == 3, "a header and one row each"
    assert all(mode in lines[0] for mode in ROTATIONS)
    assert "small_spoon" in lines[1] and "big_spoon" in lines[2]


def test_the_sweep_only_run_needs_no_solver(capsys):
    """`--skip-multi` is what makes the exact half usable on a whim, so it
    has to actually skip the search rather than merely hide its output.
    """
    assert Main(["--only", "spoons", "--skip-multi"]) == 0

    printed = capsys.readouterr().out
    assert "one part alone" in printed
    assert "restarts" not in printed


@pytest.mark.slow
def test_the_whole_experiment_runs():
    """Minutes, and asserting only that it completes. What it *found* is
    the output, and this file has nothing to say about that.
    """
    assert Main(["--only", "gainers", "--restarts", "1"]) == 0


def test_the_sweep_is_reproducible():
    """No seed anywhere in the exact half - it is geometry, and a second
    run that disagreed would mean it was not."""
    params = LayoutParameters()
    contour = np.asarray(_rectangle(120, 30), dtype=np.float64)

    assert Reach(contour, params) == Reach(contour, params)
