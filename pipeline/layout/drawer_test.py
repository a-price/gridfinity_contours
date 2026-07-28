"""Tests for drawer assignment (M9).

This is the only level of the project whose answers are exact, so the
tests are correspondingly sharper than the ones below it. Three things are
pinned here that nothing else can check.

That placements are **real**: bins inside their drawer, no two
overlapping. That INFEASIBLE means *proven* - the search finished and no
arrangement exists - which is the whole reason this level can instruct the
one below it, and which is why a budget that runs out reports EXHAUSTED
instead. And that the bounds in front of the search are one-sided, since
an over-eager bound here silently reports "buy another drawer".
"""

import pytest

from pipeline.layout.drawer import (
    DEFAULT_NODE_BUDGET,
    EXHAUSTED,
    INFEASIBLE,
    PLACED,
    AdmissibleFootprints,
    Assign,
    Drawer,
    DrawerCells,
    FreeCells,
    LargestFreeRegion,
    Slot,
    _Orientations,
)
from pipeline.layout.packer import CandidateGrids


def _occupied(footprints: dict[int, tuple[int, int]], result, drawers: list[Drawer]) -> list[set]:
    """The cells each drawer's bins actually cover, one set per drawer."""
    covered: list[set] = [set() for _ in drawers]
    for bin_id, slot in result.slots.items():
        width, height = slot.Footprint(footprints[bin_id])
        x, y = slot.cell
        for dx in range(width):
            for dy in range(height):
                covered[slot.drawer].add((x + dx, y + dy))
    return covered


def _assert_sound(footprints: dict[int, tuple[int, int]], result, drawers: list[Drawer]) -> None:
    """Every bin inside its drawer, and no two bins sharing a cell."""
    total = 0
    for bin_id, slot in result.slots.items():
        width, height = slot.Footprint(footprints[bin_id])
        x, y = slot.cell
        drawer = drawers[slot.drawer]
        assert 0 <= x and x + width <= drawer.width, f"bin {bin_id} runs off drawer {slot.drawer} horizontally"
        assert 0 <= y and y + height <= drawer.height, f"bin {bin_id} runs off drawer {slot.drawer} vertically"
        total += width * height

    covered = _occupied(footprints, result, drawers)
    assert sum(len(cells) for cells in covered) == total, "two bins share a cell"


# ------------------------------------------------------------------ drawers


def test_a_drawer_measures_in_whole_cells():
    assert DrawerCells(500, 750) == Drawer(11, 17)


def test_the_inter_bin_gap_comes_off_the_run_not_each_bin():
    """A run of n cells spans 42n - 0.5mm, because the half millimeter is
    already inside every bin's own footprint. Naive floor division would
    deny a 41.5mm drawer the cell that genuinely fits it.
    """
    assert DrawerCells(41.5, 41.5) == Drawer(1, 1)
    assert DrawerCells(83.5, 41.5) == Drawer(2, 1)


def test_a_drawer_too_small_for_one_cell_is_refused():
    with pytest.raises(ValueError, match="no whole"):
        DrawerCells(30, 30)


def test_a_bin_has_two_distinct_turns_where_a_part_has_four():
    """A part is asymmetric, so the solver tries all four quarter turns
    (`solver.FittingOrientations`). A bin's footprint is a rectangle, so 0
    and 180 degrees cover identical cells and only two footprints exist -
    enumerating two is complete, not approximate.
    """
    assert _Orientations(5, 2) == [(5, 2, False), (2, 5, True)]


def test_a_square_bin_has_only_one():
    """Nothing to gain by turning it, and trying both would rediscover
    every answer sideways.
    """
    assert _Orientations(3, 3) == [(3, 3, False)]


def test_a_drawer_holds_a_bin_at_either_quarter_turn():
    drawer = Drawer(5, 2)

    assert drawer.Holds((5, 2))
    assert drawer.Holds((2, 5)), "the same bin, turned"
    assert not drawer.Holds((6, 2))


# ---------------------------------------------------------------- placement


def test_bins_that_tile_exactly_are_placed():
    footprints = {0: (2, 1), 1: (2, 1)}
    drawers = [Drawer(2, 2)]

    result = Assign(footprints, drawers)

    assert result.outcome == PLACED
    assert len(result.slots) == 2
    _assert_sound(footprints, result, drawers)
    assert FreeCells(drawers, footprints, result) == [0]


def test_placements_never_overlap_or_escape_the_drawer():
    footprints = {0: (5, 2), 1: (3, 2), 2: (2, 1), 3: (4, 3), 4: (1, 1), 5: (6, 2)}
    drawers = [DrawerCells(500, 750)]

    result = Assign(footprints, drawers)

    assert result.outcome == PLACED
    _assert_sound(footprints, result, drawers)


def test_a_bin_is_turned_when_that_is_the_only_way_it_fits():
    footprints = {0: (4, 1)}
    drawers = [Drawer(1, 4)]

    result = Assign(footprints, drawers)

    assert result.outcome == PLACED
    assert result.slots[0].turned
    assert result.slots[0].Footprint((4, 1)) == (1, 4)


def test_bins_spread_across_drawers_when_one_will_not_hold_them():
    footprints = {0: (5, 2), 1: (5, 2)}
    drawers = [Drawer(5, 2), Drawer(5, 2)]

    result = Assign(footprints, drawers)

    assert result.outcome == PLACED
    assert {slot.drawer for slot in result.slots.values()} == {0, 1}


def test_the_leftover_room_stays_usable():
    """The property that actually matters: the next object photographed
    has to go somewhere, and free cells alone do not say whether it can.

    Stated as "can another bin still be added" rather than "is the free
    space one region", because bottom-left stability does not in fact
    guarantee the latter - measured here, 143 of 144 free cells connect
    and one is stranded behind the 1x1.
    """
    footprints = {0: (5, 2), 1: (3, 2), 2: (2, 1), 3: (4, 3), 4: (1, 1), 5: (6, 2)}
    drawers = [DrawerCells(500, 750)]

    result = Assign(footprints, drawers)
    assert result.outcome == PLACED

    with_another = Assign({**footprints, 6: (5, 2)}, drawers)

    assert with_another.outcome == PLACED, "a whole extra bin should still fit in what is left"


def test_contiguity_is_reported_rather_than_assumed():
    """`FreeCells` counts, `LargestFreeRegion` says how much of it is in
    one piece. The gap between them is exactly the space that is free and
    unusable.
    """
    footprints = {0: (2, 2)}
    drawers = [Drawer(4, 4)]

    result = Assign(footprints, drawers)

    assert FreeCells(drawers, footprints, result) == [12]
    assert LargestFreeRegion(drawers, footprints, result) == [12], "an L of free space is still one region"


def test_a_stranded_cell_shows_up_as_a_smaller_region():
    """A 1x1 hole a bin cannot reach is free space that counts for
    nothing, and the two numbers have to disagree when that happens.
    """
    footprints = {0: (2, 1), 1: (1, 2), 2: (2, 1)}
    drawers = [Drawer(3, 3)]

    result = Assign(footprints, drawers)
    assert result.outcome == PLACED

    free = FreeCells(drawers, footprints, result)[0]
    largest = LargestFreeRegion(drawers, footprints, result)[0]

    assert largest <= free
    assert free == 9 - 6


# --------------------------------------------------------------- the bounds


def test_more_cells_than_the_drawers_hold_is_refused_without_searching():
    result = Assign({0: (2, 2), 1: (1, 2)}, [Drawer(2, 2)])

    assert result.outcome == INFEASIBLE
    assert "6 cells" in result.detail and "4" in result.detail


def test_a_bin_no_drawer_could_hold_is_named():
    result = Assign({0: (2, 1), 1: (7, 1)}, [Drawer(6, 3)])

    assert result.outcome == INFEASIBLE
    assert result.unplaced == [1]
    assert "no drawer is large enough" in result.detail


def test_the_bounds_never_reject_an_assignment_that_would_have_worked():
    """One-sided, the same requirement M4 and M8 record. An over-eager
    bound at this level silently reports "buy another drawer".
    """
    exactly_full = {0: (2, 2), 1: (2, 2), 2: (2, 2), 3: (2, 2)}

    result = Assign(exactly_full, [Drawer(4, 4)])

    assert result.outcome == PLACED, "16 cells of bins into exactly 16 cells of drawer"


# ------------------------------------------------------------ proving no fit


def test_infeasible_means_proven_not_merely_unfound():
    """Three 2x2 bins into a 3x4 drawer. The area bound passes exactly -
    12 cells into 12 - so this can only be settled by searching, and the
    answer is that it genuinely does not tile.
    """
    result = Assign({0: (2, 2), 1: (2, 2), 2: (2, 2)}, [Drawer(3, 4)])

    assert result.outcome == INFEASIBLE
    assert len(result.slots) == 2, "and it says how far it got"
    assert result.unplaced == [2]


def test_a_failed_assignment_names_what_it_could_not_place():
    """Which is what makes re-grouping possible: the level below needs to
    know which footprints to stop proposing.
    """
    footprints = {index: (2, 2) for index in range(15)}

    result = Assign(footprints, [Drawer(7, 9)])

    assert result.outcome == INFEASIBLE
    assert len(result.slots) == 12, "a 7x9 drawer holds twelve 2x2 bins and no more"
    assert len(result.unplaced) == 3
    _assert_sound(footprints, result, [Drawer(7, 9)])


def test_running_out_of_budget_is_not_a_proof():
    """The distinction the whole level rests on. A search that gave up must
    never be read as "these bins do not fit".
    """
    result = Assign({0: (2, 2), 1: (2, 2), 2: (2, 2)}, [Drawer(3, 4)], budget=2)

    assert result.outcome == EXHAUSTED
    assert result.outcome != INFEASIBLE
    assert "not evidence" in result.detail


def test_the_default_budget_is_not_reached_by_a_realistic_drawer():
    footprints = {0: (5, 2), 1: (3, 2), 2: (2, 1), 3: (4, 3), 4: (1, 1), 5: (6, 2), 6: (2, 2), 7: (5, 1)}

    result = Assign(footprints, [DrawerCells(500, 750)], budget=DEFAULT_NODE_BUDGET)

    assert result.outcome == PLACED


# ------------------------------------------------------------------ refusals


def test_nothing_to_assign_is_refused():
    with pytest.raises(ValueError, match="nothing to assign"):
        Assign({}, [Drawer(2, 2)])


def test_no_drawers_is_refused():
    with pytest.raises(ValueError, match="no drawers"):
        Assign({0: (1, 1)}, [])


def test_a_fractional_footprint_is_refused():
    with pytest.raises(ValueError, match="whole cells"):
        Assign({0: (0, 2)}, [Drawer(2, 2)])


def test_a_drawer_smaller_than_a_cell_is_refused():
    with pytest.raises(ValueError, match="at least 1x1"):
        Drawer(0, 3)


# ---------------------------------------------------------- feeding it back


def test_admissible_footprints_are_those_some_drawer_can_hold():
    admissible = AdmissibleFootprints([Drawer(6, 3)], max_grid=6)

    assert (6, 3) in admissible
    assert (3, 3) in admissible
    assert (6, 4) not in admissible, "too deep for the drawer at either turn"


def test_candidate_grids_drop_what_no_drawer_can_hold():
    """The feedback edge. A bin 7 cells long cannot go in a 6-cell drawer
    at any angle, so packing one wastes the entire search below it.
    """
    admissible = AdmissibleFootprints([Drawer(4, 2)], max_grid=6)

    grids = CandidateGrids(6, admissible)

    assert (4, 2) in grids
    assert (5, 1) not in grids
    assert all(Drawer(4, 2).Holds(grid) for grid in grids)


def test_restricting_grids_keeps_them_smallest_first():
    admissible = AdmissibleFootprints([Drawer(4, 3)], max_grid=6)

    grids = CandidateGrids(6, admissible)

    areas = [n * m for n, m in grids]
    assert areas == sorted(areas)


def test_unrestricted_grids_are_unchanged():
    assert CandidateGrids(4, None) == CandidateGrids(4)


# -------------------------------------------------------------- the reports


def test_the_report_says_where_every_bin_went():
    footprints = {0: (2, 1), 1: (2, 1)}
    result = Assign(footprints, [Drawer(2, 2)])

    report = result.Report(footprints)

    assert "bin 0" in report and "drawer 0" in report
    assert "all 2 bins placed" in report


def test_the_report_of_a_failure_names_the_leftovers():
    footprints = {0: (2, 2), 1: (2, 2), 2: (2, 2)}
    result = Assign(footprints, [Drawer(3, 4)])

    report = result.Report(footprints)

    assert "infeasible" in report
    assert "could not place bins 2" in report


def test_free_cells_are_reported_per_drawer():
    """Twelve cells free across three drawers is not the same offer as
    twelve in one.
    """
    footprints = {0: (2, 2)}
    drawers = [Drawer(2, 2), Drawer(3, 3)]

    result = Assign(footprints, drawers)

    assert FreeCells(drawers, footprints, result) == [0, 9]


def test_a_slot_reports_the_footprint_as_it_sits():
    assert Slot(0, 0, (0, 0), turned=True).Footprint((5, 2)) == (2, 5)
    assert Slot(0, 0, (0, 0), turned=False).Footprint((5, 2)) == (5, 2)
