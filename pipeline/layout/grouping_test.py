"""Tests for partitioning parts across bins (M8).

Three things are pinned here that nothing else can check. That the result
is a *partition* - every part in exactly one bin, none invented and none
dropped, which a cell count alone would never reveal. That the lower bound
the search prunes with is **one-sided**: a bound that ever overestimates
discards good moves silently, and the grouping just comes back bigger
looking perfectly correct. And that every bin the search produces still
survives the independent geometric check, since a grouping is only as
sound as the layouts it is made of.
"""

from typing import Any

import numpy as np
import pytest

from pipeline.layout.drawer import AdmissibleFootprints, Drawer
from pipeline.layout.grouping import (
    FirstFit,
    Group,
    Grouping,
    Improve,
    OnePerBin,
    _Improve,
    _OnePerBin,
    _Oracle,
)
from pipeline.layout.loading import BuildParts, LoadParts
from pipeline.layout.parameters import LayoutParameters
from pipeline.layout.part import Part
from pipeline.layout.verify import CheckLayout

SPOONS = ["test_data/big_spoon.svg", "test_data/medium_spoon.svg", "test_data/small_spoon.svg"]


def _rectangle(width: float, height: float, x: float = 0.0, y: float = 0.0) -> np.ndarray:
    return np.array([[x, y], [x + width, y], [x + width, y + height], [x, y + height]], dtype=np.float64)


def _quick(**overrides) -> LayoutParameters:
    settings: dict[str, Any] = dict(restarts=6, iterations=150, patience=25)
    settings.update(overrides)
    return LayoutParameters(**settings)


def _parts(shapes: list[tuple[float, float]], params: LayoutParameters) -> dict[int, Part]:
    return BuildParts({index: _rectangle(*shape) for index, shape in enumerate(shapes)}, params)


def _rider(params: LayoutParameters) -> dict[int, Part]:
    """A long part whose bin has room to spare, and a small one that fits
    in it.

    130x25 needs a 4x1 (162.3 x 36.3mm interior) for its length alone and
    leaves most of that bin empty; the 20x20 fits alongside it with 5mm to
    spare. Separately they cost 4 + 1 cells, together 4 - so this is the
    case grouping exists for, in miniature.
    """
    return _parts([(130, 25), (20, 20)], params)


def _twins(params: LayoutParameters) -> dict[int, Part]:
    """Two parts that each fill their own bin and gain nothing by sharing.

    30x30 fits a 1x1 with 2.4mm to spare and two of them need a 2x1, so
    merging trades two one-cell bins for one two-cell bin and improves
    nothing.
    """
    return _parts([(30, 30), (30, 30)], params)


# -------------------------------------------------------------- partitioning


def test_every_part_lands_in_exactly_one_bin():
    params = _quick()
    parts = _parts([(130, 25), (20, 20), (30, 30)], params)

    grouping = Group(parts, params)

    placed = [part_id for contents in grouping.Contents() for part_id in contents]
    assert sorted(placed) == sorted(parts), "every part exactly once, none invented"


def test_every_bin_survives_the_independent_check():
    """A grouping is only as sound as the layouts it is made of, and those
    come from a stochastic solver.
    """
    params = _quick()
    parts = _parts([(130, 25), (20, 20), (30, 30)], params)

    grouping = Group(parts, params)

    for layout in grouping.bins:
        assert CheckLayout(layout, parts) == []


def test_grouping_never_costs_more_than_one_bin_each():
    params = _quick()
    parts = _rider(params)

    assert Group(parts, params).cells <= OnePerBin(parts, params).cells


def test_a_single_part_groups_into_a_single_bin():
    params = _quick()
    parts = _parts([(30, 30)], params)

    grouping = Group(parts, params)

    assert grouping.Contents() == [frozenset({0})]


def test_nothing_to_group_is_refused():
    with pytest.raises(ValueError, match="nothing to group"):
        Group({}, _quick())


def test_improving_a_grouping_of_parts_it_was_never_given_is_refused():
    params = _quick()
    parts = _twins(params)
    grouping = OnePerBin(parts, params)

    with pytest.raises(ValueError, match=r"parts \[1\]"):
        Improve({0: parts[0]}, grouping, params)


def test_the_report_names_every_bin_and_the_total():
    params = _quick()
    parts = _twins(params)

    report = OnePerBin(parts, params).Report()

    assert "bin 0" in report and "bin 1" in report
    assert "2 bins, 2 cells total" in report


# -------------------------------------------------------------------- oracle


def test_the_same_set_is_only_packed_once():
    """The search revisits sets constantly - moving a part out of a bin and
    back in asks about one already priced.
    """
    params = _quick()
    oracle = _Oracle(_twins(params), params)

    oracle.Smallest(frozenset({0, 1}))
    after_first = oracle.solver_calls
    oracle.Smallest(frozenset({0, 1}))

    assert after_first > 0, "the first call must actually have searched"
    assert oracle.solver_calls == after_first


def test_the_two_questions_share_one_cache():
    """`Smallest` is built on `FitsIn`, so asking a set's smallest size has
    already answered whether it fits that size.
    """
    params = _quick()
    oracle = _Oracle(_twins(params), params)

    layout = oracle.Smallest(frozenset({0, 1}))
    assert layout is not None
    after = oracle.solver_calls
    oracle.FitsIn(frozenset({0, 1}), layout.grid)

    assert oracle.solver_calls == after


def test_a_set_that_provably_cannot_fit_never_reaches_the_solver():
    params = _quick()
    oracle = _Oracle(_rider(params), params)

    assert oracle.FitsIn(frozenset({0}), (1, 1)) is None
    assert oracle.solver_calls == 0


def test_the_bound_never_exceeds_what_packing_actually_costs():
    """The property the whole search leans on. The bound prunes moves
    without running the solver, so a bound that ever came out *above* the
    real cost would discard improvements silently - the grouping would just
    return something bigger and look entirely correct.
    """
    params = _quick()
    parts = _parts([(130, 25), (20, 20), (30, 30)], params)
    oracle = _Oracle(parts, params)

    for ids in (frozenset({0}), frozenset({1}), frozenset({2}), frozenset({0, 1}), frozenset({1, 2})):
        bound, actual = oracle.LowerBoundCells(ids), oracle.Cells(ids)
        assert bound is not None and actual is not None
        assert bound <= actual, f"bound {bound} exceeds the {actual} cells {sorted(ids)} really needs"


def test_an_empty_bin_costs_nothing():
    """The local search empties bins, and an emptied bin is one bin fewer
    rather than an impossible one.
    """
    params = _quick()
    oracle = _Oracle(_twins(params), params)

    assert oracle.Cells(frozenset()) == 0
    assert oracle.LowerBoundCells(frozenset()) == 0


# ----------------------------------------------------------------- first fit


def test_first_fit_seats_a_small_part_in_a_big_parts_leftover_room():
    params = _quick()

    grouping = FirstFit(_rider(params), params)

    assert grouping.Contents() == [frozenset({0, 1})]
    assert grouping.cells == 4


def test_first_fit_opens_a_new_bin_when_no_open_one_takes_the_part():
    params = _quick()

    grouping = FirstFit(_twins(params), params)

    assert len(grouping.bins) == 2


def test_first_fit_does_not_grow_a_bin_to_make_room():
    """Growing is a trade - one bin gets more expensive to save another -
    and pricing that trade is the local search's job. Made greedily here,
    in whatever order the parts arrived, it would be committed to without
    ever being compared against the alternative.
    """
    params = _quick()
    parts = _twins(params)

    grouping = FirstFit(parts, params)

    assert [layout.grid for layout in grouping.bins] == [(1, 1), (1, 1)]
    assert Group(parts, params).cells == grouping.cells, "and growing would not have helped here anyway"


# --------------------------------------------------------------- local search


def test_the_local_search_merges_bins_that_should_share():
    params = _quick()
    parts = _rider(params)
    separate = OnePerBin(parts, params)
    assert separate.cells == 5, "the fixture must start with something to gain"

    improved = Improve(parts, separate, params)

    assert improved.Contents() == [frozenset({0, 1})]
    assert improved.cells == 4


def test_the_local_search_leaves_an_arrangement_it_cannot_beat():
    """Improvement has to be strict, or the search would churn between
    equal-cost groupings forever.
    """
    params = _quick()
    parts = _twins(params)
    separate = OnePerBin(parts, params)

    improved = Improve(parts, separate, params)

    assert improved.cells == separate.cells
    assert improved.Contents() == separate.Contents()


def test_hopeless_moves_are_priced_without_the_solver():
    """Two 30x30 squares cannot gain by sharing, and the bound alone says
    so - the local search should not pack a single candidate to find out.
    """
    params = _quick()
    oracle = _Oracle(_twins(params), params)
    bins = _OnePerBin(oracle, [0, 1])
    after_setup = oracle.solver_calls

    _Improve(oracle, bins)

    assert oracle.solver_calls == after_setup


def test_the_local_search_preserves_the_parts_it_was_given():
    params = _quick()
    parts = _parts([(130, 25), (20, 20), (30, 30)], params)

    improved = Improve(parts, OnePerBin(parts, params), params)

    assert improved.PartIds() == frozenset(parts)


def test_an_empty_grouping_improves_to_nothing():
    params = _quick()

    assert Improve(_twins(params), Grouping([]), params).bins == []


# ------------------------------------------------- restricted by the drawers


def test_grouping_never_proposes_a_footprint_no_drawer_can_hold():
    """The feedback edge from M9. A bin 5 cells long is useless if the
    only drawer is 4 wide, however well it packs.
    """
    admissible = AdmissibleFootprints([Drawer(4, 4)], max_grid=6)
    params = _quick(admissible_grids=admissible)
    parts = _parts([(130, 25), (20, 20), (30, 30)], params)

    grouping = Group(parts, params)

    for layout in grouping.bins:
        assert layout.grid in admissible, f"{layout.grid} does not fit the drawer"


def test_a_roomy_drawer_restricts_nothing():
    """The restriction has to be inert when it does not bite, or every
    grouping would depend on whether a drawer happened to be mentioned.
    """
    params = _quick()
    parts = _rider(params)
    roomy = _quick(admissible_grids=AdmissibleFootprints([Drawer(6, 6)], max_grid=6))

    assert Group(parts, roomy).Contents() == Group(parts, params).Contents()
    assert Group(parts, roomy).cells == Group(parts, params).cells


def test_a_part_no_admissible_bin_holds_says_so():
    """Rather than "does not fit any size", which is a confusing thing to
    read about a part that would fit a bin nobody can store.
    """
    params = _quick(admissible_grids=AdmissibleFootprints([Drawer(1, 1)], max_grid=6))
    parts = _parts([(130, 25)], params)

    with pytest.raises(ValueError, match="admissible set is restricted"):
        Group(parts, params)


# ------------------------------------------------------------- the fixtures


@pytest.mark.slow
def test_the_spoons_group_from_three_bins_into_one():
    """M8's completion criterion. One bin each costs 22 cells - the big and
    medium spoons need a 5x2 apiece for their length, and the small one a
    2x1 - while all three together fit in a single 5x2.
    """
    parts = LoadParts(SPOONS)

    separate = OnePerBin(parts)
    grouped = Group(parts)

    assert separate.cells == 22
    assert grouped.cells <= 10
    assert grouped.Contents() == [frozenset({0, 1, 2})]
    for layout in grouped.bins:
        assert CheckLayout(layout, parts) == []


@pytest.mark.slow
def test_the_local_search_alone_collapses_the_spoons():
    """Started from one bin per spoon rather than from first-fit, so the
    22-to-10 improvement is the search's own and not something the
    construction heuristic had already found.
    """
    parts = LoadParts(SPOONS)

    improved = Improve(parts, OnePerBin(parts))

    assert improved.Contents() == [frozenset({0, 1, 2})]
    assert improved.cells == 10
