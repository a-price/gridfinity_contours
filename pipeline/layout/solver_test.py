"""Tests for the fixed-grid solver (M3).

The solver is stochastic, so these assert invariants and seeded
determinism rather than exact coordinates. The one thing checked
absolutely is that a returned layout never has parts on top of each other:
a solver that fails to find an arrangement is an inconvenience, whereas
one that returns a bad arrangement produces a printed bin that does not
hold the objects.

Budgets are cut well below the defaults throughout, so the suite stays
quick; the cases are chosen to be comfortably solvable at that budget.
"""

import itertools

from dataclasses import replace

import numpy as np
import pytest

from pipeline.layout.container import BuildContainer, InteriorSpan
from pipeline.layout.energy import ComputeEnergy, PlacementEnergy
from pipeline.layout.loading import BuildParts, LoadParts
from pipeline.layout.packer import Pack
from pipeline.layout.parameters import EIGHTH_TURNS, FREE_ROTATION, QUARTER_TURNS, LayoutParameters
from pipeline.layout.placement import Placement, Pose
from pipeline.layout.solver import (
    CandidatePoses,
    FitsAtSomeAngle,
    FittingPoses,
    _ContactCorners,
    Relax,
    SolveFixedGrid,
    _ChoosePoses,
    _ConstructiveInit,
)
from pipeline.layout.verify import CheckLayout, PolygonsOverlap
from conftest import QuickParameters as _quick, Rectangle as _rectangle, SPOONS


def _three_squares(params: LayoutParameters):
    return BuildParts({i: _rectangle(30, 30) for i in range(3)}, params)


def _no_pair_overlaps(layout, parts) -> bool:
    placed = [placement.ToWorld(parts[part_id]) for part_id, placement in sorted(layout.placements.items())]
    return not any(PolygonsOverlap(placed[i], placed[j]) for i in range(len(placed)) for j in range(i + 1, len(placed)))


# ------------------------------------------------------------ orientations


def test_fitting_orientations_accepts_all_four_when_a_part_fits_either_way():
    params = _quick()
    (part,) = BuildParts({0: _rectangle(20, 20)}, params).values()

    assert FittingPoses(part, BuildContainer(2, 2, params.inset), params) == [Pose(0), Pose(1), Pose(2), Pose(3)]


def test_fitting_orientations_rejects_turning_a_long_part_across_a_narrow_bin():
    """A 100mm part in a bin 120mm one way and 36mm the other can only lie
    along the long axis - the quarter turns that stand it up are hopeless
    at every position, so the solver should never spend an attempt on one.
    """
    params = _quick()
    (part,) = BuildParts({0: _rectangle(100, 20)}, params).values()

    assert FittingPoses(part, BuildContainer(3, 1, params.inset), params) == [Pose(0), Pose(2)]


def test_fitting_orientations_is_empty_when_a_part_cannot_fit_at_all():
    params = _quick()
    (part,) = BuildParts({0: _rectangle(100, 100)}, params).values()

    assert FittingPoses(part, BuildContainer(1, 1, params.inset), params) == []


def test_attempts_walk_the_ranking_best_first():
    """The first attempt gets the best-scoring assignment, not a guess.

    A bad orientation is not a slow attempt but a doomed one, since nothing
    in the relaxation can turn a part - so the ranking is spent in order.
    """
    ranked = [{0: Pose(0), 1: Pose(2)}, {0: Pose(2), 1: Pose(0)}, {0: Pose(0), 1: Pose(0)}]

    assert [_ChoosePoses(ranked, attempt) for attempt in range(3)] == ranked


def test_attempts_past_the_ranking_cycle_rather_than_stop():
    """Running out of assignments must not end the search: the constructive
    initializer sweeps from a different corner and draws different
    positions each attempt, so the same orientations get a genuinely
    different starting arrangement.
    """
    ranked = [{0: Pose(0)}, {0: Pose(2)}]

    assert [_ChoosePoses(ranked, attempt) for attempt in range(2, 6)] == ranked * 2


# ---------------------------------------------------- constructive startup


def test_constructive_init_places_parts_without_overlapping():
    """The point of building an arrangement before relaxing it: the
    relaxation's forces reverse for deep overlap, so the descent must not
    start inside one.
    """
    params = _quick()
    parts = BuildParts({i: _rectangle(24, 20) for i in range(5)}, params)
    container = BuildContainer(3, 2, params.inset)

    for attempt in range(8):
        rng = np.random.default_rng([0, attempt])
        placements = _ConstructiveInit(parts, {i: Pose(0) for i in parts}, container, params, rng, attempt)

        placed = [placements[i].ToWorld(parts[i]) for i in sorted(parts)]
        for i in range(len(placed)):
            for j in range(i + 1, len(placed)):
                assert not PolygonsOverlap(placed[i], placed[j]), f"attempt {attempt}, parts {i} and {j}"


def test_constructive_init_solves_a_dense_bin_outright():
    """Four 50x30 parts pack into a 3x2 as an obvious 2x2 grid (103.2 wide
    against 116.4 available, 63.2 tall against 74.4), and bottom-left fill
    should find it without the relaxation being involved at all.

    Random candidate positions could not: measured at 0 successes in 30
    attempts, and no better with the budget raised sixteen-fold, because
    once parts fill most of the bin the feasible region is a vanishing
    fraction of it.
    """
    params = _quick()
    parts = BuildParts({i: _rectangle(50, 30) for i in range(4)}, params)
    container = BuildContainer(3, 2, params.inset)

    placements = _ConstructiveInit(parts, {i: Pose(0) for i in parts}, container, params, np.random.default_rng(0), 0)

    assert ComputeEnergy(parts, placements, container, params).feasible


def test_contact_positions_clear_the_rasters_own_error():
    """Contacts are offset slightly further apart than the clearance
    strictly demands, because a part placed at exactly c_pair measures as
    violating it - the distance field is rasterized, so separation comes
    back short by the discretization error.

    Without the margin every contact position prices as infeasible and the
    sweep degrades into the random search it exists to replace. This pins
    the margin to the symptom rather than to its size.
    """
    params = _quick()
    parts = BuildParts({0: _rectangle(50, 30), 1: _rectangle(50, 30)}, params)
    container = BuildContainer(3, 2, params.inset)
    placed = {0: Placement(0, np.array([params.c_wall, params.c_wall]))}

    positions = _ContactCorners(np.array([50.0, 30.0]), container, parts, placed, params)

    beside = [p for p in positions if p[0] > 50.0]
    assert beside, "expected a contact position alongside the placed part"
    for position in beside:
        candidate = {**placed, 1: Placement(1, position)}
        assert PlacementEnergy(1, parts, candidate, container, params) <= 0.0


def test_constructive_init_still_returns_placements_when_nothing_fits():
    """Best-effort, not failure: a part with nowhere to go is the
    relaxation's problem and the restart loop's, not a reason to bail out
    before either has run.
    """
    params = _quick()
    parts = BuildParts({i: _rectangle(30, 30) for i in range(6)}, params)
    container = BuildContainer(1, 1, params.inset)

    placements = _ConstructiveInit(parts, {i: Pose(0) for i in parts}, container, params, np.random.default_rng(0), 0)

    assert set(placements) == set(parts)


def test_constructive_init_places_the_largest_part_first():
    """Big parts are the constrained ones - a bin that cannot take them
    cannot take them in any order.
    """
    params = _quick()
    parts = BuildParts({0: _rectangle(10, 10), 1: _rectangle(90, 25), 2: _rectangle(30, 20)}, params)
    container = BuildContainer(3, 1, params.inset)

    placements = _ConstructiveInit(parts, {i: Pose(0) for i in parts}, container, params, np.random.default_rng(0), 0)

    # The largest had free rein, so it landed somewhere legal on its own.
    big = placements[1].ToWorld(parts[1])
    assert big[:, 0].min() >= params.c_wall - 1e-9
    assert big[:, 0].max() <= container.width - params.c_wall + 1e-9


# ------------------------------------------------------------- relaxation


def test_relax_separates_lightly_overlapping_parts():
    params = _quick()
    parts = BuildParts({0: _rectangle(30, 30), 1: _rectangle(30, 30)}, params)
    container = BuildContainer(3, 2, params.inset)
    # Touching, well inside the shallow regime the forces handle.
    placements = {0: Placement(0, np.array([20.0, 20.0])), 1: Placement(1, np.array([48.0, 24.0]))}

    settled = Relax(parts, placements, container, params, np.random.default_rng(0))

    assert settled is not None
    assert ComputeEnergy(parts, settled, container, params).feasible


def test_relax_refuses_to_descend_from_a_swallowed_part():
    """The M3 stacking criterion. Identical parts placed on top of each
    other are fully contained, where the forces point the wrong way - the
    solver must abandon rather than converge on nonsense.
    """
    params = _quick()
    parts = BuildParts({0: _rectangle(30, 30), 1: _rectangle(30, 30)}, params)
    container = BuildContainer(3, 2, params.inset)
    stacked = {0: Placement(0, np.array([20.0, 20.0])), 1: Placement(1, np.array([20.0, 20.0]))}

    assert Relax(parts, stacked, container, params, np.random.default_rng(0)) is None


def test_relax_never_returns_an_overlapping_arrangement():
    """Whatever the starting mess, a non-None result is a real answer."""
    params = _quick()
    parts = BuildParts({i: _rectangle(28, 22) for i in range(4)}, params)
    container = BuildContainer(3, 2, params.inset)
    rng = np.random.default_rng(3)

    for _ in range(12):
        start = {i: Placement(i, rng.uniform(5.0, 60.0, size=2)) for i in parts}
        settled = Relax(parts, start, container, params, rng)
        if settled is None:
            continue

        placed = [settled[i].ToWorld(parts[i]) for i in sorted(parts)]
        for i in range(len(placed)):
            for j in range(i + 1, len(placed)):
                assert not PolygonsOverlap(placed[i], placed[j])


def test_relax_gives_up_on_a_stalled_arrangement():
    """Without a stall check a hopeless bin costs the full iteration budget
    on every restart. This one cannot be solved, so it should end early.
    """
    params = _quick(iterations=4000, patience=20)
    parts = BuildParts({i: _rectangle(30, 30) for i in range(5)}, params)
    container = BuildContainer(1, 1, params.inset)
    start = {i: Placement(i, np.array([5.0 + i, 5.0])) for i in parts}

    import time

    began = time.monotonic()
    assert Relax(parts, start, container, params, np.random.default_rng(0)) is None
    # Far less than 4000 iterations' worth; the point is that it bailed.
    assert time.monotonic() - began < 20.0


# ---------------------------------------------------------------- solving


def test_three_squares_pack_into_a_one_by_three():
    """The M3 packing criterion."""
    params = _quick()
    parts = _three_squares(params)

    layout = SolveFixedGrid(parts, 1, 3, params)

    assert layout is not None
    assert layout.grid == (1, 3)
    assert set(layout.placements) == set(parts)
    assert CheckLayout(layout, parts) == []


@pytest.mark.parametrize("seed", [0, 1, 2, 3, 4])
def test_three_squares_pack_repeatably_across_seeds(seed):
    params = _quick(seed=seed)
    parts = _three_squares(params)

    layout = SolveFixedGrid(parts, 1, 3, params)

    assert layout is not None, f"seed {seed} failed a packing that fits comfortably"
    assert CheckLayout(layout, parts) == []


def test_the_same_seed_reproduces_the_same_layout():
    """The M3 determinism criterion. A layout that did not reproduce would
    not match the sheet printed alongside it.
    """
    parts = _three_squares(_quick())

    first = SolveFixedGrid(parts, 1, 3, _quick(seed=7))
    second = SolveFixedGrid(parts, 1, 3, _quick(seed=7))

    assert first is not None and second is not None
    assert first.grid == second.grid
    for part_id, placement in first.placements.items():
        assert np.array_equal(placement.position, second.placements[part_id].position)
        assert placement.orientation == second.placements[part_id].orientation


def test_an_easy_bin_packs_the_same_way_whatever_the_seed():
    """Bottom-left fill is deterministic and the first attempt uses it
    unchanged, so a bin it solves outright gives the same answer for every
    seed - the seed only starts to matter once that first attempt fails and
    the randomized restarts take over.

    Worth pinning: it means an easy layout is reproducible even if someone
    changes the seed, and it is the reason the restart loop can afford to
    be stochastic without making everyday results unstable.
    """
    parts = _three_squares(_quick())

    layouts = [SolveFixedGrid(parts, 1, 3, _quick(seed=seed)) for seed in range(4)]

    assert all(layout is not None for layout in layouts)
    positions = {tuple(np.round(layout.placements[0].position, 6)) for layout in layouts if layout}
    assert len(positions) == 1


def test_an_overfull_bin_fails_instead_of_returning_an_overlap():
    """The M3 clean-failure criterion."""
    params = _quick(restarts=3, iterations=60)
    parts = BuildParts({i: _rectangle(30, 25) for i in range(6)}, params)

    assert SolveFixedGrid(parts, 1, 1, params) is None


def test_a_part_too_large_for_the_bin_fails_immediately():
    """No orientation fits, so no search can help - this should cost
    nothing rather than the whole restart budget.
    """
    params = _quick(restarts=1000, iterations=100000)
    parts = BuildParts({0: _rectangle(200, 40)}, params)

    import time

    began = time.monotonic()
    assert SolveFixedGrid(parts, 1, 1, params) is None
    assert time.monotonic() - began < 5.0


def test_every_returned_layout_survives_the_independent_check():
    """Zero energy already implies no overlap, but the guarantee is worth
    holding to exact geometry rather than to the solver's own field.
    """
    params = _quick()
    sizes = [(24, 20), (30, 18), (16, 16), (28, 12)]

    for count in range(2, 5):
        parts = BuildParts({i: _rectangle(*sizes[i]) for i in range(count)}, params)
        layout = SolveFixedGrid(parts, 3, 2, params)

        assert layout is not None, f"{count} parts in a 3x2 should be solvable"
        assert CheckLayout(layout, parts) == []
        assert _no_pair_overlaps(layout, parts)


def test_layouts_respect_the_configured_clearances():
    """Not merely non-overlapping: the dividers between pockets have to be
    thick enough to print.
    """
    params = _quick()
    parts = BuildParts({i: _rectangle(24, 20) for i in range(4)}, params)

    layout = SolveFixedGrid(parts, 3, 2, params)

    assert layout is not None
    assert CheckLayout(layout, parts, pair_clearance=params.c_pair, wall_clearance=params.c_wall) == []


# -------------------------------------------------------------- fixtures


@pytest.mark.slow
def test_the_three_spoons_pack_into_a_five_by_two():
    """Real geometry, and the M4 target. This one genuinely needs concave
    nesting - the big and medium spoons cannot stack (41.67 + 34.89 + a
    clearance exceeds the 78.3mm interior) and cannot sit end to end
    (200.26 + 162.76 far exceeds 204.3), so the only way both fit is with
    one's bowl beside the other's handle.
    """
    params = LayoutParameters(patience=25)
    parts = LoadParts(SPOONS, params)

    layout = SolveFixedGrid(parts, 5, 2, params)

    assert layout is not None
    assert CheckLayout(layout, parts) == []
    assert _no_pair_overlaps(layout, parts)


def test_the_spoons_cannot_fit_a_five_by_one():
    """The big spoon is 41.67mm across a 36.3mm interior, so no orientation
    fits and this needs no search at all.
    """
    params = _quick()
    parts = LoadParts(SPOONS, params)

    assert SolveFixedGrid(parts, 5, 1, params) is None


def test_interior_spans_used_by_these_tests_are_what_we_think():
    # Guards the arithmetic the feasibility claims above rest on.
    assert InteriorSpan(1) == pytest.approx(36.3)
    assert InteriorSpan(3) == pytest.approx(120.3)
    assert InteriorSpan(5) == pytest.approx(204.3)


def test_every_restart_is_reported():
    """The progress hook fires once per restart, which is what lets a GUI
    show motion during the one grid size that takes real time.

    Checked against a bin nothing can pack, so the full budget runs and the
    count is deterministic - a solvable case stops at whichever attempt
    happens to work.
    """
    params = _quick(restarts=5, iterations=40)
    parts = BuildParts({i: _rectangle(30, 30) for i in range(4)}, params)
    seen = []

    layout = SolveFixedGrid(parts, 2, 1, params, on_attempt=seen.append)

    assert layout is None, "this bin should defeat the search, or the count below is not the budget"
    assert seen == [0, 1, 2, 3, 4]


def test_the_progress_hook_is_optional():
    params = _quick(restarts=2, iterations=40)
    parts = BuildParts({0: _rectangle(30, 25)}, params)

    assert SolveFixedGrid(parts, 1, 1, params) is not None


@pytest.mark.slow
def test_the_answer_does_not_depend_on_the_order_the_files_were_listed():
    """The bug this ranking was built to fix.

    Part ids come from the order contour files were named, and the search
    used to draw one orientation per part from a single seeded stream in id
    order - so which part got which draw, and therefore which arrangements
    the restarts explored, depended on how the command line was typed. Five
    of the six orderings of these three spoons found the 10-cell bin and the
    sixth returned a 12-cell one.
    """
    params = LayoutParameters()
    grids = set()

    for order in itertools.permutations(SPOONS):
        result = Pack(LoadParts(list(order), params), params)
        assert result.layout is not None, f"no bin found for {order}"
        grids.add(result.layout.grid)

    assert grids == {(5, 2)}, f"file order changed the answer: {sorted(grids)}"


# ------------------------------------------------------------- rotation modes


def test_the_candidate_poses_are_four_at_ninety_and_eight_otherwise():
    """FREE gets the eight of the 45-degree mode rather than the four of
    90. The relaxation can turn a part, but only downhill, and it will not
    cross the 45 degrees between quarter turns when everything between is
    worse - so seeding on the diagonals puts a start within 22.5 degrees of
    any angle the search might want.
    """
    assert len(CandidatePoses(_quick(rotation=QUARTER_TURNS))) == 4
    assert len(CandidatePoses(_quick(rotation=EIGHTH_TURNS))) == 8
    assert len(CandidatePoses(_quick(rotation=FREE_ROTATION))) == 8


def test_every_candidate_pose_is_a_distinct_angle():
    poses = CandidatePoses(_quick(rotation=EIGHTH_TURNS))

    turns = sorted(round(np.rad2deg(pose.total) % 360.0, 6) for pose in poses)
    assert turns == [0.0, 45.0, 90.0, 135.0, 180.0, 225.0, 270.0, 315.0]


def test_a_diagonal_pose_fits_a_bin_no_quarter_turn_does():
    """The mechanism the whole experiment is about, on a shape whose
    numbers are obvious. A 1x1 interior is 36.3mm and leaves 32.4mm once
    both wall clearances are taken, so a 40mm bar does not fit square-on -
    but that square's diagonal is 45.8mm, and the bar crosses it with room
    to spare.
    """
    params = _quick()
    (part,) = BuildParts({0: _rectangle(40, 4)}, params).values()
    container = BuildContainer(1, 1, params.inset)

    assert FittingPoses(part, container, replace(params, rotation=QUARTER_TURNS)) == []
    assert FittingPoses(part, container, replace(params, rotation=EIGHTH_TURNS)) != []


def test_fitting_poses_at_ninety_are_exactly_what_they_always_were():
    """Adding poses must not change the 90-degree answer, since every
    committed layout was packed with it.
    """
    params = _quick()
    (part,) = BuildParts({0: _rectangle(100, 20)}, params).values()

    assert FittingPoses(part, BuildContainer(3, 1, params.inset), params) == [Pose(0), Pose(2)]


# --------------------------------------------------- the bound under free rotation


def test_a_part_far_too_long_is_still_proven_not_to_fit():
    """The bound has to stay strong as well as sound. A 243mm knife is
    23mm wide, so a minimum-width test alone would say a 1x1 bin might work
    and the search would spend its whole budget finding out otherwise - the
    diameter against the diagonal is what actually rejects it.
    """
    params = _quick(rotation=FREE_ROTATION)
    (part,) = BuildParts({0: _rectangle(243, 23)}, params).values()

    assert not FitsAtSomeAngle(part, BuildContainer(1, 1, params.inset), params)


def test_the_bound_does_not_reject_a_bin_a_diagonal_fits():
    """The error this bound must never make. A bar too long for the bin
    square-on still fits across its diagonal, and rejecting that would be a
    proof of something false.
    """
    params = _quick(rotation=FREE_ROTATION)
    (part,) = BuildParts({0: _rectangle(40, 4)}, params).values()
    container = BuildContainer(1, 1, params.inset)

    assert FitsAtSomeAngle(part, container, params)
    assert not FittingPoses(part, container, replace(params, rotation=QUARTER_TURNS))


def test_a_part_that_fits_squarely_obviously_passes_the_bound():
    params = _quick(rotation=FREE_ROTATION)
    (part,) = BuildParts({0: _rectangle(20, 20)}, params).values()

    assert FitsAtSomeAngle(part, BuildContainer(2, 2, params.inset), params)


def test_a_bin_with_no_room_left_after_clearance_fits_nothing():
    params = _quick(rotation=FREE_ROTATION, wall_clearance=40.0)
    (part,) = BuildParts({0: _rectangle(5, 5)}, params).values()

    assert not FitsAtSomeAngle(part, BuildContainer(1, 1, params.inset), params)
