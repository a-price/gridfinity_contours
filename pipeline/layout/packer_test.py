"""Tests for the grid size search (M4).

Two things are pinned here that the rest of the suite cannot check:
that the bounds are *sound* - they must never reject a bin that actually
fits, since that would silently inflate every result - and that whatever
comes back survives the independent geometric check.
"""

import numpy as np
import pytest

from pipeline.layout.container import BuildContainer, InteriorSpan
from pipeline.layout.energy import LayoutParameters
from pipeline.layout.loading import BuildParts, LoadParts
from pipeline.layout.packer import (
    NOT_FOUND,
    PACKED,
    TOO_SMALL,
    CandidateGrids,
    Pack,
    ProvablyTooSmall,
    RequiredArea,
)
from pipeline.layout.verify import CheckLayout

SPOONS = ["test_data/big_spoon.svg", "test_data/medium_spoon.svg", "test_data/small_spoon.svg"]


def _rectangle(width: float, height: float, x: float = 0.0, y: float = 0.0) -> np.ndarray:
    return np.array([[x, y], [x + width, y], [x + width, y + height], [x, y + height]], dtype=np.float64)


def _quick(**overrides) -> LayoutParameters:
    settings = dict(restarts=6, iterations=150, patience=25)
    settings.update(overrides)
    return LayoutParameters(**settings)


# ------------------------------------------------------------- enumeration


def test_candidate_grids_run_smallest_area_first():
    grids = CandidateGrids(3)

    areas = [n * m for n, m in grids]
    assert areas == sorted(areas)


def test_candidate_grids_prefer_square_among_equals():
    grids = CandidateGrids(4)

    four_cells = [grid for grid in grids if grid[0] * grid[1] == 4]
    assert four_cells == [(2, 2), (4, 1)]


def test_candidate_grids_omit_the_rotations_of_what_they_already_list():
    """A 2x5 is a 5x2 turned a quarter turn, and every part can turn too,
    so the two have identical solution sets. Enumerating both would double
    the search to rediscover each answer sideways.
    """
    grids = CandidateGrids(4)

    assert all(n >= m for n, m in grids)
    assert len(grids) == len(set(grids))


def test_candidate_grids_respect_the_cap():
    assert max(max(grid) for grid in CandidateGrids(3)) == 3
    with pytest.raises(ValueError):
        CandidateGrids(0)


# ----------------------------------------------------------------- bounds


def test_required_area_counts_the_clearance_band():
    """A part claims more than its own area: half the pair clearance all
    the way round, so two parts exactly c_pair apart just touch.
    """
    params = _quick()
    parts = BuildParts({0: _rectangle(20, 10)}, params)

    required = RequiredArea(parts, params)

    assert required > 200.0
    # Loosely, area + perimeter*r + pi*r^2 for r = c_pair/2 = 1.6.
    assert required == pytest.approx(200.0 + 60.0 * 1.6 + np.pi * 1.6**2, rel=0.05)


def test_required_area_does_not_overcount_a_concave_dilation():
    """The reason dilated area is measured off the distance field rather
    than from a perimeter formula: where a concave shape's dilation folds
    into itself the formula double-counts, and an overcount would make the
    bound unsound - rejecting bins that fit.

    A comb of thin teeth is the extreme case; its gaps fill in entirely.
    """
    params = _quick(pocket_offset=2.0)
    comb = np.array(
        [[0, 0], [30, 0], [30, 20], [24, 20], [24, 6], [18, 6], [18, 20], [12, 20], [12, 6], [6, 6], [6, 20], [0, 20]],
        dtype=np.float64,
    )
    parts = BuildParts({0: comb}, params)
    radius = params.c_pair / 2.0
    (part,) = parts.values()

    perimeter = float(np.linalg.norm(np.diff(np.vstack([part.contour, part.contour[:1]]), axis=0), axis=1).sum())
    formula = part.area + perimeter * radius + np.pi * radius**2

    assert part.DilatedArea(radius) < formula, "field measurement should come in under the convex formula"
    # And never below the part itself, which would be the opposite error.
    assert part.DilatedArea(radius) > part.area


def test_dilating_beyond_the_field_is_refused():
    params = _quick()
    (part,) = BuildParts({0: _rectangle(20, 10)}, params).values()

    with pytest.raises(ValueError, match="exceeds"):
        part.DilatedArea(part.pad + 1.0)


def test_container_area_accounts_for_rounded_corners():
    container = BuildContainer(2, 2)

    assert container.area < container.width * container.height
    assert container.area == pytest.approx(InteriorSpan(2) ** 2 - (4 - np.pi) * container.radius**2)


def test_a_part_too_large_is_reported_by_extent_not_area():
    params = _quick()
    parts = BuildParts({0: _rectangle(100, 20)}, params)

    reason = ProvablyTooSmall(parts, BuildContainer(1, 1, params.inset), params)

    assert reason is not None and "does not fit" in reason


def test_parts_that_fit_individually_but_not_together_are_caught_by_area():
    params = _quick()
    parts = BuildParts({i: _rectangle(30, 30) for i in range(6)}, params)

    reason = ProvablyTooSmall(parts, BuildContainer(1, 1, params.inset), params)

    assert reason is not None and "mm^2" in reason


def test_bounds_never_reject_a_bin_that_actually_packs():
    """Soundness, which is the property that matters. A bound that wrongly
    rejects inflates every result it touches, and nothing downstream would
    notice - the packer would simply return a bigger bin and look right.
    """
    params = _quick()
    for count, (width, height), grid in [
        (1, (30, 25), (1, 1)),
        (2, (30, 30), (2, 1)),
        (4, (50, 30), (3, 2)),
        (3, (30, 30), (1, 3)),
    ]:
        parts = BuildParts({i: _rectangle(width, height) for i in range(count)}, params)
        container = BuildContainer(grid[0], grid[1], params.inset)

        assert ProvablyTooSmall(parts, container, params) is None, f"{count}x{width}x{height} in {grid}"


# ---------------------------------------------------------------- packing


@pytest.mark.parametrize(
    "count, size, cells",
    [
        (1, (30, 25), 1),  # 30 + 2*1.95 fits a 36.3 interior
        (1, (70, 25), 2),  # 70 + 3.9 needs the 78.3 of a 2-cell run
        (2, (30, 30), 2),  # 2*30 + 3.2 = 63.2 against 74.4 usable
        (4, (24, 20), 3),  # 4*24 + 3*3.2 = 105.6, still inside 116.4
    ],
    ids=["one-small", "one-long", "two-square", "four-small"],
)
def test_synthetic_sets_pack_to_their_known_cell_count(count, size, cells):
    """The M4 optimality criterion, on sets whose best packing is countable
    by hand.

    The arithmetic is spelled out per case because getting it wrong is easy
    and fails in the flattering direction: the four-part case was written
    expecting 4 cells and the packer found 3, which is genuinely better -
    all four sit in one row of a 3x1.
    """
    params = _quick()
    parts = BuildParts({i: _rectangle(*size) for i in range(count)}, params)

    result = Pack(parts, params)

    assert result.layout is not None
    assert result.cells == cells, result.Report()
    assert CheckLayout(result.layout, parts) == []


def test_pack_reports_every_size_it_rejected_and_why():
    """The M4 reporting criterion."""
    params = _quick()
    parts = BuildParts({i: _rectangle(50, 30) for i in range(4)}, params)

    result = Pack(parts, params)

    assert result.layout is not None
    outcomes = {attempt.outcome for attempt in result.attempts}
    assert PACKED in outcomes and TOO_SMALL in outcomes
    assert result.attempts[-1].outcome == PACKED

    report = result.Report()
    assert "1x1" in report and "does not fit" in report
    assert "mm^2" in report, "an area rejection should say how much was needed"


def test_pack_steps_up_rather_than_giving_up_on_a_size_it_cannot_solve():
    """Decision (a): a usable bin beats none. The size that defeated the
    search is recorded so an oversized result stays traceable.
    """
    params = _quick(restarts=1, iterations=30)
    parts = BuildParts({i: _rectangle(30, 25) for i in range(6)}, params)

    result = Pack(parts, params)

    assert result.layout is not None
    assert CheckLayout(result.layout, parts) == []
    if result.skipped:
        assert "a tighter packing may exist" in result.Report()


def test_skipped_is_empty_when_the_first_feasible_size_works():
    params = _quick()
    parts = BuildParts({0: _rectangle(30, 25)}, params)

    result = Pack(parts, params)

    assert result.skipped == []
    assert "tighter packing" not in result.Report()


def test_pack_gives_up_cleanly_when_nothing_fits():
    params = _quick(max_grid=2)
    parts = BuildParts({0: _rectangle(300, 40)}, params)

    result = Pack(parts, params)

    assert result.layout is None
    assert result.cells is None
    assert all(attempt.outcome == TOO_SMALL for attempt in result.attempts)
    assert "no grid size" in result.Report()


def test_pack_refuses_an_empty_set():
    with pytest.raises(ValueError, match="nothing to pack"):
        Pack({}, _quick())


def test_a_failed_size_is_labelled_not_found_rather_than_too_small():
    """The distinction the report exists for: geometry proving a size
    impossible is a different claim from the search running out of luck.
    """
    params = _quick(restarts=1, iterations=20)
    parts = BuildParts({i: _rectangle(30, 25) for i in range(6)}, params)

    result = Pack(parts, params)

    for attempt in result.attempts:
        if attempt.outcome == NOT_FOUND:
            assert "attempts" in attempt.detail
        elif attempt.outcome == TOO_SMALL:
            assert "does not fit" in attempt.detail or "mm^2" in attempt.detail


# --------------------------------------------------------------- fixtures


@pytest.mark.slow
def test_the_three_spoons_pack_into_a_five_by_two():
    """The M4 fixture criterion. Everything smaller is ruled out by the
    extent bound - the big spoon needs a 5-cell run one way and 2 cells the
    other - so this also exercises the bounds doing the real work.
    """
    params = LayoutParameters(patience=25)
    parts = LoadParts(SPOONS, params)

    result = Pack(parts, params)

    assert result.layout is not None
    assert result.layout.grid == (5, 2)
    assert result.cells == 10
    assert CheckLayout(result.layout, parts) == []

    rejected = [a for a in result.attempts if a.outcome == TOO_SMALL]
    assert len(rejected) == 10, result.Report()
    assert all("does not fit" in a.detail for a in rejected)


@pytest.mark.slow
def test_random_part_sets_never_produce_an_overlapping_layout():
    """The M4 validation sweep, and the gate for everything downstream.

    Every layout the packer returns is re-checked against exact polygon
    geometry rather than the rasterized field the solver used, because a
    raster artifact that confirmed itself would surface first as a printed
    bin that does not hold its objects.
    """
    rng = np.random.default_rng(0)
    checked = 0

    for trial in range(120):
        params = _quick(restarts=2, iterations=80, seed=trial)
        count = int(rng.integers(2, 6))
        parts = BuildParts(
            {i: _rectangle(float(rng.uniform(15, 55)), float(rng.uniform(10, 35))) for i in range(count)},
            params,
        )

        result = Pack(parts, params)
        if result.layout is None:
            continue

        checked += 1
        problems = CheckLayout(result.layout, parts, pair_clearance=params.c_pair, wall_clearance=params.c_wall)
        assert problems == [], f"trial {trial}: {problems}\n{result.Report()}"

    assert checked > 100, f"only {checked} trials produced a layout to check"
