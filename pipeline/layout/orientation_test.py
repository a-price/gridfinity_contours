"""Tests for the starting-orientation heuristic.

The property worth pinning here is the one that makes the heuristic
necessary at all: a single part carries no signal about which quarter turn
to use, because 180 degrees reverses its profile without changing it. Any
score that could be computed per part would be blind to the choice, so
these tests check that the score is genuinely about parts *together*.
"""

import numpy as np
import pytest

from pipeline.layout.container import BuildContainer
from pipeline.layout.loading import BuildParts, LoadParts
from pipeline.layout.orientation import Assignment, RankedAssignments, StackedWidth, WidthProfile
from pipeline.layout.parameters import LayoutParameters
from pipeline.layout.placement import RotatedSize
from pipeline.layout.solver import FittingOrientations
from conftest import Rectangle as _rectangle, SPOONS


def _wedge(length: float, thin: float, thick: float) -> np.ndarray:
    """A part narrow at one end and wide at the other - the shape the whole
    heuristic is about.
    """
    return np.array(
        [[0.0, 0.0], [length, 0.0], [length, thick], [0.0, thin]],
        dtype=np.float64,
    )


def _parts(shapes: dict[int, np.ndarray], params: LayoutParameters):
    return BuildParts(shapes, params)


def test_a_profile_spans_the_part_at_every_quarter_turn():
    """The axis convention has to match `RotatedSize`, or a profile taken
    at one orientation would describe the part at another.

    A rectangle rather than a wedge, so the claim is exact: a slanted shape
    has no single column spanning its whole bounding box, and the tolerance
    that would need would be loose enough to hide a swapped axis.
    """
    params = LayoutParameters()
    part = _parts({0: _rectangle(60.0, 20.0)}, params)[0]

    for orientation in range(4):
        profile = WidthProfile(part, orientation)
        expected = RotatedSize(part.size, orientation)
        cell = part.resolution

        assert abs(len(profile) * cell - expected[0]) <= 2 * cell, f"length wrong at {orientation}"
        assert abs(profile.max() - expected[1]) <= 2 * cell, f"width wrong at {orientation}"


def test_half_a_turn_reverses_a_profile_without_changing_it():
    """The fact that forces the score to be about pairs of parts.

    Extent, area and widest point are all identical at 0 and 180, so no
    per-part quantity can tell the two apart - and choosing between them is
    exactly what decides whether two long parts nest.
    """
    params = LayoutParameters()
    part = _parts({0: _wedge(60.0, 4.0, 20.0)}, params)[0]

    forward = WidthProfile(part, 0)
    backward = WidthProfile(part, 2)

    assert len(forward) == len(backward)
    assert np.allclose(forward, backward[::-1], atol=1e-9)
    assert forward.max() == pytest.approx(backward.max())


def test_stacking_rewards_a_wide_end_against_a_narrow_one():
    """Two wedges nose to tail stack narrower than two nose to nose, which
    is the entire signal the heuristic runs on.
    """
    params = LayoutParameters()
    part = _parts({0: _wedge(60.0, 4.0, 20.0)}, params)[0]
    columns = int(round(120.0 / params.resolution))

    forward, backward = WidthProfile(part, 0), WidthProfile(part, 2)

    opposed = StackedWidth([forward, backward], columns)
    aligned = StackedWidth([forward, forward], columns)

    assert opposed < aligned


def test_stacking_a_part_too_long_for_the_run_is_infinite():
    """Not an error: the caller is ranking candidates, and an orientation
    that cannot lie in the bin at all should simply rank last.
    """
    params = LayoutParameters()
    part = _parts({0: _wedge(60.0, 4.0, 20.0)}, params)[0]

    assert StackedWidth([WidthProfile(part, 0)], columns=10) == np.inf


@pytest.mark.slow
def test_the_spoons_get_opposed_bowls():
    """The real case. The two long spoons must overlap along the bin, so
    the only way both fit is one's bowl beside the other's handle - and the
    assignment has to work that out before the solver runs, since nothing
    in the relaxation can turn a part.
    """
    params = LayoutParameters()
    parts = LoadParts(SPOONS, params)
    container = BuildContainer(5, 2, params.inset)
    fitting = {part_id: FittingOrientations(part, container, params) for part_id, part in parts.items()}

    chosen = Assignment(parts, fitting, container, params)

    biggest, second = sorted(parts, key=lambda i: -parts[i].area)[:2]
    assert chosen[biggest] != chosen[second], f"the two large spoons were both left at {chosen[biggest]}"
    assert all(chosen[part_id] in fitting[part_id] for part_id in parts)


def test_an_assignment_only_offers_orientations_that_fit():
    params = LayoutParameters()
    parts = _parts({0: _wedge(60.0, 4.0, 20.0), 1: _wedge(50.0, 4.0, 18.0)}, params)
    container = BuildContainer(3, 2, params.inset)
    fitting = {part_id: FittingOrientations(part, container, params) for part_id, part in parts.items()}

    chosen = Assignment(parts, fitting, container, params)

    assert set(chosen) == set(parts)
    assert all(chosen[part_id] in fitting[part_id] for part_id in parts)


def _ranked(count: int, limit: int = 64):
    params = LayoutParameters()
    parts = _parts({0: _wedge(60.0, 4.0, 20.0), 1: _wedge(55.0, 4.0, 18.0)}, params)
    container = BuildContainer(3, 2, params.inset)
    fitting = {part_id: FittingOrientations(part, container, params) for part_id, part in parts.items()}
    return RankedAssignments(parts, fitting, container, params, count, limit), parts, fitting


def test_the_ranking_comes_back_best_first():
    ranked, parts, fitting = _ranked(count=8)
    container = BuildContainer(3, 2, LayoutParameters().inset)
    params = LayoutParameters()
    columns = int(round((container.width - 2 * params.c_wall) / params.resolution))

    scores = [
        StackedWidth([WidthProfile(parts[i], o) for i, o in assignment.items()], columns) for assignment in ranked
    ]

    assert scores == sorted(scores)


def test_the_ranking_is_capped_at_what_the_search_will_use():
    """No point scoring assignments the restart budget can never reach."""
    ranked, _, _ = _ranked(count=3)

    assert len(ranked) == 3


def test_every_assignment_covers_every_part_with_a_fitting_orientation():
    ranked, parts, fitting = _ranked(count=8)

    for assignment in ranked:
        assert set(assignment) == set(parts)
        assert all(assignment[part_id] in fitting[part_id] for part_id in parts)


def test_sampling_takes_over_when_there_are_too_many_permutations():
    """Above the limit the candidates are sampled rather than enumerated,
    so the cost stays flat however many parts there are - and the result is
    still a ranking, which beats an unranked draw.
    """
    ranked, parts, fitting = _ranked(count=4, limit=1)

    assert len(ranked) >= 1
    for assignment in ranked:
        assert all(assignment[part_id] in fitting[part_id] for part_id in parts)


def test_the_ranking_reproduces():
    """A bin has to pack the same way twice, so the sampled path is seeded
    like everything else.
    """
    first, _, _ = _ranked(count=4, limit=1)
    second, _, _ = _ranked(count=4, limit=1)

    assert first == second


def test_a_ranking_of_nothing_is_refused():
    with pytest.raises(ValueError, match="at least one assignment"):
        _ranked(count=0)
