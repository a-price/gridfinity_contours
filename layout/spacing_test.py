"""Tests for the spacing pass.

The property under test is not "the gaps are equal" - they cannot be, in
a bin tight enough to be worth packing. It is that the gaps get *closer
together* while the layout stays feasible, and that nothing ever pulls a
pair together on purpose.
"""

import numpy as np
import pytest

from layout.container import BuildContainer
from layout.energy import ComputeEnergy
from layout.loading import BuildParts, LoadParts
from layout.parameters import LayoutParameters
from layout.placement import Layout, Placement, RotatedSize
from layout.solver import SolveFixedGrid
from layout.spacing import Distribute, Gaps, Spread, SpringParameters
from layout.verify import CheckLayout
from conftest import QuickParameters as _quick, Rectangle as _rectangle, SPOONS


def _finite(gaps: dict) -> list[float]:
    """The gaps between parts near enough to be neighbours.

    Pairs beyond the field's reach report an enormous slack; they are not
    neighbours and the springs never act on them, so averaging them in
    would be meaningless.
    """
    return [gap for gap in gaps.values() if gap < 100.0]


def _spread_of(gaps: dict) -> float:
    finite = _finite(gaps)
    return max(finite) - min(finite)


def _three_in_a_row(params: LayoutParameters):
    """Three 20mm squares in a 2x1 bin, unevenly spaced but feasible.

    78.3mm of interior against 60mm of parts leaves too little for every
    spring to reach its rest length, so the row stays mutually blocked
    instead of simply spreading until nothing touches.

    Gaps of 3.9 and 6.0mm: lopsided, both still inside the field's reach
    so both count as neighbours, and both clear of the 3.2mm clearance.
    """
    parts = BuildParts({index: _rectangle(20, 20) for index in range(3)}, params)
    container = BuildContainer(2, 1, params.inset)
    placements = {
        0: Placement(0, np.array([4.2, 8.0])),
        1: Placement(1, np.array([28.1, 8.0])),
        2: Placement(2, np.array([54.1, 8.0])),
    }
    return parts, container, placements


# ------------------------------------------------------------ rest lengths


def test_the_springs_reach_further_than_the_clearances_they_protect():
    params = _quick()

    assert params.spacing_pair > params.c_pair_enforced
    assert params.spacing_wall > params.c_wall


def test_the_field_outreaches_the_springs():
    """A spring whose rest length sat outside the raster would read every
    neighbour as infinitely far and pull on nothing - the pass would
    silently do nothing at all.
    """
    params = _quick()

    assert params.pad > params.spacing_pair


def test_raising_the_margin_grows_the_field_to_match():
    tight, loose = _quick(spacing_margin=1.0), _quick(spacing_margin=5.0)

    assert loose.spacing_pair > tight.spacing_pair
    assert loose.pad > tight.pad


def test_spring_parameters_drive_to_the_rest_length():
    """`c_pair_enforced` adds the raster margin back on, so the override
    has to be set that much low or the springs would overshoot by it.
    """
    params = _quick()

    springs = SpringParameters(params)

    assert springs.c_pair_enforced == pytest.approx(params.spacing_pair)
    assert springs.c_wall == pytest.approx(params.spacing_wall)


def test_spring_parameters_leave_the_rest_of_the_budget_alone():
    params = _quick(seed=7, restarts=11)

    springs = SpringParameters(params)

    assert (springs.seed, springs.restarts) == (7, 11)


# ------------------------------------------------------------------- gaps


def test_gaps_are_reported_as_slack_over_the_clearance():
    # Hand-computed geometry: at a zero offset the pocket is the shape
    # written here, so the numbers below stay the ones a reader can check.
    # What packs pockets for real is the fixtures that take the default.
    params = _quick(pocket_offset=0.0)
    parts = BuildParts({0: _rectangle(20, 20), 1: _rectangle(20, 20)}, params)
    # 4mm apart edge to edge, so `4 - c_pair` of slack. Kept inside the
    # field's reach, or the pair would not register as adjacent at all.
    placements = {0: Placement(0, np.array([0.0, 0.0])), 1: Placement(1, np.array([24.0, 0.0]))}

    gaps = Gaps(parts, placements, params)

    assert gaps[(0, 1)] == pytest.approx(4.0 - params.c_pair, abs=0.1)


def test_gaps_beyond_the_fields_reach_are_not_neighbours():
    params = _quick()
    parts = BuildParts({0: _rectangle(20, 20), 1: _rectangle(20, 20)}, params)
    placements = {0: Placement(0, np.array([0.0, 0.0])), 1: Placement(1, np.array([300.0, 0.0]))}

    gaps = Gaps(parts, placements, params)

    assert gaps[(0, 1)] > 100.0, "a part across the bin should not register as adjacent"


def test_a_gap_is_measured_the_tighter_way_round():
    """One part's boundary can be near the other's field without the
    reverse holding, so a one-directional measurement can miss a contact
    the energy sees.
    """
    # Hand-computed geometry: at a zero offset the pocket is the shape
    # written here, so the numbers below stay the ones a reader can check.
    # What packs pockets for real is the fixtures that take the default.
    params = _quick(pocket_offset=0.0)
    parts = BuildParts({0: _rectangle(60, 10), 1: _rectangle(10, 10)}, params)
    placements = {0: Placement(0, np.array([0.0, 0.0])), 1: Placement(1, np.array([25.0, 14.0]))}

    gaps = Gaps(parts, placements, params)

    assert gaps[(0, 1)] == pytest.approx(4.0 - params.c_pair, abs=0.2)


# ---------------------------------------------------------------- spreading


def test_spreading_evens_out_a_lopsided_arrangement():
    """Three parts in a row, one jammed against its neighbour and one with
    room: the cramped gap should open at the roomy one's expense.

    The bin is deliberately too tight for every spring to reach its rest
    length, so the parts stay mutually blocked - which is the case that
    matters, and the one where equal compression does the work.
    """
    params = _quick()
    parts, container, placements = _three_in_a_row(params)
    assert ComputeEnergy(parts, placements, container, params).feasible, "the fixture must start feasible"

    before = Gaps(parts, placements, params)
    after = Gaps(parts, Spread(parts, placements, container, params), params)

    assert len(_finite(before)) == 2, "both pairs must be near enough to count as neighbours"
    assert _spread_of(after) < _spread_of(before)
    assert min(_finite(after)) > min(_finite(before)), "the tightest gap should have opened"


def test_blocked_springs_settle_at_equal_compression():
    """The mechanism, stated directly: identical springs that cannot reach
    their rest length balance where their compressions match, which is why
    the gaps come out even without anything measuring evenness.
    """
    params = _quick()
    parts, container, placements = _three_in_a_row(params)

    after = _finite(Gaps(parts, Spread(parts, placements, container, params), params))

    assert after[0] == pytest.approx(after[1], abs=0.05)


def test_spreading_never_returns_an_infeasible_arrangement():
    params = _quick()
    parts, container, placements = _three_in_a_row(params)

    spread = Spread(parts, placements, container, params)

    assert ComputeEnergy(parts, spread, container, params).feasible
    assert CheckLayout(Layout(grid=(2, 1), placements=spread, inset=params.inset), parts) == []


def test_spreading_a_bin_with_nowhere_to_go_leaves_it_alone():
    """Every candidate it tries is infeasible or no better, so the input
    comes back - this pass can only improve a layout or do nothing.
    """
    params = _quick()
    parts = BuildParts({0: _rectangle(30, 30)}, params)
    container = BuildContainer(1, 1, params.inset)
    placements = {0: Placement(0, np.array([params.c_wall, params.c_wall]))}

    spread = Spread(parts, placements, container, params)

    assert ComputeEnergy(parts, spread, container, params).feasible


def test_spreading_can_be_turned_off():
    params = _quick(spacing_iterations=0)
    parts = BuildParts({index: _rectangle(20, 20) for index in range(2)}, params)
    container = BuildContainer(3, 1, params.inset)
    placements = {0: Placement(0, np.array([2.0, 8.0])), 1: Placement(1, np.array([26.0, 8.0]))}

    spread = Spread(parts, placements, container, params)

    assert spread == placements


def test_spreading_leaves_orientations_alone():
    """Orientation is discrete (D1), so no force acts on it - the pass
    moves parts and must not silently turn them.
    """
    params = _quick()
    parts = BuildParts({index: _rectangle(30, 15) for index in range(2)}, params)
    container = BuildContainer(3, 1, params.inset)
    placements = {0: Placement(0, np.array([2.0, 5.0]), 1), 1: Placement(1, np.array([40.0, 5.0]), 3)}

    spread = Spread(parts, placements, container, params)

    assert {part_id: p.orientation for part_id, p in spread.items()} == {0: 1, 1: 3}


# ------------------------------------------------------------- the fixtures


@pytest.mark.slow
def test_the_spoons_come_out_more_evenly_spaced():
    """The motivating case. Before this pass one pair sat 0.44mm above the
    clearance while another had 4.79mm of room.
    """
    params = LayoutParameters()
    parts = LoadParts(SPOONS, params)
    container = BuildContainer(5, 2, params.inset)

    from layout.solver import SolveFixedGrid

    unbalanced = SolveFixedGrid(parts, 5, 2, LayoutParameters(spacing_iterations=0))
    assert unbalanced is not None

    before = Gaps(parts, unbalanced.placements, params)
    after = Gaps(parts, Spread(parts, unbalanced.placements, container, params), params)

    assert _spread_of(after) < _spread_of(before)
    assert min(_finite(after)) > min(_finite(before)), "the tightest gap should have opened"


def _box(width: float, height: float) -> np.ndarray:
    return np.array([[0.0, 0.0], [width, 0.0], [width, height], [0.0, height]], dtype=np.float64)


def _placed(parts, positions: dict[int, tuple[float, float]]) -> dict[int, Placement]:
    return {part_id: Placement(part_id, np.array(xy, dtype=np.float64)) for part_id, xy in positions.items()}


def test_a_lone_part_ends_up_centered():
    """The case that prompted this. Both energy terms are one-sided, so
    beyond its clearance a part feels nothing at all and stops wherever
    bottom-left fill dropped it - measured at 4.1mm from one wall and
    76.2mm from the opposite one.
    """
    params = _quick()
    parts = BuildParts({0: _box(40.0, 18.0)}, params)
    container = BuildContainer(3, 2, params.inset)

    settled = Distribute(parts, _placed(parts, {0: (4.1, 4.79)}), container, params)

    position = settled[0].position
    assert position[0] == pytest.approx((container.width - parts[0].size[0]) / 2.0, abs=0.01)
    assert position[1] == pytest.approx((container.height - parts[0].size[1]) / 2.0, abs=0.01)


def test_centering_is_all_that_happens_to_a_single_part():
    """Nothing to inflate away from: a lone part is its own center, so the
    scale step must leave it exactly where centering put it. Scaling its
    *corner* instead drove it into the wall.
    """
    params = _quick()
    parts = BuildParts({0: _box(40.0, 18.0)}, params)
    container = BuildContainer(5, 3, params.inset)

    settled = Distribute(parts, _placed(parts, {0: (4.1, 4.79)}), container, params)

    assert settled[0].position[0] > params.c_wall * 2, "the part was pushed against a wall"


def test_parts_spread_into_the_room_they_have():
    params = _quick()
    parts = BuildParts({0: _box(30.0, 16.0), 1: _box(30.0, 16.0)}, params)
    container = BuildContainer(5, 2, params.inset)
    before = _placed(parts, {0: (5.0, 5.0), 1: (5.0, 30.0)})

    after = Distribute(parts, before, container, params)

    gap_before = abs(before[1].position[1] - before[0].position[1])
    gap_after = abs(after[1].position[1] - after[0].position[1])
    assert gap_after > gap_before


def test_distributing_never_breaks_a_clearance():
    """Everything here is checked against the true clearances before it is
    kept, the same discipline `Spread` follows - concave parts make
    "further apart" a claim rather than a proof.
    """
    # Hand-computed geometry: at a zero offset the pocket is the shape
    # written here, so the numbers below stay the ones a reader can check.
    # What packs pockets for real is the fixtures that take the default.
    params = _quick(pocket_offset=0.0)
    parts = LoadParts(SPOONS, params)
    container = BuildContainer(5, 2, params.inset)
    layout = SolveFixedGrid(parts, 5, 2, params)
    assert layout is not None

    settled = Distribute(parts, layout.placements, container, params)

    assert ComputeEnergy(parts, settled, container, params).feasible
    assert CheckLayout(Layout(grid=(5, 2), placements=settled, inset=params.inset), parts) == []


def _wall_gaps(parts, placements, container) -> dict[int, tuple[float, float, float, float]]:
    """Each part's distance to the left, bottom, right and top walls."""
    gaps = {}
    for part_id, placement in placements.items():
        low = np.asarray(placement.position, dtype=np.float64)
        high = low + RotatedSize(parts[part_id].size, placement.orientation)
        gaps[part_id] = (low[0], low[1], container.width - high[0], container.height - high[1])
    return gaps


def test_inflating_leaves_the_arrangement_centered():
    """The inflation scales part *centers* while the centering step centers
    the arrangement's *bounding box*, so scaling undoes the centering it
    started from: whichever part is furthest out reaches its wall first and
    stops the scale, leaving every other part with whatever margin it
    happened to get. Measured before this was fixed: 7.0mm on one side
    against 2.2mm on the other, with one part jammed into a corner.

    Re-centering afterwards costs one translation, which cannot change any
    pair distance at all.
    """
    # Hand-computed geometry: at a zero offset the pocket is the shape
    # written here, so the numbers below stay the ones a reader can check.
    # What packs pockets for real is the fixtures that take the default.
    params = _quick(pocket_offset=0.0)
    contours = {0: _box(40.0, 18.0), 1: _box(30.0, 15.0), 2: _box(25.0, 12.0)}
    parts = BuildParts(contours, params)
    container = BuildContainer(4, 3, params.inset)
    before = _placed(parts, {0: (2.2, 2.2), 1: (45.9, 1.9), 2: (79.6, 1.9)})

    after = Distribute(parts, before, container, params)

    gaps = _wall_gaps(parts, after, container)
    left = min(gap[0] for gap in gaps.values())
    right = min(gap[2] for gap in gaps.values())
    bottom = min(gap[1] for gap in gaps.values())
    top = min(gap[3] for gap in gaps.values())
    assert left == pytest.approx(right, abs=0.1), f"lopsided across x: {left:.2f} vs {right:.2f}"
    assert bottom == pytest.approx(top, abs=0.1), f"lopsided across y: {bottom:.2f} vs {top:.2f}"


def test_inflating_stops_at_the_springs_wall_target():
    """The bare clearance is the legal minimum, so inflating to it spends
    every millimetre of a roomy bin's slack on the gaps between parts and
    leaves none at the wall. `spacing_wall` is what `Spread` already drives
    a wall contact to when it has the room, so stopping there makes the two
    passes agree rather than the later one overriding the earlier.
    """
    params = _quick()
    contours = {index: _box(28.0 - 2 * index, 14.0) for index in range(5)}
    parts = BuildParts(contours, params)
    container = BuildContainer(4, 3, params.inset)
    before = _placed(parts, {index: (4.0 + 32.0 * index, 4.0) for index in range(5)})

    after = Distribute(parts, before, container, params)

    closest = min(min(gap) for gap in _wall_gaps(parts, after, container).values())
    assert closest >= params.spacing_wall - 0.05
    assert closest > params.c_wall, "the legal minimum is not the target"


def test_a_tight_bin_is_not_pulled_off_its_walls():
    """The wall target caps how far the inflation may *push*; it must not
    become a force in its own right. A part the spring pass legitimately
    left closer than `spacing_wall` has nowhere to go, and moving it would
    be this pass overriding a tighter constraint than its own.
    """
    params = _quick()
    parts = BuildParts({0: _box(70.0, 30.0)}, params)
    container = BuildContainer(2, 1, params.inset)
    before = _placed(parts, {0: (params.c_wall + 0.1, params.c_wall + 0.1)})

    after = Distribute(parts, before, container, params)

    assert ComputeEnergy(parts, after, container, params).feasible
    assert CheckLayout(Layout(grid=(2, 1), placements=after, inset=params.inset), parts) == []


def test_a_full_bin_is_left_alone():
    """With no room to spare there is nothing to distribute, and the pass
    must not jostle a layout that is already as good as it gets.
    """
    params = _quick()
    parts = BuildParts({0: _box(70.0, 30.0)}, params)
    container = BuildContainer(2, 1, params.inset)
    before = _placed(parts, {0: (params.c_wall + 0.1, params.c_wall + 0.1)})

    after = Distribute(parts, before, container, params)

    assert ComputeEnergy(parts, after, container, params).feasible
