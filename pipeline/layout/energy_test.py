"""Tests for the energy and force model (M2).

The finite-difference tests are the load-bearing ones. Everything the
solver does rests on the returned forces actually being the negative
gradient of the returned energy; if they drift apart, descent stops being
descent and the failure shows up as a solver that mysteriously stalls
rather than as anything obviously wrong here.
"""

import numpy as np
import pytest

from pipeline.layout.container import DIVIDER_WIDTH_MM, MIN_WALL_MM, BuildContainer, Container
from pipeline.layout.energy import ComputeEnergy
from pipeline.layout.loading import BuildParts, LoadParts
from pipeline.layout.parameters import LayoutParameters
from pipeline.layout.part import BuildPart
from pipeline.layout.placement import Placement
from pipeline.layout.verify import MinimumSeparation, PolygonsOverlap


def _rectangle(width: float, height: float, x: float = 0.0, y: float = 0.0) -> np.ndarray:
    return np.array([[x, y], [x + width, y], [x + width, y + height], [x, y + height]], dtype=np.float64)


def _square_pair(separation: float, params: LayoutParameters):
    """Two 10mm squares whose facing edges sit `separation` apart, far from
    any wall so only the pair term is active.
    """
    parts = BuildParts({0: _rectangle(10, 10), 1: _rectangle(10, 10)}, params)
    placements = {
        0: Placement(0, np.array([40.0, 40.0])),
        1: Placement(1, np.array([40.0 + 10.0 + separation, 40.0])),
    }
    return parts, placements


def _roomy_container() -> Container:
    return BuildContainer(4, 4)


def _numeric_forces(parts, placements, container, params, step=1e-5):
    """Central-difference gradient of ComputeEnergy with respect to every
    part's position, negated to give a force.
    """
    forces = {}
    for part_id in placements:
        gradient = np.zeros(2)
        for axis in range(2):
            offset = np.zeros(2)
            offset[axis] = step

            def moved(delta):
                shifted = dict(placements)
                shifted[part_id] = Placement(
                    part_id, placements[part_id].position + delta, placements[part_id].orientation
                )
                return ComputeEnergy(parts, shifted, container, params).energy

            gradient[axis] = (moved(offset) - moved(-offset)) / (2 * step)
        forces[part_id] = -gradient
    return forces


# ------------------------------------------------------------- parameters


def test_clearances_derive_from_the_pocket_offset():
    params = LayoutParameters(pocket_offset=1.0)

    # Two pockets each cut 1mm oversize need 2mm plus a printable divider.
    assert params.c_pair == pytest.approx(2.0 + DIVIDER_WIDTH_MM)
    assert params.c_wall == pytest.approx(1.0 + MIN_WALL_MM)


def test_clearances_track_a_changed_pocket_offset():
    loose = LayoutParameters(pocket_offset=2.0)

    assert loose.c_pair == pytest.approx(4.0 + DIVIDER_WIDTH_MM)
    assert loose.c_wall == pytest.approx(2.0 + MIN_WALL_MM)


def test_explicit_clearances_override_the_derivation():
    params = LayoutParameters(pocket_offset=1.0, pair_clearance=8.0, wall_clearance=4.0)

    assert params.c_pair == 8.0
    assert params.c_wall == 4.0


def test_field_padding_covers_the_pair_clearance():
    """A field narrower than the clearance it enforces would let parts pass
    through each other at exactly the distance that matters.
    """
    for offset in (0.5, 1.0, 3.0):
        params = LayoutParameters(pocket_offset=offset)
        assert params.pad > params.c_pair


def test_energy_refuses_parts_whose_fields_are_too_small():
    params = LayoutParameters(pocket_offset=1.0)
    # Built with a field far too narrow for the clearance in force.
    parts = {0: BuildPart(_rectangle(10, 10), pad=0.5), 1: BuildPart(_rectangle(10, 10), pad=0.5)}
    placements = {0: Placement(0, np.array([20.0, 20.0])), 1: Placement(1, np.array([60.0, 20.0]))}

    with pytest.raises(ValueError, match="pass through each other"):
        ComputeEnergy(parts, placements, _roomy_container(), params)


# ------------------------------------------------------------- pair energy


def test_energy_falls_to_zero_as_squares_separate():
    """The M2 separation criterion: positive and decreasing while the parts
    are closer than the clearance, exactly zero once they are not.
    """
    params = LayoutParameters(pocket_offset=1.0)
    energies = []
    for separation in [0.0, 1.0, 2.0, 3.0]:
        parts, placements = _square_pair(separation, params)
        energies.append(ComputeEnergy(parts, placements, _roomy_container(), params).energy)

    assert all(energy > 0 for energy in energies)
    assert energies == sorted(energies, reverse=True), "energy should fall as parts separate"

    parts, placements = _square_pair(params.c_pair + 0.5, params)
    assert ComputeEnergy(parts, placements, _roomy_container(), params).energy == 0.0


def test_separated_parts_feel_no_force():
    params = LayoutParameters(pocket_offset=1.0)
    parts, placements = _square_pair(params.c_pair + 1.0, params)

    result = ComputeEnergy(parts, placements, _roomy_container(), params)

    assert result.feasible
    for force in result.forces.values():
        assert force == pytest.approx([0.0, 0.0])


def test_forces_push_along_the_separating_axis():
    params = LayoutParameters(pocket_offset=1.0)
    parts, placements = _square_pair(0.5, params)

    forces = ComputeEnergy(parts, placements, _roomy_container(), params).forces

    # Part 0 sits to the left of part 1, so it is pushed left and part 1 right.
    assert forces[0][0] < 0
    assert forces[1][0] > 0
    # The transverse component is not exactly zero - the boundary samples
    # do not land symmetrically on the raster - but it is negligible beside
    # the separating push.
    for force in forces.values():
        assert abs(force[1]) < 1e-4 * abs(force[0])


def test_pair_forces_are_equal_and_opposite():
    params = LayoutParameters(pocket_offset=1.0)
    parts, placements = _square_pair(1.0, params)

    forces = ComputeEnergy(parts, placements, _roomy_container(), params).forces

    assert forces[0] == pytest.approx(-forces[1])


def test_deeper_overlap_costs_more():
    params = LayoutParameters(pocket_offset=1.0)
    energies = []
    for separation in [2.0, 1.0, 0.0, -1.0, -2.0]:
        parts, placements = _square_pair(separation, params)
        energies.append(ComputeEnergy(parts, placements, _roomy_container(), params).energy)

    assert energies == sorted(energies), "energy should rise as parts overlap further"


def test_push_stays_separating_through_shallow_overlap():
    """Force direction is what the solver relies on; magnitude is not
    monotonic in depth, because samples deep inside another part start
    finding their nearest exit sideways and cancelling each other.
    """
    params = LayoutParameters(pocket_offset=1.0)

    for separation in [2.0, 1.0, 0.0, -1.0, -2.0, -4.0]:
        parts, placements = _square_pair(separation, params)
        forces = ComputeEnergy(parts, placements, _roomy_container(), params).forces

        assert forces[0][0] < 0, f"separation {separation} should still push part 0 left"
        assert forces[1][0] > 0, f"separation {separation} should still push part 1 right"


def test_deep_overlap_past_the_medial_axis_reverses_the_push():
    """A known and load-bearing limitation of distance-field penalties,
    pinned down here so M3 designs around it instead of rediscovering it as
    a solver that mysteriously converges on stacked parts.

    A distance field points toward the nearest way out. Once a sample is
    more than halfway through another part, that is the *far* side, so the
    force flips from separating to attracting and the energy starts falling
    as the parts merge - making coincidence a spurious minimum.

    Two 10mm squares: correct to roughly 50% overlap, reversed past it.
    """
    params = LayoutParameters(pocket_offset=1.0)

    def push(separation):
        parts, placements = _square_pair(separation, params)
        return ComputeEnergy(parts, placements, _roomy_container(), params).forces[0][0]

    assert push(-4.0) < 0, "40% overlap should still separate"
    assert push(-6.0) > 0, "60% overlap reverses - the documented failure"

    # And energy falls off toward full overlap rather than continuing to rise.
    parts, deep = _square_pair(-6.0, params)
    parts, deeper = _square_pair(-9.5, params)
    assert (
        ComputeEnergy(parts, deeper, _roomy_container(), params).energy
        < ComputeEnergy(parts, deep, _roomy_container(), params).energy
    )


def test_containment_distinguishes_crossing_from_swallowing():
    """Crossing and containment need telling apart, because they call for
    opposite responses: overlapping parts are the normal state partway
    through a relaxation and resolve themselves, whereas a swallowed part
    never recovers and the attempt should be abandoned.
    """
    params = LayoutParameters(pocket_offset=1.0)
    container = _roomy_container()
    # A big square and a small one that fits entirely within it.
    parts = BuildParts({0: _rectangle(30, 30), 1: _rectangle(6, 6)}, params)

    apart = ComputeEnergy(
        parts, {0: Placement(0, np.array([20.0, 20.0])), 1: Placement(1, np.array([60.0, 60.0]))}, container, params
    )
    crossing = ComputeEnergy(
        parts, {0: Placement(0, np.array([20.0, 20.0])), 1: Placement(1, np.array([48.0, 35.0]))}, container, params
    )
    swallowed = ComputeEnergy(
        parts, {0: Placement(0, np.array([20.0, 20.0])), 1: Placement(1, np.array([32.0, 32.0]))}, container, params
    )

    assert apart.containment == 0.0 and not apart.engulfed
    assert 0.0 < crossing.containment < 1.0, "a crossing pair is partly, not wholly, inside"
    assert swallowed.containment == 1.0 and swallowed.engulfed


def test_containment_is_scale_free():
    """The reason containment is thresholded rather than penetration depth:
    the same geometric situation at two sizes gives the same number.
    """
    params = LayoutParameters(pocket_offset=1.0)
    container = BuildContainer(6, 6)

    for scale in (1.0, 3.0):
        parts = BuildParts({0: _rectangle(30 * scale, 30 * scale), 1: _rectangle(5 * scale, 5 * scale)}, params)
        inside = ComputeEnergy(
            parts,
            {
                0: Placement(0, np.array([10.0, 10.0])),
                1: Placement(1, np.array([10.0 + 12.0 * scale, 10.0 + 12.0 * scale])),
            },
            container,
            params,
        )

        assert inside.engulfed, f"scale {scale}"


def test_zero_energy_guarantees_the_true_clearance_not_just_the_measured_one():
    """The stronger form of the guarantee, and the one M4's validation
    sweep caught being false.

    The solver reads separation off a rasterized field, which comes back
    short by up to the discretization error - so stopping at a *measured*
    c_pair left parts 3.157mm apart under a 3.200mm clearance. The energy
    therefore drives to `c_pair_enforced`, one raster cell further, and
    what exact geometry reports is what has to clear c_pair.
    """
    params = LayoutParameters(pocket_offset=1.0)
    container = BuildContainer(4, 4)
    parts = BuildParts({0: _rectangle(30, 18), 1: _rectangle(22, 26)}, params)

    rng = np.random.default_rng(11)
    checked = 0
    for _ in range(400):
        placements = {
            part_id: Placement(part_id, rng.uniform(5.0, 100.0, size=2), orientation=int(rng.integers(4)))
            for part_id in parts
        }
        if not ComputeEnergy(parts, placements, container, params).feasible:
            continue

        checked += 1
        a, b = (placements[i].ToWorld(parts[i]) for i in sorted(parts))
        assert MinimumSeparation(a, b) >= params.c_pair, "zero energy must mean the real clearance is met"

    assert checked > 20, "expected a decent number of feasible arrangements to test"


def test_zero_energy_guarantees_no_polygons_overlap():
    """Why the solver's inner loop needs no exact intersection test: energy
    zero means every boundary sample is at least c_pair from every other
    part, which is strictly stronger than not overlapping. The exact check
    is then only worth running once, on a result.

    Checked here on deliberately thin, crossing-prone shapes, since a
    sliver narrower than the sample spacing is the one way a boundary could
    slip through unnoticed.
    """
    params = LayoutParameters(pocket_offset=1.0)
    container = BuildContainer(4, 4)
    parts = BuildParts({0: _rectangle(60, 1.5), 1: _rectangle(60, 1.5)}, params)

    rng = np.random.default_rng(5)
    checked = 0
    for _ in range(200):
        placements = {
            part_id: Placement(part_id, rng.uniform(5.0, 100.0, size=2), orientation=int(rng.integers(4)))
            for part_id in parts
        }
        if not ComputeEnergy(parts, placements, container, params).feasible:
            continue

        checked += 1
        polygons = [placements[part_id].ToWorld(parts[part_id]) for part_id in parts]
        assert not PolygonsOverlap(polygons[0], polygons[1])

    assert checked > 10, "expected some feasible arrangements to actually test"


def test_deepest_penetration_reports_how_far_things_have_gone_wrong():
    """The signal M3 uses to notice it has wandered into the regime above
    and restart, rather than trusting a force that may be pointing the
    wrong way.
    """
    params = LayoutParameters(pocket_offset=1.0)

    clear_parts, clear = _square_pair(params.c_pair + 1.0, params)
    touching_parts, touching = _square_pair(0.0, params)
    deep_parts, deep = _square_pair(-5.0, params)

    assert ComputeEnergy(clear_parts, clear, _roomy_container(), params).deepest_penetration == 0.0
    assert ComputeEnergy(touching_parts, touching, _roomy_container(), params).deepest_penetration == pytest.approx(
        0.0, abs=0.1
    )
    assert ComputeEnergy(deep_parts, deep, _roomy_container(), params).deepest_penetration == pytest.approx(
        5.0, abs=0.2
    )


# ------------------------------------------------------------- wall energy


def test_a_part_straddling_the_wall_is_pushed_inward():
    """The M2 wall criterion."""
    params = LayoutParameters(pocket_offset=1.0)
    container = BuildContainer(2, 2)
    parts = BuildParts({0: _rectangle(20, 10)}, params)
    # Hanging off the left edge.
    placements = {0: Placement(0, np.array([-5.0, 20.0]))}

    result = ComputeEnergy(parts, placements, container, params)

    assert result.energy > 0
    assert result.forces[0][0] > 0, "should be pushed back to the right, into the bin"


def test_a_part_clear_of_every_wall_feels_nothing():
    params = LayoutParameters(pocket_offset=1.0)
    container = BuildContainer(3, 3)
    parts = BuildParts({0: _rectangle(20, 10)}, params)
    placements = {0: Placement(0, np.array([30.0, 30.0]))}

    result = ComputeEnergy(parts, placements, container, params)

    assert result.energy == 0.0
    assert result.forces[0] == pytest.approx([0.0, 0.0])


def test_wall_energy_respects_the_clearance_not_just_the_boundary():
    """A part fully inside the bin but closer to the wall than c_wall is
    still a violation - the divider has to be printable.
    """
    params = LayoutParameters(pocket_offset=1.0)
    container = BuildContainer(3, 3)
    parts = BuildParts({0: _rectangle(20, 10)}, params)

    snug = ComputeEnergy(parts, {0: Placement(0, np.array([0.5, 30.0]))}, container, params)
    clear = ComputeEnergy(parts, {0: Placement(0, np.array([10.0, 30.0]))}, container, params)

    assert snug.energy > 0
    assert clear.energy == 0.0


def test_a_part_is_pushed_out_of_a_corner_diagonally():
    params = LayoutParameters(pocket_offset=1.0)
    container = BuildContainer(2, 2)
    parts = BuildParts({0: _rectangle(10, 10)}, params)
    placements = {0: Placement(0, np.array([-3.0, -3.0]))}

    force = ComputeEnergy(parts, placements, container, params).forces[0]

    assert force[0] > 0 and force[1] > 0


# --------------------------------------------------- gradient consistency


def test_forces_match_finite_differences_for_colliding_parts():
    """The M2 gradient criterion, on the pair term."""
    params = LayoutParameters(pocket_offset=1.0)
    container = _roomy_container()
    parts, placements = _square_pair(0.8, params)

    analytic = ComputeEnergy(parts, placements, container, params).forces
    numeric = _numeric_forces(parts, placements, container, params)

    for part_id, force in analytic.items():
        assert force == pytest.approx(numeric[part_id], abs=1e-3, rel=1e-3), f"part {part_id}"


def test_forces_match_finite_differences_against_the_wall():
    params = LayoutParameters(pocket_offset=1.0)
    container = BuildContainer(2, 2)
    parts = BuildParts({0: _rectangle(20, 10)}, params)
    placements = {0: Placement(0, np.array([-2.0, 15.0]))}

    analytic = ComputeEnergy(parts, placements, container, params).forces
    numeric = _numeric_forces(parts, placements, container, params)

    assert analytic[0] == pytest.approx(numeric[0], abs=1e-3, rel=1e-3)


@pytest.mark.parametrize("orientation", [0, 1, 2, 3])
def test_forces_match_finite_differences_at_every_orientation(orientation):
    """Rotated parts are where a frame error hides: the field is queried in
    the target's frame and the resulting direction has to come back out
    into the bin's. A missing or transposed rotation still produces
    plausible-looking forces, but not ones that match the energy.
    """
    params = LayoutParameters(pocket_offset=1.0)
    container = _roomy_container()
    parts = BuildParts({0: _rectangle(20, 8), 1: _rectangle(16, 6)}, params)
    placements = {
        0: Placement(0, np.array([30.0, 40.0]), orientation=orientation),
        1: Placement(1, np.array([38.0, 44.0]), orientation=(orientation + 1) % 4),
    }

    result = ComputeEnergy(parts, placements, container, params)
    numeric = _numeric_forces(parts, placements, container, params)

    assert result.energy > 0, "these placements should overlap"
    for part_id, force in result.forces.items():
        assert force == pytest.approx(numeric[part_id], abs=1e-3, rel=1e-3), f"part {part_id}"


def test_forces_match_finite_differences_with_several_parts_interacting():
    params = LayoutParameters(pocket_offset=1.0)
    container = BuildContainer(3, 3)
    parts = BuildParts({0: _rectangle(20, 10), 1: _rectangle(18, 9), 2: _rectangle(14, 12)}, params)
    placements = {
        0: Placement(0, np.array([8.0, 8.0])),
        1: Placement(1, np.array([16.0, 14.0])),
        2: Placement(2, np.array([24.0, 6.0]), orientation=1),
    }

    result = ComputeEnergy(parts, placements, container, params)
    numeric = _numeric_forces(parts, placements, container, params)

    assert result.energy > 0
    for part_id, force in result.forces.items():
        assert force == pytest.approx(numeric[part_id], abs=1e-3, rel=1e-3), f"part {part_id}"


def test_energy_is_zero_exactly_when_the_arrangement_is_feasible():
    """E = 0 is the feasibility predicate, so nothing else has to agree with
    it - there is no second collision pass to drift out of sync.
    """
    params = LayoutParameters(pocket_offset=1.0)
    container = BuildContainer(5, 2)
    parts = BuildParts({0: _rectangle(40, 12), 1: _rectangle(40, 12)}, params)

    stacked = ComputeEnergy(
        parts, {0: Placement(0, np.array([10.0, 10.0])), 1: Placement(1, np.array([10.0, 14.0]))}, container, params
    )
    apart = ComputeEnergy(
        parts, {0: Placement(0, np.array([10.0, 8.0])), 1: Placement(1, np.array([10.0, 45.0]))}, container, params
    )

    assert not stacked.feasible
    assert apart.feasible
    assert apart.energy == 0.0


def test_spoons_in_a_five_by_two_report_a_finite_gradient():
    """A smoke test on real geometry: the fixtures piled on top of each
    other must produce a large but finite energy and usable forces, not a
    NaN from some degenerate sample.
    """
    params = LayoutParameters(pocket_offset=1.0)
    parts = LoadParts(["test_data/big_spoon.svg", "test_data/medium_spoon.svg", "test_data/small_spoon.svg"], params)
    container = BuildContainer(5, 2)
    placements = {part_id: Placement(part_id, np.array([20.0, 20.0])) for part_id in parts}

    result = ComputeEnergy(parts, placements, container, params)

    assert np.isfinite(result.energy) and result.energy > 0
    for force in result.forces.values():
        assert np.isfinite(force).all()
