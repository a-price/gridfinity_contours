"""Tests for the shared descent step.

The properties here are the ones both callers rely on and neither states:
that a step size means the same thing for a spoon and a washer, that no
move is long enough to leave the range where the forces are trustworthy,
and that noise shakes an attempt without being remembered.
"""

import numpy as np
import pytest

from layout.descent import Descent
from layout.parameters import LayoutParameters
from layout.part import BuildPart, Part
from layout.placement import Placement
from conftest import Rectangle as _rectangle


def _part(width: float = 10.0, height: float = 10.0) -> Part:
    return BuildPart(_rectangle(width, height), resolution=1.0, pad=2.0)


def _one(position=(0.0, 0.0), orientation: int = 0) -> tuple[dict[int, Part], dict[int, Placement]]:
    return {0: _part()}, {0: Placement(0, np.array(position, dtype=np.float64), orientation)}


def _params(**overrides) -> LayoutParameters:
    return LayoutParameters(**overrides)


# ------------------------------------------------------------- the arrangement


def test_it_starts_at_the_arrangement_it_was_given():
    parts, placements = _one((3.0, 4.0))

    current = Descent(parts, placements, _params()).Placements()

    assert current[0].position == pytest.approx([3.0, 4.0])


def test_orientations_are_carried_not_moved():
    """No force acts on orientation - it is discrete (D1) - so the descent
    holds it fixed and hands it back with every arrangement.
    """
    parts, placements = _one(orientation=3)
    descent = Descent(parts, placements, _params())

    descent.Step({0: np.array([100.0, 0.0])})

    assert descent.Placements()[0].orientation == 3


def test_stepping_leaves_the_input_placements_alone():
    """The caller keeps the arrangement it passed in - the spacing pass
    falls back to it when nothing better is found.
    """
    parts, placements = _one((5.0, 5.0))
    descent = Descent(parts, placements, _params())

    descent.Step({0: np.array([10.0, 10.0])})

    assert placements[0].position == pytest.approx([5.0, 5.0])


def test_no_force_and_no_noise_moves_nothing():
    parts, placements = _one((5.0, 5.0))
    descent = Descent(parts, placements, _params())

    descent.Step({0: np.zeros(2)})

    assert descent.Placements()[0].position == pytest.approx([5.0, 5.0])


# ------------------------------------------------------------------ the step


def test_a_part_moves_along_its_force():
    parts, placements = _one()
    descent = Descent(parts, placements, _params())

    descent.Step({0: np.array([1.0, 0.0])})

    moved = descent.Placements()[0].position
    assert moved[0] > 0.0
    assert moved[1] == pytest.approx(0.0)


def test_no_move_exceeds_the_cap():
    """Not just for stability: a part that jumps far enough can land more
    than halfway inside another, which is where the forces reverse.
    """
    params = _params(max_step=0.6)
    parts, placements = _one()
    descent = Descent(parts, placements, params)

    descent.Step({0: np.array([1e6, 1e6])})

    assert float(np.linalg.norm(descent.Placements()[0].position)) == pytest.approx(params.max_step)


def test_the_cap_keeps_the_direction():
    params = _params(max_step=0.6)
    parts, placements = _one()
    descent = Descent(parts, placements, params)

    descent.Step({0: np.array([3e6, 4e6])})

    moved = descent.Placements()[0].position
    assert moved / np.linalg.norm(moved) == pytest.approx([0.6, 0.8])


def test_one_step_size_suits_parts_of_different_sizes():
    """Force scales with a part's sample count, which says nothing about
    how badly it is placed. Dividing it out is what lets a single
    `step_scale` mean the same thing for a spoon and a washer.
    """
    params = _params(max_step=1e6)
    small, large = _part(10, 10), _part(80, 80)
    assert len(large.samples) > 4 * len(small.samples), "the fixture must have genuinely different sample counts"

    placements = {0: Placement(0, np.zeros(2))}
    force = np.array([1.0, 0.0])
    moves = []
    for part in (small, large):
        descent = Descent({0: part}, placements, params)
        # As the energy reports it: a sum over the part's own samples.
        descent.Step({0: force * len(part.samples)})
        moves.append(float(descent.Placements()[0].position[0]))

    assert moves[0] == pytest.approx(moves[1])


def test_velocity_carries_between_steps():
    """Damping is momentum, so a part under a constant force accelerates
    rather than moving the same distance every step.
    """
    params = _params(damping=0.6, max_step=1e6)
    parts, placements = _one()
    descent = Descent(parts, placements, params)
    force = {0: np.array([1.0, 0.0])}

    descent.Step(force)
    first = float(descent.Placements()[0].position[0])
    descent.Step(force)
    second = float(descent.Placements()[0].position[0]) - first

    assert second > first


# ----------------------------------------------------------------- the noise


def test_noise_moves_a_part_with_no_force_on_it():
    parts, placements = _one()
    descent = Descent(parts, placements, _params())

    descent.Step({0: np.zeros(2)}, noise=0.5, rng=np.random.default_rng(0))

    assert float(np.linalg.norm(descent.Placements()[0].position)) > 0.0


def test_noise_is_not_remembered():
    """The distinguishing property: noise is added to the move but not to
    the velocity. Letting the damping accumulate it would turn a nudge
    meant to escape a local minimum into a drift.
    """
    params = _params(damping=0.6, max_step=1e6)
    parts, placements = _one()
    descent = Descent(parts, placements, params)

    descent.Step({0: np.zeros(2)}, noise=1.0, rng=np.random.default_rng(0))
    kicked = descent.Placements()[0].position.copy()
    descent.Step({0: np.zeros(2)})

    assert descent.Placements()[0].position == pytest.approx(kicked)


def test_asking_for_noise_without_a_generator_is_refused():
    """Rather than stepping without any. A relaxation that silently lost
    its jitter would still return a layout, just a worse one, and nothing
    would say so.
    """
    parts, placements = _one()
    descent = Descent(parts, placements, _params())

    with pytest.raises(ValueError, match="seeded generator"):
        descent.Step({0: np.zeros(2)}, noise=1.0)


# --------------------------------------------------------- the free angle


def test_a_descent_that_was_not_asked_to_rotate_carries_the_angle():
    """The default, and what the 90 and 45 modes get. The angle is part of
    the arrangement and has to survive the pass; it is simply not something
    this moves.
    """
    parts, placements = _one()
    placements[0] = Placement(0, np.zeros(2), angle=0.4)
    descent = Descent(parts, placements, LayoutParameters())

    descent.Step({0: np.array([50.0, 0.0])}, torques={0: 900.0})

    assert descent.Placements()[0].angle == 0.4


def test_a_rotating_descent_turns_along_the_torque():
    parts, placements = _one()
    descent = Descent(parts, placements, LayoutParameters(), rotate=True)

    descent.Step({0: np.zeros(2)}, torques={0: 40.0})
    turned = descent.Placements()[0].angle

    assert turned > 0.0
    descent.Step({0: np.zeros(2)}, torques={0: -400.0})
    assert descent.Placements()[0].angle < turned


def test_a_rotating_descent_refuses_to_step_without_torques():
    """Silently holding the angle still would keep converging, on a
    different problem than the one asked for - the same reason noise
    without a generator raises rather than being skipped.
    """
    parts, placements = _one()
    descent = Descent(parts, placements, LayoutParameters(), rotate=True)

    with pytest.raises(ValueError, match="torques"):
        descent.Step({0: np.zeros(2)})


def test_one_step_never_moves_a_part_further_than_the_cap_allows():
    """Why the angular cap is derived from the part's radius rather than
    set in degrees. The cap exists so a step cannot carry a sample past a
    neighbour's medial axis, where the forces reverse - which is a
    statement about millimetres, so it has to become a smaller angle for a
    longer part.
    """
    params = LayoutParameters()
    long_part = {0: BuildPart(_rectangle(200.0, 10.0), resolution=1.0, pad=2.0)}
    placements = {0: Placement(0, np.zeros(2))}
    descent = Descent(long_part, placements, params, rotate=True)

    for _ in range(5):
        descent.Step({0: np.zeros(2)}, torques={0: 1e9})

    turned = descent.Placements()[0].angle
    radius = np.linalg.norm(long_part[0].samples - long_part[0].size / 2.0, axis=1).max()
    assert turned * radius <= 5 * params.max_step + 1e-9, "the tip outran the translation cap"


def test_the_same_torque_turns_a_small_part_further_than_a_large_one():
    """One `angular_step_scale` has to mean the same thing across parts,
    which is what the second moment normalization buys. Without it the
    setting would need retuning for every object size.
    """
    params = LayoutParameters()
    angles = []
    for length in (20.0, 200.0):
        parts = {0: BuildPart(_rectangle(length, 10.0), resolution=1.0, pad=2.0)}
        descent = Descent(parts, {0: Placement(0, np.zeros(2))}, params, rotate=True)
        descent.Step({0: np.zeros(2)}, torques={0: 100.0})
        angles.append(abs(descent.Placements()[0].angle))

    assert angles[0] > angles[1]
