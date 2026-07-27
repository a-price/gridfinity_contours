"""How badly an arrangement violates its clearances, and which way to move
each part to fix it.

Two terms and no others, per D3: parts repel each other, and the interior
wall repels every part inward. There is deliberately no attraction pulling
parts together - within a fixed bin every feasible arrangement is equally
good, and compaction is the grid-size search's job.
"""

from dataclasses import dataclass

import numpy as np

from pipeline.layout.container import (
    DEFAULT_INTERIOR_INSET_MM,
    DIVIDER_WIDTH_MM,
    MIN_WALL_MM,
    Container,
)
from pipeline.layout.part import DEFAULT_RESOLUTION_MM, Part
from pipeline.layout.placement import Placement, RotateVectors


@dataclass
class LayoutParameters:
    """Tunables for the packer.

    The three clearances are derived from `pocket_offset` rather than set
    independently, because they are not independent: a pocket is cut
    `pocket_offset` larger than its object so the object drops in, so two
    pockets `pair_clearance` apart leave a divider of
    `pair_clearance - 2*pocket_offset`, which has to stay printable.
    Setting them separately invites a layout whose dividers are too thin to
    print. Explicit values still override.
    """

    pocket_offset: float = 1.0
    pair_clearance: float | None = None
    wall_clearance: float | None = None
    resolution: float = DEFAULT_RESOLUTION_MM
    pair_weight: float = 1.0
    wall_weight: float = 1.0
    inset: float = DEFAULT_INTERIOR_INSET_MM
    max_grid: int = 6
    iterations: int = 400
    restarts: int = 24
    seed: int = 0

    @property
    def c_pair(self) -> float:
        """Minimum part-to-part spacing: enough for both pockets' offsets
        plus a printable divider between them.
        """
        if self.pair_clearance is not None:
            return self.pair_clearance
        return 2.0 * self.pocket_offset + DIVIDER_WIDTH_MM

    @property
    def c_wall(self) -> float:
        """Minimum part-to-wall spacing: the pocket's offset plus a
        printable wall.
        """
        if self.wall_clearance is not None:
            return self.wall_clearance
        return self.pocket_offset + MIN_WALL_MM

    @property
    def pad(self) -> float:
        """How far each part's distance field must reach beyond itself.

        A part only feels another once a sample lands inside the other's
        raster, so a field that stops short of `c_pair` would let parts pass
        straight through each other at exactly the distance the clearance
        is meant to enforce. The margin is generous because the cost is a
        few hundred kilobytes.
        """
        return self.c_pair + 2.0


@dataclass
class EnergyResult:
    """What the solver needs from one evaluation: how badly the current
    arrangement violates its constraints, and which way to move each part
    to fix it.

    `energy` is zero exactly when every clearance is satisfied, which makes
    it both the objective and the feasibility test - there is no separate
    collision pass to disagree with it.

    The other two fields exist because the forces stop being trustworthy
    once a sample passes the other part's medial axis (see ComputeEnergy),
    so the solver needs to be able to tell it has wandered into that regime
    rather than silently converging inside it.

    `deepest_penetration` is how far the worst sample has pushed inside
    another part, in mm - a diagnostic, but awkward to threshold, since how
    deep is "too deep" depends on how big the parts are.

    `containment` is the largest fraction of any one part's boundary lying
    inside another, which needs no such judgement: 1.0 means a part has
    been swallowed whole, and that is unambiguous at any scale. It is the
    signal to abort on. Note the asymmetry with crossing - two parts merely
    overlapping is the normal, expected state partway through a relaxation
    and resolves itself, whereas containment never occurs legitimately and
    never recovers, because every sample of the swallowed part is being
    pushed toward whichever wall of its captor happens to be nearest.
    """

    energy: float
    forces: dict[int, np.ndarray]
    deepest_penetration: float = 0.0
    containment: float = 0.0

    @property
    def feasible(self) -> bool:
        return self.energy <= 0.0

    @property
    def engulfed(self) -> bool:
        """Whether some part has been swallowed entirely by another - the
        unrecoverable case, worth restarting from rather than descending.
        """
        return self.containment >= 1.0


def _PenaltyAndScale(distance: np.ndarray, clearance: float, weight: float) -> tuple[np.ndarray, np.ndarray]:
    """The quadratic penalty for samples closer than `clearance`, and the
    factor by which each sample's direction is scaled to give the force.

    Energy is `weight * violation^2`, so the force is its negative
    derivative, `2 * weight * violation`, along the direction that
    increases distance.
    """
    violation = np.maximum(0.0, clearance - distance)
    return weight * violation**2, 2.0 * weight * violation


def ComputeEnergy(
    parts: dict[int, Part],
    placements: dict[int, Placement],
    container: Container,
    params: LayoutParameters,
) -> EnergyResult:
    """Total constraint violation of an arrangement, and the force on each
    part that reduces it.

    Two terms, per D3, and no others: parts repel each other, and the
    interior wall repels every part inward. There is deliberately no
    attraction pulling parts together - within a fixed bin every feasible
    arrangement is equally good, and compaction is the grid-size search's
    job, not a force's.

    The returned forces are exactly the negative gradient of the returned
    energy, so a descent step is guaranteed to reduce it (for a small
    enough step). The finite-difference test holds this to account.

    **The forces are only meaningful for shallow overlap.** A distance
    field points toward the nearest way out, so once a sample penetrates
    past the other part's medial axis, the nearest exit is out the *far*
    side and the force reverses - pushing the parts together instead of
    apart. Measured on two 10mm squares, the push stays correct to about
    50% overlap, reverses beyond it, and total energy actually *falls*
    toward full overlap, making coincident parts a spurious minimum a
    descent solver can settle into and report as converged.

    This is inherent to penalty methods on distance fields, not a defect in
    this implementation. Clamping the trusted depth band was measured and
    rejected: it fixes the mid-range but not near-coincidence, where the
    failure is symmetry rather than depth, and it buys that with an extra
    parameter and an energy that no longer matches D3.

    The mitigation belongs in the solver, and `containment` is what it
    should key on: start from an arrangement with no part inside another,
    and abandon any attempt that reaches one. Crossing needs no such
    handling - overlapping parts are the ordinary midway state of a
    relaxation, and the force resolves them correctly as long as they never
    get more than about halfway through each other.
    """
    for part_id, part in parts.items():
        if part.pad < params.c_pair:
            raise ValueError(
                f"part {part_id}'s distance field reaches {part.pad}mm beyond it, short of the "
                f"{params.c_pair}mm pair clearance - parts would pass through each other unnoticed"
            )

    forces = {part_id: np.zeros(2) for part_id in placements}
    energy = 0.0
    deepest = 0.0
    containment = 0.0

    world_samples = {part_id: placement.SamplesToWorld(parts[part_id]) for part_id, placement in placements.items()}

    for part_id, samples in world_samples.items():
        penalty, scale = _PenaltyAndScale(container.SampleDepth(samples), params.c_wall, params.wall_weight)
        energy += float(penalty.sum())
        forces[part_id] += (scale[:, None] * container.SampleDerivative(samples)).sum(axis=0)

    ordered = sorted(placements)
    for index, id_a in enumerate(ordered):
        for id_b in ordered[index + 1 :]:
            # Each part's boundary is tested against the other's field.
            # Sampling only one way is not symmetric and misses the case
            # where one part swallows another without either boundary
            # landing near the other's samples.
            for source, target in ((id_a, id_b), (id_b, id_a)):
                local = placements[target].ToLocal(parts[target], world_samples[source])
                distance = parts[target].SampleSdf(local)
                penalty, scale = _PenaltyAndScale(distance, params.c_pair, params.pair_weight)
                if not penalty.any():
                    continue

                # How much of this part's boundary is inside the other one.
                # Free here - the signs are already computed - and the only
                # scale-free way to recognize that a part has been
                # swallowed rather than merely bumped into.
                penetrating = distance < 0
                if penetrating.any():
                    deepest = max(deepest, -float(distance.min()))
                    containment = max(containment, float(penetrating.mean()))
                energy += float(penalty.sum())
                # The field is the target's, so its derivative comes back in
                # the target's frame and has to be rotated into the bin's.
                direction = RotateVectors(parts[target].SampleDerivative(local), placements[target].orientation)
                push = (scale[:, None] * direction).sum(axis=0)
                forces[source] += push
                forces[target] -= push  # equal and opposite, which is also the exact gradient

    return EnergyResult(
        energy=energy,
        forces=forces,
        deepest_penetration=max(0.0, deepest),
        containment=containment,
    )
