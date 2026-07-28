"""Moving parts along the forces an energy reports.

Two passes descend an energy: the solver relaxes a fresh arrangement until
nothing overlaps, and the spacing pass balances one that already fits. They
descend different energies toward different goals, but the step itself is
the same, and it is the part with tuning in it - the per-sample
normalization, the damping, and the cap on how far a part may move at once.
Kept here so those are decided once instead of drifting between two copies.

The cap is the subtle one. It is not only for stability: a part that jumps
several millimeters can land more than halfway through another, which is
exactly where a distance field's forces reverse (see energy.ComputeEnergy).
Capping the step keeps both passes inside the range where the gradient they
are following can be trusted.

Watching a descent lives here too, for the same reason: both passes take
the same step, so both are observed the same way.
"""

from dataclasses import dataclass
from typing import Callable

import numpy as np

from pipeline.layout.parameters import LayoutParameters
from pipeline.layout.part import Part
from pipeline.layout.placement import Placement

RELAXING = "relaxing"
SPREADING = "spreading"


@dataclass(frozen=True)
class Snapshot:
    """One iteration of one descent, for a caller that wants to see the
    arrangement move rather than only hear that it is busy.

    `packer.Progress` says which attempt has started; this says what that
    attempt is doing. It carries the placements themselves, so a consumer
    can draw the bin through `preview.LayoutShapes` exactly as the printed
    sheet is drawn - an animation assembled any other way would be a second
    drawing of the same layout, free to disagree with the first.

    `energy` means something different in each phase, which is why `phase`
    is here to say which. Relaxing drives the real clearances to zero, so
    zero is feasible; spreading drives the inflated spring clearances, which
    never reach zero and are not a feasibility test at all.

    The placements are a fresh snapshot per iteration - `Descent.Placements`
    builds new objects and the descent rebinds rather than mutates - so a
    consumer may keep them.
    """

    grid: tuple[int, int]
    attempt: int
    phase: str
    iteration: int
    placements: dict[int, Placement]
    energy: float

    def __str__(self) -> str:
        n, m = self.grid
        return f"{n}x{m} attempt {self.attempt + 1}, {self.phase} {self.iteration}, energy {self.energy:.3f}"


# What an observer is handed, and what a descent pass can report. The pass
# knows only its own iteration; everything else is context the caller
# already had, bound on by `Reporting`.
Observer = Callable[[Snapshot], None]
Reporter = Callable[[int, dict[int, Placement], float], None]


def Reporting(observer: Observer | None, grid: tuple[int, int], attempt: int, phase: str) -> Reporter | None:
    """Bind one pass's context onto a snapshot observer.

    A factory rather than a closure written at the call site, so each
    pass's values are bound by argument passing instead of by whatever the
    enclosing variables happen to hold when the callback fires - the same
    reason `packer._AttemptReporter` exists.
    """
    if observer is None:
        return None
    return lambda iteration, placements, energy: observer(Snapshot(grid, attempt, phase, iteration, placements, energy))


class Descent:
    """Positions under damped descent, with orientations carried along.

    Orientation is discrete (D1), so no force acts on it - it is held fixed
    and handed back with each arrangement rather than being something this
    can move.
    """

    def __init__(self, parts: dict[int, Part], placements: dict[int, Placement], params: LayoutParameters):
        self._params = params
        self._positions = {part_id: p.position.astype(np.float64).copy() for part_id, p in placements.items()}
        self._orientations = {part_id: p.orientation for part_id, p in placements.items()}
        self._velocities = {part_id: np.zeros(2) for part_id in placements}
        # Force scales with how many samples a part has, which is a property
        # of its size and the raster resolution rather than of how badly it
        # is placed. Dividing it out keeps one step size meaningful for a
        # spoon and a washer alike.
        self._per_sample = {part_id: 1.0 / max(1, len(parts[part_id].samples)) for part_id in placements}

    def Placements(self) -> dict[int, Placement]:
        """The current arrangement, ready to hand to an energy."""
        return {
            part_id: Placement(part_id, position, self._orientations[part_id])
            for part_id, position in self._positions.items()
        }

    def Step(self, forces: dict[int, np.ndarray], noise: float = 0.0, rng: np.random.Generator | None = None) -> None:
        """Advance every part one step along `forces`.

        `noise` is added to the move but deliberately not to the velocity:
        it is there to shake an attempt out of a local minimum, and letting
        the damping accumulate it would turn a nudge into a drift.

        Asking for noise without a generator to draw it from raises rather
        than quietly stepping without any. Everything here is seeded so a
        layout reproduces - a relaxation that silently lost its jitter
        would still return a layout, just a worse one, and nothing would
        say so.
        """
        if noise > 0.0 and rng is None:
            raise ValueError("noise needs a seeded generator to draw from")

        for part_id, position in self._positions.items():
            velocity = self._params.damping * self._velocities[part_id]
            velocity = velocity + self._params.step_scale * forces[part_id] * self._per_sample[part_id]
            self._velocities[part_id] = velocity

            move = velocity
            if rng is not None and noise > 0.0:
                move = move + rng.normal(scale=noise, size=2)

            distance = float(np.linalg.norm(move))
            if distance > self._params.max_step:
                move = move * (self._params.max_step / distance)
            self._positions[part_id] = position + move
