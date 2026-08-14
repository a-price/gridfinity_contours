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

from layout.parameters import LayoutParameters
from layout.part import Part
from layout.placement import Placement, PoseInertia, PoseRadius

RELAXING = "relaxing"
SPREADING = "spreading"

# Floor for the per-part normalizers, so a degenerate part - one whose
# samples all sit on its pivot - divides by something rather than raising.
# Small enough that no real contour ever reaches it.
_TINY = 1e-12


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
    """Positions under damped descent, and - when asked - angles alongside
    them.

    The quarter turn is always discrete (D1) and always carried rather than
    moved: nothing exerts torque on a variable that can only take four
    values, so it is handed back with each arrangement unchanged. What
    `rotate` opens up is the *free* angle on top of it, which is a real
    continuous coordinate with a real gradient, and which the FREE rotation
    mode relaxes exactly as it relaxes position.

    `rotate` is an argument rather than read from `params` because the two
    passes want different answers from the same parameters. The solver
    relaxes angles when the mode allows it; the spacing pass never does,
    since it starts from a feasible arrangement and is balancing gaps, and
    turning a part there would re-open the question its input already
    settled.
    """

    def __init__(
        self,
        parts: dict[int, Part],
        placements: dict[int, Placement],
        params: LayoutParameters,
        rotate: bool = False,
    ):
        self._params = params
        self._rotate = rotate
        self._positions = {part_id: p.position.astype(np.float64).copy() for part_id, p in placements.items()}
        self._orientations = {part_id: p.orientation for part_id, p in placements.items()}
        self._angles = {part_id: float(p.angle) for part_id, p in placements.items()}
        self._velocities = {part_id: np.zeros(2) for part_id in placements}
        self._spins = {part_id: 0.0 for part_id in placements}
        # Force scales with how many samples a part has, which is a property
        # of its size and the raster resolution rather than of how badly it
        # is placed. Dividing it out keeps one step size meaningful for a
        # spoon and a washer alike.
        self._per_sample = {part_id: 1.0 / max(1, len(parts[part_id].samples)) for part_id in placements}
        # The angular equivalents, and the reason free rotation adds only
        # one tunable rather than four. Torque grows with both the sample
        # count and how far the samples sit from the pivot, and the second
        # moment takes out both; the radius converts the millimetre limits
        # into radians. See placement.PoseInertia and placement.PoseRadius.
        self._per_moment = {part_id: 1.0 / max(_TINY, PoseInertia(parts[part_id])) for part_id in placements}
        self._per_radius = {part_id: 1.0 / max(_TINY, PoseRadius(parts[part_id])) for part_id in placements}

    def Placements(self) -> dict[int, Placement]:
        """The current arrangement, ready to hand to an energy."""
        return {
            part_id: Placement(part_id, position, self._orientations[part_id], self._angles[part_id])
            for part_id, position in self._positions.items()
        }

    def Step(
        self,
        forces: dict[int, np.ndarray],
        noise: float = 0.0,
        rng: np.random.Generator | None = None,
        torques: dict[int, float] | None = None,
    ) -> None:
        """Advance every part one step along `forces`, and along `torques`
        if this descent was told to rotate.

        `noise` is added to the move but deliberately not to the velocity:
        it is there to shake an attempt out of a local minimum, and letting
        the damping accumulate it would turn a nudge into a drift.

        Asking for noise without a generator to draw it from raises rather
        than quietly stepping without any. Everything here is seeded so a
        layout reproduces - a relaxation that silently lost its jitter
        would still return a layout, just a worse one, and nothing would
        say so. A rotating descent handed no torques raises for the same
        reason: it would keep converging, on the wrong problem.
        """
        if noise > 0.0 and rng is None:
            raise ValueError("noise needs a seeded generator to draw from")
        if self._rotate and torques is None:
            raise ValueError("a rotating descent needs torques; without them the angle is not being solved for")

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

            if self._rotate and torques is not None:
                self._angles[part_id] += self._Turn(part_id, torques[part_id], noise, rng)

    def _Turn(self, part_id: int, torque: float, noise: float, rng: np.random.Generator | None) -> float:
        """How far one part turns this step, in radians.

        The same shape as the translation above - damped velocity, then
        noise, then a cap - with every millimetre-denominated quantity
        divided by the part's own radius so that the cap and the jitter
        mean what they mean for translation: how far the part's furthest
        point is allowed to travel. That is what keeps a step from carrying
        a sample past a neighbour's medial axis, where `ComputeEnergy` says
        its own forces reverse, and it is why the cap has to scale with the
        part rather than being a fixed number of degrees.
        """
        spin = self._params.damping * self._spins[part_id]
        spin = spin + self._params.angular_step_scale * torque * self._per_moment[part_id]
        self._spins[part_id] = spin

        turn = spin
        if rng is not None and noise > 0.0:
            turn = turn + float(rng.normal(scale=noise * self._per_radius[part_id]))

        limit = self._params.max_step * self._per_radius[part_id]
        return float(np.clip(turn, -limit, limit))
