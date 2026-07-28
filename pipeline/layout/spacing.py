"""Evening out the gaps once a layout already fits.

The solver stops the moment nothing overlaps, because that is what D3's
energy measures - and every feasible arrangement scores identically. The
result fits but looks arbitrary: measured on the three spoons, one pair
sat 0.49mm above the minimum clearance while another had 23.58mm.

This pass fixes that with a spring network. Every pair of parts close
enough to feel each other, and every part close enough to a wall, gets a
quadratic spring pulling toward one shared rest length. Where they are
mutually blocked - which in a full bin is everywhere - equal springs
balance at equal compression, so the gaps come out even without anything
imposing evenness directly.

Two properties make this sound rather than merely plausible:

* No spring ever pulls. The other reading of "make the gaps the same" is
  to minimize their variance, which would actively drag a roomy gap down
  toward a cramped one - worse for the print, and worse for the thinnest
  divider, which is the number that decides whether it succeeds. Note
  this is a statement about the springs, not about every gap: a part
  balances several at once, so a generous gap can still close as its
  neighbour is pushed off something else. What cannot happen is a force
  whose purpose is to close it.
* "Close enough to feel each other" is not a threshold anyone chose. It
  is exactly the reach of the distance field, which is where a part stops
  being able to sense another at all. Parts further apart than that are
  not neighbours, and should not be equalized against each other - a
  spoon at one end of the bin has no business being spaced against one at
  the other.

Feasibility is never at risk: the springs run at *inflated* clearances,
which strictly contain the true ones, and every candidate is checked
against the true clearances before being kept.
"""

from dataclasses import replace

import numpy as np

from pipeline.layout.container import Container
from pipeline.layout.descent import Descent, Reporter
from pipeline.layout.energy import ComputeEnergy
from pipeline.layout.parameters import LayoutParameters
from pipeline.layout.part import Part
from pipeline.layout.placement import Placement


def SpringParameters(params: LayoutParameters) -> LayoutParameters:
    """The same parameters with the clearances raised to the springs' rest
    length.

    The whole pass is the existing energy evaluated against wider
    clearances - there is no second force model to keep in step with the
    first. `LayoutParameters` already takes explicit clearance overrides,
    so this is the mechanism it was built for.

    `pair_clearance` is set a raster margin low because `c_pair_enforced`
    adds that margin back on; what the energy actually drives to is
    `spacing_pair`.
    """
    return replace(
        params,
        pair_clearance=params.spacing_pair - params.raster_margin,
        wall_clearance=params.spacing_wall,
    )


def Gaps(
    parts: dict[int, Part],
    placements: dict[int, Placement],
    params: LayoutParameters,
) -> dict[tuple[int, int], float]:
    """Every pair's slack over the pair clearance, keyed by part ids.

    Slack rather than raw distance, so that zero means "exactly at the
    limit" and the number is comparable across contacts whose clearances
    differ. Measured both ways round and the smaller kept, matching how
    the energy sees a pair - one part's boundary can be near the other's
    field without the reverse being true.

    Pairs beyond the field's reach come back as a very large slack; they
    are not neighbours, and the springs will not act on them.
    """
    gaps: dict[tuple[int, int], float] = {}

    # Once per part rather than once per direction of every pair, matching
    # ComputeEnergy - rotating a spoon's several thousand boundary samples
    # is the expensive half of this.
    world_samples = {part_id: placement.SamplesToWorld(parts[part_id]) for part_id, placement in placements.items()}

    ordered = sorted(placements)
    for index, id_a in enumerate(ordered):
        for id_b in ordered[index + 1 :]:
            separation = np.inf
            for source, target in ((id_a, id_b), (id_b, id_a)):
                local = placements[target].ToLocal(parts[target], world_samples[source])
                separation = min(separation, float(parts[target].SampleSdf(local).min()))
            gaps[(id_a, id_b)] = separation - params.c_pair

    return gaps


def Spread(
    parts: dict[int, Part],
    placements: dict[int, Placement],
    container: Container,
    params: LayoutParameters,
    on_step: Reporter | None = None,
) -> dict[int, Placement]:
    """Even out the gaps in an already-feasible arrangement.

    Returns the best arrangement found that still satisfies the *true*
    clearances, falling back to the one passed in - so this can only
    improve a layout or leave it alone. That fallback is also why the
    input is required to be feasible already: nothing here would fix an
    arrangement that was not, and returning it unchanged would launder a
    bad layout through a function that looks like it validated one. The
    solver calls this only after `CheckLayout` has passed.

    Descent only, with no noise and no restarts: this starts from a
    feasible arrangement and is looking for the nearest balanced one, not
    exploring. Noise here would risk walking out of a good nesting to no
    purpose.

    `on_step` sees every candidate, including the ones this then declines
    to keep. That is the pass as it actually runs, and the alternative -
    reporting only improvements - would show a smooth march that never
    happened.
    """
    if params.spacing_iterations <= 0:
        return dict(placements)

    springs = SpringParameters(params)
    descent = Descent(parts, placements, params)

    best = dict(placements)
    best_energy = ComputeEnergy(parts, placements, container, springs).energy
    stalled = 0

    for iteration in range(params.spacing_iterations):
        current = descent.Placements()
        result = ComputeEnergy(parts, current, container, springs)

        if on_step is not None:
            on_step(iteration, current, result.energy)

        # Keep it only if the true clearances still hold. The springs push
        # strictly further than the real constraints ask, so this is a
        # guard against overshoot rather than an expected failure. The
        # short-circuit matters: the feasibility check is a second full
        # evaluation, and it is only ever needed on an improvement.
        if result.energy < best_energy and ComputeEnergy(parts, current, container, params).feasible:
            best_energy, best = result.energy, current
            stalled = 0
        else:
            stalled += 1
            # A wedged arrangement does not come unwedged by being pushed
            # harder. Without this the pass costs its full budget on every
            # layout, including the many that balance in a few iterations.
            if stalled >= params.patience:
                break

        # No noise: this starts from a feasible arrangement and is looking
        # for the nearest balanced one, not exploring.
        descent.Step(result.forces)

    return best
