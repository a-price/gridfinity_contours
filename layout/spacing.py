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

**The springs only reach a couple of millimetres, which is not the same as
using the bin.** They go slack the moment nothing is within reach of
anything else, so in a bin with room to spare a part stops a few
millimetres off the two walls bottom-left fill left it against, with the
rest of the bin empty - measured at 4.1mm from one wall and 76.2mm from
the opposite one. Lengthening the springs cannot fix that: a part only
feels another while a sample lands inside its raster, and that raster is
sized when the part is built, long before the bin is known.

Lengthening them was tried and measured. It does couple the parts - at a
42mm rest length all three pairs of a three-part 4x3 bin come within
reach - but it does not distribute them, because `SpringParameters`
inflates the wall clearance by the same margin and at bin scale every
part violates all four walls at once; the descent stalls and the
placements come back identical. Inflating only the *pair* springs does
work, and costs the tight bins the whole point of this pass: the three
spoons in 5x2 went from a gap spread of 0.04mm to 24.57mm.

So `Distribute` finishes the job geometrically rather than with a force,
by centering the arrangement and then scaling it out into the room that
is left.
"""

from dataclasses import replace

import numpy as np

from layout.container import Container
from layout.descent import Descent, Reporter
from layout.energy import ComputeEnergy
from layout.parameters import LayoutParameters
from layout.part import Part
from layout.placement import Placement


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


def Separations(
    parts: dict[int, Part],
    placements: dict[int, Placement],
) -> dict[tuple[int, int], float]:
    """How far apart every pair of parts actually is, in mm, keyed by part
    ids.

    Measured both ways round and the smaller kept, matching how the energy
    sees a pair - one part's boundary can be near the other's field without
    the reverse being true.

    Read off the rasterized distance fields, so pairs beyond a field's
    reach come back very large rather than at their true distance. That is
    the right answer for a caller asking "are these two near each other",
    and the wrong one for a caller that needs an exact figure at long
    range - `verify.MinimumSeparation` measures polygon to polygon for
    that.

    Takes no `LayoutParameters`, which is the point of it being separate
    from `Gaps`: this is a fact about where the parts ended up, not about
    what any particular parameter set would have required of them.
    """
    separations: dict[tuple[int, int], float] = {}

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
            separations[(id_a, id_b)] = separation

    return separations


def Gaps(
    parts: dict[int, Part],
    placements: dict[int, Placement],
    params: LayoutParameters,
) -> dict[tuple[int, int], float]:
    """Every pair's slack over the pair clearance, keyed by part ids.

    Slack rather than raw distance, so that zero means "exactly at the
    limit" and the number is comparable across contacts whose clearances
    differ - which is what the spacing pass wants, since it is trying to
    even out how much room each contact has to spare rather than how far
    apart things are.

    Pairs beyond the field's reach come back as a very large slack; they
    are not neighbours, and the springs will not act on them.
    """
    return {pair: separation - params.c_pair for pair, separation in Separations(parts, placements).items()}


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


def _Boxes(
    parts: dict[int, Part],
    placements: dict[int, Placement],
) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    """Each placed part's axis-aligned box, as (minimum, maximum) corners."""
    return {part_id: placement.Bounds(parts[part_id]) for part_id, placement in placements.items()}


def _Limits(container: Container, params: LayoutParameters) -> tuple[np.ndarray, np.ndarray]:
    """The box the inflation may push a part's own box out to.

    `spacing_wall` rather than the bare `c_wall`, which is what this used
    to be (plus a raster cell, so that the energy's analytic wall and
    `verify`'s polygonal one could not disagree by microns about a part
    sitting exactly on the limit).

    The bare clearance is the wrong target here, and measurably so. It is
    the *legal minimum*, so inflating to it spends every millimetre of a
    roomy bin's slack on the gaps between parts and leaves none at the
    wall: five parts in a 4x3 bin came out with one 2.2mm off the rim
    while the gaps between them ran to tens of millimetres. `spacing_wall`
    is the rest length `Spread` already drives wall contacts to when it
    has the room - the project's existing answer to "how far off a wall
    should a part sit" - so using it here makes this pass agree with the
    spring pass instead of overriding it.

    It stays strictly above the old value, so the microns argument still
    holds; and it can only reduce how far the inflation pushes, never
    increase it, so no bin that fits today stops fitting.

    Symmetric, so it does not move the center - only how far the inflation
    is allowed to go.
    """
    margin = params.spacing_wall
    return np.array([margin, margin]), np.array([container.width, container.height]) - margin


def _Shift(boxes: dict[int, tuple[np.ndarray, np.ndarray]], container: Container, params: LayoutParameters):
    """The translation that centers the whole arrangement in the bin.

    One vector for every part, so pair distances are untouched - which is
    what makes this safe by construction rather than by inspection. The
    worst wall gap can only improve: a side that had `L` and `R` ends up
    with `(L + R) / 2`, and that is at least `min(L, R)`.
    """
    low = np.min([box[0] for box in boxes.values()], axis=0)
    high = np.max([box[1] for box in boxes.values()], axis=0)
    limit_low, limit_high = _Limits(container, params)
    return ((limit_low + limit_high) - (low + high)) / 2.0


def _Centered(
    parts: dict[int, Part],
    placements: dict[int, Placement],
    container: Container,
    params: LayoutParameters,
) -> dict[int, Placement]:
    """The arrangement translated so its bounding box sits in the middle of
    the bin.

    One vector for every part, so no pair distance changes and the worst
    wall gap can only improve - which is what makes this safe to apply
    without measuring anything.
    """
    shift = _Shift(_Boxes(parts, placements), container, params)
    return {
        part_id: replace(placement, position=np.asarray(placement.position, dtype=np.float64) + shift)
        for part_id, placement in placements.items()
    }


def _Inflation(boxes, center: np.ndarray, container: Container, params: LayoutParameters) -> np.ndarray:
    """How far the arrangement can be scaled about `center` before some part
    reaches a wall, per axis.

    What scales is each part's *center*, not its corner. Scaling corners
    looks equivalent and is not: a part straddling the middle of the bin has
    its corner half a part-width off center, so scaling that corner slides
    the part bodily outward - which drove a lone part into the wall it was
    supposed to be moving away from.

    A part is only ever constrained by the wall it is moving toward, so each
    part and axis contributes at most one bound. One sitting exactly on the
    center does not move at all, and constrains nothing.
    """
    limit_low, limit_high = _Limits(container, params)
    scale = np.array([np.inf, np.inf])

    for low, high in boxes.values():
        middle = (low + high) / 2.0
        half = (high - low) / 2.0
        for axis in range(2):
            offset = middle[axis] - center[axis]
            if offset > 0:
                room = limit_high[axis] - half[axis] - center[axis]
                scale[axis] = min(scale[axis], room / offset)
            elif offset < 0:
                room = limit_low[axis] + half[axis] - center[axis]
                scale[axis] = min(scale[axis], room / offset)

    return np.maximum(1.0, np.where(np.isfinite(scale), scale, 1.0))


def _Scaled(
    parts: dict[int, Part],
    placements: dict[int, Placement],
    center: np.ndarray,
    scale: np.ndarray,
) -> dict[int, Placement]:
    """The arrangement with every part's center scaled away from `center`.

    The part keeps its size and its pose, so its anchor moves by exactly
    what its center moved - which is why this can shift `position` directly
    without re-deriving anything about how the part is turned.
    """
    moved = {}
    for part_id, placement in placements.items():
        position = np.asarray(placement.position, dtype=np.float64)
        low, high = placement.Bounds(parts[part_id])
        middle = (low + high) / 2.0
        moved[part_id] = replace(placement, position=position + (middle - center) * (scale - 1.0))
    return moved


def Distribute(
    parts: dict[int, Part],
    placements: dict[int, Placement],
    container: Container,
    params: LayoutParameters,
) -> dict[int, Placement]:
    """Spread a feasible arrangement out to use the whole bin.

    `Spread` evens out gaps that are *tight*, and stops the moment nothing
    is within a spring's reach of anything else. In a bin with room to
    spare that happens almost immediately, so a lone part ends up wherever
    bottom-left fill dropped it, a few millimetres off two walls, with the
    rest of the bin empty. Measured: a 40x18mm part in a 120x78mm bin
    settled 4.1mm from the left wall and 76.2mm from the right.

    The reason no force fixes that is `pad`. Parts only feel each other
    while a sample lands inside the other's raster, and that raster is
    sized when the part is built, long before the bin is known. Reaching
    across a roomy bin would mean rasterizing every part with a skirt as
    wide as the largest bin it might ever land in.

    So this is geometry rather than a force, in three steps that are safe
    for different reasons. Centering translates every part by one vector,
    which cannot change any pair distance at all. Inflating scales the
    positions about the center, which can only push parts further apart -
    though for concave parts "further apart" is not quite a proof, since a
    spoon slid along the line of centers can find a different part of its
    neighbour, so the result is checked rather than trusted. Then it
    centers *again*, and that step is not a tidy-up: inflation scales part
    centers while centering centers the arrangement's bounding box, so
    scaling silently undoes the centering it started from. Whichever part
    sits furthest out reaches its wall first and stops the scale, leaving
    every other part with whatever margin it happened to get - measured at
    7.0mm on one side against 2.2mm on the other, with one part jammed
    into a corner.

    It is deliberately not iterated. Re-centering frees room on the side
    the scale was pinned against, so another round would inflate into it
    and pin the other side instead, converging on parts pressed against
    both walls - the state the second centering exists to undo.

    Returns the input untouched if the improvement does not verify.
    """
    if not placements:
        return dict(placements)

    centered = _Centered(parts, placements, container, params)
    best = centered if _Holds(parts, centered, container, params) else dict(placements)

    boxes = _Boxes(parts, best)
    low = np.min([box[0] for box in boxes.values()], axis=0)
    high = np.max([box[1] for box in boxes.values()], axis=0)
    center = (low + high) / 2.0

    # Backed off rather than bisected: the whole scale is one number per
    # axis, and the first few steps down from the geometric limit are where
    # any concave surprise shows up.
    limit = _Inflation(boxes, center, container, params)
    for fraction in (1.0, 0.75, 0.5, 0.25):
        scale = 1.0 + (limit - 1.0) * fraction
        if (scale <= 1.0).all():
            break
        candidate = _Centered(parts, _Scaled(parts, best, center, scale), container, params)
        if _Holds(parts, candidate, container, params):
            return candidate

    return best


def _Holds(
    parts: dict[int, Part],
    placements: dict[int, Placement],
    container: Container,
    params: LayoutParameters,
) -> bool:
    """Whether an arrangement still satisfies the true clearances.

    The same discipline `Spread` follows: propose, then verify against what
    the layout actually has to guarantee, and keep nothing that fails.
    """
    return ComputeEnergy(parts, placements, container, params).feasible
