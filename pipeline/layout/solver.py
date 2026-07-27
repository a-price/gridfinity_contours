"""Finding an arrangement of parts inside a bin of fixed size.

The shape of the search, and why (see docs/layout.md, D4):

Parts are placed by bottom-left fill first - largest first, each swept to
the first position where it sits flush against a wall or another part
without colliding - and only then relaxed. That ordering is not
incidental. The forces from a distance field reverse once a part is more
than halfway inside another (see energy.ComputeEnergy), so an
initialization that drops parts on top of each other starts the descent
inside a trap it then has to climb out of. Starting apart leaves the
relaxation with the only job it is reliably good at: nudging shallow
clearance violations away.

The relaxation is damped descent with decaying noise. The noise is what
makes it more than gradient descent - the energy landscape is full of
local minima where two parts are wedged apart by a third, and a purely
downhill solver parks in them. Everything is seeded, so the same input
gives the same bin twice; a layout that did not reproduce would not match
the sheet printed alongside it.
"""

from typing import Callable

import numpy as np

from pipeline.layout.container import BuildContainer, Container
from pipeline.layout.energy import ComputeEnergy, LayoutParameters, PlacementEnergy
from pipeline.layout.part import Part
from pipeline.layout.placement import Layout, Placement, RotatedSize
from pipeline.layout.verify import CheckLayout


def FittingOrientations(part: Part, container: Container, params: LayoutParameters) -> list[int]:
    """The quarter turns at which a part's bounding box still fits inside
    the bin, wall clearance included.

    Worth filtering up front rather than letting the relaxation discover
    it: a 200mm spoon turned across a 78mm bin cannot be packed at any
    position, so an attempt that picks that orientation is doomed before
    its first iteration. Orientations 0 and 2 share a bounding box, as do 1
    and 3, so this usually returns four or two, and returns none exactly
    when the part cannot fit this bin in any orientation.
    """
    fitting = []
    for orientation in range(4):
        size = RotatedSize(part.size, orientation)
        if size[0] + 2 * params.c_wall <= container.width and size[1] + 2 * params.c_wall <= container.height:
            fitting.append(orientation)
    return fitting


def _ChooseOrientations(
    fitting: dict[int, list[int]],
    rng: np.random.Generator,
    attempt: int,
) -> dict[int, int]:
    """Quarter-turn orientation per part, drawn from those that fit.

    Orientation is discrete, so the relaxation cannot explore it - no
    torque acts on a part that can only sit at four angles. The restart
    loop is the only thing that ever varies it, which is why the first
    attempt takes the parts as they came where it can (PCA-aligned, long
    axis along x, which already suits a bin wider than it is tall) and
    later attempts randomize.
    """
    if attempt == 0:
        return {part_id: 0 if 0 in options else options[0] for part_id, options in fitting.items()}
    return {part_id: options[int(rng.integers(len(options)))] for part_id, options in fitting.items()}


def _PositionRange(
    part: Part,
    orientation: int,
    container: Container,
    params: LayoutParameters,
) -> tuple[np.ndarray, np.ndarray]:
    """Corner positions that keep a part's bounding box inside the bin with
    wall clearance to spare. The upper bound falls below the lower one for
    a part too big for this bin, which the caller handles.
    """
    size = RotatedSize(part.size, orientation)
    low = np.array([params.c_wall, params.c_wall])
    high = np.array([container.width, container.height]) - size - params.c_wall
    return low, high


def _Bounds(part: Part, placement: Placement) -> tuple[np.ndarray, np.ndarray]:
    """A placed part's axis-aligned bounding box. Exact, because rotation
    keeps a part's box anchored at its own origin.
    """
    return placement.position, placement.position + RotatedSize(part.size, placement.orientation)


def _ContactPositions(
    size: np.ndarray,
    container: Container,
    parts: dict[int, Part],
    placements: dict[int, Placement],
    params: LayoutParameters,
) -> list[np.ndarray]:
    """Positions where the part would sit flush against a wall or against
    an already-placed part, on each axis independently.

    This is what makes bottom-left-fill work on a crowded bin. Sampling
    positions uniformly at random cannot: once parts fill most of the
    interior, the feasible region is a vanishing fraction of the whole, and
    a random point essentially never lands in it - measured at 0 hits in 30
    attempts on a bin that packs comfortably by hand, and no better with
    the candidate budget raised sixteen-fold. Snapping to contacts searches
    where solutions actually are, and there are only `O(n^2)` of them.
    """
    high = np.array([container.width, container.height]) - size - params.c_wall
    if (high < params.c_wall).any():
        return []

    # Contacts are offset a little further than the clearance strictly
    # requires. A part placed at exactly c_pair reads as *violating* it:
    # the distance field is rasterized, so its measured separation is off
    # by up to the discretization error, and a hand-built perfect packing
    # prices at a small positive energy rather than zero. Without this
    # margin every contact position looks infeasible and the sweep falls
    # through to its random tail - which was the whole problem BLF is here
    # to solve.
    margin = params.resolution

    axes: list[list[float]] = [[params.c_wall + margin], [params.c_wall + margin]]
    for other_id, placement in placements.items():
        low_edge, high_edge = _Bounds(parts[other_id], placement)
        for axis in range(2):
            axes[axis].append(high_edge[axis] + params.c_pair_enforced + margin)  # just past it
            axes[axis].append(low_edge[axis] - size[axis] - params.c_pair_enforced - margin)  # just before it

    # Clamp into the legal range and drop duplicates, which are common once
    # several parts share an edge.
    clamped = [sorted({float(np.clip(value, params.c_wall, high[axis])) for value in axes[axis]}) for axis in range(2)]
    return [np.array([x, y]) for y in clamped[1] for x in clamped[0]]


def _NearbyPlacements(
    low: np.ndarray,
    high: np.ndarray,
    parts: dict[int, Part],
    placements: dict[int, Placement],
    params: LayoutParameters,
) -> dict[int, Placement]:
    """The already-placed parts whose bounding boxes come within the pair
    clearance of a candidate box.

    Anything further away contributes exactly zero energy, so dropping it
    is not an approximation - it is the difference between pricing a
    candidate against every part placed so far and against the two or three
    that could possibly matter.
    """
    return {
        other_id: placement
        for other_id, placement in placements.items()
        for other_low, other_high in [_Bounds(parts[other_id], placement)]
        if (low - params.c_pair_enforced < other_high).all() and (high + params.c_pair_enforced > other_low).all()
    }


def _ConstructiveInit(
    parts: dict[int, Part],
    orientations: dict[int, int],
    container: Container,
    params: LayoutParameters,
    rng: np.random.Generator,
    attempt: int = 0,
) -> dict[int, Placement]:
    """Place parts largest-first, each at the first spot that does not
    collide with what is already down.

    Largest-first is the standard nesting heuristic and the right one here:
    big parts are the constrained ones, and a bin that cannot take them
    cannot take them in any order.

    Candidates are contact positions first, swept from one corner, then a
    handful of random ones. The random tail is not vestigial - contacts are
    derived from bounding boxes, so they find shelf-like packings but never
    think to tuck a spoon's bowl into the crook of another's handle, which
    is exactly the nesting the concave shapes are there for.

    Falls back to the least-bad candidate if none is free: a part with
    nowhere to go is the relaxation's problem, not a reason to give up
    before it has run.
    """
    placements: dict[int, Placement] = {}

    # Which corner to sweep from. The first attempt sweeps from the origin,
    # making it the textbook bottom-left fill; later attempts pick a corner
    # at random so restarts explore genuinely different packings instead of
    # re-deriving the same one.
    if attempt == 0:
        corner = np.array([1.0, 1.0])
    else:
        corner = np.where(rng.random(2) < 0.5, 1.0, -1.0)

    for part_id in sorted(parts, key=lambda i: -parts[i].area):
        orientation = orientations[part_id]
        size = RotatedSize(parts[part_id].size, orientation)
        low_limit, high_limit = _PositionRange(parts[part_id], orientation, container, params)

        candidates = _ContactPositions(size, container, parts, placements, params)
        candidates.sort(key=lambda position: (corner[1] * position[1], corner[0] * position[0]))
        if (high_limit > low_limit).all():
            candidates += [rng.uniform(low_limit, high_limit) for _ in range(params.placement_tries)]
        if not candidates:
            # Too big for this bin at this orientation; pin it to the corner
            # and let the restart loop find that out.
            candidates = [low_limit]

        best = Placement(part_id, candidates[0], orientation)
        best_energy = np.inf

        for position in candidates:
            candidate = Placement(part_id, position, orientation)
            nearby = _NearbyPlacements(position, position + size, parts, placements, params)
            energy = PlacementEnergy(part_id, parts, {**nearby, part_id: candidate}, container, params)

            if energy <= 0.0:
                best = candidate
                break
            if energy < best_energy:
                best_energy, best = energy, candidate

        placements[part_id] = best

    return placements


def Relax(
    parts: dict[int, Part],
    placements: dict[int, Placement],
    container: Container,
    params: LayoutParameters,
    rng: np.random.Generator,
) -> dict[int, Placement] | None:
    """Damped descent with decaying noise, from a given starting
    arrangement. Returns the settled placements, or None if the attempt ran
    out of iterations or wandered somewhere the forces cannot be trusted.

    Exposed rather than private so a test can start it from a deliberately
    bad arrangement - stacked parts, say - and confirm it either fixes them
    or gives up, and never calls them done.
    """
    positions = {part_id: placement.position.astype(np.float64).copy() for part_id, placement in placements.items()}
    orientations = {part_id: placement.orientation for part_id, placement in placements.items()}
    velocities = {part_id: np.zeros(2) for part_id in placements}
    # Force scales with how many samples a part has, which is a property of
    # its size and the raster resolution rather than of how badly it is
    # placed. Dividing it out keeps one step size meaningful for a spoon
    # and a washer alike.
    per_sample = {part_id: 1.0 / max(1, len(parts[part_id].samples)) for part_id in placements}
    best_energy = np.inf
    stalled = 0

    for iteration in range(params.iterations):
        current = {part_id: Placement(part_id, positions[part_id], orientations[part_id]) for part_id in placements}
        result = ComputeEnergy(parts, current, container, params)

        if result.feasible:
            return current
        if result.engulfed:
            # A swallowed part is being pushed toward whichever wall of its
            # captor is nearest, which is not out. Descending from here is
            # worse than starting over.
            return None

        # Measured against the best seen rather than the previous step,
        # because the jitter makes energy fluctuate upward all the time and
        # a step-to-step test would read noise as progress.
        if result.energy < best_energy * (1.0 - 1e-3):
            best_energy = result.energy
            stalled = 0
        else:
            stalled += 1
            if stalled >= params.patience:
                return None

        cooling = 1.0 - iteration / params.iterations
        for part_id in placements:
            velocities[part_id] = (
                params.damping * velocities[part_id] + params.step_scale * result.forces[part_id] * per_sample[part_id]
            )
            move = velocities[part_id] + rng.normal(scale=params.jitter * cooling, size=2)

            distance = float(np.linalg.norm(move))
            if distance > params.max_step:
                move = move * (params.max_step / distance)
            positions[part_id] = positions[part_id] + move

    return None


def SolveFixedGrid(
    parts: dict[int, Part],
    n: int,
    m: int,
    params: LayoutParameters | None = None,
    on_attempt: Callable[[int], None] | None = None,
    cancelled: Callable[[], bool] | None = None,
) -> Layout | None:
    """Arrange every part inside an `n x m` bin, or return None if this many
    attempts could not.

    None means "not found", not "impossible" - the search is stochastic, so
    a failure here is only evidence. M4's bounds are what establish that a
    grid size is genuinely too small.

    `on_attempt` is called with each restart's index before it runs. The
    restart loop is where essentially all the time goes, so it is the only
    hook frequent enough to be worth reporting - and reporting it is the
    difference between a window that is busy and a window that looks hung.

    `cancelled` is polled at the same point. A cancelled search returns
    None like an exhausted one; telling the two apart is `Pack`'s job,
    since only it knows whether to step up to a larger bin or stop.
    """
    params = params or LayoutParameters()
    container = BuildContainer(n, m, params.inset)

    fitting = {part_id: FittingOrientations(part, container, params) for part_id, part in parts.items()}
    if not all(fitting.values()):
        # Some part does not fit this bin at any angle or position. No
        # amount of searching changes that, so say so now rather than
        # spending the whole restart budget rediscovering it.
        return None

    for attempt in range(params.restarts):
        if cancelled is not None and cancelled():
            return None
        if on_attempt is not None:
            on_attempt(attempt)

        # Seeded per attempt rather than drawn from one long stream, so an
        # attempt reproduces on its own and raising the restart budget does
        # not renumber the attempts that came before it.
        rng = np.random.default_rng([params.seed, attempt])

        orientations = _ChooseOrientations(fitting, rng, attempt)
        settled = Relax(
            parts,
            _ConstructiveInit(parts, orientations, container, params, rng, attempt),
            container,
            params,
            rng,
        )
        if settled is None:
            continue

        layout = Layout(grid=(n, m), placements=settled, inset=params.inset)
        # Zero energy already implies no overlap - every sample is at least
        # c_pair from every other part, which is strictly stronger. This
        # re-checks against exact polygon geometry anyway: it costs one test
        # per success, and the alternative to catching a bad layout here is
        # discovering it in a printed bin.
        if CheckLayout(layout, parts):
            continue
        return layout

    return None
