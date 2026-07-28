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
from pipeline.layout.descent import RELAXING, SPREADING, Descent, Observer, Reporter, Reporting
from pipeline.layout.energy import ComputeEnergy, PlacementEnergy
from pipeline.layout.orientation import RankedAssignments
from pipeline.layout.parameters import LayoutParameters
from pipeline.layout.part import CanonicalOrder, Part
from pipeline.layout.placement import Layout, Placement, RotatedSize
from pipeline.layout.spacing import Distribute, Spread
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


def _ChooseOrientations(ranked: list[dict[int, int]], attempt: int) -> dict[int, int]:
    """The orientation assignment this attempt should start from.

    Orientation is discrete, so the relaxation cannot explore it - no
    torque acts on a part that can only sit at four angles. The restart
    loop is the only thing that ever varies it, so a bad choice is not a
    slow attempt but a doomed one, and choosing well is worth doing before
    the search rather than by drawing until something works.

    `ranked` is sorted best-first by `orientation.RankedAssignments`, so
    attempts walk it in order. Cycling once it runs out is not a repeat:
    `_ConstructiveInit` sweeps from a different corner and draws different
    positions on every attempt, so the same orientations get genuinely
    different starting arrangements.
    """
    return ranked[attempt % len(ranked)]


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

    for part_id in CanonicalOrder(parts):
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
    on_step: Reporter | None = None,
) -> dict[int, Placement] | None:
    """Damped descent with decaying noise, from a given starting
    arrangement. Returns the settled placements, or None if the attempt ran
    out of iterations or wandered somewhere the forces cannot be trusted.

    Exposed rather than private so a test can start it from a deliberately
    bad arrangement - stacked parts, say - and confirm it either fixes them
    or gives up, and never calls them done.

    `on_step` sees every iteration, reported before the outcome is decided
    so that the arrangement it settles on is the last thing reported rather
    than one step short of it.
    """
    descent = Descent(parts, placements, params)
    best_energy = np.inf
    stalled = 0

    for iteration in range(params.iterations):
        current = descent.Placements()
        result = ComputeEnergy(parts, current, container, params)

        if on_step is not None:
            on_step(iteration, current, result.energy)

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
        descent.Step(result.forces, noise=params.jitter * cooling, rng=rng)

    return None


def SolveFixedGrid(
    parts: dict[int, Part],
    n: int,
    m: int,
    params: LayoutParameters | None = None,
    on_attempt: Callable[[int], None] | None = None,
    cancelled: Callable[[], bool] | None = None,
    observer: Observer | None = None,
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

    `observer`, if given, sees every iteration of both descent passes -
    orders of magnitude more events than `on_attempt`, and the level a
    caller drawing the search needs. This is where the grid size and
    attempt number get bound onto it, since neither pass knows them.
    """
    params = params or LayoutParameters()
    container = BuildContainer(n, m, params.inset)

    fitting = {part_id: FittingOrientations(part, container, params) for part_id, part in parts.items()}
    if not all(fitting.values()):
        # Some part does not fit this bin at any angle or position. No
        # amount of searching changes that, so say so now rather than
        # spending the whole restart budget rediscovering it.
        return None

    # Ranked once for the whole grid, not per attempt: the score depends on
    # the assignment as a whole, and the restart loop only walks the result.
    ranked = RankedAssignments(parts, fitting, container, params, params.restarts)

    for attempt in range(params.restarts):
        if cancelled is not None and cancelled():
            return None
        if on_attempt is not None:
            on_attempt(attempt)

        # Seeded per attempt rather than drawn from one long stream, so an
        # attempt reproduces on its own and raising the restart budget does
        # not renumber the attempts that came before it.
        rng = np.random.default_rng([params.seed, attempt])

        orientations = _ChooseOrientations(ranked, attempt)
        settled = Relax(
            parts,
            _ConstructiveInit(parts, orientations, container, params, rng, attempt),
            container,
            params,
            rng,
            Reporting(observer, (n, m), attempt, RELAXING),
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

        # Feasible, but the solver stopped at the first arrangement that
        # was - so the gaps are arbitrary. Even them out before returning.
        spread = Spread(parts, settled, container, params, Reporting(observer, (n, m), attempt, SPREADING))
        # Then out into whatever room is left. The springs only reach a
        # couple of millimetres past each clearance, so in a bin with space
        # to spare they go slack long before the parts are using it.
        spread = Distribute(parts, spread, container, params)
        balanced = Layout(grid=(n, m), placements=spread, inset=params.inset)
        return balanced if not CheckLayout(balanced, parts) else layout

    return None
