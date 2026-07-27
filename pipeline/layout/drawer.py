"""Which bins share a drawer.

The level above grouping, and the first one whose answers are exact.
Everything here is an integer count of grid cells: a drawer is `W x H`
cells, a bin is an `n x m` footprint, and the question is whether the bins
tile into the drawers without overlapping.

**There are no clearances at this level, and nothing to tune.** That is a
property of the Gridfinity spec rather than a simplification. A bin's
outer footprint is `42*n - 0.5`mm, and that half millimeter - the gap that
keeps neighbouring bins from binding - is already inside the footprint. So
bins abut exactly on the 42mm lattice and need nothing between them.
Compare every level below this one, where a clearance has to be derived,
tuned, and eventually checked against a physical print.

That is also why this level can be *exact*. The arrangement solver is
stochastic, so its failures mean "not found"; grouping inherits that,
since its cost function is that solver. Here an exhaustive search can
prove a set of bins does not fit, as a fact. `INFEASIBLE` therefore means
proven, and the search reports `EXHAUSTED` rather than stretching the word
when it runs out of budget - a distinction with the same purpose as
`packer`'s split between TOO_SMALL and NOT_FOUND.

The search is complete because it only ever considers **bottom-left
stable** positions: a bin goes where it cannot slide one cell further left
or down. Any packing can be normalized to that form by repeatedly pushing
rectangles left and down, which terminates and never introduces an
overlap, so restricting to stable positions cannot lose a solution. It is
what makes an exhaustive search tractable.

It also *tends* to leave the free space in one piece, since everything is
pushed into a corner - but it does not guarantee it, and an earlier draft
of the design claimed it did. Measured on a realistic drawer, 143 of the
144 free cells came out connected and one was stranded behind a 1x1 bin.
Contiguity is therefore reported (`LargestFreeRegion`) rather than
optimized for: turning it into an objective would mean enumerating every
complete assignment instead of stopping at the first, which is the early
exit the whole search's speed rests on.

Occupancy is one integer per drawer: a row-major bitmask, one bit per
cell, so "does this bin fit here" is a shift and an AND. Python's integers
are arbitrary precision, so this holds at any drawer size - a 500 x 750mm
drawer is 11 x 17 cells, past a machine word and still a single value.
"""

from dataclasses import dataclass, field
from typing import Iterator, Sequence

from pipeline.layout.container import BASE_GAP_MM, GRID_PITCH_MM

PLACED = "placed"
INFEASIBLE = "infeasible"
EXHAUSTED = "exhausted"

# How many search nodes to spend before giving up. Reached only by
# genuinely hard instances: a drawer that packs comfortably is solved in
# hundreds of nodes, and the bounds turn most hopeless ones away without
# searching at all. The point of having a limit is that an unbounded
# search would eventually have to be killed by hand, and a killed search
# tells you nothing - whereas EXHAUSTED at least says which question went
# unanswered.
DEFAULT_NODE_BUDGET = 200_000


@dataclass(frozen=True)
class Drawer:
    """A drawer's usable interior, in whole grid cells."""

    width: int
    height: int

    def __post_init__(self):
        if self.width < 1 or self.height < 1:
            raise ValueError(f"a drawer must be at least 1x1 cells, got {self.width}x{self.height}")

    @property
    def cells(self) -> int:
        return self.width * self.height

    def Holds(self, footprint: tuple[int, int]) -> bool:
        """Whether a bin of this footprint fits at all, at either quarter
        turn. Says nothing about what else is already in here.
        """
        n, m = footprint
        return (n <= self.width and m <= self.height) or (m <= self.width and n <= self.height)


def DrawerCells(width_mm: float, height_mm: float) -> Drawer:
    """The whole grid cells a drawer of this interior size holds.

    A run of `n` cells spans `42*n - 0.5`mm, not `42*n`: the half
    millimeter gap comes off the run as a whole, not off each bin, because
    it is already inside every bin's own footprint. Naive floor division
    would deny a 41.5mm drawer the single cell that genuinely fits it.
    """
    if width_mm <= 0 or height_mm <= 0:
        raise ValueError(f"a drawer must have positive size, got {width_mm}x{height_mm}mm")

    width = int((width_mm + BASE_GAP_MM) // GRID_PITCH_MM)
    height = int((height_mm + BASE_GAP_MM) // GRID_PITCH_MM)
    if width < 1 or height < 1:
        raise ValueError(f"a {width_mm}x{height_mm}mm drawer holds no whole {GRID_PITCH_MM}mm cell")
    return Drawer(width, height)


@dataclass(frozen=True)
class Slot:
    """Where one bin sits: which drawer, which cell is its minimum corner,
    and whether it was turned a quarter turn to get there.

    `cell` is in grid cells from the drawer's own minimum corner, so it
    multiplies straight back to millimeters by the pitch.
    """

    bin_id: int
    drawer: int
    cell: tuple[int, int]
    turned: bool = False

    def Footprint(self, footprint: tuple[int, int]) -> tuple[int, int]:
        """That bin's footprint as it sits, quarter turn applied."""
        n, m = footprint
        return (m, n) if self.turned else (n, m)


@dataclass(frozen=True)
class AssignmentResult:
    """Where every bin went, or why they could not all go anywhere.

    `outcome` separates the three answers that must never be conflated.
    PLACED means every bin is in `slots`. INFEASIBLE means the search
    finished and no arrangement exists - a fact about the geometry, and the
    one that justifies re-grouping. EXHAUSTED means the budget ran out with
    the question still open, which is not evidence about anything.
    """

    outcome: str
    slots: dict[int, Slot] = field(default_factory=dict)
    unplaced: list[int] = field(default_factory=list)
    detail: str = ""

    @property
    def placed(self) -> bool:
        return self.outcome == PLACED

    def Report(self, footprints: dict[int, tuple[int, int]] | None = None) -> str:
        lines = []
        for bin_id, slot in sorted(self.slots.items()):
            x, y = slot.cell
            size = ""
            if footprints is not None:
                n, m = slot.Footprint(footprints[bin_id])
                size = f" {n}x{m}"
            turned = ", turned" if slot.turned else ""
            lines.append(f"bin {bin_id}{size} -> drawer {slot.drawer} at cell ({x}, {y}){turned}")

        if self.placed:
            lines.append(f"all {len(self.slots)} bins placed")
        else:
            left = ", ".join(str(bin_id) for bin_id in self.unplaced)
            lines.append(f"{self.outcome}: could not place bins {left}")
            if self.detail:
                lines.append(self.detail)
        return "\n".join(lines)


def AdmissibleFootprints(drawers: Sequence[Drawer], max_grid: int) -> frozenset[tuple[int, int]]:
    """The bin footprints some drawer could hold, for feeding back into
    the grid size search.

    A bin 7 cells long cannot go in a drawer 6 cells wide at any angle,
    however few cells it uses - so proposing one wastes the whole stack
    below. This is the predicate `packer.CandidateGrids` takes.
    """
    return frozenset(
        (n, m)
        for n in range(1, max_grid + 1)
        for m in range(1, n + 1)
        if any(drawer.Holds((n, m)) for drawer in drawers)
    )


class _Exhausted(Exception):
    """The node budget ran out.

    Raised rather than returned so that a partial answer can never be
    mistaken for a completed search - the whole value of this level is
    that INFEASIBLE means proven.
    """


def _Mask(drawer: Drawer, x: int, y: int, width: int, height: int) -> int:
    """The cells a bin of `width x height` at `(x, y)` would occupy, as a
    row-major bitmask over the drawer.
    """
    row = ((1 << width) - 1) << x
    mask = 0
    for offset in range(height):
        mask |= row << ((y + offset) * drawer.width)
    return mask


def _Stable(drawer: Drawer, occupancy: int, x: int, y: int, width: int, height: int) -> bool:
    """Whether a bin here could not slide one cell further left or down.

    Restricting the search to these positions is what keeps it tractable
    without losing solutions: pushing every rectangle left and down
    terminates, never creates an overlap, and so normalizes any packing
    into one made entirely of stable positions.
    """
    if x > 0 and not (_Mask(drawer, x - 1, y, 1, height) & occupancy):
        return False
    if y > 0 and not (_Mask(drawer, x, y - 1, width, 1) & occupancy):
        return False
    return True


@dataclass
class _Context:
    footprints: list[tuple[int, int, int]]  # (bin id, n, m), largest first
    drawers: Sequence[Drawer]
    budget: int
    nodes: int = 0
    memo: dict = field(default_factory=dict)


def _Orientations(n: int, m: int) -> list[tuple[int, int, bool]]:
    """The footprint as given, plus its quarter turn when that differs."""
    return [(n, m, False)] if n == m else [(n, m, False), (m, n, True)]


def _Positions(context: _Context, index: int, state: tuple[int, ...]) -> Iterator[tuple[int, int, int, bool]]:
    """Every stable, non-overlapping spot the next bin could take, as
    `(drawer index, x, y, turned)`.
    """
    _, n, m = context.footprints[index]

    seen: set[tuple[int, int, int]] = set()
    for drawer_index, drawer in enumerate(context.drawers):
        # Two drawers of the same size holding the same thing are the same
        # drawer as far as this search is concerned, and trying both would
        # rediscover every answer once per interchangeable drawer.
        signature = (drawer.width, drawer.height, state[drawer_index])
        if signature in seen:
            continue
        seen.add(signature)

        occupancy = state[drawer_index]
        for width, height, turned in _Orientations(n, m):
            if width > drawer.width or height > drawer.height:
                continue
            for y in range(drawer.height - height + 1):
                for x in range(drawer.width - width + 1):
                    if _Mask(drawer, x, y, width, height) & occupancy:
                        continue
                    if _Stable(drawer, occupancy, x, y, width, height):
                        yield drawer_index, x, y, turned


def _Search(context: _Context, index: int, state: tuple[int, ...]) -> list[Slot]:
    """The most bins placeable from `index` onward, given what is already
    down. Raises `_Exhausted` if the budget runs out.
    """
    remaining = len(context.footprints) - index
    if remaining == 0:
        return []

    key = (index, state)
    if key in context.memo:
        return context.memo[key]

    context.nodes += 1
    if context.nodes > context.budget:
        raise _Exhausted()

    bin_id, n, m = context.footprints[index]
    best: list[Slot] = []

    for drawer_index, x, y, turned in _Positions(context, index, state):
        width, height = (m, n) if turned else (n, m)
        drawer = context.drawers[drawer_index]

        occupied = list(state)
        occupied[drawer_index] |= _Mask(drawer, x, y, width, height)

        rest = _Search(context, index + 1, tuple(occupied))
        if len(rest) + 1 > len(best):
            best = [Slot(bin_id, drawer_index, (x, y), turned)] + rest
            if len(best) == remaining:
                break  # everything left fits; nothing can beat that

    # Leaving this bin out, which only ever helps once placing it has been
    # shown not to lead anywhere better.
    if len(best) < remaining:
        rest = _Search(context, index + 1, state)
        if len(rest) > len(best):
            best = rest

    context.memo[key] = best
    return best


def _Impossible(footprints: dict[int, tuple[int, int]], drawers: Sequence[Drawer]) -> tuple[list[int], str] | None:
    """Why these bins provably cannot all be placed, or None if they might.

    Both tests are one-sided - they never reject an assignment that would
    have worked. That matters more here than anywhere below: an over-eager
    bound at this level silently reports "buy another drawer".
    """
    unplaceable = sorted(
        bin_id for bin_id, footprint in footprints.items() if not any(d.Holds(footprint) for d in drawers)
    )
    if unplaceable:
        sizes = ", ".join(f"{bin_id} ({footprints[bin_id][0]}x{footprints[bin_id][1]})" for bin_id in unplaceable)
        return unplaceable, f"no drawer is large enough for bin {sizes} at any quarter turn"

    required = sum(n * m for n, m in footprints.values())
    available = sum(drawer.cells for drawer in drawers)
    if required > available:
        return (
            sorted(footprints),
            f"bins need {required} cells, the drawers hold {available}",
        )
    return None


def Assign(
    footprints: dict[int, tuple[int, int]],
    drawers: Sequence[Drawer],
    budget: int = DEFAULT_NODE_BUDGET,
) -> AssignmentResult:
    """Fit every bin into the given drawers, or say why they do not fit.

    `footprints` maps a bin id to its `n x m` grid size - straight off a
    grouping, as `{i: layout.grid for i, layout in enumerate(grouping.bins)}`.
    Nothing about a bin's contents reaches this level, which is what keeps
    it exact.

    Bins may be turned a quarter turn, exactly as parts may be inside a
    bin.
    """
    if not footprints:
        raise ValueError("nothing to assign")
    if not drawers:
        raise ValueError("no drawers to assign into")
    for bin_id, (n, m) in sorted(footprints.items()):
        if n < 1 or m < 1:
            raise ValueError(f"bin {bin_id} has a {n}x{m} footprint; grid sizes are whole cells")

    impossible = _Impossible(footprints, drawers)
    if impossible is not None:
        unplaced, detail = impossible
        return AssignmentResult(INFEASIBLE, {}, unplaced, detail)

    # Largest first: the constrained bins are the big ones, and a drawer
    # that cannot take them cannot take them in any order.
    ordered = sorted(((bin_id, n, m) for bin_id, (n, m) in footprints.items()), key=lambda f: (-f[1] * f[2], f[0]))
    context = _Context(ordered, drawers, budget)

    try:
        slots = _Search(context, 0, tuple(0 for _ in drawers))
    except _Exhausted:
        return AssignmentResult(
            EXHAUSTED,
            {},
            sorted(footprints),
            f"gave up after {context.budget} search nodes; this is not evidence that the bins do not fit",
        )

    placed = {slot.bin_id: slot for slot in slots}
    if len(placed) == len(footprints):
        return AssignmentResult(PLACED, placed)

    unplaced = sorted(set(footprints) - set(placed))
    sizes = ", ".join(f"{bin_id} ({footprints[bin_id][0]}x{footprints[bin_id][1]})" for bin_id in unplaced)
    return AssignmentResult(
        INFEASIBLE,
        placed,
        unplaced,
        f"no arrangement fits every bin; bin {sizes} has to go somewhere else",
    )


def Occupancy(drawers: Sequence[Drawer], footprints: dict[int, tuple[int, int]], result: AssignmentResult) -> list[int]:
    """Each drawer's occupied cells, as the same row-major bitmask the
    search works in.
    """
    state = [0] * len(drawers)
    for bin_id, slot in result.slots.items():
        width, height = slot.Footprint(footprints[bin_id])
        x, y = slot.cell
        state[slot.drawer] |= _Mask(drawers[slot.drawer], x, y, width, height)
    return state


def FreeCells(drawers: Sequence[Drawer], footprints: dict[int, tuple[int, int]], result: AssignmentResult) -> list[int]:
    """Cells left over in each drawer.

    Reported per drawer rather than as a total, because "12 cells free"
    across three drawers is not the same offer as 12 in one.
    """
    return [
        drawer.cells - occupancy.bit_count()
        for drawer, occupancy in zip(drawers, Occupancy(drawers, footprints, result))
    ]


def LargestFreeRegion(
    drawers: Sequence[Drawer], footprints: dict[int, tuple[int, int]], result: AssignmentResult
) -> list[int]:
    """The biggest orthogonally-connected patch of free space in each
    drawer.

    The number that says whether the leftover room is *usable*. Free cells
    alone do not: six free cells in six separate places have room for
    nothing, and the next object photographed has to go somewhere.
    """
    regions = []
    for drawer, occupancy in zip(drawers, Occupancy(drawers, footprints, result)):
        free = {
            (x, y)
            for y in range(drawer.height)
            for x in range(drawer.width)
            if not (occupancy >> (y * drawer.width + x)) & 1
        }
        largest = 0
        while free:
            stack, region = [free.pop()], 1
            while stack:
                x, y = stack.pop()
                for neighbour in ((x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)):
                    if neighbour in free:
                        free.remove(neighbour)
                        stack.append(neighbour)
                        region += 1
            largest = max(largest, region)
        regions.append(largest)
    return regions
