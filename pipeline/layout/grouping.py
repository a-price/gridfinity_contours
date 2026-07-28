"""Partitioning parts across several bins, rather than arranging one.

Everything below M7 answers "where do these parts go in this bin". This
answers "which parts share a bin at all", with the arrangement packer as
its feasibility oracle: a candidate grouping is only as good as the bins it
implies, and the only way to know a bin's cell count is to pack it.

The objective is total cells, summed over bins. Nothing here optimizes
arrangement quality - that is `spacing.Spread`'s job, and it runs inside
every pack this calls.

The search is the standard one for bin packing, in three stages:

1. Each part alone in its own bin, which is the baseline every later
   stage has to beat and the honest denominator for any claim of
   improvement (`OnePerBin`).
2. First-fit-decreasing: parts largest-first, each into the first open bin
   that still packs *at its current size* (`FirstFit`).
3. Local search: move a part between bins, or swap two, keeping whatever
   lowers the total (`Improve`).

Stage 2 deliberately never grows a bin. A part that only fits once its
bin gets larger is not a fit, it is a trade - the bin gets more expensive
in exchange for one fewer bin elsewhere - and that trade is exactly what
stage 3 is set up to price. Letting first-fit make it greedily, in
whatever order the parts happen to arrive, would commit to it without ever
comparing it against the alternative.

What makes this affordable is that most candidates never reach the solver.
Packing a set costs seconds; proving a set cannot fit costs microseconds,
because `packer.ProvablyTooSmall` answers from areas and bounding boxes
alone. Every candidate move is priced against that bound first and
abandoned if even the bound cannot beat what it would replace. Whatever
survives is packed once and remembered - the search revisits the same sets
constantly, since moving a part out of a bin and back in asks about a set
already priced.
"""

from dataclasses import dataclass
from itertools import combinations, permutations
from typing import Callable

from pipeline.layout.container import BuildContainer
from pipeline.layout.packer import GridsFor, ProvablyTooSmall
from pipeline.layout.parameters import LayoutParameters
from pipeline.layout.part import Part
from pipeline.layout.placement import Layout
from pipeline.layout.solver import SolveFixedGrid


@dataclass(frozen=True)
class Grouping:
    """Parts partitioned into bins, each with the layout that holds it.

    A bin is just its `Layout` - the parts in it are the keys of
    `layout.placements`, so there is no second list of ids to fall out of
    step with the arrangement it describes.
    """

    bins: list[Layout]

    @property
    def cells(self) -> int:
        """Total grid cells across every bin - the quantity being
        minimized.
        """
        return sum(layout.cells for layout in self.bins)

    def Contents(self) -> list[frozenset[int]]:
        """Which parts ended up in each bin."""
        return [frozenset(layout.placements) for layout in self.bins]

    def PartIds(self) -> frozenset[int]:
        """Every part placed, across all bins."""
        return frozenset(part_id for layout in self.bins for part_id in layout.placements)

    def Report(self) -> str:
        lines = []
        for index, layout in enumerate(self.bins):
            contents = ", ".join(str(part_id) for part_id in sorted(layout.placements))
            n, m = layout.grid
            lines.append(f"bin {index}: {n}x{m} ({layout.cells} cells) holding {contents}")
        lines.append(f"{len(self.bins)} bins, {self.cells} cells total")
        return "\n".join(lines)


FILLING = "filling"
IMPROVING = "improving"


@dataclass(frozen=True)
class Step:
    """One step of the grouping search, for a caller that wants to watch it.

    `bins` is the grouping as it stands *at this moment*, and `asking`
    indexes into it - the bins whose contents the search is about to price.
    Reported before the pricing rather than after, which is what makes a
    rejected candidate drawable at all: most are turned away by the bound
    without ever being packed, so there are no layouts of the rejected
    arrangement to show. What can honestly be drawn is the grouping that
    exists and the question being asked about it.

    `accepted` marks the steps where the grouping actually changed. Those
    carry an empty `asking`, because applying a change can empty a bin and
    drop it, renumbering everything after it - an index bound to the old
    list would point at the wrong bin in the new one.
    """

    phase: str
    bins: tuple[Layout, ...]
    asking: frozenset[int]
    accepted: bool

    @property
    def cells(self) -> int:
        return sum(layout.cells for layout in self.bins)

    def __str__(self) -> str:
        mark = "took" if self.accepted else "tried"
        return f"{self.phase}: {mark} {sorted(self.asking)}, {len(self.bins)} bins / {self.cells} cells"


Observer = Callable[[Step], None]


def _Report(observer: Observer | None, phase: str, bins: list[Layout], asking, accepted: bool) -> None:
    if observer is not None:
        observer(Step(phase, tuple(bins), frozenset(asking), accepted))


class _Oracle:
    """Packing answers about subsets of one fixed set of parts.

    Every answer is memoized, and the bounds run in front of the solver.
    Both matter for the same reason: the search asks the same questions
    over and over, and the ones it can answer cheaply outnumber the ones it
    cannot by a wide margin.

    `solver_calls` counts how often that fell through to an actual search,
    which is what a test needs to tell pruning from luck.
    """

    def __init__(self, parts: dict[int, Part], params: LayoutParameters):
        self._parts = parts
        self._params = params
        self._fitted: dict[tuple[frozenset[int], tuple[int, int]], Layout | None] = {}
        self._smallest: dict[frozenset[int], Layout | None] = {}
        self._bounds: dict[frozenset[int], int | None] = {}
        self.solver_calls = 0

    def Unfittable(self, part_id: int) -> ValueError:
        """The error for a part no allowed bin size can hold."""
        return _Unfittable(part_id, self._params)

    def _Subset(self, ids: frozenset[int]) -> dict[int, Part]:
        return {part_id: self._parts[part_id] for part_id in sorted(ids)}

    def FitsIn(self, ids: frozenset[int], grid: tuple[int, int]) -> Layout | None:
        """An arrangement of exactly these parts in a bin of exactly this
        size, or None if none was found.

        None is "not found", not "impossible" - the solver is stochastic,
        and only `ProvablyTooSmall` establishes impossibility. Caching them
        together is still right: the search must not depend on how many
        times it happened to ask.
        """
        key = (ids, grid)
        if key not in self._fitted:
            self._fitted[key] = self._Solve(ids, grid)
        return self._fitted[key]

    def Smallest(self, ids: frozenset[int]) -> Layout | None:
        """The fewest-cell bin these parts pack into, or None if nothing up
        to `max_grid` holds them.

        Built on `FitsIn` rather than on `packer.Pack` so that both share
        one cache. The grouping search asks "does this set fit this size"
        and "what is this set's smallest size" about the same sets
        constantly, and answering the second re-answers the first.
        """
        if ids not in self._smallest:
            self._smallest[ids] = self._Pack(ids)
        return self._smallest[ids]

    def LowerBoundCells(self, ids: frozenset[int]) -> int | None:
        """Fewest cells a bin holding these parts could conceivably use, or
        None if no size up to `max_grid` could.

        The whole point is that this runs no solver: it is the smallest
        candidate grid the area and extent bounds do not rule out. Both
        bounds are one-sided, so a real packing can only come out at this
        size or larger - which is what lets the search discard a move on
        the bound alone.

        An empty bin is zero cells rather than impossible; the local search
        empties bins, and one that has been emptied is one bin fewer.
        """
        if not ids:
            return 0
        if ids not in self._bounds:
            self._bounds[ids] = self._Bound(ids)
        return self._bounds[ids]

    def Cells(self, ids: frozenset[int]) -> int | None:
        """What a bin holding exactly these parts actually costs, packing
        it if need be. None if they do not fit any size.
        """
        if not ids:
            return 0
        layout = self.Smallest(ids)
        return layout.cells if layout is not None else None

    def _Solve(self, ids: frozenset[int], grid: tuple[int, int]) -> Layout | None:
        if not ids:
            return None
        n, m = grid
        subset = self._Subset(ids)
        if ProvablyTooSmall(subset, BuildContainer(n, m, self._params.inset), self._params) is not None:
            return None
        self.solver_calls += 1
        return SolveFixedGrid(subset, n, m, self._params)

    def _Pack(self, ids: frozenset[int]) -> Layout | None:
        for grid in GridsFor(self._params):
            layout = self.FitsIn(ids, grid)
            if layout is not None:
                return layout
        return None

    def _Bound(self, ids: frozenset[int]) -> int | None:
        subset = self._Subset(ids)
        for n, m in GridsFor(self._params):
            if ProvablyTooSmall(subset, BuildContainer(n, m, self._params.inset), self._params) is None:
                return n * m
        return None


def _Unfittable(part_id: int, params: LayoutParameters) -> ValueError:
    """Why a part could not be given a bin of its own.

    Names the restriction when there is one: with `admissible_grids` set
    the reachable sizes are whatever the drawers can hold, and "does not
    fit any size" is a confusing thing to read about a part that would
    have fitted perfectly well in a bin nobody can store.
    """
    if params.admissible_grids is None:
        return ValueError(f"part {part_id} does not fit a bin of any size up to {params.max_grid}x{params.max_grid}")
    return ValueError(
        f"part {part_id} does not fit any of the {len(params.admissible_grids)} footprints currently "
        "allowed - the admissible set is restricted, so a bin that would hold it may exist but not be storable"
    )


def _OnePerBin(oracle: _Oracle, part_ids: list[int]) -> list[Layout]:
    bins = []
    for part_id in part_ids:
        layout = oracle.Smallest(frozenset([part_id]))
        if layout is None:
            raise oracle.Unfittable(part_id)
        bins.append(layout)
    return bins


def _FirstFit(oracle: _Oracle, parts: dict[int, Part], observer: Observer | None = None) -> list[Layout]:
    """Parts largest-first, each into the first open bin that still packs
    at that bin's current size.

    Largest-first, since the big parts are the constrained ones - the
    same heuristic and the same reason as `solver._ConstructiveInit`.
    """
    bins: list[Layout] = []
    for part_id in sorted(parts, key=lambda i: -parts[i].area):
        for index, layout in enumerate(bins):
            _Report(observer, FILLING, bins, [index], accepted=False)
            grown = oracle.FitsIn(frozenset(layout.placements) | {part_id}, layout.grid)
            if grown is not None:
                bins[index] = grown
                _Report(observer, FILLING, bins, [], accepted=True)
                break
        else:
            opened = oracle.Smallest(frozenset([part_id]))
            if opened is None:
                raise oracle.Unfittable(part_id)
            bins.append(opened)
            _Report(observer, FILLING, bins, [], accepted=True)
    return bins


def _Total(values: list[int | None]) -> int | None:
    """The sum, or None if any part of it was impossible - a grouping with
    one unpackable bin has no cost, not an infinite one.
    """
    total = 0
    for value in values:
        if value is None:
            return None
        total += value
    return total


def _Rebuild(oracle: _Oracle, bins: list[Layout], changes: dict[int, frozenset[int]]) -> list[Layout] | None:
    """The bin list with some bins' contents replaced, or None if any of
    them no longer packs. Emptied bins drop out.
    """
    updated: list[Layout | None] = list(bins)
    for index, ids in changes.items():
        if not ids:
            updated[index] = None
            continue
        layout = oracle.Smallest(ids)
        if layout is None:
            return None
        updated[index] = layout
    return [layout for layout in updated if layout is not None]


def _Improvement(oracle: _Oracle, bins: list[Layout], changes: dict[int, frozenset[int]]) -> list[Layout] | None:
    """The result of applying `changes`, if it costs strictly less than
    what it replaces.

    The bound check is what makes the search affordable, and it is sound
    only because `LowerBoundCells` never overestimates: if even the bound
    cannot beat the current cost, no packing of these sets can, and the
    solver never has to be asked.
    """
    before = sum(bins[index].cells for index in changes)

    bound = _Total([oracle.LowerBoundCells(ids) for ids in changes.values()])
    if bound is None or bound >= before:
        return None

    cost = _Total([oracle.Cells(ids) for ids in changes.values()])
    if cost is None or cost >= before:
        return None

    return _Rebuild(oracle, bins, changes)


def _Candidates(contents: list[frozenset[int]]):
    """Every move and swap available from this arrangement of bins.

    Moves first: a move can remove a bin outright, which is the largest
    single improvement available, and finding it early cuts the work the
    swaps then have to do.
    """
    for source, target in permutations(range(len(contents)), 2):
        for part_id in sorted(contents[source]):
            yield {source: contents[source] - {part_id}, target: contents[target] | {part_id}}

    for first, second in combinations(range(len(contents)), 2):
        for one in sorted(contents[first]):
            for other in sorted(contents[second]):
                yield {
                    first: contents[first] - {one} | {other},
                    second: contents[second] - {other} | {one},
                }


def _Improve(oracle: _Oracle, bins: list[Layout], observer: Observer | None = None) -> list[Layout]:
    """Apply improving moves and swaps until none is left.

    First improvement rather than best: the candidates are cheap to
    generate and expensive to price, so taking the first one that helps
    and re-generating beats pricing them all to pick the best.

    This terminates without an iteration cap because every accepted change
    strictly lowers the total cell count, which is a non-negative integer.
    """
    while True:
        contents = [frozenset(layout.placements) for layout in bins]
        for changes in _Candidates(contents):
            _Report(observer, IMPROVING, bins, changes, accepted=False)
            improved = _Improvement(oracle, bins, changes)
            if improved is not None:
                bins = improved
                _Report(observer, IMPROVING, bins, [], accepted=True)
                break
        else:
            return bins


def _RequireParts(parts: dict[int, Part]) -> None:
    if not parts:
        raise ValueError("nothing to group")


def OnePerBin(parts: dict[int, Part], params: LayoutParameters | None = None) -> Grouping:
    """Every part in a bin of its own - the baseline to measure against.

    Worth having as a function rather than a number in a comment: it is
    what grouping has to beat, and a claimed improvement against a figure
    nobody recomputes is how a regression hides.
    """
    _RequireParts(parts)
    return Grouping(_OnePerBin(_Oracle(parts, params or LayoutParameters()), sorted(parts)))


def FirstFit(
    parts: dict[int, Part],
    params: LayoutParameters | None = None,
    observer: Observer | None = None,
) -> Grouping:
    """First-fit-decreasing alone, without the local search.

    Exposed so a test can see what the search inherits and what it adds. A
    grouping that only ever ran both together could not tell an improving
    local search from one that never fires.
    """
    _RequireParts(parts)
    return Grouping(_FirstFit(_Oracle(parts, params or LayoutParameters()), parts, observer))


def Improve(
    parts: dict[int, Part],
    grouping: Grouping,
    params: LayoutParameters | None = None,
    observer: Observer | None = None,
) -> Grouping:
    """Move and swap parts between the given bins while it helps.

    Takes a grouping rather than producing one, so the search can be run
    on any starting point - including one per bin, which is the case where
    it has the most to find.
    """
    missing = sorted(grouping.PartIds() - set(parts))
    if missing:
        raise ValueError(f"grouping holds parts {missing}, which were not given")
    return Grouping(_Improve(_Oracle(parts, params or LayoutParameters()), list(grouping.bins), observer))


def Group(
    parts: dict[int, Part],
    params: LayoutParameters | None = None,
    observer: Observer | None = None,
) -> Grouping:
    """Partition parts into bins, minimizing total grid cells.

    First-fit-decreasing for a starting point, then local search. One
    oracle spans both, so the sets first-fit already priced cost the search
    nothing to revisit.

    `observer`, if given, sees both phases - see `Step`.
    """
    _RequireParts(parts)
    oracle = _Oracle(parts, params or LayoutParameters())
    return Grouping(_Improve(oracle, _FirstFit(oracle, parts, observer), observer))
