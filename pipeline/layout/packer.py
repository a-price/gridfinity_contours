"""Choosing the bin size, not just the arrangement inside one.

Candidate grids are tried smallest-area first, and the first that packs
wins. Most candidates never reach the solver at all: two cheap bounds
reject the ones that provably cannot hold the parts, which matters because
a hopeless call to SolveFixedGrid costs its entire restart budget to learn
nothing.

The distinction the report exists to preserve is between a size that was
*proven* too small and one the search merely failed on. The first is a
fact about the geometry; the second is a fact about the search having got
unlucky, and it is the reason a returned bin might be larger than
necessary. Conflating them would make an oversized result indistinguishable
from an unavoidable one.
"""

from dataclasses import dataclass, field
from typing import Callable, Collection

from pipeline.layout.container import BuildContainer, Container, GridSizes
from pipeline.layout.descent import Observer
from pipeline.layout.parameters import QUARTER_TURNS, LayoutParameters
from pipeline.layout.part import Part
from pipeline.layout.placement import Layout
from pipeline.layout.solver import FitsAtSomeAngle, FittingPoses, SolveFixedGrid

PACKED = "packed"
TOO_SMALL = "too small"
NOT_FOUND = "not found"
CANCELLED = "cancelled"


@dataclass(frozen=True)
class Progress:
    """Where the search has got to, for a caller that wants to show it.

    Reported per restart rather than per grid size, because the sizes that
    are cheap to reject are rejected instantly and the one that is not can
    take the entire restart budget. A per-grid report would sit unchanged
    for exactly the stretch a user most needs to see something moving.
    """

    grid: tuple[int, int]
    attempt: int  # zero-based restart within this grid
    restarts: int
    grids_tried: int  # candidate sizes considered so far, this one included

    def __str__(self) -> str:
        n, m = self.grid
        return f"{n}x{m}, attempt {self.attempt + 1}/{self.restarts} ({self.grids_tried} sizes tried)"


@dataclass(frozen=True)
class GridAttempt:
    """What became of one candidate grid size."""

    grid: tuple[int, int]
    outcome: str
    detail: str = ""

    @property
    def cells(self) -> int:
        return self.grid[0] * self.grid[1]

    def __str__(self) -> str:
        label = f"{self.grid[0]}x{self.grid[1]} ({self.cells} cells)"
        return f"{label}: {self.outcome}" + (f" - {self.detail}" if self.detail else "")


@dataclass(frozen=True)
class PackResult:
    """The chosen layout, if any, and the trail of everything tried.

    `layout` is None only when no candidate up to `max_grid` worked. A
    successful result can still be larger than optimal - see `skipped`.
    """

    layout: Layout | None
    attempts: list[GridAttempt] = field(default_factory=list)

    @property
    def cells(self) -> int | None:
        return self.layout.cells if self.layout else None

    @property
    def skipped(self) -> list[GridAttempt]:
        """Sizes the bounds allowed but the search could not pack.

        Non-empty means the result may be bigger than it had to be: those
        sizes were not ruled out by geometry, only by the solver running
        out of attempts. This is the trace that keeps an oversized bin
        traceable instead of silent.
        """
        return [attempt for attempt in self.attempts if attempt.outcome == NOT_FOUND]

    @property
    def cancelled(self) -> bool:
        """Whether the search was stopped rather than finished.

        Kept distinct from failure: a cancelled search says nothing about
        whether the parts fit, so nothing may conclude from it.
        """
        return any(attempt.outcome == CANCELLED for attempt in self.attempts)

    def Report(self) -> str:
        lines = [str(attempt) for attempt in self.attempts]
        if self.cancelled:
            lines.append("search cancelled - larger sizes were never tried")
        elif self.layout is None:
            lines.append("no grid size up to the configured maximum could hold these parts")
        elif self.skipped:
            smaller = ", ".join(f"{a.grid[0]}x{a.grid[1]}" for a in self.skipped)
            lines.append(f"note: {smaller} was not ruled out geometrically - a tighter packing may exist")
        return "\n".join(lines)


def CandidateGrids(max_grid: int, admissible: Collection[tuple[int, int]] | None = None) -> list[tuple[int, int]]:
    """Grid sizes worth trying, smallest first, squarest first among
    equals.

    Built on `container.GridSizes`, which is also what
    `drawer.AdmissibleFootprints` filters - the two have to enumerate the
    same `n >= m` convention, since this intersects a search against
    exactly what that says a set of drawers can hold.

    `admissible`, if given, keeps only footprints something downstream can
    accept - in practice the ones that fit a drawer, from
    `drawer.AdmissibleFootprints`. The rotation argument in `GridSizes`
    survives the restriction, since a bin turns in a drawer exactly as a
    part turns in a bin.
    """
    grids = GridSizes(max_grid)
    if admissible is not None:
        grids = [grid for grid in grids if grid in admissible]
    grids.sort(key=lambda grid: (grid[0] * grid[1], grid[0] - grid[1], grid[0]))
    return grids


def GridsFor(params: LayoutParameters) -> list[tuple[int, int]]:
    """The candidate grids this parameter set allows - its size cap and
    its admissible footprints together.

    A single place to ask, so the packer and the grouping search cannot
    end up enumerating different sets and disagreeing about what is
    packable.
    """
    return CandidateGrids(params.max_grid, params.admissible_grids)


def RequiredArea(parts: dict[int, Part], params: LayoutParameters) -> float:
    """Total area the parts claim once their clearance bands are counted.

    Each part is dilated by half the pair clearance, so two parts exactly
    `c_pair` apart have dilations that touch without overlapping. In any
    feasible layout the dilations are therefore disjoint, and their total
    cannot exceed the interior - which is what makes this a sound lower
    bound rather than a guess. Nesting never beats it.
    """
    return sum(part.DilatedArea(params.c_pair / 2.0) for part in parts.values())


def ProvablyTooSmall(parts: dict[int, Part], container: Container, params: LayoutParameters) -> str | None:
    """Why this bin cannot possibly hold these parts, or None if it might.

    Both tests are one-sided: they never reject a bin that could work, so a
    None here is genuinely "worth trying", and a reason here is final. The
    solver is only ever asked about bins that survive this.
    """
    for part_id in sorted(parts):
        part = parts[part_id]
        # Which question counts as proof depends on how freely parts may
        # turn, and getting this wrong is the one way this function can
        # cause harm. Under 90 and 45 the legal angles are a finite list, so
        # "fits at no candidate pose" *is* "fits at no legal angle". Under
        # FREE they are a continuum and that same check would reject bins a
        # part fits perfectly well on the diagonal - the knife misses a
        # six-cell bin by 0.7mm square-on and clears it at 3.9 degrees - so
        # the proof has to come from a bound that holds at every angle.
        if params.free_rotation:
            fits = FitsAtSomeAngle(part, container, params)
            detail = "at any angle"
        else:
            fits = bool(FittingPoses(part, container, params))
            detail = "at any quarter turn" if params.rotation == QUARTER_TURNS else "at any eighth turn"

        if not fits:
            size = part.size
            return (
                f"part {part_id} is {size[0]:.1f}x{size[1]:.1f}mm and does not fit a "
                f"{container.width:.1f}x{container.height:.1f}mm interior {detail}"
            )

    required = RequiredArea(parts, params)
    if required > container.area:
        return f"parts need {required:.0f}mm^2 with their clearances, interior holds {container.area:.0f}mm^2"
    return None


def _AttemptReporter(
    progress: Callable[[Progress], None] | None,
    grid: tuple[int, int],
    restarts: int,
    grids_tried: int,
) -> Callable[[int], None] | None:
    """Bind one grid size's context onto a progress callback.

    A factory rather than a closure written in the loop, so each grid's
    values are bound by argument passing instead of by whatever the loop
    variables happen to hold when the callback fires.
    """
    if progress is None:
        return None
    return lambda attempt: progress(Progress(grid, attempt, restarts, grids_tried))


def Pack(
    parts: dict[int, Part],
    params: LayoutParameters | None = None,
    progress: Callable[[Progress], None] | None = None,
    cancelled: Callable[[], bool] | None = None,
    observer: Observer | None = None,
) -> PackResult:
    """Fit every part into the smallest grid that will take them.

    Candidates are tried smallest-area first and the first success is
    returned, so the result is the smallest size *found* - not provably the
    smallest that exists, since the solver is stochastic. When a
    bounds-feasible size fails, the search steps up rather than giving up:
    a usable bin beats none, and `PackResult.skipped` records what was left
    behind so an oversized result stays traceable.

    `progress`, if given, is called as the search moves - see `Progress`.
    `observer` is the same idea at the other end of the frequency scale:
    one call per solver iteration, carrying the arrangement itself, for a
    caller drawing the search rather than reporting on it (see
    `descent.Snapshot`).

    `cancelled` is polled between and during grid sizes; a search stopped
    that way is recorded as CANCELLED rather than NOT_FOUND, because "you
    stopped me" is not evidence about the bin and must not be read as
    "this size might have worked".
    """
    params = params or LayoutParameters()
    if not parts:
        raise ValueError("nothing to pack")

    stopped = cancelled if cancelled is not None else lambda: False

    attempts: list[GridAttempt] = []
    for n, m in GridsFor(params):
        if stopped():
            attempts.append(GridAttempt((n, m), CANCELLED, "stopped before this size was tried"))
            break

        container = BuildContainer(n, m, params.inset)

        reason = ProvablyTooSmall(parts, container, params)
        if reason is not None:
            attempts.append(GridAttempt((n, m), TOO_SMALL, reason))
            continue

        reporter = _AttemptReporter(progress, (n, m), params.restarts, len(attempts) + 1)
        layout = SolveFixedGrid(parts, n, m, params, reporter, cancelled, observer)
        if layout is None:
            if stopped():
                attempts.append(GridAttempt((n, m), CANCELLED, "stopped while searching this size"))
                break
            attempts.append(GridAttempt((n, m), NOT_FOUND, f"no arrangement in {params.restarts} attempts"))
            continue

        attempts.append(GridAttempt((n, m), PACKED))
        return PackResult(layout, attempts)

    return PackResult(None, attempts)
