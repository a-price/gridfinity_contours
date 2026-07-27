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

from pipeline.layout.container import BuildContainer, Container
from pipeline.layout.energy import LayoutParameters
from pipeline.layout.part import Part
from pipeline.layout.placement import Layout
from pipeline.layout.solver import FittingOrientations, SolveFixedGrid

PACKED = "packed"
TOO_SMALL = "too small"
NOT_FOUND = "not found"


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

    def Report(self) -> str:
        lines = [str(attempt) for attempt in self.attempts]
        if self.layout is None:
            lines.append("no grid size up to the configured maximum could hold these parts")
        elif self.skipped:
            smaller = ", ".join(f"{a.grid[0]}x{a.grid[1]}" for a in self.skipped)
            lines.append(f"note: {smaller} was not ruled out geometrically - a tighter packing may exist")
        return "\n".join(lines)


def CandidateGrids(max_grid: int) -> list[tuple[int, int]]:
    """Grid sizes worth trying, smallest first, squarest first among
    equals.

    Only `n >= m` is generated. A 2x5 bin is a 5x2 rotated a quarter turn,
    and since every part can also turn a quarter turn, the two have exactly
    the same set of solutions - enumerating both would double the search to
    rediscover each answer sideways.
    """
    if max_grid < 1:
        raise ValueError(f"max_grid must be at least 1, got {max_grid}")

    grids = [(n, m) for n in range(1, max_grid + 1) for m in range(1, n + 1)]
    grids.sort(key=lambda grid: (grid[0] * grid[1], grid[0] - grid[1], grid[0]))
    return grids


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
        if not FittingOrientations(part, container, params):
            size = part.size
            return (
                f"part {part_id} is {size[0]:.1f}x{size[1]:.1f}mm and does not fit a "
                f"{container.width:.1f}x{container.height:.1f}mm interior at any quarter turn"
            )

    required = RequiredArea(parts, params)
    if required > container.area:
        return f"parts need {required:.0f}mm^2 with their clearances, interior holds {container.area:.0f}mm^2"
    return None


def Pack(parts: dict[int, Part], params: LayoutParameters | None = None) -> PackResult:
    """Fit every part into the smallest grid that will take them.

    Candidates are tried smallest-area first and the first success is
    returned, so the result is the smallest size *found* - not provably the
    smallest that exists, since the solver is stochastic. When a
    bounds-feasible size fails, the search steps up rather than giving up:
    a usable bin beats none, and `PackResult.skipped` records what was left
    behind so an oversized result stays traceable.
    """
    params = params or LayoutParameters()
    if not parts:
        raise ValueError("nothing to pack")

    attempts: list[GridAttempt] = []
    for n, m in CandidateGrids(params.max_grid):
        container = BuildContainer(n, m, params.inset)

        reason = ProvablyTooSmall(parts, container, params)
        if reason is not None:
            attempts.append(GridAttempt((n, m), TOO_SMALL, reason))
            continue

        layout = SolveFixedGrid(parts, n, m, params)
        if layout is None:
            attempts.append(GridAttempt((n, m), NOT_FOUND, f"no arrangement in {params.restarts} attempts"))
            continue

        attempts.append(GridAttempt((n, m), PACKED))
        return PackResult(layout, attempts)

    return PackResult(None, attempts)
