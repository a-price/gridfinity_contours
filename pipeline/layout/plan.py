"""The whole stack in one call: parts to bins to drawers.

The top of the layout package, and the seam
[architecture.md](../../docs/architecture.md) describes as built but
unplumbed. `Group` returns a list of layouts and `Assign` places their
footprints, but nothing joined the two outside `layout_demo`, so every
front end still spoke in terms of a single bin. This is the join, and it
is deliberately headless: a front end that wanted the whole answer had to
own the sequence, and two front ends owning it is how they come to
disagree about what the answer means.

**The feedback edge lives here.** Grouping is told which footprints the
drawers can actually hold, via `AdmissibleFootprints`, *before* it starts
- a bin seven cells long cannot go in a drawer six cells wide at any
angle, so proposing one wastes the entire stochastic search below it.
That is the one place the drawer level reaches back down, and it belongs
in whatever runs both.

**Progress is a first-class output, not a debug hook.** Grouping a real
library takes minutes, so a caller has to be able to show that the search
is alive and what it would settle for if stopped now. Both underlying
searches already report - `grouping.Step` and `drawer.Trial` - at rates
of thousands per second and in two different vocabularies. This narrows
them to one throttled `Progress` carrying the best answer so far, which
is the only thing a person watching actually wants.

The one thing it carries that is *not* an answer is first fit's opening
pass, under its own phase. Those bins hold only part of the library, but
they are real bins and they exist within a second of pressing Plan, where
the first complete grouping is a minute away on a real library. A window
with nothing to draw for that minute is the problem `Progress` exists to
solve, so the fragment is reported - labelled as a fraction of the
library rather than as an arrangement of it, and never handed back as a
plan.

Cancellation rides the same channel, because `Group` has no `cancelled`
parameter to plumb one through and the observer is the only hook into it.
The best grouping seen is kept as it goes, so a cancelled run still
returns the best answer it had rather than nothing.
"""

import json
import time
from dataclasses import dataclass, field, replace
from typing import Callable, Sequence

from pipeline.layout.drawer import (
    EXHAUSTED,
    PLACED,
    AdmissibleFootprints,
    Assign,
    AssignmentResult,
    Drawer,
    FreeCells,
    LargestFreeRegion,
    Trial,
)
from pipeline.layout.grouping import FILLING, Group, Grouping, Improve, Step
from pipeline.layout.packer import Pack
from pipeline.layout.parameters import LayoutParameters
from pipeline.layout.part import Part
from pipeline.layout.placement import Layout

# The three phases a caller can be told about, in the order they happen.
# `FILLING` is grouping's own opening pass under grouping's own name, so
# the two modules cannot come to spell it differently; the other two are
# this module's, since nothing below reports in these terms.
GROUPING = "grouping"
ASSIGNING = "assigning"

# Seconds between progress reports. The searches report thousands of times
# a second and a front end cannot draw at that rate, so this throttles at
# the source rather than leaving every caller to invent the same clock.
# Fast enough to read as live, slow enough that the drawing is a rounding
# error against the search.
DEFAULT_REPORT_INTERVAL = 0.25

# Bumped only when a reader could otherwise misinterpret an older file,
# and unit-tagged for the same reason `contour_io` is: the whole risk with
# a drawer file is reading one number as another kind of number.
DRAWER_FORMAT_VERSION = 1
DRAWER_UNITS = "cells"


def SaveDrawers(path: str, drawers: Sequence[Drawer]) -> None:
    """Write a drawer list to `path` as JSON, in whole grid cells.

    **Cells, not the millimeters they were measured from.** That
    conversion is one-way and deliberately lossy - `DrawerCells` floors a
    measurement onto the 42mm lattice - so millimeters in the file would
    record a number the system never uses again, and two drawers that
    behave identically would be stored differently. Cells are what every
    level above `DrawerCells` speaks in, and so what the file remembers.
    """
    if not drawers:
        raise ValueError("no drawers to save")

    payload = {
        "version": DRAWER_FORMAT_VERSION,
        "units": DRAWER_UNITS,
        "drawers": [{"width": drawer.width, "height": drawer.height} for drawer in drawers],
    }
    with open(path, "w") as handle:
        json.dump(payload, handle, indent=1)
        handle.write("\n")


def ReadDrawers(path: str) -> list[Drawer]:
    """Read a drawer list back, in whole grid cells.

    Here rather than in `loading.py`, which is about contours, and rather
    than in `drawer.py`, which is deliberately pure integer geometry with
    no I/O anywhere in it.

    Every drawer is validated on the way in. Everything above this treats
    a `Drawer` as trustworthy integers - the assignment search shifts
    bitmasks by them - so a fractional or negative size has to be refused
    here rather than surfacing as an empty search much later.
    """
    with open(path) as handle:
        payload = json.load(handle)

    if not isinstance(payload, dict):
        raise ValueError(f"{path} is not a drawer file")

    version = payload.get("version")
    if version != DRAWER_FORMAT_VERSION:
        raise ValueError(f"{path} is format version {version}, this build reads {DRAWER_FORMAT_VERSION}")

    units = payload.get("units")
    if units != DRAWER_UNITS:
        raise ValueError(f"{path} is in '{units}', expected '{DRAWER_UNITS}'")

    raw = payload.get("drawers")
    if not isinstance(raw, list) or not raw:
        raise ValueError(f"{path} contains no drawers")

    return [_ParseDrawerEntry(index, entry) for index, entry in enumerate(raw)]


def _ParseDrawerEntry(index: int, entry: object) -> Drawer:
    """One drawer from a file, validated into whole cells.

    The index is named rather than the entry, because a file of a dozen
    drawers with one bad one in it is otherwise a guessing game.
    """
    if not isinstance(entry, dict):
        raise ValueError(f"drawer {index} is {type(entry).__name__}, expected an object")

    size = []
    for name in ("width", "height"):
        value = entry.get(name)
        # `bool` is an `int` in Python, and `{"width": true}` is a typo
        # rather than a one-cell drawer.
        if not isinstance(value, int) or isinstance(value, bool):
            raise ValueError(f"drawer {index} has {name} {value!r}; sizes are whole grid cells")
        size.append(value)

    try:
        return Drawer(size[0], size[1])
    except ValueError as error:
        raise ValueError(f"drawer {index}: {error}") from None


@dataclass(frozen=True)
class Progress:
    """What the search has found so far, for a caller drawing it.

    Carries the best *answer* rather than the current candidate. The
    searches below report what they are trying, most of which is rejected
    a moment later, and a window redrawing that would flicker through
    thousands of arrangements nobody chose. What a person watching wants
    to know is "what would I get if I stopped it now", which is this.

    `bins` means something different in each phase, which is what `phase`
    is for. In FILLING it is the bins opened so far and accounts for only
    some of the library; from GROUPING onward it is a complete arrangement
    and therefore an answer. `assignment` arrives only once the drawer
    search starts, since until then nothing has been assigned.
    """

    phase: str
    events: int
    bins: tuple[Layout, ...] = ()
    assignment: AssignmentResult | None = None
    expected: int = 0

    @property
    def cells(self) -> int:
        return sum(layout.cells for layout in self.bins)

    @property
    def placed(self) -> int:
        """Objects accounted for by `bins`. Every one of them from GROUPING
        onward; only the ones first fit has reached before that.
        """
        return sum(len(layout.placements) for layout in self.bins)

    def __str__(self) -> str:
        if self.phase == FILLING:
            # Deliberately not phrased as an answer. These bins hold real
            # parts and are worth drawing - it is the earliest picture
            # there is - but the library is not all in them yet, and a
            # cell count over a fragment would flatter it badly.
            total = f"/{self.expected}" if self.expected else ""
            return f"first fit: {self.placed}{total} objects into {len(self.bins)} bin(s) ({self.events} steps)"

        if self.phase == GROUPING:
            if not self.bins:
                # No arrangement at all yet, complete or partial.
                return f"grouping: building the first arrangement ({self.events} steps)"
            # The caption for a picture that already shows these bins lying
            # in drawers, which is a provisional first fit and not an
            # answer - so the line has to say the drawers are still an open
            # question, or the picture claims more than the search knows.
            return (
                f"grouping: best so far {len(self.bins)} bins / {self.cells} cells "
                f"({self.events} steps); drawers not searched yet"
            )

        placed = 0 if self.assignment is None else len(self.assignment.slots)
        return f"assigning: {placed}/{len(self.bins)} bins placed ({self.events} nodes)"


Reporter = Callable[[Progress], None]


@dataclass(frozen=True)
class StoragePlan:
    """Where every object ends up: which bin it shares, and which drawer
    that bin goes in.

    Holds the parts it was built from, because every drawing downstream
    needs them and re-deriving them from contours would rasterize the
    whole library a second time.

    `cancelled` distinguishes a plan somebody stopped from one the search
    finished. The bins are still the best grouping found either way, but
    only a finished run's `assignment` says anything about whether they
    fit - which is why a cancelled plan reports no assignment at all
    rather than one computed from a grouping still being improved.
    """

    drawers: tuple[Drawer, ...]
    parts: dict[int, Part]
    layouts: dict[int, Layout]
    assignment: AssignmentResult | None = None
    grouping: Grouping | None = None
    cancelled: bool = False
    footprints: dict[int, tuple[int, int]] = field(default_factory=dict)

    @property
    def placed(self) -> bool:
        """Whether every bin found a drawer."""
        return self.assignment is not None and self.assignment.placed

    @property
    def cells(self) -> int:
        return sum(layout.cells for layout in self.layouts.values())

    def Report(self) -> str:
        """The whole answer as text: the bins, then where they went, then
        what room is left.

        Free space is reported per drawer and alongside its largest
        connected patch, because the gap between the two is exactly the
        space that is free and useless - see `drawer.LargestFreeRegion`.
        """
        lines = [f"{len(self.layouts)} bins, {self.cells} cells"]
        for bin_id in sorted(self.layouts):
            n, m = self.layouts[bin_id].grid
            contents = ", ".join(str(part) for part in sorted(self.layouts[bin_id].placements))
            lines.append(f"bin {bin_id}: {n}x{m} holding {contents}")

        if self.cancelled:
            lines.append("cancelled before the drawer search ran")
            return "\n".join(lines)
        if self.assignment is None:
            return "\n".join(lines)

        lines.append(self.assignment.Report(self.footprints))
        free = FreeCells(self.drawers, self.footprints, self.assignment)
        largest = LargestFreeRegion(self.drawers, self.footprints, self.assignment)
        for index, drawer in enumerate(self.drawers):
            lines.append(
                f"drawer {index} ({drawer.width}x{drawer.height}): "
                f"{free[index]} cells free, {largest[index]} in one piece"
            )
        return "\n".join(lines)


class _Cancelled(Exception):
    """Someone asked the search to stop.

    Raised from inside the observer because that is the only hook into
    `Group`, which takes no `cancelled` predicate. Caught in `BuildPlan`,
    which still has the best grouping seen up to that point - so stopping
    a search costs the time it had left, not the answer it had found.
    """


class _Watcher:
    """Narrows two search vocabularies into one throttled `Progress`, and
    remembers the best answer seen.

    One object across both phases rather than one per phase, so the event
    count and the best grouping carry over: the drawer search's progress
    line can say how many of *these* bins are down, which it could not if
    it had never seen the grouping.
    """

    def __init__(
        self,
        report: Reporter | None,
        cancelled: Callable[[], bool] | None,
        interval: float,
        expected: int,
    ) -> None:
        self._report = report
        self._cancelled = cancelled
        self._interval = interval
        self._expected = expected
        self._last = 0.0
        self._best_cells: int | None = None
        self.phase = FILLING
        self.events = 0
        # The best complete grouping seen, which is an answer and is what a
        # cancelled run is given. `opening` is first fit's fragment, which
        # is not an answer and is only ever drawn - keeping the two apart
        # is what stops a search stopped halfway through its opening pass
        # from handing back a floorplan missing half the library.
        self.bins: tuple[Layout, ...] = ()
        self.opening: tuple[Layout, ...] = ()
        self.assignment: AssignmentResult | None = None

    def Resume(self, start: Grouping) -> None:
        """Begin from an arrangement somebody already has.

        Without this, pressing Plan on a reloaded session would blank the
        floorplan on screen and leave it blank until the local search
        happened to improve something - which on a settled library may be
        never. The resumed arrangement is complete and is an answer, so it
        is the best so far from the first frame, and `Improve` only ever
        accepts something strictly cheaper than it.
        """
        self.phase = GROUPING
        self.bins = tuple(start.bins)
        self._best_cells = start.cells

    def OnGrouping(self, step: Step) -> None:
        """Keep the cheapest *complete* grouping seen, and before there is
        one, whatever first fit has opened so far.

        Accepted steps only, in both cases: the search reports everything
        it tries and rejects nearly all of it.

        The complete/partial split is what the phase is for. First fit
        builds its answer a part at a time, so its early steps hold a few
        parts in a few bins and score wonderfully by cell count while
        accounting for almost nothing - offering those as "best so far"
        reported one bin and two cells for a four-part library, which is
        not a worse answer but not an answer at all. They are still worth
        *drawing*, since they are the first picture the search can produce
        and they are made of real bins; so they are kept, under FILLING,
        where the report describes them as a fraction of the library
        rather than as an arrangement of it.
        """
        if not step.accepted:
            self._Tick()
            return

        if self._Complete(step):
            self.phase = GROUPING
            if self._best_cells is None or step.cells < self._best_cells:
                self._best_cells, self.bins = step.cells, step.bins
        elif self._best_cells is None:
            # Still opening bins. Once a complete grouping has been seen
            # there is a real answer to show, and a fragment must never
            # replace it.
            self.opening = step.bins
        self._Tick()

    def _Complete(self, step: Step) -> bool:
        """Whether these bins account for every part being grouped."""
        return sum(len(layout.placements) for layout in step.bins) == self._expected

    def OnAssigning(self, trial: Trial) -> None:
        # The deepest partial assignment, which is the most complete
        # picture the search has reached. It retreats constantly, and
        # showing every retreat would be a picture of backtracking rather
        # than of progress.
        if self.assignment is None or len(trial.slots) > len(self.assignment.slots):
            self.assignment = AssignmentResult(PLACED, {slot.bin_id: slot for slot in trial.slots})
        self._Tick()

    def _Tick(self) -> None:
        self.events += 1
        if self._cancelled is not None and self._cancelled():
            raise _Cancelled()
        if self._report is None:
            return

        now = time.monotonic()
        if now - self._last < self._interval:
            return
        self._last = now
        # The answer when there is one, the fragment being drawn when there
        # is not. `phase` is what tells them apart, and is FILLING for
        # exactly as long as `bins` is empty.
        self._report(Progress(self.phase, self.events, self.bins or self.opening, self.assignment, self._expected))


def BuildPlan(
    parts: dict[int, Part],
    drawers: Sequence[Drawer],
    params: LayoutParameters | None = None,
    report: Reporter | None = None,
    cancelled: Callable[[], bool] | None = None,
    interval: float = DEFAULT_REPORT_INTERVAL,
    start: Grouping | None = None,
) -> StoragePlan:
    """Group `parts` into bins and fit those bins into `drawers`.

    Runs the two searches in the only order they can go - a bin's
    footprint has to exist before it can be placed - with the feedback
    edge applied in front of both: grouping is restricted to footprints
    some drawer could hold, so the stochastic search never spends itself
    on a bin that could not be stored anyway.

    `start`, if given, is a grouping to resume from rather than to
    rediscover - the arrangement a previous session settled on. Parts it
    does not account for are opened into bins of their own and the local
    search takes it from there, which leaves every bin it cannot improve
    exactly as it was. That is what makes adding one tool to a settled
    library a matter of printing one bin instead of twelve; see `_Regroup`.

    `report` sees a throttled `Progress` through all three phases;
    `cancelled` is polled at the same points. A cancelled run returns the
    best *complete* grouping it had found with no assignment, flagged as
    `cancelled` - not an exception, because the partial answer is worth
    showing and a person who pressed Stop is not handling an error. What it
    never returns is one of first fit's fragments, however far along the
    report on screen had got: that is a picture, not a floorplan.
    """
    if not parts:
        raise ValueError("nothing to plan")
    if not drawers:
        raise ValueError("no drawers to plan into")

    params = params or LayoutParameters()
    # The feedback edge, applied before grouping rather than after a failed
    # assignment: proposing a footprint no drawer can hold wastes the whole
    # stack below it, and the drawers are already known here.
    params = _RestrictedToDrawers(params, drawers)

    watcher = _Watcher(report, cancelled, interval, expected=len(parts))
    try:
        grouping = _Regroup(parts, params, start, watcher)
    except _Cancelled:
        return StoragePlan(
            drawers=tuple(drawers),
            parts=parts,
            layouts=dict(enumerate(watcher.bins)),
            footprints={index: layout.grid for index, layout in enumerate(watcher.bins)},
            cancelled=True,
        )

    layouts = dict(enumerate(grouping.bins))
    footprints = {index: layout.grid for index, layout in layouts.items()}

    watcher.phase = ASSIGNING
    watcher.bins = tuple(grouping.bins)
    try:
        assignment = Assign(footprints, drawers, observer=watcher.OnAssigning)
    except _Cancelled:
        assignment = AssignmentResult(
            EXHAUSTED,
            {},
            sorted(footprints),
            "the drawer search was stopped; this is not evidence that the bins do not fit",
        )

    return StoragePlan(
        drawers=tuple(drawers),
        parts=parts,
        layouts=layouts,
        assignment=assignment,
        grouping=grouping,
        footprints=footprints,
    )


def _Regroup(
    parts: dict[int, Part],
    params: LayoutParameters,
    start: Grouping | None,
    watcher: "_Watcher",
) -> Grouping:
    """Partition `parts` into bins, from scratch or from `start`.

    From scratch is `Group`: first-fit for a starting point, then local
    search. From `start` is the same local search on an arrangement
    somebody already has, with any part `start` does not account for
    opened into a bin of its own first.

    **The stability this buys is the point, and it is a property of
    `Improve` rather than of anything here.** The local search only ever
    accepts a move that lowers the total cell count, and a bin no accepted
    move touched is carried through as the very same `Layout` object - the
    same grid, the same placements, to the millimetre. So adding one tool
    to a settled library reprints the bins the search could genuinely
    improve and leaves every other bin exactly as it was printed. Starting
    over would re-derive an equally good answer that shared nothing with
    the shelf of bins already sitting in the drawer.
    """
    if start is None:
        return Group(parts, params, observer=watcher.OnGrouping)

    missing = sorted(set(parts) - start.PartIds())
    bins = list(start.bins)
    for part_id in missing:
        # One bin each, at its own smallest size - the same baseline
        # `OnePerBin` uses, and the honest starting point for a part
        # nothing has yet decided anything about.
        opened = Pack({part_id: parts[part_id]}, params).layout
        if opened is None:
            raise ValueError(
                f"part {part_id} does not fit any bin these drawers can hold, so it cannot be added to this floorplan"
            )
        bins.append(opened)

    # Seeded with the grown arrangement rather than the saved one, so the
    # first frame accounts for every part - including the tool just added,
    # sitting in a bin of its own until the search finds it somewhere
    # better. The saved floorplan alone would under-count the library.
    resumed = Grouping(bins)
    watcher.Resume(resumed)
    return Improve(parts, resumed, params, observer=watcher.OnGrouping)


def _RestrictedToDrawers(params: LayoutParameters, drawers: Sequence[Drawer]) -> LayoutParameters:
    """`params` with the grid search narrowed to footprints some drawer can
    hold.

    An explicit `admissible_grids` already set by the caller wins: it is a
    narrower statement than this one can make, and silently widening it
    back out would let a bin the caller had ruled out reappear.
    """
    if params.admissible_grids is not None:
        return params
    return replace(params, admissible_grids=AdmissibleFootprints(drawers, params.max_grid))
