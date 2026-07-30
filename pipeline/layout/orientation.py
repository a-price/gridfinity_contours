"""Choosing which quarter turn each part starts at.

Orientation is the one variable the relaxation cannot explore. Nothing
exerts torque on a part that may only sit at four angles, so whatever the
restart loop picks is what that attempt is stuck with - and a bad pick is
not a slow attempt, it is a doomed one. This module picks better than
chance.

**The signal is only ever between parts, never within one.** Turning a
part 180 degrees *reverses* its width profile without changing it, so its
extent, its area and its widest point all come out identical at 0 and 180.
Any score computed on a single part in isolation is therefore blind to
exactly the choice that matters. Measured on the three spoons: a pairwise
score is not merely weak but actively misleading, ranking the two aligned
orientations best when neither of them packs.

What does work is asking how the parts stack *together*. Two long parts
that must overlap - and in a tight bin they must, since their lengths sum
to more than the bin is long - need one's wide end beside the other's
narrow end. That is a question about their profiles side by side, and it
is what `StackedWidth` measures.

The construction is greedy and therefore linear in the number of parts,
which matters: scoring every combination is exponential, and a set of
eighteen household objects has more assignments than atoms worth counting.
Greedy is not optimal, but measured against an exhaustive sweep of the
three spoons it put all four packable assignments in the top four.
"""

import itertools

import numpy as np

from pipeline.layout.container import Container
from pipeline.layout.parameters import LayoutParameters
from pipeline.layout.part import CanonicalOrder, Part

# How finely to slide one profile against another, in millimeters. This
# only orders candidate orientations, so it does not need the raster's own
# resolution - and the cost is linear in how fine it is.
SHIFT_STEP_MM = 1.0

# How many assignments are worth scoring exhaustively. Past this the
# candidates are sampled instead, which keeps the cost flat however many
# parts there are - the permutation count is exponential in them.
MAX_ENUMERATED = 64


def WidthProfile(part: Part, orientation: int) -> np.ndarray:
    """How wide the part is at each raster column along its x axis, in mm.

    Read off the part's own signed distance field, where interior is
    negative, rather than off its contour: the contour is simplified to a
    few dozen vertices, so binning them by x leaves most columns holding
    one vertex or none and reports a width of zero. The raster is exactly
    the dense sampling this needs, and it already exists.

    Trimmed to the part itself, since the field is padded on every side.
    """
    mask = np.rot90(np.asarray(part.sdf) < 0, k=orientation % 4)
    present = mask.any(axis=0)
    if not present.any():
        return np.zeros(0)

    rows = np.arange(mask.shape[0])[:, None]
    top = np.where(mask, rows, np.inf).min(axis=0)
    bottom = np.where(mask, rows, -np.inf).max(axis=0)
    width = np.where(present, (bottom - top + 1) * part.resolution, 0.0)

    first, last = int(np.argmax(present)), len(present) - int(np.argmax(present[::-1]))
    return width[first:last]


def _BestShift(total: np.ndarray, width: np.ndarray, step: int) -> tuple[float, int]:
    """Where to lay `width` along `total` to keep the combined peak lowest,
    and what that peak is.

    Every shift is priced at once rather than in a Python loop. That is
    what makes ranking whole permutations affordable: the grouping search
    calls the solver thousands of times, and a per-shift loop here turned
    up as real time in it.

    Whatever `total` already reaches outside the window is untouched by
    where this part goes, so it is folded in once at the end rather than
    compared against on every shift.
    """
    windows = np.lib.stride_tricks.sliding_window_view(total, len(width))[::step]
    peaks = (windows + width).max(axis=1)
    index = int(peaks.argmin())
    return max(float(peaks[index]), float(total.max())), index * step


def StackedWidth(profiles: list[np.ndarray], columns: int, step: int = 1) -> float:
    """The narrowest total width these profiles can be stacked into.

    Widest part first, each laid where it keeps the running total
    narrowest. A lower number means the parts nest - one's wide end against
    another's narrow end - and a bin only that tall could hold them as
    stacked bands.

    An estimate, not a bound. Parts do not have to be arranged as bands, so
    a high score does not prove anything is impossible, and nothing here
    may be used to reject a bin. It orders candidates; the solver decides.
    """
    total = np.zeros(columns)
    for width in sorted(profiles, key=lambda w: -w.max() if len(w) else 0.0):
        if not len(width) or len(width) > columns:
            return np.inf
        _, shift = _BestShift(total, width, step)
        total[shift : shift + len(width)] += width
    return float(total.max())


def _Profiles(
    parts: dict[int, Part], fitting: dict[int, list[int]], order: list[int]
) -> dict[tuple[int, int], np.ndarray]:
    """Every fitting orientation's width profile, computed once.

    Each profile rotates a part's whole distance-field mask and reduces
    over it, which is the expensive step here - `RankedAssignments` used to
    pay for it twice, once inside `Assignment` for the greedy seed and once
    more building its own copy of the same dict a few lines later.
    """
    return {
        (part_id, orientation): WidthProfile(parts[part_id], orientation)
        for part_id in order
        for orientation in fitting[part_id]
    }


def Assignment(
    parts: dict[int, Part],
    fitting: dict[int, list[int]],
    container: Container,
    params: LayoutParameters,
    profiles: dict[tuple[int, int], np.ndarray] | None = None,
) -> dict[int, int]:
    """A quarter turn per part, chosen so the parts stack narrowly.

    Built in one greedy pass rather than by scoring every combination.
    Parts are taken in canonical order - largest first, and independent of
    the order their files were listed in - and each takes the orientation
    that keeps the running profile narrowest.

    Used both as an answer in its own right and as the seed candidate when
    there are too many permutations to rank exhaustively. `profiles`, if
    given, is used instead of recomputing `WidthProfile` - the second use
    already has one lying around and there is no reason for both callers
    to pay for it.
    """
    columns = _Columns(container, params)
    step = _Step(params)
    if columns <= 0:
        return {part_id: options[0] for part_id, options in fitting.items()}

    order = CanonicalOrder(parts)
    if profiles is None:
        profiles = _Profiles(parts, fitting, order)

    total = np.zeros(columns)
    chosen: dict[int, int] = {}

    for part_id in order:
        options = fitting[part_id]
        best: tuple[float, int, int, np.ndarray] | None = None

        for orientation in options:
            width = profiles[(part_id, orientation)]
            if not len(width) or len(width) > columns:
                continue
            peak, shift = _BestShift(total, width, step)
            if best is None or peak < best[0]:
                best = (peak, orientation, shift, width)

        if best is None:
            chosen[part_id] = options[0]
            continue

        _, orientation, shift, width = best
        total[shift : shift + len(width)] += width
        chosen[part_id] = orientation

    return chosen


def _Columns(container: Container, params: LayoutParameters) -> int:
    return int(round((container.width - 2.0 * params.c_wall) / params.resolution))


def _Step(params: LayoutParameters) -> int:
    return max(1, int(round(SHIFT_STEP_MM / params.resolution)))


def _Permutations(options: list[list[int]]) -> int:
    """How many assignments exist. Multiplied rather than enumerated, since
    the whole point of asking is to avoid building the list.
    """
    total = 1
    for choices in options:
        total *= len(choices)
    return total


def RankedAssignments(
    parts: dict[int, Part],
    fitting: dict[int, list[int]],
    container: Container,
    params: LayoutParameters,
    count: int,
    enumerate_limit: int = MAX_ENUMERATED,
) -> list[dict[int, int]]:
    """Orientation assignments, most promising first.

    The restart loop walks this list instead of drawing an orientation per
    attempt. Ranking is the right shape for the problem: the score is a
    property of the assignment as a whole, so there is nothing an attempt
    could usefully decide on its own, and sorting means the good candidates
    are tried first rather than eventually.

    Exhaustive while the permutation count is small, which covers most real
    bins - parts long enough to be interesting have only two fitting
    orientations, not four. Past that limit the candidates are the greedy
    assignment plus a seeded sample, ranked the same way. A ranked sample
    is still strictly better than an unranked draw, and the cost stays
    bounded whatever the part count.

    Never empty: with nothing else to offer it returns the greedy
    assignment alone.
    """
    if count < 1:
        raise ValueError(f"a search needs at least one assignment, got count={count}")

    order = CanonicalOrder(parts)
    options = [fitting[part_id] for part_id in order]
    if not order or any(not choices for choices in options):
        return [{part_id: fitting[part_id][0] for part_id in fitting if fitting[part_id]}]

    columns, step = _Columns(container, params), _Step(params)
    if columns <= 0:
        return [Assignment(parts, fitting, container, params)]

    # Built once and handed to Assignment for the greedy seed, rather than
    # each computing its own copy - the expensive step is rotating a part's
    # whole distance-field mask, and every fitting orientation of every
    # part goes through it exactly once either way.
    profiles = _Profiles(parts, fitting, order)
    greedy = Assignment(parts, fitting, container, params, profiles)

    if _Permutations(options) <= enumerate_limit:
        candidates = list(itertools.product(*options))
    else:
        # Seeded, so a bin still packs the same way twice.
        rng = np.random.default_rng([params.seed, columns, len(order)])
        sampled = {tuple(int(rng.choice(choices)) for choices in options) for _ in range(count * 2)}
        candidates = [tuple(greedy[part_id] for part_id in order), *sorted(sampled)]

    scored = []
    for index, combination in enumerate(dict.fromkeys(candidates)):
        widths = [profiles[(part_id, orientation)] for part_id, orientation in zip(order, combination)]
        # The index breaks ties, so equally good assignments come back in a
        # fixed order rather than however the sort happened to see them.
        scored.append((StackedWidth(widths, columns, step), index, combination))

    scored.sort()
    return [dict(zip(order, combination)) for _, _, combination in scored[:count]]
