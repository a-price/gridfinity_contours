"""Choosing which pose each part starts at.

The quarter turn is the one variable the relaxation can never explore.
Nothing exerts torque on a variable with four values, so whatever the
restart loop picks is what that attempt is stuck with - and a bad pick is
not a slow attempt, it is a doomed one. This module picks better than
chance.

Under FREE rotation the free angle *is* explorable, but only downhill, and
the gradient will not carry a part across the 45 degrees between one
quarter turn and the next when everything in between is worse. So the
choice made here still decides which basin an attempt starts in, and
matters for the same reason.

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

import cv2
import numpy as np

from layout.container import Container
from layout.parameters import LayoutParameters
from layout.part import CanonicalOrder, Part
from layout.placement import Pose

# How finely to slide one profile against another, in millimeters. This
# only orders candidate orientations, so it does not need the raster's own
# resolution - and the cost is linear in how fine it is.
SHIFT_STEP_MM = 1.0

# How many assignments are worth scoring exhaustively. Past this the
# candidates are sampled instead, which keeps the cost flat however many
# parts there are - the permutation count is exponential in them.
MAX_ENUMERATED = 64


def _TurnedMask(part: Part, pose: Pose) -> np.ndarray:
    """The part's interior mask, turned to `pose`.

    Two paths, because a quarter turn deserves the exact one. `np.rot90` is
    a view and a copy with no resampling at all, so the 90-degree mode's
    profiles - and therefore its rankings, its restart order and every
    layout it has ever produced - are bit-identical to what they were
    before poses existed. An off-axis angle has no such luck and goes
    through `warpAffine`, which is nearest-neighbour on a boolean mask: a
    pixel either lands inside or it does not, so the result is still a mask
    and not a blurred one.

    The extra canvas is sized to the rotated diagonal, since a mask turned
    inside its own bounds would have its corners cut off - and the corners
    of a long thin part are the whole of its ends.
    """
    mask = np.asarray(part.sdf) < 0
    if pose.upright:
        return np.rot90(mask, k=pose.orientation % 4)

    turned = np.rot90(mask, k=pose.orientation % 4)
    height, width = turned.shape
    span = int(np.ceil(np.hypot(height, width)))
    # Negated, because the two conventions disagree. A mask row index is
    # this package's +y, so a counterclockwise turn in bin coordinates runs
    # clockwise through `getRotationMatrix2D`, whose positive angle is
    # counterclockwise on a y-down image. Without the sign, a profile
    # describes the mirror of the pose it is ranking - which still looks
    # like a plausible profile, and quietly ranks every diagonal backwards.
    matrix = cv2.getRotationMatrix2D((width / 2.0, height / 2.0), -float(np.rad2deg(pose.angle)), 1.0)
    matrix[0, 2] += (span - width) / 2.0
    matrix[1, 2] += (span - height) / 2.0
    spun = cv2.warpAffine(turned.astype(np.uint8), matrix, (span, span), flags=cv2.INTER_NEAREST)
    return spun.astype(bool)


def WidthProfile(part: Part, pose: Pose) -> np.ndarray:
    """How wide the part is at each raster column along its x axis, in mm.

    Read off the part's own signed distance field, where interior is
    negative, rather than off its contour: the contour is simplified to a
    few dozen vertices, so binning them by x leaves most columns holding
    one vertex or none and reports a width of zero. The raster is exactly
    the dense sampling this needs, and it already exists.

    Trimmed to the part itself, since the field is padded on every side.
    """
    mask = _TurnedMask(part, pose)
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
    parts: dict[int, Part], fitting: dict[int, list[Pose]], order: list[int]
) -> dict[tuple[int, Pose], np.ndarray]:
    """Every fitting pose's width profile, computed once.

    Each profile rotates a part's whole distance-field mask and reduces
    over it, which is the expensive step here - `RankedAssignments` used to
    pay for it twice, once inside `Assignment` for the greedy seed and once
    more building its own copy of the same dict a few lines later. The
    45-degree mode doubles how many there are and makes each off-axis one a
    warp rather than a transpose, so paying once matters more than it did.
    """
    return {(part_id, pose): WidthProfile(parts[part_id], pose) for part_id in order for pose in fitting[part_id]}


def Assignment(
    parts: dict[int, Part],
    fitting: dict[int, list[Pose]],
    container: Container,
    params: LayoutParameters,
    profiles: dict[tuple[int, Pose], np.ndarray] | None = None,
) -> dict[int, Pose]:
    """A pose per part, chosen so the parts stack narrowly.

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
    chosen: dict[int, Pose] = {}

    for part_id in order:
        options = fitting[part_id]
        best: tuple[float, Pose, int, np.ndarray] | None = None

        for pose in options:
            width = profiles[(part_id, pose)]
            if not len(width) or len(width) > columns:
                continue
            peak, shift = _BestShift(total, width, step)
            if best is None or peak < best[0]:
                best = (peak, pose, shift, width)

        if best is None:
            chosen[part_id] = options[0]
            continue

        _, pose, shift, width = best
        total[shift : shift + len(width)] += width
        chosen[part_id] = pose

    return chosen


def _Columns(container: Container, params: LayoutParameters) -> int:
    return int(round((container.width - 2.0 * params.c_wall) / params.resolution))


def _Step(params: LayoutParameters) -> int:
    return max(1, int(round(SHIFT_STEP_MM / params.resolution)))


def _Permutations(options: list[list[Pose]]) -> int:
    """How many assignments exist. Multiplied rather than enumerated, since
    the whole point of asking is to avoid building the list.
    """
    total = 1
    for choices in options:
        total *= len(choices)
    return total


def RankedAssignments(
    parts: dict[int, Part],
    fitting: dict[int, list[Pose]],
    container: Container,
    params: LayoutParameters,
    count: int,
    enumerate_limit: int = MAX_ENUMERATED,
) -> list[dict[int, Pose]]:
    """Pose assignments, most promising first.

    The restart loop walks this list instead of drawing a pose per attempt.
    Ranking is the right shape for the problem: the score is a property of
    the assignment as a whole, so there is nothing an attempt could
    usefully decide on its own, and sorting means the good candidates are
    tried first rather than eventually.

    Exhaustive while the permutation count is small, which covers most real
    bins at 90 degrees - parts long enough to be interesting have only two
    fitting poses there, not four. The 45-degree mode roughly doubles the
    per-part count and so pushes many more sets past the limit; that is a
    cost of the mode rather than a defect here, and the sampled path
    absorbs it. Past the limit the candidates are the greedy assignment
    plus a seeded sample, ranked the same way. A ranked sample is still
    strictly better than an unranked draw, and the cost stays bounded
    whatever the part count.

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
        #
        # Drawn as *indices* into each part's option list rather than as
        # poses, for two reasons. A Pose has no ordering, and the sort is
        # what keeps a sampled set in a fixed order across runs. And
        # `rng.choice(n)` consumes the generator exactly as the old
        # `rng.choice(orientations)` did, so the 90-degree mode samples the
        # same candidates it always has - which is what keeps every
        # committed layout and animation reproducible through this change.
        rng = np.random.default_rng([params.seed, columns, len(order)])
        drawn = {tuple(int(rng.choice(len(choices))) for choices in options) for _ in range(count * 2)}
        sampled = [tuple(choices[index] for choices, index in zip(options, picks)) for picks in sorted(drawn)]
        candidates = [tuple(greedy[part_id] for part_id in order), *sampled]

    scored = []
    for index, combination in enumerate(dict.fromkeys(candidates)):
        widths = [profiles[(part_id, pose)] for part_id, pose in zip(order, combination)]
        # The index breaks ties, so equally good assignments come back in a
        # fixed order rather than however the sort happened to see them.
        scored.append((StackedWidth(widths, columns, step), index, combination))

    scored.sort()
    return [dict(zip(order, combination)) for _, _, combination in scored[:count]]
