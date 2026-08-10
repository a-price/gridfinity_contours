"""Does letting parts turn off-axis actually pack them tighter?

    rotation_experiment.py                 # every set, every mode
    rotation_experiment.py --only spoons   # one set
    rotation_experiment.py --restarts 24   # the tuned budget, slowly

D1 rejected continuous rotation for buying "a few percent density on
diagonal-friendly shapes" at the price of a materially more complex
solver. The complexity half of that turned out to be overstated - the
distance fields are queried at arbitrary points already, so no field is
ever re-rasterized at an angle - which leaves the density half worth
measuring rather than assuming.

**This reports two different things and the difference matters.** For a
single part the answer is exact and needs no solver: sweep the contour
over every angle and ask which bins its bounding box fits, which is what
`Reach` does. For several parts it is whatever the stochastic search
managed, so a mode that comes back worse has not been shown to be worse -
only to have got unlucky at this budget. Any conclusion about the
multi-part table has to survive a raised `--restarts`.

Not a test. There is nothing here to assert: the whole point is that
nobody knows the answer yet, and a number that changed would be a result
rather than a regression. `rotation_experiment_test.py` covers the
machinery - that the sweep agrees with the packer, that the modes are
actually distinct - and deliberately pins none of the findings.

**What it found, on the fixtures, at 6 and again at 24 restarts** (the
multi-part table was identical at both, so these are not luck):

* Free rotation pays, and it pays on *whole objects that barely miss*
  rather than on tighter nesting. Alone, `medium_spoon` and `screwdriver`
  each drop 10 cells to 8 at about 20 degrees, and the knife drops 7 to 6
  at 3.9. `big_spoon`, `serving_spoon` and five others gain nothing.
* **The gains do not compound.** `medium_spoon` and `screwdriver` each
  save 2 cells alone; packed together they need 10 cells either way. Two
  parts that individually want to lie diagonally cannot both have the
  diagonal, so a bin holding both is constrained by something the
  single-part sweep does not see. This is the finding that decides how
  much free rotation is worth: it is a fix for the one-object bin, not a
  density technique.
* The one multi-part win is `outsize` - the two objects nothing else
  fits - at 21 cells square-on against 14 turned, and it is reached
  *faster* rather than slower.
* **45 degrees is a regression, not a cheap approximation.** It saves
  nothing on any fixture alone, because the angles that pay are near 4 and
  20 degrees. Worse, on the three spoons it returns 12 cells where 90
  returns 10: the extra poses do not change the top-ranked assignment, but
  they lengthen the candidate list from 16 to 24, and the eight added ones
  stand `small_spoon` on a diagonal - which `StackedWidth` ranks
  plausibly and which packs badly. At a fixed restart budget those
  attempts displace re-tries of assignments that work.
"""

import argparse
import sys
import time
from dataclasses import replace
from typing import Sequence

import numpy as np

from layout.container import BuildContainer
from layout.loading import LoadParts
from layout.packer import GridsFor, Pack
from layout.parameters import EIGHTH_TURNS, FREE_ROTATION, QUARTER_TURNS, ROTATIONS, LayoutParameters
from layout.verify import CheckLayout

# How finely the exact single-part sweep steps, in degrees. Fine enough
# that the reported angle is meaningful, and the cost is linear in it - the
# whole sweep is one vectorized rotation of a few dozen contour vertices.
SWEEP_STEP_DEG = 0.25

# Cut well below the tuned 24. The multi-part table is a comparison between
# modes rather than a claim about any one of them, so what matters is that
# every mode gets the same budget - and a budget small enough to run all
# three on every set in a coffee break is worth more here than a tight
# answer for one.
RESTARTS = 6

# The sets, chosen so that each asks a different question.
SETS: dict[str, list[str]] = {
    # The standard example. Known to want 10 cells at 90 degrees, and
    # measured to be genuinely near-optimal there - so this is the control.
    "spoons": ["small_spoon", "medium_spoon", "big_spoon"],
    # Two parts that individually gain from turning. If nesting helps at
    # all, it should show here.
    "gainers": ["medium_spoon", "screwdriver"],
    # The two objects that fit no bin at all square-on.
    "outsize": ["knife", "serving_spoon"],
    # A mixed handful, closer to what a drawer actually holds.
    "drawer": ["small_spoon", "medium_fork", "spreader", "small_measure"],
}


def Reach(contour: np.ndarray, params: LayoutParameters) -> dict[str, tuple[int, int] | None]:
    """The smallest bin one part fits, under each mode, exactly.

    No solver and no randomness: a single part in an empty bin is placeable
    exactly when its rotated bounding box clears the wall clearance on both
    axes, so sweeping the angle settles it. That makes this the trustworthy
    half of the experiment - a difference here is a fact about the
    geometry, not about how long the search ran.

    The sweep is sound for FREE despite being a sample, because it is only
    ever used to say a bin *does* fit: it can miss a narrow window and
    report a bin larger than necessary, never smaller than possible.
    """
    contour = np.asarray(contour, dtype=np.float64).reshape(-1, 2)
    angles = np.deg2rad(np.arange(0.0, 180.0, SWEEP_STEP_DEG))
    cos, sin = np.cos(angles), np.sin(angles)
    x = np.outer(cos, contour[:, 0]) - np.outer(sin, contour[:, 1])
    y = np.outer(sin, contour[:, 0]) + np.outer(cos, contour[:, 1])
    extents = np.stack([np.ptp(x, axis=1), np.ptp(y, axis=1)], axis=-1)

    # Which sweep rows each mode is allowed to use. 90 and 45 may only
    # stand where their candidate poses stand; FREE may stand anywhere.
    quarters = np.isclose(np.rad2deg(angles) % 90.0, 0.0)
    eighths = np.isclose(np.rad2deg(angles) % 45.0, 0.0)
    allowed = {QUARTER_TURNS: quarters, EIGHTH_TURNS: eighths, FREE_ROTATION: np.ones(len(angles), bool)}

    smallest: dict[str, tuple[int, int] | None] = {mode: None for mode in ROTATIONS}
    for n, m in sorted(GridsFor(params), key=lambda grid: (grid[0] * grid[1], grid[0] - grid[1])):
        container = BuildContainer(n, m, params.inset)
        room = np.array([container.width, container.height]) - 2 * params.c_wall
        # Either way round, since a bin is only ever named with n >= m.
        fits = (extents <= room).all(axis=1) | (extents[:, ::-1] <= room).all(axis=1)

        for mode, mask in allowed.items():
            if smallest[mode] is None and (fits & mask).any():
                smallest[mode] = (n, m)
    return smallest


def Cells(grid: tuple[int, int] | None) -> int:
    """A grid's cell count, with "fits nothing" sorting as worst."""
    return grid[0] * grid[1] if grid else 10**6


def _Label(grid: tuple[int, int] | None) -> str:
    return f"{grid[0]}x{grid[1]}={grid[0] * grid[1]}" if grid else "none"


def SinglePartTable(stems: Sequence[str], params: LayoutParameters) -> list[str]:
    """The exact half: what each object alone needs, per mode."""
    lines = [f"{'part':16} {'bbox mm':>13} " + " ".join(f"{mode:>9}" for mode in ROTATIONS) + "  verdict"]
    for stem in stems:
        parts = LoadParts([f"test_data/{stem}.svg"], params)
        part = next(iter(parts.values()))
        reach = Reach(part.pocket_contour, params)

        best, worst = Cells(reach[FREE_ROTATION]), Cells(reach[QUARTER_TURNS])
        verdict = "same" if best == worst else f"saves {worst - best} cells" if worst < 10**6 else "newly packable"
        size = part.size
        lines.append(
            f"{stem:16} {f'{size[0]:.0f}x{size[1]:.0f}':>13} "
            + " ".join(f"{_Label(reach[mode]):>9}" for mode in ROTATIONS)
            + f"  {verdict}"
        )
    return lines


def PackUnder(stems: Sequence[str], mode: str, params: LayoutParameters, restarts: int) -> tuple[str, float]:
    """One set packed under one mode: what it came to, and how long it
    took.

    The layout is re-checked against exact polygon geometry before being
    reported. A mode that packed tighter by placing parts through each
    other would otherwise look like the finding this experiment is hunting
    for, which is the one way it could actively mislead.
    """
    tuned = replace(params, rotation=mode, restarts=restarts)
    parts = LoadParts([f"test_data/{stem}.svg" for stem in stems], tuned)

    started = time.monotonic()
    result = Pack(parts, tuned)
    elapsed = time.monotonic() - started

    if result.layout is None:
        return "none", elapsed

    problems = CheckLayout(result.layout, parts, pair_clearance=tuned.c_pair, wall_clearance=tuned.c_wall)
    if problems:
        return f"INVALID ({len(problems)})", elapsed

    turned = sum(1 for placement in result.layout.placements.values() if placement.angle != 0.0)
    label = _Label(result.layout.grid)
    return (f"{label} ({turned} turned)" if turned else label), elapsed


def MultiPartTable(names: Sequence[str], params: LayoutParameters, restarts: int) -> list[str]:
    """The stochastic half: what the search managed, per mode, at one
    shared budget.
    """
    lines = [f"{'set':10} " + " ".join(f"{mode:>20}" for mode in ROTATIONS)]
    for name in names:
        cells, times = [], []
        for mode in ROTATIONS:
            label, elapsed = PackUnder(SETS[name], mode, params, restarts)
            cells.append(label)
            times.append(elapsed)
        lines.append(f"{name:10} " + " ".join(f"{label:>20}" for label in cells))
        lines.append(f"{'':10} " + " ".join(f"{elapsed:>19.0f}s" for elapsed in times))
    return lines


def Main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--only", choices=sorted(SETS), help="run one set rather than all of them")
    parser.add_argument("--restarts", type=int, default=RESTARTS, help=f"attempts per grid size (default: {RESTARTS})")
    parser.add_argument("--skip-multi", action="store_true", help="only the exact single-part sweep, which is seconds")
    args = parser.parse_args(argv)

    params = LayoutParameters()
    names = [args.only] if args.only else sorted(SETS)
    stems = sorted({stem for name in names for stem in SETS[name]})

    print("one part alone, exactly - no solver, no seed\n")
    print("\n".join(SinglePartTable(stems, params)))

    if not args.skip_multi:
        print(f"\n\nseveral parts, as packed at {args.restarts} restarts - stochastic, so a")
        print("worse number is not evidence of a worse mode\n")
        print("\n".join(MultiPartTable(names, params, args.restarts)))
    return 0


if __name__ == "__main__":
    sys.exit(Main())
