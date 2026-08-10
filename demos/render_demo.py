"""Reference renderings, committed so a visual change shows up in review.

    python3 -m demos.render_demo --out docs/media

The still-image counterpart to `layout_demo.py`. That one animates the
*searches*; this one photographs the *renderers*, one picture per drawing
path, and the pictures live in the repository.

**Why this is not a test.** Every renderer here already has tests, and
they assert what the drawing claims to be: that the zero level set sits on
the outline, that a clearance ring sits at the clearance, that a part
lands where the layout put it. Those state an invariant, so they fail when
the code is wrong and pass when it is merely different. What they cannot
notice is a change that keeps every invariant and still looks wrong - a
fade constant that washes the field out, a stroke weight that makes the
cell grid compete with the objects.

A stored-image assertion is the obvious answer and the wrong one here.
Eleven of this project's test modules sit downstream of a stochastic
search, so any change to the solver reddens them all at once, and the only
possible response is to re-bless the lot - which teaches nothing and
trains the reflex that lets a genuinely bad change through next time.
Measured on this repository the day the spacing pass was corrected: every
part moved, legitimately, and three test modules' worth of images would
have needed regenerating without a soul looking at one.

So these are committed artifacts rather than assertions, exactly like the
GIFs. `make previews` regenerates them, git shows the difference, and a
person decides. The failure mode of a stale reference here is a diff
somebody reads, not a red build somebody silences.

Everything drawn is **deterministic**: placements that are written out or
arithmetic (see `_Centered`), fixtures loaded from files, and the drawer
assignment, which is exhaustive and carries no RNG. Nothing calls the
stochastic solver, or the pictures would change on their own and the
diffs would mean nothing.
"""

import argparse
import os
import sys
from typing import Callable, Sequence

import cv2
import numpy as np

from layout.container import DEFAULT_INTERIOR_INSET_MM, InteriorSpan
from layout.drawer import Assign, Drawer
from layout.field import FieldView, RenderField
from layout.floorplan import RenderFloorplan
from layout.loading import BuildParts, LoadParts
from layout.parameters import LayoutParameters
from layout.part import BuildPart, Part
from layout.placement import Layout, Placement
from layout.render import RenderLayout, RenderLayouts

DEFAULT_OUT = "docs/media"

# Pocket offset the gallery is drawn at, rather than the 1mm default.
# Neither reason is decoration.
#
# It exercises a *non-default* offset, so a drawing that had quietly
# hardcoded the default - or that reached for an object where it meant a
# pocket - comes out visibly wrong here rather than plausibly right.
#
# And it is legible at these scales. Each part is drawn as two outlines,
# the pocket solid and the object dashed inside it (D5), separated by
# exactly the offset. At the 2 to 3 pixels per mm these pictures are
# rendered at, a 1mm ring collapses into one fuzzy band and the two lines
# stop being two lines; 2.5mm keeps them apart. The printed PDF has no
# such problem, being at true scale - this is a limit of the gallery, so
# the gallery is what compensates.
DEMO_POCKET_OFFSET_MM = 2.5

# Prefix, so the stills group together in a directory that also holds the
# animations and so a glob can find them without knowing their names.
PREFIX = "preview_"

# The capture fixture with the most interesting field: a spoon's bowl is
# concave enough that its dilation folds into itself, which is the case
# every synthetic rectangle in the tests fails to cover.
#
# Resolved against the repository root - one level up from `demos/` - rather
# than the working directory. The pictures have to come out identical wherever
# they are generated from, or the diffs they exist to produce would depend on
# where somebody stood.
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

SPOON = os.path.join(_ROOT, "test_data", "big_spoon.svg")


def _Rectangle(width: float, height: float) -> np.ndarray:
    return np.array([[0.0, 0.0], [width, 0.0], [width, height], [0.0, height]], dtype=np.float64)


def _Parts(shapes: dict[int, np.ndarray], params: LayoutParameters) -> dict[int, Part]:
    return BuildParts(shapes, params)


def FieldDistance(params: LayoutParameters) -> np.ndarray:
    """A real contour's distance field, contoured, with its clearance rings.

    The whole of `field.ShadeDistance` in one picture: the two hues that
    carry the sign, the fade that draws the eye to the boundary, a contour
    line every millimetre, and the two levels that mean something to the
    solver.
    """
    part = LoadParts([SPOON], params)[0]
    return RenderField(part, FieldView(), params)


def FieldGradient(params: LayoutParameters) -> np.ndarray:
    """The same field's gradient length: where it stops being a distance.

    Worth a picture of its own because the shading is *deviation* from
    unit length rather than the length itself - creases between opposing
    walls read near zero and creases between perpendicular ones near
    root two, and an earlier version shaded by raw magnitude and so
    painted half the medial axis pure white.
    """
    part = LoadParts([SPOON], params)[0]
    return RenderField(part, FieldView(gradient=True), params)


def FieldSamples(params: LayoutParameters) -> np.ndarray:
    """A corner of a small part, zoomed past the raster resolution.

    The scale at which the two things a viewer can get quietly wrong
    become visible: the staircase is the raster the solver actually reads,
    and the dots are the boundary samples that get tested against other
    parts. A regression in `ResampleBoundary` would show here as four
    corner dots and bare edges between them.
    """
    part = BuildPart(_Rectangle(12.0, 8.0), resolution=params.resolution, pad=params.pad)
    return RenderField(part, FieldView(pixels_per_mm=12.0, samples=True), params)


def Bin(params: LayoutParameters) -> np.ndarray:
    """One solved bin as the printed sheet, at screen scale.

    Placements are written out rather than solved, so this is a picture of
    `preview.LayoutShapes` and nothing else: the rim, the dashed interior,
    the dashed cell grid, and the parts drawn heaviest because they are
    the subject.
    """
    layout, parts = _Arrangement(params)
    return RenderLayout(layout, parts, pixels_per_mm=3.0)


def Bins(params: LayoutParameters) -> np.ndarray:
    """Several bins wrapped into rows - what a grouping looks like before
    any drawer has been chosen for it, which is what the floorplan window
    shows while it is still searching.

    Drawn at each bin's own true relative scale, so a wide bin reads as
    wider rather than every page being normalized to one width.
    """
    layouts, parts = _Grouping(params)
    return RenderLayouts(layouts, parts, columns=2, pixels_per_mm=2.0)


def Floorplan(params: LayoutParameters) -> np.ndarray:
    """Two drawers with those bins laid into them.

    `Assign` is exhaustive and carries no RNG, so this is reproducible
    despite being a search - and it is the picture worth having, since a
    turned bin and the 42mm lattice are both things only this drawing
    shows.
    """
    layouts, parts = _Grouping(params)
    numbered = dict(enumerate(layouts))
    footprints = {index: layout.grid for index, layout in numbered.items()}
    drawers = [Drawer(4, 3), Drawer(3, 2)]

    result = Assign(footprints, drawers)
    return RenderFloorplan(drawers, numbered, result, parts, pixels_per_mm=2.2)


def _Centered(part: Part, grid: tuple[int, int]) -> np.ndarray:
    """The position that puts `part` in the middle of a `grid` interior.

    Derived rather than written down, for the bins that hold a single part
    and are meant to look it. A literal is right for exactly one pocket
    offset: `Placement.position` anchors a part's *minimum* corner and a
    Part is its pocket, so raising the offset grows every part toward +x
    and +y from a fixed anchor and slides a hand-centred fixture off
    centre. That is not hypothetical - it is what happened to the 1x1 bin
    below when pockets became geometry (D5), which went from 4.00mm of
    slack on the near side and 4.30mm on the far to 4.00 and 2.21.

    Still deterministic and still RNG-free, which is what the pictures
    actually need: these placements are fixed, just fixed by arithmetic
    instead of by hand.
    """
    interior = np.array([InteriorSpan(grid[0]), InteriorSpan(grid[1])])
    return (interior - part.size) / 2.0


def _Arrangement(params: LayoutParameters) -> tuple[Layout, dict[int, Part]]:
    """One hand-written 3x2 arrangement, shared by the pictures that need a
    solved-looking bin without calling the solver.
    """
    parts = _Parts({0: _Rectangle(60.0, 24.0), 1: _Rectangle(40.0, 18.0)}, params)
    placements = {
        0: Placement(0, np.array([8.0, 8.0])),
        1: Placement(1, np.array([12.0, 44.0])),
    }
    return Layout(grid=(3, 2), placements=placements, inset=DEFAULT_INTERIOR_INSET_MM), parts


def _Grouping(params: LayoutParameters) -> tuple[list[Layout], dict[int, Part]]:
    """Three bins over one set of parts - a grouping, without having run
    the grouping search.

    The 3x2 keeps its two positions written out, because the point of that
    bin is two parts sharing one and neither is meant to be centred. The
    two single-part bins are centred by `_Centered` instead, which is what
    they always looked like and now stays true at any offset.
    """
    parts = _Parts(
        {
            0: _Rectangle(60.0, 24.0),
            1: _Rectangle(40.0, 18.0),
            2: _Rectangle(100.0, 30.0),
            3: _Rectangle(28.0, 28.0),
        },
        params,
    )
    layouts = [
        Layout(
            grid=(3, 2), placements={0: Placement(0, np.array([8.0, 8.0])), 1: Placement(1, np.array([12.0, 44.0]))}
        ),
        Layout(grid=(3, 1), placements={2: Placement(2, _Centered(parts[2], (3, 1)))}),
        Layout(grid=(1, 1), placements={3: Placement(3, _Centered(parts[3], (1, 1)))}),
    ]
    return layouts, parts


# One entry per picture. A table rather than a sequence of calls so that
# adding a drawing path to this gallery is a line, and so the test that
# keeps the gallery honest can walk it.
PREVIEWS: tuple[tuple[str, Callable[[LayoutParameters], np.ndarray]], ...] = (
    ("field_distance", FieldDistance),
    ("field_gradient", FieldGradient),
    ("field_samples", FieldSamples),
    ("bin", Bin),
    ("bins", Bins),
    ("floorplan", Floorplan),
)


def Write(out: str, params: LayoutParameters | None = None, names: Sequence[str] | None = None) -> list[str]:
    """Render every preview into `out`, returning what was written.

    PNG because these are compared by eye and by `git diff`, and a lossy
    format would show compression noise as though it were a change.

    Defaults to `DEMO_POCKET_OFFSET_MM` rather than to the library's own
    default - see that constant. A caller passing its own parameters gets
    them untouched, including a 1mm offset if that is what it wants.
    """
    params = params or LayoutParameters(pocket_offset=DEMO_POCKET_OFFSET_MM)
    wanted = set(names) if names else None

    written = []
    os.makedirs(out, exist_ok=True)
    for name, render in PREVIEWS:
        if wanted is not None and name not in wanted:
            continue
        path = os.path.join(out, f"{PREFIX}{name}.png")
        if not cv2.imwrite(path, render(params)):
            raise OSError(f"could not write {path}")
        written.append(path)
    return written


def Main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", default=DEFAULT_OUT, help=f"directory to write into (default: {DEFAULT_OUT})")
    parser.add_argument(
        "--only",
        action="append",
        metavar="NAME",
        choices=[name for name, _ in PREVIEWS],
        help="render just this preview; repeat for several",
    )
    args = parser.parse_args(argv)

    for path in Write(args.out, names=args.only):
        print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    sys.exit(Main())
