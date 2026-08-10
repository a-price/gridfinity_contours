"""Photograph the bin itself, so the README can show what comes out.

    python3 -m demos.solid_demo --out docs/media

Every other picture in this project is a drawing: a sheet, a field, a
floorplan, a window. This one is the object. It packs real captured
contours, writes the same `.scad` `layout_cli.py` writes, and hands it to
OpenSCAD to render - so the picture at the top of the README is the actual
solid somebody would slice, pockets and stacking lip and all, rather than
an artist's impression of one.

Worth a script rather than a saved render because the pockets come from
the layout, and the layout comes from a stochastic search. A stale picture
here would show a bin that no longer matches what the tool produces, and
would do it convincingly.

**Needs OpenSCAD on the path**, and the `gridfinity-rebuilt-openscad`
submodule checked out - the same two things `solid.py` has always needed,
and the reason this is not part of `make check`. Rendering is CGAL rather
than the fast preview, which is slower and is the point: the preview
renderer shows subtracted volumes with their surfaces fighting, and a
picture of a bin whose pocket flickers is worse than no picture.
"""

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
from typing import Sequence

import cv2
import numpy as np

from pipeline.layout.loading import BuildParts, ReadContours
from pipeline.layout.packer import Pack
from pipeline.layout.parameters import LayoutParameters
from pipeline.layout.solid import WriteScad

DEFAULT_OUT = "docs/media"

# Prefix, matching `render_demo.py`'s drawings and `screenshot_demo.py`'s
# windows, so everything the README uses sorts together in one directory.
PREFIX = "solid_"

OPENSCAD = "openscad"

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _Path(name: str) -> str:
    """A fixture path resolved against the repository root - one level up
    from `demos/` - so the render comes out the same wherever it was
    started from.
    """
    return os.path.join(_ROOT, "test_data", name)


# The three spoons the README's `layout_cli.py` example packs, so the
# picture is the bin that transcript describes rather than a different one
# that merely looks similar.
SPOONS = (_Path("small_spoon.svg"), _Path("medium_spoon.svg"), _Path("big_spoon.svg"))

# One object, which is the case `solid.py` covers on its own.
ONE = (_Path("big_spoon.svg"),)

# Big enough to read the pocket walls at README width. OpenSCAD renders
# offscreen into this, so it costs time rather than screen.
IMAGE = (1200, 900)

# OpenSCAD's own light scheme. The default is a dark background, which in
# a README that may be read in either theme reads as a hole in the page.
COLORSCHEME = "Tomorrow"

# What to leave round the bin after cropping, in pixels, and how far a
# pixel has to be from the background colour to count as part of the bin.
# The background is flat, so the tolerance only has to survive PNG
# quantization - it is not trying to be clever about edges.
TRIM_MARGIN_PX = 16
TRIM_TOLERANCE = 6

# One entry per picture: what to pack, and what to call the result. A
# table for the same reason `render_demo.PREVIEWS` is one - adding a
# picture should be a line, and the test that keeps the set honest walks
# it.
SOLIDS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("bin", SPOONS),
    ("pocket", ONE),
)


def Available() -> bool:
    """Whether OpenSCAD can be found. Checked rather than assumed, so the
    failure is one sentence instead of a `FileNotFoundError` from inside
    a subprocess call.
    """
    return shutil.which(OPENSCAD) is not None


def WriteSolid(scad_path: str, paths: Sequence[str], parameters: LayoutParameters | None = None) -> None:
    """Pack `paths` and write the bin that holds them to `scad_path`.

    The same three calls `layout_cli.py` makes, deliberately - a picture
    generated through a shortcut would stop being a picture of the tool.
    """
    parameters = parameters or LayoutParameters()

    parts = BuildParts(ReadContours(list(paths)), parameters)
    result = Pack(parts, parameters)
    if result.layout is None:
        raise ValueError(f"nothing packed: {result.Report().splitlines()[-1]}")

    WriteScad(scad_path, result.layout, parts, pocket_offset=parameters.pocket_offset)


def RenderSolid(scad_path: str, image_path: str, size: tuple[int, int] = IMAGE) -> None:
    """Render a `.scad` to a PNG with OpenSCAD.

    `--viewall --autocenter` rather than a hand-written camera: the bins
    differ in size by a factor of three, and a fixed camera framing the
    5x2 one would cut the corners off anything larger.
    """
    if not Available():
        raise OSError(f"{OPENSCAD} is not on the path; it renders the bin pictures, and only those")

    subprocess.run(
        [
            OPENSCAD,
            "-o",
            image_path,
            "--render",  # CGAL, not the preview - see the module docstring
            "--autocenter",
            "--viewall",
            f"--imgsize={size[0]},{size[1]}",
            f"--colorscheme={COLORSCHEME}",
            scad_path,
        ],
        check=True,
        capture_output=True,
        text=True,
    )


def Trim(image: np.ndarray, margin: int = TRIM_MARGIN_PX) -> np.ndarray:
    """Crop the flat background OpenSCAD leaves round the bin.

    `--viewall` frames the object with room to spare on all four sides,
    and a bin is a wide flat thing, so most of a square render is empty
    page - at README width that shrinks the pockets to nothing. Cropping
    is framing rather than retouching: nothing about the bin changes, it
    just stops being surrounded.

    An image that is entirely background comes back untouched, since a
    zero-size crop would be harder to diagnose than a blank picture.
    """
    background = image[0, 0].astype(int)
    solid = np.any(np.abs(image.astype(int) - background) > TRIM_TOLERANCE, axis=2)

    rows = np.flatnonzero(solid.any(axis=1))
    columns = np.flatnonzero(solid.any(axis=0))
    if not rows.size or not columns.size:
        return image

    height, width = solid.shape
    top, bottom = max(0, rows[0] - margin), min(height, rows[-1] + 1 + margin)
    left, right = max(0, columns[0] - margin), min(width, columns[-1] + 1 + margin)
    return image[top:bottom, left:right]


def Write(out: str, names: Sequence[str] | None = None) -> list[str]:
    """Render every solid into `out`, returning the pictures written.

    The `.scad` is a temporary file rather than a fourth artifact beside
    the picture. `solid.GenerateScad` writes the library include as an
    absolute path - deliberately, so the file renders from any directory -
    which makes it specific to the machine that generated it. Committed,
    it would show up as a diff on every checkout that regenerated it.
    Anyone wanting the program itself gets a portable one from
    `layout_cli.py`.
    """
    wanted = set(names) if names else None

    written = []
    os.makedirs(out, exist_ok=True)
    with tempfile.TemporaryDirectory() as scratch:
        for name, paths in SOLIDS:
            if wanted is not None and name not in wanted:
                continue

            scad_path = os.path.join(scratch, f"{name}.scad")
            image_path = os.path.join(out, f"{PREFIX}{name}.png")
            WriteSolid(scad_path, paths)
            RenderSolid(scad_path, image_path)

            image = cv2.imread(image_path)
            if image is None:
                raise OSError(f"{OPENSCAD} wrote nothing readable to {image_path}")
            if not cv2.imwrite(image_path, Trim(image)):
                raise OSError(f"could not write {image_path}")

            written.append(image_path)
    return written


def Main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", default=DEFAULT_OUT, help=f"directory to write into (default: {DEFAULT_OUT})")
    parser.add_argument(
        "--only",
        action="append",
        metavar="NAME",
        choices=[name for name, _ in SOLIDS],
        help="render just this solid; repeat for several",
    )
    args = parser.parse_args(argv)

    if not Available():
        print(f"{OPENSCAD} is not on the path - install it, or skip this target", file=sys.stderr)
        return 1

    for path in Write(args.out, names=args.only):
        print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    sys.exit(Main())
