"""Pack contours into Gridfinity bins from the command line.

    layout_cli.py test_data/*.svg --out spoons

Takes contour files - either a JSON dump from the GUI (contour_io) or any
SVG this project has written - finds the smallest grid size they pack
into, and writes a true-scale preview to print.

Headless on purpose. Extracting contours needs a photo and a person
clicking; packing them needs neither, and the search is slow enough that
tying it to a live session would make it painful to run twice.
"""

import argparse
import sys
from typing import Sequence

import numpy as np

from pipeline.contour_io import LoadContours, SaveContours
from pipeline.layout.energy import LayoutParameters
from pipeline.layout.packer import Pack
from pipeline.layout.preview import WriteLayoutPdf, WriteLayoutSvg
from pipeline.layout.svg import BuildParts, LoadSvgContours


def ReadContours(paths: Sequence[str]) -> dict[int, np.ndarray]:
    """Every contour across the given files, renumbered from zero.

    Ids are assigned by order encountered rather than carried over from the
    inputs, because two files dumped from two sessions both start at 0 and
    silently dropping half the contours to a key collision would look
    exactly like a packing that went well.
    """
    contours: dict[int, np.ndarray] = {}
    for path in paths:
        loaded = LoadContours(path) if path.lower().endswith(".json") else dict(enumerate(LoadSvgContours(path)))
        for _, points in sorted(loaded.items()):
            contours[len(contours)] = points
    if not contours:
        raise ValueError("no contours found in the given files")
    return contours


def BuildParser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("inputs", nargs="+", metavar="FILE", help="contour dumps (.json) or SVGs to pack")
    parser.add_argument("--out", default="layout", help="output basename; writes .svg and .pdf (default: layout)")
    parser.add_argument("--dump-contours", metavar="FILE", help="also write the loaded contours as a JSON dump")
    parser.add_argument("--max-grid", type=int, default=None, help="largest grid dimension to try")
    parser.add_argument("--seed", type=int, default=None, help="search seed; a different one is a different attempt")
    parser.add_argument("--restarts", type=int, default=None, help="attempts per grid size before moving up")
    parser.add_argument(
        "--pocket-offset",
        type=float,
        default=None,
        metavar="MM",
        help="how much larger than its object each pocket is cut; sets both clearances",
    )
    parser.add_argument("--resolution", type=float, default=None, metavar="MM", help="distance field resolution")
    return parser


def ParametersFrom(args: argparse.Namespace) -> LayoutParameters:
    """A LayoutParameters with only the flags the user actually passed
    applied, so unset flags keep the tuned defaults rather than
    re-specifying them here where they would drift.
    """
    overrides = {
        "max_grid": args.max_grid,
        "seed": args.seed,
        "restarts": args.restarts,
        "pocket_offset": args.pocket_offset,
        "resolution": args.resolution,
    }
    return LayoutParameters(**{name: value for name, value in overrides.items() if value is not None})


def Main(argv: Sequence[str] | None = None) -> int:
    args = BuildParser().parse_args(argv)
    params = ParametersFrom(args)

    contours = ReadContours(args.inputs)
    if args.dump_contours:
        SaveContours(args.dump_contours, contours)
    print(f"loaded {len(contours)} contours from {len(args.inputs)} file(s)")

    parts = BuildParts(contours, params)
    result = Pack(parts, params)
    print(result.Report())

    if result.layout is None:
        return 1

    svg_path, pdf_path = f"{args.out}.svg", f"{args.out}.pdf"
    WriteLayoutSvg(svg_path, result.layout, parts)
    WriteLayoutPdf(pdf_path, result.layout, parts)

    n, m = result.layout.grid
    print(f"packed {len(parts)} parts into {n}x{m} ({result.cells} cells)")
    print(f"wrote {svg_path} and {pdf_path}")
    return 0


if __name__ == "__main__":
    sys.exit(Main())
