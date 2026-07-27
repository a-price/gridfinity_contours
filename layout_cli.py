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
import signal
import sys
from contextlib import contextmanager
from typing import Callable, Iterator, Sequence, TextIO

from pipeline.contour_io import SaveContours
from pipeline.layout.loading import BuildParts, ReadContours
from pipeline.layout.packer import Pack, Progress
from pipeline.layout.parameters import LayoutParameters
from pipeline.layout.preview import WriteLayoutPdf, WriteLayoutSvg
from pipeline.layout.solid import DEFAULT_HEIGHT_UNITS, WriteScad


def ShouldShowProgress(stream: TextIO, quiet: bool) -> bool:
    """Whether a live progress line is worth drawing.

    Off when the output is not a terminal: the line works by rewriting
    itself with a carriage return, which is unreadable noise in a log file
    or a pipe. Off when asked, for the same reason a script might want.
    """
    return not quiet and hasattr(stream, "isatty") and stream.isatty()


class ProgressLine:
    """One console line that rewrites itself as the search moves.

    Padded back out to the widest text yet written, because a shorter
    message would otherwise leave the tail of the previous one on screen -
    "attempt 9/24" over "attempt 10/24" reads as "attempt 9/244".
    """

    def __init__(self, stream: TextIO) -> None:
        self._stream = stream
        self._width = 0

    def Update(self, progress: Progress) -> None:
        text = f"  searching {progress}"
        self._stream.write("\r" + text.ljust(self._width))
        self._stream.flush()
        self._width = max(self._width, len(text))

    def Clear(self) -> None:
        """Wipe the line, so the report that follows starts clean."""
        if self._width:
            self._stream.write("\r" + " " * self._width + "\r")
            self._stream.flush()
            self._width = 0


@contextmanager
def Interruptible() -> Iterator[Callable[[], bool]]:
    """Turn the first Ctrl-C into a request to stop, not a traceback.

    Yields the predicate to hand `Pack` as `cancelled`. A search stopped
    this way still returns everything it learned, so the sizes already
    ruled out get reported instead of thrown away.

    The default handler is restored immediately, so a second Ctrl-C - from
    someone who means it - interrupts as usual rather than being swallowed
    by a search that is between polls.
    """
    stopped = False

    def handle(signum, frame) -> None:
        nonlocal stopped
        stopped = True
        signal.signal(signal.SIGINT, previous)
        print("\nstopping - press Ctrl-C again to quit now", flush=True)

    previous = signal.signal(signal.SIGINT, handle)
    try:
        yield lambda: stopped
    finally:
        signal.signal(signal.SIGINT, previous)


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
    parser.add_argument("--quiet", action="store_true", help="suppress the live progress line")
    parser.add_argument(
        "--height",
        type=int,
        default=DEFAULT_HEIGHT_UNITS,
        metavar="UNITS",
        help=f"bin height in 7mm Gridfinity units (default: {DEFAULT_HEIGHT_UNITS})",
    )
    parser.add_argument("--no-scad", action="store_true", help="write only the preview, not the OpenSCAD bin")
    parser.add_argument(
        "--solid-offset",
        type=float,
        default=None,
        metavar="MM",
        help="cut the pockets at a different tolerance than the layout was packed with (default: the same)",
    )
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
    line = ProgressLine(sys.stdout) if ShouldShowProgress(sys.stdout, args.quiet) else None

    with Interruptible() as interrupted:
        result = Pack(
            parts,
            params,
            progress=None if line is None else line.Update,
            cancelled=interrupted,
        )
    if line is not None:
        line.Clear()

    print(result.Report())

    if result.cancelled:
        # 130 is the shell's convention for "killed by SIGINT", so a script
        # wrapping this can tell "you stopped it" from "it did not fit".
        return 130
    if result.layout is None:
        return 1

    written = [f"{args.out}.svg", f"{args.out}.pdf"]
    WriteLayoutSvg(written[0], result.layout, parts)
    WriteLayoutPdf(written[1], result.layout, parts)

    n, m = result.layout.grid
    print(f"packed {len(parts)} parts into {n}x{m} ({result.cells} cells)")

    if not args.no_scad:
        scad_path = f"{args.out}.scad"
        # The tolerance the pockets are cut at is a property of the printer,
        # not of the arrangement, so it can differ from the one the layout
        # reserved room for - within what that layout actually left.
        offset = params.pocket_offset if args.solid_offset is None else args.solid_offset
        try:
            WriteScad(scad_path, result.layout, parts, pocket_offset=offset, height_units=args.height)
            written.append(scad_path)
        except ValueError as error:
            # The layout is still good and its preview is already written;
            # only the solid could not be cut at this tolerance.
            print(f"could not generate the solid: {error}")

    print("wrote " + ", ".join(written))
    return 0


if __name__ == "__main__":
    sys.exit(Main())
