"""Tests for the headless packing CLI (M5).

The end-to-end path matters more than any one function here: the point of
the milestone is that a real captured contour set packs from a command
line and produces something printable.
"""

import io
import os
import re
import signal
from unittest import mock

import matplotlib
import numpy as np
import pytest

matplotlib.use("Agg")  # headless: no window should pop up when saving the PDF

from layout_cli import (
    BuildParser,
    Interruptible,
    Main,
    ParametersFrom,
    ProgressLine,
    ReadContours,
    ShouldShowProgress,
)
from export.contour_io import LoadContours, SaveContours
from pipeline.layout.container import DIVIDER_WIDTH_MM, MIN_WALL_MM
from pipeline.layout.loading import LoadSvgContours
from pipeline.layout.packer import Progress
from pipeline.layout.parameters import LayoutParameters
from conftest import Rectangle as _rectangle, SPOONS

# ------------------------------------------------------------------ inputs


def test_contours_load_from_an_svg():
    contours = ReadContours(["test_data/big_spoon.svg"])

    assert len(contours) == 1
    assert contours[0].shape[1] == 2


def test_contours_load_from_a_json_dump(tmp_path):
    path = str(tmp_path / "dump.json")
    SaveContours(path, {0: _rectangle(20.0, 10.0), 1: _rectangle(8.0, 4.0)})

    contours = ReadContours([path])

    assert len(contours) == 2


def test_the_two_formats_can_be_mixed_in_one_run(tmp_path):
    path = str(tmp_path / "dump.json")
    SaveContours(path, {0: _rectangle(20.0, 10.0)})

    contours = ReadContours([path, "test_data/small_spoon.svg"])

    assert len(contours) == 2


def test_contours_are_renumbered_across_files(tmp_path):
    """Two dumps from two sessions both start at id 0. Carrying ids over
    would drop one of every colliding pair, which looks exactly like a
    packing that went unusually well.
    """
    first, second = str(tmp_path / "a.json"), str(tmp_path / "b.json")
    SaveContours(first, {0: _rectangle(20.0, 10.0)})
    SaveContours(second, {0: _rectangle(30.0, 12.0)})

    contours = ReadContours([first, second])

    assert sorted(contours) == [0, 1]


def test_reading_a_file_with_no_contours_raises(tmp_path):
    empty = tmp_path / "empty.svg"
    empty.write_text('<?xml version="1.0"?><svg xmlns="http://www.w3.org/2000/svg" width="1mm" viewBox="0 0 1 1"/>')

    with pytest.raises(ValueError):
        ReadContours([str(empty)])


# -------------------------------------------------------------- parameters


def test_unset_flags_keep_the_tuned_defaults():
    # Restating the defaults in the CLI is how they drift apart; only what
    # the user actually passed should override.
    args = BuildParser().parse_args(["in.svg"])

    assert ParametersFrom(args) == LayoutParameters()


def test_passed_flags_override():
    args = BuildParser().parse_args(["in.svg", "--seed", "7", "--max-grid", "3", "--pocket-offset", "0.5"])

    params = ParametersFrom(args)

    assert (params.seed, params.max_grid, params.pocket_offset) == (7, 3, 0.5)


def test_pocket_offset_no_longer_moves_the_clearances():
    """It used to set both, because a part was an object and the room its
    pocket would need had to be reserved in the clearance. The parts are
    pockets now, so the offset is already in the shape being packed and
    the clearances are the printable divider and wall, flat. Adding it
    here as well would count it twice.
    """
    args = BuildParser().parse_args(["in.svg", "--pocket-offset", "2.0"])

    params = ParametersFrom(args)

    assert params.pocket_offset == pytest.approx(2.0)
    assert params.c_pair == pytest.approx(DIVIDER_WIDTH_MM)
    assert params.c_wall == pytest.approx(MIN_WALL_MM)


# ---------------------------------------------------------------- end to end


def test_a_small_set_packs_and_writes_both_files(tmp_path, capsys):
    out = str(tmp_path / "layout")
    dump = str(tmp_path / "dump.json")
    SaveContours(dump, {0: _rectangle(20.0, 10.0), 1: _rectangle(18.0, 12.0)})

    assert Main([dump, "--out", out, "--max-grid", "2"]) == 0

    assert (tmp_path / "layout.svg").exists()
    assert (tmp_path / "layout.pdf").exists()
    assert (tmp_path / "layout.pdf").read_bytes().startswith(b"%PDF")
    assert "packed 2 parts" in capsys.readouterr().out


def test_the_written_preview_holds_one_polygon_per_part(tmp_path):
    out = str(tmp_path / "layout")
    dump = str(tmp_path / "dump.json")
    SaveContours(dump, {0: _rectangle(20.0, 10.0), 1: _rectangle(18.0, 12.0)})

    Main([dump, "--out", out, "--max-grid", "2"])

    assert len(LoadSvgContours(f"{out}.svg")) == 2


def test_a_set_that_cannot_fit_reports_failure_without_writing(tmp_path, capsys):
    # Exit status is what a script reads; a preview written anyway would be
    # a picture of nothing.
    out = str(tmp_path / "layout")
    dump = str(tmp_path / "dump.json")
    SaveContours(dump, {0: _rectangle(500.0, 400.0)})

    assert Main([dump, "--out", out, "--max-grid", "2"]) == 1

    assert not (tmp_path / "layout.svg").exists()
    assert "too small" in capsys.readouterr().out


def test_the_report_names_every_size_it_tried(tmp_path, capsys):
    out = str(tmp_path / "layout")
    dump = str(tmp_path / "dump.json")
    SaveContours(dump, {0: _rectangle(60.0, 30.0)})

    Main([dump, "--out", out, "--max-grid", "2"])

    printed = capsys.readouterr().out
    assert "1x1" in printed and "2x1" in printed


def test_contours_can_be_dumped_on_the_way_through(tmp_path):
    """The dump is what makes a clicked-once photo reusable offline, so it
    has to be writable from the same run that reads the SVGs.
    """
    out = str(tmp_path / "layout")
    dump = str(tmp_path / "dump.json")

    Main(["test_data/small_spoon.svg", "--out", out, "--dump-contours", dump, "--max-grid", "2"])

    assert len(LoadContours(dump)) == 1


@pytest.mark.slow
def test_the_real_spoons_pack_from_the_command_line(tmp_path, capsys):
    """M5's own done-when: a real captured contour set, packed headlessly."""
    out = str(tmp_path / "spoons")

    assert Main([*SPOONS, "--out", out]) == 0

    printed = capsys.readouterr().out
    assert "loaded 3 contours" in printed
    assert "packed 3 parts into 5x2" in printed
    assert len(LoadSvgContours(f"{out}.svg")) == 3


# ----------------------------------------------------------------- progress


class _FakeTty(io.StringIO):
    """A real text stream that claims to be (or not to be) a terminal."""

    def __init__(self, tty: bool = True) -> None:
        super().__init__()
        self._tty = tty

    def isatty(self) -> bool:
        return self._tty


def _progress(grid=(2, 1), attempt=0, restarts=24, grids_tried=3):
    return Progress(grid, attempt, restarts, grids_tried)


def _segments(stream: _FakeTty) -> list[str]:
    """What was written between carriage returns - one entry per redraw."""
    return stream.getvalue().split("\r")[1:]


def test_progress_is_shown_on_a_terminal_and_not_in_a_pipe():
    """The line works by rewriting itself with a carriage return, which is
    unreadable noise in a log file.
    """
    assert ShouldShowProgress(_FakeTty(tty=True), quiet=False)
    assert not ShouldShowProgress(_FakeTty(tty=False), quiet=False)
    assert not ShouldShowProgress(_FakeTty(tty=True), quiet=True)


def test_progress_rewrites_one_line_rather_than_scrolling():
    # 24 restarts across a dozen candidate sizes would otherwise bury the
    # report under hundreds of lines.
    stream = _FakeTty()
    line = ProgressLine(stream)

    line.Update(_progress(attempt=0))
    line.Update(_progress(attempt=1))

    assert "\n" not in stream.getvalue()
    assert len(_segments(stream)) == 2
    assert "attempt 2/24" in _segments(stream)[-1]


def test_a_shorter_message_does_not_leave_the_tail_of_a_longer_one():
    """ "attempt 9/24" written over "attempt 10/24" would read as
    "attempt 9/244".
    """
    stream = _FakeTty()
    line = ProgressLine(stream)

    line.Update(_progress(attempt=9, grids_tried=100))
    line.Update(_progress(attempt=8, grids_tried=1))

    longer, shorter = _segments(stream)
    assert shorter.strip() != longer.strip(), "the two redraws should differ"
    assert len(shorter) == len(longer), "the shorter message must be padded over the longer one"


def test_clearing_wipes_the_line_for_the_report():
    stream = _FakeTty()
    line = ProgressLine(stream)
    line.Update(_progress())

    line.Clear()

    drawn, blanked, after = _segments(stream)
    assert blanked.strip() == "", "the line should be overwritten with spaces"
    assert len(blanked) == len(drawn), "and be wide enough to cover what was there"
    assert after == "", "leaving the cursor at the start of a clean line"


def test_clearing_a_line_that_never_drew_writes_nothing():
    stream = _FakeTty()

    ProgressLine(stream).Clear()

    assert stream.getvalue() == ""


def test_a_piped_run_emits_no_control_characters(tmp_path, capsys):
    # capsys' replacement stdout is not a tty, which is exactly the case
    # this checks.
    dump = str(tmp_path / "dump.json")
    SaveContours(dump, {0: _rectangle(20.0, 10.0)})

    Main([dump, "--out", str(tmp_path / "layout"), "--max-grid", "2"])

    assert "\r" not in capsys.readouterr().out


# -------------------------------------------------------------- interruption


def test_an_interrupted_search_reports_what_it_learned(tmp_path, capsys):
    """A cancelled search still knows which sizes it ruled out, and those
    are facts worth printing rather than throwing away.
    """
    dump = str(tmp_path / "dump.json")
    SaveContours(dump, {0: _rectangle(60.0, 25.0)})

    with Interruptible() as interrupted:
        assert not interrupted()
        os.kill(os.getpid(), signal.SIGINT)
        assert interrupted(), "the first Ctrl-C should ask the search to stop, not raise"

    assert "stopping" in capsys.readouterr().out


def test_the_handler_is_restored_afterwards():
    before = signal.getsignal(signal.SIGINT)

    with Interruptible():
        pass

    assert signal.getsignal(signal.SIGINT) is before


def test_a_second_interrupt_is_not_swallowed():
    """Someone pressing Ctrl-C twice means it, and a search between polls
    would otherwise ignore them.
    """
    with Interruptible():
        os.kill(os.getpid(), signal.SIGINT)
        with pytest.raises(KeyboardInterrupt):
            os.kill(os.getpid(), signal.SIGINT)


def test_a_cancelled_run_exits_with_the_signal_convention(tmp_path):
    """130 lets a wrapping script tell "you stopped it" from "it did not
    fit", which 1 would conflate.
    """
    dump = str(tmp_path / "dump.json")
    SaveContours(dump, {0: _rectangle(20.0, 10.0)})
    out = str(tmp_path / "layout")

    with mock.patch("layout_cli.Interruptible") as interruptible:
        interruptible.return_value.__enter__ = lambda self: (lambda: True)
        interruptible.return_value.__exit__ = lambda self, *args: None
        status = Main([dump, "--out", out, "--max-grid", "2"])

    assert status == 130
    assert not (tmp_path / "layout.svg").exists()


# -------------------------------------------------------------------- solid


def test_a_pack_also_writes_the_openscad_bin(tmp_path):
    out = str(tmp_path / "layout")
    dump = str(tmp_path / "dump.json")
    SaveContours(dump, {0: _rectangle(20.0, 10.0)})

    Main([dump, "--out", out, "--max-grid", "2"])

    assert "bin_render(bin)" in (tmp_path / "layout.scad").read_text()


def test_the_solid_can_be_skipped(tmp_path):
    out = str(tmp_path / "layout")
    dump = str(tmp_path / "dump.json")
    SaveContours(dump, {0: _rectangle(20.0, 10.0)})

    Main([dump, "--out", out, "--max-grid", "2", "--no-scad"])

    assert not (tmp_path / "layout.scad").exists()
    assert (tmp_path / "layout.svg").exists()


def test_the_bin_height_is_settable(tmp_path):
    out = str(tmp_path / "layout")
    dump = str(tmp_path / "dump.json")
    SaveContours(dump, {0: _rectangle(20.0, 10.0)})

    Main([dump, "--out", out, "--max-grid", "2", "--height", "6"])

    assert "height_mm = 42.0000" in (tmp_path / "layout.scad").read_text()


def test_a_solid_that_cannot_be_cut_does_not_lose_the_preview(tmp_path, capsys):
    """The layout is still good and the preview still worth having - only
    the tolerance is unbuildable, so that alone should be reported.

    Uses --solid-offset rather than --pocket-offset: the latter widens the
    clearances the layout packs to as well, so the arrangement would just
    make room and the guard would never fire.
    """
    out = str(tmp_path / "layout")
    dump = str(tmp_path / "dump.json")
    SaveContours(dump, {0: _rectangle(20.0, 10.0), 1: _rectangle(18.0, 12.0)})

    assert Main([dump, "--out", out, "--max-grid", "2", "--solid-offset", "6.0"]) == 0

    assert (tmp_path / "layout.svg").exists()
    assert not (tmp_path / "layout.scad").exists()
    assert "could not generate the solid" in capsys.readouterr().out


def _PocketExtents(scad: str) -> list[float]:
    """The longer bounding-box side of every `polygon(...)` in a .scad, in
    mm - a rotation-proof stand-in for "how big did that pocket come out",
    since a quarter turn swaps which axis is which.
    """
    extents = []
    for body in re.findall(r"polygon\(points=\[(.*?)\]\)", scad):
        points = np.array([[float(x), float(y)] for x, y in re.findall(r"\[(-?[\d.]+), (-?[\d.]+)\]", body)])
        extents.append(float((points.max(axis=0) - points.min(axis=0)).max()))
    return extents


def test_the_solid_tolerance_can_differ_from_the_layouts(tmp_path):
    """The freedom M7 exists to preserve: the tolerance is a property of
    the printer, so changing it re-cuts the solid instead of invalidating
    the arrangement.
    """
    out = str(tmp_path / "layout")
    dump = str(tmp_path / "dump.json")
    SaveContours(dump, {0: _rectangle(20.0, 10.0), 1: _rectangle(18.0, 12.0)})

    Main([dump, "--out", out, "--max-grid", "2", "--pocket-offset", "1.0", "--solid-offset", "0.4"])

    # No `offset(r = ...)` to look for any more - the pocket is the polygon.
    # So the check is geometric: cut at 0.4 rather than the 1.0 it packed
    # at, each pocket is 1.2mm narrower across than the one the layout
    # reserved room for.
    scad = (tmp_path / "layout.scad").read_text()
    assert "offset(" not in scad
    assert sorted(_PocketExtents(scad)) == pytest.approx([18.0 + 0.8, 20.0 + 0.8], abs=0.15)
