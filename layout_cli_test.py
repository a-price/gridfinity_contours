"""Tests for the headless packing CLI (M5).

The end-to-end path matters more than any one function here: the point of
the milestone is that a real captured contour set packs from a command
line and produces something printable.
"""

import matplotlib
import numpy as np
import pytest

matplotlib.use("Agg")  # headless: no window should pop up when saving the PDF

from layout_cli import BuildParser, Main, ParametersFrom, ReadContours
from pipeline.contour_io import LoadContours, SaveContours
from pipeline.layout.energy import LayoutParameters
from pipeline.layout.loading import LoadSvgContours

SPOONS = ["test_data/big_spoon.svg", "test_data/medium_spoon.svg", "test_data/small_spoon.svg"]


def _rectangle(width: float, height: float, x: float = 0.0, y: float = 0.0) -> np.ndarray:
    return np.array([[x, y], [x + width, y], [x + width, y + height], [x, y + height]], dtype=np.float64)


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


def test_pocket_offset_moves_both_clearances_together():
    # They are derived from it rather than set independently, so a divider
    # cannot be left too thin to print.
    args = BuildParser().parse_args(["in.svg", "--pocket-offset", "2.0"])

    params = ParametersFrom(args)

    assert params.c_pair == pytest.approx(5.2)
    assert params.c_wall == pytest.approx(2.95)


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
