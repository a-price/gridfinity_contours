"""Tests for layout_demo.

The demo exists to be re-run, so what these check is that the commands in
the README still work end to end - not that any particular arrangement
comes out, which is the solver's business and is pinned by its own tests.
"""

import argparse
import numpy as np
import pytest
from PIL import Image, ImageSequence

import demos.layout_demo as layout_demo
from demos.layout_demo import (
    BuildParser,
    DrawerRecorder,
    GroupRecorder,
    ParametersFrom,
    PackRecorder,
    ParseDrawer,
    Recording,
)
from pipeline.layout.descent import RELAXING, Snapshot
from pipeline.layout.drawer import PLACED, AssignmentResult, Drawer, Slot, Trial
from pipeline.layout.loading import BuildParts, ReadContours
from pipeline.layout.packer import Pack
from pipeline.layout.parameters import LayoutParameters
from pipeline.layout.placement import Layout, Placement
from pipeline.layout.render import RenderLayout
from conftest import QuickParameters, SPOONS


def _quick() -> LayoutParameters:
    """Small enough to run in a test, large enough to actually settle.

    Thinner than the shared budget because these tests record every frame
    of the search, so the budget sets how much rendering they do too.
    """
    return QuickParameters(restarts=2, iterations=60, patience=15)


def _parts():
    params = _quick()
    return BuildParts(ReadContours(SPOONS), params), params


def _snapshot(iteration: int, placements: dict[int, Placement]) -> Snapshot:
    return Snapshot((5, 2), 0, RELAXING, iteration, placements, 1.0)


def _frames(path) -> int:
    with Image.open(path) as gif:
        assert gif.format == "GIF"
        return sum(1 for _ in ImageSequence.Iterator(gif))


# --------------------------------------------------------------- parameters


def test_unset_flags_keep_the_tuned_defaults():
    args = BuildParser().parse_args(["pack", "in.svg"])

    assert ParametersFrom(args) == LayoutParameters()


def test_passed_flags_override():
    args = BuildParser().parse_args(["pack", "in.svg", "--seed", "7", "--max-grid", "3", "--restarts", "2"])

    params = ParametersFrom(args)

    assert (params.seed, params.max_grid, params.restarts) == (7, 3, 2)


def test_the_pack_command_writes_a_playable_gif(tmp_path):
    out = tmp_path / "pack.gif"
    code = layout_demo.Main(
        ["pack", *SPOONS, "--out", str(out), "--restarts", "2", "--every", "6", "--pixels-per-mm", "1.5"]
    )

    assert code == 0
    # More than one, or it is a picture rather than an animation - and a
    # single frame is exactly what a silently-broken observer produces.
    assert _frames(out) > 1


def test_the_drawer_command_writes_a_playable_gif(tmp_path):
    out = tmp_path / "drawer.gif"
    code = layout_demo.Main(
        [
            "drawer",
            *SPOONS,
            "--out",
            str(out),
            "--drawer",
            "260x180",
            "--restarts",
            "2",
            "--every",
            "1",
            "--pixels-per-mm",
            "0.6",
        ]
    )

    assert code == 0
    assert _frames(out) > 1


def test_the_last_pack_frame_is_the_layout_that_was_solved():
    """The animation must not end one step short of its own answer.

    `Spread` returns the best arrangement it saw rather than its last, so
    the final snapshot is generally *not* the returned layout - which is
    why the demo appends the result rather than trusting the last frame.
    """
    parts, params = _parts()
    recorder = PackRecorder(parts, params, pixels_per_mm=1.5, every=6)
    result = Pack(parts, params, observer=recorder)
    assert result.layout is not None

    recorder.Draw(result.layout)
    assert np.array_equal(recorder.frames[-1], RenderLayout(result.layout, parts, 1.5))


def test_the_observer_sees_the_search_step_up_a_size():
    """Snapshots carry the grid they belong to, not just the final one.

    This is what makes a frame drawable at all: a snapshot rendered against
    the wrong bin would show parts hanging outside their walls.
    """
    parts, params = _parts()
    grids = set()
    Pack(parts, params, observer=lambda snapshot: grids.add(snapshot.grid))

    assert grids, "the solver reported nothing"
    assert all(n >= m for n, m in grids), f"candidate grids should be n>=m, got {sorted(grids)}"


def test_both_descent_phases_are_reported():
    parts, params = _parts()
    phases = set()
    result = Pack(parts, params, observer=lambda snapshot: phases.add(snapshot.phase))

    assert result.layout is not None
    # Relaxing alone would mean the animation stops before the spacing pass
    # that produced the layout it ends on.
    assert phases == {"relaxing", "spreading"}


def test_a_drawer_frame_draws_every_drawer_side_by_side():
    """One image per frame, however many drawers there are - a bin moving
    between them is the thing worth seeing.
    """
    parts, params = _parts()
    layouts = {0: Layout(grid=(2, 1), placements={}, inset=params.inset)}
    drawers = [Drawer(3, 2), Drawer(2, 2)]

    recorder = DrawerRecorder(drawers, layouts, parts, pixels_per_mm=1.0, every=1)
    recorder(Trial(0, (Slot(0, 0, (0, 0)),)))

    single = DrawerRecorder([drawers[0]], layouts, parts, pixels_per_mm=1.0, every=1)
    single.Draw(AssignmentResult(PLACED, {0: Slot(0, 0, (0, 0))}))

    assert recorder.frames[0].shape[1] > single.frames[0].shape[1]


def test_sampling_keeps_one_frame_in_every_n():
    parts, params = _parts()
    placements = {part_id: Placement(part_id, np.array([10.0, 10.0]), 0) for part_id in parts}
    recorder = PackRecorder(parts, params, pixels_per_mm=1.0, every=4)

    for iteration in range(12):
        recorder(_snapshot(iteration, placements))

    assert recorder.steps == 12
    assert len(recorder.frames) == 3


def test_recording_stops_at_the_cap_and_says_so():
    parts, params = _parts()
    placements = {part_id: Placement(part_id, np.array([10.0, 10.0]), 0) for part_id in parts}
    recorder = PackRecorder(parts, params, pixels_per_mm=1.0, every=1, max_frames=5)

    for iteration in range(20):
        recorder(_snapshot(iteration, placements))

    assert len(recorder.frames) == 5
    assert recorder.truncated
    assert recorder.steps == 20, "the cap stops recording, not the search"


def test_the_cap_does_not_stop_a_caller_holding_on_the_answer():
    """Which is when it matters most: a truncated recording still has to
    end on the result rather than wherever it was cut off.
    """
    parts, params = _parts()
    recorder = PackRecorder(parts, params, pixels_per_mm=1.0, every=1, max_frames=2)
    placements = {part_id: Placement(part_id, np.array([10.0, 10.0]), 0) for part_id in parts}

    for iteration in range(10):
        recorder(_snapshot(iteration, placements))
    recorder.Draw(Layout(grid=(5, 2), placements=placements, inset=params.inset))

    assert len(recorder.frames) == 3


def test_sampling_every_zero_is_refused():
    with pytest.raises(ValueError, match="at least every step"):
        Recording(every=0)


def test_a_drawer_is_given_in_millimeters():
    # 42mm cells with the half-millimeter gap taken off the run as a whole,
    # not off each cell - so 210mm is five cells, not four.
    assert ParseDrawer("210x340") == Drawer(5, 8)
    assert ParseDrawer("170X130") == Drawer(4, 3)


def test_a_drawer_can_also_be_given_in_cells():
    """Shared with the window through `drawer.ParseDrawer`, so the two
    front ends cannot disagree about what a drawer is.
    """
    assert ParseDrawer("5x8 cells") == Drawer(5, 8)


@pytest.mark.parametrize("text", ["500", "500x400x300", "widexdeep", "10x10"])
def test_an_unusable_drawer_is_refused(text):
    # argparse's own exception type, so a bad flag is a usage message
    # rather than a traceback.
    with pytest.raises(argparse.ArgumentTypeError, match="drawer"):
        ParseDrawer(text)


def test_the_group_command_writes_a_playable_gif(tmp_path):
    out = tmp_path / "group.gif"
    code = layout_demo.Main(
        ["group", *SPOONS, "--out", str(out), "--restarts", "2", "--every", "1", "--pixels-per-mm", "0.6"]
    )

    assert code == 0
    assert _frames(out) > 1


def test_a_group_frame_marks_only_the_bins_being_asked_about():
    """The mark is the whole point of the grouping animation: the
    arrangement changes on the rare accepted move, so without it the frames
    would be identical for long stretches with nothing to show for the
    hundreds of candidates priced.
    """
    parts, params = _parts()
    layouts = [Layout(grid=(2, 1), placements={}, inset=params.inset) for _ in range(2)]

    recorder = GroupRecorder(parts, columns=2, pixels_per_mm=1.0, every=1)
    recorder.Draw(layouts)
    recorder.Draw(layouts, asking=frozenset([1]))

    plain, marked = recorder.frames
    assert plain.shape == marked.shape
    assert not np.array_equal(plain, marked)
    # Only the second bin is marked, so the left half is untouched.
    half = plain.shape[1] // 2
    assert np.array_equal(plain[:, :half], marked[:, :half])


def test_group_columns_stay_fixed_as_bins_disappear():
    """A column count that followed the bin count would reflow the
    survivors into different rows every time one was merged away.
    """
    parts, params = _parts()
    layouts = [Layout(grid=(2, 1), placements={}, inset=params.inset) for _ in range(4)]

    recorder = GroupRecorder(parts, columns=2, pixels_per_mm=1.0, every=1)
    recorder.Draw(layouts)
    recorder.Draw(layouts[:2])

    four, two = recorder.frames
    assert four.shape[1] == two.shape[1], "two bins should still fill one row of two"
    assert four.shape[0] > two.shape[0], "and drop from two rows to one"
