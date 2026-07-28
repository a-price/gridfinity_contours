"""Tests for layout_demo.

The demo exists to be re-run, so what these check is that the command in
the README still works end to end - not that any particular arrangement
comes out, which is the solver's business and is pinned by its own tests.
"""

import numpy as np
import pytest
from PIL import Image, ImageSequence

import layout_demo
from layout_demo import Recorder
from pipeline.layout.descent import RELAXING, Snapshot
from pipeline.layout.loading import BuildParts, ReadContours
from pipeline.layout.packer import Pack
from pipeline.layout.parameters import LayoutParameters
from pipeline.layout.placement import Placement
from pipeline.layout.render import RenderLayout

SPOONS = ["test_data/small_spoon.svg", "test_data/medium_spoon.svg"]


def _quick() -> LayoutParameters:
    """Small enough to run in a test, large enough to actually settle."""
    return LayoutParameters(restarts=2, iterations=60, patience=15)


def _parts():
    params = _quick()
    return BuildParts(ReadContours(SPOONS), params), params


def _snapshot(iteration: int, placements: dict[int, Placement]) -> Snapshot:
    return Snapshot((5, 2), 0, RELAXING, iteration, placements, 1.0)


def test_the_readme_command_writes_a_playable_gif(tmp_path):
    out = tmp_path / "pack.gif"
    code = layout_demo.Main(
        ["pack", *SPOONS, "--out", str(out), "--restarts", "2", "--every", "6", "--pixels-per-mm", "1.5"]
    )

    assert code == 0
    with Image.open(out) as gif:
        assert gif.format == "GIF"
        # More than one, or it is a picture rather than an animation - and a
        # single frame is exactly what a silently-broken observer produces.
        assert sum(1 for _ in ImageSequence.Iterator(gif)) > 1


def test_the_last_frame_is_the_layout_that_was_solved(tmp_path):
    """The animation must not end one step short of its own answer.

    `Spread` returns the best arrangement it saw rather than its last, so
    the final snapshot is generally *not* the returned layout - which is
    why the demo appends the result rather than trusting the last frame.
    """
    parts, params = _parts()
    recorder = Recorder(parts, params, pixels_per_mm=1.5, every=6)
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


def test_sampling_keeps_one_frame_in_every_n():
    parts, params = _parts()
    placements = {part_id: Placement(part_id, np.array([10.0, 10.0]), 0) for part_id in parts}
    recorder = Recorder(parts, params, pixels_per_mm=1.0, every=4)

    for iteration in range(12):
        recorder(_snapshot(iteration, placements))

    assert recorder.snapshots == 12
    assert len(recorder.frames) == 3


def test_recording_stops_at_the_cap_and_says_so():
    parts, params = _parts()
    placements = {part_id: Placement(part_id, np.array([10.0, 10.0]), 0) for part_id in parts}
    recorder = Recorder(parts, params, pixels_per_mm=1.0, every=1, max_frames=5)

    for iteration in range(20):
        recorder(_snapshot(iteration, placements))

    assert len(recorder.frames) == 5
    assert recorder.truncated
    assert recorder.snapshots == 20, "the cap stops recording, not the search"


def test_sampling_every_zero_is_refused():
    parts, params = _parts()
    with pytest.raises(ValueError, match="at least every iteration"):
        Recorder(parts, params, every=0)
