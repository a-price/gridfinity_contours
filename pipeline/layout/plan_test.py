"""Tests for the whole-stack plan: parts to bins to drawers.

Two things carry the value here. The feedback edge - grouping must never
be allowed to propose a bin no drawer could hold - and progress, which is
not decoration at this level: the search runs for minutes, so "what would
I get if I stopped now" has to be a real answer at every moment rather
than something only the end produces.
"""

import json

import pytest

from pipeline.layout.drawer import Drawer
from pipeline.layout.loading import BuildParts
from pipeline.layout.plan import (
    ASSIGNING,
    DRAWER_FORMAT_VERSION,
    FILLING,
    GROUPING,
    BuildPlan,
    Progress,
    ReadDrawers,
    SaveDrawers,
)
from conftest import QuickParameters as _quick, Rectangle as _rectangle


def _parts(count: int = 4, params=None):
    params = params or _quick(max_grid=3)
    return BuildParts({index: _rectangle(60.0 - 8.0 * index, 30.0) for index in range(count)}, params), params


# ------------------------------------------------------------ drawer files


def test_a_drawer_list_round_trips(tmp_path):
    path = str(tmp_path / "drawers.json")
    drawers = [Drawer(11, 9), Drawer(4, 3)]

    SaveDrawers(path, drawers)

    assert ReadDrawers(path) == drawers


def test_a_drawer_file_records_cells_not_millimetres(tmp_path):
    """The mm-to-cells conversion is one-way and lossy, so millimetres in
    the file would record a number the system never uses again - and two
    drawers that behave identically would be stored differently.
    """
    path = str(tmp_path / "drawers.json")
    SaveDrawers(path, [Drawer(11, 9)])

    payload = json.loads((tmp_path / "drawers.json").read_text())

    assert payload["units"] == "cells"
    assert payload["version"] == DRAWER_FORMAT_VERSION
    assert payload["drawers"] == [{"width": 11, "height": 9}]


def test_saving_no_drawers_is_refused(tmp_path):
    with pytest.raises(ValueError, match="no drawers"):
        SaveDrawers(str(tmp_path / "drawers.json"), [])


def test_a_file_from_a_future_format_is_refused(tmp_path):
    path = tmp_path / "drawers.json"
    path.write_text(json.dumps({"version": 99, "units": "cells", "drawers": [{"width": 2, "height": 2}]}))

    with pytest.raises(ValueError, match="format version"):
        ReadDrawers(str(path))


def test_a_file_in_the_wrong_units_is_refused(tmp_path):
    """The whole risk with a drawer file is reading one kind of number as
    another: 500x400 is a fine drawer in mm and an absurd one in cells.
    """
    path = tmp_path / "drawers.json"
    path.write_text(json.dumps({"version": DRAWER_FORMAT_VERSION, "units": "mm", "drawers": [{"width": 500}]}))

    with pytest.raises(ValueError, match="expected 'cells'"):
        ReadDrawers(str(path))


@pytest.mark.parametrize(
    "entry",
    [
        {"width": 2.5, "height": 2},
        {"width": True, "height": 2},
        {"width": 2},
        {"width": 0, "height": 2},
        [2, 2],
    ],
)
def test_a_drawer_that_is_not_whole_cells_is_refused(tmp_path, entry):
    """Everything above this shifts bitmasks by these integers, so a bad
    one has to be caught here rather than surfacing as an empty search.
    """
    path = tmp_path / "drawers.json"
    path.write_text(json.dumps({"version": DRAWER_FORMAT_VERSION, "units": "cells", "drawers": [entry]}))

    with pytest.raises(ValueError, match="drawer 0"):
        ReadDrawers(str(path))


def test_an_empty_drawer_list_is_refused(tmp_path):
    path = tmp_path / "drawers.json"
    path.write_text(json.dumps({"version": DRAWER_FORMAT_VERSION, "units": "cells", "drawers": []}))

    with pytest.raises(ValueError, match="no drawers"):
        ReadDrawers(str(path))


# ------------------------------------------------------------ the feedback edge


def test_grouping_is_restricted_to_footprints_a_drawer_can_hold():
    """The edge the architecture describes: a bin longer than the drawer
    is wasted work at every level below, so it is ruled out before the
    stochastic search ever sees it.
    """
    parts, params = _parts(2, _quick(max_grid=6))
    drawers = [Drawer(2, 2)]

    plan = BuildPlan(parts, drawers, params)

    for layout in plan.layouts.values():
        assert drawers[0].Holds(layout.grid), f"{layout.grid} cannot go in a 2x2 drawer"


def test_an_explicit_admissible_set_is_not_widened():
    """A caller who has already narrowed the footprints has made a
    stronger statement than this one can; quietly widening it back out
    would let a bin they had ruled out reappear.

    Pinned to 2x2 rather than 1x1 so the test can tell obedience from
    coincidence: these parts would otherwise pack into a single cell, so
    an ignored restriction would still produce a plausible answer.
    """
    params = _quick(max_grid=3, admissible_grids=frozenset({(2, 2)}))
    parts = BuildParts({0: _rectangle(20.0, 20.0)}, params)

    plan = BuildPlan(parts, [Drawer(6, 6)], params)

    assert [layout.grid for layout in plan.layouts.values()] == [(2, 2)]


# ------------------------------------------------------------------- planning


def test_a_plan_places_every_bin_when_there_is_room():
    parts, params = _parts(4)

    plan = BuildPlan(parts, [Drawer(6, 6)], params)

    assert plan.placed
    assert plan.assignment is not None
    assert set(plan.assignment.slots) == set(plan.layouts)
    assert not plan.cancelled


def test_a_plan_reports_where_everything_went():
    parts, params = _parts(3)

    report = BuildPlan(parts, [Drawer(6, 6)], params).Report()

    assert "bins" in report
    assert "cells free" in report
    assert "in one piece" in report, "free space is useless unless it is connected"


def test_planning_with_no_parts_or_no_drawers_is_refused():
    parts, params = _parts(2)

    with pytest.raises(ValueError, match="nothing to plan"):
        BuildPlan({}, [Drawer(4, 4)], params)
    with pytest.raises(ValueError, match="no drawers"):
        BuildPlan(parts, [], params)


def test_bins_that_do_not_all_fit_are_reported_not_raised():
    """Bins that individually could go in the drawer but together cannot
    is an answer, not an error - and the one answer that justifies buying
    another drawer. Only this level can state it as a fact.
    """
    parts, params = _parts(4)

    plan = BuildPlan(parts, [Drawer(3, 1)], params)

    assert not plan.placed
    assert plan.assignment is not None
    assert plan.assignment.detail


def test_a_drawer_too_small_for_any_bin_refuses_before_searching():
    """Different from the above, and worth keeping apart: here no bin the
    parts could ever occupy is storable at all, so the feedback edge
    leaves grouping with nothing to choose from. That is a broken request
    rather than a negative answer, and it says which part caused it.
    """
    parts, params = _parts(4)

    with pytest.raises(ValueError, match="does not fit"):
        BuildPlan(parts, [Drawer(1, 1)], params)


# --------------------------------------------------------------- resuming


def test_resuming_leaves_an_unimproved_bin_byte_identical():
    """The stability the resume flow rests on, and it is a property of
    `Improve` rather than of anything in `plan`: a bin no accepted move
    touched is carried through as the very same `Layout`. That is what
    makes adding one tool a matter of printing one bin.
    """
    parts, params = _parts(4)
    first = BuildPlan(parts, [Drawer(6, 6)], params)

    resumed = BuildPlan(parts, [Drawer(6, 6)], params, start=first.grouping)

    assert first.grouping is not None and resumed.grouping is not None
    kept = [layout for layout in resumed.grouping.bins if any(layout is before for before in first.grouping.bins)]
    assert kept, "resuming an already-settled grouping should change nothing"


def test_resuming_opens_a_bin_for_a_part_it_does_not_know():
    parts, params = _parts(4)
    first = BuildPlan(parts, [Drawer(6, 6)], params)
    assert first.grouping is not None

    grown = dict(parts)
    grown[99] = BuildParts({99: _rectangle(50.0, 25.0)}, params)[99]
    resumed = BuildPlan(grown, [Drawer(6, 6)], params, start=first.grouping)

    assert resumed.grouping is not None
    assert resumed.grouping.PartIds() == set(grown)


def test_resuming_never_loses_a_part():
    parts, params = _parts(3)
    first = BuildPlan(parts, [Drawer(6, 6)], params)
    assert first.grouping is not None

    resumed = BuildPlan(parts, [Drawer(6, 6)], params, start=first.grouping)

    assert resumed.grouping is not None
    assert resumed.grouping.PartIds() == set(parts)


def test_resuming_reports_the_floorplan_it_started_from():
    """Pressing Plan on a reloaded session must not blank the picture. The
    arrangement being resumed is already an answer, so it is the best so
    far from the first report - never a FILLING fragment, since first fit
    does not run at all here.
    """
    parts, params = _parts(4)
    first = BuildPlan(parts, [Drawer(6, 6)], params)
    seen: list[Progress] = []

    BuildPlan(parts, [Drawer(6, 6)], params, report=seen.append, interval=0.0, start=first.grouping)

    assert seen
    assert seen[0].phase == GROUPING
    assert seen[0].placed == len(parts), "every object accounted for from the first frame"
    assert not any(progress.phase == FILLING for progress in seen)


def test_a_resumed_report_counts_a_newly_added_tool():
    """Seeded with the grown arrangement rather than the saved one, so the
    new tool shows in a bin of its own until the search finds it somewhere
    better. The saved floorplan alone would under-count the library.
    """
    parts, params = _parts(3)
    first = BuildPlan(parts, [Drawer(6, 6)], params)
    grown = dict(parts)
    grown[99] = BuildParts({99: _rectangle(50.0, 25.0)}, params)[99]
    seen: list[Progress] = []

    BuildPlan(grown, [Drawer(6, 6)], params, report=seen.append, interval=0.0, start=first.grouping)

    assert seen and seen[0].placed == len(grown)


def test_resuming_with_a_part_no_drawer_can_hold_says_which():
    parts, params = _parts(2, _quick(max_grid=6))
    first = BuildPlan(parts, [Drawer(6, 6)], params)
    assert first.grouping is not None

    grown = dict(parts)
    grown[99] = BuildParts({99: _rectangle(400.0, 30.0)}, params)[99]

    with pytest.raises(ValueError, match="cannot be added to this floorplan|does not fit"):
        BuildPlan(grown, [Drawer(6, 6)], params, start=first.grouping)


# ---------------------------------------------------------------- pinning


def test_a_pinned_bin_comes_back_exactly_as_it_went_in():
    """The whole claim. A pinned bin is one already printed and sitting in
    a drawer, so an arrangement that improved it by rearranging it would
    be describing a bin nobody owns.
    """
    parts, params = _parts(4)
    first = BuildPlan(parts, [Drawer(6, 6)], params)
    assert first.grouping is not None
    held = first.grouping.bins[0]

    plan = BuildPlan(parts, [Drawer(6, 6)], params, pinned=[held])

    assert plan.layouts[0] is held, "not merely equal - the same bin"
    assert plan.pinned == frozenset([0])


def test_pinned_bins_lead_so_a_pin_keeps_its_number():
    """Re-planning renumbers bins. If a pin moved with them, the panel
    would tick a different bin every time and nobody could follow it.
    """
    parts, params = _parts(5)
    first = BuildPlan(parts, [Drawer(6, 6)], params)
    assert first.grouping is not None and len(first.grouping.bins) > 1
    held = first.grouping.bins[-1]

    plan = BuildPlan(parts, [Drawer(6, 6)], params, pinned=[held])
    again = BuildPlan(parts, [Drawer(6, 6)], params, pinned=[held])

    assert plan.pinned == again.pinned == frozenset([0])
    assert plan.layouts[0] is again.layouts[0] is held


def test_the_parts_in_a_pinned_bin_are_placed_exactly_once():
    """They are held out of the grouping search entirely, so the danger is
    the opposite of losing them - grouping them a second time would place
    the same object in two bins.
    """
    parts, params = _parts(5)
    first = BuildPlan(parts, [Drawer(6, 6)], params)
    assert first.grouping is not None

    plan = BuildPlan(parts, [Drawer(6, 6)], params, pinned=[first.grouping.bins[0]])

    placed = [part_id for layout in plan.layouts.values() for part_id in layout.placements]
    assert sorted(placed) == sorted(parts)
    assert len(placed) == len(set(placed))


def test_pinning_every_bin_runs_no_grouping_search_at_all():
    """Nothing left to group. The drawer search still runs, since where a
    pinned bin sits on the shelf is not what was pinned.
    """
    parts, params = _parts(4)
    first = BuildPlan(parts, [Drawer(6, 6)], params)
    assert first.grouping is not None
    seen: list[Progress] = []

    plan = BuildPlan(parts, [Drawer(6, 6)], params, report=seen.append, interval=0.0, pinned=list(first.grouping.bins))

    assert list(plan.layouts.values()) == list(first.grouping.bins)
    assert plan.placed
    assert {progress.phase for progress in seen} == {ASSIGNING}


def test_a_pinned_bin_is_still_given_a_drawer_slot():
    """Pinning holds the bin together, not still. Sliding a bin along a
    shelf costs nothing, and refusing to would turn a pin into an
    obstacle.
    """
    parts, params = _parts(4)
    first = BuildPlan(parts, [Drawer(6, 6)], params)
    assert first.grouping is not None

    plan = BuildPlan(parts, [Drawer(6, 6)], params, pinned=[first.grouping.bins[0]])

    assert plan.assignment is not None
    assert 0 in plan.assignment.slots


def test_pinned_bins_are_in_every_report():
    """A pinned bin is one somebody owns. Dropping it from the picture
    while the search ran would make the drawer on screen look emptier than
    the drawer in the room.
    """
    parts, params = _parts(5)
    first = BuildPlan(parts, [Drawer(6, 6)], params)
    assert first.grouping is not None
    held = first.grouping.bins[0]
    seen: list[Progress] = []

    BuildPlan(parts, [Drawer(6, 6)], params, report=seen.append, interval=0.0, pinned=[held])

    assert seen
    for progress in seen:
        assert progress.bins[0] is held
    assert seen[-1].placed == len(parts), "and the count reaches the whole library"


def test_pinning_narrows_what_the_search_has_to_think_about():
    """The other half of why this is worth having: adding one tool to a
    settled library should be a search over one part, not over thirty.
    Measured against the same search unpinned, since a fixed threshold
    would drift with the tuning and prove nothing either way.
    """
    parts, params = _parts(5)
    first = BuildPlan(parts, [Drawer(6, 6)], params)
    assert first.grouping is not None

    def _steps(pinned) -> int:
        seen: list[Progress] = []
        BuildPlan(parts, [Drawer(6, 6)], params, report=seen.append, interval=0.0, pinned=pinned)
        return len([progress for progress in seen if progress.phase != ASSIGNING])

    assert _steps(list(first.grouping.bins[:-1])) < _steps([]) / 2


def test_pinning_and_resuming_together_do_not_group_a_part_twice():
    """They describe the same bins from opposite directions, and a start
    still holding the pinned parts would have the search group them a
    second time.
    """
    parts, params = _parts(5)
    first = BuildPlan(parts, [Drawer(6, 6)], params)
    assert first.grouping is not None

    plan = BuildPlan(parts, [Drawer(6, 6)], params, start=first.grouping, pinned=[first.grouping.bins[0]])

    placed = [part_id for layout in plan.layouts.values() for part_id in layout.placements]
    assert sorted(placed) == sorted(parts)
    assert len(placed) == len(set(placed))


def test_a_pin_naming_a_part_nobody_loaded_is_refused():
    """It would place a part that does not exist, and look plausible right
    up until the bins were printed.
    """
    parts, params = _parts(3)
    first = BuildPlan(parts, [Drawer(6, 6)], params)
    assert first.grouping is not None
    fewer = {0: parts[0]}

    with pytest.raises(ValueError, match="not in this library"):
        BuildPlan(fewer, [Drawer(6, 6)], params, pinned=list(first.grouping.bins))


def test_two_pins_claiming_the_same_part_are_refused():
    parts, params = _parts(3)
    first = BuildPlan(parts, [Drawer(6, 6)], params)
    assert first.grouping is not None
    twice = [first.grouping.bins[0], first.grouping.bins[0]]

    with pytest.raises(ValueError, match="already holds"):
        BuildPlan(parts, [Drawer(6, 6)], params, pinned=twice)


def test_stopping_before_a_complete_grouping_hands_back_no_pinned_bins_either():
    """The pinned bins alone are not an arrangement of the library, and a
    plan that looked like one would be saved and printed as one.
    """
    parts, params = _parts(5)
    first = BuildPlan(parts, [Drawer(6, 6)], params)
    assert first.grouping is not None
    seen: list[Progress] = []

    plan = BuildPlan(
        parts,
        [Drawer(6, 6)],
        params,
        report=seen.append,
        cancelled=lambda: bool(seen),
        interval=0.0,
        pinned=[first.grouping.bins[0]],
    )

    assert plan.cancelled
    assert plan.layouts == {}
    assert plan.pinned == frozenset()


def test_the_report_names_the_pinned_bins():
    parts, params = _parts(4)
    first = BuildPlan(parts, [Drawer(6, 6)], params)
    assert first.grouping is not None

    plan = BuildPlan(parts, [Drawer(6, 6)], params, pinned=[first.grouping.bins[0]])

    assert "(pinned)" in plan.Report()


# ------------------------------------------------------------------ progress


def test_progress_arrives_from_every_phase():
    parts, params = _parts(3)
    seen: list[Progress] = []

    BuildPlan(parts, [Drawer(6, 6)], params, report=seen.append, interval=0.0)

    phases = {progress.phase for progress in seen}
    assert phases == {FILLING, GROUPING, ASSIGNING}


def test_the_first_bins_are_reported_before_any_complete_grouping():
    """The earliest picture there is. On a real library the first complete
    grouping is a minute away, and a window with nothing to draw for that
    minute is indistinguishable from a hung one.
    """
    parts, params = _parts(4)
    seen: list[Progress] = []

    BuildPlan(parts, [Drawer(6, 6)], params, report=seen.append, interval=0.0)

    opening = [progress for progress in seen if progress.phase == FILLING and progress.bins]
    assert opening, "first fit's bins should be drawable as it opens them"
    assert seen.index(opening[0]) < next(i for i, p in enumerate(seen) if p.phase == GROUPING)


def test_the_opening_pass_is_reported_as_a_fraction_not_an_arrangement():
    """These bins hold real parts, but not all of them, and a cell count
    over a fragment flatters it badly - 1 bin / 2 cells for a four-part
    library is not a worse answer, it is not an answer.
    """
    parts, params = _parts(4)
    seen: list[Progress] = []

    BuildPlan(parts, [Drawer(6, 6)], params, report=seen.append, interval=0.0)

    partial = [progress for progress in seen if progress.phase == FILLING and progress.bins]
    assert partial
    assert any(progress.placed < len(parts) for progress in partial), "it should be caught mid-fill"
    for progress in partial:
        assert "best so far" not in str(progress)
        assert f"/{len(parts)} objects" in str(progress)


def test_the_best_so_far_always_holds_every_part():
    """Whatever is labelled GROUPING is an answer, and an answer accounts
    for the whole library. That is the line between the two phases.
    """
    parts, params = _parts(4)
    seen: list[Progress] = []

    BuildPlan(parts, [Drawer(6, 6)], params, report=seen.append, interval=0.0)

    reported = [progress for progress in seen if progress.phase == GROUPING and progress.bins]
    assert reported, "the search should reach a complete grouping"
    for progress in reported:
        assert progress.placed == len(parts)


def test_the_best_so_far_never_gets_worse():
    parts, params = _parts(4)
    seen: list[Progress] = []

    BuildPlan(parts, [Drawer(6, 6)], params, report=seen.append, interval=0.0)

    cells = [progress.cells for progress in seen if progress.phase == GROUPING and progress.bins]
    assert cells == sorted(cells, reverse=True)


def test_progress_is_throttled():
    """The searches report thousands of times a second and no front end can
    draw at that rate, so the throttle is at the source rather than left
    for every caller to reinvent.
    """
    parts, params = _parts(4)
    every, throttled = [], []

    BuildPlan(parts, [Drawer(6, 6)], params, report=every.append, interval=0.0)
    BuildPlan(parts, [Drawer(6, 6)], params, report=throttled.append, interval=60.0)

    assert len(throttled) < len(every)


def test_a_progress_line_distinguishes_having_no_answer_yet():
    """Before first-fit completes there is genuinely nothing to report,
    and saying so beats reporting the fragment it is holding.
    """
    parts, params = _parts(2)
    plan = BuildPlan(parts, [Drawer(6, 6)], params)

    empty = str(Progress(GROUPING, 12))
    found = str(Progress(GROUPING, 13, tuple(plan.layouts.values())))

    assert "building the first" in empty
    assert "best so far" in found
    assert f"{plan.cells} cells" in found


def test_a_progress_line_counts_placed_bins_while_assigning():
    parts, params = _parts(2)
    plan = BuildPlan(parts, [Drawer(6, 6)], params)
    assert plan.assignment is not None

    line = str(Progress(ASSIGNING, 9, tuple(plan.layouts.values()), plan.assignment))

    assert f"{len(plan.assignment.slots)}/{len(plan.layouts)} bins placed" in line


# --------------------------------------------------------------- cancelling


def test_cancelling_keeps_the_best_answer_so_far():
    """Someone who has watched for two minutes and seen the answer stop
    improving should be able to keep it. Stopping costs the time the
    search had left, not the answer it had found.
    """
    parts, params = _parts(4)
    seen: list[Progress] = []

    # Stop on the first event after a complete grouping has been reported,
    # rather than after a fixed count - the search is stochastic and a
    # count that lands past the end of it would test nothing.
    plan = BuildPlan(
        parts,
        [Drawer(6, 6)],
        params,
        report=seen.append,
        cancelled=lambda: any(progress.phase == GROUPING and progress.bins for progress in seen),
        interval=0.0,
    )

    assert plan.cancelled
    assert plan.layouts, "the best grouping found should survive the stop"
    assert sum(len(layout.placements) for layout in plan.layouts.values()) == len(parts)
    assert plan.assignment is None, "a grouping still being improved says nothing about drawers"
    assert "cancelled" in plan.Report()


def test_stopping_during_the_opening_pass_hands_back_no_floorplan():
    """The fragment on screen is a picture, not an answer. Handing it back
    would produce a session holding half a library that reads as a whole
    one - and the bins in it would be printed.
    """
    parts, params = _parts(4)
    seen: list[Progress] = []

    plan = BuildPlan(
        parts,
        [Drawer(6, 6)],
        params,
        report=seen.append,
        cancelled=lambda: any(progress.phase == FILLING and progress.bins for progress in seen),
        interval=0.0,
    )

    assert plan.cancelled
    assert plan.layouts == {}, "a fragment must never survive as a plan"


def test_cancelling_immediately_still_returns_a_plan():
    """No answer yet is a state to report, not an exception - the person
    pressing Stop is not handling an error.
    """
    parts, params = _parts(3)

    plan = BuildPlan(parts, [Drawer(6, 6)], params, cancelled=lambda: True)

    assert plan.cancelled
    assert plan.layouts == {}
    assert not plan.placed


def test_an_uncancelled_search_is_not_flagged():
    parts, params = _parts(3)

    plan = BuildPlan(parts, [Drawer(6, 6)], params, cancelled=lambda: False)

    assert not plan.cancelled
