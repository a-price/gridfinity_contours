"""Tests for saving a floorplan and resuming it.

The flow these are about: a library was planned months ago and half of it
is printed and sitting in a drawer. A new tool arrives. What has to be
true for that to work is that part ids still mean what they meant, that
the arrangement comes back exactly as saved rather than re-solved, and
that resuming tells you which bins you have to print again.
"""

import json
from dataclasses import replace

import jsonschema
import numpy as np
import pytest

from layout.drawer import ASSIGNMENT_SCHEMA, INFEASIBLE, SLOT_SCHEMA, AssignmentResult, Drawer
from layout.loading import BuildParts
from layout.placement import LAYOUT_SCHEMA, PLACEMENT_SCHEMA, Layout, Placement
from layout.plan import BuildPlan, StoragePlan
from layout.session import (
    SESSION_SCHEMA,
    Changes,
    LoadSession,
    SaveSession,
    Verify,
    _BinPayload,
    _SlotPayload,
)
from layout.parameters import FREE_ROTATION, QUARTER_TURNS
from conftest import QuickParameters as _quick, Rectangle as _rectangle


def _library(count: int = 4) -> dict[int, np.ndarray]:
    return {index: _rectangle(60.0 - 8.0 * index, 30.0) for index in range(count)}


def _planned(tmp_path, contours=None, params=None, drawers=None):
    """A saved session, and everything that went into it."""
    contours = _library() if contours is None else contours
    params = params or _quick(max_grid=3)
    drawers = drawers or [Drawer(6, 6)]

    plan = BuildPlan(BuildParts(contours, params), drawers, params)
    path = str(tmp_path / "session.json")
    SaveSession(path, plan, contours, params)
    return path, plan, contours, params


# ------------------------------------------------------------- round trip


def test_a_session_round_trips(tmp_path):
    path, plan, contours, params = _planned(tmp_path)

    session = LoadSession(path)

    assert sorted(session.contours) == sorted(contours)
    assert session.drawers == list(plan.drawers)
    assert len(session.grouping.bins) == len(plan.layouts)
    assert session.grouping.cells == plan.cells


def test_the_arrangement_comes_back_exactly_as_saved(tmp_path):
    """Not re-solved. The bins in the drawer were printed from these
    placements, and a session that quietly re-derived them would describe
    a shelf of bins nobody owns.
    """
    path, plan, _, _ = _planned(tmp_path)

    session = LoadSession(path)

    for saved, loaded in zip(plan.layouts.values(), session.grouping.bins):
        assert loaded.grid == saved.grid
        assert sorted(loaded.placements) == sorted(saved.placements)
        for part_id, placement in saved.placements.items():
            assert loaded.placements[part_id].position == pytest.approx(placement.position)
            assert loaded.placements[part_id].orientation == placement.orientation


def test_the_contours_are_stored_not_referenced(tmp_path):
    """Part ids come from the order files are read in, so a session that
    pointed at source files would mean something different the moment one
    was renamed - and a grouping is a statement about ids.
    """
    path, _, contours, _ = _planned(tmp_path)

    payload = json.loads(open(path).read())

    assert sorted(int(key) for key in payload["contours"]) == sorted(contours)
    assert len(payload["contours"]["0"]) == len(contours[0])


def test_the_geometry_parameters_travel_with_it(tmp_path):
    """The placements satisfied the clearances they were solved against,
    so the settings that produced them are part of the answer.
    """
    params = _quick(max_grid=3, pocket_offset=2.5, resolution=0.5)
    path, _, _, _ = _planned(tmp_path, params=params)

    session = LoadSession(path)

    assert session.parameters.pocket_offset == pytest.approx(2.5)
    assert session.parameters.resolution == pytest.approx(0.5)
    assert session.parameters.c_pair == pytest.approx(params.c_pair)


def test_the_search_budget_does_not_travel_with_it():
    """How hard the last run looked says nothing about whether its answer
    is still valid, and inheriting somebody else's restart count would be
    surprising rather than helpful.
    """
    from layout.session import GEOMETRY_FIELDS

    assert "restarts" not in GEOMETRY_FIELDS
    assert "seed" not in GEOMETRY_FIELDS
    assert "admissible_grids" not in GEOMETRY_FIELDS, "derived from the drawers by BuildPlan"


def test_the_assignment_travels_with_it(tmp_path):
    path, plan, _, _ = _planned(tmp_path)
    assert plan.assignment is not None

    session = LoadSession(path)

    assert session.assignment is not None
    assert set(session.assignment.slots) == set(plan.assignment.slots)
    for bin_id, slot in plan.assignment.slots.items():
        assert session.assignment.slots[bin_id].cell == slot.cell
        assert session.assignment.slots[bin_id].drawer == slot.drawer


def test_an_infeasible_assignments_unplaced_bins_and_detail_travel_with_it(tmp_path):
    """`AssignmentResult` carries more than `outcome` and `slots` - which
    bins an INFEASIBLE or EXHAUSTED search could not place, and why, is
    the part that outcome most needs to explain. Losing them on the way
    through a session file used to leave a reloaded floorplan's `Report()`
    saying "could not place bins" with nothing after it.
    """
    path, plan, contours, params = _planned(tmp_path)
    bin_id = next(iter(plan.layouts))
    broken = AssignmentResult(INFEASIBLE, slots={}, unplaced=[bin_id], detail="no drawer is large enough")
    SaveSession(path, replace(plan, assignment=broken), contours, params)

    session = LoadSession(path)

    assert session.assignment == broken


# ------------------------------------------------------------- wire schema


def test_a_placements_payload_matches_its_own_schema(tmp_path):
    """The real dict `_BinPayload` writes for one placement, validated
    against the schema kept beside `Placement` - not a hand-copied
    example, so a field added to one and not the other fails here rather
    than shipping quietly.
    """
    _, plan, _, _ = _planned(tmp_path)
    layout = next(iter(plan.layouts.values()))

    jsonschema.validate(_BinPayload(layout)["placements"][0], PLACEMENT_SCHEMA)


def test_a_bins_payload_matches_its_own_schema(tmp_path):
    _, plan, _, _ = _planned(tmp_path)
    layout = next(iter(plan.layouts.values()))

    jsonschema.validate(_BinPayload(layout), LAYOUT_SCHEMA)


def test_a_slots_payload_matches_its_own_schema(tmp_path):
    _, plan, _, _ = _planned(tmp_path)
    assert plan.assignment is not None
    slot = next(iter(plan.assignment.slots.values()))

    jsonschema.validate(_SlotPayload(slot), SLOT_SCHEMA)


def test_an_assignments_real_payload_matches_its_own_schema(tmp_path):
    """Same fixture the `unplaced`/`detail` bug was found with - the
    payload `SaveSession` actually writes to disk for an INFEASIBLE
    assignment, read back and validated end to end.
    """
    path, plan, contours, params = _planned(tmp_path)
    broken = AssignmentResult(
        INFEASIBLE, slots={}, unplaced=[next(iter(plan.layouts))], detail="no drawer is large enough"
    )
    SaveSession(path, replace(plan, assignment=broken), contours, params)

    payload = json.loads(open(path).read())

    jsonschema.validate(payload["assignment"], ASSIGNMENT_SCHEMA)


def test_a_saved_session_matches_its_own_schema(tmp_path):
    path, _, _, _ = _planned(tmp_path)

    payload = json.loads(open(path).read())

    jsonschema.validate(payload, SESSION_SCHEMA)


def test_a_saved_session_with_an_infeasible_assignment_matches_its_own_schema(tmp_path):
    """The exact fixture the `unplaced`/`detail` bug was found and fixed
    with, validated end to end this time rather than piece by piece.
    """
    path, plan, contours, params = _planned(tmp_path)
    broken = AssignmentResult(
        INFEASIBLE, slots={}, unplaced=[next(iter(plan.layouts))], detail="no drawer is large enough"
    )
    SaveSession(path, replace(plan, assignment=broken), contours, params)

    payload = json.loads(open(path).read())

    jsonschema.validate(payload, SESSION_SCHEMA)


# ----------------------------------------------------------------- pinning


def test_pins_survive_a_round_trip(tmp_path):
    """A pin is a statement about the physical world - this one is printed,
    leave it alone - so re-ticking a dozen boxes every time the file is
    opened is how a pin gets lost.
    """
    contours = _library()
    params = _quick(max_grid=3)
    plan = BuildPlan(BuildParts(contours, params), [Drawer(6, 6)], params)
    held = BuildPlan(BuildParts(contours, params), [Drawer(6, 6)], params, pinned=[plan.layouts[0]])
    path = str(tmp_path / "pinned.json")
    SaveSession(path, held, contours, params)

    session = LoadSession(path)

    assert session.pinned == held.pinned
    assert sorted(session.grouping.bins[0].placements) == sorted(held.layouts[0].placements)


def test_a_session_with_nothing_pinned_says_so(tmp_path):
    path, _, _, _ = _planned(tmp_path)

    assert LoadSession(path).pinned == frozenset()


def test_a_pin_naming_a_bin_the_session_does_not_have_is_refused(tmp_path):
    """Clipping it would quietly unpin a bin somebody has already printed,
    which is the failure the whole feature exists to prevent.
    """
    path, _, _, _ = _planned(tmp_path)
    payload = json.loads(open(path).read())
    payload["pinned"] = [99]
    (tmp_path / "broken.json").write_text(json.dumps(payload))

    with pytest.raises(ValueError, match="is pinned"):
        LoadSession(str(tmp_path / "broken.json"))


def test_an_older_session_without_pins_still_loads(tmp_path):
    """The field was added after the format; a file written before it is
    a file with nothing pinned, not a broken one.
    """
    path, _, _, _ = _planned(tmp_path)
    payload = json.loads(open(path).read())
    del payload["pinned"]
    (tmp_path / "older.json").write_text(json.dumps(payload))

    assert LoadSession(str(tmp_path / "older.json")).pinned == frozenset()


# ---------------------------------------------------------------- refusals


def test_saving_without_a_floorplan_is_refused(tmp_path):
    empty = StoragePlan(drawers=(Drawer(2, 2),), parts={}, layouts={})

    with pytest.raises(ValueError, match="nothing to save|no floorplan"):
        SaveSession(str(tmp_path / "s.json"), empty, {}, _quick())


def test_saving_without_the_contours_it_places_is_refused(tmp_path):
    """A session missing a contour would reload as a grouping referring to
    a part that does not exist.
    """
    _, plan, contours, params = _planned(tmp_path)

    with pytest.raises(ValueError, match="whose contours were not given"):
        SaveSession(str(tmp_path / "partial.json"), plan, {0: contours[0]}, params)


def test_a_session_from_a_future_format_is_refused(tmp_path):
    path = tmp_path / "s.json"
    path.write_text(json.dumps({"version": 99, "units": "mm"}))

    with pytest.raises(ValueError, match="format version"):
        LoadSession(str(path))


def test_a_bin_placing_an_unknown_part_is_refused(tmp_path):
    path, _, _, _ = _planned(tmp_path)
    payload = json.loads(open(path).read())
    payload["bins"][0]["placements"][0]["part"] = 99
    (tmp_path / "broken.json").write_text(json.dumps(payload))

    with pytest.raises(ValueError, match="no contour for"):
        LoadSession(str(tmp_path / "broken.json"))


# ------------------------------------------------------------------ verify


def test_a_reloaded_floorplan_still_satisfies_its_own_clearances(tmp_path):
    path, _, _, _ = _planned(tmp_path)
    session = LoadSession(path)

    problems = Verify(session, BuildParts(session.contours, session.parameters))

    assert problems == []


def test_a_floorplan_reloaded_under_wider_clearances_says_so(tmp_path):
    """A session outlives the settings it was made under. One solved at a
    1mm pocket offset and reopened at 6mm looks settled and is not, and
    that has to be reported rather than assumed away.
    """
    path, _, _, _ = _planned(tmp_path)
    session = LoadSession(path)
    widened = replace(session, parameters=replace(session.parameters, pocket_offset=6.0))

    problems = Verify(widened, BuildParts(widened.contours, widened.parameters))

    assert problems, "a 6mm pocket offset cannot hold an arrangement solved for 1mm"
    assert "bin " in problems[0]


# -------------------------------------------------------------- growing it


def test_a_new_contour_gets_an_id_nothing_else_has_used(tmp_path):
    """Ids must never change meaning between sessions, or the saved
    grouping stops describing the parts it was written about.
    """
    path, _, _, _ = _planned(tmp_path)
    session = LoadSession(path)

    grown, added = session.Grown({0: _rectangle(50.0, 25.0)})

    assert added == [max(session.contours) + 1]
    assert set(grown.contours) == set(session.contours) | set(added)
    for part_id in session.contours:
        assert grown.contours[part_id] is session.contours[part_id]


def test_growing_leaves_the_saved_grouping_alone(tmp_path):
    path, _, _, _ = _planned(tmp_path)
    session = LoadSession(path)

    grown, _ = session.Grown({0: _rectangle(50.0, 25.0)})

    assert grown.grouping is session.grouping


# ------------------------------------------------------------ what changed


def test_an_untouched_bin_is_reported_as_unchanged(tmp_path):
    """The question a resumed session is actually asked: what has to come
    off the printer again.
    """
    path, _, _, params = _planned(tmp_path)
    session = LoadSession(path)
    grown, _ = session.Grown({0: _rectangle(50.0, 25.0)})

    resumed = BuildPlan(BuildParts(grown.contours, params), grown.drawers, params, start=session.grouping)
    assert resumed.grouping is not None
    kept, changed = Changes(session.grouping, resumed.grouping)

    assert kept, "adding one object should not invalidate every bin"
    assert len(kept) + len(changed) == len(resumed.grouping.bins)


def test_a_bin_whose_parts_moved_counts_as_changed():
    """Contents alone is not enough. Two bins holding the same parts in
    different positions are different bins as far as a printed pocket is
    concerned, and calling one unchanged is the single error this must not
    make.
    """
    before = Layout(grid=(2, 1), placements={0: Placement(0, np.array([5.0, 5.0]))})
    moved = Layout(grid=(2, 1), placements={0: Placement(0, np.array([9.0, 5.0]))})

    kept, changed = Changes(_grouping(before), _grouping(moved))

    assert kept == []
    assert changed == [0]


def test_a_bin_that_only_turned_counts_as_changed():
    before = Layout(grid=(2, 1), placements={0: Placement(0, np.array([5.0, 5.0]), orientation=0)})
    turned = Layout(grid=(2, 1), placements={0: Placement(0, np.array([5.0, 5.0]), orientation=1)})

    assert Changes(_grouping(before), _grouping(turned)) == ([], [0])


def test_an_identical_grouping_is_entirely_unchanged():
    layout = Layout(grid=(2, 1), placements={0: Placement(0, np.array([5.0, 5.0]))})

    assert Changes(_grouping(layout), _grouping(layout)) == ([0], [])


def _grouping(*layouts):
    from layout.grouping import Grouping

    return Grouping(list(layouts))


# ---------------------------------------------------------- free rotation


def test_a_free_angle_survives_the_round_trip(tmp_path):
    """A layout packed with free rotation has to reload as the same layout.
    An angle silently dropped on load would place every pocket square-on -
    a drawing that still looks reasonable, describing bins that do not fit
    the objects they were cut for.
    """
    contours, params = _library(), _quick(max_grid=3, rotation=FREE_ROTATION)
    parts = BuildParts(contours, params)
    plan = BuildPlan(parts, [Drawer(6, 6)], params)

    bin_id = min(plan.layouts)
    layout = plan.layouts[bin_id]
    part_id, placement = next(iter(layout.placements.items()))
    turned = replace(layout, placements={**layout.placements, part_id: replace(placement, angle=0.35)})
    plan = replace(plan, layouts={**plan.layouts, bin_id: turned})

    path = str(tmp_path / "turned.json")
    SaveSession(path, plan, contours, params)

    reloaded = LoadSession(path).grouping.bins
    assert any(p.angle == pytest.approx(0.35) for layout in reloaded for p in layout.placements.values())


def test_a_session_written_before_free_rotation_still_loads(tmp_path):
    """The angle defaults rather than being required. Every placement in an
    older session is upright by construction, which is exactly what the
    default says - so refusing to read one would be refusing a file whose
    meaning is unambiguous.
    """
    path, _, _, _ = _planned(tmp_path)

    payload = json.loads(open(path).read())
    for entry in payload["bins"]:
        for placement in entry["placements"]:
            del placement["angle"]
    payload["parameters"].pop("rotation", None)
    with open(path, "w") as handle:
        json.dump(payload, handle)

    reloaded = LoadSession(path)

    assert all(p.angle == 0.0 for layout in reloaded.grouping.bins for p in layout.placements.values())
    assert reloaded.parameters.rotation == QUARTER_TURNS


def test_the_rotation_mode_travels_with_a_session(tmp_path):
    """It decides which arrangements are legal, so reloading under a
    different mode would hold placements the current settings say cannot
    exist - and a re-pack would disagree with the drawing beside it.
    """
    path, _, _, _ = _planned(tmp_path, params=_quick(max_grid=3, rotation=FREE_ROTATION))

    assert LoadSession(path).parameters.rotation == FREE_ROTATION
