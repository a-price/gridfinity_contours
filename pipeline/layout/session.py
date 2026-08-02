"""Saving a floorplan and picking it up again.

A library grows one object at a time. You photograph a new tool, and the
question is not "where does everything go" - that was answered months ago
and half the answer is already printed and sitting in a drawer - but
"where does *this* go, and what do I have to reprint". A session file is
what makes that the question the tools can be asked.

**The file holds the contours themselves, not paths to them.** Part ids
are assigned by `ReadContours` in the order files are encountered, so a
saved grouping that referred to source files would silently mean something
different the moment a file was renamed, moved, or loaded alongside one
more. A grouping is a statement about ids, so the file that records one
has to record what those ids *are*.

**Parameters are saved too, and checked rather than trusted.** The
placements in a session satisfied the clearances they were solved
against; reloading under a wider pocket offset would produce a floorplan
that looked settled and was not. `LoadSession` restores what the layouts
were built with, and `Verify` says whether they still hold - the same
propose-then-check discipline `spacing` and `Distribute` follow one level
down.

What this deliberately does *not* store is the assignment's derived
reporting, or `admissible_grids`. Both are recomputed: the first is a
function of the bins and drawers, and the second is derived from the
drawers by `plan.BuildPlan`, where saving a stale copy would override the
rule that an explicit set wins.
"""

import json
from dataclasses import dataclass, fields, replace
from typing import Any

import numpy as np

from pipeline.layout.drawer import PLACED, AssignmentResult, Drawer, Slot
from pipeline.layout.grouping import Grouping
from pipeline.layout.parameters import LayoutParameters
from pipeline.layout.part import Part
from pipeline.layout.placement import Layout, Placement
from pipeline.layout.plan import StoragePlan
from pipeline.layout.verify import CheckLayout

SESSION_FORMAT_VERSION = 1

# Millimetres for anything continuous, whole cells for anything on the
# lattice. Tagged for the same reason the drawer file is: the risk with
# these files is reading one kind of number as another kind of number.
SESSION_UNITS = "mm"

# Parameters restored with a session. Deliberately not every field of
# `LayoutParameters`: these are the ones the saved *geometry* depends on,
# so reloading under a different value would invalidate the placements.
# The search budget is not among them - how hard the last run looked has
# no bearing on whether its answer is still valid, and inheriting somebody
# else's restart count would be surprising rather than helpful.
GEOMETRY_FIELDS = ("pocket_offset", "pair_clearance", "wall_clearance", "resolution", "inset", "max_grid")


@dataclass(frozen=True)
class Session:
    """A floorplan and everything needed to resume it.

    Contours rather than parts, because parts are a rasterization of
    contours under a particular resolution and rebuilding them is cheap
    and unambiguous - whereas storing a raster would fix a decision the
    reloading run is entitled to change.
    """

    contours: dict[int, np.ndarray]
    drawers: list[Drawer]
    grouping: Grouping
    parameters: LayoutParameters
    assignment: AssignmentResult | None = None
    # Indices into `grouping.bins` that were held fixed rather than
    # searched for. Saved because a pin is a statement about the physical
    # world - "this one is printed, leave it alone" - and re-ticking a
    # dozen boxes every time the file is opened is how a pin gets lost.
    pinned: frozenset[int] = frozenset()

    def Grown(self, contours: dict[int, np.ndarray]) -> tuple["Session", list[int]]:
        """This session with more contours added, and the ids they got.

        New ids continue past the highest already in use rather than
        filling gaps, so an id never changes meaning between one session
        and the next - which is the whole reason the grouping can be
        resumed at all.
        """
        combined = dict(self.contours)
        added = []
        for points in contours.values():
            part_id = max(combined, default=-1) + 1
            combined[part_id] = points
            added.append(part_id)
        return replace(self, contours=combined), added


def SaveSession(path: str, plan: StoragePlan, contours: dict[int, np.ndarray], params: LayoutParameters) -> None:
    """Write a floorplan and its inputs to `path`.

    Takes the contours alongside the plan because a `StoragePlan` holds
    rasterized parts, and a part cannot be turned back into the contour it
    came from - `BuildPart` PCA-aligns and resamples on the way in.
    """
    if not plan.layouts:
        raise ValueError("no floorplan to save")

    placed = {part_id for layout in plan.layouts.values() for part_id in layout.placements}
    missing = sorted(placed - set(contours))
    if missing:
        raise ValueError(f"the floorplan places parts {missing}, whose contours were not given")

    payload: dict[str, Any] = {
        "version": SESSION_FORMAT_VERSION,
        "units": SESSION_UNITS,
        "parameters": {name: getattr(params, name) for name in GEOMETRY_FIELDS},
        "contours": {
            str(part_id): np.asarray(points, dtype=np.float64).reshape(-1, 2).tolist()
            for part_id, points in sorted(contours.items())
        },
        "drawers": [{"width": drawer.width, "height": drawer.height} for drawer in plan.drawers],
        "bins": [_BinPayload(plan.layouts[bin_id]) for bin_id in sorted(plan.layouts)],
        # Positions in the `bins` list above, which is written in bin id
        # order - so these stay meaningful without a second identity
        # scheme for bins.
        "pinned": sorted(plan.pinned),
    }
    if plan.assignment is not None:
        payload["assignment"] = {
            "outcome": plan.assignment.outcome,
            "slots": [_SlotPayload(slot) for _, slot in sorted(plan.assignment.slots.items())],
        }

    with open(path, "w") as handle:
        json.dump(payload, handle, indent=1)
        handle.write("\n")


def LoadSession(path: str) -> Session:
    """Read a session back.

    The layouts are reconstructed exactly as saved rather than re-solved.
    That is the point: the bins in the drawer were printed from these
    placements, and a resumed session that quietly re-derived them would
    describe a shelf of bins nobody owns.
    """
    with open(path) as handle:
        payload = json.load(handle)

    if not isinstance(payload, dict):
        raise ValueError(f"{path} is not a session file")

    version = payload.get("version")
    if version != SESSION_FORMAT_VERSION:
        raise ValueError(f"{path} is format version {version}, this build reads {SESSION_FORMAT_VERSION}")

    units = payload.get("units")
    if units != SESSION_UNITS:
        raise ValueError(f"{path} is in '{units}', expected '{SESSION_UNITS}'")

    contours = _Contours(path, payload.get("contours"))
    drawers = _Drawers(path, payload.get("drawers"))
    bins = _Bins(path, payload.get("bins"), contours)

    return Session(
        contours=contours,
        drawers=drawers,
        grouping=Grouping(bins),
        parameters=_Parameters(payload.get("parameters")),
        assignment=_Assignment(payload.get("assignment"), len(bins)),
        pinned=_Pinned(payload.get("pinned"), len(bins)),
    )


def Verify(session: Session, parts: dict[int, Part]) -> list[str]:
    """Every way the saved floorplan fails its own clearances, as
    human-readable strings. Empty means it still holds.

    Checked rather than trusted, because a session outlives the settings
    it was made under: reloading a floorplan solved at a 1mm pocket offset
    under a 3mm one leaves placements that look settled and are not. This
    runs the same independent polygon checks `verify.py` gives the CLI,
    which share no code with the solver that produced the arrangement.
    """
    problems = []
    for index, layout in enumerate(session.grouping.bins):
        for problem in CheckLayout(layout, parts, session.parameters.c_pair, session.parameters.c_wall):
            problems.append(f"bin {index}: {problem}")
    return problems


def Changes(before: Grouping, after: Grouping) -> tuple[list[int], list[int]]:
    """Which bins of `after` are unchanged from `before`, and which are
    not, as two lists of indices into `after`.

    The question a resumed session is actually asked: *what do I have to
    reprint*. A bin whose contents and arrangement both survived is a bin
    already sitting in the drawer; anything else has to come off the
    printer again.

    Compared by contents *and* placements, not contents alone. Two bins
    holding the same parts in different positions are different bins as
    far as a printed pocket is concerned, and reporting one as unchanged
    would be the one error this function must not make.
    """
    previous = {_Signature(layout) for layout in before.bins}

    kept, changed = [], []
    for index, layout in enumerate(after.bins):
        (kept if _Signature(layout) in previous else changed).append(index)
    return kept, changed


def _Signature(layout: Layout) -> tuple:
    """A bin's identity for reprint purposes: its size, and every part in
    it at the position and angle it sits.
    """
    return (
        layout.grid,
        tuple(
            (
                part_id,
                round(float(placement.position[0]), 6),
                round(float(placement.position[1]), 6),
                placement.orientation,
            )
            for part_id, placement in sorted(layout.placements.items())
        ),
    )


def _BinPayload(layout: Layout) -> dict:
    return {
        "grid": list(layout.grid),
        "inset": layout.inset,
        "placements": [
            {
                "part": part_id,
                "position": [float(placement.position[0]), float(placement.position[1])],
                "orientation": placement.orientation,
            }
            for part_id, placement in sorted(layout.placements.items())
        ],
    }


def _SlotPayload(slot: Slot) -> dict:
    return {"bin": slot.bin_id, "drawer": slot.drawer, "cell": list(slot.cell), "turned": slot.turned}


def _Contours(path: str, raw: Any) -> dict[int, np.ndarray]:
    if not isinstance(raw, dict) or not raw:
        raise ValueError(f"{path} contains no contours")

    contours = {}
    for key, points in raw.items():
        try:
            part_id = int(key)
        except ValueError:
            raise ValueError(f"contour key '{key}' is not an integer id") from None

        array = np.asarray(points, dtype=np.float64)
        if array.ndim != 2 or array.shape[1] != 2 or len(array) < 3:
            raise ValueError(f"contour {part_id} has shape {array.shape}, expected (N, 2) with at least 3 points")
        if not np.isfinite(array).all():
            raise ValueError(f"contour {part_id} contains a non-finite coordinate")
        contours[part_id] = array
    return contours


def _Drawers(path: str, raw: Any) -> list[Drawer]:
    if not isinstance(raw, list) or not raw:
        raise ValueError(f"{path} contains no drawers")
    return [Drawer(int(entry["width"]), int(entry["height"])) for entry in raw]


def _Bins(path: str, raw: Any, contours: dict[int, np.ndarray]) -> list[Layout]:
    if not isinstance(raw, list) or not raw:
        raise ValueError(f"{path} contains no bins")

    bins = []
    for index, entry in enumerate(raw):
        placements = {}
        for placement in entry["placements"]:
            part_id = int(placement["part"])
            if part_id not in contours:
                raise ValueError(f"bin {index} places part {part_id}, which the session has no contour for")
            placements[part_id] = Placement(
                part_id,
                np.asarray(placement["position"], dtype=np.float64),
                int(placement["orientation"]),
            )
        if not placements:
            raise ValueError(f"bin {index} holds no parts")
        grid = tuple(int(value) for value in entry["grid"])
        bins.append(Layout(grid=(grid[0], grid[1]), placements=placements, inset=float(entry["inset"])))
    return bins


def _Parameters(raw: Any) -> LayoutParameters:
    """The saved geometry settings, over this build's defaults.

    Unknown names are ignored rather than refused, so a session written by
    a build that tracked one more setting still loads here - it just loads
    without it, which is what a default is for.
    """
    known = {field.name for field in fields(LayoutParameters)}
    given = raw if isinstance(raw, dict) else {}
    return replace(LayoutParameters(), **{name: value for name, value in given.items() if name in known})


def _Pinned(raw: Any, bins: int) -> frozenset[int]:
    """The pinned bin indices, refused rather than clipped if they do not
    name a bin this session has.

    Silently dropping one would quietly unpin a bin somebody has already
    printed, which is the failure this whole feature exists to prevent.
    """
    if not isinstance(raw, list):
        return frozenset()

    pinned = set()
    for value in raw:
        index = int(value)
        if not 0 <= index < bins:
            raise ValueError(f"bin {index} is pinned, but this session has {bins} bins")
        pinned.add(index)
    return frozenset(pinned)


def _Assignment(raw: Any, bins: int) -> AssignmentResult | None:
    if not isinstance(raw, dict):
        return None

    slots = {}
    for entry in raw.get("slots", []):
        bin_id = int(entry["bin"])
        if not 0 <= bin_id < bins:
            raise ValueError(f"the assignment places bin {bin_id}, which this session does not have")
        cell = tuple(int(value) for value in entry["cell"])
        slots[bin_id] = Slot(bin_id, int(entry["drawer"]), (cell[0], cell[1]), bool(entry.get("turned", False)))

    return AssignmentResult(str(raw.get("outcome", PLACED)), slots)
