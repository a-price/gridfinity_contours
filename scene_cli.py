"""Render a saved floorplan session's bins into a 3D scene, one file per
drawer.

    scene_cli.py floorplan.json --out scene

Reads a session the same way `floorplan_gui.py` does, places every bin at
its real position and orientation in its drawer (`layout.scene.
WriteScene`), then renders each drawer's `.scad` to a mesh with OpenSCAD
(`--format`, default `3mf`). A session saved before its drawer search
finished carries no assignment; one is computed here rather than refused -
the same fallback `panels.floorplan_panel.FloorplanPanel._Assigned` uses on
reload, since the drawer search is exact and fast and there is nothing to
lose by running it.

Headless, like `layout_cli.py`: turning a saved plan into a scene needs
neither a live session nor a new search.
"""

import argparse
import sys
from typing import Sequence

from export.scad_writer import Available, RenderScad
from layout.drawer import Assign, AssignmentResult
from layout.loading import BuildParts
from layout.plan import StoragePlan
from layout.scene import WriteScene
from layout.session import LoadSession, Verify
from layout.solid import DEFAULT_HEIGHT_UNITS


def BuildParser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("session", metavar="FILE", help="a saved floorplan session (.json)")
    parser.add_argument(
        "--out", default="scene", help="output basename; writes one .scad (+ mesh) per drawer (default: scene)"
    )
    parser.add_argument("--format", default="3mf", help="mesh format OpenSCAD writes, by extension (default: 3mf)")
    parser.add_argument(
        "--pocket-offset",
        type=float,
        default=None,
        metavar="MM",
        help="cut pockets at a different tolerance than the session was packed with (default: the session's own)",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=DEFAULT_HEIGHT_UNITS,
        metavar="UNITS",
        help=f"bin height in 7mm Gridfinity units (default: {DEFAULT_HEIGHT_UNITS})",
    )
    parser.add_argument("--no-render", action="store_true", help="write the .scad files only, skip the OpenSCAD render")
    return parser


def _Assigned(plan: StoragePlan) -> AssignmentResult:
    """`plan`'s drawer assignment, searched for now if it has none.

    Mirrors `panels.floorplan_panel.FloorplanPanel._Assigned`: a session
    saved while its grouping search was still running carries bins and no
    assignment, and the drawer search alone is cheap enough that redoing it
    here beats refusing the export.
    """
    if plan.assignment is not None:
        return plan.assignment
    return Assign(plan.footprints, plan.drawers)


def Main(argv: Sequence[str] | None = None) -> int:
    args = BuildParser().parse_args(argv)
    fmt = args.format.lstrip(".")

    session = LoadSession(args.session)
    parts = BuildParts(session.contours, session.parameters)
    print(f"loaded {len(session.grouping.bins)} bins, {len(session.drawers)} drawer(s) from {args.session}")

    problems = Verify(session, parts)
    if problems:
        print(f"warning: {len(problems)} clearance problem(s) in this session: {problems[0]}", file=sys.stderr)

    layouts = dict(enumerate(session.grouping.bins))
    plan = StoragePlan(
        drawers=tuple(session.drawers),
        parts=parts,
        layouts=layouts,
        assignment=session.assignment,
        footprints={index: layout.grid for index, layout in layouts.items()},
        pinned=session.pinned,
    )
    assignment = _Assigned(plan)
    print(assignment.Report(plan.footprints))

    offset = session.parameters.pocket_offset if args.pocket_offset is None else args.pocket_offset
    report = WriteScene(args.out, plan, assignment=assignment, pocket_offset=offset, height_units=args.height)
    for bin_id, problem in sorted(report.problems.items()):
        print(f"bin {bin_id} could not be cut: {problem}", file=sys.stderr)
    scad_paths = report.written
    if not scad_paths:
        print("no drawer holds a cuttable bin; nothing to render")
        return 1
    print("wrote " + ", ".join(scad_paths))

    if args.no_render:
        return 0
    if not Available():
        print("openscad is not on the path; wrote .scad only", file=sys.stderr)
        return 1

    written = []
    for scad_path in scad_paths:
        mesh_path = f"{scad_path.removesuffix('.scad')}.{fmt}"
        RenderScad(scad_path, mesh_path)
        written.append(mesh_path)
    print("rendered " + ", ".join(written))
    return 0


if __name__ == "__main__":
    sys.exit(Main())
