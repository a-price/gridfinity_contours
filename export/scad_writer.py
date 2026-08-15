"""OpenSCAD as this project's fifth output format: several placed modules
composed into one program, and a subprocess call to `openscad` that turns
that program into a mesh.

Symmetrical with `svg_writer`/`pdf_writer`: `ScadPart` is the `.scad`
analogue of `Shape` - already positioned in the frame it should be placed
in, with nothing left for this module to know about what a bin or a
drawer is. `layout/solid.py` and `layout/scene.py` are the callers that
know that; this only ever sees OpenSCAD source and millimetre coordinates,
the same discipline that keeps `export/` free of Qt, a solver, or any
other project concept (see `export/__init__.py`).

Rendering needs the `openscad` binary on the path. `Available()` checks
rather than assumes, so a missing binary is one clear sentence instead of
a `FileNotFoundError` from inside a subprocess call - the same discipline
`demos/solid_demo.py` already established as the project's one other place
that shells out to it.
"""

import shutil
import subprocess
from dataclasses import dataclass
from typing import Sequence

OPENSCAD = "openscad"


@dataclass(frozen=True)
class ScadPart:
    """One named OpenSCAD module, and where to place it in a scene.

    `module` is a complete `module <name>() { ... }` definition -
    `layout.solid.GenerateBinModule` writes exactly this. `name` is
    repeated alongside it rather than parsed back out, since a caller
    already had to choose a unique name to write that definition with in
    the first place.

    `x`, `y` and `degrees` are millimetres and a rotation about Z, in
    OpenSCAD's own y-up frame - already resolved by the caller, the same
    way a `Shape`'s points are already in page coordinates by the time
    `svg_writer` sees them.
    """

    name: str
    module: str
    x: float = 0.0
    y: float = 0.0
    degrees: float = 0.0


def WriteScadScene(path: str, library_path: str, parts: Sequence[ScadPart]) -> None:
    """Write several placed modules to one `.scad` program.

    One shared header rather than one per part - each `ScadPart.module` is
    written with no `include`/`use` of its own (`GenerateBinModule`
    deliberately omits one), so a file with a dozen bins in it does not
    include the library a dozen times.
    """
    if not parts:
        raise ValueError("no parts to write")

    header = f"include <{library_path}/standard.scad>\nuse <{library_path}/bin.scad>\n"
    modules = "\n".join(part.module for part in parts)
    placements = "\n".join(
        f"translate([{part.x:.4f}, {part.y:.4f}, 0]) rotate([0, 0, {part.degrees:.4f}]) {part.name}();"
        for part in parts
    )

    with open(path, "w") as f:
        f.write(f"{header}\n{modules}\n{placements}\n")


def Available() -> bool:
    """Whether the `openscad` binary can be found."""
    return shutil.which(OPENSCAD) is not None


def RenderScad(scad_path: str, out_path: str) -> None:
    """Render a `.scad` file to `out_path`, in whatever format its
    extension names - the same inference OpenSCAD's own `-o` makes, so a
    caller gets STL, 3MF or anything else OpenSCAD supports just by naming
    it.

    No `--render`/image flags here, unlike `demos/solid_demo.RenderSolid`:
    those only mean something when `-o` names a PNG. For a mesh format
    OpenSCAD always does the full CGAL evaluation regardless, which is
    what a slicer needs to see a manifold solid rather than the fast
    preview's self-intersecting one.
    """
    if not Available():
        raise OSError(f"{OPENSCAD} is not on the path; it renders the .scad this project writes into a mesh")

    subprocess.run(
        [OPENSCAD, "-o", out_path, scad_path],
        check=True,
        capture_output=True,
        text=True,
    )
