"""Tests for scad_writer (M2).

No OpenSCAD needed for `WriteScadScene`, which only ever assembles text.
`RenderScad`/`Available()` are covered the same way
`demos/solid_demo_test.py` covers a missing binary - by monkeypatching
`shutil.which` rather than requiring the real thing.
"""

import os
import shutil

import pytest

from export.scad_writer import Available, RenderScad, ScadPart, WriteScadScene

_LIBRARY_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "gridfinity-rebuilt-openscad", "src", "core")
)


def _part(name: str, x: float = 0.0, y: float = 0.0, degrees: float = 0.0) -> ScadPart:
    return ScadPart(name, module=f"module {name}() {{ cube([1, 1, 1]); }}", x=x, y=y, degrees=degrees)


def test_write_scad_scene_raises_on_no_parts(tmp_path):
    with pytest.raises(ValueError):
        WriteScadScene(str(tmp_path / "empty.scad"), "/lib", [])


def test_the_header_is_written_once(tmp_path):
    """Not once per part - `layout.solid.GenerateBinModule` deliberately
    omits its own header, so the scene is the only place it appears.
    """
    path = tmp_path / "scene.scad"

    WriteScadScene(str(path), "/lib", [_part("bin_0"), _part("bin_1")])

    scad = path.read_text()
    assert scad.count("include <") == 1
    assert scad.count("use <") == 1
    assert "/lib/standard.scad" in scad


def test_every_part_gets_its_own_module_and_placement(tmp_path):
    path = tmp_path / "scene.scad"

    WriteScadScene(str(path), "/lib", [_part("bin_0", x=1.0, y=2.0), _part("bin_1", x=3.0, y=-4.0, degrees=-90.0)])

    scad = path.read_text()
    assert "module bin_0() {" in scad and "module bin_1() {" in scad
    assert "translate([1.0000, 2.0000, 0]) rotate([0, 0, 0.0000]) bin_0();" in scad
    assert "translate([3.0000, -4.0000, 0]) rotate([0, 0, -90.0000]) bin_1();" in scad


def test_parts_are_placed_in_the_order_given(tmp_path):
    """Placement calls come after every module definition, so a part
    later in the list can never be instantiated before its own module (or
    anyone else's) is defined.
    """
    path = tmp_path / "scene.scad"

    WriteScadScene(str(path), "/lib", [_part("bin_0"), _part("bin_1")])

    scad = path.read_text()
    last_module = max(scad.rindex("module bin_0()"), scad.rindex("module bin_1()"))
    first_placement = min(scad.index("bin_0();"), scad.index("bin_1();"))
    assert last_module < first_placement


# --------------------------------------------------------------- rendering


def test_it_says_so_rather_than_crashing_without_openscad(monkeypatch, tmp_path):
    """`FileNotFoundError` from inside a subprocess call is a poor way to
    learn that a build tool is missing.
    """
    import export.scad_writer as scad_writer

    monkeypatch.setattr(scad_writer.shutil, "which", lambda _: None)

    assert not Available()
    with pytest.raises(OSError, match="not on the path"):
        RenderScad("scene.scad", str(tmp_path / "scene.3mf"))


@pytest.mark.slow
@pytest.mark.skipif(shutil.which("openscad") is None, reason="openscad is not installed")
def test_a_scene_of_two_parts_renders_to_a_mesh(tmp_path):
    """Two ordinary cubes rather than a real bin - this is exercising
    `WriteScadScene`/`RenderScad`'s own mechanics (several modules, each
    placed and instantiated, rendering as one file), not the pocket
    geometry `layout.solid`/`layout.scene` are responsible for getting
    right.
    """
    scad = tmp_path / "scene.scad"
    stl = tmp_path / "scene.stl"
    parts = [
        ScadPart("cube_0", "module cube_0() { cube([10, 10, 10]); }", x=0.0, y=0.0),
        ScadPart("cube_1", "module cube_1() { cube([10, 10, 10]); }", x=50.0, y=0.0, degrees=45.0),
    ]

    WriteScadScene(str(scad), _LIBRARY_PATH, parts)
    RenderScad(str(scad), str(stl))

    assert stl.stat().st_size > 0
