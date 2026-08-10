"""Tests for the reference renderings.

These deliberately do **not** compare against the committed images. That
would be the stored-image assertion `render_demo` exists to avoid, and it
would fail on every intentional change to how anything looks - which is
most of the changes these pictures are for.

What is worth pinning is that the gallery keeps working: that every entry
renders, that nothing comes out blank, and above all that the output is
*deterministic*. That last one is the load-bearing property. A picture
that varies run to run makes every diff meaningless and would quietly
turn `make previews` into a source of noise in every commit.
"""

import ast
import pathlib

import numpy as np
import pytest

import demos.render_demo as render_demo
from demos.render_demo import PREFIX, PREVIEWS, Main, Write
from pipeline.layout.parameters import LayoutParameters


def _names() -> list[str]:
    return [name for name, _ in PREVIEWS]


# --------------------------------------------------------------- rendering


@pytest.mark.parametrize("name", _names())
def test_every_preview_renders_something(name, tmp_path):
    """Blank would satisfy a shape check and tell a reviewer nothing, so
    each picture has to have both background and ink in it.
    """
    (path,) = Write(str(tmp_path), names=[name])

    image = _read(path)
    assert image.ndim == 3 and image.shape[2] == 3
    assert image.max() > 200, "expected a light background"
    assert image.min() < 100, "expected something drawn on it"


def test_the_gallery_is_drawn_at_a_non_default_pocket_offset():
    """Deliberate, and worth pinning so a later tidy-up does not quietly
    restore the default.

    A gallery drawn at the library's own default cannot show a drawing
    that hardcoded that default, or one that reached for an object where
    it meant a pocket - both come out looking exactly right. It also has
    to be big enough to see: each part is drawn as a pocket and the
    object inside it, and at these render scales a 1mm ring between them
    is one fuzzy band rather than two lines.
    """
    assert render_demo.DEMO_POCKET_OFFSET_MM != LayoutParameters().pocket_offset
    assert render_demo.DEMO_POCKET_OFFSET_MM > LayoutParameters().pocket_offset


def test_the_gallery_covers_every_drawing_path(tmp_path):
    """The point of the gallery is that each renderer has a picture. A
    path added without one is a path whose appearance nothing watches.
    """
    written = Write(str(tmp_path))

    assert len(written) == len(PREVIEWS)
    assert {path.rsplit(PREFIX, 1)[1].removesuffix(".png") for path in written} == set(_names())


def test_previews_are_deterministic(tmp_path):
    """The property the whole scheme rests on. If a picture varied between
    runs, every `make previews` would produce a diff that meant nothing,
    and the real ones would be lost among them.
    """
    first = Write(str(tmp_path / "a"))
    second = Write(str(tmp_path / "b"))

    for left, right in zip(first, second):
        assert _bytes(left) == _bytes(right), f"{left} is not reproducible"


def test_previews_do_not_depend_on_the_working_directory(tmp_path, monkeypatch):
    """The fixtures are resolved against the module rather than the
    caller, so the pictures cannot depend on where `make previews` was run
    from - which would otherwise show up as a spurious diff.
    """
    expected = _bytes(Write(str(tmp_path / "here"), names=["field_distance"])[0])

    monkeypatch.chdir(tmp_path)
    elsewhere = _bytes(Write(str(tmp_path / "there"), names=["field_distance"])[0])

    assert elsewhere == expected


def test_the_gallery_never_reaches_for_a_stochastic_search():
    """A preview built by the packer would move whenever the solver's
    tuning did, for reasons no reviewer could tell apart from a rendering
    change - and that is precisely the noise this gallery exists to avoid
    generating.

    Checked as an import rather than by patching the search: `grouping`
    binds `SolveFixedGrid` at import time, so patching the solver module
    would leave its caller untouched and this would pass while meaning
    nothing. Reading the module's own imports cannot be fooled that way.
    """
    tree = ast.parse(pathlib.Path(render_demo.__file__).read_text())

    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module)
        elif isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)

    stochastic = {
        "pipeline.layout.solver",
        "pipeline.layout.packer",
        "pipeline.layout.grouping",
        "pipeline.layout.plan",
    }
    assert not (imported & stochastic), f"a reference rendering must not run a search: {imported & stochastic}"


def test_a_preview_is_not_sensitive_to_the_seed(tmp_path):
    """The other half of the same claim, from the outside: change the one
    parameter that steers every stochastic search and nothing moves.
    """
    plain = _bytes(Write(str(tmp_path / "a"), LayoutParameters(seed=0), names=["floorplan"])[0])
    reseeded = _bytes(Write(str(tmp_path / "b"), LayoutParameters(seed=99), names=["floorplan"])[0])

    assert plain == reseeded


# ------------------------------------------------------------------ the cli


def test_the_command_writes_the_whole_gallery(tmp_path, capsys):
    assert Main(["--out", str(tmp_path)]) == 0

    written = sorted(path.name for path in tmp_path.iterdir())
    assert written == sorted(f"{PREFIX}{name}.png" for name in _names())
    assert "wrote" in capsys.readouterr().out


def test_one_preview_can_be_rendered_alone(tmp_path):
    """A few seconds is cheap for the set, and less than that matters when
    iterating on one drawing.
    """
    assert Main(["--out", str(tmp_path), "--only", "bin"]) == 0

    assert [path.name for path in tmp_path.iterdir()] == [f"{PREFIX}bin.png"]


def test_an_unknown_preview_is_refused(tmp_path):
    with pytest.raises(SystemExit):
        Main(["--out", str(tmp_path), "--only", "nonesuch"])


def _read(path: str) -> np.ndarray:
    import cv2

    image = cv2.imread(path)
    assert image is not None, f"{path} is not a readable image"
    return image


def _bytes(path: str) -> bytes:
    with open(path, "rb") as handle:
        return handle.read()
