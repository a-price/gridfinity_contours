"""Tests for the rendered bin pictures.

Split three ways by what each part needs. Cropping is pure array work and
is tested on hand-built images, which is where the only real logic lives.
Writing the `.scad` needs the packer but not OpenSCAD, so it runs as a
slow test. Actually rendering needs OpenSCAD on the path, so it is skipped
where there is none rather than failing - a machine without it can still
run the suite, exactly as one without graphviz can still build the docs.
"""

import os

import numpy as np
import pytest

import demos.solid_demo as solid_demo
from demos.solid_demo import ONE, PREFIX, SOLIDS, SPOONS, Available, Trim, WriteSolid


def _Image(fill: int = 245, size: tuple[int, int] = (100, 120)) -> np.ndarray:
    return np.full((size[0], size[1], 3), fill, dtype=np.uint8)


# ------------------------------------------------------------------ cropping


def test_cropping_keeps_the_object_and_drops_the_page():
    """The whole job. `--viewall` frames a wide flat bin inside a square
    render, so most of the picture is empty page and the pockets shrink to
    nothing at README width.
    """
    image = _Image()
    image[40:60, 30:90] = 20

    trimmed = Trim(image, margin=0)

    assert trimmed.shape[:2] == (20, 60)
    assert trimmed.min() == 20


def test_cropping_leaves_the_margin_it_is_asked_for():
    image = _Image()
    image[40:60, 30:90] = 20

    assert Trim(image, margin=5).shape[:2] == (30, 70)


def test_a_margin_cannot_run_off_the_edge():
    """An object touching the frame is the case that would otherwise ask
    for a negative index and silently crop from the wrong side.
    """
    image = _Image()
    image[0:10, 30:60] = 20  # against the top edge

    assert Trim(image, margin=20).shape[:2] == (30, 70)


def test_the_background_is_read_from_a_corner():
    """The one assumption cropping makes, and it holds by construction:
    `--viewall --autocenter` centres the bin, so the corners are page.
    Stated as a test because a render that ever filled the frame would
    make this silently keep the whole picture rather than crop wrongly.
    """
    image = _Image()
    image[40:60, 30:90] = 20

    assert tuple(image[0, 0]) != (20, 20, 20)
    assert Trim(image, margin=0).shape[:2] == (20, 60)


def test_an_empty_render_is_left_alone():
    """A blank picture is easier to diagnose than a zero-size one, and a
    zero-size crop is what `cv2.imwrite` refuses without saying why.
    """
    blank = _Image()

    assert Trim(blank).shape == blank.shape


def test_cropping_ignores_compression_noise():
    """The background is flat, so the tolerance only has to survive PNG
    quantization - it is not trying to be clever about edges.
    """
    image = _Image()
    image[10:20, 10:20] = 245 + solid_demo.TRIM_TOLERANCE - 1
    image[40:60, 30:90] = 20

    assert Trim(image, margin=0).shape[:2] == (20, 60)


# ------------------------------------------------------------------ fixtures


def test_every_fixture_it_names_is_there():
    for path in [*SPOONS, *ONE]:
        assert os.path.exists(path), f"{path} is missing"


def test_the_pictures_are_of_different_things():
    """One bin holding several objects and one holding a single object are
    the two cases the README distinguishes - `layout_cli.py` against
    `solid.py`. Two pictures of the same thing would document neither.
    """
    assert len(SPOONS) > 1 and len(ONE) == 1


def test_it_says_so_rather_than_crashing_without_openscad(monkeypatch, tmp_path):
    """`FileNotFoundError` from inside a subprocess call is a poor way to
    learn that a build tool is missing.
    """
    monkeypatch.setattr(solid_demo.shutil, "which", lambda _: None)

    assert not Available()
    with pytest.raises(OSError, match="not on the path"):
        solid_demo.RenderSolid("bin.scad", str(tmp_path / "bin.png"))


# ------------------------------------------------------------------ the bins


@pytest.mark.slow
def test_the_scad_it_writes_is_the_bin_the_packer_found(tmp_path):
    """Packed through the same three calls `layout_cli.py` makes, so the
    picture is of the tool rather than of a shortcut past it.
    """
    path = str(tmp_path / "bin.scad")

    WriteSolid(path, SPOONS)

    program = open(path).read()
    assert "new_bin(" in program
    assert program.count("polygon(points=") == len(SPOONS), "one pocket per object"


@pytest.mark.slow
@pytest.mark.skipif(not Available(), reason="OpenSCAD renders these pictures, and only these")
def test_a_rendered_bin_is_a_picture_of_something(tmp_path):
    """About ten seconds of CGAL per bin. Asserts only that a render
    happened and was cropped to it - what the bin *looks* like is the
    judgement the committed picture exists for.
    """
    import cv2

    (image,) = solid_demo.Write(str(tmp_path), names=["pocket"])

    assert image.endswith(f"{PREFIX}pocket.png")
    rendered = cv2.imread(image)
    assert rendered is not None
    assert rendered.shape[0] < 900, "an uncropped render is the full frame"
    assert len(np.unique(rendered.reshape(-1, 3), axis=0)) > 8, "a solid colour is not a bin"


def test_the_gallery_names_are_usable_on_the_command_line():
    """`--only` is choice-checked against this table, so a name with a
    space or a slash in it would be unselectable.
    """
    for name, _ in SOLIDS:
        assert name.isidentifier()
