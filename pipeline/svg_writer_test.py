import re

import numpy as np
import pytest

from pipeline.svg_writer import Shape, WriteShapesSvg, WriteSvg

# A 20x10 rectangle - deliberately non-square, so its PCA axes are
# unambiguous (a square's principal axes are degenerate/arbitrary).
_RECT = np.array([[0.0, 0.0], [20.0, 0.0], [20.0, 10.0], [0.0, 10.0]], dtype=np.float32)


def _rotate(points: np.ndarray, degrees: float) -> np.ndarray:
    theta = np.radians(degrees)
    rotation = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]], dtype=np.float32)
    return points @ rotation.T


def test_write_svg_raises_on_no_contours(tmp_path):
    with pytest.raises(ValueError):
        WriteSvg(str(tmp_path / "empty.svg"), {})


def test_write_svg_writes_a_polygon_per_contour(tmp_path):
    contours = {0: _RECT, 1: np.array([[0.0, 0.0], [8.0, 0.0], [8.0, 4.0], [0.0, 4.0]], dtype=np.float32)}
    path = tmp_path / "out.svg"

    WriteSvg(str(path), contours)

    svg = path.read_text()
    assert svg.startswith('<?xml version="1.0" encoding="UTF-8"?>')
    assert svg.count("<polygon") == 2
    # Canvas is sized to the largest contour's PCA-local extent.
    assert 'width="20.0000mm" height="10.0000mm"' in svg


def test_write_svg_aligns_a_rotated_contour_to_its_pca_axes(tmp_path):
    # If rotation weren't normalized away, a tilted input would produce a
    # visibly different (and non-level) polygon. Instead, PCA alignment
    # should make a rotated copy of the same rectangle come out identical
    # to the unrotated one.
    level_path = tmp_path / "level.svg"
    tilted_path = tmp_path / "tilted.svg"

    WriteSvg(str(level_path), {0: _RECT})
    WriteSvg(str(tilted_path), {0: _rotate(_RECT, 25)})

    level_svg = level_path.read_text()
    tilted_svg = tilted_path.read_text()

    assert level_svg == tilted_svg
    assert 'width="20.0000mm" height="10.0000mm"' in level_svg
    # Path coordinates are scaled by 96/25.4 (see _SVG_USER_UNITS_PER_MM),
    # so tools that ignore the mm-unit width/height still import this at
    # the right real-world size.
    assert 'points="0.0000,0.0000 75.5906,0.0000 75.5906,37.7953 0.0000,37.7953"' in level_svg


def test_write_svg_viewbox_is_scaled_for_96dpi_importers(tmp_path):
    # width/height stay true mm; the viewBox is pre-scaled by 96/25.4 so
    # importers that assume "1 user unit = 1px @ 96dpi" and ignore the mm
    # suffix entirely (e.g. Fusion 360) still recover the correct
    # real-world size instead of coming out ~3.78x too small.
    path = tmp_path / "out.svg"

    WriteSvg(str(path), {0: _RECT})

    svg = path.read_text()
    match = re.search(r'viewBox="0 0 ([\d.]+) ([\d.]+)"', svg)
    assert match is not None, "no viewBox found in the written SVG"
    viewbox_width, viewbox_height = float(match.group(1)), float(match.group(2))

    assert viewbox_width == pytest.approx(20.0 * 96 / 25.4, abs=1e-3)
    assert viewbox_height == pytest.approx(10.0 * 96 / 25.4, abs=1e-3)


def test_write_svg_aligns_a_translated_contour_to_its_pca_axes(tmp_path):
    # Likewise, translation shouldn't survive into the output - each
    # contour is re-centered onto its own PCA-local frame regardless of
    # where it happened to sit in real-world (calibrated) space.
    origin_path = tmp_path / "origin.svg"
    shifted_path = tmp_path / "shifted.svg"

    WriteSvg(str(origin_path), {0: _RECT})
    WriteSvg(str(shifted_path), {0: _RECT + np.array([500.0, 300.0], dtype=np.float32)})

    assert origin_path.read_text() == shifted_path.read_text()


def test_write_shapes_svg_draws_coordinates_as_given(tmp_path):
    # The core writer's whole reason to exist: a layout's coordinates must
    # reach the page untouched. Aligning them, as WriteSvg does, would
    # stack every part back onto the origin.
    path = tmp_path / "shapes.svg"
    offset = _RECT + np.array([30.0, 5.0], dtype=np.float32)

    WriteShapesSvg(str(path), [Shape(offset)], 100.0, 50.0)

    svg = path.read_text()
    scale = 96 / 25.4
    assert f'points="{30.0 * scale:.4f},{5.0 * scale:.4f}' in svg
    assert 'width="100.0000mm" height="50.0000mm"' in svg


def test_closed_shapes_are_polygons_and_open_ones_are_polylines(tmp_path):
    # layout.loading.LoadSvgContours reads only <polygon>, so this is what
    # keeps a preview's bin outline from reading back as another object to
    # pack.
    path = tmp_path / "mixed.svg"

    WriteShapesSvg(str(path), [Shape(_RECT), Shape(_RECT, closed=False)], 20.0, 10.0)

    svg = path.read_text()
    assert svg.count("<polygon") == 1
    assert svg.count("<polyline") == 1


def test_stroke_width_and_dashes_are_scaled_with_the_coordinates(tmp_path):
    # A user unit is not a millimeter here (see _SVG_USER_UNITS_PER_MM). An
    # unscaled stroke would draw a hairline on geometry 3.78x larger.
    path = tmp_path / "dashed.svg"
    scale = 96 / 25.4

    WriteShapesSvg(str(path), [Shape(_RECT, stroke_width=0.5, dashes=(2.0, 1.0))], 20.0, 10.0)

    svg = path.read_text()
    assert f'stroke-width="{0.5 * scale:.4f}"' in svg
    assert f'stroke-dasharray="{2.0 * scale:.4f},{1.0 * scale:.4f}"' in svg


def test_solid_shapes_carry_no_dasharray(tmp_path):
    path = tmp_path / "solid.svg"

    WriteShapesSvg(str(path), [Shape(_RECT)], 20.0, 10.0)

    assert "stroke-dasharray" not in path.read_text()


def test_write_shapes_svg_rejects_an_empty_or_degenerate_canvas(tmp_path):
    with pytest.raises(ValueError, match="no shapes"):
        WriteShapesSvg(str(tmp_path / "a.svg"), [], 20.0, 10.0)
    with pytest.raises(ValueError, match="positive"):
        WriteShapesSvg(str(tmp_path / "b.svg"), [Shape(_RECT)], 0.0, 10.0)
