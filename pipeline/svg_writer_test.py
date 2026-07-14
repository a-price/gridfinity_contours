import numpy as np
import pytest

from pipeline.svg_writer import WriteSvg

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
    assert 'points="0.0000,0.0000 20.0000,0.0000 20.0000,10.0000 0.0000,10.0000"' in level_svg


def test_write_svg_aligns_a_translated_contour_to_its_pca_axes(tmp_path):
    # Likewise, translation shouldn't survive into the output - each
    # contour is re-centered onto its own PCA-local frame regardless of
    # where it happened to sit in real-world (calibrated) space.
    origin_path = tmp_path / "origin.svg"
    shifted_path = tmp_path / "shifted.svg"

    WriteSvg(str(origin_path), {0: _RECT})
    WriteSvg(str(shifted_path), {0: _RECT + np.array([500.0, 300.0], dtype=np.float32)})

    assert origin_path.read_text() == shifted_path.read_text()
