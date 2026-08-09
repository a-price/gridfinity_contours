"""Tests for the shared rasterizer and signed distance field.

The rasterizer is checked against matplotlib's point-in-polygon test
rather than against a hand-written mask, for the reason `verify` gives:
an independent implementation cannot confirm its own artifacts. That
matters most for the row-span slicing, whose whole claim is that it
changes performance and nothing else.
"""

import numpy as np
import pytest
from matplotlib.path import Path

from pipeline.layout.raster import FieldGrid, RasterizePolygon, SignedDistanceField
from conftest import Rectangle as _rectangle


def _oracle(pixels: np.ndarray, height: int, width: int) -> np.ndarray:
    """Which pixel centers matplotlib puts inside the polygon."""
    grid_x, grid_y = np.meshgrid(np.arange(width, dtype=np.float64), np.arange(height, dtype=np.float64))
    centers = np.stack([grid_x.ravel(), grid_y.ravel()], axis=-1)
    return Path(pixels, closed=False).contains_points(centers).reshape(height, width)


# ------------------------------------------------------------- the mask


@pytest.mark.parametrize("trial", range(50))
def test_rasterize_agrees_with_an_independent_point_in_polygon_test(trial):
    rng = np.random.default_rng(trial)
    # Offset off the integer lattice, so no edge lands exactly on a pixel
    # center where the two implementations' boundary conventions differ.
    polygon = rng.uniform(-2.0, 22.0, size=(int(rng.integers(3, 12)), 2)) + 0.317

    mask = RasterizePolygon(polygon, height=20, width=20)

    assert np.array_equal(mask, _oracle(polygon, height=20, width=20))


def test_rasterize_handles_horizontal_edges():
    """A rectangle is entirely horizontal and vertical edges, and the
    horizontal ones are skipped outright - they cross no horizontal ray.
    """
    square = _rectangle(8, 6, x=1.5, y=1.5)

    assert np.array_equal(RasterizePolygon(square, 12, 12), _oracle(square, 12, 12))


def test_rasterize_ignores_geometry_outside_the_raster():
    """Rows an edge cannot straddle are skipped, so an edge lying entirely
    above or below the raster must contribute nothing rather than wrap.
    """
    far_above = _rectangle(8, 6, x=1.5, y=40.0)

    assert not RasterizePolygon(far_above, 12, 12).any()


# -------------------------------------------------------------- the grid


def test_field_grid_pads_the_polygon_on_every_side():
    origin, height, width = FieldGrid(_rectangle(20, 10, x=5, y=7), pad=2.0, resolution=0.5)

    assert origin == pytest.approx([3.0, 5.0])
    assert (width, height) == (48, 28)  # (20 + 4) / 0.5, (10 + 4) / 0.5


# ------------------------------------------------------------- the field


def test_signed_distance_is_negative_inside_and_positive_outside():
    square = _rectangle(10, 10)
    origin, height, width = FieldGrid(square, pad=2.0, resolution=0.25)

    sdf = SignedDistanceField(square, origin, height, width, 0.25)

    center = ((np.array([5.0, 5.0]) - origin) / 0.25 - 0.5).round().astype(int)
    corner = ((np.array([-1.0, -1.0]) - origin) / 0.25 - 0.5).round().astype(int)
    assert sdf[center[1], center[0]] == pytest.approx(-5.0, abs=0.25)
    assert sdf[corner[1], corner[0]] == pytest.approx(np.sqrt(2.0), abs=0.25)
