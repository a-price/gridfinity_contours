"""Tests for the Gridfinity bin interior: the spec-derived spans, the
envelope polygon, and the analytic distance function into it."""

import numpy as np
import pytest

from pipeline.layout.container import (
    BASE_GAP_MM,
    GRID_PITCH_MM,
    BuildContainer,
    GridSizes,
    InteriorSpan,
)
from pipeline.layout.verify import DistanceToBoundary, PolygonInside
from conftest import Rectangle as _rectangle


def test_interior_span_matches_the_gridfinity_spec():
    # A 1x1's usable interior is the commonly-quoted ~36mm: 42 - 0.5 gap
    # - 2 * 2.6 lip intrusion.
    assert InteriorSpan(1) == pytest.approx(36.3)
    assert InteriorSpan(5) == pytest.approx(204.3)


def test_interior_span_grows_by_one_pitch_per_cell():
    for cells in range(1, 7):
        assert InteriorSpan(cells + 1) - InteriorSpan(cells) == pytest.approx(GRID_PITCH_MM)


def test_interior_span_without_a_lip_gives_back_the_intrusion():
    assert InteriorSpan(1, inset=0.0) == pytest.approx(GRID_PITCH_MM - BASE_GAP_MM)


def test_grid_sizes_omit_the_rotations_of_what_they_already_list():
    grids = GridSizes(4)

    assert all(n >= m for n, m in grids)
    assert len(grids) == len(set(grids))


def test_grid_sizes_respect_the_cap():
    assert max(max(grid) for grid in GridSizes(3)) == 3
    with pytest.raises(ValueError):
        GridSizes(0)


def test_interior_envelope_spans_the_usable_interior():
    envelope = BuildContainer(5, 2).Polygon()

    assert envelope.min(axis=0) == pytest.approx([0.0, 0.0])
    assert envelope.max(axis=0) == pytest.approx([InteriorSpan(5), InteriorSpan(2)])


def test_interior_envelope_corners_are_rounded_inward():
    envelope = BuildContainer(2, 2).Polygon()

    # A rounded rectangle excludes its own bounding box corners.
    assert not PolygonInside(_rectangle(0.2, 0.2, -0.05, -0.05), envelope)


def test_building_a_container_rejects_a_degenerate_grid():
    with pytest.raises(ValueError):
        BuildContainer(0, 3)


# ------------------------------------------------- distance to the wall

# -------------------------------------------------------------- container


def test_container_depth_is_positive_inside_and_negative_outside():
    container = BuildContainer(2, 2)

    assert container.SampleDepth(np.array([[container.width / 2, container.height / 2]]))[0] > 0
    assert container.SampleDepth(np.array([[-5.0, container.height / 2]]))[0] < 0


def test_container_depth_measures_distance_to_the_wall():
    container = BuildContainer(3, 2)

    # 4mm in from the left wall, and 4mm out past it.
    assert container.SampleDepth(np.array([[4.0, container.height / 2]]))[0] == pytest.approx(4.0, abs=0.01)
    assert container.SampleDepth(np.array([[-4.0, container.height / 2]]))[0] == pytest.approx(-4.0, abs=0.01)


def test_container_depth_agrees_with_its_own_polygon():
    """The analytic rounded-rectangle distance against an exact measurement
    from the tessellated boundary - two unrelated routes to the same number.
    """
    container = BuildContainer(3, 2)
    polygon = container.Polygon(segments_per_corner=64)
    x = np.linspace(2.0, container.width - 2.0, 25)
    y = np.linspace(2.0, container.height - 2.0, 15)
    query = np.stack(np.meshgrid(x, y), axis=-1).reshape(-1, 2)

    assert container.SampleDepth(query) == pytest.approx(DistanceToBoundary(query, polygon), abs=0.02)


def test_container_derivative_points_inward_from_every_wall():
    container = BuildContainer(3, 2)
    width, height = container.width, container.height

    assert container.SampleDerivative(np.array([[2.0, height / 2]]))[0] == pytest.approx([1.0, 0.0], abs=1e-6)
    assert container.SampleDerivative(np.array([[width - 2.0, height / 2]]))[0] == pytest.approx([-1.0, 0.0], abs=1e-6)
    assert container.SampleDerivative(np.array([[width / 2, 2.0]]))[0] == pytest.approx([0.0, 1.0], abs=1e-6)
    assert container.SampleDerivative(np.array([[width / 2, height - 2.0]]))[0] == pytest.approx([0.0, -1.0], abs=1e-6)


def test_container_derivative_matches_finite_differences():
    container = BuildContainer(3, 2)
    rng = np.random.default_rng(0)
    query = rng.uniform([-5.0, -5.0], [container.width + 5.0, container.height + 5.0], size=(200, 2))
    step = 1e-6

    for axis in range(2):
        offset = np.zeros(2)
        offset[axis] = step
        numeric = (container.SampleDepth(query + offset) - container.SampleDepth(query - offset)) / (2 * step)

        assert container.SampleDerivative(query)[:, axis] == pytest.approx(numeric, abs=1e-4)


def test_container_span_matches_the_layout_it_reports():
    container = BuildContainer(5, 2)

    assert container.width == pytest.approx(InteriorSpan(5))
    assert container.height == pytest.approx(InteriorSpan(2))
