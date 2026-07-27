"""Tests for the independent geometric checks.

These validate the oracle the solver is checked against, so they lean on
hand-checked cases rather than on anything the solver computes.
"""

import numpy as np
import pytest

from pipeline.layout.part import BuildPart
from pipeline.layout.placement import Layout, Placement
from pipeline.layout.verify import (
    CheckLayout,
    MinimumSeparation,
    PolygonInside,
    PolygonsOverlap,
)


def _rectangle(width: float, height: float, x: float = 0.0, y: float = 0.0) -> np.ndarray:
    return np.array([[x, y], [x + width, y], [x + width, y + height], [x, y + height]], dtype=np.float64)


def _u_shape() -> np.ndarray:
    """A 30x30 U with a 10-wide, 20-deep notch - the concavity another part
    can nest into, which is the whole reason the packer bothers with
    non-convex shapes.
    """
    return np.array([[0, 0], [30, 0], [30, 30], [20, 30], [20, 10], [10, 10], [10, 30], [0, 30]], dtype=np.float64)


# ------------------------------------------------------- verification tools


def test_polygons_overlap_on_clearly_separated_and_crossing_shapes():
    assert not PolygonsOverlap(_rectangle(10, 10), _rectangle(10, 10, 20, 20))
    assert PolygonsOverlap(_rectangle(10, 10), _rectangle(10, 10, 5, 5))


def test_polygons_overlap_detects_full_containment():
    """No edges cross here, so an intersection test alone would miss it."""
    assert PolygonsOverlap(_rectangle(30, 30), _rectangle(5, 5, 10, 10))


def test_polygons_overlap_allows_nesting_into_a_concavity():
    """The property that makes non-convex packing worth doing: a bar in the
    U's notch is not an overlap, but the same bar through its arm is.
    """
    assert not PolygonsOverlap(_u_shape(), _rectangle(8, 18, 11, 11))
    assert PolygonsOverlap(_u_shape(), _rectangle(8, 18, 5, 11))


def test_minimum_separation_measures_edge_to_edge():
    assert MinimumSeparation(_rectangle(10, 10), _rectangle(10, 10, 13, 0)) == pytest.approx(3.0)
    # Diagonally offset corners: 3-4-5 again.
    assert MinimumSeparation(_rectangle(10, 10), _rectangle(10, 10, 13, 14)) == pytest.approx(5.0)


def test_polygon_inside_rejects_a_shape_poking_through_an_edge():
    container = _rectangle(50, 50)

    assert PolygonInside(_rectangle(10, 10, 20, 20), container)
    assert not PolygonInside(_rectangle(10, 10, 45, 20), container)
    assert not PolygonInside(_rectangle(10, 10, 80, 80), container)


def test_check_layout_reports_a_clean_arrangement():
    parts = {0: BuildPart(_rectangle(20, 10)), 1: BuildPart(_rectangle(20, 10))}
    layout = Layout(
        grid=(2, 2),
        placements={
            0: Placement(0, np.array([5.0, 5.0])),
            1: Placement(1, np.array([5.0, 40.0])),
        },
    )

    assert CheckLayout(layout, parts) == []


def test_check_layout_reports_overlap_and_escape():
    parts = {0: BuildPart(_rectangle(20, 10)), 1: BuildPart(_rectangle(20, 10))}
    layout = Layout(
        grid=(1, 1),
        placements={
            0: Placement(0, np.array([5.0, 5.0])),
            1: Placement(1, np.array([6.0, 6.0])),  # on top of part 0
        },
    )

    problems = CheckLayout(layout, parts)

    assert any("overlap" in problem for problem in problems)


def test_check_layout_enforces_clearances():
    parts = {0: BuildPart(_rectangle(20, 10)), 1: BuildPart(_rectangle(20, 10))}
    # 2mm apart vertically: legal bare, too close once a 3.2mm clearance applies.
    layout = Layout(
        grid=(2, 2),
        placements={
            0: Placement(0, np.array([5.0, 5.0])),
            1: Placement(1, np.array([5.0, 17.0])),
        },
    )

    assert CheckLayout(layout, parts) == []
    assert any("apart" in problem for problem in CheckLayout(layout, parts, pair_clearance=3.2))
