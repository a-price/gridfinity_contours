"""Tests for the Gridfinity bin interior."""

import numpy as np
import pytest

from pipeline.layout.container import GRID_PITCH_MM, BASE_GAP_MM, InteriorEnvelope, InteriorSpan
from pipeline.layout.verify import PolygonInside


def _rectangle(width: float, height: float, x: float = 0.0, y: float = 0.0) -> np.ndarray:
    return np.array([[x, y], [x + width, y], [x + width, y + height], [x, y + height]], dtype=np.float64)


def _l_shape() -> np.ndarray:
    """A 30x30 L with a 20x20 bite taken out of the top right corner."""
    return np.array([[0, 0], [30, 0], [30, 10], [10, 10], [10, 30], [0, 30]], dtype=np.float64)


def _u_shape() -> np.ndarray:
    """A 30x30 U with a 10-wide, 20-deep notch - the concavity another part
    can nest into, which is the whole reason the packer bothers with
    non-convex shapes.
    """
    return np.array([[0, 0], [30, 0], [30, 30], [20, 30], [20, 10], [10, 10], [10, 30], [0, 30]], dtype=np.float64)


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


def test_interior_envelope_spans_the_usable_interior():
    envelope = InteriorEnvelope(5, 2)

    assert envelope.min(axis=0) == pytest.approx([0.0, 0.0])
    assert envelope.max(axis=0) == pytest.approx([InteriorSpan(5), InteriorSpan(2)])


def test_interior_envelope_corners_are_rounded_inward():
    envelope = InteriorEnvelope(2, 2)

    # A rounded rectangle excludes its own bounding box corners.
    assert not PolygonInside(_rectangle(0.2, 0.2, -0.05, -0.05), envelope)


def test_interior_envelope_rejects_a_degenerate_grid():
    with pytest.raises(ValueError):
        InteriorEnvelope(0, 3)
