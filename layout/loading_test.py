"""Tests for SVG contour loading, and for the test_data fixtures."""

import pytest

from layout.container import BuildContainer, InteriorSpan
from layout.loading import LoadParts, LoadSvgContours
from layout.parameters import LayoutParameters
from layout.part import BuildPart, PolygonArea
from conftest import SPOONS

# ---------------------------------------------------------------- fixtures


def test_svg_fixtures_load_at_their_measured_sizes():
    """The M1 fixture criterion. These files are 1:1 mm while the current
    writer pre-scales by 96/25.4, so a loader that hardcoded either
    constant would be 3.78x wrong on one of them.
    """
    expected = {
        "test_data/big_spoon.svg": (200.2648, 41.67),
        "test_data/medium_spoon.svg": (162.76, 34.89),
        "test_data/small_spoon.svg": (73.93, 14.20),
    }

    for path, (length, width) in expected.items():
        (contour,) = LoadSvgContours(path)
        extent = contour.max(axis=0) - contour.min(axis=0)

        assert extent == pytest.approx([length, width], abs=0.01), path


def test_svg_scale_is_derived_from_the_viewbox_not_assumed(tmp_path):
    """The same geometry in both conventions must load identically."""
    template = (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        '<svg xmlns="http://www.w3.org/2000/svg" width="100.0000mm" height="50.0000mm" '
        'viewBox="0 0 {vw} {vh}">\n'
        '  <polygon points="{points}" fill="none" stroke="black" />\n'
        "</svg>\n"
    )
    scale = 96.0 / 25.4
    one_to_one = tmp_path / "plain.svg"
    prescaled = tmp_path / "prescaled.svg"
    one_to_one.write_text(template.format(vw=100.0, vh=50.0, points="0,0 100,0 100,50 0,50"))
    prescaled.write_text(
        template.format(
            vw=100.0 * scale,
            vh=50.0 * scale,
            points=" ".join(f"{x * scale},{y * scale}" for x, y in [(0, 0), (100, 0), (100, 50), (0, 50)]),
        )
    )

    (plain_contour,) = LoadSvgContours(str(one_to_one))
    (scaled_contour,) = LoadSvgContours(str(prescaled))

    assert plain_contour == pytest.approx(scaled_contour, abs=1e-6)
    assert plain_contour.max(axis=0) == pytest.approx([100.0, 50.0])


def test_svg_loader_refuses_units_it_cannot_interpret(tmp_path):
    path = tmp_path / "pixels.svg"
    path.write_text(
        '<svg xmlns="http://www.w3.org/2000/svg" width="100" height="50" viewBox="0 0 100 50">'
        '<polygon points="0,0 100,0 100,50" /></svg>'
    )

    with pytest.raises(ValueError, match="no unit"):
        LoadSvgContours(str(path))


def test_spoon_fixtures_build_parts_with_the_expected_footprints():
    """The measured lengths are the *objects*; what `LoadParts` hands back
    is their pockets, so the two are checked separately.

    Stating the growth as a bound rather than a fourth measured number is
    the point: it is exactly the guarantee `pocket` sells. Never less than
    twice the offset, because the trace is deliberately one-sided so a
    pocket always contains the ideal dilation; never more than the raster
    cell plus simplification tolerance it spends per side.
    """
    params = LayoutParameters()
    parts = LoadParts(SPOONS, params)

    assert len(parts) == 3
    objects = sorted(
        float(part.object_contour[:, 0].max() - part.object_contour[:, 0].min()) for part in parts.values()
    )
    assert objects == pytest.approx([73.93, 162.76, 200.26], abs=0.05)

    slack = 2.0 * (params.pocket_resolution + params.pocket_simplify)
    for part in parts.values():
        span = float(part.object_contour[:, 0].max() - part.object_contour[:, 0].min())
        assert 2.0 * params.pocket_offset <= float(part.size[0]) - span <= 2.0 * params.pocket_offset + slack


def test_spoon_areas_match_their_contours():
    """Guards the raster pipeline against a silent scale error: a part's
    area is computed from the aligned polygon, so it must still agree with
    the area of the contour as loaded.
    """
    for path, expected in zip(SPOONS, [3413.9, 2355.5, 436.6]):
        (contour,) = LoadSvgContours(path)
        part = BuildPart(contour)

        assert part.area == pytest.approx(expected, rel=0.01), path
        assert part.area == pytest.approx(PolygonArea(contour), rel=1e-6)


def test_big_spoon_barely_clears_a_five_cell_run():
    """The design's motivating edge case: 200.26mm against a 204.3mm
    interior leaves 0.04mm once a 2mm wall clearance applies at each end -
    an order of magnitude under the raster resolution. M4's extent bound
    has to treat that as infeasible rather than merely tight.
    """
    (contour,) = LoadSvgContours("test_data/big_spoon.svg")
    part = BuildPart(contour)
    wall_clearance = 2.0

    slack = InteriorSpan(5) - (float(part.size[0]) + 2 * wall_clearance)

    assert 0 < slack < part.resolution


def test_spoons_fit_a_five_by_two_with_room_to_spare():
    """The M4 target, checked here at the geometry level: the three spoons
    have somewhere to go in a 5x2, so a later packing failure is the
    solver's doing and not an impossible bin.
    """
    parts = LoadParts(SPOONS)
    envelope = BuildContainer(5, 2).Polygon()
    interior_area = PolygonArea(envelope)

    assert sum(part.area for part in parts.values()) < 0.5 * interior_area
    for part in parts.values():
        assert part.size[0] < InteriorSpan(5)
        assert part.size[1] < InteriorSpan(2)
