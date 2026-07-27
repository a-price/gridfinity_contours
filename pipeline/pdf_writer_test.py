import re

import matplotlib

matplotlib.use("Agg")  # headless: no window should pop up when saving the PDF

import numpy as np
import pytest

from pipeline.pdf_writer import WritePdf, WriteShapesPdf
from pipeline.svg_writer import Shape

_RECT = np.array([[0.0, 0.0], [20.0, 0.0], [20.0, 10.0], [0.0, 10.0]], dtype=np.float32)


def test_write_pdf_raises_on_no_contours(tmp_path):
    with pytest.raises(ValueError):
        WritePdf(str(tmp_path / "empty.pdf"), {})


def test_write_pdf_writes_a_pdf_file(tmp_path):
    path = tmp_path / "out.pdf"

    WritePdf(str(path), {0: _RECT})

    assert path.exists()
    assert path.read_bytes().startswith(b"%PDF")


def test_write_pdf_page_size_matches_the_aligned_extent_in_mm(tmp_path):
    # A 20x10mm rectangle should produce a page sized 20/25.4 x 10/25.4 in
    # (the PDF's MediaBox is in points - 1/72 in).
    path = tmp_path / "out.pdf"

    WritePdf(str(path), {0: _RECT})

    match = re.search(rb"/MediaBox \[ 0 0 ([\d.]+) ([\d.]+) \]", path.read_bytes())
    assert match is not None, "no MediaBox found in the written PDF"
    width_pt, height_pt = float(match.group(1)), float(match.group(2))

    assert width_pt == pytest.approx(20.0 / 25.4 * 72, abs=0.01)
    assert height_pt == pytest.approx(10.0 / 25.4 * 72, abs=0.01)


def test_write_shapes_pdf_page_size_is_the_canvas_not_the_content(tmp_path):
    # The core writer is handed a page size rather than deriving one from
    # the geometry: a layout's page is the bin, whether or not any part
    # reaches its edge.
    path = tmp_path / "shapes.pdf"

    WriteShapesPdf(str(path), [Shape(_RECT)], 84.0, 42.0)

    match = re.search(rb"/MediaBox \[ 0 0 ([\d.]+) ([\d.]+) \]", path.read_bytes())
    assert match is not None, "no MediaBox found in the written PDF"
    assert float(match.group(1)) == pytest.approx(84.0 / 25.4 * 72, abs=0.01)
    assert float(match.group(2)) == pytest.approx(42.0 / 25.4 * 72, abs=0.01)


def test_write_shapes_pdf_draws_dashed_and_open_shapes(tmp_path):
    path = tmp_path / "styled.pdf"

    WriteShapesPdf(
        str(path),
        [Shape(_RECT), Shape(_RECT, closed=False, stroke="#808080", dashes=(2.0, 1.0))],
        20.0,
        10.0,
    )

    assert path.read_bytes().startswith(b"%PDF")


def test_write_shapes_pdf_rejects_an_empty_or_degenerate_page(tmp_path):
    with pytest.raises(ValueError, match="no shapes"):
        WriteShapesPdf(str(tmp_path / "a.pdf"), [], 20.0, 10.0)
    with pytest.raises(ValueError, match="positive"):
        WriteShapesPdf(str(tmp_path / "b.pdf"), [Shape(_RECT)], 20.0, -1.0)
