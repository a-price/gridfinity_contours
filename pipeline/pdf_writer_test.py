import re

import matplotlib

matplotlib.use("Agg")  # headless: no window should pop up when saving the PDF

import numpy as np
import pytest

from pipeline.pdf_writer import WritePdf

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
