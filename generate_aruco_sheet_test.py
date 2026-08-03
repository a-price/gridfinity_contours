import matplotlib

matplotlib.use("Agg")  # headless: no window should pop up when saving the PDF

from pipeline.calibration import ArucoParameters
from generate_aruco_sheet import MARKER_SIZE_MM, GenerateSheet, MarkerPositions


def test_generate_sheet_writes_a_pdf(tmp_path):
    output_path = tmp_path / "sheet.pdf"

    GenerateSheet(str(output_path))

    assert output_path.exists()
    assert output_path.read_bytes().startswith(b"%PDF")


def test_the_extension_decides_the_format(tmp_path):
    """A PDF is what you print - its page size is unambiguous, which is
    the property a sheet whose whole purpose is real-world scale needs.
    A PNG is what the docs show. Forcing PDF regardless of the name wrote
    a PDF called `.png`, which nothing would open.
    """
    output_path = tmp_path / "sheet.png"

    GenerateSheet(str(output_path))

    assert output_path.read_bytes().startswith(b"\x89PNG")


def test_marker_positions_are_inside_the_page_with_margin():
    from generate_aruco_sheet import MARGIN_MM, PAGE_HEIGHT_MM, PAGE_WIDTH_MM

    half = MARKER_SIZE_MM / 2
    for x, y in MarkerPositions().values():
        assert MARGIN_MM <= x - half and x + half <= PAGE_WIDTH_MM - MARGIN_MM
        assert MARGIN_MM <= y - half and y + half <= PAGE_HEIGHT_MM - MARGIN_MM


def test_defaults_match_aruco_calibration_parameters():
    # A freshly printed sheet should work with ArucoCalibration's defaults
    # with no configuration - if either side's constants drift, this fails.
    assert MARKER_SIZE_MM == ArucoParameters().marker_size_mm
    assert MarkerPositions() == ArucoParameters().marker_positions_mm
