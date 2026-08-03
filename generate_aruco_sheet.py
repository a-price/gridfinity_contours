"""Generates a letter-size calibration sheet with ArUco markers at known
real-world (mm) positions, for use with pipeline.calibration.ArucoCalibration.

Usage:
    python3 generate_aruco_sheet.py [output.pdf]

Print at 100% scale ("actual size" / no "fit to page") - any scaling
invalidates the mm positions this script reports. After printing, copy the
reported positions into ArucoCalibration.parameters.marker_positions_mm
(they already match ArucoParameters' defaults, so no size/dictionary
changes are needed if you don't touch the constants below).

The format follows the extension, so `sheet.png` writes a picture of the
sheet rather than a PDF named `.png`. PDF is what you print: its page size
is unambiguous, which is exactly the property a sheet whose whole purpose
is real-world scale needs.
"""

import sys

import cv2
import matplotlib.pyplot as plt

from pipeline.calibration import (
    ARUCO_MARKER_SIZE_MM,
    ARUCO_SHEET_MARGIN_MM,
    DefaultArucoMarkerPositions,
    PaperCalibration,
)
from pipeline.core import FixQtOpenCvPluginPath

FixQtOpenCvPluginPath()


PAGE_WIDTH_MM = PaperCalibration.WIDTH_MM
PAGE_HEIGHT_MM = PaperCalibration.HEIGHT_MM
MM_PER_INCH = 25.4

# Sourced from calibration.py so this sheet and ArucoParameters' defaults
# can never drift apart.
ARUCO_DICTIONARY = cv2.aruco.DICT_4X4_50
MARKER_SIZE_MM = ARUCO_MARKER_SIZE_MM
MARGIN_MM = ARUCO_SHEET_MARGIN_MM
MARKER_PIXELS = 800  # raster resolution per marker, for a crisp print

# Only reached by a raster format. A PDF carries the markers as images at
# their own resolution, so this changes nothing about the sheet you print.
DEFAULT_DPI = 150


def MarkerPositions() -> dict[int, tuple[float, float]]:
    """Marker IDs 0-3, one near each corner (top-left origin, y down - the
    same convention as PaperCalibration/ArucoCalibration), inset far enough
    from the page edge to survive typical printer margins.
    """
    return DefaultArucoMarkerPositions(MARKER_SIZE_MM, PAGE_WIDTH_MM, PAGE_HEIGHT_MM, MARGIN_MM)


def GenerateSheet(output_path: str, dpi: int = DEFAULT_DPI) -> None:
    dictionary = cv2.aruco.getPredefinedDictionary(ARUCO_DICTIONARY)
    positions = MarkerPositions()
    half = MARKER_SIZE_MM / 2

    fig = plt.figure(figsize=(PAGE_WIDTH_MM / MM_PER_INCH, PAGE_HEIGHT_MM / MM_PER_INCH))
    ax = fig.add_axes((0, 0, 1, 1))  # fill the whole page, no margins
    ax.set_xlim(0, PAGE_WIDTH_MM)
    ax.set_ylim(PAGE_HEIGHT_MM, 0)  # origin top-left, y increases downward
    ax.axis("off")

    for marker_id, (cx, cy) in positions.items():
        marker_image = cv2.aruco.generateImageMarker(dictionary, marker_id, MARKER_PIXELS)
        ax.imshow(
            marker_image,
            cmap="gray",
            vmin=0,
            vmax=255,
            extent=(cx - half, cx + half, cy + half, cy - half),
            interpolation="nearest",
        )
        ax.text(
            cx,
            cy - half - 2,
            f"ID {marker_id}",
            ha="center",
            va="bottom",
            fontsize=8,
        )

    ax.text(
        PAGE_WIDTH_MM / 2,
        PAGE_HEIGHT_MM - 5,
        "Print at 100% scale (actual size, not “fit to page”)",
        ha="center",
        va="bottom",
        fontsize=7,
    )

    # Format from the extension rather than forced, so this one function
    # writes both the sheet you print and the picture of it in the README.
    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)

    print(f"Wrote {output_path}")
    print("Marker positions (mm, top-left origin) for ArucoCalibration.parameters:")
    print(f"  marker_size_mm = {MARKER_SIZE_MM}")
    print("  marker_positions_mm = {")
    for marker_id, (x, y) in positions.items():
        print(f"      {marker_id}: ({x:.1f}, {y:.1f}),")
    print("  }")


def main() -> None:
    output_path = sys.argv[1] if len(sys.argv) > 1 else "aruco_calibration_sheet.pdf"
    GenerateSheet(output_path)


if __name__ == "__main__":
    main()
