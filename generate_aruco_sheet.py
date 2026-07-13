"""Generates a letter-size PDF calibration sheet with ArUco markers at
known real-world (mm) positions, for use with calibration.ArucoCalibration.

Usage:
    python3 generate_aruco_sheet.py [output.pdf]

Print at 100% scale ("actual size" / no "fit to page") - any scaling
invalidates the mm positions this script reports. After printing, copy the
reported positions into ArucoCalibration.parameters.marker_positions_mm
(they already match ArucoParameters' defaults, so no size/dictionary
changes are needed if you don't touch the constants below).
"""

import os
import sys

import cv2
import matplotlib.pyplot as plt
from PyQt5.QtCore import QLibraryInfo


from calibration import PaperCalibration

# Fix PyQt5 / OpenCV collision
os.environ["QT_QPA_PLATFORM_PLUGIN_PATH"] = QLibraryInfo.location(
    QLibraryInfo.PluginsPath
)


PAGE_WIDTH_MM = PaperCalibration.WIDTH_MM
PAGE_HEIGHT_MM = PaperCalibration.HEIGHT_MM
MM_PER_INCH = 25.4

# Match ArucoParameters' defaults so a freshly printed sheet works with
# ArucoCalibration out of the box.
ARUCO_DICTIONARY = cv2.aruco.DICT_4X4_50
MARKER_SIZE_MM = 20.0
MARKER_PIXELS = 800  # raster resolution per marker, for a crisp print

MARGIN_MM = 15.0  # marker-edge-to-page-edge clearance


def MarkerPositions() -> dict[int, tuple[float, float]]:
    """Marker IDs 0-3, one near each corner (top-left origin, y down - the
    same convention as PaperCalibration/ArucoCalibration), inset far enough
    from the page edge to survive typical printer margins.
    """
    inset = MARGIN_MM + MARKER_SIZE_MM / 2
    return {
        0: (inset, inset),
        1: (PAGE_WIDTH_MM - inset, inset),
        2: (PAGE_WIDTH_MM - inset, PAGE_HEIGHT_MM - inset),
        3: (inset, PAGE_HEIGHT_MM - inset),
    }


def GenerateSheet(output_path: str) -> None:
    dictionary = cv2.aruco.getPredefinedDictionary(ARUCO_DICTIONARY)
    positions = MarkerPositions()
    half = MARKER_SIZE_MM / 2

    fig = plt.figure(
        figsize=(PAGE_WIDTH_MM / MM_PER_INCH, PAGE_HEIGHT_MM / MM_PER_INCH)
    )
    ax = fig.add_axes((0, 0, 1, 1))  # fill the whole page, no margins
    ax.set_xlim(0, PAGE_WIDTH_MM)
    ax.set_ylim(PAGE_HEIGHT_MM, 0)  # origin top-left, y increases downward
    ax.axis("off")

    for marker_id, (cx, cy) in positions.items():
        marker_image = cv2.aruco.generateImageMarker(
            dictionary, marker_id, MARKER_PIXELS
        )
        ax.imshow(
            marker_image,
            cmap="gray",
            vmin=0,
            vmax=255,
            extent=(cx - half, cx + half, cy + half, cy - half),
            interpolation="nearest",
        )
        ax.text(
            cx, cy - half - 2, f"ID {marker_id}", ha="center", va="bottom", fontsize=8
        )

    ax.text(
        PAGE_WIDTH_MM / 2,
        PAGE_HEIGHT_MM - 5,
        "Print at 100% scale (actual size, not “fit to page”)",
        ha="center",
        va="bottom",
        fontsize=7,
    )

    fig.savefig(output_path, format="pdf", dpi=300)
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
