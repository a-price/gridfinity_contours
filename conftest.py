"""Fixtures and helpers shared across the test suite.

Everything here was previously copied into each test file that needed it -
`Rectangle` into nineteen of them, `qapp` into four, the quick parameter
budget into seven. The copies had not drifted, but two of them were the
kind that would: `SPOONS` is a list of paths, and `QuickParameters` is the
dial that trades how thoroughly the suite searches against how long it
takes. Neither should have to be found in seven places to be changed once.

Test files import from here under the local names they already used, so
the call sites are untouched:

    from conftest import QuickParameters as _quick, Rectangle as _rectangle
"""

from dataclasses import replace

import numpy as np
import pytest

from layout.parameters import LayoutParameters
from qt_utils.headless import UseOffscreenQt

# Set before any test module imports PyQt5, so a Qt test needs no display.
# Lives here rather than in each Qt test file because conftest is imported
# first, which is the only ordering that reliably works. Shared with the
# two demos that photograph windows - see qt_utils/headless.py for what
# the font half is for, which is not obvious and is Windows-only.
UseOffscreenQt()

# The three capture fixtures that make up the standard packing example -
# three separate photo sessions of three real spoons.
SPOONS = ["test_data/big_spoon.svg", "test_data/medium_spoon.svg", "test_data/small_spoon.svg"]


def Rectangle(width: float, height: float, x: float = 0.0, y: float = 0.0) -> np.ndarray:
    """An axis-aligned rectangular contour in millimeters.

    The workhorse test shape: its area, extent and clearances are all
    obvious by inspection, so a test that fails says something about the
    code rather than about the fixture.
    """
    return np.array([[x, y], [x + width, y], [x + width, y + height], [x, y + height]], dtype=np.float64)


def QuickParameters(**overrides) -> LayoutParameters:
    """Default parameters on a budget small enough for a test suite.

    The search is stochastic, so this is a real tradeoff rather than a
    formality: too small a budget and comfortably packable fixtures start
    failing intermittently, too large and the suite crawls. Files that need
    a different budget pass overrides rather than restating the baseline.

    `pocket_resolution` is the second dial, and it is the expensive one.
    Dilating an object is a fine raster whose cost is quadratic in the
    cell: at the 0.05mm default a single spoon takes 139ms against
    `BuildPart`'s own 9ms, and the suite builds parts in twenty-three
    files. At 0.2mm that is sixteen times less raster, and it costs the
    tests nothing they measure - a pocket 0.22mm oversize instead of
    0.07mm still packs, still verifies, and no fixture here asserts on a
    pocket outline finely enough to notice. Anything that does asks for
    the default back.
    """
    return replace(
        LayoutParameters(restarts=6, iterations=150, patience=25, pocket_resolution=0.2),
        **overrides,
    )


@pytest.fixture(scope="session")
def qapp():
    """One QApplication for the whole session.

    Session-scoped because Qt allows exactly one per process, and imported
    inside the fixture so that a run of the non-Qt tests never loads it.
    """
    from PyQt5.QtWidgets import QApplication

    return QApplication.instance() or QApplication([])
