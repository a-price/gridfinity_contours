"""A solved arrangement, and (M3) the search that produces one.

Only the result type so far; the relaxation that drives the energy to zero
lands here next. See docs/layout_roadmap.md.
"""

from dataclasses import dataclass

import numpy as np

from pipeline.layout.container import DEFAULT_INTERIOR_INSET_MM, BuildContainer, Container
from pipeline.layout.placement import Placement


@dataclass(frozen=True)
class Layout:
    """A solved arrangement: the grid size chosen and where every part
    landed inside it.
    """

    grid: tuple[int, int]
    placements: dict[int, Placement]
    inset: float = DEFAULT_INTERIOR_INSET_MM

    @property
    def cells(self) -> int:
        return self.grid[0] * self.grid[1]

    def Interior(self) -> Container:
        """The bin interior this layout was packed into."""
        return BuildContainer(self.grid[0], self.grid[1], self.inset)

    def Envelope(self) -> np.ndarray:
        """That interior's boundary as a polygon."""
        return self.Interior().Polygon()
