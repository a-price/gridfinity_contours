import matplotlib.pyplot as plt
import numpy as np

from pipeline.svg_writer import AlignContoursToPca

MM_PER_INCH = 25.4


def WritePdf(path: str, contours: dict[int, np.ndarray]) -> None:
    """Writes `contours` to a PDF at the same PCA-aligned scale as
    WriteSvg (1 unit = 1mm). A PDF's page size is unambiguous, unlike an
    SVG's - not every viewer or print path honors an SVG's embedded
    physical units, so this is the reliable thing to print at "actual
    size" instead.
    """
    aligned, width, height = AlignContoursToPca(contours)

    fig = plt.figure(figsize=(width / MM_PER_INCH, height / MM_PER_INCH))
    ax = fig.add_axes((0, 0, 1, 1))  # fill the whole page, no margins
    ax.set_xlim(0, width)
    ax.set_ylim(height, 0)  # match the SVG's origin: top-left, y down
    ax.axis("off")

    for points in aligned.values():
        closed = np.vstack([points, points[:1]])
        ax.plot(closed[:, 0], closed[:, 1], "-", color="black", linewidth=0.5)

    fig.savefig(path, format="pdf")
    plt.close(fig)
