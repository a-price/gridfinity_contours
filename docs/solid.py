"""The original single-contour bin generator.

Superseded by [layout/solid.py](../layout/solid.py), which
takes a whole `Layout` and cuts a pocket per part.

This one differences against `bin_render(...)` from the outside and then
unions `bin_render_base` back on top. That is sound - the outer union is
what restores the base the difference removed - but it leaves the pocket
depth implicit, since a bare `linear_extrude()` cuts 100mm upward from
z=0 and the floor ends up wherever the base happens to start. The newer
generator passes cutouts as children of `bin_render`, which is the
mechanism the library provides, and states the depth.

Kept as the worked example of that older technique rather than for use:
for anything with more than one object in it, `layout_cli.py` takes a
whole set of contours and writes the `.scad` for you. Paste a contour into
the `points` array at the bottom and run it:

    .venv/bin/python3 docs/solid.py     # writes test.scad into the cwd

**Run it from the repository root, and render it there too.** The `.scad`
it emits includes the vendored library by *relative* path, and OpenSCAD
resolves those against the file's own directory - so a `test.scad`
written anywhere else renders an empty object, and says so only as a
warning it then exits 0 on. `layout.solid` avoids this by baking an
absolute `LIBRARY_PATH`; this one predates that and is left as it was.
"""

from typing import Sequence
import math
import numpy as np


def generate_gridfinity_scad(box, points: Sequence):
    scad_header = """
include <gridfinity-rebuilt-openscad/src/core/standard.scad>
use <gridfinity-rebuilt-openscad/src/core/gridfinity-rebuilt-utility.scad>
use <gridfinity-rebuilt-openscad/src/core/gridfinity-rebuilt-holes.scad>
use <gridfinity-rebuilt-openscad/src/core/bin.scad>
"""

    def to_gf_units(x):
        return math.ceil(x / 42.0)

    grid_x = to_gf_units(box[0][1] - box[0][0])
    grid_y = to_gf_units(box[1][1] - box[1][0])

    center = [(box[0][0] + box[0][1]) / 2.0, (box[1][0] + box[1][1]) / 2.0]

    bin = f"""
binxx = new_bin(
    grid_size = [{grid_x}, {grid_y}],
    height_mm = 7 * 3,
    include_lip = true
);
"""

    render = f"""
render()
union() {{
    bin_render_base(binxx);
    difference() {{
        bin_render(binxx);
        linear_extrude() {{
            offset(r=1) {{
                polygon(points=[{",".join([f"[{p[0]-center[0]},{p[1]-center[1]}]" for p in points])}]);
            }}
        }}
    }}
}}
"""
    return scad_header + bin + render


if __name__ == "__main__":
    # points = np.array([[0,0],[10,0],[10,20],[5,15],[0,20]])
    points = (
        np.array(
            [
                [0.00, 333.68],
                [12.48, 400.55],
                [45.02, 446.69],
                [87.84, 474.23],
                [151.91, 481.80],
                [585.40, 423.44],
                [1040.05, 387.09],
                [1355.85, 375.79],
                [1772.42, 385.58],
                [1895.75, 406.88],
                [2006.98, 443.10],
                [2217.86, 575.34],
                [2321.99, 621.28],
                [2438.78, 650.15],
                [2558.59, 659.93],
                [2755.81, 630.39],
                [2905.97, 553.77],
                [2964.82, 494.93],
                [2995.20, 446.38],
                [3020.83, 371.22],
                [3023.56, 322.53],
                [2994.38, 207.09],
                [2949.78, 143.10],
                [2894.26, 94.79],
                [2749.14, 25.01],
                [2651.51, 4.26],
                [2553.37, 0.00],
                [2335.83, 34.98],
                [2232.80, 78.85],
                [2012.28, 217.84],
                [1917.19, 255.89],
                [1788.75, 281.75],
                [1424.77, 297.37],
                [944.89, 278.83],
                [748.44, 260.11],
                [713.69, 269.70],
                [626.42, 258.99],
                [610.11, 245.13],
                [143.54, 188.90],
                [89.37, 198.62],
                [44.33, 226.83],
                [14.28, 272.24],
            ]
        )
        / 30.0
    )

    box = [
        [min(points[:, 0]), max(points[:, 0])],
        [min(points[:, 1]), max(points[:, 1])],
    ]
    with open("test.scad", "w") as f:
        f.write(generate_gridfinity_scad(box, points.tolist()))
