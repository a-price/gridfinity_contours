include <gridfinity-rebuilt-openscad-2.0.0/src/core/standard.scad>
use <gridfinity-rebuilt-openscad-2.0.0/src/core/gridfinity-rebuilt-utility.scad>
use <gridfinity-rebuilt-openscad-2.0.0/src/core/gridfinity-rebuilt-holes.scad>
use <gridfinity-rebuilt-openscad-2.0.0/src/core/bin.scad>

bin22 = new_bin(
    grid_size = [2, 2],
    height_mm = 7 * 3,
    include_lip = true
);

bin_render(bin22);