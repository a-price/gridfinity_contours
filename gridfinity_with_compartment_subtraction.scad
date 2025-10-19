include <gridfinity-rebuilt-openscad-2.0.0/src/core/standard.scad>
use <gridfinity-rebuilt-openscad-2.0.0/src/core/gridfinity-rebuilt-utility.scad>
use <gridfinity-rebuilt-openscad-2.0.0/src/core/gridfinity-rebuilt-holes.scad>
use <gridfinity-rebuilt-openscad-2.0.0/src/core/bin.scad>

bin22 = new_bin(
    grid_size = [2, 2],
    height_mm = 7 * 3,
    include_lip = true
);

/*
render()
union() {
    bin_render_wall(bin22);
    bin_render_base(bin22);
    difference() {
        bin_render_infill(bin22);
        linear_extrude(){
         polygon(points=[[0,0],[10,0],[10,20],[5,15],[0,20]]);
        }
    }
}
*/

render()
difference() {
    bin_render(bin22);
    linear_extrude(){
         polygon(points=[[0,0],[10,0],[10,20],[5,15],[0,20]]);
        }
    }

//test = union(br, cube(1));
//render(test);