"""The file formats this project writes.

    svg_writer   contours as SVG, at true millimetre scale
    pdf_writer   the same shapes as a printable PDF, one or many pages
    contour_io   the JSON dump that carries contours between sessions
    gif_writer   frames as an animation, for the documentation
    scad_writer  placed OpenSCAD modules as one program, and a mesh render
    json_writer  the compact JSON every writer above (and layout/session.py,
                 layout/plan.py) hands its payload to

Every module here turns this project's shapes into somebody else's format
and does nothing else: no Qt, no solver, no capture. Capture and layout
both import these; these import neither, and nothing else of this
project's but `geometry`.

`contour_io` is the load-bearing one. Rectification puts every contour in
real millimetres, so a dump written by one capture session composes with
one written by another - which is the only reason contours from unrelated
photographs can be packed into the same bin. architecture.md calls that
frame the project's universal currency; this is where it is written down.

Grouped by what they are rather than by who imports them today. That is
why `gif_writer` is here despite only `demos/` using it: it writes a file
format, the same as its neighbours, and a package defined by its current
fan-in would have to be re-sorted every time a caller moved.
"""
