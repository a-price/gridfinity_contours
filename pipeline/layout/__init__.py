"""Packing contours into Gridfinity bins.

    container   the bin interior, from the Gridfinity spec
    part        a contour and its signed distance field
    placement   a part positioned in a bin
    parameters  everything tunable, in one place
    energy      how badly an arrangement violates its clearances
    descent     moving parts along the forces an energy reports
    solver      a solved arrangement, and the search for one
    packer      choosing the bin size, and the bounds that rule one out
    spacing     evening out the gaps once a layout already fits
    loading     getting parts in, from SVGs or contours you already have
    preview     drawing a solved layout at true scale
    render      the same drawing, rasterized for the screen
    solid       the printable bin, as OpenSCAD
    verify      independent checks, sharing no code with the solver

Design: docs/layout.md. Build order: docs/layout_roadmap.md.
"""
