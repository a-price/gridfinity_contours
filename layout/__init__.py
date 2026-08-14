"""Packing contours into Gridfinity bins.

    container   the bin interior, from the Gridfinity spec
    raster      a millimeter polygon as a signed distance field
    pocket      an object's outline grown into the shape cut for it
    part        a pocket, its field, and the object it was cut for
    field       looking at the field a part is packed by
    placement   a part positioned in a bin
    parameters  everything tunable, in one place
    energy      how badly an arrangement violates its clearances
    descent     moving parts along the forces an energy reports
    orientation which pose each part starts the search at
    solver      a solved arrangement, and the search for one
    packer      choosing the bin size, and the bounds that rule one out
    grouping    partitioning parts across several bins
    drawer      which bins share a drawer, in whole grid cells
    plan        the whole stack in one call: parts to bins to drawers
    session     saving a floorplan and picking it up again
    floorplan   a whole drawer drawn at true scale, bins and objects
    spacing     evening out the gaps once a layout already fits
    loading     getting parts in, from SVGs or contours you already have
    preview     drawing a solved layout at true scale
    render      the same drawing, rasterized for the screen
    solid       the printable bin, as OpenSCAD
    verify      independent checks, sharing no code with the solver

Design: docs/layout.md. Build order: docs/layout_roadmap.md.
"""
