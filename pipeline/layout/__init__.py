"""Packing contours into Gridfinity bins.

    container   the bin interior, from the Gridfinity spec
    part        a contour and its signed distance field
    placement   a part positioned in a bin
    energy      how badly an arrangement violates its clearances
    solver      a solved arrangement, and the search for one
    packer      choosing the bin size, and the bounds that rule one out
    loading     getting parts in, from SVGs or contours you already have
    preview     drawing a solved layout at true scale
    verify      independent checks, sharing no code with the solver

Design: docs/layout.md. Build order: docs/layout_roadmap.md.
"""
