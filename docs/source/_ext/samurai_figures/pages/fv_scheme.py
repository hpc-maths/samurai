"""The figures of ``howto/fv_scheme.md``: the indices of a wider stencil.

``samurai::line_stencil<dim, d>(-1, 0, 1, 2)`` (``include/samurai/stencil.hpp``)
puts its offsets in the stencil in the order given, and the flux function receives
the values in that order: ``u[0]`` to ``u[3]`` are the cells at -1, 0, 1 and 2 from
the origin cell. The default stencil of size 2 is ``line_stencil_from<dim, d, 2>(0)``,
the cells at 0 and 1, read as ``u[0]`` and ``u[1]``. On one level,
``flux_based_scheme__lin_hom.hpp`` gives the left cell of the face
``h_factor(h, h) * flux`` and the right cell its opposite, ``h_factor`` being the face
measure over the cell measure, 1 / h; ``flux_based_scheme__nonlin.hpp`` does the same
with ``fluxes[1] = -fluxes[0]`` from ``FluxDefinition``.
"""

from ..draw import (
    HAIR,
    HEAVY,
    INK,
    INK2,
    RED,
    RWASH,
    WASH,
    line,
    patterns,
    rect,
    text,
)
from ..registry import DIAGRAM_WIDTH, Drawing, figure
from .finite_volume_schemes import default_stencil

# The stencil of the page, line_stencil<dim, d>(-1, 0, 1, 2), in its order.
WIDE = (-1, 0, 1, 2)

# The width of the hatched bands on each side of the origin cell, clear of its labels.
ORIGIN_BAND = 22


def _code(x, y, s, size=11.5, fill=INK2):
    """Set ``s`` in the mono, centered on x."""
    return text(x, y, s, anchor="middle", size=size, fill=fill, cls="sm-fig-code")


def _cell(x, y, w, h, origin, p):
    """Draw a cell of ``w`` by ``h`` at (x, y), hatched in red for the origin of the stencil.

    The hatching of the origin fills two bands along its sides and leaves its middle
    in the red wash, so that its labels stay legible in both themes without a halo.
    """
    if not origin:
        return rect(x, y, w, h, fill=WASH)
    band = ORIGIN_BAND
    return (
        rect(x, y, w, h, fill=RWASH, stroke="none")
        + rect(x, y, band, h, fill=f"url(#{p}-r)", stroke="none")
        + rect(x + w - band, y, band, h, fill=f"url(#{p}-r)", stroke="none")
        + rect(x, y, w, h, stroke=RED, sw=HAIR)
    )


def _row(x0, y, cw, ch, offsets, p):
    """Draw the cells of ``offsets`` on the row y, each with its value and its offset.

    The cell at offset k sits at x0 + (k + 1) cw, so that the offset -1 starts at x0.
    """
    g = ""
    for i, offset in enumerate(offsets):
        x = x0 + (offset + 1) * cw
        origin = offset == 0
        g += _cell(x, y, cw, ch, origin, p)
        g += _code(x + cw / 2, y + 25, f"u[{i}]", size=15, fill=INK)
        g += _code(x + cw / 2, y + 44, "{" + f"{offset},0" + "}")
    return g


def _sign(x, y, sign):
    """Write the term ``sign`` flux / h the face adds to the scheme in a cell, centered on x."""
    return text(x, y, f"{sign} flux / h", anchor="middle", size=16, fill=INK, cls="sm-fig-math")


@figure
def wide_stencil(p):
    """Draw the stencil ``line_stencil(-1, 0, 1, 2)`` over the default stencil of size 2.

    Figure of howto/fv_scheme.md: each cell holds the value the flux function reads
    (``u[0]`` to ``u[3]``) and its offset; the origin cell is red, the face is
    red, and the two cells of the face carry the term the face adds to the value of the
    scheme there, the divergence of the flux.
    """
    x0, cw, ch = 184, 104, 56
    top, gap = 66, 34
    face = x0 + 2 * cw
    inner = default_stencil(2)
    if inner != (0, 2) or WIDE[1:3] != tuple(range(*inner)):
        raise ValueError(f"the default stencil of size 2 is not the inner pair: {inner}")
    rows = (
        ("size 4", "line_stencil(-1, 0, 1, 2)", "sm-fig-code", WIDE),
        ("size 2", "default stencil", "sm-fig-label", tuple(range(*inner))),
    )
    g = ""
    for r, (size, how, face_cls, offsets) in enumerate(rows):
        y = top + r * (ch + gap)
        g += _row(x0, y, cw, ch, offsets, p)
        g += text(x0 - 16, y + 24, size, anchor="end", size=12, fill=INK)
        g += text(x0 - 16, y + 41, how, anchor="end", size=10.5, fill=INK2, cls=face_cls)
    bottom = top + 2 * ch + gap
    # the face, through both rows, and the flux across it
    g += line(face, top - 36, face, bottom + 6, stroke=RED, sw=HEAVY)
    g += line(
        face - 26, top - 12, face + 26, top - 12, stroke=RED, extra=f'marker-end="url(#{p}-ar)"'
    )
    g += _sign(face - cw / 2, top - 26, "+")
    g += _sign(face + cw / 2, top - 26, "−")
    g += text(x0 - 16, top - 26, "scheme value", anchor="end", size=12, fill=INK)
    # the direction, beside the wide row
    xr, yr = x0 + 4 * cw + 14, top + ch / 2
    g += line(xr, yr, xr + 34, yr, stroke=INK2, extra=f'marker-end="url(#{p}-a)"')
    g += text(xr + 17, yr - 8, "x", anchor="middle", size=15, fill=INK2, cls="sm-fig-math")
    return Drawing(
        DIAGRAM_WIDTH,
        bottom + 12,
        "Two rows of cells along x around a red face. The upper row, size 4, line_stencil(-1, 0, "
        "1, 2), has four cells u[0], u[1], u[2] and u[3], at the offsets {-1,0}, {0,0}, {1,0} and"
        " {2,0}. The lower row, the default stencil of size 2, has two cells under u[1] and u[2] "
        "of the upper row, named u[0] and u[1], at the offsets {0,0} and {1,0}. In both rows the "
        "origin cell {0,0} is shaded red, hatched along its sides, and the red face lies between "
        "it and the cell at {1,0}. Above the face, a red arrow points along x, the direction of a"
        " positive flux. Above the cells, labeled scheme value: the face adds plus flux over h to"
        " the scheme in the cell on its left and minus flux over h in the cell on its right.",
        patterns(p),
        g,
    )
