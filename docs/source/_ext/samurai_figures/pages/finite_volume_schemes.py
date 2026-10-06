"""The figures of ``reference/finite_volume_schemes.md``: the faces and the stencils of a flux.

The figures follow ``include/samurai/interface.hpp`` and the flux-based schemes of
``include/samurai/schemes/fv/flux_based/``. At a level jump l / l + 1,
``for_each_interior_interface__level_jump_direction`` finds one face per fine cell
(``intersection(coarse, translate(fine, -direction)).on(level + 1)``) and reads the
stencil on level l + 1, its origin being a ghost inside the coarse cell; the face
measure is that of level l + 1 (``h_factor(h_lp1, h_l)``). The default stencil of
``FluxDefinition`` is ``line_stencil_from<dim, d, stencil_size>(-stencil_size / 2 + 1)``.
"""

from ..draw import (
    EMPTY,
    FIG,
    HAIR,
    HEAVY,
    INK,
    INK2,
    INK3,
    RED,
    WASH,
    cell_row,
    line,
    patterns,
    rect,
    text,
)
from ..registry import DIAGRAM_WIDTH, Drawing, figure


def _math(chunks, size):
    """Build serif italic markup from ``chunks``, a ``("sub", s)`` chunk set as a subscript.

    The other chunks are plain strings. The baseline comes back after each
    subscript, so that a chunk can follow it.
    """
    s, low, raised = "", 0.3 * size, False
    for chunk in chunks:
        if isinstance(chunk, tuple):
            s += f'<tspan dy="{low:g}" font-size="{0.72 * size:g}">{chunk[1]}</tspan>'
            raised = True
        elif raised:
            # dy moves the first glyph of its tspan: the chunk after a subscript carries it
            s += f'<tspan dy="{-low:g}">{chunk}</tspan>'
            raised = False
        else:
            s += chunk
    return s


def _label(x, y, chunks, size=17, fill=INK, anchor="middle"):
    """Set a math label at (x, y) from ``chunks``, see ``_math``."""
    return text(x, y, _math(chunks, size), anchor=anchor, size=size, fill=fill, cls="sm-fig-math")


def _measures(x, y, face, cell):
    """Write the measure of the face and that of the left cell, on each side of x."""
    return _label(x - 54, y, face, size=15, fill=INK2) + _label(x + 54, y, cell, size=15, fill=INK2)


def _flux_arrow(p, x1, x2, y):
    """Draw the flux across a face, a red arrow from x1 to x2 on the line y."""
    return line(x1, y, x2, y, stroke=RED, extra=f'marker-end="url(#{p}-ar)"')


@figure
def face_fluxes(p):
    """Draw the face between two cells of one level, then the faces of a level jump.

    Figure of reference/finite_volume_schemes.md. On the right, the coarse cell of
    level l faces two cells of level l + 1; each fine face has its own flux,
    computed between the fine cell and a ghost of level l + 1 inside the coarse
    cell, as ``for_each_interior_interface__level_jump_direction`` does.
    """
    s, top = 96, 34  # a cell of the same-level pair, 96 units wide
    xa = 64
    face = xa + s
    g = rect(xa, top, s, s, fill=WASH) + rect(face, top, s, s, fill=WASH)
    g += line(face, top - 8, face, top + s + 8, stroke=RED, sw=HEAVY)
    g += text(face + 6, top - 4, "F", size=17, fill=RED, cls="sm-fig-math")
    g += _flux_arrow(p, face - 30, face + 30, top + 30)
    g += _label(xa + s / 2, top + 70, ["V", ("sub", "L")])
    g += _label(face + s / 2, top + 70, ["V", ("sub", "R")])
    level_y, measure_y = 158, 184
    g += text(face, level_y, "level <tspan class='sm-fig-math'>l</tspan>", anchor="middle", size=12)
    g += _measures(face, measure_y, ["|F| = h"], ["|V", ("sub", "L"), "| = h²"])

    # the level jump: a coarse cell of level l, two fine cells of level l + 1
    g += line(330, 8, 330, measure_y + 6, stroke=FIG, sw=EMPTY)
    f, ctop = 56, 26  # a fine cell is 56 units wide, the coarse cell 112
    xc = 420
    jump = xc + 2 * f
    g += rect(xc, ctop, 2 * f, 2 * f, fill=WASH)
    for k in range(2):
        y = ctop + k * f
        g += rect(jump, y, f, f, fill=WASH)
    # the ghosts of level l + 1 inside the coarse cell, along the face
    g += line(jump - f, ctop, jump - f, ctop + 2 * f, stroke=INK3, dash="3 3")
    g += line(jump - f, ctop + f, jump, ctop + f, stroke=INK3, dash="3 3")
    g += rect(xc, ctop, 2 * f, 2 * f, sw=HEAVY)
    for k in range(2):
        y = ctop + k * f
        # one face per fine cell: the red face stops short of the edge between them
        g += line(jump, y + (3 if k else -8), jump, y + f + (8 if k else -3), stroke=RED, sw=HEAVY)
        g += _flux_arrow(p, jump - 26, jump + 26, y + 18)
        g += _label(jump - f / 2, y + 45, ["G"], size=15, fill=INK2)
        g += _label(jump + f / 2, y + 45, ["V", ("sub", "R")], size=15)
    g += _label(xc + f / 2, ctop + f + 6, ["V", ("sub", "L")])
    g += text(
        xc + f, level_y, "level <tspan class='sm-fig-math'>l</tspan>", anchor="middle", size=12
    )
    g += text(
        jump + f / 2,
        level_y,
        "level <tspan class='sm-fig-math'>l</tspan> + 1",
        anchor="middle",
        size=12,
    )
    g += _measures(
        jump,
        measure_y,
        ["|F| = h", ("sub", "l+1")],
        ["|V", ("sub", "L"), "| = h", ("sub", "l"), "²"],
    )
    return Drawing(
        DIAGRAM_WIDTH,
        measure_y + 12,
        "Two parts. Left, two square cells of the same level l, V_L and V_R, share a face F,"
        " drawn in red, with a red arrow across it from V_L to V_R: |F| = h and |V_L| = h"
        " squared. Right, a level jump: a coarse cell V_L of level l, twice as wide, faces two"
        " cells V_R of level l + 1 stacked on its right. Its right half is cut by dashed lines"
        " into two ghosts G of level l + 1, one beside each fine cell. The edge between the coarse"
        " cell and the fine cells is two red faces, one per fine cell, each with its own red"
        " arrow from the ghost G to the fine cell V_R: |F| = h at level l + 1, |V_L| = h at level"
        " l, squared.",
        patterns(p),
        g,
    )


def _offset_cell(x, y, c, label, origin, p):
    """Draw a cell of side ``c`` at (x, y) with its direction vector, hatched for the origin."""
    fill, stroke, sw = (f"url(#{p}-r)", RED, HAIR) if origin else (WASH, INK, 1)
    return rect(x, y, c, c, fill=fill, stroke=stroke, sw=sw) + text(
        x + c / 2,
        y + c / 2 + 4,
        label,
        anchor="middle",
        size=11.5,
        fill=INK,
        cls="sm-fig-code",
        knock=origin,
    )


@figure
def stencil_offsets(p):
    """Draw the stencils {{-1,0}, {0,0}, {1,0}, {2,0}} and {{0,-1}, {0,0}, {0,1}, {0,2}}.

    Figure of reference/finite_volume_schemes.md: each cell holds the direction
    vector that captures it, the origin cell is hatched, the face is red.
    """
    c = 52
    arrow = f'marker-end="url(#{p}-a)"'
    # the x-direction: a row of four cells, the face between {0,0} and {1,0}
    xr, yr = 120, 92
    g = ""
    for k, label in enumerate(("{-1,0}", "{0,0}", "{1,0}", "{2,0}")):
        g += _offset_cell(xr + k * c, yr, c, label, k == 1, p)
    face = xr + 2 * c
    g += line(face, yr - 8, face, yr + c + 8, stroke=RED, sw=HEAVY)
    g += line(face - c / 2, yr - 18, face + c / 2, yr - 18, stroke=INK2, extra=arrow)
    g += text(face - c / 2, yr + c + 22, "origin", anchor="middle", size=11.5, fill=RED)
    # the y-direction: a column of four cells, the face between {0,0} and {0,1}
    xc, yc = 490, 14
    for k, label in enumerate(("{0,2}", "{0,1}", "{0,0}", "{0,-1}")):
        g += _offset_cell(xc, yc + k * c, c, label, k == 2, p)
    face_y = yc + 2 * c
    g += line(xc - 8, face_y, xc + c + 8, face_y, stroke=RED, sw=HEAVY)
    g += line(xc + c + 18, face_y + c / 2, xc + c + 18, face_y - c / 2, stroke=INK2, extra=arrow)
    g += text(xc - 12, face_y + c / 2 + 4, "origin", anchor="end", size=11.5, fill=RED)
    caption_y = yc + 4 * c + 26
    g += text(xr + 2 * c, caption_y, "x-direction", anchor="middle", size=12, fill=INK)
    g += text(xc + c / 2, caption_y, "y-direction", anchor="middle", size=12, fill=INK)
    return Drawing(
        DIAGRAM_WIDTH,
        caption_y + 10,
        "Two stencils of four cells. Left, in the x-direction, a row of cells labeled {-1,0},"
        " {0,0}, {1,0} and {2,0}; the origin cell {0,0} is hatched in red, the red face lies"
        " between {0,0} and {1,0}, and an arrow above it points right. Right, in the y-direction,"
        " a column of cells labeled from bottom to top {0,-1}, {0,0}, {0,1} and {0,2}; the origin"
        " cell {0,0} is hatched, the red face lies between {0,0} and {0,1}, and an arrow beside"
        " it points up.",
        patterns(p),
        g,
    )


DEFAULT_SIZES = (2, 3, 4, 6)


def default_stencil(size):
    """Return the offsets [from, from + size) of the default stencil of ``size`` cells.

    ``FluxDefinition`` starts it at ``-size / 2 + 1``, the division of C++ that
    rounds toward zero.
    """
    start = -(size // 2) + 1
    return start, start + size


@figure
def default_stencils(p):
    """Draw the default stencils of sizes 2, 3, 4 and 6 against the face.

    Figure of reference/finite_volume_schemes.md, from ``FluxDefinition``: the
    offsets of each stencil, its origin cell hatched, and the stencil radius
    counted from the face.
    """
    x0, cw, ch, pitch, top = 167, 56, 26, 64, 30
    first, last = -2, 4  # the offsets drawn, [-2, 4)
    face = x0 + (0 - first + 1) * cw
    g = text(face - cw / 2, top - 10, "origin", anchor="middle", size=11.5, fill=RED)
    for r, size in enumerate(DEFAULT_SIZES):
        y = top + r * pitch
        a, b = default_stencil(size)
        g += cell_row(x0, y, cw, last, [(a, b)], ch=ch, start=first, label_size=11)
        g += rect(x0 - first * cw, y, cw, ch, fill=f"url(#{p}-r)", stroke=RED, sw=HAIR)
        g += line(face, y - 4, face, y + ch + 4, stroke=RED, sw=HEAVY)
        g += text(x0 - 16, y + 18, f"size {size}", anchor="end", size=12, fill=INK)
        radius = max(-a + 1, b - 1)  # the cells on each side of the face
        g += text(x0 + (last - first) * cw + 16, y + 18, f"radius {radius}", size=11.5, fill=INK2)
    axis = top + len(DEFAULT_SIZES) * pitch + 4
    g += line(x0, axis, x0 + (last - first) * cw, axis)
    for k in range(first, last):
        xm = x0 + (k - first + 0.5) * cw
        g += line(xm, axis, xm, axis + 5)
        g += text(xm, axis + 18, f"{k}", anchor="middle", size=11.5, cls="sm-fig-code")
    g += line(face, axis - 4, face, axis + 4, stroke=RED, sw=HEAVY)
    g += text(x0 - 16, axis + 18, "offset", anchor="end", size=11.5)
    return Drawing(
        DIAGRAM_WIDTH,
        axis + 26,
        "Four rows of cells over the offsets -2 to 3, a red face between the offsets 0 and 1 in"
        " every row, the origin cell at offset 0 hatched in red. Size 2 reads the offsets [0, 2),"
        " radius 1; size 3 reads [0, 3), radius 2; size 4 reads [-1, 3), radius 2; size 6 reads"
        " [-2, 4), radius 3.",
        patterns(p),
        g,
    )
