"""The figures of ``tutorial/level_set.md``: the stencils of the two schemes of the demo.

Both come from ``demos/FiniteVolume/level_set_schemes.hpp``. The flux of
``make_level_set_convection`` has 4 cells, so ``FluxDefinition`` gives it the default
stencil ``line_stencil_from<dim, d, 4>(-1)``: the offsets -1 to 2 in the direction d,
read as ``phi[0]`` to ``phi[3]``. The scheme of ``make_level_set_reinitialization``
uses ``star_stencil<2, 2>()`` (``include/samurai/stencil.hpp``), whose center has the
index ``cfg::center_index`` = 2, and finds the cells at -2, -1, +1 and +2 in the
direction d with the formulas of its loop, which ``star_indices`` repeats.
"""

from ..draw import (
    HAIR,
    HEAVY,
    INK,
    INK2,
    RED,
    RWASH,
    WASH,
    bracket,
    line,
    patterns,
    rect,
    text,
)
from ..registry import DIAGRAM_WIDTH, Drawing, figure

# The default stencil of a flux on 4 cells in the x-direction, offsets [-1, 3).
FLUX_OFFSETS = (-1, 0, 1, 2)

# star_stencil<2, 2>(), in the order of stencil.hpp and of the comment of the demo.
STAR = (
    ("left2", (-2, 0)),
    ("left", (-1, 0)),
    ("center", (0, 0)),
    ("right", (1, 0)),
    ("right2", (2, 0)),
    ("bottom2", (0, -2)),
    ("bottom", (0, -1)),
    ("top", (0, 1)),
    ("top2", (0, 2)),
)
CENTER = 2  # cfg::center_index of StarStencilSchemeConfig, the radius


def star_indices(d):
    """Return the indices m2, m1, p1, p2 that the loop of the demo computes for direction d.

    They are the indices in ``STAR`` of the cells at -2, -1, +1 and +2 along d; the
    function checks it.
    """
    m2 = 0 if d == 0 else 4 * d + 1
    m1 = m2 + 1
    p1 = 3 if d == 0 else m2 + 2
    p2 = p1 + 1
    for index, step in zip((m2, m1, p1, p2), (-2, -1, 1, 2)):
        vector = [0, 0]
        vector[d] = step
        if STAR[index][1] != tuple(vector):
            raise ValueError(f"index {index} is not the cell {vector} of the star stencil")
    return m2, m1, p1, p2


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


def _code(x, y, s, size=11.5, fill=INK2):
    """Set ``s`` in the mono, centered on x."""
    return text(x, y, s, anchor="middle", size=size, fill=fill, cls="sm-fig-code")


def _cell(x, y, w, h, origin, p, band=0):
    """Draw a cell of ``w`` by ``h`` at (x, y), in red for the origin of the stencil.

    The origin is filled with the red wash and hatched only in two bands of width
    ``band`` along its sides, clear of its labels, so that they stay legible in both
    themes without a halo.
    """
    if not origin:
        return rect(x, y, w, h, fill=WASH)
    return (
        rect(x, y, w, h, fill=RWASH, stroke="none")
        + rect(x, y, band, h, fill=f"url(#{p}-r)", stroke="none")
        + rect(x + w - band, y, band, h, fill=f"url(#{p}-r)", stroke="none")
        + rect(x, y, w, h, stroke=RED, sw=HAIR)
    )


def _vector(v):
    """Write the vector ``v`` as stencil.hpp does, ``{-1,0}``."""
    return "{" + ",".join(str(k) for k in v) + "}"


def _minus(k):
    """Write ``k`` with a typographic minus, for the math labels."""
    return f"−{-k}" if k < 0 else f"+{k}"


@figure
def flux_stencil(p):
    """Draw the four cells of the flux in the x-direction, under their three indices.

    Figure of tutorial/level_set.md: the cell of the math (k - 1 to k + 2), the value
    in the lambda (``phi[0]`` to ``phi[3]``) and the offset of the stencil ({-1,0} to
    {2,0}); the face lies between ``phi[1]`` and ``phi[2]``. The brackets mark the
    values of phi the flux reads, besides the velocity, for each sign of u.
    """
    x0, cw, ch = 152, 112, 62
    face_y, math_y, top = 18, 50, 66
    face = x0 + 2 * cw
    g = ""
    for i, offset in enumerate(FLUX_OFFSETS):
        x = x0 + i * cw
        xm = x + cw / 2
        g += _cell(x, top, cw, ch, offset == 0, p, band=22)
        k = "k" if offset == 0 else f"k{_minus(offset)}"
        g += _label(xm, math_y, ["\u03d5", ("sub", f"j, {k}, h")])
        g += _code(xm, top + 27, f"phi[{i}]", size=15, fill=INK)
        g += _code(xm, top + 48, _vector((offset, 0)))
    g += line(face, face_y + 13, face, top + ch + 6, stroke=RED, sw=HEAVY)
    g += _label(face, face_y, ["F", ("sub", "j, k+1/2, h")], fill=RED)
    # the direction of the flux, beside the row
    xr, yr = x0 + 4 * cw + 14, top + ch / 2
    g += line(xr, yr, xr + 34, yr, stroke=INK2, extra=f'marker-end="url(#{p}-a)"')
    g += _label(xr + 17, yr - 8, ["x"], size=15, fill=INK2)
    keys = (
        (math_y, "notation"),
        (top + 27, "code"),
        (top + 48, "offset"),
    )
    for y, key in keys:
        g += text(x0 - 16, y, key, anchor="end", size=12, fill=INK)
    # the values of phi each sign of u reads: phi[0] to phi[2], or phi[1] to phi[3]
    rows = ((["u ≥ 0"], (-1, 2)), (["u &lt; 0"], (0, 3)))
    y = top + ch + 24
    for chunks, (a, b) in rows:
        xa, xb = x0 + (a + 1) * cw, x0 + (b + 1) * cw
        g += bracket(xa, xb, y, f"[{a}, {b})")
        g += _label(x0 - 16, y + 5, chunks, size=15, anchor="end")
        y += 44
    return Drawing(
        DIAGRAM_WIDTH,
        y - 12,
        "A row of four cells along x, each with three labels. Above the cells, the notation of"
        " the math: phi with indices j, k - 1, h, then k, k + 1 and k + 2. Inside, the value in"
        " the code, phi[0] to phi[3], and under it the offset of the stencil, {-1,0}, {0,0},"
        " {1,0} and {2,0}. The cell phi[1], offset {0,0}, is shaded red and hatched along its"
        " sides: it is the origin of the stencil. A red face between phi[1] and phi[2] carries"
        " the flux F with indices j, k + 1/2, h, and an arrow on the right points along x. Under"
        " the row, two brackets: for u at least 0, the offsets [-1, 2), the cells phi[0] to"
        " phi[2]; for u less than 0, the offsets [0, 3), the cells phi[1] to phi[3].",
        patterns(p),
        g,
    )


@figure
def star_stencil(p):
    """Draw the star stencil of radius 2 with the index of each cell in the scheme.

    Figure of tutorial/level_set.md: each cell holds its index in ``cells``, its name
    in the comment of the demo and its vector in ``star_stencil<2, 2>()``; the center
    (index 2) is shaded red and hatched along its sides. On the right, the indices the
    loop over the directions computes, from ``star_indices``.
    """
    c, x0, y0 = 76, 24, 10
    # the cells, the center last so that its red outline stays on top of its neighbors
    cells = [
        (x0 + (dx + 2) * c, y0 + (2 - dy) * c, k == CENTER) for k, (_, (dx, dy)) in enumerate(STAR)
    ]
    g = "".join(_cell(x, y, c, c, center, p, band=14) for x, y, center in cells if not center)
    g += "".join(_cell(x, y, c, c, center, p, band=14) for x, y, center in cells if center)
    for index, (name, (dx, dy)) in enumerate(STAR):
        x, y = cells[index][0], cells[index][1]
        g += _code(x + c / 2, y + 26, f"{index}", size=17, fill=INK)
        g += text(x + c / 2, y + 47, name, anchor="middle", size=10.5, fill=INK2)
        g += _code(x + c / 2, y + 66, _vector((dx, dy)), size=10.5)
    # the axes, in the empty corner at the bottom left
    ax, ay = x0 + 12, y0 + 5 * c - 14
    arrow = f'marker-end="url(#{p}-a)"'
    g += line(ax, ay, ax + 40, ay, stroke=INK2, extra=arrow)
    g += line(ax, ay, ax, ay - 40, stroke=INK2, extra=arrow)
    g += _label(ax + 50, ay + 5, ["x"], size=15, fill=INK2)
    g += _label(ax, ay - 50, ["y"], size=15, fill=INK2)
    # the indices of the loop over the directions
    tx, cols, ty, pitch = 450, 48, 172, 30
    heads = ("d", "m2", "m1", "p1", "p2")
    for j, head in enumerate(heads):
        g += _code(tx + j * cols, ty, head, size=13, fill=INK)
    g += line(tx - cols / 2, ty + 9, tx + (len(heads) - 0.5) * cols, ty + 9, stroke=INK, sw=HAIR)
    for d in range(2):
        y = ty + 9 + (d + 1) * pitch
        g += _code(tx, y, f"{d}", size=13, fill=INK)
        for j, index in enumerate(star_indices(d)):
            g += _code(tx + (j + 1) * cols, y, f"{index}", size=13)
    g += text(
        tx + 2 * cols, ty - 30, "indices read by the loop", anchor="middle", size=12, fill=INK
    )
    return Drawing(
        DIAGRAM_WIDTH,
        y0 + 5 * c + 4,
        "A star of nine cells: a center with two cells on each side along x and along y. Each"
        " cell holds its index in the stencil, its name and its vector. Along x, from left to"
        " right: 0 left2 {-2,0}, 1 left {-1,0}, 2 center {0,0}, shaded red and hatched along its"
        " sides, 3 right {1,0}, 4 right2 {2,0}. Along y, from bottom to top: 5 bottom2 {0,-2}, 6"
        " bottom {0,-1}, then the center, 7 top {0,1}, 8 top2 {0,2}. Arrows at the bottom left"
        " show x and y. On the right, a table of the indices read by the loop: for d = 0, m2 ="
        " 0, m1 = 1, p1 = 3, p2 = 4; for d = 1, m2 = 5, m1 = 6, p1 = 7, p2 = 8.",
        patterns(p),
        g,
    )
