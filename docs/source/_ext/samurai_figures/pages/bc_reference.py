"""The figures of ``reference/bc.md``: the cells a boundary condition reads and writes.

The figures follow ``include/samurai/bc/``. ``DirichletImpl<order>`` reads and writes
the cells of ``line_stencil<dim, 0, 2 * order>()``, the offsets
``[-order + 1, order + 1)`` from the boundary cell on the right boundary: ``order``
cells inside, then ``order`` ghosts. It places the cell values at the cell centers
and the value of the condition at the boundary face, and writes the polynomial
through them at the centers of the ghosts.

``update_further_ghosts_by_polynomial_extrapolation`` (``bc/apply_field_bc.hpp``)
fills the ghost layers beyond those of the conditions, one at a time: the layer k
uses ``PolynomialExtrapolation<Field, min(2k, 6)>`` on the stencil that ends on the
ghost of the layer. ``update_outer_corners_by_polynomial_extrapolation`` fills the
corner blocks: the diagonal ghost of the layer k takes the value of the inner
diagonal cell at the offset 1 - k from the corner cell, and the other ghosts of the
block copy the diagonal ghost that has the same offset along the first axis.

The numbers checked below come from a 1D and a 2D program on a uniform mesh of level
3 on [0, 1)^dim, which sets the field, attaches the condition and calls
``samurai::update_ghost_mr``.
"""

from html import escape

from ..draw import (
    EMPTY,
    FIG,
    HAIR,
    HEAVY,
    INK,
    INK2,
    INK3,
    PAPER,
    RED,
    RULE,
    RWASH,
    WASH,
    bracket,
    circle,
    line,
    patterns,
    rect,
    text,
)
from ..plots import Frame, curve, lagrange
from ..registry import DIAGRAM_WIDTH, Drawing, figure

GHOST_DASH = "3 3"

# --- numbers --------------------------------------------------------------------------


def dirichlet2_ghosts(u_m1, u_0, v):
    """Return the two ghosts of ``Dirichlet<2>``, as ``DirichletImpl<2>`` writes them.

    ``u_m1`` and ``u_0`` are the cells at the offsets -1 and 0, ``v`` the value of the
    condition. The coefficients are those of ``bc/dirichlet.hpp``.
    """
    return 8 / 3 * v + u_m1 / 3 - 2 * u_0, 8 * v + 2 * u_m1 - 9 * u_0


def extrapolate(values):
    """Return the value ``PolynomialExtrapolation`` writes from the stencil ``values``.

    ``values`` are the cells of the stencil but the last one, which is the ghost to
    fill; the coefficients are those of ``bc/polynomial_extrapolation.hpp``.
    """
    coefficients = {1: (1,), 3: (1, -3, 3), 5: (1, -5, 10, -10, 5)}[len(values)]
    return sum(c * u for c, u in zip(coefficients, values))


def extrapolation_stencil(layer):
    """Return the offsets [a, b) of the stencil that fills the ghost of ``layer``.

    It ends on the ghost and holds ``min(2 * layer, 6)`` cells, as in
    ``update_further_ghosts_by_polynomial_extrapolation``.
    """
    size = min(2 * layer, 6)
    return layer - size + 1, layer + 1


# Printed by the 1D program: u(x) = x^2 at the cell centers of the level 3, Dirichlet<1>
# with the value 1, ghost width 3. The offsets -2 to 3 from the last cell, index 7.
PROBE_DIRICHLET1 = {
    -2: 0.472656,
    -1: 0.660156,
    0: 0.878906,
    1: 1.121094,
    2: 1.386719,
    3: 1.683594,
}

# Printed by the 2D program: u = 10 i + j on the cells of the level 3, Dirichlet<1>
# with the value 0, ghost width 3. The corner block of the top-right corner, (i, j) in
# [8, 11) x [8, 11), the corner cell being (7, 7).
PROBE_CORNER = {
    (8, 8): 77,
    (8, 9): 77,
    (8, 10): 77,
    (9, 8): 66,
    (9, 9): 66,
    (9, 10): 66,
    (10, 8): 55,
    (10, 9): 55,
    (10, 10): 55,
}


def check_layers():
    """Recompute the ghosts of ``PROBE_DIRICHLET1`` and raise if one differs."""
    u = {k: v for k, v in PROBE_DIRICHLET1.items() if k <= 0}
    u[1] = 2 * 1.0 - u[0]  # DirichletImpl<1>: 2v - u_0
    for layer in (2, 3):
        a, b = extrapolation_stencil(layer)
        u[layer] = extrapolate([u[k] for k in range(a, b - 1)])
    for k, value in PROBE_DIRICHLET1.items():
        if abs(u[k] - value) > 1e-5:
            raise ValueError(f"ghost {k}: {u[k]} computed, {value} printed by the program")


def corner_values(corner, width):
    """Return the corner block of the top-right corner of ``u = 10 i + j``.

    ``corner`` is the index of the corner cell on both axes, ``width`` the ghost width.
    The diagonal ghost of the layer k mirrors the inner diagonal cell at the offset
    1 - k; the others copy the diagonal ghost with the same i.
    """
    block = {}
    for di in range(1, width + 1):
        mirrored = corner + 1 - di
        for dj in range(1, width + 1):
            block[(corner + di, corner + dj)] = 10 * mirrored + mirrored
    if block != PROBE_CORNER:
        raise ValueError(f"corner block {block} differs from the program output {PROBE_CORNER}")
    return block


# --- drawing --------------------------------------------------------------------------


def _math(x, y, s, sub="", size=16, fill=INK, anchor="middle", knock=True):
    """Set the serif italic name ``s`` at (x, y), with the subscript ``sub``."""
    low = 0.3 * size
    body = s + (f'<tspan dy="{low:g}" font-size="{0.72 * size:g}">{sub}</tspan>' if sub else "")
    return text(x, y, body, anchor=anchor, size=size, fill=fill, cls="sm-fig-math", knock=knock)


# The style of each role of a cell: its fill, its outline, the weight and the dash of
# the outline, and the priority of the outline on an edge two cells share.
ROLES = {
    "read": (WASH, INK, RULE, "", 2),
    "ghost-read": (WASH, INK, RULE, GHOST_DASH, 2),
    "write": (RWASH, RED, RULE, GHOST_DASH, 3),
    "write-hatch": (None, RED, RULE, GHOST_DASH, 3),
    "inner": ("none", FIG, EMPTY, "", 1),
    "ghost": ("none", FIG, EMPTY, GHOST_DASH, 0),
}


def _grid(x0, y0, cw, ch, cells, p):
    """Draw ``cells``, a dict from (column, row) to a role of ``ROLES``, rows downward.

    ``"read"`` is an inner cell a stencil reads, ``"ghost-read"`` a ghost it reads,
    ``"write"`` a ghost it writes (``"write-hatch"`` hatched), ``"inner"`` and
    ``"ghost"`` the cells outside the stencil.
    Ghosts are dashed. Each edge is drawn once, in the style of the cell of higher
    priority on its two sides, so that the dashes of two neighbours never overlap.
    """
    g, edges = "", {}
    for (i, j), role in cells.items():
        fill = ROLES[role][0] or f"url(#{p}-r)"
        if fill != "none":
            g += rect(x0 + i * cw, y0 + j * ch, cw, ch, fill=fill, stroke="none")
        for edge in (
            (i, j, i + 1, j),
            (i, j + 1, i + 1, j + 1),
            (i, j, i, j + 1),
            (i + 1, j, i + 1, j + 1),
        ):
            if edge not in edges or ROLES[role][4] > ROLES[edges[edge]][4]:
                edges[edge] = role
    for (i1, j1, i2, j2), role in edges.items():
        _, stroke, sw, dash, _ = ROLES[role]
        g += line(
            x0 + i1 * cw, y0 + j1 * ch, x0 + i2 * cw, y0 + j2 * ch, stroke=stroke, sw=sw, dash=dash
        )
    return g


def _offset_axis(x0, cw, first, last, y, label=True):
    """Draw the offsets ``first`` to ``last - 1`` under a row, in the mono, and the face tick."""
    g = line(x0, y, x0 + (last - first) * cw, y)
    for k in range(first, last):
        xm = x0 + (k - first + 0.5) * cw
        g += line(xm, y, xm, y + 5)
        g += text(xm, y + 21, f"{k}", anchor="middle", size=11.5, cls="sm-fig-code")
    face = x0 + (1 - first) * cw
    g += line(face, y - 4, face, y + 4, stroke=RED, sw=HEAVY)
    if label:
        g += text(x0 - 16, y + 21, "offset", anchor="end", size=11.5)
    return g


@figure
def dirichlet_order_2(p):
    """Draw ``Dirichlet<2>`` on the right boundary.

    Figure of reference/bc.md. The two inner cells at the offsets -1 and 0 and the
    value v at the boundary face define the polynomial p of degree 2; the ghosts at
    the offsets 1 and 2 receive p at their centers.
    """
    u_m1, u_0, v = 1.0, 1.6, 1.7
    g1, g2 = dirichlet2_ghosts(u_m1, u_0, v)
    poly = lagrange([(-1, u_m1), (0, u_0), (0.5, v)])
    if abs(poly(1) - g1) > 1e-12 or abs(poly(2) - g2) > 1e-12:
        raise ValueError("the Dirichlet<2> coefficients do not match the polynomial")

    first, last, cw, ch = -2, 3, 84, 30
    x0 = (DIAGRAM_WIDTH - (last - first) * cw) / 2
    fr = Frame(x0, 34, (last - first) * cw, 120, xlim=(first - 0.5, last - 0.5), ylim=(1.0, 1.8))
    row = fr.bottom + 26
    face = fr.X(0.5)

    roles = {
        (k - first, 0): "inner" if k < -1 else ("read" if k <= 0 else "write")
        for k in range(first, last)
    }
    g = _grid(x0, row, cw, ch, roles, p)
    # the centers: guides from the row up to the values
    for k, value in ((-1, u_m1), (0, u_0), (1, g1), (2, g2)):
        xc = fr.X(k)
        g += line(xc, row, xc, fr.Y(value) + 4, stroke=INK3, sw=HAIR, dash="2 3")
        g += circle(xc, row + ch / 2, 1.8, fill=RED if k > 0 else INK)
    g += line(face, fr.y - 14, face, row + ch + 6, stroke=RED, sw=HEAVY)
    g += text(face, fr.y - 23, "boundary face", anchor="middle", size=11.5, fill=RED)
    g += curve(fr, poly, lo=-1, hi=2)
    g += circle(fr.X(-1), fr.Y(u_m1), 3, fill=INK)
    g += circle(fr.X(0), fr.Y(u_0), 3, fill=INK)
    g += circle(face, fr.Y(v), 3.4, fill=RED)
    for k, value in ((1, g1), (2, g2)):
        g += circle(fr.X(k), fr.Y(value), 3, fill=PAPER, stroke=RED, sw=RULE)
    g += _math(fr.X(-1) - 12, fr.Y(u_m1) + 4, "u", "−1", anchor="end")
    g += _math(fr.X(0) - 13, fr.Y(u_0) - 8, "u", "0", anchor="end")
    g += _math(face - 8, fr.Y(v) - 10, "v", fill=RED, anchor="end")
    g += _math(fr.X(2) + 12, fr.Y(g2) + 5, "p", anchor="start")

    # the stencil of the condition and the offsets
    a, b = -1, 3
    g += bracket(
        x0 + (a - first) * cw,
        x0 + (b - first) * cw,
        row + ch + 10,
        f"[{a}, {b})",
        size=11,
        label_dy=17,
    )
    g += text(fr.X(-0.5), row + ch + 27, "inner cells", anchor="middle", size=11.5, fill=INK)
    g += text(fr.X(1.5), row + ch + 27, "ghosts", anchor="middle", size=11.5, fill=RED)
    axis = row + ch + 46
    g += _offset_axis(x0, cw, first, last, axis)
    return Drawing(
        DIAGRAM_WIDTH,
        axis + 29,
        "A row of five cells over the offsets -2 to 2 from the boundary cell, on the right"
        " boundary of the domain. The red boundary face lies between the offsets 0 and 1. The"
        " inner cells at the offsets -1 and 0, shaded, are read; the ghosts at the offsets 1"
        " and 2, dashed and red, are written; the cell at -2 is not used. Above the row, a"
        " parabola p goes through the values u_-1 and u_0 at the centers of the two inner"
        " cells and through the value v, in red, on the boundary face. Open red circles mark"
        " the values of p at the centers of the two ghosts.",
        patterns(p),
        g,
    )


LAYER_ROWS = (
    (1, "condition", "Dirichlet<1>"),
    (2, "extrapolation, degree 2", ""),
    (3, "extrapolation, degree 4", ""),
)


@figure
def ghost_layers(p):
    """Draw the three ghost layers of a ghost width 3 under ``Dirichlet<1>``.

    Figure of reference/bc.md. The layer 1 comes from the condition, on the offsets
    [0, 2); the layers 2 and 3 from ``PolynomialExtrapolation`` on the stencils of 4
    and 6 cells that end on their ghost, [-1, 3) and [-2, 4).
    """
    check_layers()
    first, last, cw, ch, pitch, top = -2, 4, 56, 26, 64, 28
    x0 = 150
    face = x0 + (1 - first) * cw
    g = text(face, top - 13, "boundary face", anchor="middle", size=11.5, fill=RED)
    for r, (layer, what, code) in enumerate(LAYER_ROWS):
        y = top + r * pitch
        a, b = (0, 2) if layer == 1 else extrapolation_stencil(layer)
        roles = {}
        for k in range(first, last):
            if not a <= k < b:
                role = "inner" if k <= 0 else "ghost"
            elif k == b - 1:
                role = "write"
            else:
                role = "read" if k <= 0 else "ghost-read"
            roles[(k - first, 0)] = role
        g += _grid(x0, y, cw, ch, roles, p)
        g += line(face, y - 4, face, y + ch + 4, stroke=RED, sw=HEAVY)
        g += bracket(
            x0 + (a - first) * cw,
            x0 + (b - first) * cw,
            y + ch + 9,
            f"[{a}, {b})",
            size=11,
            label_dy=17,
        )
        g += text(x0 - 16, y + 18, f"layer {layer}", anchor="end", size=12, fill=INK)
        right = x0 + (last - first) * cw + 16
        if code:
            what += f' <tspan class="sm-fig-code" fill="{INK}">{escape(code)}</tspan>'
        g += text(right, y + 18, what, size=11.5, fill=INK2)
    axis = top + len(LAYER_ROWS) * pitch + 4
    g += _offset_axis(x0, cw, first, last, axis)
    return Drawing(
        DIAGRAM_WIDTH,
        axis + 29,
        "Three rows of cells over the offsets -2 to 3 from the boundary cell, a red boundary"
        " face between the offsets 0 and 1, ghosts dashed. Layer 1, filled by the condition"
        " Dirichlet<1>: the stencil [0, 2) reads the boundary cell and writes the ghost at"
        " offset 1. Layer 2, filled by extrapolation of degree 2: the stencil [-1, 3) of 4"
        " cells reads the offsets -1, 0 and the ghost 1, and writes the ghost at offset 2."
        " Layer 3, filled by extrapolation of degree 4: the stencil [-2, 4) of 6 cells reads"
        " the offsets -2 to 2 and writes the ghost at offset 3.",
        patterns(p),
        g,
    )


@figure
def corner_ghosts(p):
    """Draw the corner block of the top-right corner of a 2D mesh, ghost width 3.

    Figure of reference/bc.md, with the values the 2D program prints for
    ``u = 10 i + j``: the diagonal ghosts mirror the inner diagonal cells about the
    corner, the other ghosts copy the diagonal ghost of their column.
    """
    corner, width = 7, 3
    block = corner_values(corner, width)
    c = 44
    lo, hi = corner - width + 1, corner + width + 1  # the indices drawn, [5, 11)
    n = hi - lo
    x0, top = (DIAGRAM_WIDTH - n * c) / 2 + 10, 34

    def X(i):
        return x0 + (i - lo) * c

    def Y(j):
        return top + (hi - 1 - j) * c

    roles, values = {}, {}
    for i in range(lo, hi):
        for j in range(lo, hi):
            key = (i - lo, hi - 1 - j)
            if i <= corner and j <= corner:
                roles[key] = "read"
                values[(i, j)] = 10 * i + j
            elif i <= corner or j <= corner:
                roles[key] = "ghost"
            else:
                roles[key] = "write-hatch" if i == j else "write"
                values[(i, j)] = block[(i, j)]
    g = _grid(X(lo), Y(hi - 1), c, c, roles, p)
    # the inner diagonal cells the diagonal ghosts mirror, outlined in red inside
    for k in range(lo, corner + 1):
        g += rect(X(k) + 2.5, Y(k) + 2.5, c - 5, c - 5, stroke=RED)
    for (i, j), value in values.items():
        g += text(
            X(i) + c / 2,
            Y(j) + c / 2 + 4,
            f"{value}",
            anchor="middle",
            size=11.5,
            fill=INK,
            cls="sm-fig-code",
        )
    # the domain: its right and top boundaries, and its corner
    xb, yb = X(corner + 1), Y(corner)
    g += line(X(lo), yb, xb, yb, sw=HEAVY) + line(xb, yb, xb, Y(lo) + c, sw=HEAVY)
    g += circle(xb, yb, 3, fill=RED)
    # the indices
    for k in range(lo, hi):
        g += text(
            X(k) + c / 2, Y(lo) + c + 18, f"{k}", anchor="middle", size=11.5, cls="sm-fig-code"
        )
        g += text(X(lo) - 10, Y(k) + c / 2 + 4, f"{k}", anchor="end", size=11.5, cls="sm-fig-code")
    g += _math(X(hi) + 14, Y(lo) + c + 18, "i", size=15, anchor="start", knock=False)
    g += _math(X(lo) - 10, Y(hi - 1) - 10, "j", size=15, anchor="end", knock=False)
    return Drawing(
        DIAGRAM_WIDTH,
        Y(lo) + c + 28,
        "The top-right corner of a 2D mesh, cells indexed by i and j from 5 to 10, the field"
        " u = 10 i + j. Inside the domain, the 3 by 3 cells with i and j from 5 to 7 hold 55"
        " to 77; the diagonal cells 55, 66 and 77 have a red outline. A heavy rule marks the"
        " right and top boundaries, a red dot their corner. The ghosts beside the boundaries"
        " are dashed and blank. The 3 by 3 corner block of ghosts is red: its diagonal ghosts,"
        " hatched, hold 77, 66 and 55 going outward, the mirror image of the inner diagonal"
        " about the corner; every other ghost holds the value of the diagonal ghost of its"
        " column, so the columns i = 8, 9 and 10 hold 77, 66 and 55.",
        patterns(p),
        g,
    )
