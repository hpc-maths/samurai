"""The figures of ``LBM/schemes.md``: velocity set, time step, stream, walls and blocks.

The figures follow ``include/samurai/schemes/lbm/``. ``LBMScheme::operator()`` fills
the ghosts of ``f`` (``update_ghost_mr``), streams, then collides in every cell:
``m = M f``, equilibrium and relaxation, the optional source, ``f = M^-1 m``. On a
level ``l = L - j``, ``LBMScheme::stream`` applies the stencil of ``build_stencil``:
the prediction of stencil radius 1 (``prediction_stencil_radius`` 1, order 3) of the
fine cells ``[-c, 2^j - c)`` summed by ``detail::get_prediction``, times ``1 / 2^(j dim)``.
``stream_weights`` computes the same weights; for ``c = (1, 0)`` the code gives
-1/16, 9/16, 9/16, -1/16 at ``j = 1``. The reflection of the walls is
``LbmReflectionImpl`` of ``boundary.hpp``.
"""

from collections import defaultdict
from fractions import Fraction

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
    arrow,
    bracket,
    fig_label,
    line,
    path,
    patterns,
    rect,
    text,
)
from ..flows import Lane, Step, chain
from ..registry import DIAGRAM_WIDTH, PLATE_WIDTH, Drawing, figure

# The D2Q4 velocities in the order of demos/LBM/new_D2Q4_advection.cpp.
D2Q4 = ((1, 0), (0, 1), (-1, 0), (0, -1))
STAR = "⋆"
MINUS = "−"


def _math(chunks, size):
    """Build serif italic markup from ``chunks``.

    A ``("sub", s)`` chunk is set as a subscript, a ``("sup", s)`` chunk as a
    superscript, a ``("label", s)`` chunk upright in the sans, a string on the
    baseline. The baseline comes back after each shifted chunk, so that a chunk can
    follow it.
    """
    s, shift = "", 0.0
    for chunk in chunks:
        kind, body = chunk if isinstance(chunk, tuple) else ("", chunk)
        target = {"sub": 0.3 * size, "sup": -0.38 * size}.get(kind, 0.0)
        attributes = f' dy="{target - shift:g}"' if target != shift else ""
        if kind in ("sub", "sup"):
            attributes += f' font-size="{0.72 * size:g}"'
        elif kind == "label":
            attributes += ' class="sm-fig-label" font-style="normal"'
        s += f"<tspan{attributes}>{body}</tspan>" if attributes else body
        shift = target
    return s


def _label(x, y, chunks, size=15, fill=INK, anchor="middle"):
    """Set a math label at (x, y) from ``chunks``, see ``_math``."""
    return text(x, y, _math(chunks, size), anchor=anchor, size=size, fill=fill, cls="sm-fig-math")


def _vector(v):
    """Write a lattice velocity as "(1, 0)", with a true minus sign."""
    return "(" + ", ".join(f"{MINUS}{-c}" if c < 0 else str(c) for c in v) + ")"


@figure
def d2q4_velocities(p):
    """Draw the four velocities of D2Q4 from a cell to its neighbors, in the order of the code.

    Figure of LBM/schemes.md.
    """
    s, cx, cy = 70, DIAGRAM_WIDTH / 2, 34 + 1.5 * 70
    g = ""
    for vx, vy in D2Q4:
        g += rect(cx - s / 2 + vx * s, cy - s / 2 - vy * s, s, s, stroke=FIG, sw=EMPTY)
    g += rect(cx - s / 2, cy - s / 2, s, s, fill=WASH, stroke=INK)
    ink = f'marker-end="url(#{p}-ai)"'
    # where the label of each velocity stands, beyond the neighbor cell
    places = {
        (1, 0): (cx + 1.5 * s + 14, cy + 5, "start"),
        (0, 1): (cx, cy - 1.5 * s - 12, "middle"),
        (-1, 0): (cx - 1.5 * s - 14, cy + 5, "end"),
        (0, -1): (cx, cy + 1.5 * s + 24, "middle"),
    }
    for a, (vx, vy) in enumerate(D2Q4):
        g += line(cx, cy, cx + vx * s, cy - vy * s, stroke=INK, sw=HEAVY, extra=ink)
        x, y, anchor = places[(vx, vy)]
        g += text(
            x,
            y,
            _math(["c", ("sub", a), ("label", f" = {_vector((vx, vy))}")], 16),
            anchor=anchor,
            size=16,
            fill=INK,
            cls="sm-fig-math",
        )
    g += f'<circle cx="{cx:g}" cy="{cy:g}" r="2.6" fill="{INK}"/>'
    return Drawing(
        DIAGRAM_WIDTH,
        cy + 1.5 * s + 36,
        "A shaded cell with its four neighbors drawn faint. Four arrows leave its center, one to"
        " the center of each neighbor: c_0 = (1, 0) to the east, c_1 = (0, 1) to the north,"
        " c_2 = (-1, 0) to the west and c_3 = (0, -1) to the south.",
        arrow(f"{p}-ai", INK),
        g,
    )


@figure
def lbm_time_step(p):
    """Draw one call of ``scheme(f, m)`` as a chain of its steps, in the order of the code.

    Figure of LBM/schemes.md. ``LBMScheme::operator()`` runs ``update_ghost_mr``,
    ``stream`` and ``collide``; ``collide`` computes the moments, relaxes them and
    computes the distributions back, in every cell.
    """
    lane = Lane(x=12, y=44, width=DIAGRAM_WIDTH - 24, box=104)
    inv = f'M<tspan dy="-6" font-size="10.5">{MINUS}1</tspan><tspan dy="6"> m</tspan>'
    steps = (
        Step("fill ghosts", code="update_ghost_mr(f)"),
        Step("stream"),
        Step("m = M f", "math"),
        Step("relax"),
        Step(f'f = {inv}<tspan dy="-5" font-size="10.5">{STAR}</tspan>', "math"),
    )
    star = f'm<tspan dy="-5" font-size="10.5">{STAR}</tspan>'
    flow = chain(p, lane, steps, ("f", "f", "m", star))
    pitch = lane.pitch(len(steps))
    g = flow.body
    # scheme(f, m) runs the whole chain
    left, right, top = lane.x, lane.x + lane.width, 22
    g += path(f"M{left} {top + 6}V{top}H{right}V{top + 6}", stroke=INK3, sw=HAIR) + text(
        (left + right) / 2,
        top - 6,
        "scheme(f, m)",
        anchor="middle",
        size=11,
        fill=INK,
        cls="sm-fig-code",
    )
    # collide: the last three steps, in every cell
    xa, xb = lane.x + 2 * pitch, lane.x + 4 * pitch + lane.box
    yb = lane.y + lane.height + 8
    g += path(f"M{xa:g} {yb - 4}V{yb}H{xb:g}V{yb - 4}", stroke=INK3, sw=HAIR)
    g += text((xa + xb) / 2, yb + 16, "collide, in every cell", anchor="middle", size=12)
    return Drawing(
        DIAGRAM_WIDTH,
        flow.height + 4,
        "Five boxes in a row, linked by arrows, under a bracket labeled scheme(f, m): fill ghosts,"
        " with update_ghost_mr(f) under it, then stream, then m = M f, then relax, then"
        " f = M^-1 m*. The arrows carry f, f, m and m*. A bracket under the last three boxes reads"
        " collide, in every cell.",
        patterns(p),
        g,
    )


def _fine_weights(i, j):
    """Weigh the coarse values in the radius-1 prediction of fine cell ``i``, ``j`` levels down.

    The prediction has a stencil radius of 1, so it is of order 3. The coarse cells are
    numbered from the cell 0 that holds the fine cells ``[0, 2^j)``. Each level predicts
    a child of cell ``k`` as ``f_k + sigma / 8 (f_(k-1) - f_(k+1))``, ``sigma`` being +1
    for the even child.
    """
    if j == 0:
        return {i: Fraction(1)}
    k, sigma = i >> 1, 1 if i % 2 == 0 else -1
    weights = defaultdict(Fraction)
    for offset, sign in ((k, 0), (k - 1, sigma), (k + 1, -sigma)):
        for o, w in _fine_weights(offset, j - 1).items():
            weights[o] += w * (Fraction(sign, 8) if sign else 1)
    return weights


# The taps of LBMScheme::build_stencil for c = (1, 0) on the D2Q4 mesh of the page, dumped
# from the code: the weights of offsets -2 to 2, for j = 1 and j = 2.
BUILD_STENCIL = {
    1: {-2: Fraction(-1, 16), -1: Fraction(9, 16), 0: Fraction(9, 16), 1: Fraction(-1, 16)},
    2: {
        -2: Fraction(-9, 256),
        -1: Fraction(33, 128),
        0: Fraction(27, 32),
        1: Fraction(-9, 128),
        2: Fraction(1, 256),
    },
}


def stream_weights(c, j):
    """Compute the stream stencil of a velocity ``c`` along one axis, ``j`` levels above the finest.

    Return ``{offset: weight}``: the new value of a cell is the sum of the weights
    times the values of the cells at these offsets. It is the mean of the fine
    cells ``[-c, 2^j - c)`` predicted from the cells of the level, as
    ``LBMScheme::build_stencil``; the other axis sums to the identity.
    """
    weights = defaultdict(Fraction)
    for i in range(-c, 2**j - c):
        for o, w in _fine_weights(i, j).items():
            weights[o] += w / 2**j
    return {o: w for o, w in sorted(weights.items()) if w}


def _fraction(w):
    """Write a weight as "9/16" or "-1/16", with a true minus sign."""
    return (MINUS if w < 0 else "") + (
        f"{abs(w.numerator)}/{w.denominator}" if w.denominator > 1 else f"{abs(w)}"
    )


# The three columns of fine cells of a stream that change a cell C, and their fills.
ENTERS, STAYS, LEAVES = "enters", "stays", "leaves"


def _fill(p, kind):
    """Return the fill of a column of fine cells of ``kind``."""
    return {ENTERS: f"url(#{p}-r)", STAYS: WASH, LEAVES: f"url(#{p}-k)"}[kind]


def _coarse_row(p, x0, y, s, columns, fine=True):
    """Draw the cells W, C and E of side ``s`` from x0, C in a heavy rule.

    ``columns`` maps a column of fine cells, numbered 0 to 5 from the west edge of
    W, to its kind. With ``fine``, hairlines cut each cell into 2 by 2 fine cells.
    """
    h = s / 2
    g = ""
    for col, kind in columns.items():
        g += rect(x0 + col * h, y, h, s, fill=_fill(p, kind), stroke="none")
    for k in range(3):
        if fine:
            x = x0 + k * s
            g += line(x + h, y, x + h, y + s, stroke=INK3, sw=HAIR)
            g += line(x, y + h, x + s, y + h, stroke=INK3, sw=HAIR)
    g += rect(x0, y, 3 * s, s, stroke=INK)
    g += line(x0 + s, y, x0 + s, y + s) + line(x0 + 2 * s, y, x0 + 2 * s, y + s)
    g += rect(x0 + s, y, s, s, sw=HEAVY)
    for k, name in enumerate("WCE"):
        g += _label(x0 + (k + 0.5) * s, y + s + 20, [name], size=16)
    return g


def _stream_caption(cx, y, n, name, level):
    """Write the "Fig. n. name" of a stream figure and, under it, the level it works on.

    In ``level``, "L" is set in the serif italic, as in the text.
    """
    level = level.replace("L", '<tspan class="sm-fig-math">L</tspan>')
    return fig_label(cx, y, n, name) + text(cx, y + 20, level, anchor="middle", size=12.5)


@figure
def stream_coarse_cell(p):
    """Draw the stream of f_0 into a cell C one level above the finest, step by step.

    Figure of LBM/schemes.md, in the 2 by 2 layout of a plate. Fig. 1 holds the
    values of W, C and E at level L - 1; Fig. 2 predicts them at level L; Fig. 3
    moves every fine value by c_0 = (1, 0); Fig. 4 averages the four fine values of
    C back to level L - 1. The fills mark the fine columns that enter C, stay in C
    and leave C.
    """
    s, half = 88, PLATE_WIDTH / 2
    art = 3 * s
    x_left, x_right = (half - art) / 2, half + (half - art) / 2
    top1, top2 = 34, 250
    g = ""
    # Fig. 1: one value per cell of level L - 1
    for k in range(3):
        g += rect(x_left + k * s, top1, s, s, fill=WASH, stroke="none")
    g += _coarse_row(p, x_left, top1, s, {}, fine=False)
    # Fig. 2: the prediction; the east column of W, the two columns of C
    g += _coarse_row(p, x_right, top1, s, {1: ENTERS, 2: STAYS, 3: LEAVES})
    # Fig. 3: every fine value moves one fine cell east
    g += _coarse_row(p, x_left, top2, s, {2: ENTERS, 3: STAYS, 4: LEAVES})
    ya = top2 - 12
    g += line(
        x_left + 0.75 * s, ya, x_left + 1.25 * s, ya, stroke=INK2, extra=f'marker-end="url(#{p}-a)"'
    )
    g += _label(x_left + 1.25 * s + 8, ya + 5, ["c", ("sub", 0)], anchor="start")
    # Fig. 4: the four fine values of C, then their mean at level L - 1
    xc = x_right
    g += rect(xc, top2, s / 2, s, fill=_fill(p, ENTERS), stroke="none")
    g += rect(xc + s / 2, top2, s / 2, s, fill=WASH, stroke="none")
    g += line(xc + s / 2, top2, xc + s / 2, top2 + s, stroke=INK3, sw=HAIR)
    g += line(xc, top2 + s / 2, xc + s, top2 + s / 2, stroke=INK3, sw=HAIR)
    g += rect(xc, top2, s, s, sw=HEAVY)
    xm = xc + 2 * s
    g += line(
        xc + s + 12,
        top2 + s / 2,
        xm - 12,
        top2 + s / 2,
        stroke=INK2,
        extra=f'marker-end="url(#{p}-a)"',
    )
    g += text(xc + 1.5 * s, top2 + s / 2 - 8, "mean", anchor="middle", size=12)
    g += rect(xm, top2, s, s, fill=WASH, stroke=INK, sw=HEAVY)
    for x in (xc + s / 2, xm + s / 2):
        g += _label(x, top2 + s + 20, ["C"], size=16)
    # the captions of the four figures
    y1, y2 = top1 + s + 58, top2 + s + 58
    g += _stream_caption(half / 2, y1, 1, "coarse values", "level L − 1")
    g += _stream_caption(half * 1.5, y1, 2, "predict", "from level L − 1 to level L")
    g += _stream_caption(half / 2, y2, 3, "stream", "at level L")
    g += _stream_caption(half * 1.5, y2, 4, "average", "from level L to level L − 1")
    panel = top2 - top1
    panels = tuple(
        (col * half, top - 26, half, panel - 8, "") for top in (top1, top2) for col in (0, 1)
    )
    return Drawing(
        PLATE_WIDTH,
        y2 + 32,
        "Four figures in two rows, each on a row of three square cells W, C and E, C outlined in a"
        " heavy rule. Fig. 1, coarse values at level L - 1: the three cells are shaded, one value"
        " each. Fig. 2, predict: each cell is cut into 2 by 2 fine cells; the east column of W is"
        " hatched in red, the west column of C is shaded, the east column of C is ruled. Fig. 3,"
        " stream at level L: an arrow c_0 points east and every column has moved one fine cell"
        " east: the red hatching fills the west column of C, the shading its east column, and the"
        " ruling the west column of E. Fig. 4, average: the cell C with its hatched west column and"
        " shaded east column, an arrow labeled mean, and the cell C at level L - 1, shaded.",
        patterns(p),
        g,
        panels,
    )


@figure
def stream_stencil(p):
    """Draw the stream stencil of c_0 = (1, 0) for j = 1 and j = 2, from ``stream_weights``.

    Figure of LBM/schemes.md. Each cell holds the vector that reaches it from the
    cell C, hatched; the weight is written under it.
    """
    c, x0, pitch, top = 84, 172, 62, 30
    offsets = range(-2, 3)
    g = ""
    for k, o in enumerate(offsets):
        g += text(
            x0 + (k + 0.5) * c,
            top - 10,
            f"{{{o},0}}",
            anchor="middle",
            size=11.5,
            fill=INK,
            cls="sm-fig-code",
        )
    for row, j in enumerate((1, 2)):
        y = top + row * pitch
        weights = stream_weights(1, j)
        if weights != BUILD_STENCIL[j]:
            raise ValueError(f"the stream weights for j = {j} differ from build_stencil: {weights}")
        g += _label(x0 - 26, y + c / 4 + 5, ["j = ", str(j)], size=16, anchor="end")
        for k, o in enumerate(offsets):
            x = x0 + k * c
            if o not in weights:
                g += rect(x, y, c, c / 2, stroke=FIG, sw=EMPTY)
                continue
            origin = o == 0
            fill, stroke = (f"url(#{p}-r)", RED) if origin else (WASH, INK)
            g += rect(x, y, c, c / 2, fill=fill, stroke=stroke, sw=HAIR if origin else 1)
            g += text(
                x + c / 2,
                y + c / 4 + 5,
                _fraction(weights[o]),
                anchor="middle",
                size=13,
                fill=INK,
                cls="sm-fig-code",
                knock=origin,
            )
    return Drawing(
        DIAGRAM_WIDTH,
        top + pitch + c / 2 + 12,
        "Two rows of cells of one level, under the column headings {-2,0}, {-1,0}, {0,0}, {1,0}"
        " and {2,0}; the cell {0,0} is hatched in red. Row j = 1 holds the weights -1/16, 9/16,"
        " 9/16 and -1/16 in the cells {-2,0} to {1,0}; the cell {2,0} is empty. Row j = 2 holds"
        " -9/256, 33/128, 27/32, -9/128 and 1/256 in the cells {-2,0} to {2,0}.",
        patterns(p),
        g,
    )


@figure
def bounce_back(p):
    """Draw the bounce-back at the left wall of the D1Q3 dam break, in three rows.

    Figure of LBM/schemes.md. The velocities are c_0 = 0, c_1 = 1 and c_2 = -1, as
    in demos/LBM/new_D1Q3_shallow_waters_dam.cpp; the opposite of c_2 is c_1.
    """
    w, h, pitch, x0, top = 140, 46, 84, 200, 40
    ink = f'marker-end="url(#{p}-ai)"'
    names = ("after the collision", "fill ghosts", "stream")
    star = ["f", ("sub", 2), ("sup", STAR)]
    # BounceBack without a wall equilibrium: sigma = 1 and r = 0, f_1 = f_2*
    value = ["f", ("sub", 2), ("sup", STAR)]
    g = ""
    for row, name in enumerate(names):
        y = top + row * pitch
        g += text(x0 - 30, y + h / 2 + 5, name, anchor="end", size=12.5, fill=INK)
        g += rect(x0, y, w, h, stroke=INK3, dash="3 3")
        g += rect(x0 + w, y, 2 * w, h, stroke=INK)
        g += line(x0 + 2 * w, y, x0 + 2 * w, y + h)
        g += rect(x0 + w, y, w, h, fill=WASH, stroke=INK)
        # the wall, between the ghost and the inner cell
        g += line(x0 + w, y - 8, x0 + w, y + h + 8, stroke=INK, sw=HEAVY)
    cx, gx, ay = x0 + 1.5 * w, x0 + 0.5 * w, 32
    # row 1: f_2* leaves the inner cell towards the wall
    y = top
    g += line(cx + 26, y + ay, cx - 26, y + ay, stroke=INK, sw=HEAVY, extra=ink)
    g += _label(cx, y + 18, star)
    # row 2: the ghost receives the reflection of the inner f_2*
    y = top + pitch
    g += line(cx + 26, y + ay, cx - 26, y + ay, stroke=INK3, extra=f'marker-end="url(#{p}-a)"')
    g += _label(cx, y + 18, star, fill=INK3)
    g += line(
        gx - 26, y + ay, gx + 26, y + ay, stroke=RED, sw=HEAVY, extra=f'marker-end="url(#{p}-ar)"'
    )
    g += _label(gx, y + 18, ["f", ("sub", 1), "\u2002=\u2002"] + value, size=14, fill=RED)
    # row 3: the stream brings it into the inner cell
    y = top + 2 * pitch
    g += line(
        cx - 26, y + ay, cx + 26, y + ay, stroke=RED, sw=HEAVY, extra=f'marker-end="url(#{p}-ar)"'
    )
    g += _label(cx, y + 18, ["f", ("sub", 1), "\u2002=\u2002"] + value, size=14, fill=RED)
    # the names of the cells and of the wall
    yb = top + 2 * pitch + h + 22
    g += text(gx, yb, "ghost", anchor="middle", size=12)
    g += text(cx, yb, "inner cell", anchor="middle", size=12)
    g += text(x0 + 2.5 * w, yb, "next cell", anchor="middle", size=12)
    g += text(x0 + w, top - 14, "wall", anchor="middle", size=12, fill=INK)
    return Drawing(
        DIAGRAM_WIDTH,
        yb + 10,
        "Three rows, each a dashed ghost cell, then a shaded inner cell and the next cell, with a"
        " heavy wall line between the ghost and the inner cell. Row after the collision: in the"
        " inner cell, f_2* points west, towards the wall. Row fill ghosts: f_2* is drawn faint and,"
        " in the ghost, a red arrow points east, labeled f_1 = f_2*. Row stream: a red arrow points"
        " east in the inner cell, labeled f_1 = f_2*.",
        patterns(p) + arrow(f"{p}-ai", INK),
        g,
    )


@figure
def d1q222_layout(p):
    """Draw the six components of the D1Q222 fields and the three velocity schemes they form.

    Figure of LBM/schemes.md, after demos/LBM/new_D1Q222_euler_sod.cpp: three D1Q2
    schemes with the velocities {+1, -1}, in the order of the arguments of
    ``make_lbm_scheme``; the conserved moments m_0, m_2 and m_4 have s = 0.
    """
    c, x0, fy, my = 84, 88, 52, 196
    g = ""
    head = f'marker-end="url(#{p}-a)"'
    for k in range(6):
        x = x0 + k * c
        g += rect(x, fy, c, 40, fill=WASH, stroke=INK)
        g += _label(x + c / 2, fy + 26, ["f", ("sub", k)], size=17)
        # the velocity of the component, +1 for an even index, -1 for an odd one
        xm, d = x + c / 2, 14 if k % 2 == 0 else -14
        g += line(xm - d, fy - 14, xm + d, fy - 14, stroke=INK2, extra=head)
        conserved = k % 2 == 0
        fill, stroke = (f"url(#{p}-r)", RED) if conserved else (WASH, INK)
        g += rect(x, my, c, 40, fill=fill, stroke=stroke, sw=HAIR if conserved else 1)
        g += _label(x + c / 2, my + 26, ["m", ("sub", k)], size=17)
    g += text(x0 - 14, fy + 25, "f", anchor="end", size=13, fill=INK2, cls="sm-fig-code")
    g += text(x0 - 14, my + 25, "m", anchor="end", size=13, fill=INK2, cls="sm-fig-code")
    names = ("ρ", "q = ρu", "E")
    for b in range(3):
        xa, xb = x0 + 2 * b * c, x0 + 2 * (b + 1) * c
        # a gap between two brackets keeps the open end of each one visible
        g += bracket(xa + 4, xb - 4, fy + 50, f"[{2 * b}, {2 * b + 2})", size=11)
        g += text(
            (xa + xb) / 2, fy + 90, f"velocity scheme {b + 1}", anchor="middle", size=12, fill=INK
        )
        g += line((xa + xb) / 2, fy + 100, (xa + xb) / 2, my - 8, stroke=INK2, extra=head)
        g += _label((xa + xb) / 2 + 8, (fy + 100 + my) / 2 + 9, ["M"], size=15, anchor="start")
        g += _label(xa + c / 2, my + 64, [names[b]], size=15, fill=RED)
    # every equilibrium reads the whole moment vector
    yb = my + 80
    g += path(f"M{x0} {yb - 5}V{yb}H{x0 + 6 * c}V{yb - 5}", stroke=INK3, sw=HAIR)
    g += text(
        x0 + 3 * c,
        yb + 18,
        "mm, read by the equilibrium of each velocity scheme",
        anchor="middle",
        size=12,
    )
    return Drawing(
        DIAGRAM_WIDTH,
        yb + 28,
        "Two rows of six cells. The top row, the field f, holds f_0 to f_5; an arrow above each"
        " cell points east for f_0, f_2 and f_4 and west for f_1, f_3 and f_5. Brackets under it"
        " group the components [0, 2), [2, 4) and [4, 6) into velocity schemes 1, 2 and 3. An"
        " arrow labeled M goes from each group down to the bottom row, the field m, which holds"
        " m_0 to m_5; m_0, m_2 and m_4 are hatched in red and named rho, q = rho u and E. A"
        " bracket under the whole row reads mm, read by the equilibrium of each velocity scheme.",
        patterns(p),
        g,
    )
