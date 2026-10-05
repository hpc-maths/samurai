"""The figures of the documentation, one function each.

A figure function takes the prefix of the ids of its SVG elements and returns a
``Drawing``. Diagrams are drawn 680 units wide, the text column; plates 711 units
wide, the inside of a plate frame. The ``plate`` and ``diagram`` directives
(``_ext/plates.py``) refer to a figure by the name of its function.
"""

from math import cos, sin

from .draw import (
    EMPTY,
    FIG,
    HAIR,
    HEAVY,
    INK,
    INK2,
    INK3,
    PAPER,
    RED,
    RWASH,
    WASH,
    bracket,
    callout,
    cell_row,
    fig_label,
    line,
    num,
    path,
    patterns,
    rect,
    text,
)
from .mesh import build_mesh, circle_refine, draw_mesh
from .registry import Drawing, figure

PLATE_WIDTH = 711
DIAGRAM_WIDTH = 680


def _number_line(x0, y, cw, n, size=11.5):
    """The axis under rows of level-0 cells, with the indices 0 to n."""
    g = line(x0, y, x0 + n * cw, y)
    for k in range(n + 1):
        g += line(x0 + k * cw, y, x0 + k * cw, y + 5) + text(
            x0 + k * cw, y + 18, k, anchor="middle", size=size
        )
    return g + text(x0 - 14, y + 18, "level 0", anchor="end", size=11.5)


# ------------------------------------------------------------------- diagrams


@figure
def boolean_operations(_p):
    """reference/subset.md: the four results of set_algebra_boolean.cpp."""
    x0, cw, n, pitch = 178, 34, 14, 58
    rows = [
        ("A", [(0, 5), (10, 13)], "ink"),
        ("B", [(4, 8)], "ink"),
        ("union_(a, b)", [(0, 8), (10, 13)], "red"),
        ("intersection(a, b)", [(4, 5)], "red"),
        ("difference(a, b)", [(0, 4), (10, 13)], "red"),
        ("difference(b, a)", [(5, 8)], "red"),
    ]
    g = ""
    for r, (name, cells, tone) in enumerate(rows):
        # the results sit under a rule, a little apart from the operands
        y = 14 + r * pitch + (22 if r >= 2 else 0)
        g += cell_row(x0, y, cw, n, cells, ch=24, tone=tone, label_size=11)
        if tone == "ink":
            g += text(x0 - 14, y + 17, name, anchor="end", size=14, fill=INK, cls="sm-fig-math")
        else:
            g += text(x0 - 14, y + 17, name, anchor="end", size=11.5, fill=RED, cls="sm-fig-code")
    g += line(24, 16 + 2 * pitch, DIAGRAM_WIDTH - 8, 16 + 2 * pitch, stroke=FIG)
    axis = 32 + 6 * pitch
    g += _number_line(x0, axis, cw, n)
    return Drawing(
        DIAGRAM_WIDTH,
        axis + 26,
        "Two 1D sets on the cells 0 to 13 of level 0. A holds [0, 5) and [10, 13), B holds [4, 8)."
        " In red, their union union_(a, b) is [0, 8) and [10, 13), their intersection is the single"
        " cell [4, 5), A minus B, difference(a, b), is [0, 4) and [10, 13), and B minus A,"
        " difference(b, a), is [5, 8).",
        "",
        g,
    )


@figure
def set_projection(_p):
    """reference/subset.md: set_algebra_levels.cpp, two operands of two levels and
    the set projection of their intersection."""
    x0, unit, pitch = 200, 116, 58  # a level-0 cell is 116 units wide
    rows = [
        ("level 1", None, 1, 6, [(0, 4)], "ink"),
        ("level 0", None, 0, 3, [(1, 3)], "ink"),
        ("intersection", "level 1", 1, 6, [(2, 4)], "red"),
        (".on(0)", "level 0", 0, 3, [(1, 2)], "red"),
        (".on(3)", "level 3", 3, 24, [(8, 16)], "red"),
    ]
    top = 14
    bottom = top + 5 * pitch + 22
    g = ""
    # the edges of the cell 1 of level 0, through every row
    for k in (1, 2):
        g += line(
            x0 + k * unit, top - 6, x0 + k * unit, bottom - 16, stroke=RED, sw=0.8, dash="2 3"
        )
    for r, (name, sub, level, n, cells, tone) in enumerate(rows):
        y = top + r * pitch + (22 if r >= 2 else 0)
        # a level-l cell is 2^l times narrower than a level-0 cell
        g += cell_row(x0, y, unit / 2**level, n, cells, ch=24, tone=tone, label_size=11)
        if sub:
            g += text(x0 - 14, y + 11, name, anchor="end", size=11.5, fill=RED, cls="sm-fig-code")
            g += text(x0 - 14, y + 25, sub, anchor="end", size=11)
        else:
            g += text(x0 - 14, y + 17, name, anchor="end", size=12, fill=INK)
    g += line(24, top + 2 * pitch + 2, DIAGRAM_WIDTH - 8, top + 2 * pitch + 2, stroke=FIG)
    g += text(x0 + 3 * unit + 18, top + pitch + 17, "compared as they are:", size=11)
    g += text(x0 + 3 * unit + 18, top + pitch + 32, "[1, 3),", size=11, cls="sm-fig-code")
    g += text(x0 + 3 * unit + 18, top + pitch + 46, "which is wrong", size=11)
    axis = bottom - 4
    g += line(x0, axis, x0 + 3 * unit, axis)
    for k in range(4):
        g += line(x0 + k * unit, axis, x0 + k * unit, axis + 5) + text(
            x0 + k * unit, axis + 18, k, anchor="middle", size=11.5
        )
    g += text(x0 - 14, axis + 18, "level 0", anchor="end", size=11.5)
    return Drawing(
        DIAGRAM_WIDTH,
        axis + 26,
        "Two levels of a 1D mesh, drawn at their true size. Level 1 holds the cells 0 to 3, [0, 4)."
        " Level 0 holds the cells 1 and 2, [1, 3), each one twice as wide as a cell of level 1, so"
        " that the cell 1 of level 0 covers the cells 2 and 3 of level 1. In red, their"
        " intersection is [2, 4) on level 1; brought to level 0 with on(0) it is [1, 2), and"
        " brought to level 3 with on(3) it is [8, 16). Dotted red guides mark the edges of the cell"
        " 1 of level 0.",
        "",
        g,
    )


@figure
def level_rows(_p):
    """tutorial/getting_started.md: the levels 4 to 8 of the program, over the
    first quarter of the x axis."""
    x0, quarter, pitch, top = 84, 512, 56, 34
    g = line(x0, top - 10, x0 + quarter, top - 10)
    for v, label in ((0, "0"), (0.25, "1/16"), (0.5, "1/8"), (0.75, "3/16"), (1, "1/4")):
        g += line(x0 + v * quarter, top - 15, x0 + v * quarter, top - 10)
        g += text(x0 + v * quarter, top - 20, label, anchor="middle", size=11)
    g += text(x0 - 14, top - 16, "x", anchor="end", size=13, fill=INK, cls="sm-fig-math")
    for level in range(4, 9):
        # [0, 1/4) holds 2^level / 4 cells of side 1/2^level
        y, n = top + (level - 4) * pitch, 2 ** (level - 2)
        cw = quarter / n
        red = level == 8
        for k in range(1, n):
            g += rect(x0 + k * cw, y, cw, 22, stroke=FIG, sw=EMPTY)
        g += rect(x0, y, cw, 22, fill=RWASH if red else WASH, stroke=RED if red else INK)
        g += bracket(x0, x0 + quarter, y + 31, f"[0, {n})", color=RED if red else INK, size=11)
        g += text(
            x0 - 14, y + 16, f"level {level}", anchor="end", size=12, fill=RED if red else INK
        )
        g += text(
            x0 + quarter + 16, y + 16, f"side 1/{2**level}", size=11.5, fill=RED if red else INK2
        )
    return Drawing(
        DIAGRAM_WIDTH,
        top + 5 * pitch - 4,
        "Five rows of cells over the first quarter of the x axis, from 0 to 1/4. Level 4 has 4"
        " cells of side 1/16, level 5 has 8, level 6 has 16, level 7 has 32 and level 8, in red, 64"
        " cells of side 1/256. The first cell of each row is filled; under each row a half-open"
        " bracket gives the cell indices [0, 4), [0, 8), [0, 16), [0, 32), [0, 64).",
        "",
        g,
    )


# --------------------------------------------------------------------- plates


@figure
def transported_disc(p):
    """tutorial/getting_started.md: the disc at t = 0 and at t = 0.3, with the
    mesh refined along its edge. Drawn with levels 3 to 6, the program uses 4 to 8."""
    min_level, max_level, radius = 3, 6, 0.2
    start, end = (0.3, 0.3), (0.6, 0.6)  # the centre moves by a t = (1, 1) * 0.3
    m0 = build_mesh(min_level, max_level, circle_refine(*start, radius))
    m1 = build_mesh(min_level, max_level, circle_refine(*end, radius))
    size, oy, xa, xb = 300, 14, 34, 380

    def X0(u):
        return xa + u * size

    def X1(u):
        return xb + u * size

    def Y(v):
        return oy + (1 - v) * size

    g = draw_mesh(p, m0, xa, oy, size, max_level, disc=(*start, radius))
    g += draw_mesh(p, m1, xb, oy, size, max_level, disc=(*end, radius), ghost=(*start, radius))
    g += line(X1(0.3), Y(0.3), X1(0.585), Y(0.585), stroke=RED, extra=f'marker-end="url(#{p}-ar)"')
    g += text(
        X1(0.56),
        Y(0.3) + 4,
        '<tspan font-weight="600" font-style="normal">a</tspan> t',
        size=15,
        fill=RED,
        cls="sm-fig-math",
        knock=True,
    )
    edge = (start[0] + radius * cos(0.9), start[1] + radius * sin(0.9))
    g += callout(X0(edge[0]), Y(edge[1]), X0(0.62), Y(0.75), "a", red=True)
    g += callout(X0(0.3), Y(0.3), X0(0.38), Y(0.42), "b")
    g += callout(X0(0.88), Y(0.12), X0(0.8), Y(0.3), "c")
    g += callout(X1(0.3), Y(0.3), X1(0.15), Y(0.15), "d")
    g += fig_label(xa + size / 2, oy + size + 46, 1, "t = 0")
    g += fig_label(xb + size / 2, oy + size + 46, 2, "t = 0.3")
    return Drawing(
        PLATE_WIDTH,
        oy + size + 58,
        "Two square meshes. Left, at t = 0, the finest cells form a band along a circle near the"
        " lower left corner; right, at t = 0.3, the circle has moved up and right along the"
        " velocity (1, 1), the band of fine cells has moved with it, and a dotted circle marks the"
        " start.",
        patterns(p),
        g,
        panels=((0, 0, 356, 372, ""), (350, 0, 361, 372, "")),
    )


def _loop_station(k):
    """The centre of the k-th figure of the time step plate."""
    return 92 + k * 176


@figure
def time_step(p):
    """tutorial/getting_started.md: the four statements of the time loop."""
    stations = [
        ("adapt", "MRadaptation(mra_config)"),
        ("resize", "unp1.resize()"),
        ("scheme", "unp1 = u - dt * conv(u)"),
        ("swap", "samurai::swap(u, unp1)"),
    ]
    cx, top = _loop_station, 22
    arrow, red_arrow = f'marker-end="url(#{p}-a)"', f'marker-end="url(#{p}-ar)"'
    # 1, adapt: a coarse cell refined in one quadrant, the new cells hatched
    x, y = cx(0) - 46, top
    g = rect(x, y, 92, 92, sw=HEAVY)
    g += line(x + 46, y, x + 46, y + 92, sw=EMPTY) + line(x, y + 46, x + 92, y + 46, sw=EMPTY)
    for a in range(2):
        for b in range(2):
            g += rect(x + 46 + a * 23, y + b * 23, 23, 23, fill=f"url(#{p}-r)", sw=HAIR)
    # 2, resize: the storage of unp1 follows the new mesh
    x, y = cx(1) - 56, top + 16
    for k in range(6):
        g += rect(x + k * 19, y, 19, 19, fill=f"url(#{p}-k)", sw=EMPTY)
    for k in range(8):
        new = k > 5
        g += rect(
            x + k * 19 - 9,
            y + 46,
            19,
            19,
            fill=f"url(#{p}-{'r' if new else 'k'})",
            sw=HAIR if new else EMPTY,
        )
    g += line(x + 57, y + 23, x + 57, y + 42, stroke=INK2, extra=arrow)
    g += text(x - 10, y + 14, "old", anchor="end", size=12, fill=INK, cls="sm-fig-math")
    g += text(x - 18, y + 60, "new", anchor="end", size=12, fill=INK, cls="sm-fig-math")
    # 3, scheme: a coarse cell beside two fine ones, the fluxes across their face
    x, y = cx(2) - 56, top + 10
    g += rect(x, y, 56, 56, fill=f"url(#{p}-k)")
    g += rect(x + 56, y, 28, 28) + rect(x + 56, y + 28, 28, 28)
    g += line(x + 56, y - 6, x + 56, y + 62, stroke=RED, sw=2)
    g += line(x + 44, y + 14, x + 70, y + 14, stroke=RED, extra=red_arrow)
    g += line(x + 44, y + 42, x + 70, y + 42, stroke=RED, extra=red_arrow)
    g += line(x - 18, y + 28, x + 8, y + 28, stroke=INK2, extra=arrow)
    g += rect(x + 84, y, 28, 28, stroke=INK3, dash="3 3") + rect(
        x + 84, y + 28, 28, 28, stroke=INK3, dash="3 3"
    )
    g += text(x + 98, y + 78, "ghosts", anchor="middle", size=12, fill=INK, cls="sm-fig-math")
    # 4, swap: the two fields exchanged
    x, y = cx(3) - 52, top + 12
    g += rect(x, y, 38, 22, fill=PAPER) + text(
        x + 19, y + 16, "u", anchor="middle", size=14, fill=INK, cls="sm-fig-math"
    )
    g += rect(x + 66, y, 38, 22, fill=PAPER) + text(
        x + 85, y + 15, "unp1", anchor="middle", size=10.5, fill=INK, cls="sm-fig-code"
    )
    g += path(
        f"M{x + 19} {y + 28} C{x + 19} {y + 68}, {x + 85} {y + 38}, {x + 85} {y + 72}",
        stroke=RED,
        extra=red_arrow,
    )
    g += path(
        f"M{x + 85} {y + 28} C{x + 85} {y + 68}, {x + 19} {y + 38}, {x + 19} {y + 72}",
        stroke=RED,
        extra=red_arrow,
    )
    for k, (name, code) in enumerate(stations):
        g += fig_label(cx(k), top + 132, k + 1, name)
        g += text(cx(k), top + 152, code, anchor="middle", size=10.5, cls="sm-fig-code")
        if k < 3:
            g += line(cx(k) + 66, top + 46, cx(k) + 110, top + 46, stroke=INK2, extra=arrow)
    # back to the first statement, until the final time
    yb = top + 196
    g += path(f"M{cx(3)} {top + 164} V{yb} H{cx(0)} V{top + 168}", extra=arrow)
    mid = (cx(0) + cx(3)) / 2
    g += rect(mid - 140, yb - 12, 280, 22, fill=PAPER, stroke="none")
    g += text(
        mid,
        yb + 5,
        "t += dt, and again until t = T = 0.3",
        anchor="middle",
        size=14,
        fill=INK,
        cls="sm-fig-math",
    )
    g += callout(cx(2), top + 4, cx(2) + 24, top - 12, "a", red=True)
    # one panel per station, cut between the stations; resize is the widest
    half = (
        (0, 0, 168, 182, "half"),
        (168, 0, 190, 182, "half"),
        (358, 0, 178, 182, "half"),
        (536, 0, 175, 182, "half"),
    )
    return Drawing(
        PLATE_WIDTH,
        yb + 16,
        "Four stations of one time step. Adapt: a coarse cell refined in one quadrant, new cells"
        " hatched. Resize: the storage of unp1 grows to the new mesh. Scheme: fluxes cross the face"
        " between a coarse cell and two fine cells, marked in red, with ghost cells beyond. Swap: u"
        " and unp1 are exchanged. An arrow returns from the last station to the first until t ="
        " 0.3.",
        patterns(p),
        g,
        panels=(*half, (mid - 160, 196, 320, 36, "")),
    )


@figure
def plus_sign_operations(p):
    """reference/subset.md: set_algebra_expand_contract.cpp, every result drawn
    from the intervals the program prints."""
    plus = [(1, 1), (0, 2), (1, 2), (2, 2), (1, 3)]
    # y = 0 and 4: [0, 3); y = 1 to 3: [-1, 4)
    expanded = [(i, j) for i in range(3) for j in range(5)] + [
        (i, j) for i in (-1, 3) for j in range(1, 4)
    ]
    # y = 0 and 4: [1, 2); y = 1 to 3: [0, 3)
    expanded_y = [(1, 0), (1, 4)] + [(i, j) for i in range(3) for j in range(1, 4)]
    grids = [
        ("plus", None, plus, False),
        ("expand(plus, 1)", None, expanded, True),
        ("expand(plus, 1,", "{false, true})", expanded_y, True),
        ("contract(plus, 1)", None, [(1, 2)], True),
        ("translate(plus,", "{1, -1})", [(i + 1, j - 1) for i, j in plus], True),
    ]
    c = 19
    gw, gh, gap, oy = 6 * c, 5 * c, 30.5, 26
    s = ""
    for q, (name, name2, cells, result) in enumerate(grids):
        ox = 18 + q * (gw + gap)

        def P(i, j, ox=ox):
            """The top left corner of the cell (i, j), x from -1 to 4, y from 0 to 4 upwards."""
            return ox + (i + 1) * c, oy + (4 - j) * c

        for i in range(-1, 5):
            for j in range(5):
                s += rect(*P(i, j), c, c, stroke=FIG, sw=EMPTY)
        for i, j in cells:
            if result:
                s += rect(*P(i, j), c, c, fill=f"url(#{p}-r)", stroke=RED, sw=HAIR)
            else:
                s += rect(*P(i, j), c, c, fill=WASH, stroke=INK)
        if result:
            a, b = P(1, 3)
            outline = f"M{num(a)} {num(b)}h{c}v{c}h{c}v{c}h-{c}v{c}h-{c}v-{c}h-{c}v-{c}h{c}z"
            s += f'<path d="{outline}" fill="none" stroke="{INK}" stroke-width="{HEAVY}"/>'
        for i in range(-1, 5):
            s += text(P(i, 0)[0] + c / 2, oy + gh + 14, i, anchor="middle", size=10.5)
        if q == 0:
            for j in range(5):
                s += text(ox - 6, P(0, j)[1] + c / 2 + 4, j, anchor="end", size=10.5)
        s += fig_label(ox + gw / 2, oy + gh + 42, q + 1)
        s += text(
            ox + gw / 2,
            oy + gh + 61,
            name,
            anchor="middle",
            size=10.5,
            fill=RED if q else INK,
            cls="sm-fig-code",
        )
        if name2:
            s += text(
                ox + gw / 2,
                oy + gh + 75,
                name2,
                anchor="middle",
                size=10.5,
                fill=RED,
                cls="sm-fig-code",
            )
    contracted = 18 + 3 * (gw + gap)
    s += callout(
        contracted + 2.5 * c, oy + 2.5 * c, contracted + 4.7 * c, oy + 0.6 * c, "a", red=True
    )
    expanded_x = 18 + gw + gap
    s += callout(expanded_x + 0.5 * c, oy + 0.5 * c, expanded_x + 0.3 * c, 12, "b")
    return Drawing(
        PLATE_WIDTH,
        oy + gh + 86,
        "Five 2D grids of x from -1 to 4 and y from 0 to 4. The plus sign holds the cell (1, 2) and"
        " its four axis neighbors. expand(plus, 1) fills a 5 by 5 square without its four corners."
        " Expanding only along y gives a column of 5 cells at x = 1 and a column of 3 cells at x ="
        " 0 and x = 2. contract(plus, 1) keeps only the center cell (1, 2). translate(plus, {1,"
        " -1}) moves the plus sign one cell right and one cell down. The results are hatched in red"
        " under the outline of the plus sign.",
        patterns(p),
        s,
        panels=tuple((18 + q * (gw + gap) - 14, 0, 144, 207, "half") for q in range(5)),
    )
