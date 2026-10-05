"""The figures of ``tutorial/getting_started.md``: the levels, the disc and the time loop."""

from math import cos, sin

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
    RWASH,
    WASH,
    bracket,
    callout,
    fig_label,
    line,
    path,
    patterns,
    rect,
    text,
)
from ..mesh import build_mesh, circle_refine, draw_mesh
from ..registry import DIAGRAM_WIDTH, PLATE_WIDTH, Drawing, figure


@figure
def level_rows(_p):
    """Draw the levels 4 to 8 of the program over the first quarter of the x axis.

    Figure of tutorial/getting_started.md.
    """
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


@figure
def transported_disc(p):
    """Draw the disc at t = 0 and at t = 0.3, with the mesh refined along its edge.

    Figure of tutorial/getting_started.md. It is drawn with levels 3 to 6, while
    the program uses 4 to 8.
    """
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
