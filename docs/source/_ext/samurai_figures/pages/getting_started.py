"""The figures of ``tutorial/getting_started.md``: the levels, the disc and the time loop."""

from math import cos, sin

from ..draw import (
    EMPTY,
    FIG,
    HAIR,
    HEAVY,
    INK,
    INK2,
    PAPER,
    RED,
    RULE,
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
from ..flows import Loop, Row, Station, cycle
from ..mesh import build_mesh, check_outline, circle_refine, draw_mesh
from ..plots import Frame, axes, detail, line_key, predict_children, steps
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
def child_details(p):
    """Draw the details of the two children of a cell, in one dimension.

    Figure of tutorial/getting_started.md. The values are made up; the prediction is
    the prediction of samurai with a stencil radius of 1 (order 3), and the real values
    of the children average to the parent, so that the two details are opposite.
    """
    fr = Frame(30, 26, 420, 170, xlim=(0, 3))
    coarse = (0.30, 0.55, 0.92)
    predicted = predict_children(*coarse)
    real = (0.31, 0.79)  # their mean is the parent value, coarse[1]
    g = line(fr.x - 20, fr.bottom, fr.right + 20, fr.bottom)
    g += axes(fr, xticks=[(k, "") for k in range(4)])
    g += line(fr.X(1.5), fr.bottom, fr.X(1.5), fr.bottom + 4, sw=EMPTY)
    g += steps(fr, [(k, k + 1, v) for k, v in enumerate(coarse)], sw=HEAVY, joined=False)
    for k in range(2):
        name = f'd<tspan baseline-shift="sub" font-size="9">{k + 1}</tspan>'
        g += detail(fr, 1 + k / 2, 1.5 + k / 2, predicted[k], real[k], name, p)
    for k, (name, tone) in enumerate((("neighbor", INK2), ("parent", INK), ("neighbor", INK2))):
        g += text(fr.X(k + 0.5), fr.bottom + 22, f"{name}, level ℓ − 1", "middle", 12, tone)
    g += line_key(
        fr.right + 44,
        40,
        [
            ("value at level ℓ − 1", INK, HEAVY, ""),
            ("predicted at level ℓ", INK2, RULE, "4 3"),
            ("value at level ℓ", RED, HEAVY, ""),
        ],
    )
    return Drawing(
        DIAGRAM_WIDTH,
        236,
        "Three neighboring cells of level ℓ − 1 with their values, the middle one the parent."
        " Its two children at level ℓ have a predicted value, dashed, computed from the parent"
        " and its neighbors, and a real value, in red. The details d1 and d2 are the gaps"
        " between them: d1 points down and d2 up, by the same length.",
        patterns(p),
        g,
    )


@figure
def transported_disc(p):
    """Draw the disc at t = 0 and at t = 0.3, with the mesh refined along its edge.

    Figure of tutorial/getting_started.md. It is drawn with levels 3 to 6, while
    the program uses 4 to 8, and with the disc of the program.

    The circle of the program grazes the corners of coarser cells, down to 0.12
    units at the size of the plate, so refining only the cells it crosses lets its
    heavy outline spill out of the hatched band. The band takes every cell that
    comes within a heavy rule of the circle instead: still narrower than the band
    of the program, whose details reach several cells away from the edge.
    """
    min_level, max_level, radius = 3, 6, 0.2
    start, end = (0.3, 0.3), (0.6, 0.6)  # the centre moves by a t = (1, 1) * 0.3
    size, oy, xa, xb = 300, 14, 34, 380

    def band_mesh(centre):
        leaves = build_mesh(min_level, max_level, circle_refine(*centre, radius, HEAVY / size))
        check_outline(leaves, (*centre, radius), max_level, size)
        return leaves

    m0, m1 = band_mesh(start), band_mesh(end)

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


# The stations of the time step plate. Each box holds the widest statement,
# MRadaptation(mra_config), which the mono sets 151.2 units wide at 10.5, so that
# the panel of a station shows its art, its name and its code whole. The four boxes
# fill the width of the plate, which leaves 22 units for each arrow.
TIME_STEP_ROW = Row(x0=80, pitch=184, box=158, top=22, gap=2)
TIME_STEP_STATIONS = (
    Station("adapt", "MRadaptation(mra_config)"),
    Station("resize", "unp1.resize()"),
    Station("scheme", "unp1 = u - dt * conv(u)"),
    Station("swap", "samurai::swap(u, unp1)"),
)


def _adapt_art(p, cx, top):
    """Draw a coarse cell of four quadrants, one refined: kept cells ruled, new ones hatched."""
    x, y, h = cx - 46, top, 46
    g = ""
    for a, b in ((0, 0), (0, 1), (1, 1)):
        g += rect(x + a * h, y + b * h, h, h, fill=f"url(#{p}-k)", sw=EMPTY)
    for a in range(2):
        for b in range(2):
            g += rect(x + h + a * 23, y + b * 23, 23, 23, fill=f"url(#{p}-r)", sw=HAIR)
    return g + rect(x, y, 2 * h, 2 * h, sw=HEAVY)


def _resize_art(p, cx, top):
    """Draw the storage of unp1 before and after: the 4 cells of Fig. 1, then 3 of them, 4 new."""
    arrow, s, y = f'marker-end="url(#{p}-a)"', 16, top + 20
    g = text(cx, y - 7, "old", anchor="middle", size=12, fill=INK, cls="sm-fig-math")
    for k in range(4):
        g += rect(cx - 2 * s + k * s, y, s, s, fill=f"url(#{p}-k)", sw=EMPTY)
    g += line(cx, y + s + 4, cx, y + 36, stroke=INK2, extra=arrow)
    for k in range(7):
        new = k > 2
        g += rect(
            cx - 3.5 * s + k * s,
            y + 40,
            s,
            s,
            fill=f"url(#{p}-{'r' if new else 'k'})",
            sw=HAIR if new else EMPTY,
        )
    return g + text(
        cx, y + 40 + s + 16, "new", anchor="middle", size=12, fill=INK, cls="sm-fig-math"
    )


def _scheme_art(p, cx, top):
    """Draw a coarse cell beside two fine ones and the fluxes across their face.

    At a level jump the scheme computes one flux per fine face, on the fine level:
    its stencil starts from a ghost of the fine level inside the coarse cell, in the
    half along the face (``for_each_interior_interface__level_jump_direction``,
    include/samurai/interface.hpp). The two ghosts take the place of that half: they
    are drawn dashed on the wash, the size of the fine cells, and the solid outline
    of the coarse cell runs around its ruled half only.
    """
    red_arrow = f'marker-end="url(#{p}-ar)"'
    x, y = cx - 42, top + 10
    g = rect(x, y, 28, 56, fill=f"url(#{p}-k)", stroke="none")
    g += rect(x + 28, y, 28, 56, fill=WASH, stroke="none")
    g += path(f"M{x + 28} {y} H{x} V{y + 56} H{x + 28}", stroke=INK)
    g += rect(x + 28, y, 28, 28, stroke=INK2, dash="3 2")
    g += rect(x + 28, y + 28, 28, 28, stroke=INK2, dash="3 2")
    g += rect(x + 56, y, 28, 28) + rect(x + 56, y + 28, 28, 28)
    g += line(x + 56, y - 6, x + 56, y + 62, stroke=RED, sw=2)
    g += line(x + 36, y + 14, x + 76, y + 14, stroke=RED, extra=red_arrow)
    g += line(x + 36, y + 42, x + 76, y + 42, stroke=RED, extra=red_arrow)
    g += text(x + 42, y + 80, "ghosts", anchor="middle", size=12, fill=INK, cls="sm-fig-math")
    return g + callout(x + 56, y - 6, x + 80, top - 12, "a", red=True)


def _swap_art(p, cx, top):
    """Draw the two fields exchanged."""
    red_arrow = f'marker-end="url(#{p}-ar)"'
    x, y = cx - 52, top + 12
    g = rect(x, y, 38, 22, fill=PAPER) + text(
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
    return g


@figure
def time_step(p):
    """tutorial/getting_started.md: the four statements of the time loop."""
    row = TIME_STEP_ROW
    flow = cycle(p, row, TIME_STEP_STATIONS, Loop("t += dt, and again until t = T = 0.3", 280))
    g = "".join(
        art(p, row.center(k), row.top)
        for k, art in enumerate((_adapt_art, _resize_art, _scheme_art, _swap_art))
    )
    return Drawing(
        PLATE_WIDTH,
        flow.height,
        "Four stations of one time step. Adapt: a coarse cell of four quadrants, three kept and"
        " ruled, one refined into four new cells hatched in red. Resize: the storage of unp1 goes"
        " from the four old cells to the three kept cells and the four new ones. Scheme: a coarse"
        " cell beside two fine cells, the face between them in red; two fine ghost cells, dashed,"
        " fill the half of the coarse cell along the face, and one flux arrow runs from each ghost"
        " to its fine cell across the face. Swap: u and unp1 are exchanged. An arrow returns from"
        " the last station to the first until t = 0.3.",
        patterns(p),
        g + flow.body,
        panels=flow.panels,
    )
