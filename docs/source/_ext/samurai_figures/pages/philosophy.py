"""The figures of ``philosophy.md``: intervals, a set expression, a flux, a detail.

The first three figures share one graded 1D mesh of [0, 1) with levels 3 to 5,
``MESH``. Its intervals, its counts, the set expression of the set algebra plate
and the cells each flux reads are those a samurai program prints for this mesh
(an ``mra::make_mesh`` built from the same ``CellList``, ``make_graduation``
leaving it unchanged, ``for_each_interior_interface__same_level`` and
``__level_jump_direction`` with a four-cell flux stencil).
"""

from math import tanh

from ..cells import from_intervals
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
    WASH,
    bracket,
    cell_row,
    fig_label,
    line,
    num,
    path,
    patterns,
    rect,
    text,
)
from ..plots import (
    Frame,
    axes,
    cell_averages,
    detail,
    level_threshold,
    line_key,
    parent_averages,
    predict_children,
    steps,
)
from ..registry import DIAGRAM_WIDTH, PLATE_WIDTH, Drawing, figure

# The rows (start, end) of each level, as the program adds them to its CellList.
MESH = {3: [(0, 3), (7, 8)], 4: [(6, 8), (12, 14)], 5: [(16, 24)]}


def check_graded_1d(leaves):
    """Raise ``ValueError`` unless the 1D cells ``leaves`` tile [0, 1) and are graded.

    A cell is ``(level, i)``, covering [i / 2^level, (i + 1) / 2^level). Two
    neighboring cells must differ by at most one level, as in every samurai mesh.
    """
    top = max(level for level, _ in leaves)
    fine = {}
    for level, i in leaves:
        s = 2 ** (top - level)
        for k in range(i * s, (i + 1) * s):
            if k in fine:
                raise ValueError(f"the cells overlap at the cell {k} of level {top}")
            fine[k] = level
    if sorted(fine) != list(range(2**top)):
        raise ValueError("the cells do not cover [0, 1) exactly once")
    for k in range(2**top - 1):
        if abs(fine[k] - fine[k + 1]) > 1:
            raise ValueError(f"the mesh is not graded at the cell {k + 1} of level {top}")


def intervals(leaves, level):
    """Return the intervals [a, b) of the cells of ``level``, one per run of indices."""
    ks = sorted(i for lv, i in leaves if lv == level)
    runs = []
    for k in ks:
        if runs and runs[-1][1] == k:
            runs[-1][1] = k + 1
        else:
            runs.append([k, k + 1])
    return [tuple(r) for r in runs]


def _leaves():
    """Return the cells of ``MESH``, checked."""
    leaves = from_intervals(MESH)
    check_graded_1d(leaves)
    return leaves


@figure
def graded_intervals(_p):
    """philosophy.md: a graded 1D mesh, its cells and its intervals, level by level."""
    leaves = _leaves()
    x0, w, top, ch, pitch = 74, 480, 34, 22, 60
    col_cells, col_intervals = 594, 648
    g = text(col_cells, 16, "cells", anchor="middle", size=11.5, fill=INK2)
    g += text(col_intervals, 16, "intervals", anchor="middle", size=11.5, fill=RED)
    # the mesh: every cell at its own size on one line
    for level, i in leaves:
        cw = w / 2**level
        g += rect(x0 + i * cw, top, cw, ch, fill=WASH, stroke=INK)
    g += text(x0 - 12, top + 15, "mesh", anchor="end", size=12, fill=INK)
    total_intervals = sum(len(intervals(leaves, level)) for level in MESH)
    g += text(col_cells, top + 15, len(leaves), anchor="middle", size=12.5, fill=INK)
    g += text(col_intervals, top + 15, total_intervals, anchor="middle", size=12.5, fill=RED)
    rule = top + ch + 16
    g += line(24, rule, DIAGRAM_WIDTH - 8, rule, stroke=FIG)
    # one row a level, in the grid of that level
    y = rule + 16
    for level in sorted(MESH):
        cw = w / 2**level
        runs = intervals(leaves, level)
        g += cell_row(x0, y, cw, 2**level, runs, ch=ch, brackets=False)
        for a, b in runs:
            g += bracket(x0 + a * cw, x0 + b * cw, y + ch + 9, f"[{a}, {b})", color=RED, size=11)
        g += text(x0 - 12, y + 15, f"level {level}", anchor="end", size=12, fill=INK)
        count = sum(b - a for a, b in runs)
        g += text(col_cells, y + 15, count, anchor="middle", size=12, fill=INK2)
        g += text(col_intervals, y + 15, len(runs), anchor="middle", size=12, fill=RED)
        y += pitch
    axis = y - 6
    g += line(x0, axis, x0 + w, axis)
    for v, label in ((0, "0"), (0.25, "1/4"), (0.5, "1/2"), (0.75, "3/4"), (1, "1")):
        g += line(x0 + v * w, axis, x0 + v * w, axis + 5)
        g += text(x0 + v * w, axis + 18, label, anchor="middle", size=11)
    g += text(x0 - 12, axis + 18, "x", anchor="end", size=13, fill=INK, cls="sm-fig-math")
    return Drawing(
        DIAGRAM_WIDTH,
        axis + 26,
        "A graded 1D mesh of the segment from 0 to 1, with 16 cells of levels 3, 4 and 5, drawn"
        " first on one line and then one row a level. Level 3 holds the intervals [0, 3) and"
        " [7, 8), 4 cells in 2 intervals; level 4 holds [6, 8) and [12, 14), 4 cells in 2"
        " intervals; level 5 holds [16, 24), 8 cells in 1 interval. The mesh has 16 cells and 5"
        " intervals.",
        "",
        g,
    )


def _guides(x0, cw, y1, y2):
    """Dotted red guides at the edges of the cell 2 of level 3, from y1 to y2."""
    return "".join(
        line(x0 + k * cw, y1, x0 + k * cw, y2, stroke=RED, sw=0.8, dash="2 3") for k in (2, 3)
    )


def _row_name(x, y, name, level, red=False):
    """Write the name of a row in the mono and its level under it, right-aligned at x."""
    g = text(x, y + 10, name, anchor="end", size=11.5, fill=RED if red else INK, cls="sm-fig-code")
    return g + text(x, y + 24, f"level {level}", anchor="end", size=11)


@figure
def right_neighbor_finer(p):
    """philosophy.md: the cells of level l whose right neighbor is at level l + 1, l = 3.

    The intervals are those the program prints for ``MESH``: translate gives
    [5, 7) and [11, 13) on level 4, the intersection [5, 6) on level 4, and its
    set projection [2, 3) on level 3.
    """
    leaves = _leaves()
    x0, w, ch = 168, 512, 22
    c3, c4 = w / 8, w / 16
    shifted = [(a - 1, b - 1) for a, b in intervals(leaves, 4)]
    inter = [(5, 6)]
    on_l = [(2, 3)]
    # the intersection of level 3 brought to level 4 and the shifted level 4
    fine3 = {k for a, b in intervals(leaves, 3) for k in range(2 * a, 2 * b)}
    moved = {k for a, b in shifted for k in range(a, b)}
    if sorted(fine3 & moved) != [5] or sorted({k // 2 for k in fine3 & moved}) != [2]:
        raise ValueError("the set expression does not give the cell 2 of level 3")
    centre = x0 + w / 2
    g = ""
    panels = []

    # Fig. 1: the operands
    y1, y2 = 30, 88
    g += _guides(x0, c3, y1 - 8, y2 + ch + 4)
    g += cell_row(x0, y1, c3, 8, intervals(leaves, 3), ch=ch, label_size=11)
    g += cell_row(x0, y2, c4, 16, intervals(leaves, 4), ch=ch, label_size=11)
    g += _row_name(x0 - 14, y1, "cells[l]", 3) + _row_name(x0 - 14, y2, "cells[l + 1]", 4)
    g += fig_label(centre, 160, 1)
    g += text(centre, 178, "cells[l], cells[l + 1]", anchor="middle", size=10.5, cls="sm-fig-code")
    panels.append((0, 0, PLATE_WIDTH, 190, "wide"))

    # Fig. 2: level l + 1 moved one cell to the left
    y = 230
    g += _guides(x0, c3, y - 18, y + ch + 4)
    for a, b in intervals(leaves, 4):
        g += rect(x0 + a * c4, y, (b - a) * c4, ch, stroke=INK3, sw=RULE, dash="3 3")
    g += cell_row(x0, y, c4, 16, shifted, ch=ch, label_size=11)
    for a, b in intervals(leaves, 4):
        # the arrow runs over the cell that the interval leaves behind
        xa = x0 + b * c4 - 4
        g += line(xa, y - 9, xa - c4 + 8, y - 9, stroke=INK2, extra=f'marker-end="url(#{p}-a)"')
    g += _row_name(x0 - 14, y, "translate", 4)
    g += fig_label(centre, 302, 2)
    g += text(
        centre, 320, "translate(cells[l + 1], {-1})", anchor="middle", size=10.5, cls="sm-fig-code"
    )
    panels.append((0, 196, PLATE_WIDTH, 136, "wide"))

    # Fig. 3: the intersection on level l + 1, then its set projection on level l
    y1, y2 = 362, 420
    g += _guides(x0, c3, y1 - 8, y2 + ch + 4)
    g += cell_row(x0, y1, c4, 16, inter, ch=ch, tone="red", label_size=11)
    g += cell_row(x0, y2, c3, 8, on_l, ch=ch, tone="red", label_size=11)
    for a, b in on_l:
        g += rect(x0 + a * c3, y2, (b - a) * c3, ch, fill=f"url(#{p}-r)", stroke=RED, sw=HAIR)
    g += _row_name(x0 - 14, y1, "intersection", 4, red=True)
    g += _row_name(x0 - 14, y2, ".on(l)", 3, red=True)
    g += fig_label(centre, 492, 3)
    g += text(
        centre,
        510,
        "intersection(cells[l], translate(cells[l + 1], {-1})).on(l)",
        anchor="middle",
        size=10.5,
        fill=RED,
        cls="sm-fig-code",
    )
    panels.append((0, 338, PLATE_WIDTH, 184, "wide"))
    return Drawing(
        PLATE_WIDTH,
        522,
        "Three figures on the cells of levels 3 and 4 of a 1D mesh, with l = 3. Figure 1: level 3"
        " holds [0, 3) and [7, 8), level 4 holds [6, 8) and [12, 14). Figure 2: level 4 translated"
        " one cell to the left gives [5, 7) and [11, 13), drawn over the dashed outline of the"
        " original cells. Figure 3, in red: the intersection of level 3 with the translated level 4"
        " is the cell [5, 6) of level 4, and brought to level 3 with on(l) it is the cell [2, 3),"
        " the only cell of level 3 whose right neighbor is at level 4. Dotted red guides mark the"
        " edges of the cell 2 of level 3.",
        patterns(p),
        g,
        panels=tuple(panels),
    )


@figure
def level_jump_flux(_p):
    """philosophy.md: one flux at a same-level face and at a level jump.

    The window is [1/8, 3/4) of ``MESH``. The program reports the four cells the
    flux reads: 18 to 21 of level 5 at the face between the cells 19 and 20 of
    level 5, and 4 to 7 of level 4 at the face between the cell 2 of level 3 and
    the cell 6 of level 4, where 4 and 5 are prediction ghosts.
    """
    leaves = _leaves()
    lo, hi = 1 / 8, 3 / 4
    x0, w, ch = 116, 548, 24
    unit = w / (hi - lo)

    def X(u):
        return x0 + (u - lo) * unit

    def cell(level, i):
        return X(i / 2**level), unit / 2**level

    row1, row2 = 44, 108
    g = ""
    for level, i in leaves:
        x, cw = cell(level, i)
        if x < X(lo) - 1e-9 or x + cw > X(hi) + 1e-9:
            continue
        g += rect(x, row1, cw, ch, fill=WASH, stroke=INK)
        g += text(x + cw / 2, row1 + 16, i, anchor="middle", size=10, fill=INK2, cls="sm-fig-code")
    for level, (a, _) in ((3, (1, 3)), (4, (6, 8)), (5, (16, 24))):
        g += text(cell(level, a)[0] + 8, row1 - 8, f"level {level}", size=11)
    g += text(x0 - 12, row1 + 16, "mesh", anchor="end", size=12, fill=INK)
    g += text(x0 - 12, row2 + 16, "read by", anchor="end", size=12, fill=INK)
    g += text(x0 - 12, row2 + 31, "the flux", anchor="end", size=12, fill=INK)

    stencils = (
        ("jump", 4, (4, 5, 6, 7), (4, 5), 3 / 8),
        ("same", 5, (18, 19, 20, 21), (), 20 / 32),
    )
    for _, level, ks, ghosts, face in stencils:
        for k in ks:
            x, cw = cell(level, k)
            if k in ghosts:
                g += rect(x, row2, cw, ch, fill=PAPER, stroke=INK3, sw=RULE, dash="3 2")
                tone = INK3
            else:
                g += rect(x, row2, cw, ch, fill=WASH, stroke=INK)
                tone = INK2
            g += text(
                x + cw / 2, row2 + 16, k, anchor="middle", size=10, fill=tone, cls="sm-fig-code"
            )
        # a brace under the four cells, with the level the flux is computed on
        xa, xb = cell(level, ks[0])[0], sum(cell(level, ks[-1]))
        yb = row2 + ch + 6
        g += path(
            f"M{num(xa + 1)} {num(yb)}v4H{num(xb - 1)}v-4M{num(X(face))} {num(yb + 4)}v6",
            stroke=INK3,
            sw=HAIR,
        )
        g += text(X(face), yb + 24, f"computed on level {level}", anchor="middle", size=11.5)
        # the face, through both rows
        g += line(X(face), row1 - 22, X(face), row2 + ch + 2, stroke=RED, sw=HEAVY)
        g += text(X(face), row1 - 27, "F", anchor="middle", size=15, fill=RED, cls="sm-fig-math")
    xg = cell(4, 4)[0]
    g += text(xg - 8, row2 + 11, "prediction", anchor="end", size=11, fill=INK2)
    g += text(xg - 8, row2 + 25, "ghosts", anchor="end", size=11, fill=INK2)
    return Drawing(
        DIAGRAM_WIDTH,
        row2 + ch + 40,
        "A window of a graded 1D mesh with cells 1 and 2 of level 3, 6 and 7 of level 4 and 16 to"
        " 23 of level 5. Two faces are marked in red, each with its flux F. At the face between"
        " the cell 2 of level 3 and the cell 6 of level 4, the flux reads the cells 4, 5, 6 and 7"
        " of level 4 and is computed on level 4; the cells 4 and 5, dashed, are prediction ghosts"
        " that fill the coarse cell 2. At the face between the cells 19 and 20 of level 5, the"
        " same flux reads the cells 18 to 21 of level 5 and is computed on level 5.",
        "",
        g,
    )


def front(x):
    """Return the field of the detail diagram at x: a smooth front from 0 to 1 at 0.43."""
    return 0.5 * (1 + tanh((x - 0.43) / 0.01))


@figure
def detail_threshold(p):
    """philosophy.md: predicted values, details and the threshold, on levels 3 and 4.

    The values are the cell averages of ``front`` on level 4 and their means on
    level 3; the prediction is samurai's (``predict_children``) and the threshold
    is ``eps_l`` of include/samurai/mr/adapt.hpp for epsilon 0.08, children on
    level 4 and a maximum level of 5.
    """
    children = cell_averages(front, 16, samples=400)
    parents = parent_averages(children)
    eps = level_threshold(0.08, 4, 5)
    fr = Frame(70, 34, 580, 240, xlim=(0, 5 / 8), ylim=(-0.1, 1.1))
    h = 1 / 16
    g = steps(fr, [(k / 8, (k + 1) / 8, parents[k]) for k in range(5)], sw=HEAVY, joined=False)
    outcome = {}
    for k in (1, 3):
        low, high = predict_children(parents[k - 1], parents[k], parents[k + 1])
        details = []
        for c, predicted in ((0, low), (1, high)):
            a, b = (2 * k + c) * h, (2 * k + c + 1) * h
            real = children[2 * k + c]
            details.append(real - predicted)
            band_top, band_bottom = fr.Y(predicted + eps), fr.Y(predicted - eps)
            g += rect(
                fr.X(a) + 4,
                band_top,
                fr.X(b) - fr.X(a) - 8,
                band_bottom - band_top,
                fill=WASH,
                stroke="none",
            )
            if fr.Y(predicted) - fr.Y(real) > 14 or fr.Y(real) - fr.Y(predicted) > 14:
                label = f'd<tspan baseline-shift="sub" font-size="10">{c + 1}</tspan>'
                g += detail(fr, a, b, predicted, real, label, p)
            else:
                g += line(
                    fr.X(a) + 4,
                    fr.Y(predicted),
                    fr.X(b) - 4,
                    fr.Y(predicted),
                    stroke=INK2,
                    dash="4 3",
                )
                g += line(fr.X(a) + 4, fr.Y(real), fr.X(b) - 4, fr.Y(real), stroke=RED, sw=HEAVY)
        if abs(details[0] + details[1]) > 1e-12:
            raise ValueError("the two details of a cell must be opposite")
        outcome[k] = abs(details[0]) > eps
        # the boundary between the two children, on the axis
        g += line(fr.X((2 * k + 1) * h), fr.bottom, fr.X((2 * k + 1) * h), fr.bottom + 4, sw=EMPTY)
    if outcome != {1: False, 3: True}:
        raise ValueError("the details should stay under the threshold at 1 and exceed it at 3")
    g += axes(
        fr,
        [(0, "0"), (1 / 8, "1/8"), (1 / 4, "1/4"), (3 / 8, "3/8"), (1 / 2, "1/2"), (5 / 8, "5/8")],
        [(0, "0"), (1, "1")],
        yname="u",
    )
    g += text(fr.X(3 / 16), fr.bottom + 38, "below the threshold:", anchor="middle", size=11.5)
    g += text(fr.X(3 / 16), fr.bottom + 53, "the children are merged", anchor="middle", size=11.5)
    g += text(
        fr.X(7 / 16), fr.bottom + 38, "above the threshold:", anchor="middle", size=11.5, fill=RED
    )
    g += text(
        fr.X(7 / 16), fr.bottom + 53, "the children are kept", anchor="middle", size=11.5, fill=RED
    )
    kx, ky = fr.x + 18, fr.y + 8
    g += line_key(
        kx,
        ky,
        [
            ("value of a cell, level 3", INK, HEAVY, ""),
            ("value predicted for a child, level 4", INK2, RULE, "4 3"),
            ("value of the child, level 4", RED, HEAVY, ""),
        ],
    )
    g += rect(kx, ky + 3 * 22 - 6, 22, 12, fill=WASH, stroke="none")
    g += text(
        kx + 30,
        ky + 3 * 22 + 4,
        'threshold, predicted value ± ε<tspan baseline-shift="sub" font-size="9">4</tspan>',
        size=12,
        fill=INK2,
    )
    return Drawing(
        DIAGRAM_WIDTH,
        fr.bottom + 62,
        "The values of a field u that rises from 0 to 1 in a sharp front near x = 0.43, on five"
        " cells of level 3 from 0 to 5/8, drawn as heavy bars. For two of them the children of"
        " level 4 are drawn: a dashed bar for the value predicted from the cell and its two"
        " neighbors, a red bar for the real value, and a grey band of half-height epsilon_4 = 0.04"
        " around the prediction. On the flat cell from 1/8 to 1/4 the prediction is exact, the"
        " details are zero and the children are merged. On the cell of the front, from 3/8 to"
        " 1/2, the real values lie far outside the band, the details d1 and d2 are opposite, and"
        " the children are kept.",
        patterns(p),
        g,
    )
