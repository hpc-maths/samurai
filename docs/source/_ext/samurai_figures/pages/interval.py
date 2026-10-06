"""The figures of ``tutorial/interval.md``: from cells to intervals, and how a level is stored.

The 1D mesh is the one of ``snippet/interval/interval_1d.cpp`` and the 2D mesh the one
of ``demos/tutorial/2D_mesh.cpp``; their intervals, indices and offsets are those the
two programs print (``interval_1d_output.txt`` and ``2d_mesh_representation_output.txt``).
"""

from ..arrays import Array, connector, draw_array, interval_token, token_row
from ..cells import Frame, draw_cells, from_intervals
from ..draw import (
    EMPTY,
    FIG,
    HAIR,
    HEAVY,
    INK,
    INK2,
    RED,
    RWASH,
    WASH,
    bracket,
    callout,
    cell_row,
    fig_label,
    line,
    patterns,
    rect,
    text,
)
from ..registry import DIAGRAM_WIDTH, PLATE_WIDTH, Drawing, figure

# The CellList of interval_1d.cpp: level -> intervals (start, end).
MESH_1D = {0: [(0, 2), (5, 6)], 1: [(4, 7), (8, 10)], 2: [(14, 16)]}

# The CellList of 2D_mesh.cpp: level -> rows (y, start, end).
MESH_2D = {
    0: [(0, 0, 4), (1, 0, 1), (1, 3, 4), (2, 0, 1), (2, 3, 4), (3, 0, 3)],
    1: [(2, 2, 6), (3, 2, 6), (4, 2, 4), (4, 5, 6), (5, 2, 6), (6, 6, 8), (7, 6, 7)],
    2: [(8, 8, 10), (9, 8, 10), (14, 14, 16), (15, 14, 16)],
}

# The level cell arrays of 2D_mesh.cpp along y, as the program prints them:
# the y intervals with their index, the y-offset array and the x intervals.
ARRAYS_2D = {
    1: {
        "y": [(2, 8, -2)],
        "offsets": [0, 1, 2, 4, 5, 6, 7],
        "x": [(2, 6), (2, 6), (2, 4), (5, 6), (2, 6), (6, 8), (6, 7)],
    },
    2: {
        "y": [(8, 10, -8), (14, 16, -12)],
        "offsets": [0, 1, 2, 3, 4],
        "x": [(8, 10), (8, 10), (14, 16), (14, 16)],
    },
}


def _axis(x0, unit, y, ticks, name="x"):
    """A number line under rows of cells, with a tick and a label at each value of ``ticks``.

    ``ticks`` are ``(value, label)``; ``unit`` is the drawing width of one unit of value.
    """
    g = line(x0 + ticks[0][0] * unit, y, x0 + ticks[-1][0] * unit, y)
    for v, label in ticks:
        g += line(x0 + v * unit, y, x0 + v * unit, y + 5)
        g += text(x0 + v * unit, y + 18, label, anchor="middle", size=11.5)
    return g + text(x0 - 14, y + 4, name, anchor="end", size=13, fill=INK, cls="sm-fig-math")


def _cells(x0, unit, y, cells, ch=24, red=False):
    """Cells ``(a, b)`` of real bounds, drawn at their true width on the row y."""
    fill, stroke = (RWASH, RED) if red else (WASH, INK)
    return "".join(
        rect(x0 + a * unit, y, (b - a) * unit, ch, fill=fill, stroke=stroke) for a, b in cells
    )


def _real(v):
    """A real bound of the introduction, written without a trailing .0."""
    return f"{v:g}"


@figure
def cells_of_a_line(_p):
    """tutorial/interval.md: five cells of real bounds on the segment [0, 13)."""
    x0, unit, y = 60, 42, 30
    cells = [(0, 4), (4, 5), (5, 7), (7, 9), (9, 13)]
    g = _cells(x0, unit, y, cells, ch=26)
    for k, (a, b) in enumerate(cells):
        g += text(x0 + (a + b) / 2 * unit, y + 18, k + 1, anchor="middle", size=12, fill=INK)
        # neighbours share a bound: their brackets alternate on two lines
        yb = y + 26 + 12 + (k % 2) * 34
        g += bracket(x0 + a * unit, x0 + b * unit, yb, f"[{a}, {b})", size=11)
    axis = y + 26 + 12 + 34 + 34
    ticks = [(v, _real(v)) for v in (0, 4, 5, 7, 9, 13)]
    g += _axis(x0, unit, axis, ticks)
    g += text(x0 - 14, y + 18, "cell", anchor="end", size=12, fill=INK)
    return Drawing(
        DIAGRAM_WIDTH,
        axis + 26,
        "A segment from 0 to 13 cut into five cells, numbered 1 to 5 from left to right:"
        " [0, 4), [4, 5), [5, 7), [7, 9) and [9, 13). Each cell has a half-open bracket under"
        " it, and the brackets of neighbouring cells alternate on two lines.",
        "",
        g,
    )


def _width_rows(x0, unit, top, pitch, rows):
    """Rows of cells by width, each labelled on the left, with the bracket of each interval.

    ``rows`` are ``(width, cells, intervals, red)``: the cells of that width, the
    intervals they merge into, and whether the brackets are red.
    """
    g = ""
    for r, (width, cells, intervals, red) in enumerate(rows):
        y = top + r * pitch
        g += _cells(x0, unit, y, cells)
        g += text(x0 - 14, y + 16, f"width {width}", anchor="end", size=12, fill=INK)
        for a, b in intervals:
            g += bracket(
                x0 + a * unit,
                x0 + b * unit,
                y + 24 + 9,
                f"[{a}, {b})",
                color=RED if red else INK,
                size=11,
            )
    return g


@figure
def cells_by_width(_p):
    """tutorial/interval.md: the five cells of ``cells_of_a_line`` sorted by width."""
    x0, unit, top, pitch = 90, 42, 14, 62
    rows = [
        (1, [(4, 5)], [(4, 5)], False),
        (2, [(5, 7), (7, 9)], [(5, 9)], True),
        (4, [(0, 4), (9, 13)], [(0, 4), (9, 13)], False),
    ]
    g = _width_rows(x0, unit, top, pitch, rows)
    axis = top + 3 * pitch
    g += _axis(x0, unit, axis, [(v, _real(v)) for v in (0, 4, 5, 7, 9, 13)])
    return Drawing(
        DIAGRAM_WIDTH,
        axis + 26,
        "The five cells of the segment [0, 13) on three rows, one per cell width. Width 1 holds"
        " the cell [4, 5). Width 2 holds the cells [5, 7) and [7, 9), merged into one interval"
        " [5, 9) drawn in red. Width 4 holds the cells [0, 4) and [9, 13).",
        "",
        g,
    )


@figure
def overlapping_widths(_p):
    """tutorial/interval.md: cells of three widths that cover the same parts of [0, 13)."""
    x0, unit, top, pitch = 90, 42, 14, 62
    rows = [
        (1, [(1, 2), (2, 3), (10, 11), (11, 12)], [(1, 3), (10, 12)], False),
        (2, [(3, 5), (5, 7), (7, 9)], [(3, 9)], False),
        (4, [(0, 4), (9, 13)], [(0, 4), (9, 13)], False),
    ]
    g = _width_rows(x0, unit, top, pitch, rows)
    # the parts of the segment covered by two cells, under a rule
    y = top + 3 * pitch + 6
    g += line(24, y - 12, DIAGRAM_WIDTH - 8, y - 12, stroke=FIG)
    for a, b in ((1, 4), (10, 12)):
        g += bracket(x0 + a * unit, x0 + b * unit, y + 6, f"[{a}, {b})", color=RED, size=11)
    g += text(x0 - 14, y + 10, "covered twice", anchor="end", size=12, fill=RED)
    axis = y + 40
    g += _axis(x0, unit, axis, [(v, _real(v)) for v in (0, 1, 3, 4, 9, 10, 12, 13)])
    return Drawing(
        DIAGRAM_WIDTH,
        axis + 26,
        "Cells of three widths on the segment [0, 13). Width 1 holds [1, 3) and [10, 12), width 2"
        " holds [3, 9), width 4 holds [0, 4) and [9, 13). In red under a rule, the parts covered"
        " by two cells: [1, 4) and [10, 12).",
        "",
        g,
    )


@figure
def printed_interval(_p):
    """tutorial/interval.md: the interval of level 2 that interval_1d.cpp prints."""
    parts = interval_token(14, 16, -6, 1, position=0, red=("index",))
    g = token_row(DIAGRAM_WIDTH / 2, 36, parts, size=24, anchor="middle", label_size=12)
    return Drawing(
        DIAGRAM_WIDTH,
        96,
        "The printed interval 0->[14,16)@-6:1 with a label under each part: 0 is the position,"
        " 14 the start, 16 the end, -6 the index, in red, and 1 the step.",
        "",
        g,
    )


@figure
def interval_on_two_levels(_p):
    """tutorial/interval.md: the same part of the line as an interval of two levels."""
    x0, unit, top, pitch = 200, 140, 14, 62
    g = ""
    rows = [("l", 2), ("l + 1", 4)]
    for r, (level, n) in enumerate(rows):
        y = top + r * pitch
        cw = 2 * unit / n
        g += cell_row(x0, y, cw, n, [(0, n)], ch=24, label_size=11)
        for k in range(n):
            g += text(x0 + (k + 0.5) * cw, y + 16, k, anchor="middle", size=12, fill=INK)
        g += text(
            x0 - 14,
            y + 16,
            f'level <tspan class="sm-fig-math">{level}</tspan>',
            anchor="end",
            size=12,
            fill=INK,
        )
    axis = top + 2 * pitch
    ticks = [(0, "0"), (1, "Δx"), (2, "2Δx")]
    g += line(x0, axis, x0 + 2 * unit, axis)
    for v, label in ticks:
        g += line(x0 + v * unit, axis, x0 + v * unit, axis + 5)
        g += text(
            x0 + v * unit, axis + 19, label, anchor="middle", size=13, fill=INK2, cls="sm-fig-math"
        )
    g += text(x0 - 14, axis + 4, "x", anchor="end", size=13, fill=INK, cls="sm-fig-math")
    return Drawing(
        DIAGRAM_WIDTH,
        axis + 28,
        "The segment from 0 to 2 Δx, where Δx is the cell length of level l. At level l it holds"
        " the cells 0 and 1, the interval [0, 2). At level l + 1, drawn at true size under it, the"
        " same segment holds the cells 0 to 3, the interval [0, 4).",
        "",
        g,
    )


@figure
def nested_levels(_p):
    """tutorial/interval.md: a cell of level l + 2 inside a cell of level l, and one across."""
    unit, top, pitch = 112, 26, 54  # a cell of level l is 112 units wide
    lefts = (120, 412)
    fine = unit / 4
    g = ""
    for q, x0 in enumerate(lefts):
        # level l: the cells 0 and 1, the cell 0 is the one compared against
        g += cell_row(x0, top, unit, 2, [(0, 1)], ch=24, brackets=False)
        for k in (0, 1):
            g += text(x0 + (k + 0.5) * unit, top + 16, k, anchor="middle", size=12, fill=INK)
        y = top + pitch
        for k in range(8):
            g += rect(x0 + k * fine, y, fine, 24, stroke=FIG, sw=EMPTY)
        # the edges of the cell 0 of level l, down through the row of level l + 2
        for k in (0, 1):
            g += line(x0 + k * unit, top - 6, x0 + k * unit, y + 24, stroke=RED, sw=0.8, dash="2 3")
        if q == 0:
            g += rect(x0 + 3 * fine, y, fine, 24, fill=WASH, stroke=INK)
            g += bracket(x0 + 3 * fine, x0 + 4 * fine, y + 33, "[3, 4)", size=11)
        else:
            xa = x0 + 3.5 * fine
            g += rect(xa, y, fine, 24, fill=WASH, stroke=INK)
            g += line(xa - 8, y - 8, xa + fine + 8, y + 32, stroke=RED, sw=HEAVY)
            g += line(xa - 8, y + 32, xa + fine + 8, y - 8, stroke=RED, sw=HEAVY)
            g += text(
                xa + fine / 2, y + 52, "across the edge", anchor="middle", size=11.5, fill=RED
            )
    for r, level in enumerate(("l", "l + 2")):
        g += text(
            lefts[0] - 14,
            top + r * pitch + 16,
            f'level <tspan class="sm-fig-math">{level}</tspan>',
            anchor="end",
            size=12,
            fill=INK,
        )
    return Drawing(
        DIAGRAM_WIDTH,
        top + pitch + 62,
        "Two drawings of the cells 0 and 1 of level l, with the 8 cells of level l + 2 under them"
        " at true size. Left, the cell [3, 4) of level l + 2 lies inside the cell [0, 1) of level"
        " l. Right, a cell of level l + 2 shifted by half its width crosses the edge between the"
        " two cells of level l; it is crossed out in red.",
        "",
        g,
    )


@figure
def three_levels_1d(_p):
    """tutorial/interval.md: the mesh of interval_1d.cpp and its intervals level by level."""
    x0, unit, top, pitch = 100, 92, 14, 62
    # the mesh: the cells of every level side by side, at true size
    g = ""
    for level, intervals in MESH_1D.items():
        cw = unit / 2**level
        for a, b in intervals:
            for k in range(a, b):
                g += rect(x0 + k * cw, top, cw, 24, fill=WASH, stroke=INK)
    g += rect(x0, top, 6 * unit, 24, stroke=INK, sw=HEAVY)
    g += text(x0 - 14, top + 16, "mesh", anchor="end", size=12, fill=INK)
    g += line(24, top + 40, DIAGRAM_WIDTH - 8, top + 40, stroke=FIG)
    for level, intervals in MESH_1D.items():
        y = top + 56 + level * pitch
        cw = unit / 2**level
        g += cell_row(x0, y, cw, 6 * 2**level, intervals, ch=24, label_size=11)
        for a, b in intervals:
            for k in range(a, b):
                g += text(
                    x0 + (k + 0.5) * cw,
                    y + 16,
                    k,
                    anchor="middle",
                    size=10.5 if level else 12,
                    fill=INK,
                )
        g += text(x0 - 14, y + 16, f"level {level}", anchor="end", size=12, fill=INK)
    axis = top + 56 + 3 * pitch
    g += _axis(x0, unit, axis, [(v, str(v)) for v in range(7)])
    return Drawing(
        DIAGRAM_WIDTH,
        axis + 26,
        "The 1D mesh of interval_1d.cpp over x from 0 to 6, then its cells on one row per level"
        " at true size, each numbered and with the half-open bracket of its interval. Level 0"
        " holds [0, 2) and [5, 6), level 1 holds [4, 7) and [8, 10), level 2 holds [14, 16)."
        " The cells of the three levels fill the segment without overlapping.",
        "",
        g,
    )


def _index_scale(x, y, size, n, horizontal):
    """The indices 0 to n - 1 of one level along a side of the mesh, in a row of boxes."""
    step = size / n
    g = ""
    for k in range(n):
        if horizontal:
            bx, by, w, h = x + k * step, y, step, 18
        else:
            bx, by, w, h = x, y + size - (k + 1) * step, 22, step
        g += rect(bx, by, w, h, stroke=FIG, sw=EMPTY)
        g += text(bx + w / 2, by + h / 2 + 3.6, k, anchor="middle", size=10, fill=INK2)
    return g


@figure
def mesh_2d(p):
    """tutorial/interval.md: the mesh of 2D_mesh.cpp, with the indices of each level."""
    size, ox, oy = 320, 250, 30
    frame = Frame((0, 0, 4, 4), ox, oy, size)
    g = draw_cells(p, from_intervals(MESH_2D), frame)
    # the indices of levels 0, 1 and 2, the finest next to the mesh
    for level in range(3):
        n = 4 * 2**level
        g += _index_scale(ox, oy + size + 8 + (2 - level) * 22, size, n, True)
        g += _index_scale(ox - 8 - 22 - (2 - level) * 26, oy, size, n, False)
        g += text(
            ox - 12, oy + size + 8 + (2 - level) * 22 + 13, f"level {level}", anchor="end", size=11
        )
        g += text(ox - 8 - 11 - (2 - level) * 26, oy - 8, level, anchor="middle", size=11)
    g += text(ox - 8 - 22 - 2 * 26 - 8, oy - 8, "level", anchor="end", size=11)
    return Drawing(
        DIAGRAM_WIDTH,
        oy + size + 8 + 3 * 22 + 4,
        "The 2D mesh of 2D_mesh.cpp on the square from 0 to 4. Cells of level 0 fill the edges"
        " of the square, cells of level 1 a block in the middle and the upper right corner, and"
        " two blocks of four cells of level 2, hatched in red, sit in the middle and in the upper"
        " right corner. Under the mesh and to its left, the cell indices of levels 0, 1 and 2:"
        " 0 to 3, 0 to 7 and 0 to 15.",
        patterns(p),
        g,
    )


def _level_arrays(p, level, x, top, row):
    """The y, y-offset and x arrays of a level of 2D_mesh.cpp, with the lookup of ``row``.

    Return the drawing, the bottom of the x array's positions and the centre of the x array.
    """
    data = ARRAYS_2D[level]
    ys, offsets, xs = data["y"], data["offsets"], data["x"]
    # the y interval that holds the row, and the place of the row in y-offset
    k, (_, _, index) = next((k, iv) for k, iv in enumerate(ys) if iv[0] <= row < iv[1])
    j = row + index
    ylabels = [f"[{a},{b})@{i}" for a, b, i in ys]
    xlabels = [f"[{a},{b})" for a, b in xs]
    wy = 12 + max(len(s) for s in ylabels) * 6.6
    wx = 8 + max(len(s) for s in xlabels) * 6.6
    ya = Array(x, top, wy, len(ys))
    yo = Array(x, top + 64, 30, len(offsets))
    xa = Array(x, top + 140, wx, len(xs))
    arrow = f"{p}-ar"
    g = draw_array(ya, ylabels, red=(k,), name="y")
    # the lookup, next to the connector it labels
    g += text(
        max(ya.cx(k), yo.cx(j)) + 16,
        (ya.bottom + 20 + yo.y) / 2 + 4,
        f"{row} + ({index}) = {j}",
        size=10.5,
        fill=RED,
        cls="sm-fig-code",
    )
    g += connector(ya.cx(k), ya.bottom + 20, yo.cx(j), yo.y, red=True, arrow=arrow)
    g += draw_array(yo, offsets, red=(j, j + 1), name="y-offset")
    # the x intervals of the row: from y-offset[j] to y-offset[j + 1], excluded
    first, end = offsets[j], offsets[j + 1]
    g += connector(yo.cx(j), yo.bottom + 20, xa.left(first) + 3, xa.y, red=True, arrow=arrow)
    g += connector(yo.cx(j + 1), yo.bottom + 20, xa.left(end) - 3, xa.y, red=True, arrow=arrow)
    g += draw_array(xa, xlabels, red=range(first, end), name="x")
    return g, xa.bottom + 14, (xa.x + xa.right) / 2


@figure
def level_lookup(p):
    """tutorial/interval.md: the mesh of 2D_mesh.cpp and the lookup of a row in two levels."""
    size, ox, oy = 250, 36, 92
    frame = Frame((0, 0, 4, 4), ox, oy, size)
    rows = {(1, i, 4) for i in (2, 3, 5)} | {(2, i, 14) for i in (14, 15)}
    g = draw_cells(
        p,
        from_intervals(MESH_2D),
        frame,
        hatch=False,
        fill=lambda leaf: RWASH if leaf in rows else None,
        xticks=[(v, str(v)) for v in range(5)],
        yticks=[(v, str(v)) for v in range(5)],
    )
    for leaf in sorted(rows):
        g += rect(*frame.cell(leaf), stroke=RED, sw=HAIR)
    g += callout(*frame.center((1, 2, 4)), frame.x(0.5), frame.y(2.25), "a", red=True)
    g += callout(*frame.center((2, 15, 14)), frame.x(4) + 18, frame.y(3.625), "b", red=True)
    g += fig_label(ox + size / 2, oy + size + 52, 1)
    x = 396
    fig2, bottom2, center2 = _level_arrays(p, 1, x, 16, 4)
    g += fig2 + fig_label(center2, bottom2 + 32, 2, "level 1")
    top3 = bottom2 + 58
    fig3, bottom3, center3 = _level_arrays(p, 2, x, top3, 14)
    g += fig3 + fig_label(center3, bottom3 + 32, 3, "level 2")
    height = bottom3 + 44
    return Drawing(
        PLATE_WIDTH,
        height,
        "Left, the 2D mesh of 2D_mesh.cpp with two rows of cells in red: a, the row y = 4 of"
        " level 1, cells 2, 3 and 5; b, the row y = 14 of level 2, cells 14 and 15. Right, the"
        " arrays of level 1: y holds [2,8)@-2, so the row y = 4 is read at y-offset[4 - 2] ="
        " y-offset[2]; y-offset[2] = 2 and y-offset[3] = 4 give the entries 2 and 3 of x,"
        " [2,4) and [5,6). Below, the arrays of level 2: y holds [8,10)@-8 and [14,16)@-12, the"
        " row y = 14 is read at y-offset[14 - 12] = y-offset[2]; y-offset[2] = 2 and"
        " y-offset[3] = 3 give the entry 2 of x, [14,16).",
        patterns(p),
        g,
        panels=(
            (0, oy - 20, 320, size + 90, ""),
            (310, 0, 401, top3 - 10, ""),
            (310, top3 - 10, 401, height - top3 + 10, ""),
        ),
    )
