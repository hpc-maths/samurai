"""The figures of ``tutorial/operator_on_subset.md``, from ``demos/tutorial/set_operator.cpp``.

The mesh of the program has two levels on the cells [0, 10) of level 0: level 0
holds [0, 10), level 1 holds [2, 6) and [11, 15). Every interval and value drawn
here is one the program prints.
"""

from ..draw import (
    HAIR,
    HEAVY,
    INK,
    INK2,
    RED,
    RULE,
    RWASH,
    WASH,
    callout,
    cell_row,
    fig_label,
    line,
    patterns,
    rect,
    text,
)
from ..registry import DIAGRAM_WIDTH, PLATE_WIDTH, Drawing, figure

# The two levels of the mesh, as half-open intervals of cells.
LEVEL0 = [(0, 10)]
LEVEL1 = [(2, 6), (11, 15)]
# The field u after the field projection: level 1 holds its index along x, the
# cells 1, 2 and 6 of level 0 the mean of their two children, the others 0.
U1 = {k: k for a, b in LEVEL1 for k in range(a, b)}
U0 = {k: 0 for k in range(10)} | {1: 2.5, 2: 4.5, 6: 12.5}


def _axis(x0, y, cw, n):
    """The level-0 number line under the rows, with the indices 0 to n."""
    g = line(x0, y, x0 + n * cw, y)
    for k in range(n + 1):
        g += line(x0 + k * cw, y, x0 + k * cw, y + 5)
        g += text(x0 + k * cw, y + 18, k, anchor="middle", size=11.5)
    return g + text(x0 - 14, y + 18, "level 0", anchor="end", size=11.5)


def _row_name(x, y, name, level):
    """The name of a row of results, its expression in red mono over its level."""
    return text(x, y + 11, name, anchor="end", size=11.5, fill=RED, cls="sm-fig-code") + text(
        x, y + 25, level, anchor="end", size=11
    )


@figure
def two_level_mesh(_p):
    """Draw the two levels of the mesh of set_operator.cpp at their true size."""
    x0, cw, top, pitch = 80, 28, 14, 70  # a level-1 cell is 28 units wide
    g = cell_row(x0, top, cw, 20, LEVEL1, ch=24)
    g += cell_row(x0, top + pitch, 2 * cw, 10, LEVEL0, ch=24)
    for y, name in ((top, "level 1"), (top + pitch, "level 0")):
        g += text(x0 - 14, y + 17, name, anchor="end", size=12, fill=INK)
    axis = top + 2 * pitch + 2
    g += _axis(x0, axis, 2 * cw, 10)
    return Drawing(
        DIAGRAM_WIDTH,
        axis + 26,
        "Two levels of a 1D mesh, drawn at their true size over the cells 0 to 9 of level 0."
        " Level 1 holds two intervals, [2, 6) and [11, 15). Level 0 holds the ten cells [0, 10),"
        " each one twice as wide as a cell of level 1, so that the cell 1 of level 0 covers the"
        " cells 2 and 3 of level 1.",
        "",
        g,
    )


def _values(x0, y, cw, values, color, size=11):
    """The values of a field written in the middle of its cells."""
    g = ""
    for k, v in values.items():
        g += text(
            x0 + (k + 0.5) * cw,
            y + 16.5,
            f"{v:g}",
            anchor="middle",
            size=size,
            fill=color,
            cls="sm-fig-code",
        )
    return g


@figure
def subset_projection(p):
    """Draw the intersection, its set projection, the contraction and the field projection.

    Figure of tutorial/operator_on_subset.md. Fig. 1 is the attempt that stops
    with std::out_of_range, Fig. 2 the contracted set expression, Fig. 3 the
    field u that the program prints after the field projection.
    """
    x0, cw, ch = 161, 26, 24  # a level-1 cell is 26 units wide, a level-0 cell 52
    mid = x0 + 10 * cw
    g = ""

    # Fig. 1: the intersection on level 1, then on level 0, where 5 and 7 fail
    y1, y0 = 30, 100
    g += cell_row(x0, y1, cw, 20, LEVEL1, ch=ch, tone="red", label_size=11)
    for k in (10, 15):  # the children of the cells 5 and 7 that level 1 does not hold
        g += rect(x0 + k * cw, y1, cw, ch, stroke=RED, sw=RULE, dash="3 2")
    g += cell_row(x0, y0, 2 * cw, 10, [(1, 3), (5, 8)], ch=ch, tone="red", label_size=11)
    for k in (5, 7):
        g += rect(x0 + 2 * k * cw, y0, 2 * cw, ch, fill=f"url(#{p}-r)", stroke=RED, sw=HAIR)
    g += _row_name(x0 - 14, y1, "intersection", "level 1")
    g += _row_name(x0 - 14, y0, ".on(0)", "level 0")
    g += callout(x0 + 15.5 * cw, y1 + ch / 2, x0 + 16.6 * cw, y1 - 14, "b", red=True)
    g += callout(x0 + 15 * cw, y0 + ch / 2, x0 + 16.6 * cw, y0 - 24, "a", red=True)
    g += fig_label(mid, y0 + 80, 1, "the attempt")

    # Fig. 2: the contraction of level 1, then its set projection on level 0
    y1, y0 = 222, 292
    g += cell_row(x0, y1, cw, 20, [(3, 5), (12, 14)], ch=ch, tone="red", label_size=11)
    for a, b in LEVEL1:  # the outline of the operand, level 1
        g += rect(x0 + a * cw, y1, (b - a) * cw, ch, stroke=INK, sw=HEAVY)
    g += cell_row(x0, y0, 2 * cw, 10, [(1, 3), (6, 7)], ch=ch, tone="red", label_size=11)
    g += _row_name(x0 - 14, y1, "contract(ca[1], 1)", "level 1")
    g += _row_name(x0 - 14, y0, ".on(0)", "level 0")
    g += callout(x0 + 14.5 * cw, y1 + ch / 2, x0 + 16.6 * cw, y1 - 14, "c")
    g += fig_label(mid, y0 + 80, 2, "after contraction")

    # Fig. 3: the field u, the means of the children written in the red cells
    y1, y0 = 414, 470
    g += cell_row(x0, y1, cw, 20, LEVEL1, ch=ch, brackets=False)
    g += _values(x0, y1, cw, U1, INK)
    for k in range(10):
        red = k in (1, 2, 6)
        g += rect(
            x0 + 2 * k * cw, y0, 2 * cw, ch, fill=RWASH if red else WASH, stroke=RED if red else INK
        )
    for k, v in U0.items():
        g += _values(x0, y0, 2 * cw, {k: v}, RED if k in (1, 2, 6) else INK2)
    for k in (1, 2, 6):  # each red cell reads its two children
        for c in (2 * k, 2 * k + 1):
            xc = x0 + (c + 0.5) * cw
            g += line(xc, y1 + ch + 3, xc, y0 - 3, stroke=INK2, extra=f'marker-end="url(#{p}-a)"')
    g += text(x0 - 14, y1 + 9, "u", anchor="end", size=14, fill=INK, cls="sm-fig-math")
    g += text(x0 - 14, y1 + 25, "level 1", anchor="end", size=11)
    g += text(x0 - 14, y0 + 9, "u", anchor="end", size=14, fill=INK, cls="sm-fig-math")
    g += text(x0 - 14, y0 + 25, "level 0", anchor="end", size=11)
    g += fig_label(mid, y0 + 56, 3, "field projection")

    height = y0 + 68
    return Drawing(
        PLATE_WIDTH,
        height,
        "Three figures on the mesh of the program, level 1 above level 0, at true size."
        " Fig. 1: the intersection of the two levels is [2, 6) and [11, 15) on level 1; its set"
        " projection on level 0 is [1, 3) and [5, 8). The cells 5 and 7 of level 0 are hatched:"
        " their children 10 and 15, drawn dashed, are not in level 1, so reading them fails."
        " Fig. 2: contract(ca[1], 1) is [3, 5) and [12, 14) inside the outline of level 1; on"
        " level 0 it gives [1, 3) and [6, 7). Fig. 3: the field u, with the values 2 to 5 and 11"
        " to 14 on level 1; arrows go from the children to the cells 1, 2 and 6 of level 0, which"
        " hold 2.5, 4.5 and 12.5, while the other cells of level 0 hold 0.",
        patterns(p),
        g,
        panels=(
            (0, 0, PLATE_WIDTH, 192, "wide"),
            (0, 192, PLATE_WIDTH, 192, "wide"),
            (0, 384, PLATE_WIDTH, height - 384, "wide"),
        ),
    )
