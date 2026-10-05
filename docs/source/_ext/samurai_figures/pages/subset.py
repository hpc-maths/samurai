"""The figures of ``reference/subset.md``: set operations and the set projection."""

from ..draw import (
    EMPTY,
    FIG,
    HAIR,
    HEAVY,
    INK,
    RED,
    WASH,
    callout,
    cell_row,
    fig_label,
    line,
    num,
    patterns,
    rect,
    text,
)
from ..registry import DIAGRAM_WIDTH, PLATE_WIDTH, Drawing, figure


def _number_line(x0, y, cw, n, size=11.5):
    """The axis under rows of level-0 cells, with the indices 0 to n."""
    g = line(x0, y, x0 + n * cw, y)
    for k in range(n + 1):
        g += line(x0 + k * cw, y, x0 + k * cw, y + 5) + text(
            x0 + k * cw, y + 18, k, anchor="middle", size=size
        )
    return g + text(x0 - 14, y + 18, "level 0", anchor="end", size=11.5)


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
    """Draw two operands of two levels and the set projection of their intersection.

    Figure of reference/subset.md, from set_algebra_levels.cpp.
    """
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
def plus_sign_operations(p):
    """Draw the expansion, contraction and translation of a plus sign.

    Figure of reference/subset.md, from set_algebra_expand_contract.cpp: every
    result is drawn from the intervals the program prints.
    """
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
