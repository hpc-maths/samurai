"""The figures of ``tutorial/algorithm.md``: the shifted interval of the upwind scheme."""

from ..draw import (
    EMPTY,
    FIG,
    INK,
    INK2,
    PAPER,
    RED,
    RULE,
    WASH,
    bracket,
    line,
    num,
    patterns,
    rect,
    text,
)
from ..registry import DIAGRAM_WIDTH, Drawing, figure

# The mesh of step 2 of demos/tutorial/AMR_1D_Burgers: the box [-3, 3] at level 6.
LEVEL = 6
BOX = (-3, 3)
# Step 1 prints the mesh, whose level 6 holds "cells = 0->[0,384)@0:1": one interval.
INTERVAL = (0, 384)
# The program does not print cell centers. This one was read from cell.center(0) of
# the cell 0, printed by a for_each_cell over the same mesh: the cell 0 starts at the
# corner -3 of the box.
CENTER_0 = -2.99219

# The cells drawn at each end of the interval; the cells between them are cut out.
LEFT = range(-1, 6)
RIGHT = range(378, 384)


def _check():
    """Check the intervals of update_sol against the mesh, and return them.

    Return ``ii``, ``ii - 1`` and ``interval - 1``. Raise ``ValueError`` when the
    mesh does not hold ``INTERVAL``, when a cell of ``ii - 1`` is not the left
    neighbor of a cell of ``ii`` inside the mesh, or when ``interval - 1`` does
    not ask for exactly one cell outside the mesh, the cell -1 left of the box.
    """
    start, end = INTERVAL
    dx = 2.0**-LEVEL
    if (BOX[1] - BOX[0]) * 2**LEVEL != end - start:
        raise ValueError(f"the box {BOX} does not hold {end - start} cells of level {LEVEL}")
    ii = (start + 1, end - 1)
    shifted = (ii[0] - 1, ii[1] - 1)
    wrong = (start - 1, end - 1)
    for k in range(*shifted):
        if not (start <= k < end and k + 1 in range(*ii)):
            raise ValueError(f"the cell {k} of ii - 1 is not the left neighbor of a cell of ii")
    # the cell k spans [x0 + k dx, x0 + (k + 1) dx), the cell 0 starting at the box corner
    x0 = CENTER_0 - dx / 2
    if abs(x0 - BOX[0]) > 1e-5:
        raise ValueError(f"the cell 0 starts at {x0}, not at the corner {BOX[0]} of the box")
    outside = [k for k in range(*wrong) if not start <= k < end]
    if outside != [-1] or x0 + (outside[0] + 1) * dx > BOX[0] + 1e-5:
        raise ValueError(f"interval - 1 should leave the box by the cell -1 only, not {outside}")
    return ii, shifted, wrong


def _break(x, y1, y2, color, slant=4, gap=6):
    """Cut out the cells between two ends of a row: a paper band between two slanted lines."""
    a, b = x - gap / 2, x + gap / 2
    band = (
        f'<path d="M{num(a + slant)} {num(y1)}H{num(b + slant)}L{num(b - slant)} {num(y2)}'
        f'H{num(a - slant)}Z" fill="{PAPER}" stroke="none"/>'
    )
    return (
        band
        + line(a + slant, y1, a - slant, y2, stroke=color, sw=RULE)
        + line(b + slant, y1, b - slant, y2, stroke=color, sw=RULE)
    )


@figure
def shifted_interval(p):
    """Draw interval, ii and ii - 1 of update_sol, and the cell interval - 1 asks for.

    Figure of tutorial/algorithm.md. Only the two ends of the interval [0, 384)
    of level 6 are drawn, on either side of a break mark.
    """
    ii, shifted, wrong = _check()
    x0, cw, ch, gap = 132, 36, 24, 52
    names = x0 - 14

    def X(k):
        """Return the left edge of the cell k; the edges 7 to 377 are cut out."""
        if k <= LEFT[-1] + 1:
            return x0 + (k - LEFT[0]) * cw
        return x0 + len(LEFT) * cw + gap + (k - RIGHT[0]) * cw

    cut = (X(LEFT[-1] + 1) + X(RIGHT[0])) / 2

    def row(y, cells, below):
        """Draw the two ends of the row [a, b) at y, its bracket above or below, and the cut."""
        a, b = cells
        fill, stroke = WASH, INK
        # the cells 6 to 377 run on between the two ends, under the outlines of the cells drawn
        cut_a, cut_b = X(LEFT[-1] + 1), X(RIGHT[0])
        g = rect(cut_a, y, cut_b - cut_a, ch, fill=fill, stroke="none")
        for k in [*LEFT, *RIGHT]:
            if k < INTERVAL[0] and k < a:
                continue
            if a <= k < b:
                g += rect(X(k), y, cw, ch, fill=fill, stroke=stroke)
            elif k >= INTERVAL[0]:
                g += rect(X(k), y, cw, ch, stroke=FIG, sw=EMPTY)
        for edge in (y, y + ch):
            g += line(cut_a, edge, cut_b, edge, stroke=stroke)
        yb = y + ch + 9 if below else y - 9
        g += bracket(
            X(a), X(b), yb, f"[{a}, {b})", color=stroke, size=11, label_dy=15 if below else -9
        )
        top, bottom = (y - 4, yb + 3) if below else (yb - 3, y + ch + 4)
        return g + _break(cut, top, bottom, stroke)

    def name(y, s):
        """Write the name of the row at y in the mono, right-aligned before the cells."""
        return text(names, y + 16, s, anchor="end", size=11.5, fill=INK, cls="sm-fig-code")

    ya, yb, yc, yd = 50, 110, 168, 252
    g = ""
    # the edges of the box, where the mesh ends, in the red of the cell outside it
    for k, edge in ((INTERVAL[0], "\u22123"), (INTERVAL[1], "3")):
        g += line(X(k), 20, X(k), yd + ch + 15, stroke=RED, sw=RULE, dash="2 3")
        g += text(X(k), 14, f'<tspan class="sm-fig-math">x</tspan> = {edge}', anchor="middle")
    g += row(ya, INTERVAL, below=False) + name(ya, "interval")
    g += row(yb, ii, below=False) + name(yb, "ii")
    g += row(yc, shifted, below=True) + name(yc, "ii - 1")
    # each cell of ii - 1 points to the cell of ii it is the left neighbor of
    for k in [*LEFT, *RIGHT]:
        if shifted[0] <= k < shifted[1] and k + 1 in [*LEFT, *RIGHT]:
            g += line(
                X(k) + cw / 2,
                yc - 3,
                X(k + 1) + cw / 2,
                yb + ch + 3,
                stroke=INK2,
                extra=f'marker-end="url(#{p}-a)"',
            )
    g += line(24, yd - 18, DIAGRAM_WIDTH - 8, yd - 18, stroke=FIG)
    g += row(yd, wrong, below=True) + name(yd, "interval - 1")
    # the cell -1 that interval - 1 asks for is not in the mesh
    xm = X(-1)
    g += rect(xm, yd, cw, ch, fill=PAPER, stroke=RED, sw=RULE, dash="3 3")
    g += line(xm + 9, yd + 5, xm + cw - 9, yd + ch - 5, stroke=RED, sw=RULE)
    g += line(xm + 9, yd + ch - 5, xm + cw - 9, yd + 5, stroke=RED, sw=RULE)
    g += text(
        xm, yd + ch + 48, "cell \u22121 is left of the box, not in the mesh", size=11, fill=RED
    )
    # the number line of level 6, cut like the rows
    axis = yd + ch + 70
    g += line(X(LEFT[0]), axis, X(RIGHT[-1] + 1), axis, stroke=INK)
    for k in [*LEFT, LEFT[-1] + 1, *RIGHT, RIGHT[-1] + 1]:
        g += line(X(k), axis, X(k), axis + 5, stroke=INK)
        g += text(X(k), axis + 18, str(k).replace("-", "\u2212"), anchor="middle", size=10.5)
    g += _break(cut, axis - 5, axis + 6, INK2)
    g += text(names, axis + 18, f"level {LEVEL}", anchor="end", size=11.5)
    return Drawing(
        DIAGRAM_WIDTH,
        axis + 26,
        f"The cells of level {LEVEL} of the 1D mesh of the box [-3, 3], numbered from 0 at x = -3,"
        f" with only the two ends drawn: the cells {LEFT[0] + 1} to {LEFT[-1]} and"
        f" {RIGHT[0]} to {RIGHT[-1]}, on either side of a break mark that cuts out the cells"
        f" {LEFT[-1] + 1} to {RIGHT[0] - 1}. interval is [{INTERVAL[0]}, {INTERVAL[1]}), every"
        f" cell of the mesh. ii is [{ii[0]}, {ii[1]}), the interval without its first and last"
        f" cells. ii - 1 is [{shifted[0]}, {shifted[1]}), and an arrow goes from each"
        " of its cells to the cell on its right in ii, the cell it is the left neighbor of."
        f" Under a rule, interval - 1 is [{wrong[0]}, {wrong[1]}): its first cell, -1, is drawn"
        " dashed and crossed in red, because it lies left of the box and is not in the mesh."
        " Dotted red guides mark the edges of the box, x = -3 and x = 3, where the mesh ends.",
        patterns(p),
        g,
    )
