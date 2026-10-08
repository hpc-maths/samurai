"""The figure of ``howto/mesh.md``: the ghost ring of the uniform mesh.

The mesh is the one of ``howto/snippet/mesh/uniform.cpp``: the unit box at level 4,
with ``UniformConfig<dim>``, whose ghost width is ``default_config::ghost_width``,
1 (``include/samurai/samurai_config.hpp``). ``UniformMesh::update_sub_mesh``
(``include/samurai/uniform_mesh.hpp``) builds ``cells_and_ghosts`` row by row: each
interval of ``cells`` is widened by the ghost width on both ends and copied to the
rows within the ghost width of its own. The ring is therefore square and holds the
four corner ghosts. The two counts drawn are read from ``uniform_output.txt``, the
output of the program, and checked against the sizes of the two squares.
"""

import re
from pathlib import Path

from ..draw import (
    HAIR,
    HEAVY,
    INK,
    INK2,
    INK3,
    PAPER,
    RULE,
    WASH,
    bracket,
    circle,
    line,
    rect,
    text,
)
from ..registry import DIAGRAM_WIDTH, Drawing, figure

LEVEL = 4
GHOST_WIDTH = 1  # default_config::ghost_width, kept by UniformConfig<dim>
OUTPUT = Path(__file__).parents[3] / "howto" / "snippet" / "mesh" / "uniform_output.txt"


def uniform_counts(path=OUTPUT):
    """Return the two counts ``uniform.cpp`` prints, cells and cells and ghosts.

    Raise ``ValueError`` when they are not the squares of the level-4 mesh without
    and with its ring of ghosts.
    """
    output = path.read_text(encoding="utf-8")
    counts = {}
    for name in ("cells", "cells and ghosts"):
        match = re.search(rf"^{name}: (\d+)$", output, re.MULTILINE)
        if match is None:
            raise ValueError(f'{path.name} has no line "{name}: <count>"')
        counts[name] = int(match.group(1))
    n = 2**LEVEL
    side = n + 2 * GHOST_WIDTH
    if counts["cells"] != n * n or counts["cells and ghosts"] != side * side:
        raise ValueError(
            f"{path.name} prints {counts['cells']} and {counts['cells and ghosts']} cells,"
            f" not {n} x {n} and {side} x {side}"
        )
    return n, side, counts["cells"], counts["cells and ghosts"]


def _ring(x0, y0, c, side, g):
    """Draw the ghosts of a square of ``side`` cells whose outer ``g`` rows are the ring.

    The ring is washed; its outline and the edges between its cells are dashed. The
    edges are drawn as lines, not as one rectangle per ghost, so that two ghosts
    never draw the same dashed edge twice.
    """
    full = side * c
    s = rect(x0, y0, full, full, fill=WASH, stroke="none")
    s += rect(x0 + g * c, y0 + g * c, full - 2 * g * c, full - 2 * g * c, fill=PAPER, stroke="none")
    dashed = {"stroke": INK3, "sw": RULE, "dash": "3 3"}
    for k in range(1, side):
        xk, yk = x0 + k * c, y0 + k * c
        # the edges across the top and the bottom strips, then across the side strips
        s += line(xk, y0, xk, y0 + g * c, **dashed)
        s += line(xk, y0 + full - g * c, xk, y0 + full, **dashed)
        if g <= k <= side - g:
            s += line(x0, yk, x0 + g * c, yk, **dashed)
            s += line(x0 + full - g * c, yk, x0 + full, yk, **dashed)
    for k in range(1, g):
        # the inner rows of a ring wider than one cell
        for edge in (k, side - k):
            s += line(x0, y0 + edge * c, x0 + full, y0 + edge * c, **dashed)
            s += line(x0 + edge * c, y0, x0 + edge * c, y0 + full, **dashed)
    return s + rect(x0, y0, full, full, **dashed)


def _swatch(x, y, c, ghost):
    """Draw a legend cell: a ghost dashed on the wash, or a cell outlined in ink."""
    if ghost:
        return rect(x, y, c, c, fill=WASH, stroke=INK3, dash="3 3")
    return rect(x, y, c, c, stroke=INK)


@figure
def ghost_ring(_p):
    """Draw the 16 by 16 cells of level 4 and the ring of ghosts that makes 18 by 18.

    Figure of howto/mesh.md, at "Check the mesh". The counts come from
    ``uniform_output.txt``; under the mesh, the intervals of one row of ``cells``
    and of ``cells_and_ghosts`` along x.
    """
    n, side, n_cells, n_all = uniform_counts()
    g = GHOST_WIDTH
    c, x0, y0 = 15, 82, 20
    full = side * c
    xa, xb = x0 + g * c, x0 + (g + n) * c  # the box, from index 0 to index n
    ya, yb = y0 + g * c, y0 + (g + n) * c
    s = _ring(x0, y0, c, side, g)
    for k in range(1, n):
        s += line(xa + k * c, ya, xa + k * c, yb, stroke=INK2, sw=HAIR)
        s += line(xa, ya + k * c, xb, ya + k * c, stroke=INK2, sw=HAIR)
    s += rect(xa, ya, n * c, n * c, sw=HEAVY)

    # the intervals of a row along x: [0, 16) for the cells, [-1, 17) with the ghosts
    rows = y0 + full + 12
    s += bracket(xa, xb, rows, f"[0, {n})")
    s += bracket(x0, x0 + full, rows + 36, f"[{-g}, {n + g})", color=INK3)
    s += text(x0 + full + 14, rows + 4, "cells", size=11.5)
    s += text(x0 + full + 14, rows + 40, "cells and ghosts", size=11.5)

    # the legend, on the right of the mesh
    lx, lc = x0 + full + 64, 15
    first, second = y0 + 58, y0 + 154
    s += _swatch(lx, first - 12, lc, ghost=False)
    s += text(lx + lc + 12, first, f"cell of level {LEVEL}", size=12, fill=INK)
    s += text(lx + lc + 12, first + 24, f"cells: {n_cells}", size=12, fill=INK, cls="sm-fig-code")
    s += text(lx + lc + 12, first + 44, f"{n} × {n} cells, inside the box", size=11.5)
    s += _swatch(lx, second - 12, lc, ghost=True)
    s += text(lx + lc + 12, second, "ghost cell", size=12, fill=INK)
    s += text(
        lx + lc + 12,
        second + 24,
        f"cells and ghosts: {n_all}",
        size=12,
        fill=INK,
        cls="sm-fig-code",
    )
    ghosts = n_all - n_cells
    s += text(
        lx + lc + 12, second + 44, f"{side} × {side} = {n_cells} cells + {ghosts} ghosts", size=11.5
    )
    s += text(lx + lc + 12, second + 62, f"{ghosts} = 4 sides × {n} + 4 corners", size=11.5)

    # a leader from the top right corner ghost to its note
    cx, cy = x0 + full - c / 2, y0 + c / 2
    s += line(cx, cy, lx - 10, cy, stroke=INK, sw=HAIR) + circle(cx, cy, 1.8, fill=INK)
    s += text(lx - 6, cy + 4, "corner ghost", size=11.5, fill=INK2, knock=True)
    return Drawing(
        DIAGRAM_WIDTH,
        rows + 36 + 24,
        f"A square of {n} by {n} cells of level {LEVEL}, outlined by the heavy edge of the box,"
        f" inside a ring of ghost cells one cell wide, dashed on a grey wash, which makes a"
        f" square of {side} by {side}; the ring includes its four corner ghosts, and a leader"
        f" points to the top right one. Under the"
        f" square, the interval of one row along x: [0, {n}) for the cells, [{-g}, {n + g}) for"
        f" the cells and ghosts. Beside it: cells: {n_cells}, {n} times {n} cells inside the"
        f" box; cells and ghosts: {n_all}, {side} times {side}, that is {n_cells} cells and"
        f" {ghosts} ghosts, four sides of {n} and four corners.",
        "",
        s,
    )
