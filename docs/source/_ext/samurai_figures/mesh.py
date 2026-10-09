"""Graded quadtree meshes of the unit square, as samurai builds them.

A cell is ``(level, i, j)``: at level l it has the side 1/2^l and covers
[i/2^l, (i+1)/2^l) x [j/2^l, (j+1)/2^l), half-open like every samurai interval.
samurai meshes are graded: two cells that touch, by a face or by a corner, differ
by at most one level. ``build_mesh`` enforces it and ``check_graded`` proves it, so a
plate never shows a mesh samurai could not produce.
"""

from math import hypot

from .draw import EMPTY, HAIR, HEAVY, INK, RULE, circle, line, num, rect, text

CHILDREN = ((0, 0), (1, 0), (0, 1), (1, 1))


def _finest_grid(leaves, max_level):
    """Map each cell of the finest level to the level of the leaf that covers it."""
    grid = {}
    for level, i, j in leaves:
        s = 2 ** (max_level - level)
        for a in range(s):
            for b in range(s):
                grid[i * s + a, j * s + b] = level
    return grid


def build_mesh(min_level, max_level, refine):
    """Build a graded mesh.

    Start from the uniform mesh at ``min_level``, split every cell for which
    ``refine(level, i, j)`` is true, down to ``max_level``, then split the cells
    that touch a cell more than one level finer until none is left.
    """
    leaves = []

    def split(level, i, j):
        if level < max_level and refine(level, i, j):
            for a, b in CHILDREN:
                split(level + 1, 2 * i + a, 2 * j + b)
        else:
            leaves.append((level, i, j))

    n = 2**min_level
    for i in range(n):
        for j in range(n):
            split(min_level, i, j)

    changed = True
    while changed:
        grid = _finest_grid(leaves, max_level)
        changed = False
        graded = []
        for level, i, j in leaves:
            s = 2 ** (max_level - level)
            x0, y0 = i * s, j * s
            ring = [(x0 + k, y0 - 1) for k in range(-1, s + 1)] + [
                (x0 + k, y0 + s) for k in range(-1, s + 1)
            ]
            ring += [(x0 - 1, y0 + k) for k in range(s)] + [(x0 + s, y0 + k) for k in range(s)]
            finest = max(grid.get(c, -1) for c in ring)
            if finest > level + 1:
                changed = True
                graded += [(level + 1, 2 * i + a, 2 * j + b) for a, b in CHILDREN]
            else:
                graded.append((level, i, j))
        leaves = graded
    check_graded(leaves, max_level)
    return leaves


def check_graded(leaves, max_level):
    """Raise ``ValueError`` when two touching cells differ by more than one level."""
    grid = _finest_grid(leaves, max_level)
    n = 2**max_level
    if len(grid) != n * n:
        raise ValueError("the cells of the mesh do not cover the unit square exactly once")
    for (x, y), level in grid.items():
        for dx, dy in ((1, 0), (0, 1), (1, 1), (1, -1)):
            other = grid.get((x + dx, y + dy))
            if other is not None and abs(level - other) > 1:
                raise ValueError(
                    f"the mesh is not graded: a level-{level} cell touches a level-{other} cell"
                )


def _circle_gap(cx, cy, r, level, i, j):
    """Return the distance from the circle of centre (cx, cy) and radius r to a cell.

    The distance is 0 when the circle crosses the cell ``(level, i, j)``.
    """
    h = 2.0**-level
    x0, y0 = i * h, j * h
    dmin = hypot(max(x0, min(cx, x0 + h)) - cx, max(y0, min(cy, y0 + h)) - cy)
    dmax = hypot(max(abs(cx - x0), abs(cx - x0 - h)), max(abs(cy - y0), abs(cy - y0 - h)))
    if dmin > r:
        return dmin - r
    return max(r - dmax, 0.0)


def circle_refine(cx, cy, r, margin=0.0):
    """Refine the cells that come within ``margin`` of the circle of centre (cx, cy), radius r.

    With the default ``margin`` of 0, these are the cells the circle crosses.
    """

    def refine(level, i, j):
        return _circle_gap(cx, cy, r, level, i, j) <= margin

    return refine


def check_outline(leaves, disc, max_level, size):
    """Raise ``ValueError`` unless the outline of the circle lies in cells of ``max_level``.

    Every cell the circle ``disc`` (cx, cy, r) crosses must be of ``max_level``, and
    the circle, drawn ``size`` units for the unit square, must pass at least the
    width of its heavy rule from every coarser cell, so that the rule stays inside
    the hatched cells.
    """
    for level, i, j in leaves:
        if level == max_level:
            continue
        gap = _circle_gap(*disc, level, i, j) * size
        if gap == 0:
            raise ValueError(f"the circle crosses the cell {(level, i, j)} of level {level}")
        if gap < HEAVY:
            raise ValueError(
                f"the circle passes {gap:.2f} units from the cell {(level, i, j)} of level"
                f" {level}: its outline would seem to cross it"
            )


def draw_mesh(p, leaves, ox, oy, size, max_level, disc=None, ghost=None, hatch=True, axes=True):
    """Draw a mesh of the unit square in a square of side ``size`` at (ox, oy), y up.

    The cells of ``max_level`` are hatched in red. ``disc`` (cx, cy, r) is a disc
    where the solution is constant, ruled in fine ink lines and outlined; ``ghost``
    is a dotted circle. ``p`` prefixes the ids, see ``draw.patterns``.
    """

    def X(u):
        return ox + u * size

    def Y(v):
        return oy + (1 - v) * size

    s = ""
    if disc:
        clip = f"{p}-c{round(ox)}"
        s += f'<clipPath id="{clip}">{circle(X(disc[0]), Y(disc[1]), disc[2] * size)}</clipPath>'
        s += (
            f'<rect x="{num(ox)}" y="{num(oy)}" width="{num(size)}" height="{num(size)}"'
            f' fill="url(#{p}-k)" clip-path="url(#{clip})"/>'
        )
    for level, i, j in leaves:
        h = size * 2.0**-level
        fine = level == max_level
        fill = f"url(#{p}-r)" if fine and hatch else "none"
        s += rect(
            ox + i * h,
            oy + size - (j + 1) * h,
            h,
            h,
            fill=fill,
            stroke=INK,
            sw=HAIR if fine else EMPTY,
        )
    if disc:
        s += circle(X(disc[0]), Y(disc[1]), disc[2] * size, stroke=INK, sw=HEAVY)
    if ghost:
        s += circle(X(ghost[0]), Y(ghost[1]), ghost[2] * size, stroke=INK, sw=RULE, dash="2 3")
    s += rect(ox, oy, size, size, stroke=INK, sw=HEAVY)
    if axes:
        for v, label in ((0, "0"), (0.5, "1/2"), (1, "1")):
            s += line(X(v), oy + size, X(v), oy + size + 5) + text(
                X(v), oy + size + 18, label, anchor="middle", size=11
            )
            if v:
                s += line(ox - 5, Y(v), ox, Y(v)) + text(
                    ox - 8, Y(v) + 4, label, anchor="end", size=11
                )
    return s
