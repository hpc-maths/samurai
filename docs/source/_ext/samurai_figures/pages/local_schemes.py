"""The figure of ``reference/local_schemes.md``: the order of the cells of a star stencil.

A cell-based scheme of configuration ``StarStencilSchemeConfig<..., radius, ...>``
reads the cells of ``star_stencil<dim, radius>()`` (``include/samurai/stencil.hpp``),
set by ``CellBasedSchemeDefinitionBase``; ``coeffs[i]``, ``jac[i]`` and ``cells[i]``
refer to the i-th cell of that stencil. ``PRINTED`` is the output of a program that
builds such a scheme in 2D for the radii 0, 1 and 2 and prints ``cfg::stencil_size``,
``cfg::center_index`` and ``A.stencil().stencil`` row by row. ``STARS`` gives each
cell its name, from the comments of ``star_stencil``; ``check_order`` raises if its
order is not the printed one.
"""

from ..draw import HAIR, INK, INK2, RED, RWASH, WASH, line, patterns, rect, text
from ..registry import DIAGRAM_WIDTH, Drawing, figure

# The output of the program, for StarStencilSchemeConfig of radius 0, 1 and 2 in 2D.
PRINTED = """\
radius 0: stencil_size 1, center_index 0
  0 {0,0}
radius 1: stencil_size 5, center_index 1
  0 {-1,0}
  1 {0,0}
  2 {1,0}
  3 {0,-1}
  4 {0,1}
radius 2: stencil_size 9, center_index 2
  0 {-2,0}
  1 {-1,0}
  2 {0,0}
  3 {1,0}
  4 {2,0}
  5 {0,-2}
  6 {0,-1}
  7 {0,1}
  8 {0,2}
"""

# The width of the hatched bands on each side of the center cell, clear of its labels.
CENTER_BAND = 10

# The cells drawn for each radius, in the order of star_stencil<2, radius>().
STARS = {
    0: (("center", (0, 0)),),
    1: (
        ("left", (-1, 0)),
        ("center", (0, 0)),
        ("right", (1, 0)),
        ("bottom", (0, -1)),
        ("top", (0, 1)),
    ),
    2: (
        ("left2", (-2, 0)),
        ("left", (-1, 0)),
        ("center", (0, 0)),
        ("right", (1, 0)),
        ("right2", (2, 0)),
        ("bottom2", (0, -2)),
        ("bottom", (0, -1)),
        ("top", (0, 1)),
        ("top2", (0, 2)),
    ),
}


def _parse(printed):
    """Read ``printed`` into ``{radius: (center_index, (vector, ...))}``.

    Raise ``ValueError`` if the rows of a stencil are out of order or if their count
    is not its ``stencil_size``.
    """
    stencils, radius = {}, None
    for row in printed.splitlines():
        if row.startswith("radius"):
            head, tail = row.split(":")
            radius = int(head.split()[1])
            size, center = (int(part.split()[-1]) for part in tail.split(","))
            stencils[radius] = (size, center, [])
        else:
            index, vector = row.split()
            vectors = stencils[radius][2]
            if int(index) != len(vectors):
                raise ValueError(f"radius {radius}: row {index} is out of order")
            vectors.append(tuple(int(k) for k in vector.strip("{}").split(",")))
    for r, (size, _, vectors) in stencils.items():
        if len(vectors) != size:
            raise ValueError(f"radius {r}: {len(vectors)} cells, stencil_size = {size}")
    return {r: (center, tuple(vectors)) for r, (_, center, vectors) in stencils.items()}


def check_order():
    """Return the center index of each radius, after checking ``STARS`` against ``PRINTED``.

    Raise ``ValueError`` if a drawn stencil differs from the printed one, in its
    cells or in their order, or if the center index is not the radius.
    """
    printed = _parse(PRINTED)
    if set(printed) != set(STARS):
        raise ValueError(f"the program prints the radii {sorted(printed)}, not {sorted(STARS)}")
    centers = {}
    for radius, cells in STARS.items():
        center, vectors = printed[radius]
        drawn = tuple(vector for _, vector in cells)
        if drawn != vectors:
            raise ValueError(f"radius {radius}: drawn in the order {drawn}, printed {vectors}")
        if center != radius or cells[center][1] != (0, 0):
            raise ValueError(f"radius {radius}: center_index {center} is not the center cell")
        centers[radius] = center
    return centers


def _vector(v):
    """Write the vector ``v`` as stencil.hpp does, ``{-1,0}``."""
    return "{" + ",".join(str(k) for k in v) + "}"


def _code(x, y, s, size=10.5, fill=INK2):
    """Set ``s`` in the mono, centered on x."""
    return text(x, y, s, anchor="middle", size=size, fill=fill, cls="sm-fig-code")


def _cell(x, y, c, center, p):
    """Draw a cell of side ``c`` at (x, y), shaded and hatched in red for the center.

    The hatching of the center fills two bands along its sides and leaves its middle
    in the red wash, so that its labels stay legible in both themes without a halo.
    """
    if not center:
        return rect(x, y, c, c, fill=WASH)
    band = CENTER_BAND
    return (
        rect(x, y, c, c, fill=RWASH, stroke="none")
        + rect(x, y, band, c, fill=f"url(#{p}-r)", stroke="none")
        + rect(x + c - band, y, band, c, fill=f"url(#{p}-r)", stroke="none")
        + rect(x, y, c, c, stroke=RED, sw=HAIR)
    )


def _star(p, x, y, c, radius, center):
    """Draw the star stencil of ``radius`` with cells of side ``c``, its top left corner at (x, y).

    Each cell holds its index in the stencil, its name and its vector; the center is
    shaded in red and hatched along its sides.
    """
    g = ""
    for index, (name, (dx, dy)) in enumerate(STARS[radius]):
        cx, cy = x + (dx + radius) * c, y + (radius - dy) * c
        g += _cell(cx, cy, c, index == center, p)
        g += _code(cx + c / 2, cy + 22, f"{index}", size=16, fill=INK)
        g += text(cx + c / 2, cy + 37, name, anchor="middle", size=10, fill=INK2)
        g += _code(cx + c / 2, cy + 51, _vector((dx, dy)), size=10)
    return g


@figure
def star_stencils(p):
    """Draw the stencils of radius 0, 1 and 2 in 2D, each cell with its index.

    Figure of reference/local_schemes.md, at "Stencil-based cell schemes". The
    centers of the three stencils lie on one row; the axes sit in the empty corner
    of the stencil of radius 2.
    """
    centers = check_order()
    c, gap, y0 = 58, 52, 10
    # The size label of radius 0 is wider than its cell: center the art and that label
    # together. Source Code Pro advances 0.6 em per glyph.
    half = 0.6 * 11 * len(f"stencil_size = {len(STARS[0])}") / 2
    x0 = (DIAGRAM_WIDTH - 9 * c - 2 * gap + half - c / 2) / 2
    left = {0: x0, 1: x0 + c + gap, 2: x0 + 4 * c + 2 * gap}
    base = y0 + 5 * c + 24  # the labels of the three stencils share a baseline
    g = ""
    for radius in (0, 1, 2):
        side = 2 * radius + 1
        x, y = left[radius], y0 + (2 - radius) * c
        g += _star(p, x, y, c, radius, centers[radius])
        xm = x + side * c / 2
        g += text(xm, base, f"radius {radius}", anchor="middle", size=13, fill=INK)
        g += _code(xm, base + 17, f"stencil_size = {len(STARS[radius])}", size=11)
    # the axes, in the empty corner at the bottom left of the stencil of radius 2
    ax, ay = left[2] + 14, y0 + 5 * c - 14
    arrow = f'marker-end="url(#{p}-a)"'
    g += line(ax, ay, ax + 36, ay, stroke=INK2, extra=arrow)
    g += line(ax, ay, ax, ay - 36, stroke=INK2, extra=arrow)
    g += text(ax + 46, ay + 5, "x", anchor="middle", size=15, fill=INK2, cls="sm-fig-math")
    g += text(ax, ay - 44, "y", anchor="middle", size=15, fill=INK2, cls="sm-fig-math")
    return Drawing(
        DIAGRAM_WIDTH,
        base + 26,
        "Three stencils in 2D side by side, their centers on one row. Each cell holds its"
        " index in the stencil, its name and its vector; the center is shaded in red"
        " and hatched along its sides."
        " Radius 0, stencil_size = 1: the cell 0 center {0,0} alone. Radius 1, stencil_size"
        " = 5: 0 left {-1,0}, 1 center {0,0}, 2 right {1,0}, 3 bottom {0,-1}, 4 top {0,1}."
        " Radius 2, stencil_size = 9: along x from left to right, 0 left2 {-2,0}, 1 left"
        " {-1,0}, 2 center {0,0}, 3 right {1,0}, 4 right2 {2,0}; along y from bottom to top,"
        " 5 bottom2 {0,-2}, 6 bottom {0,-1}, then the center, 7 top {0,1}, 8 top2 {0,2}."
        " Arrows at the bottom left of the last stencil show x and y.",
        patterns(p),
        g,
    )
