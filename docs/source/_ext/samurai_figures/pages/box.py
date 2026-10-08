"""The figures of ``howto/box.md``: the domains of the snippets and the smallest hole.

The corners of the boxes are read from the snippets the page includes,
``howto/snippet/box/2d_union.cpp`` and ``2d_box_with_hole.cpp``, so that a change
in a snippet changes the drawing, and a call the parser does not recognize stops
the build. The diagram of the smallest hole draws the check of
``Mesh_base::compute_scaling_factor`` (``include/samurai/mesh.hpp``) for the radius
``mesh_config`` enforces by default; both are read from the headers and the build
stops if their source text changes.
"""

import math
import re
from fractions import Fraction
from pathlib import Path

from ..draw import (
    EMPTY,
    HEAVY,
    INK,
    INK2,
    INK3,
    PAPER,
    RED,
    RWASH,
    WASH,
    bracket,
    circle,
    line,
    num,
    path,
    rect,
    text,
)
from ..registry import DIAGRAM_WIDTH, Drawing, figure

SOURCE = Path(__file__).resolve().parents[3]
SNIPPETS = SOURCE / "howto" / "snippet" / "box"
INCLUDE = SOURCE.parents[1] / "include" / "samurai"

# DomainBuilder<dim> domain({x0, y0}, {x1, y1}); then domain.add(...) or domain.remove(...)
_CORNERS = r"\{\s*([^{}]+?)\s*\},\s*\{\s*([^{}]+?)\s*\}"
_BUILDER = re.compile(r"DomainBuilder<dim>\s+domain\(" + _CORNERS + r"\);")
_CALL = re.compile(r"domain\.(add|remove)\(" + _CORNERS + r"\);")

# The check of a hole in Mesh_base::compute_scaling_factor, and the smallest radius of
# mesh_config, as the headers write them.
_HOLE_CHECK = "box.min_length() < 2 * largest_cell_length * max_stencil_radius()"
_MIN_RADIUS = re.compile(r"m_max_stencil_radius = std::max\(m_max_stencil_radius, (\d+)\);")


def _point(source):
    """Parse the point ``{x, y}`` of a snippet into fractions."""
    point = tuple(Fraction(v.strip()) for v in source.split(","))
    if len(point) != 2:
        raise ValueError(f"the corner {{{source}}} is not a 2D point")
    return point


def read_domain(name):
    """Read the boxes of the snippet ``name``: ``[(call, lower, upper), ...]`` in call order.

    ``call`` is ``"DomainBuilder"`` for the box of the constructor, then ``"add"`` or
    ``"remove"``. Raise ``ValueError`` when the snippet has no ``DomainBuilder``.
    """
    code = (SNIPPETS / name).read_text(encoding="utf-8")
    first = _BUILDER.search(code)
    if not first:
        raise ValueError(f"{name}: no 'DomainBuilder<dim> domain({{...}}, {{...}});' found")
    boxes = [("DomainBuilder", _point(first[1]), _point(first[2]))]
    boxes += [(m[1], _point(m[2]), _point(m[3])) for m in _CALL.finditer(code)]
    return boxes


def hole_rule():
    """Return the smallest radius samurai uses, after checking the source of the hole check."""
    mesh = (INCLUDE / "mesh.hpp").read_text(encoding="utf-8")
    if _HOLE_CHECK not in mesh:
        raise ValueError(f"include/samurai/mesh.hpp no longer checks '{_HOLE_CHECK}'")
    found = _MIN_RADIUS.search((INCLUDE / "mesh_config.hpp").read_text(encoding="utf-8"))
    if not found:
        raise ValueError(
            "include/samurai/mesh_config.hpp: the smallest stencil radius is not found"
        )
    return int(found[1])


def _region(boxes):
    """Cover the domain with the cells of a grid fine enough for every corner.

    samurai applies every ``remove`` after every ``add``. Returns the step of the
    grid and the set of its cells ``(i, j)``, the cell ``(i, j)`` covering
    [i, i + 1) x [j, j + 1) in steps.
    """
    values = [v for _, lower, upper in boxes for v in (*lower, *upper)]
    # the largest step that divides every coordinate
    den = math.lcm(*(v.denominator for v in values))
    step = Fraction(math.gcd(*(int(v * den) for v in values)) or 1, den)

    def cells(lower, upper):
        return {
            (i, j)
            for i in range(int(lower[0] / step), int(upper[0] / step))
            for j in range(int(lower[1] / step), int(upper[1] / step))
        }

    added = set().union(*(cells(lo, up) for call, lo, up in boxes if call != "remove"))
    removed = set().union(*(cells(lo, up) for call, lo, up in boxes if call == "remove"))
    return step, added - removed


def _edges(grid):
    """Return the edges of the region ``grid`` covers, as unit segments ``(x1, y1, x2, y2)``."""
    edges = []
    for x, y in grid:
        if (x, y - 1) not in grid:
            edges.append((x, y, x + 1, y))
        if (x, y + 1) not in grid:
            edges.append((x, y + 1, x + 1, y + 1))
        if (x - 1, y) not in grid:
            edges.append((x, y, x, y + 1))
        if (x + 1, y) not in grid:
            edges.append((x + 1, y, x + 1, y + 1))
    return sorted(edges)


def _point_label(point):
    """Write the point ``(x, y)`` as the snippet does, without a trailing ``.0``."""
    return f"({num(float(point[0]))}, {num(float(point[1]))})"


# The places a corner label tries, in order, as (anchor, dx, dy) from its dot: the
# lower corner first below on its left, the upper corner first above on its right.
_BELOW_LEFT, _BELOW_RIGHT = ("end", -5, 16), ("start", 5, 16)
_ABOVE_LEFT, _ABOVE_RIGHT = ("end", -5, -9), ("start", 5, -9)
_LOWER = (_BELOW_LEFT, _BELOW_RIGHT, _ABOVE_LEFT, _ABOVE_RIGHT)
_UPPER = (_ABOVE_RIGHT, _ABOVE_LEFT, _BELOW_RIGHT, _BELOW_LEFT)
# the room a corner label keeps from every line, dot and other label
CLEARANCE = 4


def _text_box(x, y, s, anchor, size):
    """Return the box ``(x0, y0, x1, y1)`` of a mono text, from the advance of 0.6 em."""
    w = 0.6 * size * len(s)
    x0 = {"start": x, "middle": x - w / 2, "end": x - w}[anchor]
    return x0, y - 0.8 * size, x0 + w, y + 0.22 * size


def _clear(box, lines, boxes, dots):
    """Tell whether ``box``, grown by the clearance, meets no line, box or dot."""
    x0, y0, x1, y1 = box
    # a line counts with half its heavy stroke, a dot with its radius
    m, half = CLEARANCE, HEAVY / 2
    for a, b, c, d in lines:
        if min(a, c) - half < x1 + m and max(a, c) + half > x0 - m:
            if min(b, d) - half < y1 + m and max(b, d) + half > y0 - m:
                return False
    for a, b, c, d in boxes:
        if a < x1 + m and c > x0 - m and b < y1 + m and d > y0 - m:
            return False
    return all(
        not (x0 - m - 2.6 < a < x1 + m + 2.6 and y0 - m - 2.6 < b < y1 + m + 2.6) for a, b in dots
    )


def _domain_panel(boxes, ox, oy, size, box_labels):
    """Draw the domain of ``boxes`` in a square of ``size`` units, its top left corner at (ox, oy).

    The added boxes are shaded and outlined in dashes, the removed ones painted over
    in paper; the edge of the domain is the heavy rule; a dot marks each corner a
    call passes, with its coordinates. ``box_labels`` places the name of each call:
    ``(u, v)`` in the coordinates of the program.
    """
    xs = [v for _, lower, upper in boxes for v in (lower[0], upper[0])]
    ys = [v for _, lower, upper in boxes for v in (lower[1], upper[1])]
    xmin, ymax = min(xs), max(ys)
    extent = max(max(xs) - xmin, ymax - min(ys))
    unit = size / float(extent)

    def x(u):
        return ox + float(u - xmin) * unit

    def y(v):
        return oy + float(ymax - v) * unit

    g = ""
    # samurai applies every remove after every add: paint them in that order
    for call, lower, upper in sorted(boxes, key=lambda b: b[0] == "remove"):
        w, h = float(upper[0] - lower[0]) * unit, float(upper[1] - lower[1]) * unit
        g += rect(x(lower[0]), y(upper[1]), w, h, fill=PAPER if call == "remove" else WASH, sw=0)
    for call, lower, upper in boxes:
        if call != "remove":
            w, h = float(upper[0] - lower[0]) * unit, float(upper[1] - lower[1]) * unit
            g += rect(x(lower[0]), y(upper[1]), w, h, stroke=INK2, sw=EMPTY, dash="3 3")
    step, grid = _region(boxes)
    d = "".join(
        f"M{num(x(a * step))} {num(y(b * step))}L{num(x(c * step))} {num(y(e * step))}"
        for a, b, c, e in _edges(grid)
    )
    g += path(d, stroke=INK, sw=HEAVY, extra='stroke-linecap="square"')
    # the edges of the domain and the outlines of the boxes, to keep the labels off them
    lines = [(x(a * step), y(b * step), x(c * step), y(e * step)) for a, b, c, e in _edges(grid)]
    for _, lower, upper in boxes:
        x0, y0, x1, y1 = x(lower[0]), y(lower[1]), x(upper[0]), y(upper[1])
        lines += [(x0, y0, x1, y0), (x0, y1, x1, y1), (x0, y0, x0, y1), (x1, y0, x1, y1)]
    taken = []
    for (call, lower, upper), (u, v) in zip(boxes, box_labels):
        # 11 units, or less when the box is too narrow to keep the clearance inside it
        room = float(upper[0] - lower[0]) * unit - 2 * CLEARANCE - HEAVY
        size = min(11, int(2 * room / (0.6 * len(call))) / 2)
        box = _text_box(x(u), y(v) + 0.3 * size, call, "middle", size)
        if not _clear(box, lines, taken, []):
            raise ValueError(f"the label {call} does not keep {CLEARANCE} units clear")
        g += text(
            x(u),
            y(v) + 0.3 * size,
            call,
            anchor="middle",
            size=num(size),
            fill=INK,
            cls="sm-fig-code",
        )
        taken.append(box)
    dots = [(x(u), y(v)) for _, lower, upper in boxes for u, v in (lower, upper)]
    for _, lower, upper in boxes:
        for corner, places in ((lower, _LOWER), (upper, _UPPER)):
            cx, cy = x(corner[0]), y(corner[1])
            label = _point_label(corner)
            others = [d for d in dots if d != (cx, cy)]
            for anchor, dx, dy in places:
                box = _text_box(cx + dx, cy + dy, label, anchor, 10.5)
                if _clear(box, lines, taken, others):
                    break
            else:
                raise ValueError(f"no place for the label {label} keeps {CLEARANCE} units clear")
            taken.append(box)
            g += circle(cx, cy, 2.6, fill=INK)
            g += text(
                cx + dx,
                cy + dy,
                label,
                anchor=anchor,
                size=10.5,
                fill=INK,
                cls="sm-fig-code",
            )
    return g


@figure
def box_domains(_p):
    """Draw the domains of 2d_union.cpp and 2d_box_with_hole.cpp with their corners.

    Figure of howto/box.md. The corners are read from the snippets.
    """
    union, hole = read_domain("2d_union.cpp"), read_domain("2d_box_with_hole.cpp")
    if [c for c, _, _ in union] != ["DomainBuilder", "add"]:
        raise ValueError("2d_union.cpp: expected the constructor and one add")
    if [c for c, _, _ in hole] != ["DomainBuilder", "remove"]:
        raise ValueError("2d_box_with_hole.cpp: expected the constructor and one remove")
    size, oy, xa, xb = 230, 26, 64, 398
    bottom = oy + size + 44

    def center(lower, upper):
        return ((lower[0] + upper[0]) / 2, (lower[1] + upper[1]) / 2)

    # the name of each call at the center of its box; in the box with a hole, the name
    # of the constructor up and to the left of the hole, clear of its corner labels
    (_, a0, a1), (_, b0, b1) = union
    (_, c0, c1), (_, h0, h1) = hole
    g = _domain_panel(union, xa, oy, size, [center(a0, a1), center(b0, b1)])
    g += _domain_panel(
        hole, xb, oy, size, [((c0[0] + h0[0]) / 2, (h1[1] + c1[1]) / 2), center(h0, h1)]
    )
    for x0, name in ((xa, "2d_union.cpp"), (xb, "2d_box_with_hole.cpp")):
        g += text(
            x0 + size / 2, bottom - 4, name, anchor="middle", size=11, fill=INK3, cls="sm-fig-code"
        )

    def corners(lower, upper):
        return f"{_point_label(lower)} to {_point_label(upper)}"

    return Drawing(
        DIAGRAM_WIDTH,
        bottom,
        "Two domains drawn from their boxes. Left, an L shape: the box of the DomainBuilder"
        f" constructor from {corners(a0, a1)} and the box of add from {corners(b0, b1)}, joined"
        " into one domain. Right, the square of the constructor from"
        f" {corners(c0, c1)} with the box of remove from {corners(h0, h1)} cut out as a hole."
        " Each corner passed to a call is marked by a dot and its coordinates.",
        "",
        g,
    )


def _ghost_case(x0, y0, cw, ch, hole, radius):
    """Draw a row of cells across a hole of ``hole`` cells, with the ghost band of each edge.

    The cells are numbered from the left edge of the hole: the domain holds ``radius``
    cells on each side, the hole the cells 0 to ``hole - 1``. Returns the drawing and
    the cells both bands claim.
    """
    left, right = set(range(radius)), set(range(hole - radius, hole))
    both = left & right

    def cx(k):
        return x0 + (k + radius) * cw

    g = ""
    for k in [*range(-radius, 0), *range(hole, hole + radius)]:
        g += rect(cx(k), y0, cw, ch, fill=WASH, stroke=INK)
    for k in range(hole):
        if k in both:
            # a plain red wash, not the hatching, so that the number stays legible
            g += rect(cx(k), y0, cw, ch, fill=RWASH, stroke="none")
        g += rect(cx(k), y0, cw, ch, stroke=RED if k in both else INK2, sw=EMPTY, dash="3 3")
    # the number of each cell, counted from the left edge of the hole
    for k in range(-radius, hole + radius):
        tone = RED if k in both else INK2
        g += text(cx(k) + cw / 2, y0 + ch / 2 + 4, num(k), anchor="middle", size=11, fill=tone)
    # the edges of the hole are edges of the domain
    for k in (0, hole):
        g += line(cx(k), y0 - 4, cx(k), y0 + ch + 4, sw=HEAVY)
    # the ghosts of the left edge above the row, those of the right edge under it
    g += bracket(
        cx(min(left)),
        cx(max(left) + 1),
        y0 - 12,
        f"[{min(left)}, {max(left) + 1})",
        size=11,
        label_dy=-8,
    )
    g += bracket(
        cx(min(right)),
        cx(max(right) + 1),
        y0 + ch + 12,
        f"[{min(right)}, {max(right) + 1})",
        size=11,
        label_dy=17,
    )
    return g, both


@figure
def hole_ghost_bands(_p):
    """Draw the check of a hole: two bands of r ghosts, one per edge, must not overlap.

    Figure of howto/box.md, for the radius mesh_config uses at least (2): a hole of
    2r cells passes the check of Mesh_base::compute_scaling_factor, a hole of 2r - 1
    cells does not.
    """
    radius = hole_rule()
    cw, ch, x0 = 32, 28, 196
    # the names of the bands in one column, right of the longer row
    xr = x0 + 4 * radius * cw
    g = ""
    pitch, top = 112, 34
    for row, hole in enumerate((2 * radius, 2 * radius - 1)):
        y0 = top + row * pitch
        case, both = _ghost_case(x0, y0, cw, ch, hole, radius)
        # the check of mesh.hpp, in cells of the minimum level
        if bool(both) != (hole < 2 * radius):
            raise ValueError("the bands overlap exactly when the hole is shorter than 2r")
        g += case
        g += text(x0 - 22, y0 + 11, f"hole of {hole} cells", anchor="end", size=12, fill=INK)
        if both:
            g += text(
                x0 - 22,
                y0 + 27,
                f'{hole} &lt; 2<tspan class="sm-fig-math">r</tspan>:' " too small",
                anchor="end",
                size=11.5,
                fill=RED,
            )
            k = min(both)
            g += text(
                xr + 16,
                y0 + ch / 2 + 4,
                f"[{k}, {k + 1}) in both bands",
                size=11,
                fill=RED,
                cls="sm-fig-code",
            )
        else:
            g += text(
                x0 - 22,
                y0 + 27,
                f'{hole} = 2<tspan class="sm-fig-math">r</tspan>:' " accepted",
                anchor="end",
                size=11.5,
                fill=INK2,
            )
        g += text(xr + 16, y0 - 8, "ghosts of the left edge", size=11, fill=INK2)
        g += text(xr + 16, y0 + ch + 16, "ghosts of the right edge", size=11, fill=INK2)
    height = top + pitch + ch + 40
    return Drawing(
        DIAGRAM_WIDTH,
        height,
        f"Two rows of cells of the minimum level across a hole, for a stencil radius r = {radius}."
        f" Each row has {radius} domain cells on each side of the hole. Above the row, the"
        f" {radius} ghost cells of the left edge; under it, the {radius} ghost cells of the right"
        f" edge. With a hole of {2 * radius} cells the two bands meet without overlapping and"
        f" the hole is accepted. With a hole of {2 * radius - 1} cells the middle cell is in both"
        " bands and the hole is too small.",
        "",
        g,
    )
