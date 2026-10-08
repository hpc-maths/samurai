"""The figures of ``howto/bc.md``: the ghost values the boundary conditions write.

Both programs of the page build 4 by 4 cells of length h = 0.25 on the unit square,
attach their conditions and print the first layer of ghosts along the sides of the
square, with the center of each ghost and its value (``snippet/bc/print_ghosts.hpp``).
The figures read those values from the checked-in outputs
``bc_dirichlet_neumann_output.txt`` and ``bc_regions_output.txt``, and the initial
values of the fields from the programs themselves. Before drawing, they recompute
every printed ghost from the conditions of the program and raise if one differs.

Every ghost along a side holds its printed value, and a missing one raises; a ghost a
condition writes is drawn in red. ``print_ghosts`` skips the corner ghosts, which no
condition writes (``update_ghost_mr`` fills them from the inner cells): they stay blank.
"""

import re
from pathlib import Path

from ..draw import EMPTY, FIG, HEAVY, INK, INK2, RED, RULE, RWASH, WASH, line, path, rect, text
from ..registry import DIAGRAM_WIDTH, Drawing, figure

SNIPPETS = Path(__file__).resolve().parents[3] / "howto" / "snippet" / "bc"

N = 4  # cells per side
H = 1 / N  # cell length
C = 44  # drawn cell size
GHOST_DASH = "3 3"
SIDES = ("left", "right", "bottom", "top")

# --- numbers --------------------------------------------------------------------------

_HEADER = re.compile(r"^(.+) ghosts:$")
_GHOST = re.compile(r"^\s+\((\S+), (\S+)\): (.+)$")
_FIELD = re.compile(r'make_(?:scalar|vector)_field<double(?:, (\d+))?>\("(\w+)", mesh, ([^)]*)\)')


def _value(s):
    """Parse a printed value: ``-0.75`` or ``{-1.,  3.}``, returned as a tuple."""
    return tuple(float(v) for v in s.strip("{} ").split(","))


def _index(x):
    """Return the index of the cell whose center is ``x``: -1 for a ghost before 0."""
    return round(x / H - 0.5)


def read_ghosts(program):
    """Read the output of ``program``: ``{"u, left": {(i, j): values}}``.

    ``(i, j)`` are the indices of the ghost, ``values`` a tuple of its components.
    """
    blocks, current = {}, ""
    for row in (SNIPPETS / f"{program}_output.txt").read_text(encoding="utf-8").splitlines():
        if match := _HEADER.match(row):
            current = match[1]
            blocks[current] = {}
        elif current and (match := _GHOST.match(row)):
            blocks[current][(_index(float(match[1])), _index(float(match[2])))] = _value(match[3])
        else:
            raise ValueError(f"{program}_output.txt: unexpected line {row!r}")
    return blocks


def read_fields(program):
    """Return the initial value of each field of ``program``, read from its source.

    A vector field built from one value holds it in every component.
    """
    source = (SNIPPETS / f"{program}.cpp").read_text(encoding="utf-8")
    fields = {}
    for n_comp, name, values in _FIELD.findall(source):
        values = _value(values)
        fields[name] = values * int(n_comp) if n_comp and len(values) == 1 else values
    return fields


def _check(program, block, ghosts, expected):
    """Raise if the ghosts printed in ``block`` differ from ``expected(i, j)``."""
    for (i, j), printed in ghosts.items():
        want = expected(i, j)
        if len(want) != len(printed) or any(abs(a - b) > 1e-12 for a, b in zip(want, printed)):
            raise ValueError(
                f"{program}, {block} ghost ({i}, {j}): {printed} printed, {want} expected"
            )


def dirichlet(value, inner):
    """Return the ghost of ``Dirichlet<1>``, ``2 v - u_0``, component by component."""
    return tuple(2 * v - u for v, u in zip(value, inner))


def neumann(value, inner):
    """Return the ghost of ``Neumann<1>``, ``u_0 + h v``, component by component."""
    return tuple(u + H * v for v, u in zip(value, inner))


def dirichlet_neumann_ghosts():
    """Return the fields and the printed ghosts of ``bc_dirichlet_neumann.cpp``, checked.

    u = 1 has ``Dirichlet<1>`` 0 on the left, ``Dirichlet<1>`` y on the right and
    ``Neumann<1>`` 0 at the bottom and the top; v = (1, 1) has ``Dirichlet<1>`` (0, 2)
    on the whole boundary.
    """
    program = "bc_dirichlet_neumann"
    fields, ghosts = read_fields(program), read_ghosts(program)
    u, v = fields["u"], fields["v"]
    expected = {
        "u, left": lambda i, j: dirichlet((0,), u),
        "u, right": lambda i, j: dirichlet(((j + 0.5) * H,), u),
        "u, bottom": lambda i, j: neumann((0,), u),
        "u, top": lambda i, j: neumann((0,), u),
    }
    for side in SIDES:
        expected[f"v, {side}"] = lambda i, j: dirichlet((0, 2), v)
    if set(ghosts) != set(expected):
        raise ValueError(f"{program}: blocks {sorted(ghosts)}, expected {sorted(expected)}")
    for block, check in expected.items():
        _check(program, block, ghosts[block], check)
    return fields, ghosts


def column_ghosts():
    """Return w and the printed ghosts of the column region of ``bc_regions.cpp``, checked.

    w = 0 has ``Dirichlet<1>`` 1 on the cells of the left column only, so the ghosts
    beside that column receive 2 and the other ghosts keep 0.
    """
    program = "bc_regions"
    fields, ghosts = read_fields(program), read_ghosts(program)
    w = fields["w"]
    blocks = [f"w, {side}" for side in SIDES]
    for block in blocks:
        _check(program, block, ghosts[block], lambda i, j: dirichlet((1,), w) if i <= 0 else w)
    return w, _merge(ghosts, blocks)


def _merge(ghosts, blocks):
    """Merge the printed ``blocks`` into one map, and raise unless it has every side ghost."""
    merged = {key: value for block in blocks for key, value in ghosts[block].items()}
    missing = [cell for cell in _all_cells() if _is_side_ghost(*cell) and cell not in merged]
    if missing:
        raise ValueError(f"{blocks}: the side ghosts {missing} are not printed")
    return merged


# --- drawing --------------------------------------------------------------------------

# The style of each kind of cell: its fill, its outline, the weight and the dash of the
# outline, and the priority of the outline on an edge two cells share.
ROLES = {
    "written": (RWASH, RED, RULE, GHOST_DASH, 3),
    "region": (WASH, INK, RULE, "", 2),
    "inner": ("none", FIG, EMPTY, "", 1),
    "ghost": ("none", FIG, EMPTY, GHOST_DASH, 0),
}


def _is_side_ghost(i, j):
    """Tell whether (i, j) is a ghost beside a side, not a corner ghost nor an inner cell."""
    return [0 <= i < N, 0 <= j < N].count(False) == 1


class Grid:
    """The 4 by 4 cells and their first ghost layer, ``C`` units per cell.

    ``(x0, y0)`` is the top-left corner of the top-left corner ghost. Cells are
    indexed as in samurai, ``i`` to the right and ``j`` upward, from -1 to 4.
    """

    def __init__(self, x0, y0):
        """Place the grid with the top-left corner of its ghost layer at (x0, y0)."""
        self.x0, self.y0 = x0, y0

    def x(self, i):
        """Return the left edge of the column ``i``."""
        return self.x0 + (i + 1) * C

    def y(self, j):
        """Return the top edge of the row ``j``."""
        return self.y0 + (N - j) * C

    @property
    def right(self):
        """Return the right edge of the ghost layer."""
        return self.x(N + 1)

    @property
    def bottom(self):
        """Return the bottom edge of the ghost layer."""
        return self.y(-2)

    def cells(self, roles):
        """Draw the cells, ``roles`` mapping (i, j) to a key of ``ROLES``.

        Each edge is drawn once, in the style of the cell of higher priority on its
        two sides, so that the dashes of two neighbours never overlap. The edges on
        the boundary of the domain are left to ``boundary``.
        """
        g, edges = "", {}
        for (i, j), role in roles.items():
            fill = ROLES[role][0]
            if fill != "none":
                g += rect(self.x(i), self.y(j), C, C, fill=fill, stroke="none")
            for edge in (
                (i, j, i + 1, j),
                (i, j + 1, i + 1, j + 1),
                (i, j, i, j + 1),
                (i + 1, j, i + 1, j + 1),
            ):
                if edge not in edges or ROLES[role][4] > ROLES[edges[edge]][4]:
                    edges[edge] = role
        for (i1, j1, i2, j2), role in edges.items():
            if _on_boundary(i1, j1, i2, j2):
                continue
            _, stroke, sw, dash, _ = ROLES[role]
            g += line(self.x(i1), self.y(j1 - 1), self.x(i2), self.y(j2 - 1), stroke, sw, dash)
        return g

    def boundary(self, red=None):
        """Draw the boundary of the domain in ink, and the polyline ``red`` over it in red.

        ``red`` is a list of mesh nodes (i, j), the corners of the cells.
        """
        g = rect(self.x(0), self.y(N - 1), N * C, N * C, stroke=INK, sw=HEAVY)
        if red:
            d = " ".join(
                f"{'M' if k == 0 else 'L'}{self.x(i)},{self.y(j - 1)}"
                for k, (i, j) in enumerate(red)
            )
            g += path(d, stroke=RED, sw=HEAVY, extra='stroke-linejoin="miter"')
        return g

    def value(self, i, j, s, fill=INK, size=11):
        """Write the value ``s`` in the mono at the center of the cell (i, j)."""
        return text(
            self.x(i) + C / 2, self.y(j) + C / 2 + 4, s, "middle", size, fill, "sm-fig-code"
        )


def _on_boundary(i1, j1, i2, j2):
    """Tell whether the edge from node (i1, j1) to node (i2, j2) lies on the boundary."""
    if i1 == i2:
        return i1 in (0, N) and 0 <= min(j1, j2) < N
    return j1 in (0, N) and 0 <= min(i1, i2) < N


def _show(values):
    """Format the components of a value as the figures print them: ``-0.75``, ``-1, 3``."""
    return ", ".join(f"{v:g}" for v in values)


def _all_cells():
    """Return every (i, j) of the grid, ghosts included."""
    return [(i, j) for i in range(-1, N + 1) for j in range(-1, N + 1)]


# The baseline skip from the name of a condition to its formula: the boxes of the two
# lines keep 4 units between them.
LINE = 23


def _condition(x, y, name, formula, anchor="middle"):
    """Write a condition: its name in the sans over its formula in the serif italic."""
    return text(x, y, name, anchor, 11.5, INK2) + text(
        x, y + LINE, formula, anchor, 14, INK, "sm-fig-math"
    )


def _field_name(grid, name):
    """Write the name of the field in the serif, centered over the grid."""
    return text((grid.x0 + grid.right) / 2, grid.y0 - 57, name, "middle", 17, INK, "sm-fig-math")


def _field_grid(grid, inner, printed):
    """Draw a field: its inner values, its side ghosts with their values, blank corners."""
    roles = {}
    g = ""
    size = 11 if len(inner) == 1 else 10  # a pair of values must clear the dashed edges
    for i, j in _all_cells():
        if 0 <= i < N and 0 <= j < N:
            roles[(i, j)] = "inner"
            g += grid.value(i, j, _show(inner), INK2, size)
        else:
            roles[(i, j)] = "written" if _is_side_ghost(i, j) else "ghost"
    for (i, j), values in printed.items():
        g += grid.value(i, j, _show(values), INK, size)
    return grid.cells(roles) + g + grid.boundary(red=[(0, 0), (N, 0), (N, N), (0, N), (0, 0)])


DERIVATIVE = "\u2202u/\u2202n = 0"
BOLD_V = '<tspan font-weight="700" font-style="normal">v</tspan>'


@figure
def ghost_values(p):
    """Draw the ghost values of ``bc_dirichlet_neumann.cpp`` on its 4 by 4 mesh.

    Figure of howto/bc.md, "Check the ghost values". The fields u and v, the
    condition of each side and the ghosts the program prints along every side.
    """
    del p
    fields, ghosts = dirichlet_neumann_ghosts()
    label = 44  # room for the condition of a side
    gap = 20
    width = label + 10 + 6 * C + 10 + label + gap + 6 * C
    left = (DIAGRAM_WIDTH - width) / 2
    top = 78
    gu = Grid(left + label + 10, top)
    gv = Grid(left + width - 6 * C, top)

    g = _field_grid(gu, fields["u"], _merge(ghosts, [f"u, {side}" for side in SIDES]))
    g += _field_grid(gv, fields["v"], _merge(ghosts, [f"v, {side}" for side in SIDES]))
    middle = gu.y(N // 2 - 1) - 8
    centre = (gu.x0 + gu.right) / 2
    g += _condition(gu.x0 - 10, middle, "Dirichlet", "u = 0", "end")
    g += _condition(gu.right + 10, middle, "Dirichlet", "u = y", "start")
    g += _condition(centre, gu.y0 - 13 - LINE, "Neumann", DERIVATIVE)
    g += _condition(centre, gu.bottom + 20, "Neumann", DERIVATIVE)
    v_formula = f"{BOLD_V} = (0, 2)"
    g += _condition((gv.x0 + gv.right) / 2, gv.bottom + 20, "Dirichlet, every side", v_formula)
    g += _field_name(gu, "u") + _field_name(gv, BOLD_V)
    return Drawing(
        DIAGRAM_WIDTH,
        gu.bottom + 52,
        "Two grids of 4 by 4 cells, each in a ring of dashed ghost cells, the boundary of the"
        " domain in red. Left, the field u, equal to 1 in every cell: the left side is"
        " labeled Dirichlet u = 0 and its four ghosts hold -1; the right side is labeled"
        " Dirichlet u = y and its ghosts hold -0.75, -0.25, 0.25 and 0.75 from bottom to"
        " top; the bottom and top sides are labeled Neumann du/dn = 0 and their ghosts"
        " hold 1. Right, the vector field v, equal to (1, 1) in every cell, with Dirichlet"
        " (0, 2) on every side: all its ghosts along the sides hold (-1, 3). The ghosts"
        " along the sides are washed in red; the four corner ghosts of each grid are"
        " blank, with a faint outline.",
        "",
        g,
    )


@figure
def column_region(p):
    """Draw the left column region of ``bc_regions.cpp`` and the ghosts it writes.

    Figure of howto/bc.md, "By cells". The cells of the left column touch the left
    side, and the first and the last of them the bottom and the top sides: the
    condition writes the ghosts beyond those faces.
    """
    del p
    w, ghosts = column_ghosts()
    grid = Grid((DIAGRAM_WIDTH - 6 * C) / 2, 22)
    roles, g = {}, ""
    for i, j in _all_cells():
        if 0 <= i < N and 0 <= j < N:
            roles[(i, j)] = "region" if i == 0 else "inner"
            g += grid.value(i, j, _show(w), INK if i == 0 else INK2)
        elif i == 0 or (i == -1 and 0 <= j < N):
            roles[(i, j)] = "written"
        else:
            roles[(i, j)] = "ghost"
    for (i, j), value in ghosts.items():
        g += grid.value(i, j, _show(value), INK if roles[(i, j)] == "written" else INK2)
    g = grid.cells(roles) + g + grid.boundary(red=[(1, 0), (0, 0), (0, N), (1, N)])
    g += _condition(grid.x0 - 10, grid.y(N // 2 - 1) - 8, "Dirichlet", "w = 1", "end")
    return Drawing(
        DIAGRAM_WIDTH,
        grid.bottom + 12,
        "A grid of 4 by 4 cells in a ring of dashed ghost cells; w is 0 in every cell. The"
        " left column of cells, shaded, is the region of the condition labeled Dirichlet"
        " w = 1. Its faces on the boundary are red: the four on the left side, the bottom"
        " face of its bottom cell and the top face of its top cell. The ghosts beyond these"
        " faces are washed in red: the four left ghosts hold 2, the ghost below the column"
        " and the ghost above it hold 2. The other ghosts along the sides, with a faint"
        " dashed outline, hold 0. The four corner ghosts are blank.",
        "",
        g,
    )
