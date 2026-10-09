"""Meshes as a samurai program produced them, from intervals or an exported cell list.

``mesh.py`` builds graded meshes of the unit square from a refinement rule. This
module draws the cells a program really has, given in one of two ways:

- ``from_intervals``: the intervals of each level, written as the program writes
  them in its ``CellList`` (``cl[1][{4}].add_interval({2, 4})`` is the row
  ``(4, 2, 4)`` of level 1);
- ``load_cells``: the JSON file that ``docs/tools/export_cells.py`` writes from the
  ``.h5`` file of the program, for a mesh Python cannot rebuild (a random mesh, a
  mesh adapted to a solution, the subdomains of an MPI run). Keep the JSON file
  in ``samurai_figures/data/``.

A cell is ``(level, i, j)``. It covers [i/2^l, (i+1)/2^l) x [j/2^l, (j+1)/2^l) in
level-0 units, half-open like every samurai interval, and the point ``(u, v)`` of
level-0 units lies at ``origin + scaling_factor * (u, v)`` in the coordinates of
the program. A ``Frame`` places a box of those coordinates in the drawing; the box
can be any part of the domain, not only the unit square.

``draw_cells`` checks the mesh before drawing it: the cells must not overlap and
the mesh must be graded, unless the figure says otherwise with ``overlap=True`` or
``graded=False``. A figure that passes ``graded=False`` shows a mesh samurai does
not keep, so its legend must say so.
"""

import json
from dataclasses import dataclass

from .draw import EMPTY, HAIR, HEAVY, INK, PAPER, line, num, path, rect, text

FORMAT, VERSION = "samurai-cells", 1


def from_intervals(levels):
    """The cells of the intervals ``levels``, sorted by level, then y, then x.

    ``levels`` maps a level (an int, or a string as in the JSON file) to its rows
    ``(y, start, end)``: the cells ``start`` to ``end - 1`` along x on the row
    ``y``. A cell is ``(level, i, j)``. In 1D a row is ``(start, end)`` and a cell
    ``(level, i)``.
    """
    leaves = []
    for level, rows in levels.items():
        for *others, start, end in rows:
            leaves += [(int(level), i, *others) for i in range(start, end)]
    return sorted(leaves, key=lambda c: (c[0], c[2:], c[1]))


@dataclass(frozen=True)
class Exported:
    """The mesh of a JSON file written by ``docs/tools/export_cells.py``.

    ``leaves`` are its cells; ``ranks`` gives the MPI rank of each cell, in the
    same order, or is None when the program ran without MPI. ``source`` names the
    program and the options that produced it.
    """

    dim: int
    leaves: tuple
    ranks: tuple
    scaling_factor: float
    origin: tuple
    source: str


def load_cells(file):
    """Read the JSON file ``file`` (a path) written by ``export_cells.py``.

    The JSON files of the figures live in ``samurai_figures/data/``.

    Raise ``ValueError`` for a file of another format or version.
    """
    with open(file, encoding="utf-8") as stream:
        document = json.load(stream)
    if document.get("format") != FORMAT or document.get("version") != VERSION:
        raise ValueError(f"{file} is not a {FORMAT} file of version {VERSION}")
    if "ranks" in document:
        leaves, ranks = [], []
        for part in document["ranks"]:
            cells = from_intervals(part["levels"])
            leaves += cells
            ranks += [part["rank"]] * len(cells)
        ranks = tuple(ranks)
    else:
        leaves, ranks = from_intervals(document["levels"]), None
    return Exported(
        document["dim"],
        tuple(leaves),
        ranks,
        document["scaling_factor"],
        tuple(document["origin"]),
        document.get("source", ""),
    )


def _overlaps(leaves):
    """The pairs (coarse, fine) of cells where the fine cell lies in the coarse one."""
    present = set(leaves)
    levels = sorted({leaf[0] for leaf in leaves})
    pairs = []
    for level, i, j in leaves:
        for coarse in levels:
            if coarse >= level:
                break
            shift = level - coarse
            # >> floors, so negative indices find their parent too
            parent = (coarse, i >> shift, j >> shift)
            if parent in present:
                pairs.append((parent, (level, i, j)))
    return pairs


def check_disjoint(leaves):
    """Raise ``ValueError`` when two cells overlap, or a cell is listed twice."""
    if len(set(leaves)) != len(leaves):
        raise ValueError("a cell is listed twice")
    pairs = _overlaps(leaves)
    if pairs:
        coarse, fine = pairs[0]
        raise ValueError(
            f"the cells overlap: {fine} lies in {coarse}; pass overlap=True to draw them so"
        )


def _finest(leaves):
    """The finest level of ``leaves`` and the level seen at each of its cells.

    Where cells overlap, the finest one is seen, as it is drawn on top.
    """
    top = max(leaf[0] for leaf in leaves)
    grid = {}
    for level, i, j in sorted(leaves):
        s = 2 ** (top - level)
        for a in range(s):
            for b in range(s):
                grid[i * s + a, j * s + b] = level
    return top, grid


def check_graded(leaves):
    """Raise ``ValueError`` when two touching cells differ by more than one level.

    Two cells touch when they share a face or a corner, as in samurai. The cells
    can cover any box and leave holes, as the domain of a program can; cells on
    both sides of a hole do not touch. Where cells overlap, the finest one counts.
    """
    top, grid = _finest(leaves)

    def cell(x, y, level):
        shift = top - level
        return (level, x >> shift, y >> shift)

    for (x, y), level in grid.items():
        for dx, dy in ((1, 0), (0, 1), (1, 1), (1, -1)):
            other = grid.get((x + dx, y + dy))
            if other is not None and abs(level - other) > 1:
                raise ValueError(
                    f"the mesh is not graded: the cell {cell(x, y, level)} touches the cell"
                    f" {cell(x + dx, y + dy, other)}; pass graded=False only if the figure"
                    " shows such a mesh"
                )


def _boundary(grid):
    """The edges of the region covered by ``grid``, merged into segments.

    Each segment is ``(x1, y1, x2, y2)`` in cells of the finest level.
    """
    horizontal, vertical = {}, {}
    for x, y in grid:
        if (x, y - 1) not in grid:
            horizontal.setdefault(y, []).append(x)
        if (x, y + 1) not in grid:
            horizontal.setdefault(y + 1, []).append(x)
        if (x - 1, y) not in grid:
            vertical.setdefault(x, []).append(y)
        if (x + 1, y) not in grid:
            vertical.setdefault(x + 1, []).append(y)

    def runs(starts):
        starts = sorted(set(starts))
        out, first = [], starts[0]
        for prev, cur in zip(starts, starts[1:] + [None]):
            if cur != prev + 1:
                out.append((first, prev + 1))
                first = cur
        return out

    segments = [(a, y, b, y) for y, xs in horizontal.items() for a, b in runs(xs)]
    segments += [(x, a, x, b) for x, ys in vertical.items() for a, b in runs(ys)]
    return sorted(segments)


@dataclass(frozen=True)
class Frame:
    """A box of the program's coordinates, placed in the drawing.

    The box ``(xmin, ymin, xmax, ymax)`` is drawn ``width`` units wide, its top left
    corner at (ox, oy), y up. ``scaling_factor`` and ``origin`` are the mesh's: a
    cell of level l has the side ``scaling_factor / 2**l`` and the cell ``(0, 0)``
    of every level starts at ``origin``. ``Exported`` carries both for an exported
    mesh.
    """

    box: tuple
    ox: float
    oy: float
    width: float
    scaling_factor: float = 1.0
    origin: tuple = (0.0, 0.0)

    @property
    def unit(self):
        """Drawing units per unit of the program's coordinates."""
        return self.width / (self.box[2] - self.box[0])

    @property
    def height(self):
        """The height of the box in the drawing."""
        return (self.box[3] - self.box[1]) * self.unit

    def x(self, u):
        """The drawing abscissa of the coordinate ``u``."""
        return self.ox + (u - self.box[0]) * self.unit

    def y(self, v):
        """The drawing ordinate of the coordinate ``v``, y up."""
        return self.oy + (self.box[3] - v) * self.unit

    def corner(self, level, i, j):
        """The lower left corner of a cell, in the program's coordinates."""
        h = self.scaling_factor * 2.0**-level
        return self.origin[0] + i * h, self.origin[1] + j * h

    def cell(self, leaf):
        """The drawing rectangle ``(x, y, w, h)`` of the cell ``leaf``."""
        level, i, j = leaf
        u, v = self.corner(level, i, j)
        side = self.scaling_factor * 2.0**-level * self.unit
        return self.x(u), self.y(v) - side, side, side

    def center(self, leaf):
        """The drawing point at the center of the cell ``leaf``, for a callout."""
        x, y, w, h = self.cell(leaf)
        return x + w / 2, y + h / 2

    def inside(self, leaf):
        """Whether the cell ``leaf`` meets the inside of the box."""
        x, y, w, h = self.cell(leaf)
        right, bottom = self.x(self.box[2]), self.y(self.box[1])
        return x < right and x + w > self.ox and y < bottom and y + h > self.oy


def _axes(frame, xticks, yticks):
    """Ticks and labels under and left of the box, as in ``mesh.draw_mesh``."""
    s = ""
    bottom = frame.oy + frame.height
    for u, label in xticks:
        s += line(frame.x(u), bottom, frame.x(u), bottom + 5)
        s += text(frame.x(u), bottom + 18, label, anchor="middle", size=11)
    for v, label in yticks:
        s += line(frame.ox - 5, frame.y(v), frame.ox, frame.y(v))
        s += text(frame.ox - 8, frame.y(v) + 4, label, anchor="end", size=11)
    return s


def draw_cells(
    p,
    leaves,
    frame,
    graded=True,
    overlap=False,
    hatch=True,
    fill=None,
    xticks=(),
    yticks=(),
):
    """Draw the 2D cells ``leaves`` in ``frame``, with ticks ``(value, label)``.

    The cells of the finest level are hatched in red (``hatch=False`` leaves them
    blank) and outlined with the hairline, the others with the faint outline. The
    edge of the domain, holes included, is the heavy rule. ``fill(leaf)`` returns
    the fill of a cell, or None for the default, such as the fill of its rank.
    Cells outside the box are left out, cells across its edge are cut.

    With ``overlap=True`` a finer cell is drawn over the coarser cell it lies in,
    and the coarser cell is outlined with the heavy rule on top, so that both stay
    visible. ``graded=False`` skips ``check_graded``. ``p`` prefixes the ids, see
    ``draw.patterns``.
    """
    if any(len(leaf) != 3 for leaf in leaves):
        raise ValueError("draw_cells draws 2D cells (level, i, j)")
    if not overlap:
        check_disjoint(leaves)
    if graded:
        check_graded(leaves)
    top, grid = _finest(leaves)
    pairs = _overlaps(leaves)
    covered, under = {coarse for coarse, _ in pairs}, {fine for _, fine in pairs}

    cells = ""
    for leaf in sorted(leaves):
        if not frame.inside(leaf):
            continue
        x, y, w, h = frame.cell(leaf)
        finest = leaf[0] == top
        own = fill(leaf) if fill else None
        if own is None:
            own = f"url(#{p}-r)" if finest and hatch else (PAPER if leaf in under else "none")
        cells += rect(x, y, w, h, fill=own, stroke=INK, sw=HAIR if finest else EMPTY)
    for leaf in sorted(covered):
        if frame.inside(leaf):
            cells += rect(*frame.cell(leaf), stroke=INK, sw=HEAVY)

    # the edge of the domain, in cells of the finest level
    h = frame.scaling_factor * 2.0**-top
    x0, y0 = frame.origin
    d = "".join(
        f"M{num(frame.x(x0 + a * h))} {num(frame.y(y0 + b * h))}"
        f"L{num(frame.x(x0 + c * h))} {num(frame.y(y0 + e * h))}"
        for a, b, c, e in _boundary(grid)
    )
    cells += path(d, stroke=INK, sw=HEAVY, extra='stroke-linecap="square"')

    # Cut the cells at the box, but not the heavy rule of a domain edge that lies
    # on the box: grow the box by the heavy rule on those sides.
    xs, ys = [x for x, _ in grid], [y for _, y in grid]
    left, right = frame.x(x0 + min(xs) * h), frame.x(x0 + (max(xs) + 1) * h)
    top_edge, bottom = frame.y(y0 + (max(ys) + 1) * h), frame.y(y0 + min(ys) * h)
    x1, x2 = frame.ox, frame.ox + frame.width
    y1, y2 = frame.oy, frame.oy + frame.height
    x1 -= HEAVY if left >= x1 - 1e-6 else 0
    x2 += HEAVY if right <= x2 + 1e-6 else 0
    y1 -= HEAVY if top_edge >= y1 - 1e-6 else 0
    y2 += HEAVY if bottom <= y2 + 1e-6 else 0
    clip = f"{p}-cl{round(frame.ox)}-{round(frame.oy)}"
    s = (
        f'<clipPath id="{clip}"><rect x="{num(x1)}" y="{num(y1)}" width="{num(x2 - x1)}"'
        f' height="{num(y2 - y1)}"/></clipPath><g clip-path="url(#{clip})">{cells}</g>'
    )
    return s + _axes(frame, xticks, yticks)
