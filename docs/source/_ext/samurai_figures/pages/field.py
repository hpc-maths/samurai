"""The figures of ``tutorial/field.md``: where the values of a field are stored.

The meshes are those of the interval tutorial (``pages/interval.py``). The x
intervals and their indices are read from the outputs of the two programs,
``interval_1d_output.txt`` and ``2d_mesh_representation_output.txt``: the entry of
the cell ``k`` of an interval is ``k + index``. Reading a file checks that its cells
are the ones the interval tutorial draws, and that the entries number them from the
coarsest level to the finest, row by row and from left to right, from 0 without a
gap; a mismatch raises ``ValueError`` and fails the build.
"""

import re
from pathlib import Path

from ..arrays import MONO_ADVANCE, Array, connector, draw_array, interval_token, token_row
from ..cells import Frame, draw_cells, from_intervals
from ..draw import (
    HAIR,
    INK,
    INK2,
    INK3,
    RED,
    RWASH,
    cell_row,
    circle,
    fig_label,
    line,
    num,
    path,
    patterns,
    rect,
    text,
)
from ..registry import DIAGRAM_WIDTH, PLATE_WIDTH, Drawing, figure
from .interval import ARRAYS_2D, MESH_1D, MESH_2D

OUTPUTS = Path(__file__).resolve().parents[3] / "tutorial" / "snippet" / "interval"

# A printed interval of a level cell array: position->[start,end)@index:step.
INTERVAL = re.compile(r"(\d+)->\[(-?\d+),(-?\d+)\)@(-?\d+):(\d+)")


def read_cell_array(name):
    """Read a CellArray printed by ``operator<<`` from the output file ``name``.

    Return ``{level: {dim: (intervals, offsets)}}``, ``intervals`` a list of
    ``(start, end, index)`` and ``offsets`` the offsets printed for the dimension.
    Raise ``ValueError`` on an interval whose step is not 1 or whose position is not
    its place in the printed list.
    """
    levels = {}
    parts = re.split(r"Level (\d+)", (OUTPUTS / name).read_text(encoding="utf-8"))
    for level, block in zip(parts[1::2], parts[2::2]):
        dims = re.split(r"dim (\d+)", block)
        levels[int(level)] = {}
        for d, body in zip(dims[1::2], dims[2::2]):
            intervals = []
            for n, (pos, a, b, i, step) in enumerate(INTERVAL.findall(body)):
                if (int(pos), int(step)) != (n, 1):
                    raise ValueError(f"{name}: unexpected interval {pos}->[{a},{b})@{i}:{step}")
                intervals.append((int(a), int(b), int(i)))
            offsets = [int(v) for v in re.findall(r"\(\d+: (-?\d+)\)", body)]
            levels[int(level)][int(d)] = (intervals, offsets)
    return levels


def _number(rows):
    """Check the entries of ``rows`` and return the entry of each cell.

    ``rows`` are ``(level, others, start, end, index)`` in the order of the cell
    array; the cell ``k`` of a row is stored at ``k + index``. The entries must be
    0, 1, 2, ... in that order.
    """
    entries, n = {}, 0
    for level, others, start, end, index in rows:
        for k in range(start, end):
            if k + index != n:
                raise ValueError(
                    f"the cell {k} of [{start}, {end})@{index} at level {level} is stored at"
                    f" {k + index}, expected {n}"
                )
            entries[(level, k, *others)] = n
            n += 1
    return entries


def _read_1d():
    """Read the x intervals of interval_1d.cpp by level, and the entry of each cell."""
    levels = {
        level: dims[0][0] for level, dims in read_cell_array("interval_1d_output.txt").items()
    }
    cells = {level: [(a, b) for a, b, _ in ivs] for level, ivs in levels.items()}
    if cells != MESH_1D:
        raise ValueError(f"interval_1d_output.txt holds {cells}, the interval tutorial {MESH_1D}")
    rows = [(level, (), a, b, i) for level, ivs in sorted(levels.items()) for a, b, i in ivs]
    return levels, _number(rows)


def _read_2d():
    """Read the rows ``(y, start, end, index)`` of 2D_mesh.cpp by level, and the entries."""
    printed = read_cell_array("2d_mesh_representation_output.txt")
    levels = {}
    for level, dims in sorted(printed.items()):
        xs, ys, offsets = dims[0][0], dims[1][0], dims[1][1]
        rows = []
        for ya, yb, yi in ys:
            for y in range(ya, yb):
                j = y + yi
                rows += [(y, *iv) for iv in xs[offsets[j] : offsets[j + 1]]]
        levels[level] = rows
        arrays = ARRAYS_2D.get(level)
        if arrays and (arrays["y"], arrays["offsets"], arrays["x"]) != (
            ys,
            offsets,
            [(a, b) for a, b, _ in xs],
        ):
            raise ValueError(f"the arrays of level {level} differ from the interval tutorial")
    cells = {level: [(y, a, b) for y, a, b, _ in rows] for level, rows in levels.items()}
    if cells != MESH_2D:
        raise ValueError(
            f"2d_mesh_representation_output.txt holds {cells}, the interval tutorial {MESH_2D}"
        )
    rows = [(level, (y,), a, b, i) for level, rs in levels.items() for y, a, b, i in rs]
    return levels, _number(rows)


LEVELS_1D, ENTRIES_1D = _read_1d()
LEVELS_2D, ENTRIES_2D = _read_2d()

# The examples of the page: an interval, its index and the entries of its cells.
EXAMPLE_1D = (2, 14, 16, -6, (8, 9))
EXAMPLE_2D = (2, 15, 14, 16, 21, (35, 36))
if (
    (14, 16, -6) not in LEVELS_1D[2]
    or tuple(ENTRIES_1D[(2, k)] for k in (14, 15)) != EXAMPLE_1D[4]
    or (15, 14, 16, 21) not in LEVELS_2D[2]
    or tuple(ENTRIES_2D[(2, k, 15)] for k in (14, 15)) != EXAMPLE_2D[5]
):
    raise ValueError("the examples of tutorial/field.md do not match the program outputs")


def _sums(x, y, start, end, index, size=16):
    """Draw the entry of each cell of [start, end)@index, one sum per line, centred at x.

    The last line labels its columns: the cell, the index and the entry. Return the
    drawing and the baseline of the last line.
    """
    shown = f"({index})" if index < 0 else str(index)
    g = ""
    for n, k in enumerate(range(start, end)):
        parts = [
            (str(k), "cell", INK),
            (" + ", None, INK3),
            (shown, "index", RED),
            (" = ", None, INK3),
            (str(k + index), "entry", RED),
        ]
        if n < end - start - 1:
            parts = [(s, None, color) for s, _, color in parts]
        g += token_row(x, y + n * 1.5 * size, parts, size=size, anchor="middle")
    return g, y + (end - start - 1) * 1.5 * size


def _crossings(curve, boxes):
    """Return the boxes ``(x, y, w, h)`` that the cubic ``curve`` passes through."""
    (x1, y1), (cx1, cy1), (cx2, cy2), (x2, y2) = curve
    hits = set()
    for n in range(201):
        t = n / 200
        u = 1 - t
        x = u**3 * x1 + 3 * u * u * t * cx1 + 3 * u * t * t * cx2 + t**3 * x2
        y = u**3 * y1 + 3 * u * u * t * cy1 + 3 * u * t * t * cy2 + t**3 * y2
        hits |= {b for b in boxes if b[0] < x < b[0] + b[2] and b[1] < y < b[1] + b[3]}
    return hits


@figure
def field_storage_1d(p):
    """tutorial/field.md: the cells of interval_1d.cpp and the entries that store them."""
    x0, unit, pitch, ch = 112, 92, 44, 24
    level_2, start, end, index, entries = EXAMPLE_1D
    # The array is above the mesh: a connector that leaves the centre of a cell
    # upwards crosses the coarser rows inside their slots, never along a slot edge.
    arr = Array(PLATE_WIDTH / 2 - 200, 52, 40, len(ENTRIES_1D))
    top = arr.bottom + 100
    rows, boxes, links = "", {}, []
    for level, ivs in LEVELS_1D.items():
        y, cw = top + level * pitch, unit / 2**level
        red = level == level_2
        rows += cell_row(
            x0,
            y,
            cw,
            6 * 2**level,
            [(a, b) for a, b, _ in ivs],
            ch=ch,
            tone="red" if red else "ink",
            brackets=False,
        )
        rows += text(
            x0 - 14, y + 16, f"level {level}", anchor="end", size=12, fill=RED if red else INK
        )
        for a, b, _ in ivs:
            for k in range(a, b):
                boxes[(level, k)] = (x0 + k * cw, y, cw, ch)
                rows += text(
                    x0 + (k + 0.5) * cw,
                    y + 16,
                    k,
                    anchor="middle",
                    size=10.5 if level == 2 else 11.5,
                    fill=RED if red else INK,
                )
    # one connector from each cell to its entry; it may cross empty slots, not a cell
    for cell, entry in ENTRIES_1D.items():
        bx, by, bw, _ = boxes[cell]
        x1, y1, x2, y2 = bx + bw / 2, by, arr.cx(entry), arr.bottom
        ym = (y1 + y2) / 2
        others = [b for c, b in boxes.items() if c != cell]
        if _crossings(((x1, y1), (x1, ym), (x2, ym), (x2, y2)), others):
            raise ValueError(f"the connector of the cell {cell} crosses another cell")
        links.append((cell[0] == level_2, connector(x1, y1, x2, y2, red=cell[0] == level_2)))
    g = "".join(s for _, s in sorted(links, key=lambda link: link[0])) + rows
    g += draw_array(arr, red=entries, name="u", name_cls="sm-fig-math", positions=False)
    # the position of each entry above it, and the entries of each level above them
    for e in range(arr.n):
        g += text(
            arr.cx(e), arr.y - 7, e, anchor="middle", size=10.5, fill=RED if e in entries else INK2
        )
    for level in LEVELS_1D:
        mine = [e for (lv, _), e in ENTRIES_1D.items() if lv == level]
        xa, xb, yg = arr.left(min(mine)) + 3, arr.left(max(mine) + 1) - 3, arr.y - 26
        g += path(
            f"M{num(xa)} {num(yg + 4)}V{num(yg)}H{num(xb)}V{num(yg + 4)}", stroke=INK3, sw=HAIR
        )
        g += text((xa + xb) / 2, yg - 6, f"level {level}", anchor="middle", size=11, fill=INK2)
    bottom1 = top + 2 * pitch + ch
    g += fig_label(PLATE_WIDTH / 2, bottom1 + 38, 1)

    # Fig. 2: the printed interval of level 2 and the entries its index gives
    top2 = bottom1 + 60
    split = top2 - 12
    g += line(24, split, PLATE_WIDTH - 24, split, stroke=INK3, sw=HAIR)
    g += token_row(
        PLATE_WIDTH / 2,
        top2 + 30,
        interval_token(start, end, index, red=("index",)),
        anchor="middle",
    )
    sums, last = _sums(PLATE_WIDTH / 2, top2 + 96, start, end, index)
    g += sums
    g += fig_label(PLATE_WIDTH / 2, last + 62, 2)
    height = last + 74
    return Drawing(
        PLATE_WIDTH,
        height,
        "Fig. 1: at the top, the field array u of 10 entries, grouped by level: entries 0 to"
        " 2 for level 0, 3 to 7 for level 1, 8 and 9, in red, for level 2. Under it, the cells"
        " of the 1D mesh of interval_1d.cpp on one row per level, each with its x index: level"
        " 0 holds 0, 1 and 5, level 1 holds 4, 5, 6, 8 and 9, level 2, in red, holds 14 and 15."
        " A connector links each cell to its entry; the cell 5 of level 0, at the right end of"
        " the mesh, is entry 2, before the cells of level 1. Fig. 2: the printed interval"
        " [14,16)@-6:1 of level 2 with its index -6 in red, and the entries of its cells:"
        " 14 + (-6) = 8 and 15 + (-6) = 9.",
        patterns(p),
        g,
        panels=(
            (0, 0, PLATE_WIDTH, split, "wide"),
            (PLATE_WIDTH / 2 - 150, split, 300, height - split, ""),
        ),
    )


@figure
def field_storage_2d(p):
    """tutorial/field.md: the mesh of 2D_mesh.cpp with the entry of each cell."""
    level, row, start, end, index, _ = EXAMPLE_2D
    size, ox, oy = 336, 20, 30
    frame = Frame((0, 0, 4, 4), ox, oy, size)
    red = {(level, k, row) for k in range(start, end)}
    g = draw_cells(
        p,
        from_intervals(MESH_2D),
        frame,
        hatch=False,
        fill=lambda leaf: RWASH if leaf in red else None,
    )
    for leaf in sorted(red):
        g += rect(*frame.cell(leaf), stroke=RED, sw=HAIR)
    for leaf, entry in ENTRIES_2D.items():
        x, y = frame.center(leaf)
        fs = (13, 12, 10)[leaf[0]]
        g += text(
            x,
            y + 0.36 * fs,
            entry,
            anchor="middle",
            size=fs,
            fill=RED if leaf in red else INK,
        )
    # the row, its printed interval and the entries of its cells
    cx = (ox + size + DIAGRAM_WIDTH) / 2 + 6
    _, yr = frame.center((level, start, row))
    parts = interval_token(start, end, index, red=("index",))
    width = MONO_ADVANCE * 18 * len("".join(s for s, _, _ in parts))
    g += line(ox + size, yr, cx - width / 2 - 12, yr, stroke=RED, sw=HAIR)
    g += circle(ox + size, yr, 1.8, fill=RED)
    g += text(
        cx,
        yr - 22,
        f'level {level}, row <tspan class="sm-fig-math">y</tspan> = {row}',
        anchor="middle",
        size=12,
        fill=RED,
    )
    g += token_row(cx, yr + 6, parts, anchor="middle")
    sums, _ = _sums(cx, yr + 84, start, end, index)
    g += sums
    return Drawing(
        DIAGRAM_WIDTH,
        oy + size + 16,
        "The 2D mesh of 2D_mesh.cpp with the entry of each cell in the field array: 0 to 10"
        " for the cells of level 0, 11 to 28 for level 1 and 29 to 36 for level 2, each level"
        " numbered row by row from the bottom and from left to right. The row y = 15 of level"
        " 2, in the upper right corner, is in red; a leader links it to its printed interval"
        " [14,16)@21:1, whose index 21 gives the entries 14 + 21 = 35 and 15 + 21 = 36.",
        patterns(p),
        g,
    )
