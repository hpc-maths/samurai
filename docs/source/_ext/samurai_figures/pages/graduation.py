"""The figure of ``tutorial/graduation.md``: what a graded mesh allows.

The three meshes of the figure share one layout on the unit square: a cell of
level 1 in the bottom left quarter, cells of level 2 elsewhere, and one cell of
level 2 replaced by its four children of level 3. Only the place of that cell
changes. ``_jumps`` lists the touching cells more than one level apart and the
figure checks the claim of each mesh with it: none, a face contact, a corner
contact. A run of samurai on the same meshes agrees: ``make_graduation`` leaves
the first one as it is and refines the cell of level 1 in the other two.
"""

from ..cells import Frame, draw_cells
from ..draw import HEAVY, INK, INK2, RED, circle, line, patterns, text
from ..registry import DIAGRAM_WIDTH, Drawing, figure


def _mesh(refined):
    """Return the cells of the mesh where the level 2 cell ``refined`` is split into level 3."""
    cells = [(1, 0, 0)]
    for j in range(4):
        for i in range(4):
            if i < 2 and j < 2:
                continue
            if (i, j) == refined:
                cells += [(3, 2 * i + a, 2 * j + b) for b in (0, 1) for a in (0, 1)]
            else:
                cells.append((2, i, j))
    return cells


def _box(cell, top):
    """Return the closed box ``(x0, y0, x1, y1)`` of ``cell`` in cells of level ``top``."""
    level, i, j = cell
    s = 2 ** (top - level)
    return i * s, j * s, (i + 1) * s, (j + 1) * s


def _jumps(cells):
    """Return the pairs of touching cells more than one level apart, with their contact.

    Each item is ``(coarse, fine, kind)``: ``kind`` is ``"face"`` when the two cells
    share part of a side, ``"corner"`` when they share only a vertex.
    """
    top = max(c[0] for c in cells)
    out = []
    for a in cells:
        for b in cells:
            if b[0] - a[0] < 2:
                continue
            ax0, ay0, ax1, ay1 = _box(a, top)
            bx0, by0, bx1, by1 = _box(b, top)
            # the length of the overlap along each axis; negative: they do not touch
            dx = min(ax1, bx1) - max(ax0, bx0)
            dy = min(ay1, by1) - max(ay0, by0)
            if dx < 0 or dy < 0:
                continue
            if dx > 0 and dy > 0:
                raise ValueError(f"the cells {a} and {b} overlap")
            out.append((a, b, "face" if dx + dy > 0 else "corner"))
    return out


# the three meshes: the level 2 cell split into level 3, and the jumps it must give
CASES = (
    ((3, 3), None),
    ((2, 0), "face"),
    ((2, 2), "corner"),
)


@figure
def graduation_allowed(p):
    """Draw a graded mesh, a face jump of two levels and a corner jump of two levels.

    Figure of tutorial/graduation.md. Each mesh is checked with ``_jumps``: the
    first has no jump, the second only face jumps, the third only corner jumps.
    The cells of each jump are hatched in red, and the side or the vertex they
    share is drawn in red.
    """
    size, top, pitch, x0 = 176, 48, 228, 24
    heads = (
        ("graded", "face or corner: one level apart at most"),
        ("face contact", "they share part of a side"),
        ("corner contact", "they share only a vertex"),
    )
    g = ""
    for k, ((refined, kind), (title, note)) in enumerate(zip(CASES, heads)):
        cells = _mesh(refined)
        jumps = _jumps(cells)
        kinds = {j[2] for j in jumps}
        if kinds != ({kind} if kind else set()):
            raise ValueError(f"the mesh {k + 1} has the jumps {kinds}, not {kind}")
        offending = {c for a, b, _ in jumps for c in (a, b)}
        ox = x0 + k * pitch
        frame = Frame((0, 0, 1, 1), ox, top, size)
        g += draw_cells(
            p,
            cells,
            frame,
            graded=not jumps,
            hatch=False,
            fill=lambda c, hit=offending: f"url(#{p}-r)" if c in hit else None,
        )
        for cell in cells:
            x, y = frame.center(cell)
            g += text(
                x,
                y + 4,
                cell[0],
                anchor="middle",
                size=10.5 if cell[0] == 3 else 12,
                fill=RED if cell in offending else INK2,
                knock=cell in offending,
            )
        # the side part or the vertex the cells of a jump share
        for a, b, _ in jumps:
            t = max(a[0], b[0])
            ax0, ay0, ax1, ay1 = _box(a, t)
            bx0, by0, bx1, by1 = _box(b, t)
            u0, u1 = max(ax0, bx0) / 2**t, min(ax1, bx1) / 2**t
            v0, v1 = max(ay0, by0) / 2**t, min(ay1, by1) / 2**t
            if kind == "face":
                g += line(frame.x(u0), frame.y(v0), frame.x(u1), frame.y(v1), stroke=RED, sw=HEAVY)
            else:
                g += circle(frame.x(u0), frame.y(v0), 3.5, fill=RED)
        cx = ox + size / 2
        g += text(cx, top - 26, title, anchor="middle", size=12, fill=INK)
        g += text(cx, top - 11, note, anchor="middle", size=11)
        verdict = "levels 1 and 3: not allowed" if jumps else "allowed"
        g += text(
            cx, top + size + 22, verdict, anchor="middle", size=11.5, fill=RED if jumps else INK
        )
    return Drawing(
        DIAGRAM_WIDTH,
        top + size + 34,
        "Three meshes of the unit square, each cell labeled with its level: a cell of level 1"
        " in the bottom left quarter, cells of level 2 elsewhere, and four cells of level 3 that"
        " replace one cell of level 2. Left, the level 3 cells are in the top right corner and"
        " touching cells are at most one level apart: allowed. Middle, the level 3 cells sit"
        " right of the level 1 cell and share part of its right side, a face contact two levels"
        " apart: not allowed. Right, the level 3 cells sit above and right of the level 1 cell"
        " and one of them shares only its top right vertex, a corner contact two levels apart:"
        " not allowed. The cells of each jump are hatched in red, the shared side or vertex is"
        " drawn in red.",
        patterns(p),
        g,
    )
