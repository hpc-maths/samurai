"""The figures of ``tutorial/graduation_case_1.md``: the random mesh and its graduation.

The meshes of the plate are the two files the demo saves with ``--with-corner``
and its default levels. Run
``tutorial-graduation-case-1 --with-corner --save-debug-fields --path run``, then
export ``run/graduation_case_1_before_graduation.h5`` with
``docs/tools/export_cells.py`` into ``data/graduation_case_1_before.json``, and
``run/graduation_case_1_after_graduation.h5`` into
``data/graduation_case_1_after.json``, each with ``--source`` naming the
program and the file. The diagrams redo in Python the tagging loop of the demo
(``_tag``), on the exported mesh or on a small 1D mesh; with the stencil of
``--with-corner`` it gives the same graded mesh as the demo, which
``graduation_meshes`` checks.
"""

from pathlib import Path

from ..cells import Frame, draw_cells, load_cells
from ..draw import (
    EMPTY,
    FIG,
    HAIR,
    HEAVY,
    INK,
    INK2,
    RED,
    RULE,
    RWASH,
    WASH,
    callout,
    fig_label,
    line,
    patterns,
    rect,
    text,
)
from ..registry import DIAGRAM_WIDTH, PLATE_WIDTH, Drawing, figure

DATA = Path(__file__).resolve().parent.parent / "data"

# the stencils of the demo: the axis directions by default, and with --with-corner
# the axis directions followed by the diagonal ones, in the order of the demo
AXIS = ((1, 0), (-1, 0), (0, 1), (0, -1))
DIAGONAL = ((1, 1), (-1, -1), (-1, 1), (1, -1))
WITH_CORNER = AXIS + DIAGONAL


def _meshes():
    """The cells of the mesh before and after the graduation, as the demo saves them."""
    before = load_cells(DATA / "graduation_case_1_before.json").leaves
    after = load_cells(DATA / "graduation_case_1_after.json").leaves
    return before, after


def _tag(cells, stencil):
    """The cells one pass of the demo tags, and the translated cells that tag them.

    For each level and each level ``level_below <= level - 2``, a cell of
    ``level`` translated by ``s`` that lands in a cell of ``level_below`` tags it:
    this is ``intersection(translate(ca[level], s), ca[level_below])
    .on(level_below)``. ``cells`` are ``(level, i, ...)``, in 1D or 2D; the result
    maps each tagged cell to the list of ``(cell, s)`` that reach it.
    """
    present = set(cells)
    levels = sorted({c[0] for c in cells})
    tagged = {}
    for level, *index in sorted(cells):
        for below in range(levels[0], level - 1):
            shift = level - below
            for s in stencil:
                # >> floors, as the set projection to a coarser level does
                coarse = (below, *((k + d) >> shift for k, d in zip(index, s)))
                if coarse in present:
                    tagged.setdefault(coarse, []).append(((level, *index), s))
    return tagged


def _split(cells, tagged):
    """The mesh where each tagged cell is replaced by its children."""
    out = []
    for level, *index in cells:
        if (level, *index) not in tagged:
            out.append((level, *index))
        elif len(index) == 1:
            out += [(level + 1, 2 * index[0] + a) for a in (0, 1)]
        else:
            i, j = index
            out += [(level + 1, 2 * i + a, 2 * j + b) for b in (0, 1) for a in (0, 1)]
    return sorted(out)


@figure
def graduation_meshes(p):
    """Draw the mesh of the demo before and after the graduation, side by side.

    Figure of tutorial/graduation_case_1.md, from the two files the demo saves
    with ``--with-corner``. The cells the graduation adds are hatched in red.
    Raise ``ValueError`` when the tagging loop of the demo, replayed with ``_tag``,
    does not turn the first mesh into the second.
    """
    before, after = _meshes()
    mesh = sorted(before)
    while tagged := _tag(mesh, WITH_CORNER):
        mesh = _split(mesh, tagged)
    if set(mesh) != set(after):
        raise ValueError("the tagging loop of the demo does not give the exported graded mesh")
    added = set(after) - set(before)
    size, oy, xa, xb = 330, 14, 16, 365
    fa, fb = Frame((0, 0, 1, 1), xa, oy, size), Frame((0, 0, 1, 1), xb, oy, size)
    # the mesh the demo generates is not graded: this is the figure's point
    g = draw_cells(p, before, fa, graded=False, hatch=False)
    g += draw_cells(
        p, after, fb, hatch=False, fill=lambda leaf: f"url(#{p}-r)" if leaf in added else None
    )
    # a: the level 2 cell [3, 4) x [1, 2) touches the level 7 cell (110, 64) by its top face
    g += callout(*fa.center((7, 110, 64)), *fa.center((4, 14, 6)), "a")
    # b: in place of that cell, cells of levels 3 to 6 step up to the level 7 cells
    g += callout(*fb.center((4, 13, 6)), *fb.center((5, 30, 10)), "b", red=True)
    g += fig_label(xa + size / 2, oy + size + 32, 1, "before")
    g += fig_label(xb + size / 2, oy + size + 32, 2, "after")
    return Drawing(
        PLATE_WIDTH,
        oy + size + 46,
        f"Two square meshes of the unit square. Left, the mesh the demo generates, {len(before)}"
        " cells from level 2 to level 7, where small cells touch cells many times larger: a"
        " level 2 cell at the right touches level 7 cells along its top face. Right, the same"
        f" mesh after the graduation, {len(after)} cells; the {len(added)} cells the graduation"
        " added are hatched in red and step between small and large cells one level at a time.",
        patterns(p),
        g,
        panels=((0, 0, 362, oy + size + 46, ""), (349, 0, 362, oy + size + 46, "")),
    )


def _dashed_cell(x, y, c, inside):
    """A translated cell: red where it lands in a coarse cell, a dashed outline elsewhere."""
    if inside:
        return rect(x, y, c, c, fill=RWASH, stroke=RED)
    return rect(x, y, c, c, stroke=INK2, sw=EMPTY, dash="3 2")


@figure
def graduation_translation(p):
    """Draw one translation of the demo on a window of its mesh, then its set projection.

    Figure of tutorial/graduation_case_1.md: the window [0, 1/4) x [1/2, 3/4) of
    the mesh before the graduation, the cells of level 5 translated by s = {1, 1},
    their intersection with the cells of level 3 and its set projection on level 3.
    """
    before, _ = _meshes()
    c, n, y0 = 20, 8, 16  # the window holds the level-5 cells [0, 8) x [16, 24)
    top = 30
    xs = (64, 290, 516)

    def fine(level, i, j):
        """Whether a cell lies in the window, in level-5 units."""
        s = 2 ** (5 - level)
        return 0 <= i * s < n and y0 <= j * s < y0 + n

    window = [leaf for leaf in before if fine(*leaf)]
    fives = [leaf for leaf in window if leaf[0] == 5]
    threes = [leaf for leaf in window if leaf[0] == 3]
    tagged = _tag(before, ((1, 1),))
    hits = sorted(
        (cell[1] + 1, cell[2] + 1)
        for coarse, cells in tagged.items()
        if coarse in threes
        for cell, _ in cells
        if cell[0] == 5
    )

    def box(ox, level, i, j):
        """The drawing rectangle of a cell in the panel at ``ox``."""
        s = 2 ** (5 - level) * c
        return ox + i * s, top + (y0 + n) * c - (j + 1) * s, s, s

    def frame(ox):
        """The cells of the window, in faint outline."""
        g = ""
        for leaf in window:
            g += rect(*box(ox, *leaf), stroke=FIG, sw=EMPTY)
        return g

    def axes(ox, level, edges):
        """Ticks of the cell edges of ``level`` under and left of the panel."""
        g = ""
        s = 2 ** (5 - level)
        for k in range(0, n // s + 1):
            x = ox + k * s * c
            g += line(x, top + n * c, x, top + n * c + 4, stroke=INK2, sw=HAIR)
            g += text(x, top + n * c + 15, k, anchor="middle", size=10)
        for k in range(edges[0], edges[1] + 1):
            y = top + n * c - (k * s - y0) * c
            g += line(ox - 4, y, ox, y, stroke=INK2, sw=HAIR)
            g += text(ox - 6, y + 3.5, k, anchor="end", size=10)
        return g

    # 1, the operands: the cells of level 5, filled, and of level 3, in heavy rule
    ox = xs[0]
    g = frame(ox)
    for leaf in fives:
        g += rect(*box(ox, *leaf), fill=WASH, stroke=INK, sw=HAIR)
    for leaf in threes:
        g += rect(*box(ox, *leaf), stroke=INK, sw=HEAVY)
    g += axes(ox, 5, (y0, y0 + n))
    # 2, the level 5 cells translated by s, in red where they land in a level 3 cell
    ox = xs[1]
    g += frame(ox)
    for leaf in threes:
        g += rect(*box(ox, *leaf), stroke=INK, sw=HEAVY)
    for _, i, j in fives:
        if fine(5, i + 1, j + 1):
            g += _dashed_cell(*box(ox, 5, i + 1, j + 1)[:2], c, (i + 1, j + 1) in hits)
    x1, y1, _, _ = box(ox, 5, 3, 19)
    g += rect(x1, y1, c, c, fill=WASH, stroke=INK, sw=HAIR)
    g += line(
        x1 + c / 2,
        y1 + c / 2,
        x1 + 1.42 * c,
        y1 - 0.42 * c,
        stroke=RED,
        extra=f'marker-end="url(#{p}-ar)"',
    )
    g += axes(ox, 5, (y0, y0 + n))
    # 3, the set projection on level 3: the coarse cells the hits fall in
    ox = xs[2]
    g += frame(ox)
    on3 = sorted({(3, i >> 2, j >> 2) for i, j in hits})
    for leaf in threes:
        tag = leaf in on3
        g += rect(*box(ox, *leaf), fill=f"url(#{p}-r)" if tag else "none", stroke=INK, sw=HEAVY)
    g += axes(ox, 3, (y0 // 4, (y0 + n) // 4))
    rows = sorted({j for _, j in hits})
    columns = sorted({i for i, _ in hits})
    yl = top + n * c + 38
    for k, (code, word) in enumerate((("ca[5]", "filled"), ("ca[3]", "heavy outline"))):
        g += text(xs[0], yl + k * 17, code, size=11, fill=INK, cls="sm-fig-code")
        g += text(xs[0] + 44, yl + k * 17, word, size=11)
    g += text(xs[1], yl, "translate(ca[5], {1, 1})", size=11, fill=INK, cls="sm-fig-code")
    g += text(
        xs[1],
        yl + 17,
        f"∩ ca[3]: [{columns[0]}, {columns[-1] + 1})",
        size=11,
        fill=RED,
        cls="sm-fig-code",
    )
    g += text(xs[1], yl + 34, f"on the rows {rows[0]} to {rows[-1]}", size=11)
    coarse_rows = sorted({j for _, _, j in on3})
    coarse_cols = sorted({i for _, i, _ in on3})
    g += text(
        xs[2],
        yl,
        f".on(3): [{coarse_cols[0]}, {coarse_cols[-1] + 1})",
        size=11,
        fill=RED,
        cls="sm-fig-code",
    )
    g += text(xs[2], yl + 17, f"on the rows {coarse_rows[0]} and {coarse_rows[-1]}", size=11)
    for k, title in enumerate(("level 5 and level 3", "translated, on level 5", "on level 3")):
        g += text(xs[k] + n * c / 2, top - 12, title, anchor="middle", size=11.5, fill=INK)
    return Drawing(
        DIAGRAM_WIDTH,
        yl + 44,
        "Three views of the window [0, 1/4) x [1/2, 3/4) of the mesh before the graduation."
        " Left, the cells of level 5 are filled and the two cells of level 3 at the right of"
        " the window have a heavy outline. Middle, the level 5 cells are moved one cell right and"
        f" one cell up; the {len(hits)} moved cells that land in a level 3 cell are red: on level"
        f" 5 they are [{columns[0]}, {columns[-1] + 1}) on the rows {rows[0]} to {rows[-1]}. An"
        " arrow shows the move of one cell. Right, brought to level 3 they become"
        f" [{coarse_cols[0]}, {coarse_cols[-1] + 1}) on the rows {coarse_rows[0]} and"
        f" {coarse_rows[-1]}: both level 3 cells are hatched in red, tagged for refinement.",
        patterns(p),
        g,
    )


def _vector(s):
    """The stencil row ``s`` as the demo writes it, ``{1, -1}``."""
    return "{" + ", ".join(str(k) for k in s) + "}"


@figure
def graduation_directions(p):
    """Draw a face contact and a corner contact against the two stencils of the demo.

    Figure of tutorial/graduation_case_1.md. A cell of level l, filled, touches a
    cell of level l - 2, four times larger, in heavy rule: by a part of a side
    (face contact) or by a vertex only (corner contact). Each stencil of the demo
    moves the small cell by each of its vectors; the coarse cell is tagged when a
    moved copy lands in it, as ``intersection(translate(ca[level], s),
    ca[level_below])`` finds it.
    """
    c, top, pitch = 24, 44, 218
    # the panels show the cells [0, 6) x [-1, 5) of level l; the coarse cell
    # covers [2, 6) x [1, 5)
    coarse = (2, 1, 6, 5)
    contacts = (
        ("face contact", "they share part of a side", (1, 1)),
        ("corner contact", "they share only a vertex", (1, 0)),
    )
    stencils = (
        ("axis stencil", "the default", AXIS),
        (
            "axis and diagonal stencil",
            'with <tspan class="sm-fig-code">--with-corner</tspan>',
            WITH_CORNER,
        ),
    )
    xs, key = (282, 482), 140
    ar, ar_red = f'marker-end="url(#{p}-a)"', f'marker-end="url(#{p}-ar)"'

    def corner(ox, oy, i, j):
        """The top left corner of the cell (i, j) of level l in a panel."""
        return ox + i * c, oy + (4 - j) * c

    def center(ox, oy, i, j):
        """The center of the cell (i, j) of level l in a panel."""
        x, y = corner(ox, oy, i, j)
        return x + c / 2, y + c / 2

    g = ""
    for col, (name, note, _) in enumerate(contacts):
        g += text(xs[col] + 3 * c, top - 26, name, anchor="middle", size=12, fill=INK)
        g += text(xs[col] + 3 * c, top - 11, note, anchor="middle", size=11)
    for row, (name, note, stencil) in enumerate(stencils):
        oy = top + row * pitch
        # the stencil: a cell and its vectors, each labeled
        g += text(key, oy + 8, name, anchor="middle", size=12, fill=INK)
        cx, cy, half = key, oy + 80, 10
        g += rect(cx - half, cy - half, 2 * half, 2 * half, fill=WASH, stroke=INK)
        for sx, sy in stencil:
            # y up in the mesh, down in the drawing
            dx, dy = sx, -sy
            g += line(
                cx + dx * (half + 2),
                cy + dy * (half + 2),
                cx + dx * 34,
                cy + dy * 34,
                stroke=INK2,
                extra=ar,
            )
            lx, ly = cx + dx * 42, cy + dy * 42 + 4
            anchor = "middle" if dx == 0 else ("start" if dx > 0 else "end")
            g += text(lx, ly, _vector((sx, sy)), anchor=anchor, size=10.5, cls="sm-fig-code")
        g += text(key, oy + 152, note, anchor="middle", size=11)
        for col, (_, _, (fi, fj)) in enumerate(contacts):
            ox = xs[col]
            for i in range(6):
                for j in range(-1, 5):
                    g += rect(*corner(ox, oy, i, j), c, c, stroke=FIG, sw=EMPTY)
            hits = [
                (sx, sy)
                for sx, sy in stencil
                if coarse[0] <= fi + sx < coarse[2] and coarse[1] <= fj + sy < coarse[3]
            ]
            x0, y0 = corner(ox, oy, coarse[0], coarse[3] - 1)
            fill = f"url(#{p}-r)" if hits else "none"
            g += rect(x0, y0, 4 * c, 4 * c, fill=fill, stroke=INK, sw=HEAVY)
            for sx, sy in stencil:
                g += _dashed_cell(*corner(ox, oy, fi + sx, fj + sy), c, (sx, sy) in hits)
            g += rect(*corner(ox, oy, fi, fj), c, c, fill=WASH, stroke=INK)
            fx, fy = center(ox, oy, fi, fj)
            for sx, sy in stencil:
                hit = (sx, sy) in hits
                tx, ty = center(ox, oy, fi + sx, fj + sy)
                # from the center of the cell to the center of its copy
                g += line(
                    fx,
                    fy,
                    tx - sx * c * 0.1,
                    ty + sy * c * 0.1,
                    stroke=RED if hit else INK2,
                    sw=RULE if hit else HAIR,
                    extra=ar_red if hit else ar,
                )
            if hits:
                vectors = ", ".join(_vector(s) for s in hits)
                verbs = "lands" if len(hits) == 1 else "land"
                verdict = (f'<tspan class="sm-fig-code">{vectors}</tspan> {verbs} in it', "tagged")
            else:
                verdict = ("no copy lands in it", "not tagged")
            for k, words in enumerate(verdict):
                g += text(
                    ox + 3 * c,
                    oy + 6 * c + 20 + 15 * k,
                    words,
                    anchor="middle",
                    size=11,
                    fill=RED if hits else INK2,
                )
    return Drawing(
        DIAGRAM_WIDTH,
        top + 2 * pitch - 26,
        "A table of four small grids. Columns: a filled cell of level l touches a coarse cell of"
        " level l - 2, four times larger, by part of a side (face contact) or by a vertex only"
        " (corner contact). Rows: the axis stencil {1, 0}, {-1, 0}, {0, 1}, {0, -1}, the"
        " default, and with --with-corner the same four vectors followed by the diagonal ones"
        " {1, 1}, {-1, -1}, {-1, 1}, {1, -1}, each stencil drawn as labeled arrows from a cell."
        " In each grid the small cell is moved by every vector; a copy that lands in the coarse"
        " cell is red and the coarse cell is hatched, tagged. Axis, face contact: {1, 0} lands"
        " in it. Axis, corner contact: no copy lands in it, the coarse cell is not tagged. Axis"
        " and diagonal, face contact: {1, 0} and {1, 1} land in it. Axis and diagonal, corner"
        " contact: {1, 1} lands in it.",
        patterns(p),
        g,
    )


@figure
def graduation_passes(p):
    """Draw the passes of the tagging loop on a 1D mesh, until it is graded.

    Figure of tutorial/graduation_case_1.md: the cell [0, 1/2) of level 1 next to
    cells of level 5. Each pass is computed with ``_tag``, the 1D form of the
    demo's loop, with the stencil {1}, {-1}.
    """
    mesh = [(1, 0)] + [(5, i) for i in range(16, 32)]
    rows = [("initial mesh", mesh, set())]
    while True:
        tagged = _tag(mesh, ((1,), (-1,)))
        if not tagged:
            break
        new = _split(mesh, tagged)
        rows.append((f"pass {len(rows)}", new, set(new) - set(mesh)))
        mesh = new
    x0, width, ch, pitch, top = 112, 544, 26, 50, 10
    g = ""
    for r, (name, cells, added) in enumerate(rows):
        y = top + r * pitch
        for level, i in cells:
            w = width / 2**level
            x = x0 + i * w
            red = (level, i) in added
            g += rect(
                x,
                y,
                w,
                ch,
                fill=RWASH if red else "none",
                stroke=RED if red else INK,
                sw=RULE,
            )
            g += text(
                x + w / 2,
                y + ch / 2 + 4,
                level,
                anchor="middle",
                size=10.5,
                fill=RED if red else INK2,
            )
        g += text(x0 - 14, y + ch / 2 + 4, name, anchor="end", size=11.5, fill=INK)
    passes = len(rows)
    y = top + passes * pitch - 6
    g += text(
        x0 - 14,
        y + 4,
        f"pass {passes}",
        anchor="end",
        size=11.5,
        fill=INK,
    )
    g += text(x0, y + 4, "no cell tagged: the mesh is graded and the loop stops", size=11.5)
    axis = y + 22
    g += line(x0, axis, x0 + width, axis, stroke=INK2)
    for v, label in ((0, "0"), (0.25, "1/4"), (0.5, "1/2"), (0.75, "3/4"), (1, "1")):
        g += line(x0 + v * width, axis, x0 + v * width, axis + 5, stroke=INK2)
        g += text(x0 + v * width, axis + 18, label, anchor="middle", size=11)
    g += text(x0 - 14, axis + 18, "x", anchor="end", size=13, fill=INK, cls="sm-fig-math")
    return Drawing(
        DIAGRAM_WIDTH,
        axis + 26,
        "Four rows of a 1D mesh of [0, 1), each cell labeled with its level. The initial mesh has"
        " one cell of level 1 on [0, 1/2) and sixteen cells of level 5 on [1/2, 1). Pass 1 splits"
        " the level 1 cell into two level 2 cells, pass 2 splits the right one into two level 3"
        " cells, pass 3 splits the right level 3 cell into two level 4 cells; the new cells are"
        " red. Pass 4 tags no cell: the mesh is graded.",
        patterns(p),
        g,
    )
