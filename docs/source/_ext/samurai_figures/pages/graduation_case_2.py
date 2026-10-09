"""The figures of ``tutorial/graduation_case_2.md``: overlaps removed, then the graduation.

The three meshes are the ones ``tutorial-graduation-case-2 --with-corner`` saves,
with its default levels, exported with ``docs/tools/export_cells.py`` into ``data/``.
"""

from pathlib import Path

from ..cells import Frame, draw_cells, load_cells
from ..draw import (
    EMPTY,
    FIG,
    INK,
    INK2,
    RED,
    RWASH,
    WASH,
    bracket,
    callout,
    fig_label,
    line,
    patterns,
    rect,
    text,
)
from ..registry import DIAGRAM_WIDTH, PLATE_WIDTH, Drawing, figure

DATA = Path(__file__).resolve().parent.parent / "data"


def _mesh(name):
    """Return the cells of the mesh the demo saves as ``graduation_case_2_<name>``."""
    return load_cells(DATA / f"graduation_case_2_{name}.json").leaves


@figure
def overlap_removal(p):
    """Draw the initial mesh, the mesh without overlaps and the graded mesh of the demo.

    Figure of tutorial/graduation_case_2.md. The cells that the removal of the
    overlaps adds are filled with the ink wash, the cells that the graduation adds
    are hatched in red; both are set differences of the exported meshes.
    """
    initial, without, graded = _mesh("initial"), _mesh("without_intersection"), _mesh("graduated")
    added, refined = set(without) - set(initial), set(graded) - set(without)
    size, oy, gap = 215, 34, 33
    xs = [(PLATE_WIDTH - 3 * size - 2 * gap) / 2 + k * (size + gap) for k in range(3)]
    frames = [Frame((0, 0, 1, 1), x, oy, size) for x in xs]

    g = draw_cells(p, initial, frames[0], graded=False, overlap=True, hatch=False)
    g += draw_cells(
        p,
        without,
        frames[1],
        graded=False,
        hatch=False,
        fill=lambda leaf: WASH if leaf in added else None,
    )
    g += draw_cells(
        p,
        graded,
        frames[2],
        hatch=False,
        fill=lambda leaf: f"url(#{p}-r)" if leaf in refined else None,
    )
    for k, name in enumerate(("initial", "without overlap", "graded")):
        g += fig_label(xs[k] + size / 2, oy + size + 34, k + 1, name)
        g += text(
            xs[k] + size / 2,
            oy + size + 52,
            f"{len((initial, without, graded)[k])} cells",
            anchor="middle",
            size=11.5,
        )

    # the letters sit in the margin above each mesh, their leaders run down to the cell
    for k, leaf, letter in (
        (0, (5, 13, 29), "a"),  # a fine cell drawn over the coarse cells it lies in
        (1, (3, 1, 7), "b"),  # a cell the overlap removal added
        (2, (4, 5, 15), "c"),  # a cell the graduation added
    ):
        x, y = frames[k].center(leaf)
        g += callout(x, y, x + 12, oy - 14, letter, red=letter == "c")

    height = oy + size + 62
    return Drawing(
        PLATE_WIDTH,
        height,
        f"Three square meshes of the unit square, levels 0 to 6. Left, the initial mesh of"
        f" {len(initial)} cells: the cell of level 0 covers the whole square and smaller cells of"
        f" levels 1 to 6 lie on top of it and of each other, the covered cells outlined with the"
        f" heavy rule. Middle, the mesh without overlap, {len(without)} cells of levels 3 to 6,"
        f" not graded: the {len(added)} cells added by the removal of the overlaps are filled"
        f" with a gray wash. Right, the graded mesh of {len(graded)} cells: the {len(refined)}"
        " cells added by the graduation are hatched in red and form a layer of intermediate"
        " sizes around the finest cells.",
        patterns(p),
        g,
        panels=tuple((x - gap / 2, 0, size + gap, height, "") for x in xs),
    )


def _loop(levels, tagger):
    """Run one loop of the demo on a 1D mesh until the mesh no longer changes.

    ``levels`` maps a level to its set of cell indices; ``tagger(mesh)`` returns the
    tagged cells of each level. Each pass replaces every tagged cell by its two
    children, as the ``CellList`` of the demo does. Return the list of
    ``(mesh, tags)``, one per iteration the demo prints; the last one has no tag.
    """
    passes = []
    mesh = {level: set(cells) for level, cells in levels.items() if cells}
    while True:
        tags = {level: cells for level, cells in tagger(mesh).items() if cells}
        passes.append((mesh, tags))
        new = {}
        for level, cells in mesh.items():
            kept = cells - tags.get(level, set())
            children = {c for i in tags.get(level, ()) for c in (2 * i, 2 * i + 1)}
            for lev, part in ((level, kept), (level + 1, children)):
                if part:
                    new.setdefault(lev, set()).update(part)
        if new == mesh:
            return passes
        mesh = new


def _overlap_tags(mesh):
    """``intersection(ca[level], ca[level_below]).on(level_below)`` for each pair of levels."""
    tags = {}
    for level in mesh:
        for below in mesh:
            if below < level:
                on = {i >> (level - below) for i in mesh[level]}
                tags.setdefault(below, set()).update(on & mesh[below])
    return tags


def _graduation_tags(mesh):
    """``intersection(translate(ca[level], s), ca[level_below]).on(level_below)``.

    For each pair of levels ``level_below < level - 1`` and each ``s`` of the 1D
    stencil {-1, 1}, to which both stencils of the demo reduce in 1D.
    """
    tags = {}
    for level in mesh:
        for below in mesh:
            if below < level - 1:
                for s in (-1, 1):
                    on = {(i + s) >> (level - below) for i in mesh[level]}
                    tags.setdefault(below, set()).update(on & mesh[below])
    return tags


def remove_overlaps(levels):
    """Run the overlap removal of the demo on a 1D mesh, see ``_loop``."""
    return _loop(levels, _overlap_tags)


def graduate(levels):
    """Run the graduation of the demo on a 1D mesh, see ``_loop``."""
    return _loop(levels, _graduation_tags)


def _finest(mesh):
    """Return the level of the cell over each position of the finest level of ``mesh``.

    Raise ``ValueError`` when two cells cover the same position or a position is
    left uncovered in the cell 0 of level 0.
    """
    top = max(mesh)
    seen = {}
    for level, cells in mesh.items():
        for i in cells:
            for x in range(i << (top - level), (i + 1) << (top - level)):
                if x in seen:
                    raise ValueError(f"the 1D mesh {mesh} has overlapping cells")
                seen[x] = level
    if sorted(seen) != list(range(2**top)):
        raise ValueError(f"the 1D mesh {mesh} does not cover the cell of level 0")
    return [seen[x] for x in range(2**top)]


def check_graded_1d(mesh):
    """Raise ``ValueError`` unless ``mesh`` is disjoint and its neighbors differ by one level."""
    finest = _finest(mesh)
    for a, b in zip(finest, finest[1:]):
        if abs(a - b) > 1:
            raise ValueError(f"the 1D mesh {mesh} is not graded: levels {a} and {b} touch")


def _runs(cells):
    """Return the half-open intervals (a, b) of a set of indices."""
    out = []
    for i in sorted(cells):
        if out and out[-1][1] == i:
            out[-1][1] = i + 1
        else:
            out.append([i, i + 1])
    return [tuple(r) for r in out]


@figure
def overlap_passes(p):
    """Draw the overlap removal, then the graduation, pass after pass on a 1D mesh.

    Figure of tutorial/graduation_case_2.md, at "Remove the overlaps". The mesh has
    levels 0 to 3; the passes come from ``remove_overlaps`` and ``graduate``, the two
    loops of the demo written in Python. The mesh the overlap removal leaves has no
    overlap but is not graded; the graduation grades it, which is checked.
    """
    top_level = 3
    first = remove_overlaps({0: {0}, top_level: {5}})
    second = graduate(first[-1][0])
    _finest(first[-1][0])
    final = second[-1][0]
    check_graded_1d(final)
    added = {level: cells - first[-1][0].get(level, set()) for level, cells in final.items()}

    x0, unit, col, pitch, ch = 70, 128, 152, 52, 22
    # the title, the headings, the rows and the two lines of notes of a step
    block = 46 + (top_level + 1) * pitch + 24
    g = ""

    def step(oy, title, passes, last):
        """Draw one loop at ``oy``; ``last`` writes the note under its last iteration."""
        s = text(x0 - 58, oy + 10, title, size=12, fill=INK, cls="sm-fig-label")
        top = oy + 46
        for level in range(top_level + 1):
            y = top + level * pitch
            s += text(x0 - 12, y + 15, f"level {level}", anchor="end", size=11.5, fill=INK)
        for k, (mesh, tags) in enumerate(passes):
            ox = x0 + k * col
            s += text(
                ox + unit / 2, top - 14, f"iteration {k}", anchor="middle", size=11.5, fill=INK
            )
            for level in range(top_level + 1):
                y, cw = top + level * pitch, unit / 2**level
                cells, tagged = mesh.get(level, set()), tags.get(level, set())
                new = added.get(level, set()) if mesh is final else set()
                for i in range(2**level):
                    if i in tagged:
                        s += rect(ox + i * cw, y, cw, ch, fill=RWASH, stroke=RED)
                    elif i in new:
                        s += rect(ox + i * cw, y, cw, ch, fill=f"url(#{p}-r)", stroke=INK)
                    elif i in cells:
                        s += rect(ox + i * cw, y, cw, ch, fill=WASH, stroke=INK)
                    else:
                        s += rect(ox + i * cw, y, cw, ch, stroke=FIG, sw=EMPTY)
                for a, b in _runs(tagged):
                    s += bracket(
                        ox + a * cw, ox + b * cw, y + ch + 9, f"[{a}, {b})", color=RED, size=11
                    )
            if tags:
                # from this iteration to the next, between their headings
                s += line(
                    ox + unit / 2 + 40,
                    top - 18,
                    ox + col + unit / 2 - 40,
                    top - 18,
                    stroke=INK2,
                    extra=f'marker-end="url(#{p}-a)"',
                )
            else:
                s += last(ox, top)
        return s

    def not_graded(ox, top):
        """Mark the jump of two levels at 1/2, in red, under the last overlap iteration."""
        xm = ox + unit / 2
        s = line(xm, top + pitch - 6, xm, top + 3 * pitch + ch + 6, stroke=RED, sw=0.8, dash="2 3")
        y = top + (top_level + 1) * pitch - 8
        s += text(ox + unit / 2, y, "no tag: the loop stops", anchor="middle", size=11, fill=INK2)
        s += text(ox + unit / 2, y + 15, "not graded yet", anchor="middle", size=11, fill=RED)
        return s

    def graded(ox, top):
        """Write the note under the last graduation iteration."""
        y = top + (top_level + 1) * pitch - 8
        s = text(ox + unit / 2, y, "no tag: the loop stops", anchor="middle", size=11, fill=INK2)
        s += text(ox + unit / 2, y + 15, "graded", anchor="middle", size=11, fill=INK)
        return s

    g += step(0, "Remove the overlaps", first, not_graded)
    oy2 = block + 26
    g += line(24, oy2 - 16, DIAGRAM_WIDTH - 8, oy2 - 16, stroke=FIG)
    g += step(oy2, "Make the mesh graded", second, graded)
    height = oy2 + block - 4
    return Drawing(
        DIAGRAM_WIDTH,
        height,
        "The two loops of the demo on a 1D mesh of levels 0 to 3, each iteration drawn as four"
        " rows of cells over the cell of level 0, tagged cells in red. Remove the overlaps,"
        " iteration 0: the mesh holds the cell 0 of level 0 and the cell 5 of level 3, which"
        " lies inside it; the cell of level 0 is tagged, [0, 1). Iteration 1: it is replaced by"
        " the cells 0 and 1 of level 1, and the cell 1 of level 1 is tagged, [1, 2). Iteration"
        " 2: the cells 2 and 3 of level 2 replace it, and the cell 2 of level 2 is tagged, [2,"
        " 3). Iteration 3: the mesh holds the cell 0 of level 1, the cell 3 of level 2 and the"
        " cells 4 and 5 of level 3; nothing overlaps and the loop stops, but the cell of level 1"
        " touches the cell 4 of level 3 at 1/2, marked by a red dotted line: the mesh is not"
        " graded yet. Make the mesh graded, iteration 0: the cells of level 3 shifted by one"
        " cell reach the cell 0 of level 1, which is tagged, [0, 1). Iteration 1: it is replaced"
        " by the cells 0 and 1 of level 2, hatched in red; no cell is tagged, the loop stops and"
        " the mesh is graded.",
        patterns(p),
        g,
    )
