r"""The figures of ``tutorial/graduation_case_3.md``: a refinement along a curve, graded.

The meshes of the plate are the files the demo saves with ``--maximum-level 4``,
the other options left to their defaults. Run::

    ./tutorial-graduation-case-3 --maximum-level 4 --with-graduation=false \
        --save-debug-fields --path run --filename without_graduation
    ./tutorial-graduation-case-3 --maximum-level 4 \
        --save-debug-fields --path run --filename with_graduation

then export ``run/<name>.h5`` with ``docs/tools/export_cells.py`` into
``data/graduation_case_3_<name>.json``, with ``--source`` naming the program and
its options. ``run_demo`` redoes in Python the loop of the demo: the figures
check that it ends on the exported meshes, and the diagram takes one interval of
its graduation step.
"""

import math
from pathlib import Path

from ..arrays import Array, draw_array, even_elements, odd_elements, strided_row
from ..cells import Frame, check_graded, draw_cells, load_cells
from ..draw import (
    HAIR,
    INK,
    INK2,
    RED,
    RULE,
    bracket,
    callout,
    cell_row,
    fig_label,
    line,
    path,
    patterns,
    rect,
    text,
)
from ..registry import DIAGRAM_WIDTH, PLATE_WIDTH, Drawing, figure

DATA = Path(__file__).resolve().parent.parent / "data"

# the options of the demo for the figures, and the box of the domain
START_LEVEL, MAX_LEVEL = 1, 4
DOMAIN = (-2.0, -2.0, 2.0, 2.0)
# the stencil of the graduation step of the demo, in its order
STENCIL = ((1, 1), (-1, -1), (-1, 1), (1, -1))


def curve(t):
    """Return the point of the curve of the demo at the parameter ``t``."""
    return math.sin(3 * t + math.pi * 0.5), math.sin(2 * t)


def samples(max_level):
    """Return the points of the curve the demo tests, ``1 << (max_level + 4)`` of them.

    ``t`` is summed step by step, as in the demo, so that the points are the same
    doubles.
    """
    npoints = 1 << (max_level + 4)
    dt = 2.0 * math.pi / npoints
    t, out = 0.0, []
    for _ in range(npoints):
        out.append(curve(t))
        t += dt
    return out


def _criterion(mesh, points):
    """Return the cells of ``mesh`` with a point of ``points`` strictly inside.

    This is the test of the demo: ``corner < xc < corner + dx`` along x and y.
    """
    tags = set()
    for level in {c[0] for c in mesh}:
        dx = 2.0**-level
        for x, y in points:
            i, j = math.floor(x / dx), math.floor(y / dx)
            if i * dx < x and y > j * dx and (level, i, j) in mesh:
                tags.add((level, i, j))
    return tags


def _subset(mesh, level, s):
    """Return the rows of ``intersection(translate(ca[level], s), ca[level - 1])``.

    The set expression is computed on ``level``: the result maps a row ``j_f`` to its
    intervals ``(start, end)`` of fine cells, sorted.
    """
    fine = {(i + s[0], j + s[1]) for lev, i, j in mesh if lev == level}
    coarse = {(i, j) for lev, i, j in mesh if lev == level - 1}
    cells = sorted((j, i) for i, j in fine if (i >> 1, j >> 1) in coarse)
    rows = {}
    for j, i in cells:
        runs = rows.setdefault(j, [])
        if runs and runs[-1][1] == i:
            runs[-1][1] = i + 1
        else:
            runs.append([i, i + 1])
    return {j: [tuple(r) for r in runs] for j, runs in rows.items()}


def _graduation(mesh, tags, record=None):
    """Add to ``tags`` the coarse cells the graduation step of the demo tags.

    From the finest level down, for each row ``s`` of the stencil, the fine
    positions of the set expression whose cell moved by ``-s`` is tagged tag the
    cell ``>> 1`` of ``level - 1``; the even and the odd elements are handled
    apart, as in the demo. ``record(level, s, j_f, interval, tags)`` is called
    on each interval before it changes ``tags``.
    """
    top = max(c[0] for c in mesh)
    for level in range(top, 1, -1):
        for s in STENCIL:
            for j_f, intervals in _subset(mesh, level, s).items():
                for interval in intervals:
                    if record:
                        record(level, s, j_f, interval, set(tags))
                    for part in (even_elements(*interval), odd_elements(*interval)):
                        if part is None:
                            continue
                        for i_f in range(*part):
                            if (level, i_f - s[0], j_f - s[1]) in tags:
                                tags.add((level - 1, i_f >> 1, j_f >> 1))


def run_demo(start_level, max_level, with_graduation, record=None):
    """Run the loop of the demo and return the cells of its final mesh, sorted.

    Each pass tags the cells the curve goes through, adds the tags of the
    graduation step when ``with_graduation`` is true, and replaces every tagged
    cell below ``max_level`` by its four children, until the mesh no longer
    changes. ``record(pass, level, s, j_f, interval, tags)`` is called on each
    interval of the graduation step.
    """
    n = 2 << start_level
    mesh = {(start_level, i, j) for i in range(-n, n) for j in range(-n, n)}
    points = samples(max_level)
    passes = 0
    while True:
        tags = _criterion(mesh, points)
        if with_graduation:
            k = passes
            _graduation(mesh, tags, record and (lambda *a, k=k: record(k, *a)))
        new = set()
        for level, i, j in mesh:
            if (level, i, j) in tags and level < max_level:
                new |= {(level + 1, 2 * i + a, 2 * j + b) for a in (0, 1) for b in (0, 1)}
            else:
                new.add((level, i, j))
        passes += 1
        if new == mesh:
            return sorted(mesh)
        mesh = new


def _mesh(name, with_graduation):
    """Return the cells the demo saves as ``name``, checked against ``run_demo``."""
    cells = load_cells(DATA / f"graduation_case_3_{name}.json").leaves
    if sorted(cells) != run_demo(START_LEVEL, MAX_LEVEL, with_graduation):
        raise ValueError(f"the Python loop of the demo does not give the mesh {name}")
    return cells


def _finest_owner(leaves):
    """Map each cell of the finest level to the level of the cell over it."""
    top = max(c[0] for c in leaves)
    owner = {}
    for level, i, j in leaves:
        s = 2 ** (top - level)
        for a in range(s):
            for b in range(s):
                owner[i * s + a, j * s + b] = level
    return top, owner


def check_curve(leaves, margin):
    """Return the cells of ``leaves`` the curve enters deeper than ``margin``.

    The curve is followed with a step much finer than a cell; ``margin`` is in
    units of the domain. Only the cells below the finest level are returned: on
    a refined mesh, the drawn curve must stay in the cells of the finest level.
    """
    top, owner = _finest_owner(leaves)
    h = 2.0**-top
    out = set()
    n = 20000
    for k in range(n):
        x, y = curve(2 * math.pi * k / n)
        level = owner[math.floor(x / h), math.floor(y / h)]
        if level == top:
            continue
        side = 2.0**-level
        i, j = math.floor(x / side), math.floor(y / side)
        depth = min(x - i * side, (i + 1) * side - x, y - j * side, (j + 1) * side - y)
        if depth > margin:
            out.add((level, i, j))
    return out


def _curve_path(frame):
    """Return the ``d`` of the curve drawn in ``frame``."""
    n = 720
    d = ""
    for k in range(n + 1):
        x, y = curve(2 * math.pi * k / n)
        d += f"{'M' if k == 0 else 'L'}{frame.x(x):.2f} {frame.y(y):.2f}"
    return d


def _counts(leaves):
    """Return the number of cells per level, as ``{level: count}``."""
    out = {}
    for level, *_ in leaves:
        out[level] = out.get(level, 0) + 1
    return dict(sorted(out.items()))


@figure
def refined_meshes(p):
    """Draw the mesh the demo refines along the curve, without and with graduation.

    Figure of tutorial/graduation_case_3.md. The cells of the finest level are
    hatched in red and the curve is drawn over both meshes. The mesh without
    graduation is not graded, which is the point of the figure; the graded mesh
    is checked, and the curve must stay in its cells of the finest level.
    """
    without = _mesh("without_graduation", False)
    with_ = _mesh("with_graduation", True)
    check_graded(with_)
    # the labels of the y ticks sit in the left margin, left of the first mesh only
    size, oy, gap, left = 324, 14, 30, 22
    xs = [left + k * (size + gap) for k in range(2)]
    frames = [Frame(DOMAIN, x, oy, size) for x in xs]
    # half the stroke of the curve, in units of the domain
    stroke = 1.2
    margin = stroke / 2 / frames[0].unit
    if check_curve(with_, margin):
        raise ValueError("the curve leaves the cells of the finest level of the graded mesh")
    # the demo tests a finite number of points of the curve: between two of them,
    # the curve can cut the corner of a coarser cell of the mesh without graduation
    clipped = check_curve(without, margin)

    g = ""
    ticks = [(v, f"{v:g}") for v in (-2, -1, 0, 1, 2)]
    for frame, cells, graded in ((frames[0], without, False), (frames[1], with_, True)):
        g += draw_cells(
            p,
            cells,
            frame,
            graded=graded,
            xticks=ticks,
            yticks=ticks if frame is frames[0] else (),
        )
        g += path(_curve_path(frame), stroke=INK, sw=stroke, extra='fill="none"')
    # a: a cell of level 1 touches cells of level 4 along its left face; b: the same
    # place, where cells of levels 2 and 3 step down from level 4
    fa, fb = frames
    g += callout(*fa.center((1, 2, -1)), fa.x(1.75), fa.y(-1.1), "a")
    g += callout(*fb.center((3, 9, -2)), fb.x(1.75), fb.y(-1.1), "b", red=True)
    for k, name in enumerate(("without graduation", "with graduation")):
        g += fig_label(xs[k] + size / 2, oy + size + 46, k + 1, name)
    height = oy + size + 60
    counts = [_counts(cells) for cells in (without, with_)]
    return Drawing(
        PLATE_WIDTH,
        height,
        "Two square meshes of the domain [-2, 2) x [-2, 2), with the same closed curve drawn"
        " over both: a figure of Lissajous that loops twice through the center and fills"
        f" [-1, 1] x [-1, 1]. Left, the mesh refined without graduation, {len(without)} cells"
        f" of levels 1 to 4: the {counts[0][4]} cells of level 4, hatched in red, follow the"
        " curve in a thin band, and large cells of levels 1 to 3 touch them directly. Right,"
        f" the mesh refined with graduation, {len(with_)} cells: the {counts[1][4]} cells of"
        " level 4 form a wider band, and rings of cells of levels 3 and 2 separate them from"
        f" the cells of level 1. {len(clipped)} coarse cells of the left mesh are cut at a"
        " corner by the curve.",
        patterns(p),
        g,
        panels=(
            (0, 0, left + size + gap / 2, height, ""),
            (xs[1] - gap / 2, 0, size + gap, height, ""),
        ),
    )


# the interval of the diagram: pass, level, row s of the stencil, row j_f, interval
EXAMPLE = (3, 4, (1, 1), 12, (5, 11))


def _example():
    """Return the tags of the demo's run when it reaches the interval ``EXAMPLE``.

    Raise ``ValueError`` when the graduation step of the run does not visit it.
    """
    seen = []

    def record(*args):
        if args[:5] == EXAMPLE:
            seen.append(args[5])

    run_demo(START_LEVEL, MAX_LEVEL, True, record)
    if len(seen) != 1:
        raise ValueError(f"the graduation step does not visit the interval {EXAMPLE} once")
    return seen[0]


@figure
def even_odd_split(p):
    """Draw how one interval of the graduation step tags the cells of ``level - 1``.

    Figure of tutorial/graduation_case_3.md, at "Change the tags to keep the mesh
    graded". The interval is ``EXAMPLE``, taken from the run of the plate. As in
    the demo, ``i_f`` is first its even elements, then its odd elements, both of
    step 2; for each, the mask is ``tag(level, i_f - s[0], j_f - s[1])``, one
    value per element of ``i_f``, and ``i_f >> 1`` gives the cells of
    ``level - 1`` it tags where the mask is true.
    """
    _, level, s, j_f, (a, b) = EXAMPLE
    tags = _example()
    cw, ch, x0 = 48, 26, 300
    lx = x0 - cw - 16  # the right edge of the labels

    def fx(i):
        """Return the left edge of the fine cell ``i``."""
        return x0 + (i - a) * cw

    def cx(k):
        """Return the left edge of the coarse cell ``k``."""
        return fx(2 * k)

    def label(y, code, note):
        """Write the label of a row: its expression in the mono, a note under it."""
        s_ = text(lx, y + ch / 2 + 1, code, anchor="end", size=11.5, fill=INK, cls="sm-fig-code")
        return s_ + text(lx, y + ch / 2 + 17, note, anchor="end", size=10.5)

    def numbers(y, cells, width, left, knock=False):
        """Write the index of each cell in it, on a paper halo with ``knock``."""
        s_ = ""
        for k in cells:
            s_ += text(
                left(k) + width / 2,
                y + ch / 2 + 4,
                k,
                anchor="middle",
                size=11,
                fill=INK,
                cls="sm-fig-code",
                knock=knock,
            )
        return s_

    y = 14
    g = label(y, "interval", f"level {level}, row {j_f}")
    g += cell_row(x0, y, cw, b, [(a, b)], ch=ch, start=a)
    g += numbers(y, range(a, b), cw, fx)

    y += 100
    halves = []
    for name, split in (("even", even_elements), ("odd", odd_elements)):
        part = split(a, b)
        members = list(range(*part))
        # mask = tag(level, i_f - s[0], j_f - s[1]): one value per element of i_f
        mask = [(level, i - s[0], j_f - s[1]) in tags for i in members]
        # i_f >> 1, as Interval::operator>>= computes it
        c0, c1 = part[0] >> 1, ((part[1] - 1) >> 1) + 1
        if c1 - c0 != len(mask):
            raise ValueError("i_f >> 1 and the mask do not have the same number of elements")
        halves.append((name, part, mask, (c0, c1)))

        g += label(y, f"i_f = interval.{name}_elements()", f"level {level}, row {j_f}")
        g += strided_row(x0, y, cw, b, part, ch=ch, start=a, tone="ink")
        g += numbers(y, members, cw, fx)

        ym = y + ch + 44
        moved = (part[0] - s[0], part[1] - s[0])
        g += label(
            ym,
            f"mask = tag({level}, i_f - {s[0]}, {j_f - s[1]})",
            f"tags of [{moved[0]}, {moved[1]}), step 2, row {j_f - s[1]}",
        )
        for i, m in zip(members, mask):
            g += draw_array(
                Array(fx(i), ym, cw, 1, ch),
                ["true" if m else "false"],
                red=[0] if m else [],
                positions=False,
            )
        tagged = {i >> 1 for i, m in zip(members, mask) if m}

        yc = ym + ch + 52
        for i, m in zip(members, mask):
            g += line(
                fx(i) + cw / 2,
                ym + ch + 8,
                cx(i >> 1) + cw,
                yc - 6,
                stroke=RED if m else INK2,
                sw=RULE if m else HAIR,
                extra=f'marker-end="url(#{p}-ar)"' if m else f'marker-end="url(#{p}-a)"',
            )
        for k in range(c0, c1):
            fill = f"url(#{p}-r)" if k in tagged else "none"
            g += rect(cx(k), yc, 2 * cw, ch, fill=fill, stroke=INK)
        g += numbers(yc, range(c0, c1), 2 * cw, cx, knock=True)
        g += bracket(cx(c0), cx(c1), yc + ch + 9, f"[{c0}, {c1})")
        g += label(yc, "i_c = i_f >> 1", f"level {level - 1}, row {j_f >> 1}")
        y = yc + ch + 88

    height = y - 60

    def describe(name, part, mask, coarse):
        """Describe one half of the split for the aria-label."""
        values = ", ".join(f"{i}: {'true' if m else 'false'}" for i, m in zip(range(*part), mask))
        return (
            f"Its {name} elements form [{part[0]}, {part[1]}) with step 2; the mask computed on"
            f" them is, element by element, {values}; shifted right by one they give the cells"
            f" [{coarse[0]}, {coarse[1]}) of level {level - 1}, one per element, joined by"
            " arrows."
        )

    return Drawing(
        DIAGRAM_WIDTH,
        height,
        f"One interval of the graduation step, [{a}, {b}) on the row {j_f} of level {level},"
        f" for the row s = {{{s[0]}, {s[1]}}} of the stencil. "
        + " ".join(describe(*half) for half in halves)
        + " The mask values that are true and their arrows are red, and the coarse cell they"
        " reach is hatched in red: it is tagged.",
        patterns(p),
        g,
    )
