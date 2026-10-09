"""The figures of ``howto/adapt.md``: the adaptation loop and the adaptation in 1D.

The detail diagram of the page is ``child_details``, from ``getting_started.py``.
"""

import re
from math import exp
from pathlib import Path

from ..cells import Frame as CellFrame
from ..cells import draw_cells, from_intervals, load_cells
from ..draw import (
    EMPTY,
    FIG,
    HAIR,
    HEAVY,
    INK,
    INK2,
    RED,
    RULE,
    WASH,
    callout,
    fig_label,
    line,
    patterns,
    rect,
    text,
)
from ..flows import Loop, Row, Station, cycle
from ..plots import Frame, axes, level_rows, level_threshold, log_ticks, steps
from ..registry import PLATE_WIDTH, Drawing, figure

HERE = Path(__file__).parent
# The mesh of mr_adapt.cpp run with --max-level 5 --mr-eps 1e-2. mr_adapt.cpp does
# not save its mesh, so it was exported from a copy of it:
#   1. copy docs/source/howto/snippet/adapt/mr_adapt.cpp, add
#      #include <samurai/io/hdf5.hpp> and, right after MRadaptation(mra_config, v);,
#      samurai::save(std::filesystem::path("export"), "adapted", mesh);
#   2. build the copy like the snippets (-DBUILD_SNIPPETS=ON) and run it with
#      --max-level 5 --mr-eps 1e-2 --save-debug-fields; it prints the same cell
#      counts as mr_adapt with these options (16, 144 and 192 cells at levels 3 to 5);
#   3. python docs/tools/export_cells.py export/adapted.h5 \
#          docs/source/_ext/samurai_figures/data/adapt.json --source "..."
ADAPTED = HERE.parent / "data" / "adapt.json"
# The output of mr_adapt_1d.cpp, written by the target update_snippet_outputs
OUTPUT_1D = HERE.parents[2] / "howto" / "snippet" / "adapt" / "mr_adapt_1d_output.txt"

# --- the adaptation loop -----------------------------------------------------------------

# Each box holds the widest name, "coarsen or refine", about 133 units in the serif
# italic at 14; the four boxes fill the width of the plate and leave 24 units for
# each arrow.
LOOP_STATIONS = (
    Station("compute details", "compute_detail"),
    Station("tag", "mr_criteria"),
    Station("coarsen or refine", "CellFlag::refine"),
    Station("graduate", "make_graduation"),
)
LOOP_NOTE = (
    'at most <tspan class="sm-fig-code" font-style="normal" font-size="12">'
    "max_level - min_level</tspan> times"
)


def _block(x, y, cells, fills):
    """Draw a block of 92 by 92 units at (x, y), ``cells`` (i, j, side) on the grid of ``side``.

    ``fills`` maps a cell to its fill; the other cells are left blank. The block has
    the heavy outline.
    """
    g = ""
    for i, j, s in cells:
        fill = fills.get((i, j, s), "none")
        g += rect(x + i * s, y + j * s, s, s, fill=fill, sw=HAIR if fill != "none" else EMPTY)
    return g + rect(x, y, 92, 92, sw=HEAVY)


# The 4 by 4 cells of level l in the blocks of the stations 2 to 4; the 2 by 2 cells
# at the top left are coarsened into one cell of level l - 1, the 2 by 2 cells at the
# bottom right refined into 4 by 4 cells of level l + 1.
_FINE = [(i, j, 23) for i in range(4) for j in range(4)]
_TOP_LEFT = [(i, j, 23) for i in (0, 1) for j in (0, 1)]
_BOTTOM_RIGHT = [(i, j, 23) for i in (2, 3) for j in (2, 3)]
_SPLIT = [(i, j, 11.5) for i in range(4, 8) for j in range(4, 8)]
_MERGED = (0, 0, 46)


def _detail_art(p, cx, top):
    """Draw a parent between two neighbors at level l - 1, its two children at level l under it.

    Above the cells, their values: the level l - 1 values in ink, the predicted values
    of the children dashed, their real values in red. The prediction is the one of
    samurai with a stencil of radius 1, parent -/+ (right - left) / 8, ``plots.predict_children``;
    the real values average
    to the parent like the predicted ones, so the two details are opposite.
    """
    w, x0 = 40, cx - 60
    fr = Frame(x0, top + 4, 3 * w, 52, xlim=(0, 3))
    left, parent, right = 0.1, 0.5, 0.9
    slope = (right - left) / 8
    predicted = (parent - slope, parent + slope)
    real = (parent - 0.36, parent + 0.36)
    heads = f'marker-start="url(#{p}-ar)" marker-end="url(#{p}-ar)"'
    g = ""
    # the values of level l - 1
    for k, v in enumerate((left, parent, right)):
        g += line(fr.X(k) + 3, fr.Y(v), fr.X(k + 1) - 3, fr.Y(v), stroke=INK, sw=HEAVY)
    # the predicted and the real values of the children, and their details
    for k in range(2):
        xa, xb = fr.X(1 + k / 2) + 2, fr.X(1.5 + k / 2) - 2
        xm = (xa + xb) / 2
        g += line(xa, fr.Y(predicted[k]), xb, fr.Y(predicted[k]), stroke=INK2, dash="3 2")
        g += line(xa, fr.Y(real[k]), xb, fr.Y(real[k]), stroke=RED, sw=HEAVY)
        g += line(xm, fr.Y(predicted[k]), xm, fr.Y(real[k]), stroke=RED, extra=heads)
    for k, (label, anchor) in enumerate(((f"{chr(0x2212)}d", "end"), ("d", "start"))):
        y = (fr.Y(predicted[k]) + fr.Y(real[k])) / 2 + 4
        x = fr.X(1) - 4 if k == 0 else fr.X(2) + 4
        g += text(x, y, label, anchor=anchor, size=13, fill=RED, cls="sm-fig-math")
    # the cells: level l - 1, then the two children of the parent, as wide as it
    y = top + 62
    for k in range(3):
        parent_cell = k == 1
        g += rect(
            x0 + k * w,
            y,
            w,
            14,
            fill=WASH if parent_cell else "none",
            stroke=INK if parent_cell else FIG,
            sw=RULE if parent_cell else EMPTY,
        )
    for k in range(2):
        g += rect(x0 + w + k * w / 2, y + 16, w / 2, 14, fill=WASH)
    return g


def _tag_art(p, cx, top):
    """Draw level l cells: the top left ones to coarsen (ruled), the bottom right to refine."""
    fills = {c: f"url(#{p}-k)" for c in _TOP_LEFT}
    fills.update({c: f"url(#{p}-r)" for c in _BOTTOM_RIGHT})
    return _block(cx - 46, top, _FINE, fills)


def _apply_art(p, cx, top):
    """Draw the tags applied: one coarse cell, ruled, and 16 fine ones, hatched; not graded yet.

    The coarse cell (level l - 1) and the fine cells (level l + 1) touch by a corner,
    two levels apart: the callout points at it.
    """
    x = cx - 46
    cells = [_MERGED] + [c for c in _FINE if c not in _TOP_LEFT and c not in _BOTTOM_RIGHT]
    fills = {_MERGED: f"url(#{p}-k)", **{c: f"url(#{p}-r)" for c in _SPLIT}}
    g = _block(x, top, cells + _SPLIT, fills)
    g += text(
        cx, top + 108, "not graded yet", anchor="middle", size=12, fill=RED, cls="sm-fig-math"
    )
    return g + callout(x + 46, top + 46, cx - 58, top - 12, "a", red=True)


def _graduate_art(p, cx, top):
    """Draw the coarse cell refined back into its four cells of level l, hatched: graded."""
    fills = {c: f"url(#{p}-r)" for c in _TOP_LEFT}
    cells = [c for c in _FINE if c not in _BOTTOM_RIGHT] + _SPLIT
    return _block(cx - 46, top, cells, fills)


@figure
def adapt_loop(p):
    """Draw the uniform mesh, the adapted mesh, and the loop of one call of the adaptation.

    Figure of howto/adapt.md. The two meshes are those of ``mr_adapt.cpp`` run with
    ``--max-level 5 --mr-eps 1e-2``: before the call every cell is at level 5, after
    it the cells are those of ``adapt.json``. The loop follows ``Adapt::operator()``
    and ``Adapt::harten`` in include/samurai/mr/adapt.hpp.
    """
    size, top, margin = 216, 16, 28
    xa, xb = margin, PLATE_WIDTH - margin - size
    adapted = load_cells(ADAPTED)
    uniform = from_intervals({5: [(j, 0, 32) for j in range(32)]})

    def frame(x):
        return CellFrame((0, 0, 1, 1), x, top, size)

    g = draw_cells(p, uniform, frame(xa), hatch=False)
    g += draw_cells(p, adapted.leaves, frame(xb))
    # the call between the meshes
    mid, y = PLATE_WIDTH / 2, top + size / 2
    g += line(xa + size + 14, y, xb - 14, y, stroke=INK2, extra=f'marker-end="url(#{p}-a)"')
    g += text(
        mid, y - 10, "MRadaptation(mra_config)", anchor="middle", size=10.5, cls="sm-fig-code"
    )
    g += text(
        mid, y + 22, "the loop of Fig. 3", anchor="middle", size=13, fill=INK, cls="sm-fig-math"
    )
    label_y = top + size + 32
    g += fig_label(xa + size / 2, label_y, 1, "uniform")
    g += fig_label(xb + size / 2, label_y, 2, "adapted")
    # the loop
    row = Row(
        x0=PLATE_WIDTH / 2 - 1.5 * 176, pitch=176, box=140, top=label_y + 52, gap=6, first=None
    )
    flow = cycle(p, row, LOOP_STATIONS, Loop(LOOP_NOTE))
    arts = (_detail_art, _tag_art, _apply_art, _graduate_art)
    g += "".join(art(p, row.center(k), row.top) for k, art in enumerate(arts))
    loop_y = flow.height + 30
    g += fig_label(mid, loop_y, 3, "one call")
    # on a phone: the two meshes without the arrow between them (it starts 14 units
    # from each mesh), the four stations, then the note of the loop with the label of
    # Fig. 3, the panel of the note (inside its paper band, see flows.cycle)
    # stretched down to that label
    *stations, (x, y_note, w, _h, kind) = flow.panels
    mesh_h = label_y + 10
    panels = (
        (xa - 10, 0, size + 20, mesh_h, "half"),
        (xb - 10, 0, size + 20, mesh_h, "half"),
        *stations,
        (x, y_note, w, loop_y + 8 - y_note, kind),
    )
    counts = {}
    for level, *_ in adapted.leaves:
        counts[level] = counts.get(level, 0) + 1
    levels = ", ".join(f"{n} cells at level {level}" for level, n in sorted(counts.items()))
    return Drawing(
        PLATE_WIDTH,
        loop_y + 10,
        "Top: a square mesh of 32 by 32 cells, all at level 5, and an arrow labelled"
        " MRadaptation(mra_config) to the adapted mesh, with " + levels + "; the finest"
        " cells, hatched in red, form a block around the center. Bottom: the loop of one call,"
        " four stations linked by arrows. Compute details: a parent between two neighbors,"
        " its two children under it, and above them the predicted values, dashed, and the real"
        " values in red, at opposite distances minus d and d. Tag: a block of 4 by 4 cells, the"
        " top left quarter ruled to coarsen and the bottom right quarter hatched to refine."
        " Coarsen or refine: the top left quarter is now one coarse cell and the bottom right"
        " quarter 16 fine cells; they touch by a corner, two levels apart, not graded yet."
        " Graduate: the coarse cell is split back into four hatched cells. An arrow returns from"
        " the last station to the first, at most max_level minus min_level times.",
        patterns(p),
        g + flow.body,
        panels=panels,
    )


# --- the adaptation in one dimension -----------------------------------------------------

_LINE = re.compile(
    r"level (?P<level>\d+): threshold (?P<eps>\S+), (?P<n>\d+) cells(?P<runs>( \[\d+, \d+\))*)$"
)


def read_output_1d(file=OUTPUT_1D):
    """Read the thresholds and the cells of each level printed by ``mr_adapt_1d``.

    Return ``(thresholds, leaves)``: ``thresholds`` maps each level to its threshold,
    ``leaves`` lists the cells ``(level, k)``. Raise ``ValueError`` when a line does
    not parse, when the count of a level does not match its intervals, or when the
    total does not match.
    """
    thresholds, leaves, total = {}, [], None
    for raw in Path(file).read_text(encoding="utf-8").splitlines():
        match = _LINE.match(raw)
        if match:
            level = int(match["level"])
            thresholds[level] = float(match["eps"])
            cells = [
                (level, k)
                for a, b in re.findall(r"\[(\d+), (\d+)\)", match["runs"])
                for k in range(int(a), int(b))
            ]
            if len(cells) != int(match["n"]):
                raise ValueError(
                    f"{file}: level {level} counts {match['n']} cells, lists" f" {len(cells)}"
                )
            leaves += cells
        elif raw.startswith("total: "):
            total = int(raw.split()[1])
        else:
            raise ValueError(f"{file}: cannot read the line {raw!r}")
    if total != len(leaves):
        raise ValueError(f"{file}: the total {total} does not match the {len(leaves)} cells")
    return thresholds, leaves


def check_mesh_1d(leaves):
    """Raise ``ValueError`` unless the 1D cells tile [0, 1) and the mesh is graded."""
    cells = sorted((k / 2**level, (k + 1) / 2**level, level) for level, k in leaves)
    if cells[0][0] != 0 or cells[-1][1] != 1:
        raise ValueError("the cells do not cover [0, 1)")
    for (_, b, level), (a, _, other) in zip(cells, cells[1:]):
        if a != b:
            raise ValueError(f"the cells leave a gap or overlap at {b}")
        if abs(level - other) > 1:
            raise ValueError(f"the mesh is not graded at {b}: levels {level} and {other}")


def bump(x):
    """Return the field of mr_adapt_1d.cpp at ``x``."""
    return exp(-200 * (x - 0.5) ** 2)


@figure
def adapt_1d(p):
    """Draw the bump of mr_adapt_1d.cpp, its adapted mesh and the threshold of each level.

    Figure of howto/adapt.md, drawn from mr_adapt_1d_output.txt: the cells and the
    thresholds are those the program prints. The thresholds are checked against
    ``level_threshold``, the formula of include/samurai/mr/adapt.hpp.
    """
    thresholds, leaves = read_output_1d()
    check_mesh_1d(leaves)
    max_level, min_level = max(thresholds), min(thresholds)
    epsilon = thresholds[max_level]
    for level, eps_l in thresholds.items():
        expected = level_threshold(epsilon, level, max_level)
        if abs(eps_l - expected) > 0.01 * expected:
            raise ValueError(f"the threshold of level {level} is {eps_l}, not {expected}")
    finest = max(level for level, _ in leaves)
    x0, w = 120, 470

    # Fig. 1, the bump as the program sets it: its value at the center of each cell
    fr = Frame(x0, 26, w, 140, ylim=(0, 1.08))
    cells = sorted(
        (k / 2**level, (k + 1) / 2**level, bump((k + 0.5) / 2**level), level) for level, k in leaves
    )
    g = steps(
        fr,
        [(a, b, v) for a, b, v, _ in cells],
        sw=HEAVY,
        fills=[f"url(#{p}-r)" if level == finest else None for *_, level in cells],
    )
    g += axes(
        fr,
        xticks=[(0, "0"), (0.5, "0.5"), (1, "1")],
        yticks=[(0, "0"), (1, "1")],
        xname="x",
        yname="u",
    )
    g += callout(fr.X(0.5), fr.Y(1), fr.X(0.66), fr.Y(0.98), "a", red=True)
    g += callout(fr.X(1 / 16), fr.Y(0), fr.X(0.12), fr.Y(0.32), "b")
    g += fig_label(x0 + w / 2, fr.bottom + 46, 1)

    # Fig. 2, the cells of each level. level_rows starts at the coarsest level that
    # holds cells: the empty levels above it, from the minimum level of the mesh, get
    # the dashed line and the "0 cells" of an empty row, in the style of level_rows.
    rows_y = fr.bottom + 82
    present = sorted({level for level, _ in leaves})
    for r, level in enumerate(range(min_level, present[0])):
        yy = rows_y + r * 30
        g += line(x0, yy + 16, x0 + w, yy + 16, stroke=FIG, dash="3 3")
        g += text(x0 - 12, yy + 12, f"level {level}", anchor="end", size=12, fill=INK)
        g += text(x0 + w + 12, yy + 12, "0 cells", size=11.5, fill=INK2)
    g += level_rows(x0, rows_y + (present[0] - min_level) * 30, w, leaves, p)
    rows_bottom = rows_y + (present[-1] - min_level) * 30 + 16
    g += fig_label(x0 + w / 2, rows_bottom + 34, 2)

    # Fig. 3, the threshold of each level, on a logarithmic scale
    bars = Frame(x0, rows_bottom + 92, w, 120, ylim=(3e-5, 2e-3), ylog=True)
    g += axes(
        bars,
        yticks=log_ticks(-4, -3),
        yname='ϵ<tspan baseline-shift="sub"' ' font-size="10">ℓ</tspan>',
    )
    pitch = w / len(thresholds)
    for q, level in enumerate(sorted(thresholds)):
        cx, top = bars.x + (q + 0.5) * pitch, bars.Y(thresholds[level])
        red = level == max_level
        if red:
            g += rect(cx - 17, top, 34, bars.bottom - top, fill=f"url(#{p}-r)", stroke=RED, sw=HAIR)
        else:
            g += rect(cx - 17, top, 34, bars.bottom - top, fill=WASH)
        tone = RED if red else INK
        g += text(cx, bars.bottom + 17, f"level {level}", anchor="middle", size=11.5, fill=tone)
        ratio = 2 ** (max_level - level)
        label = "ϵ" if ratio == 1 else f"ϵ/{ratio}"
        g += text(
            cx,
            top - 7,
            label,
            anchor="middle",
            size=13,
            fill=RED if red else INK2,
            cls="sm-fig-math",
        )
    bars_label = bars.bottom + 50
    g += fig_label(x0 + w / 2, bars_label, 3)

    counts = ", ".join(
        f"{sum(1 for lv, _ in leaves if lv == level)} at level {level}" for level in present
    )
    # on a phone each figure keeps its natural size and scrolls sideways: scaled to
    # the screen, the labels of the levels would be too small to read
    left, right = x0 - 70, x0 + w + 70
    panels = (
        (left, 0, right - left, fr.bottom + 56, "wide"),
        (left, fr.bottom + 56, right - left, rows_bottom + 44 - fr.bottom - 56, "wide"),
        (left, rows_bottom + 44, right - left, bars_label + 10 - rows_bottom - 44, "wide"),
    )
    return Drawing(
        PLATE_WIDTH,
        bars_label + 10,
        "Three figures of one dimension. Top: the bump u(x) = exp(-200 (x - 1/2)^2) on [0, 1)"
        " sampled at the centers of the adapted cells and drawn as steps, one per cell; the"
        " steps of the finest cells are hatched in red. Middle: the adapted mesh, one row per"
        " level, with "
        + counts
        + f" cells; the finest cells sit on the bump. Bottom: the threshold of each level from"
        f" {min_level} to {max_level}, on a logarithmic scale, halving at each coarser level,"
        f" from epsilon at level {max_level} to epsilon/{2 ** (max_level - min_level)} at"
        f" level {min_level}.",
        patterns(p),
        g,
        panels=panels,
    )
