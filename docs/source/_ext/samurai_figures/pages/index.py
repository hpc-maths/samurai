"""The figure of ``index.md``: the adaptive mesh and the intervals that store it."""

from math import cos, hypot, sin

from ..draw import (
    EMPTY,
    HEAVY,
    HAIR,
    INK,
    INK2,
    INK3,
    PAPER,
    RED,
    RULE,
    WASH,
    bracket,
    callout,
    fig_label,
    line,
    patterns,
    rect,
    text,
)
from ..mesh import build_mesh, check_graded, circle_refine, draw_mesh
from ..registry import PLATE_WIDTH, Drawing, figure

MIN_LEVEL, MAX_LEVEL = 2, 6
# The circle passes at least 1.77 units (at the size of Fig. 1) from every cell
# coarser than the finest level, so its outline, drawn with the heavy rule, stays in
# the red cells all around; a circle that grazes a coarse cell seems to leave them.
DISC = (0.469, 0.531, 0.302)
# The row a-a, a row of cells of the finest level, just above the center of the disc.
ROW = 34


def row_intervals(leaves, min_level, max_level, row):
    """Return the intervals of each level that the row ``row`` of ``max_level`` crosses.

    The row ``row`` of ``max_level`` lies in the row ``row >> (max_level - l)`` of
    each coarser level l. The result maps each level to its half-open intervals
    ``(start, end)`` of cell indices, counted in cells of that level. Raise
    ``ValueError`` if the levels do not cover the row exactly once.
    """
    intervals, covered = {}, 0
    for level in range(min_level, max_level + 1):
        j = row >> (max_level - level)
        runs = []
        for i in sorted(i for lv, i, jj in leaves if lv == level and jj == j):
            if runs and runs[-1][1] == i:
                runs[-1][1] = i + 1
            else:
                runs.append([i, i + 1])
        intervals[level] = [tuple(run) for run in runs]
        covered += sum((b - a) * 2 ** (max_level - level) for a, b in runs)
    if covered != 2**max_level:
        raise ValueError(f"the levels cover {covered} of the {2**max_level} cells of the row")
    return intervals


def check_outline(leaves, disc, max_level, size):
    """Raise ``ValueError`` unless the outline of the circle lies in cells of ``max_level``.

    Every cell the circle ``disc`` (cx, cy, r) crosses must be of ``max_level``, and
    the circle, drawn ``size`` units for the unit square, must pass at least the
    width of its heavy rule from every coarser cell, so that the rule stays inside
    the hatched cells.
    """
    cx, cy, r = disc
    for level, i, j in leaves:
        if level == max_level:
            continue
        h = 2.0**-level
        x0, y0 = i * h, j * h
        dmin = hypot(max(x0, min(cx, x0 + h)) - cx, max(y0, min(cy, y0 + h)) - cy)
        dmax = hypot(max(abs(cx - x0), abs(cx - x0 - h)), max(abs(cy - y0), abs(cy - y0 - h)))
        if dmin <= r <= dmax:
            raise ValueError(f"the circle crosses the cell {(level, i, j)} of level {level}")
        gap = (dmin - r if dmin > r else r - dmax) * size
        if gap < HEAVY:
            raise ValueError(
                f"the circle passes {gap:.2f} units from the cell {(level, i, j)} of level"
                f" {level}: its outline would seem to cross it"
            )


def _mesh_figure(p, leaves, ox, oy, size):
    """Fig. 1: the mesh, the row a-a across it and the callouts."""

    def X(u):
        return ox + u * size

    def Y(v):
        return oy + (1 - v) * size

    cx, cy, r = DISC
    g = draw_mesh(p, leaves, ox, oy, size, MAX_LEVEL, disc=DISC, axes=False)
    # the axes; the tick 1/2 on the y axis would sit on the row a-a, so only 1 is marked
    for v, label in ((0, "0"), (0.5, "1/2"), (1, "1")):
        g += line(X(v), oy + size, X(v), oy + size + 5)
        g += text(X(v), oy + size + 18, label, anchor="middle", size=11)
    g += line(ox - 5, Y(1), ox, Y(1)) + text(ox - 8, Y(1) + 4, "1", anchor="end", size=11)
    # the row a-a, through the centers of its cells
    y = Y((ROW + 0.5) / 2**MAX_LEVEL)
    g += line(ox - 14, y, ox + size + 14, y, dash="12 3 2 3")
    g += text(ox - 19, y + 5, "a", anchor="end", size=15, fill=INK, cls="sm-fig-math")
    g += text(ox + size + 19, y + 5, "a", size=15, fill=INK, cls="sm-fig-math")
    edge = (cx + r * cos(-0.75), cy + r * sin(-0.75))
    g += callout(X(edge[0]), Y(edge[1]), X(0.9), Y(0.08), "b", red=True)
    g += callout(X(0.0625), Y(0.9375), X(0.19), Y(0.81), "c")
    g += callout(X(cx - 0.05), Y(cy + 0.06), X(cx + 0.08), Y(cy + 0.16), "d")
    return g


def _section(p, intervals, rx, rw, top, pitch):
    """Fig. 3: the cells of the row a-a at each level, with their intervals."""
    g = ""
    for level in range(MIN_LEVEL, MAX_LEVEL + 1):
        y, cw = top + (level - MIN_LEVEL) * pitch, rw / 2**level
        fine = level == MAX_LEVEL
        color = RED if fine else INK
        g += line(rx, y + 18, rx + rw, y + 18, stroke=INK3, dash="3 3")
        g += text(rx - 14, y + 14, f"level {level}", anchor="end", size=12, fill=color)
        if not intervals[level]:
            note = "no cell of this level in the row"
            w = 0.385 * 11.5 * len(note) + 16
            g += rect(rx + (rw - w) / 2, y + 10, w, 14, fill=PAPER, stroke="none")
            g += text(rx + rw / 2, y + 21, note, anchor="middle", size=11.5, fill=INK3)
        # a label that would come closer than 8 units to the one before drops a line
        ends = [float("-inf"), float("-inf")]
        for a, b in intervals[level]:
            for k in range(a, b):
                g += rect(
                    rx + k * cw,
                    y,
                    cw,
                    18,
                    fill=f"url(#{p}-r)" if fine else WASH,
                    stroke=color,
                    sw=HAIR if fine else RULE,
                )
            label = f"[{a}, {b})"
            mid, half = rx + (a + b) / 2 * cw, 0.3 * 10.5 * len(label) + 2
            drop = 0 if mid - half >= ends[0] + 8 else 1
            ends[drop] = mid + half
            g += bracket(
                rx + a * cw,
                rx + b * cw,
                y + 25,
                label,
                color=color,
                size=10.5,
                r=2.4,
                label_dy=13 + 12 * drop,
            )
    return g


@figure
def adaptive_mesh(p):
    """index.md: a graded mesh around a disc, a cell of each level, the intervals of a row."""
    leaves = build_mesh(MIN_LEVEL, MAX_LEVEL, circle_refine(*DISC))
    size, ox, oy = 340, 62, 14
    check_graded(leaves, MAX_LEVEL)
    check_outline(leaves, DISC, MAX_LEVEL, size)
    intervals = row_intervals(leaves, MIN_LEVEL, MAX_LEVEL, ROW)
    g = _mesh_figure(p, leaves, ox, oy, size)
    base = oy + size + 46
    g += fig_label(ox + size / 2, base, 1)

    # Fig. 2 and its "Fig. 2.", centered on the height of the mesh
    fx = ox + size + 76
    heights = [size * 2.0**-level for level in range(MIN_LEVEL, MAX_LEVEL + 1)]
    stack = sum(max(h, 12) for h in heights) + 14 * (len(heights) - 1)
    y = y2 = oy + (size - stack - 36) / 2
    for level, h in zip(range(MIN_LEVEL, MAX_LEVEL + 1), heights):
        fine = level == MAX_LEVEL
        g += rect(
            fx,
            y,
            h,
            h,
            fill=f"url(#{p}-r)" if fine else "none",
            stroke=RED if fine else INK,
            sw=HAIR if fine else EMPTY,
        )
        ty = y + min(h, 22) / 2 + 4
        g += text(fx + 104, ty, f"level {level}", size=12, fill=RED if fine else INK)
        g += text(fx + 158, ty, f"side 1/{2**level}", size=11.5, fill=RED if fine else INK2)
        y += max(h, 12) + 14
    # the column runs from fx to the end of "side 1/64", 200 units further
    g += fig_label(fx + 100, y + 22, 2)

    rx, rw, pitch = 80, 594, 62
    top = base + 52
    g += _section(p, intervals, rx, rw, top, pitch)
    end = top + 4 * pitch + 66
    g += fig_label(rx + rw / 2, end, 3, "Section a-a")
    return Drawing(
        PLATE_WIDTH,
        end + 12,
        "A square mesh built from level 2 to level 6 and refined around a circle; the finest"
        " cells, hatched in red, follow the circle, the cells coarsen away from it and inside the"
        " disc, and a dash-dotted line a-a crosses the mesh just above the center of the disc."
        " Beside it, one cell of each level from 2 to 6 at the same scale. Below, the cells that"
        " the row a-a crosses, level by level, each run marked as a half-open interval of indices:"
        + _describe(intervals)
        + ".",
        patterns(p),
        g,
        panels=(
            # Fig. 1 and Fig. 2 at the same scale: two panels of the same width, the
            # cells of Fig. 2 as far from the left edge as the mesh of Fig. 1
            (ox - 40, 0, 420, base + 14, ""),
            (fx - 40, y2 - 16, 420, y + 22 - y2 + 30, ""),
            (0, top - 18, PLATE_WIDTH, end + 12 - top + 18, "wide"),
        ),
    )


def _describe(intervals):
    """Write the intervals of each level in words, for the description of the drawing."""
    parts = []
    for level, runs in intervals.items():
        if runs:
            parts.append(f" level {level}, " + ", ".join(f"[{a}, {b})" for a, b in runs))
        else:
            parts.append(f" no cell of level {level}")
    return ";".join(parts)
