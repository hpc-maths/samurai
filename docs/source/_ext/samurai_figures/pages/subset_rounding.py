"""The outward rounding of a set projection, a figure of ``reference/subset.md``.

The interval [3, 6) of level 2 is brought to levels 1 and 0 with ``on``.
``a.on(level)`` builds a ``Projection`` (``subset/projection.hpp``). To a coarser
level, its traverser of the outermost direction (x in 1D, y in 2D, z in 3D) is a
``LastDimProjectionTraverser``, which shifts the bounds with ``coarsen_start`` and
``coarsen_end`` (``subset/traversers/utils.hpp``): the start is divided by 2^s and
rounded down, the end divided by 2^s and rounded up. The inner directions shift
their intervals with ``Interval::operator>>`` (``interval.hpp``), whose
``operator>>=`` applies the same formula. In 1D, the case drawn here, only
``coarsen_start`` and ``coarsen_end`` run.

``coarsen`` below computes the coarse intervals with that rule, and the figure
raises if they differ from what ``snippet/set_algebra/set_algebra_rounding.cpp``
prints (``set_algebra_rounding_output.txt``).
"""

import re
from pathlib import Path

from ..draw import FIG, HAIR, INK, RED, cell_row, line, patterns, rect, text
from ..registry import DIAGRAM_WIDTH, Drawing, figure

# The fine interval and its level, which the labels of the figure describe.
FINE_LEVEL, FINE = 2, (3, 6)
OUTPUT = (
    Path(__file__).resolve().parents[3]
    / "reference/snippet/set_algebra/set_algebra_rounding_output.txt"
)


def read_output(path=OUTPUT):
    """Return the level and the interval of each line the snippet prints, by name.

    A line reads ``a.on(1) (level 1): [1,3)@0:1``; the result maps ``"a.on(1)"``
    to ``(1, (1, 3))``.
    """
    printed = {}
    for row in path.read_text(encoding="utf-8").splitlines():
        m = re.fullmatch(r"(\S+) \(level (\d+)\): \[(-?\d+),(-?\d+)\)@.*", row)
        if not m:
            raise ValueError(f"{path.name}: unexpected line {row!r}")
        printed[m[1]] = (int(m[2]), (int(m[3]), int(m[4])))
    return printed


def coarsen(interval, shift):
    """Return the interval brought ``shift`` levels down: start floored, end ceiled."""
    start, end = interval
    return start >> shift, ((end - 1) >> shift) + 1


def partly_covered(interval, shift):
    """Return the coarse cells that hold some, but not all, of the fine ``interval``."""
    start, end = interval
    cells = []
    for k in range(*coarsen(interval, shift)):
        lo, hi = k << shift, (k + 1) << shift
        if not (start <= lo and hi <= end):
            cells.append(k)
    return cells


def _check():
    """Raise if the snippet output differs from the interval drawn or from the rule.

    The snippet must print ``FINE`` at ``FINE_LEVEL``, and for ``a.on(1)`` and
    ``a.on(0)`` the coarse intervals that ``coarsen`` computes.
    """
    printed = read_output()
    if printed.get("a") != (FINE_LEVEL, FINE):
        raise ValueError(f"the snippet builds {printed.get('a')}, the figure draws {FINE}")
    for level in range(FINE_LEVEL):
        name = f"a.on({level})"
        computed = (level, coarsen(FINE, FINE_LEVEL - level))
        if printed.get(name) != computed:
            raise ValueError(
                f"{name}: the rule gives {computed}, the program prints {printed.get(name)}"
            )


def _row_name(x, y, name, level, color):
    """Write the name of a row, its expression in mono over its level."""
    return text(x, y + 11, name, anchor="end", size=11.5, fill=color, cls="sm-fig-code") + text(
        x, y + 25, level, anchor="end", size=11
    )


def _rule(x, y, shift):
    """Write the rounding of the two bounds for a shift of ``shift`` levels."""
    q = 2**shift
    a, b = FINE
    start, end = coarsen(FINE, shift)
    return text(x, y + 11, f"start {a}/{q} = {a / q:g}, down to {start}", size=11) + text(
        x, y + 26, f"end {b}/{q} = {b / q:g}, up to {end}", size=11
    )


@figure
def set_projection_rounding(p):
    """Draw [3, 6) of level 2 and its set projections on levels 1 and 0, at true size."""
    _check()
    x0, unit, ch, pitch = 104, 172, 24, 58  # a level-0 cell is 172 units wide
    n0 = 2  # the drawing spans the cells [0, 2) of level 0
    top = 14
    rows = [(FINE_LEVEL, top)] + [(lv, top + r * pitch + 22) for r, lv in enumerate((1, 0), 1)]
    axis = rows[-1][1] + pitch - 4
    g = ""
    # the bounds of the fine interval, through every row
    for b in FINE:
        x = x0 + b * unit / 2**FINE_LEVEL
        g += line(x, top - 6, x, axis - 12, stroke=RED, sw=0.8, dash="2 3")
    for level, y in rows:
        cw = unit / 2**level
        n = n0 * 2**level
        if level == FINE_LEVEL:
            g += cell_row(x0, y, cw, n, [FINE], ch=ch, label_size=11)
            g += _row_name(x0 - 14, y, "a", f"level {level}", INK)
            continue
        shift = FINE_LEVEL - level
        g += cell_row(x0, y, cw, n, [coarsen(FINE, shift)], ch=ch, tone="red", label_size=11)
        for k in partly_covered(FINE, shift):
            g += rect(x0 + k * cw, y, cw, ch, fill=f"url(#{p}-r)", stroke=RED, sw=HAIR)
        g += _row_name(x0 - 14, y, f"a.on({level})", f"level {level}", RED)
        g += _rule(x0 + n0 * unit + 24, y, shift)
    g += line(24, top + pitch + 2, DIAGRAM_WIDTH - 8, top + pitch + 2, stroke=FIG)
    g += line(x0, axis, x0 + n0 * unit, axis)
    for k in range(n0 + 1):
        g += line(x0 + k * unit, axis, x0 + k * unit, axis + 5) + text(
            x0 + k * unit, axis + 18, k, anchor="middle", size=11.5
        )
    g += text(x0 - 14, axis + 18, "level 0", anchor="end", size=11.5)
    # the key of the hatching
    key = axis + 42
    g += rect(x0, key - 11, 22, 14, fill=f"url(#{p}-r)", stroke=RED, sw=HAIR)
    g += text(
        x0 + 32,
        key,
        'a coarse cell only partly covered by <tspan class="sm-fig-code">[3, 6)</tspan>',
        size=11.5,
    )
    return Drawing(
        DIAGRAM_WIDTH,
        key + 10,
        "Three levels of a 1D mesh, drawn at their true size over the cells 0 and 1 of level 0."
        " Level 2 holds the interval a = [3, 6). Brought to level 1 with a.on(1), it becomes"
        " [1, 3): the start 3 / 2 is rounded down to 1 and the end 6 / 2 is rounded up to 3. The"
        " cell 1 of level 1 is hatched, because a holds only one of its two children. Brought to"
        " level 0 with a.on(0), it becomes [0, 2): 3 / 4 is rounded down to 0 and 6 / 4 up to 2."
        " Both cells of level 0 are hatched, because each holds only part of a. Dotted red guides"
        " mark the bounds of a through every row.",
        patterns(p),
        g,
    )
