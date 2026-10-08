"""The figures of ``howto/loop.md``: the arguments that ``for_each_interval`` passes.

The mesh is the one of ``snippet/loop/for_each_interval_mesh.cpp``: a multiresolution
mesh of the unit square with all its cells at the maximum level. The figure reads
the lines the program prints from ``for_each_interval_mesh_output.txt``, the
checked-in output that the page shows, and draws the rows of those lines.
"""

import re
from pathlib import Path

from ..arrays import token_row
from ..draw import (
    HAIR,
    HEAVY,
    INK,
    INK2,
    INK3,
    RED,
    RULE,
    bracket,
    line,
    num,
    path,
    patterns,
    rect,
    text,
)
from ..registry import DIAGRAM_WIDTH, Drawing, figure

OUTPUT = (
    Path(__file__).resolve().parents[3]
    / "howto"
    / "snippet"
    / "loop"
    / "for_each_interval_mesh_output.txt"
)

# A line of the program: "Level: 5, x: [0,32)@498:1, y: 0".
LINE = re.compile(
    r"Level: (?P<level>\d+), x: (?P<interval>\[(?P<start>-?\d+),(?P<end>-?\d+)\)"
    r"@-?\d+:(?P<step>\d+)), y: (?P<y>-?\d+)"
)


def read_calls(file=OUTPUT):
    """Read the calls of the function in the output ``file``, skipping the ``...`` line.

    Return a list of ``(line, level, start, end, step, y)``, one per printed line.
    Raise ``ValueError`` for a line of another form.
    """
    calls = []
    for raw in Path(file).read_text(encoding="utf-8").splitlines():
        if raw.strip() in ("", "..."):
            continue
        m = LINE.fullmatch(raw.strip())
        if m is None:
            raise ValueError(f"{file}: unexpected line {raw!r}")
        values = (int(m[k]) for k in ("level", "start", "end", "step", "y"))
        calls.append((raw.strip(), *values))
    return calls


def check_uniform(calls):
    """Check that ``calls`` are the rows of a mesh of one level, one interval per row.

    The figure draws whole rows of 2^level cells, the first call at the bottom row
    and the last one at the top row. Return the level. Raise ``ValueError`` when
    the output tells otherwise.
    """
    level = calls[0][1]
    n = 2**level
    for raw, lv, start, end, step, _ in calls:
        if (lv, start, end, step) != (level, 0, n, 1):
            raise ValueError(f"{raw!r} is not the whole row [0, {n}) of level {level}")
    ys = [call[5] for call in calls]
    if ys != sorted(set(ys)) or ys[0] != 0 or ys[-1] != n - 1:
        raise ValueError(f"the printed rows {ys} do not run from 0 to {n - 1}")
    return level


def _printed_line(raw, y):
    """Set the first printed line in the mono, with its three arguments labelled under them."""
    m = LINE.fullmatch(raw)
    parts, k = [], 0
    for group, label in (("level", "level"), ("interval", "i"), ("y", "index[0]")):
        a, b = m.span(group)
        parts += [(raw[k:a], None, INK2), (raw[a:b], label, RED)]
        k = b
    return token_row(DIAGRAM_WIDTH / 2, y, parts, size=15, anchor="middle", label_size=11.5)


@figure
def interval_arguments(p):
    """Draw the first call of for_each_interval and the cells its arguments designate.

    Figure of howto/loop.md, from for_each_interval_mesh_output.txt.
    """
    calls = read_calls()
    level = check_uniform(calls)
    n = 2**level
    first, *_ = calls
    rows = [call[5] for call in calls]

    x0, cw, top, gap = 132, 13, 86, 30
    # the printed rows from the top: the last one, a break, then the others
    ytop = {rows[-1]: top}
    for r, row in enumerate(reversed(rows[:-1])):
        ytop[row] = top + cw + gap + r * cw
    bottom = ytop[0] + cw
    right = x0 + n * cw

    g = _printed_line(first[0], 30)
    for row, y in ytop.items():
        for k in range(n):
            if row == first[5]:
                g += rect(x0 + k * cw, y, cw, cw, fill=f"url(#{p}-r)", stroke=RED, sw=HAIR)
            else:
                g += rect(x0 + k * cw, y, cw, cw, stroke=INK, sw=HAIR)
        if row != first[5]:
            # the row of the call carries its arguments instead
            g += text(x0 - 8, y + cw / 2 + 4, row, anchor="end", size=11, fill=INK2)
    # the edge of the domain: whole at the bottom and the top, dashed across the break
    brk_a, brk_b = top + cw, top + cw + gap
    g += path(
        f"M{num(x0)} {num(brk_a)}V{num(top)}H{num(right)}V{num(brk_a)}"
        f"M{num(x0)} {num(brk_b)}V{num(bottom)}H{num(right)}V{num(brk_b)}",
        stroke=INK,
        sw=HEAVY,
        extra='stroke-linecap="square"',
    )
    for x in (x0, right):
        g += line(x, brk_a + 4, x, brk_b - 4, stroke=INK3, sw=RULE, dash="2 3")
    g += text(
        (x0 + right) / 2,
        (brk_a + brk_b) / 2 + 4,
        f"rows {rows[-2] + 1} to {rows[-1] - 1}, one interval each",
        anchor="middle",
        size=11,
        fill=INK3,
    )

    # level: the level of the row of the call, from its left end
    yr = ytop[first[5]] + cw / 2
    g += line(x0 - 16, yr, x0 - 4, yr, stroke=RED, sw=HAIR)
    g += text(
        x0 - 20, yr + 4, f"level = {level}", anchor="end", size=12, fill=RED, cls="sm-fig-code"
    )
    g += text(x0 - 20, yr + 19, f"cells 1/{n} wide", anchor="end", size=10.5, fill=INK2)

    # i: the half-open interval under the row
    g += bracket(x0, right, bottom + 9, f"i = [{first[2]}, {first[3]})", color=RED, size=12)

    # index[0]: the row, from its right end
    g += line(right + 4, yr, right + 16, yr, stroke=RED, sw=HAIR)
    g += text(
        right + 20,
        yr + 4,
        f"index[0] = {first[5]}",
        size=12,
        fill=RED,
        cls="sm-fig-code",
    )

    return Drawing(
        DIAGRAM_WIDTH,
        bottom + 40,
        f"The first line the program prints, {first[0]}, with its parts labelled level, i and"
        f" index[0]. Under it, rows {rows[-1]}, {', '.join(str(r) for r in rows[-2::-1])} of the"
        f" level-{level} mesh, {n} cells each, with a break for the rows in between; every row has"
        f" level {level}. The bottom row, hatched in red, is the interval of that first call: a"
        f" label at its left end reads level = {level}, cells 1/{n} wide, the half-open bracket"
        f" under it reads i = [{first[2]}, {first[3]}), and a label at its right end reads"
        f" index[0] = {first[5]}.",
        patterns(p),
        g,
    )
