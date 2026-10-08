"""The figure of ``howto/geometry.md``: the mesh adapted around the disc obstacle.

The mesh is the one ``docs/source/howto/snippet/geometry/disc_obstacle.cpp`` saves
in ``disc_obstacle.h5``. Build the ``disc_obstacle`` target with
``-DBUILD_SNIPPETS=ON``, run it, then export its file with
``docs/tools/export_cells.py`` into ``data/geometry.json``, ``--source`` naming the
program and the file. The figure checks the exported cells against the counts per
level that the program prints in ``disc_obstacle_output.txt``.
"""

import re
from collections import Counter
from math import hypot
from pathlib import Path

from ..cells import Frame, draw_cells, load_cells
from ..draw import (
    EMPTY,
    HAIR,
    HEAVY,
    INK,
    INK2,
    PAPER,
    RED,
    RULE,
    circle,
    line,
    num,
    path,
    patterns,
    rect,
    text,
)
from ..registry import DIAGRAM_WIDTH, Drawing, figure

DATA = Path(__file__).resolve().parent.parent / "data"
OUTPUT = (
    Path(__file__).resolve().parents[3]
    / "howto"
    / "snippet"
    / "geometry"
    / "disc_obstacle_output.txt"
)

# the disc of the program: the level-set function is |x - (0.5, 0.5)| - 0.2
CX, CY, R = 0.5, 0.5, 0.2
MIN_LEVEL, MAX_LEVEL = 2, 6
# the window of the channel that the second view enlarges
WINDOW = (0.25, 0.25, 0.75, 0.75)


def _printed_counts():
    """Return the number of cells per level that the program prints after adaptation."""
    lines = OUTPUT.read_text(encoding="utf-8")
    return {int(level): int(n) for level, n in re.findall(r"^level (\d+): (\d+)$", lines, re.M)}


def _mesh():
    """Return the cells of the adapted mesh, checked against the output of the program."""
    leaves = load_cells(DATA / "geometry.json").leaves
    counts = dict(Counter(leaf[0] for leaf in leaves))
    printed = _printed_counts()
    if counts != printed:
        raise ValueError(
            f"data/geometry.json has the cells per level {counts}, the program prints {printed}:"
            " export disc_obstacle.h5 again"
        )
    return leaves


def _check_hole(leaves):
    """Raise ``ValueError`` unless the hole is the level 6 cells with their center in the disc.

    The program keeps a cell of the uniform mesh at ``MAX_LEVEL`` when its center
    lies outside the disc; adaptation only merges cells of the domain. So the
    cells of ``MAX_LEVEL`` the mesh does not cover over [0, 2) x [0, 1) must be
    exactly the ones with their center in the disc.
    """
    n = 2**MAX_LEVEL
    covered = set()
    for level, i, j in leaves:
        s = 2 ** (MAX_LEVEL - level)
        covered.update((i * s + a, j * s + b) for a in range(s) for b in range(s))
    h = 1 / n
    for x in range(2 * n):
        for y in range(n):
            inside = hypot((x + 0.5) * h - CX, (y + 0.5) * h - CY) <= R
            if inside == ((x, y) in covered):
                where = "is covered" if inside else "is not covered"
                raise ValueError(f"the cell ({MAX_LEVEL}, {x}, {y}) {where}, against the program")


def _check_edge(leaves, unit, sw):
    """Raise ``ValueError`` unless the drawn circle stays in the cells of ``MAX_LEVEL``.

    The circle, drawn with the stroke ``sw`` at ``unit`` drawing units per unit of
    length, must pass at least half its stroke plus a heavy rule away from every
    coarser cell, so that its outline lies in the hatched cells or in the hole and
    never seems to cross a coarse cell.
    """
    for level, i, j in leaves:
        if level == MAX_LEVEL:
            continue
        h = 2.0**-level
        x0, y0 = i * h, j * h
        dmin = hypot(max(x0, min(CX, x0 + h)) - CX, max(y0, min(CY, y0 + h)) - CY)
        dmax = hypot(max(abs(CX - x0), abs(CX - x0 - h)), max(abs(CY - y0), abs(CY - y0 - h)))
        gap = (dmin - R if dmin > R else R - dmax) * unit
        if dmin <= R <= dmax or gap < sw / 2 + HEAVY:
            raise ValueError(
                f"the circle passes {gap:.2f} units from the cell {(level, i, j)} of level {level}"
            )


def _circle(frame, clip=""):
    """Return the exact circle in ``frame``, clipped to the path ``clip`` when given."""
    s = circle(frame.x(CX), frame.y(CY), R * frame.unit, stroke=RED, sw=RULE)
    return f'<g clip-path="url(#{clip})">{s}</g>' if clip else s


def _corner_marks(leaves, frame, arm=7):
    """Return the four L-shaped marks at the corners of ``WINDOW`` in ``frame``.

    Each mark runs ``arm`` drawing units along the two sides of the window from
    its corner, in secondary ink over a paper halo, so that it stands out from
    the cell edges it lies on. Raise ``ValueError`` if a mark, halo included,
    would touch a cell of ``MAX_LEVEL`` or the hole, where it would cover the
    hatching or the staircase.
    """
    xa, ya, xb, yb = WINDOW
    # the arm, then half the width of the halo past its square end
    reach = (arm + (HEAVY + 3) / 2) / frame.unit
    boxes = []
    for u, sx in ((xa, 1), (xb, -1)):
        for v, sy in ((ya, 1), (yb, -1)):
            boxes.append((min(u, u + sx * reach), min(v, v + sy * reach), reach))
    for x0, y0, size in boxes:
        for level, i, j in leaves:
            h = 2.0**-level
            overlaps = i * h < x0 + size and x0 < (i + 1) * h and j * h < y0 + size
            if overlaps and y0 < (j + 1) * h and level == MAX_LEVEL:
                raise ValueError(f"a corner mark covers the level {level} cell {(level, i, j)}")
        if (
            hypot(
                max(abs(x0 + size / 2 - CX) - size / 2, 0),
                max(abs(y0 + size / 2 - CY) - size / 2, 0),
            )
            <= R
        ):
            raise ValueError("a corner mark reaches the hole")
    d = ""
    for u, sx in ((xa, 1), (xb, -1)):
        for v, sy in ((ya, -1), (yb, 1)):
            x, y = frame.x(u), frame.y(v)
            d += f"M{num(x + sx * arm)} {num(y)}H{num(x)}V{num(y + sy * arm)}"
    halo = path(d, stroke=PAPER, sw=HEAVY + 3, extra='stroke-linecap="square"')
    return halo + path(d, stroke=INK2, sw=HEAVY, extra='stroke-linecap="square"')


def _key(p, x, y, unit):
    """Return the key: one cell of each level at the scale of the channel, then the lines."""
    g = text(x, y, "one cell of each level, at the scale", size=11.5)
    g += text(x, y + 15, "of the channel", size=11.5)
    y += 32
    for level in range(MAX_LEVEL, MIN_LEVEL - 1, -1):
        side = unit * 2.0**-level
        fine = level == MAX_LEVEL
        g += rect(
            x,
            y,
            side,
            side,
            fill=f"url(#{p}-r)" if fine else "none",
            stroke=INK,
            sw=HAIR if fine else EMPTY,
        )
        ty = y + min(side, 14) / 2 + 4
        g += text(x + 92, ty, f"level {level}", size=12, fill=RED if fine else INK)
        g += text(x + 144, ty, f"side 1/{2**level}", size=11.5, fill=RED if fine else INK2)
        y += max(side, 10) + 12
    y += 10
    g += line(x, y, x + 34, y, stroke=INK, sw=HEAVY)
    g += text(x + 46, y + 4, "edge of the domain", size=11.5)
    y += 22
    g += line(x, y, x + 34, y, stroke=RED, sw=RULE)
    g += text(x + 46, y + 4, "exact circle, radius 0.2", size=11.5, fill=RED)
    return g, y


@figure
def disc_obstacle_mesh(p):
    """Draw the mesh of ``disc_obstacle.cpp`` after adaptation, and an enlarged window.

    Figure of howto/geometry.md. The cells come from ``data/geometry.json``; the
    cells of level 6 are hatched in red, the edge of the domain (the channel and
    the hole) is the heavy rule, and the exact circle is drawn over the staircase.
    Corner marks frame on the channel the window that the second view enlarges.
    """
    leaves = _mesh()
    _check_hole(leaves)
    ox, oy, width = 40, 12, 620
    channel = Frame((0, 0, 2, 1), ox, oy, width)
    g = draw_cells(
        p,
        leaves,
        channel,
        xticks=[(v, f"{v:g}") for v in (0, 0.5, 1, 1.5, 2)],
        yticks=[(v, f"{v:g}") for v in (0, 0.5, 1)],
    )
    g += _circle(channel)
    g += _corner_marks(leaves, channel)

    top = oy + channel.height + 62
    zoom = Frame(WINDOW, ox, top, 380)
    ticks = [(v, f"{v:g}") for v in (0.25, 0.375, 0.5, 0.625, 0.75)]
    # the corner 0.25 is labeled once, under the x axis: its two labels would touch
    g += draw_cells(p, leaves, zoom, xticks=ticks, yticks=ticks[1:])
    clip = f"{p}-zoom"
    g += (
        f'<clipPath id="{clip}"><rect x="{zoom.ox}" y="{zoom.oy}" width="{zoom.width}"'
        f' height="{zoom.height}"/></clipPath>'
    )
    g += _circle(zoom, clip)
    g += rect(zoom.ox, zoom.oy, zoom.width, zoom.height)
    g += text(
        zoom.ox,
        top - 12,
        '<tspan class="sm-fig-code">[0.25, 0.75) × [0.25, 0.75)</tspan>, the square framed by'
        " the corner marks above, enlarged",
        size=11.5,
        fill=INK,
    )
    for frame in (channel, zoom):
        _check_edge(leaves, frame.unit, RULE)

    key, bottom = _key(p, ox + zoom.width + 40, top + 16, channel.unit)
    g += key
    counts = Counter(leaf[0] for leaf in leaves)
    per_level = ", ".join(f"{counts[level]} of level {level}" for level in sorted(counts))
    return Drawing(
        DIAGRAM_WIDTH,
        max(top + zoom.height + 32, bottom + 12),
        f"The channel [0, 2) x [0, 1) meshed with {len(leaves)} cells: {per_level}. A disc of"
        " radius 0.2 centered at (0.5, 0.5) has no cells; its edge is a staircase of level 6"
        " cells, hatched in red, and a red circle marks the exact edge, passing through that"
        " staircase. Around the disc the band of level 6 cells gives way to cells of levels 5,"
        " 4 and 3, and the right of the channel is at level 2; no coarser cell reaches the"
        " circle. Four corner marks frame the square [0.25, 0.75) x [0.25, 0.75) around the disc;"
        " below, that square is enlarged: the"
        " circle runs through the steps of the staircase, across kept level 6 cells and through"
        " the hole. Beside it, a key shows one cell of each level from"
        " 6 to 2 at the scale of the channel, the heavy rule of the edge of the domain and the"
        " red line of the exact circle.",
        patterns(p),
        g,
    )
