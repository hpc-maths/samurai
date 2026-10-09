"""The figures of ``howto/mpi.md``: the subdomains of 4 ranks, before and after load balancing.

The two meshes are the files ``mpi_adapt_size_4.h5`` that the snippet
``howto/snippet/mpi/mpi_adapt.cpp`` saves on 4 processes, once without options and
once with ``--load-balancing-at 1``. Run both with ``--save-debug-fields``, each in
its own directory, then export them with ``docs/tools/export_cells.py`` into
``data/mpi_adapt_np4.json`` and ``data/mpi_adapt_np4_load_balancing.json``, each
with ``--source`` naming the run. The files keep the cells of each rank.

The figure checks what it draws against the page: the cells per rank are the
counts the program prints, the mesh is graded and covers the unit square, each
rank of the first run owns one band of rows of the start level, and the ranks of
the second run are the cut of the Hilbert curve that ``partition.partition``
computes on the same cells.
"""

import re
from fractions import Fraction
from pathlib import Path

from ..cells import check_disjoint, check_graded, load_cells
from ..draw import FIG, HAIR, INK, INK2, fig_label, line, rect, text
from ..partition import draw_partition, partition, rank_patterns
from ..registry import PLATE_WIDTH, Drawing, figure

DATA = Path(__file__).resolve().parent.parent / "data"

# the mesh of the snippet: max_level 7, which is also its start level
MAX_LEVEL = 7
N_RANKS = 4
# the outputs of the two runs, shown on the page
OUTPUTS = Path(__file__).resolve().parents[3] / "howto" / "snippet" / "mpi"


def _printed(name):
    """Read the cells per rank that the output ``name`` of the snippet prints."""
    counts = {}
    for row in (OUTPUTS / name).read_text(encoding="utf-8").splitlines():
        match = re.fullmatch(r"rank (\d+): (\d+) cells", row)
        if match:
            counts[int(match[1])] = int(match[2])
    if sorted(counts) != list(range(N_RANKS)):
        raise ValueError(f"{name} does not print the cells of {N_RANKS} ranks")
    return tuple(counts[r] for r in range(N_RANKS))


def _counts(ranks):
    """Count the cells of each rank in ``ranks``, which maps each cell to its rank."""
    return tuple(sum(1 for r in ranks.values() if r == k) for k in range(N_RANKS))


def _load(name, counts):
    """Read the rank of each cell of an exported run and check it against the page."""
    exported = load_cells(DATA / name)
    ranks = dict(zip(exported.leaves, exported.ranks))
    found = _counts(ranks)
    if found != counts:
        raise ValueError(f"{name} has {found} cells per rank, the page prints {counts}")
    leaves = list(ranks)
    check_disjoint(leaves)
    check_graded(leaves)
    if sum(Fraction(1, 4**level) for level, _, _ in leaves) != 1:
        raise ValueError(f"the cells of {name} do not cover the unit square")
    return ranks


def _runs():
    """Read and check the ranks of the cells without and with load balancing."""
    rows = _load("mpi_adapt_np4.json", _printed("mpi_adapt_np4_output.txt"))
    balanced = _load(
        "mpi_adapt_np4_load_balancing.json", _printed("mpi_adapt_load_balancing_output.txt")
    )
    if set(rows) != set(balanced):
        raise ValueError("the two runs do not have the same cells")
    # the mesh constructor gives rank r the rows [32 r, 32 r + 32) of level 7
    band = 2**MAX_LEVEL // N_RANKS
    for (level, i, j), r in rows.items():
        lo, hi = j << (MAX_LEVEL - level), (j + 1) << (MAX_LEVEL - level)
        if not (r * band <= lo and hi <= (r + 1) * band):
            raise ValueError(f"the cell {(level, i, j)} of rank {r} is outside its band of rows")
    if partition(list(balanced), MAX_LEVEL, N_RANKS) != balanced:
        raise ValueError("the balanced ranks are not the cut of the Hilbert curve")
    return rows, balanced


def _key(p, x, y, counts):
    """Draw the key under a mesh: the fill, the number and the cells of each rank."""
    g = ""
    for r, count in enumerate(counts):
        kx, ky = x + (r % 2) * 168, y + (r // 2) * 22
        g += rect(kx, ky, 15, 15, fill=f"url(#{p}-rank{r})", stroke=INK, sw=HAIR)
        g += text(kx + 23, ky + 12, f"rank {r}", size=11.5, fill=INK)
        g += text(kx + 72, ky + 12, f"{count} cells", size=11.5)
    return g


def _bands(ox, oy, size):
    """Draw the ticks of y = 0, 1/4, 1/2, 3/4 and 1 at the left of a mesh, and its name.

    The name *y* sits above the end of the axis, as ``plots.py`` places it.
    """
    g = ""
    for k, label in enumerate(("0", "1/4", "1/2", "3/4", "1")):
        y = oy + size - k * size / N_RANKS
        g += line(ox - 5, y, ox, y, stroke=INK2, sw=HAIR)
        g += text(ox - 8, y + 4, label, anchor="end", size=11)
    g += text(ox, oy - 14, "y", anchor="middle", size=14, fill=INK, cls="sm-fig-math")
    return g


def _stroke(cell):
    """Give the outline color of a cell: faint for the two finest levels.

    Their cells are 2.3 and 4.7 units wide; with ink outlines the zone around
    the peak turns dark on a phone and hides the tone of its rank.
    """
    return FIG if cell[0] >= MAX_LEVEL - 1 else INK2


@figure
def mpi_subdomains(p):
    """Draw the 4 subdomains of the snippet without and with load balancing.

    Figure of howto/mpi.md, from the two runs of ``mpi_adapt`` on 4 processes.
    Fig. 1: the bands of rows the mesh constructor gives each rank, kept through
    the adaptation. Fig. 2: the cut of the Hilbert order of the cells after
    ``--load-balancing-at 1``.
    """
    rows, balanced = _runs()
    size, oy, xa, xb = 300, 34, 42, 391
    ga, gb = (xa, oy, size), (xb, oy, size)
    g = _bands(xa, oy, size) + draw_partition(p, rows, MAX_LEVEL, ga, cell_stroke=_stroke)
    g += draw_partition(p, balanced, MAX_LEVEL, gb, cell_stroke=_stroke)
    ky = oy + size + 18
    before, after = _counts(rows), _counts(balanced)
    g += _key(p, xa, ky, before) + _key(p, xb, ky, after)
    height = ky + 82
    g += fig_label(xa + size / 2, ky + 66, 1, "without load balancing")
    # an option of the program: in the code face, not as an italic qualifier
    g += text(
        xb + size / 2,
        ky + 66,
        'Fig. 2. <tspan class="sm-fig-code" font-size="12">--load-balancing-at 1</tspan>',
        anchor="middle",
        size=14,
        fill=INK,
        cls="sm-fig-sc",
    )
    total = len(rows)
    levels = sorted({level for level, _, _ in rows})

    def listed(counts):
        """Write ``counts`` as "a, b, c and d"."""
        return ", ".join(map(str, counts[:-1])) + f" and {counts[-1]}"

    return Drawing(
        PLATE_WIDTH,
        height,
        f"Two copies of the adapted mesh of the program on 4 processes, {total} cells of the"
        f" unit square from level {levels[0]} to level {levels[-1]}, the finest cells around"
        " the peak at (0.3, 0.3). Each cell is filled with the tone and pattern of its rank,"
        " the cells of the two finest levels have a fainter outline, and heavy rules separate"
        " the ranks. Left, without load balancing: four horizontal"
        " bands of equal height, rank 0 for y from 0 to 1/4 up to rank 3 for y from 3/4 to 1,"
        f" with {listed(before)} cells. Right, with --load-balancing-at 1: four connected"
        f" subdomains of {listed(after)} cells, cut along the Hilbert order of the cells; rank 1"
        " holds a small region around the peak, rank 3 most of the square in coarse cells.",
        rank_patterns(p),
        g,
        panels=((0, 0, 357, height, ""), (349, 0, 362, height, "")),
    )
