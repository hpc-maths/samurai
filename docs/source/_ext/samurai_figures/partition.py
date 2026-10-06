"""The Hilbert and Morton orders of the leaves of a mesh, and their cut between ranks.

samurai shares a mesh between MPI processes with a space-filling curve
(``include/samurai/load_balancing/strategies/sfc.hpp``): every cell gets the key of
its place on the curve, the cells are sorted by key, and the curve is cut into one
segment of equal load per rank. This module computes the same keys and the same cuts
for the leaves ``(level, i, j)`` of a mesh from ``mesh.py``, so that a figure shows
the order and the ranks samurai would give, and draws them. The C++ files cited
below are under ``include/samurai/load_balancing/``.

Ranks are drawn with the four rank tones of the theme (``--sm-rank-0`` to
``--sm-rank-3`` in ``_static/css/samurai.css``), each with its own pattern, so that
color is never the only cue.
"""

from .draw import HAIR, HEAVY, INK, INK2, RED, RULE, circle, line, num, rect

MORTON, HILBERT = "morton", "hilbert"

RANK_TONES = tuple(f"var(--sm-rank-{r})" for r in range(4))


def normalized_box(leaves, max_level):
    """The shift and the extent of the box of the leaves, at ``max_level``.

    Return ``((xmin, ymin), (w, h))``: the smallest index of each direction and the
    number of ``max_level`` cells the box spans in it. Same as
    ``sfc_normalized_box`` (strategies/sfc.hpp, lines 41 to 69) for a mesh whose
    domain is the box of its cells.
    """
    lo, hi = [None, None], [None, None]
    for level, i, j in leaves:
        s = max_level - level
        for d, k in enumerate((i, j)):
            a, b = k << s, (k + 1) << s
            lo[d] = a if lo[d] is None else min(lo[d], a)
            hi[d] = b if hi[d] is None else max(hi[d], b)
    return (lo[0], lo[1]), (hi[0] - lo[0], hi[1] - lo[1])


def morton_key(x, y):
    """The Morton key of (x, y): the bits of x at the even places, of y at the odd ones.

    ``Morton::key_2d`` (sfc/morton.hpp, lines 34 to 38); the extent of the box does
    not change it (lines 51 to 55).
    """
    key = 0
    for b in range(32):
        key |= ((x >> b) & 1) << (2 * b) | ((y >> b) & 1) << (2 * b + 1)
    return key


def _sgn(v):
    return (v > 0) - (v < 0)


def _half(v):
    """``v / 2`` rounded toward zero, as C++ integer division rounds."""
    return -(-v // 2) if v < 0 else v // 2


# _in_region and _gilbert keep the arguments and the variables of their C++
# originals, so that the two can be compared line by line.
# pylint: disable=too-many-arguments,too-many-positional-arguments
# pylint: disable=too-many-locals,too-many-return-statements


def _in_region(px, py, x, y, ax, ay, bx, by):
    """``Hilbert::in_region`` (sfc/hilbert.hpp, lines 109 to 125)."""
    ex, ey = ax + bx, ay + by
    fx, fy = x + ex - _sgn(ex), y + ey - _sgn(ey)
    return min(x, fx) <= px <= max(x, fx) and min(y, fy) <= py <= max(y, fy)


def _gilbert(px, py, x, y, ax, ay, bx, by):
    """``Hilbert::gilbert_xy2d`` (sfc/hilbert.hpp, lines 130 to 194), line for line.

    The place of (px, py) on the generalized Hilbert curve of the region at (x, y)
    spanned by the major direction (ax, ay) and the minor direction (bx, by).
    """
    w, h = abs(ax + ay), abs(bx + by)
    dax, day, dbx, dby = _sgn(ax), _sgn(ay), _sgn(bx), _sgn(by)
    if h == 1:
        return dax * (px - x) + day * (py - y)
    if w == 1:
        return dbx * (px - x) + dby * (py - y)
    ax2, ay2, bx2, by2 = _half(ax), _half(ay), _half(bx), _half(by)
    w2, h2 = abs(ax2 + ay2), abs(bx2 + by2)
    if 2 * w > 3 * h:
        if w2 & 1 and w > 2:
            ax2, ay2 = ax2 + dax, ay2 + day
        if _in_region(px, py, x, y, ax2, ay2, bx, by):
            return _gilbert(px, py, x, y, ax2, ay2, bx, by)
        return abs((ax2 + ay2) * (bx + by)) + _gilbert(
            px, py, x + ax2, y + ay2, ax - ax2, ay - ay2, bx, by
        )
    if h2 & 1 and h > 2:
        bx2, by2 = bx2 + dbx, by2 + dby
    if _in_region(px, py, x, y, bx2, by2, ax2, ay2):
        return _gilbert(px, py, x, y, bx2, by2, ax2, ay2)
    base = abs((bx2 + by2) * (ax2 + ay2))
    if _in_region(px, py, x + bx2, y + by2, ax, ay, bx - bx2, by - by2):
        return base + _gilbert(px, py, x + bx2, y + by2, ax, ay, bx - bx2, by - by2)
    base += abs((ax + ay) * ((bx - bx2) + (by - by2)))
    return base + _gilbert(
        px,
        py,
        x + (ax - dax) + (bx2 - dbx),
        y + (ay - day) + (by2 - dby),
        -bx2,
        -by2,
        -(ax - ax2),
        -(ay - ay2),
    )


# pylint: enable=too-many-arguments,too-many-positional-arguments
# pylint: enable=too-many-locals,too-many-return-statements


def hilbert_key(x, y, w, h):
    """The Hilbert key of (x, y) in a box of ``w`` by ``h`` cells.

    ``Hilbert::key_2d(p, n)`` (sfc/hilbert.hpp, lines 68 to 82): the generalized
    Hilbert curve of the box, laid along its longer side. This is the key the SFC
    strategy uses, through ``sfc_cell_key`` (strategies/sfc.hpp, line 89).
    """
    if not (w >= 1 and h >= 1 and 0 <= x < w and 0 <= y < h):
        raise ValueError(f"({x}, {y}) is outside the {w} x {h} box of the curve")
    if w >= h:
        return _gilbert(x, y, 0, 0, w, 0, 0, h)
    return _gilbert(x, y, 0, 0, 0, h, w, 0)


def cell_key(curve, cell, max_level, shift, extent):
    """The key of the leaf ``cell = (level, i, j)`` on ``curve``.

    ``sfc_cell_key`` (strategies/sfc.hpp, lines 76 to 90): the indices of the cell
    are brought to ``max_level`` and shifted by the smallest index of the box,
    ``shift``, from ``normalized_box``.
    """
    level, i, j = cell
    x = (i << (max_level - level)) - shift[0]
    y = (j << (max_level - level)) - shift[1]
    if curve == MORTON:
        return morton_key(x, y)
    if curve == HILBERT:
        return hilbert_key(x, y, *extent)
    raise ValueError(f"unknown curve {curve!r}, expected {MORTON!r} or {HILBERT!r}")


def sfc_order(leaves, max_level, curve=HILBERT):
    """The leaves sorted by their key on ``curve``, the order the SFC strategy uses."""
    shift, extent = normalized_box(leaves, max_level)
    return sorted(leaves, key=lambda c: cell_key(curve, c, max_level, shift, extent))


def partition(leaves, max_level, n_ranks, curve=HILBERT, weight=None):
    """The rank of each leaf, as a dict, once the curve is cut into equal loads.

    The leaves are sorted on ``curve`` and cut into ``n_ranks`` segments of equal
    load, ``weight(cell)`` per leaf (1 when ``weight`` is None), as
    ``SFC::partition`` does with one atom per cell: a cut key is the smallest key
    whose cumulative load reaches a multiple of ``W_total / n_ranks``
    (``sfc_equal_weight_cuts_gather``, strategies/sfc.hpp, lines 187 to 235), and
    a leaf goes to the rank equal to the number of cut keys at or below its key
    (lines 493 to 496). Loads are floats, compared as samurai compares them.
    """
    shift, extent = normalized_box(leaves, max_level)
    keyed = sorted(
        (cell_key(curve, c, max_level, shift, extent), 1.0 if weight is None else weight(c), c)
        for c in leaves
    )
    chunk = sum(w for _, w, _ in keyed) / n_ranks
    cuts, pre = [], 0.0
    for key, w, _ in keyed:
        pre += w
        while len(cuts) < n_ranks - 1 and pre >= (len(cuts) + 1) * chunk:
            cuts.append(key + 1)
    cuts += [keyed[-1][0] + 1] * (n_ranks - 1 - len(cuts))
    return {c: sum(1 for s in cuts if s <= key) for key, _, c in keyed}


def rank_patterns(p):
    """The fills of the four ranks, ``{p}-rank0`` to ``{p}-rank3``.

    Each fill is the tone of the rank crossed by a pattern of its own in secondary
    ink, so that the ranks stay apart without color: rank 0 is hatched at 45
    degrees, rank 1 dotted, rank 2 hatched at 135 degrees, rank 3 cross-hatched.
    """
    gap = 4.5

    def hatch(r, angle, crossed=False):
        lines = f'<line x1="0" y1="0" x2="0" y2="{gap}" stroke="{INK2}" stroke-width="{HAIR}"/>'
        if crossed:
            lines += (
                f'<line x1="0" y1="0" x2="{gap}" y2="0" stroke="{INK2}" stroke-width="{HAIR}"/>'
            )
        return (
            f'<pattern id="{p}-rank{r}" patternUnits="userSpaceOnUse" width="{gap}"'
            f' height="{gap}" patternTransform="rotate({angle})"><rect width="{gap}"'
            f' height="{gap}" fill="{RANK_TONES[r]}"/>{lines}</pattern>'
        )

    dots = (
        f'<pattern id="{p}-rank1" patternUnits="userSpaceOnUse" width="{gap}" height="{gap}">'
        f'<rect width="{gap}" height="{gap}" fill="{RANK_TONES[1]}"/>'
        f"{circle(gap / 2, gap / 2, 0.7, fill=INK2)}</pattern>"
    )
    return hatch(0, 45) + dots + hatch(2, -45) + hatch(3, 45, crossed=True)


def _cell_box(square, cell):
    """The corner and the side of a leaf in ``square = (ox, oy, size)``, y up."""
    ox, oy, size = square
    level, i, j = cell
    h = size * 2.0**-level
    return ox + i * h, oy + size - (j + 1) * h, h


def draw_partition(p, ranks, max_level, square):
    """Draw the leaves in the fills of their ranks, the subdomains outlined.

    ``ranks`` maps each leaf of a mesh of the unit square to its rank, as
    ``partition`` returns it; the mesh is drawn in ``square = (ox, oy, size)``,
    y up. The fills are ``rank_patterns(p)``. The edges between two ranks are
    heavy rules. The theme has four rank tones, so ranks go from 0 to 3.
    """
    if not set(ranks.values()) <= set(range(len(RANK_TONES))):
        raise ValueError(f"a figure draws at most {len(RANK_TONES)} ranks")
    s = ""
    for cell, r in ranks.items():
        x, y, h = _cell_box(square, cell)
        s += rect(x, y, h, h, fill=f"url(#{p}-rank{r})", stroke=INK, sw=HAIR)
    for edge in _rank_edges(ranks, max_level):
        s += line(*_grid_points(square, max_level, edge), sw=HEAVY)
    return s + rect(*square, square[2], stroke=INK, sw=HEAVY)


def _grid_points(square, max_level, edge):
    """The ends of ``edge``, in max_level cells, in ``square = (ox, oy, size)``, y up."""
    ox, oy, size = square
    u = size / 2**max_level
    x1, y1, x2, y2 = edge
    return ox + x1 * u, oy + size - y1 * u, ox + x2 * u, oy + size - y2 * u


def _rank_edges(ranks, max_level):
    """The segments of the max_level grid between two ranks, in max_level cells.

    Each segment ``(x1, y1, x2, y2)`` is a run of edges along one grid line.
    """
    owner = _rank_grid(ranks, max_level)
    n = 2**max_level
    segments = []
    for a in range(1, n):
        x_runs = _runs([owner[a - 1, t] != owner[a, t] for t in range(n)])
        y_runs = _runs([owner[t, a - 1] != owner[t, a] for t in range(n)])
        segments += [(a, t0, a, t1) for t0, t1 in x_runs]
        segments += [(t0, a, t1, a) for t0, t1 in y_runs]
    return segments


def _rank_grid(ranks, max_level):
    """The rank of the leaf over each cell of ``max_level``, as a dict."""
    owner = {}
    for (level, i, j), r in ranks.items():
        k = 2 ** (max_level - level)
        for a in range(k):
            for b in range(k):
                owner[i * k + a, j * k + b] = r
    return owner


def _runs(flags):
    """The half-open runs [t0, t1) where ``flags`` is true."""
    runs, start = [], None
    for t, flag in enumerate([*flags, False]):
        if flag and start is None:
            start = t
        elif not flag and start is not None:
            runs.append((start, t))
            start = None
    return runs


def draw_curve(order, square, marker="", stroke=RED):
    """Draw the curve through the centers of the leaves, in ``order``.

    The mesh covers the unit square, drawn in ``square = (ox, oy, size)``, y up. A
    filled dot marks the first leaf; ``marker`` is the id of an arrow head for the
    last one (see ``draw.arrow``).
    """
    centers = []
    for cell in order:
        x, y, h = _cell_box(square, cell)
        centers.append((x + h / 2, y + h / 2))
    points = " ".join(f"{num(x)},{num(y)}" for x, y in centers)
    end = f' marker-end="url(#{marker})"' if marker else ""
    return (
        f'<polyline points="{points}" fill="none" stroke="{stroke}" stroke-width="{RULE}"'
        f' stroke-linejoin="round"{end}/>' + circle(*centers[0], 2.4, fill=stroke)
    )
