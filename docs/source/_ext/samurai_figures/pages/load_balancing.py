"""The figures of ``reference/load_balancing.md``: a curve and its cut, a diffusion call, a graph.

The plate and the diffusion diagram share one graded mesh of the unit square,
``plate_mesh()``, levels 2 to 4 around a disc. The keys, the order and the cut of
the plate come from ``partition.py``. The diffusion call and the cell graph are
rewritten here from ``strategies/diffusion.hpp`` and ``graph.hpp`` (under
``include/samurai/load_balancing/``). A samurai program checked them on the same
meshes: ``SFC<Hilbert>`` on 4 processes gave the ranks and the keys of
``partition.py``, ``Diffusion`` on 2 processes moved the cells
``diffusion_call`` moves and reported the same unmet flux, and ``build_cell_graph``
on 2 processes gave the edges of ``cell_graph``.
"""

from fractions import Fraction

from ..draw import (
    EMPTY,
    HAIR,
    HEAVY,
    INK,
    INK2,
    INK3,
    PAPER,
    RED,
    RULE,
    callout,
    circle,
    fig_label,
    line,
    num,
    patterns,
    rect,
    text,
)
from ..mesh import build_mesh, check_graded, circle_refine
from ..partition import (
    HILBERT,
    cell_key,
    draw_curve,
    draw_partition,
    normalized_box,
    partition,
    rank_patterns,
    sfc_order,
)
from ..registry import DIAGRAM_WIDTH, PLATE_WIDTH, Drawing, figure

MIN_LEVEL, MAX_LEVEL = 2, 4
DISC = (0.35, 0.4, 0.2)
N_RANKS = 4

# The diffusion call starts from two subdomains: rank 0 holds the cells left of
# x = 3/4, rank 1 the column right of it.
DIFFUSION_SPLIT = 0.75

# The cell graph is drawn on a smaller mesh, cut into two ranks by the Hilbert curve.
GRAPH_LEVELS = (2, 3)
GRAPH_DISC = (0.3, 0.3, 0.15)

# The defaults of ``DiffusionOptions`` (strategies/diffusion.hpp, lines 33 to 53).
FLUX_THRESHOLD, DIFFUSION_ITERATIONS, MIN_RETAINED = 0.01, 50, 0.1

FACES = ((1, 0), (-1, 0), (0, 1), (0, -1))


def plate_mesh():
    """Return the graded mesh of the plate and of the diffusion diagram."""
    leaves = build_mesh(MIN_LEVEL, MAX_LEVEL, circle_refine(*DISC))
    check_graded(leaves, MAX_LEVEL)
    return leaves


def graph_mesh():
    """Return the graded mesh of the cell graph diagram."""
    leaves = build_mesh(*GRAPH_LEVELS, circle_refine(*GRAPH_DISC))
    check_graded(leaves, GRAPH_LEVELS[1])
    return leaves


# -- the diffusion call (strategies/diffusion.hpp) ---------------------------------


def cybenko_fluxes(loads, neighbours):
    """Return the flux of every edge ``(i, j)`` of the process graph, as a dict.

    ``detail::diffusion_fluxes`` (lines 90 to 143), run by every rank in lockstep:
    at each iteration rank i adds ``(L_j - L_i) / (max(d_i, d_j) + 1)`` to the flux
    towards j and to its load, until every term is at most 1e-3 times the average
    load; fluxes below ``flux_threshold`` times the average load are then dropped.
    A negative flux is load that i gives to j.
    """
    avg = sum(loads) / len(loads)
    scale = avg if avg > 0 else 1.0
    loads = list(loads)
    flux = {(i, j): 0.0 for i, ns in neighbours.items() for j in ns}
    for _ in range(DIFFUSION_ITERATIONS):
        converged, new = True, list(loads)
        for i, ns in neighbours.items():
            for j in ns:
                t = (loads[j] - loads[i]) / (max(len(ns), len(neighbours[j])) + 1)
                flux[i, j] += t
                new[i] += t
                converged = converged and abs(t) <= 1e-3 * scale
        loads = new
        if converged:
            break
    return {e: (0.0 if abs(f) < FLUX_THRESHOLD * scale else f) for e, f in flux.items()}


def _coarse(cell, ref):
    """Return the cell of level ``ref`` that holds ``cell``."""
    level, i, j = cell
    return i >> (level - ref), j >> (level - ref)


def give_to_neighbour(flags, rank, neighbour_cells, neighbour, ref, budget):
    """Peel the layers of ``rank`` towards ``neighbour``, for a load ``budget``.

    ``Diffusion::give_to_neighbour`` (lines 383 to 532), with a uniform weight. The
    front starts on the cells of the neighbour, brought to level ``ref``; each layer
    is the set of cells of level ``ref`` still owned by ``rank`` that share a face
    with the front, walked row by row from the bottom, left to right. A coarse cell
    goes whole, with every finer cell it holds, and only when its load fits in what
    is left of the budget. ``flags`` is updated in place.

    Return the load left to give and the layers, each a pair: the coarse cells of
    the layer, and the ones the peel visited, as ``(coarse cell, load, ceded)``; the
    last one visited is the coarse cell that stopped the peel, when one did.
    """
    owned = {_coarse(c, ref) for c, f in flags.items() if f == rank}
    front = {_coarse(c, ref) for c in neighbour_cells}
    layers, remaining = [], budget
    while remaining > 0:
        ring = sorted(
            (c for c in owned if any((c[0] - dx, c[1] - dy) in front for dx, dy in FACES)),
            key=lambda c: (c[1], c[0]),
        )
        if not ring:
            break
        layer, ceded, stop = [], set(), False
        for cc in ring:
            inside = [c for c, f in flags.items() if f == rank and _coarse(c, ref) == cc]
            load = float(len(inside))
            if load > remaining and load > 0:
                layer.append((cc, load, False))
                stop = True
                break
            for c in inside:
                flags[c] = neighbour
            remaining -= load
            layer.append((cc, load, True))
            ceded.add(cc)
            if remaining <= 0:
                stop = True
                break
        layers.append((ring, layer))
        front |= ceded
        owned -= ceded
        if stop:
            break
    return remaining, layers


def _connected(cells, ref):
    """Return True when the cells of level ``ref`` holding ``cells`` are face-connected."""
    blocks = {_coarse(c, ref) for c in cells}
    if not blocks:
        return True
    seen, stack = set(), [next(iter(blocks))]
    while stack:
        b = stack.pop()
        if b in seen:
            continue
        seen.add(b)
        stack += [(b[0] + dx, b[1] + dy) for dx, dy in FACES if (b[0] + dx, b[1] + dy) in blocks]
    return seen == blocks


def diffusion_call(start, neighbours, ref):
    """Return the ranks after one call of ``Diffusion::partition`` (lines 249 to 353).

    ``start`` maps each cell to its rank, ``neighbours`` each rank to its MPI
    neighbours. The weight is uniform. Return the new ranks, the fluxes, and per
    giving rank its layers and its unmet flux. The connectivity repair of the
    strategy is not rewritten: the function raises when a kept region splits, the
    case where the repair would act.
    """
    n = len(neighbours)
    loads = [sum(1 for f in start.values() if f == r) for r in range(n)]
    flux = cybenko_fluxes(loads, neighbours)
    flags, peels = dict(start), {}
    for rank in range(n):
        out = {j: -flux[rank, j] for j in neighbours[rank] if flux[rank, j] < 0}
        total, cap = sum(out.values()), (1 - MIN_RETAINED) * loads[rank]
        if total > cap:
            out = {j: f * cap / total for j, f in out.items()}
        for j, budget in sorted(out.items(), key=lambda item: (-item[1], item[0])):
            neighbour_cells = [c for c, f in start.items() if f == j]
            unmet, layers = give_to_neighbour(flags, rank, neighbour_cells, j, ref, budget)
            peels[rank, j] = (layers, max(unmet, 0.0))
        if not _connected([c for c, f in flags.items() if f == rank], ref):
            raise ValueError(f"the cells rank {rank} keeps split: the repair would act")
    return flags, flux, peels


# -- the cell graph (graph.hpp) ------------------------------------------------------


def face_adjacent(a, b, max_level):
    """Return True when the cells ``a`` and ``b`` share a face, a corner is not enough."""
    touching = 0
    for d in (1, 2):
        a0, a1 = a[d] << (max_level - a[0]), (a[d] + 1) << (max_level - a[0])
        b0, b1 = b[d] << (max_level - b[0]), (b[d] + 1) << (max_level - b[0])
        if a0 < b1 and b0 < a1:
            continue
        if a1 != b0 and b1 != a0:
            return False
        touching += 1
    return touching == 1


def cell_graph(leaves, max_level):
    """Return the edges of the cell graph of ``build_cell_graph`` (lines 128 to 297).

    One edge of weight 1 per interface between two face-adjacent cells, at one level
    or across a level jump, whatever the ranks of the two cells: a coarse cell gets
    one edge per finer cell along its face, and no edge to a cell it touches by a
    corner only.
    """
    return [
        (a, b)
        for k, a in enumerate(leaves)
        for b in leaves[k + 1 :]
        if face_adjacent(a, b, max_level)
    ]


# -- drawing helpers -------------------------------------------------------------------


def _box(square, cell):
    """Return the corner and the side of ``cell`` in ``square = (ox, oy, size)``, y up."""
    ox, oy, size = square
    level, i, j = cell
    h = size * 2.0**-level
    return ox + i * h, oy + size - (j + 1) * h, h


def _center(square, cell):
    x, y, h = _box(square, cell)
    return x + h / 2, y + h / 2


def _segment_hits_box(p, q, box):
    """Return True when the segment pq crosses the box ``(x0, y0, x1, y1)``."""
    x0, y0, x1, y1 = box
    t0, t1 = 0.0, 1.0
    for d, lo, hi in ((0, x0, x1), (1, y0, y1)):
        delta = q[d] - p[d]
        if delta == 0:
            if not lo <= p[d] <= hi:
                return False
            continue
        a, b = (lo - p[d]) / delta, (hi - p[d]) / delta
        t0, t1 = max(t0, min(a, b)), min(t1, max(a, b))
        if t0 > t1:
            return False
    return True


def _clear(points, box):
    """Raise ``ValueError`` when the polyline ``points`` crosses ``box``."""
    for p, q in zip(points, points[1:]):
        if _segment_hits_box(p, q, box):
            raise ValueError(f"a label at {box} crosses the curve")


def _coarse_grid(square, ref, sw=RULE, stroke=INK):
    """Draw the lines of the cells of level ``ref`` inside ``square``."""
    ox, oy, size = square
    s = ""
    for k in range(1, 2**ref):
        v = k * size / 2**ref
        s += line(ox + v, oy, ox + v, oy + size, stroke=stroke, sw=sw)
        s += line(ox, oy + v, ox + size, oy + v, stroke=stroke, sw=sw)
    return s


def _swatch(p, x, y, rank, label):
    """Draw the fill of ``rank`` as a small square, followed by ``label``."""
    return rect(x, y - 9, 11, 11, fill=f"url(#{p}-rank{rank})", stroke=INK, sw=HAIR) + text(
        x + 16, y, label, size=11.5
    )


def _tag(x, y, label, color=INK):
    """Write the mono ``label`` centered at (x, y) on a ground of paper."""
    w = 6.0 * len(label) + 6
    return rect(x - w / 2, y - 8, w, 16, fill=PAPER, stroke=color, sw=HAIR) + text(
        x, y + 3.5, label, anchor="middle", size=10, fill=color, cls="sm-fig-code"
    )


# -- the plate: the Hilbert curve and its cut ---------------------------------------------


def _key_labels(square, leaves, order, keys):
    """Mark the lower-left cell of ``MAX_LEVEL`` of each cell of ``MIN_LEVEL`` and write its key.

    The key of a cell is the key of that corner cell. Each label is placed in the
    first free spot around the corner cell and checked against the curve.
    """
    points = [_center(square, c) for c in order]
    sub = square[2] / 2**MAX_LEVEL
    s = ""
    for cell in leaves:
        if cell[0] != MIN_LEVEL:
            continue
        x, y, h = _box(square, cell)
        cx, cy = x, y + h - sub
        s += rect(cx, cy, sub, sub, stroke=INK, sw=HAIR, dash="1.5 1.5")
        label = str(keys[cell])
        w = 6.0 * len(label) + 2
        spots = [(cx + 2, cy - 4), (cx + sub + 3, cy + sub - 4)]
        for tx, ty in spots:
            box = (tx - 1, ty - 9, tx + w, ty + 2)
            try:
                _clear(points, box)
            except ValueError:
                continue
            s += text(tx, ty, label, size=10, fill=INK, cls="sm-fig-code", knock=True)
            break
        else:
            raise ValueError(f"no free spot for the key of {cell}")
    return s


def cut_keys(order, ranks, keys):
    """Return the cut keys between the ranks of ``order``, checked against ``ranks``.

    The cut key r is the smallest key below which the cumulative load reaches
    r times the average load per rank: one more than the key of the last cell of
    rank r - 1, as ``partition`` computes it. A cell goes to the rank equal to the
    number of cut keys at or below its key.
    """
    cuts = [keys[a] + 1 for a, b in zip(order, order[1:]) if ranks[a] != ranks[b]]
    for cell in order:
        if ranks[cell] != sum(1 for c in cuts if c <= keys[cell]):
            raise ValueError(f"the cut keys {cuts} do not give the rank of {cell}")
    return cuts


def _cut_strip(p, x0, width, y, order, ranks, keys):
    """Draw the cells in curve order, one slot each, with the cut keys between the ranks."""
    n = len(order)
    slot = width / n
    s = ""
    for k, cell in enumerate(order):
        s += rect(
            x0 + k * slot,
            y,
            slot,
            24,
            fill=f"url(#{p}-rank{ranks[cell]})",
            stroke=INK,
            sw=HAIR,
        )
    s += rect(x0, y, width, 24, stroke=INK, sw=HEAVY)
    cuts = cut_keys(order, ranks, keys)
    starts = [0] + [sum(1 for c in order if keys[c] < cut) for cut in cuts] + [n]
    chunk = n / N_RANKS
    for r, (a, b) in enumerate(zip(starts, starts[1:])):
        mid = x0 + (a + b) / 2 * slot
        s += text(mid, y + 44, f"rank {r}", anchor="middle", size=11.5, fill=INK)
        s += text(mid, y + 59, f"{b - a} cells", anchor="middle", size=11)
        if a:
            xa = x0 + a * slot
            s += line(xa, y - 14, xa, y + 30, stroke=INK, sw=HEAVY)
            s += text(xa, y - 19, f"cut key {cuts[r - 1]}", anchor="middle", size=11, fill=INK)
            s += text(
                xa, y + 76, f"{a} \u2265 {num(r * chunk)}", anchor="middle", size=10.5, fill=INK3
            )
    s += text(x0, y - 19, f"key {keys[order[0]]}", anchor="start", size=11)
    s += text(x0 + width, y - 19, f"key {keys[order[-1]]}", anchor="end", size=11)
    return s


@figure
def hilbert_cut(p):
    """reference/load_balancing.md: the Hilbert curve through a graded mesh, then its cut."""
    leaves = plate_mesh()
    order = sfc_order(leaves, MAX_LEVEL, HILBERT)
    shift, extent = normalized_box(leaves, MAX_LEVEL)
    keys = {c: cell_key(HILBERT, c, MAX_LEVEL, shift, extent) for c in leaves}
    ranks = partition(leaves, MAX_LEVEL, N_RANKS, HILBERT)
    size, top = 300, 14
    left = (30, top, size)
    right = (PLATE_WIDTH - 30 - size, top, size)

    g = ""
    for cell in leaves:
        x, y, h = _box(left, cell)
        g += rect(x, y, h, h, stroke=INK, sw=HAIR if cell[0] == MAX_LEVEL else EMPTY)
    g += rect(*left, size, stroke=INK, sw=HEAVY)
    g += draw_curve(order, left, marker=f"{p}-ar")
    g += _key_labels(left, leaves, order, keys)
    base = top + size + 30
    g += fig_label(left[0] + size / 2, base, 1)

    g += draw_partition(p, ranks, MAX_LEVEL, right)
    g += fig_label(right[0] + size / 2, base, 2)

    strip_y = base + 56
    g += _cut_strip(p, 30, PLATE_WIDTH - 60, strip_y, order, ranks, keys)
    end = strip_y + 108
    g += fig_label(PLATE_WIDTH / 2, end, 3)

    counts = [sum(1 for r in ranks.values() if r == k) for k in range(N_RANKS)]
    return Drawing(
        PLATE_WIDTH,
        end + 12,
        f"Left, a graded mesh of the unit square with {len(leaves)} cells of levels"
        f" {MIN_LEVEL} to {MAX_LEVEL}, and the Hilbert curve that visits every cell in"
        f" the order of its key; each cell of level {MIN_LEVEL} shows the key of its"
        f" lower-left cell of level {MAX_LEVEL}. Right, the same mesh cut into"
        f" {N_RANKS} ranks of "
        + ", ".join(str(c) for c in counts)
        + " cells, one fill per rank. Below, the cells in curve order, one slot each,"
        " with the cut keys between the ranks.",
        patterns(p) + rank_patterns(p),
        g,
        panels=(
            (0, 0, PLATE_WIDTH / 2, base + 14, ""),
            (PLATE_WIDTH / 2, 0, PLATE_WIDTH / 2, base + 14, ""),
            (0, strip_y - 36, PLATE_WIDTH, end + 12 - strip_y + 36, "wide"),
        ),
    )


# -- the diffusion diagram -----------------------------------------------------------------


@figure
def diffusion_layers(p):
    """reference/load_balancing.md: what one call of Diffusion moves between two ranks."""
    leaves = plate_mesh()
    start = {c: (0 if (c[1] + 0.5) / 2 ** c[0] < DIFFUSION_SPLIT else 1) for c in leaves}
    after, flux, peels = diffusion_call(start, {0: [1], 1: [0]}, MIN_LEVEL)
    layers, unmet = peels[0, 1]
    before_loads = [sum(1 for f in start.values() if f == r) for r in (0, 1)]
    after_loads = [sum(1 for f in after.values() if f == r) for r in (0, 1)]
    size, top = 250, 34
    left = (40, top, size)
    right = (DIAGRAM_WIDTH - 40 - size, top, size)

    g = text(left[0], top - 12, "Before the call", size=12, fill=INK)
    g += text(right[0], top - 12, "After one call", size=12, fill=INK)
    for square, flags in ((left, start), (right, after)):
        g += draw_partition(p, flags, MAX_LEVEL, square)
        g += _coarse_grid(square, MIN_LEVEL)
    # the arrow of the call, with the flux it acts on
    ay = top + size / 2
    g += line(
        left[0] + size + 14, ay, right[0] - 14, ay, stroke=RED, extra=f'marker-end="url(#{p}-ar)"'
    )
    g += text(DIAGRAM_WIDTH / 2, ay - 9, "one call", anchor="middle", size=11, fill=INK)
    g += text(DIAGRAM_WIDTH / 2, ay + 19, f"flux {num(-flux[0, 1])}", anchor="middle", size=11)

    # the ceded coarse cells and the one that stopped the peel
    unit = size / 2**MIN_LEVEL
    for _, layer in layers:
        for (i, j), load, ceded in layer:
            x, y = right[0] + i * unit, top + size - (j + 1) * unit
            if ceded:
                g += _tag(x + unit / 2, y + unit / 2, f"+{num(load)}")
            else:
                g += rect(x + 3, y + 3, unit - 6, unit - 6, stroke=RED, sw=RULE, dash="3 2")
                g += _tag(x + unit / 2, y + unit / 2, f"{num(load)} > {num(unmet)}", RED)
    # each layer under the column of coarse cells it is made of
    by = top + size + 12
    for n, (ring, _) in enumerate(layers, 1):
        cols = {i for i, _ in ring}
        if len(cols) != 1 or len(ring) != 2**MIN_LEVEL:
            raise ValueError("each layer of this diagram is expected to be a whole column")
        col = cols.pop()
        xa, xb = right[0] + col * unit + 4, right[0] + (col + 1) * unit - 4
        g += line(xa, by, xb, by, stroke=INK2, sw=RULE)
        g += line(xa, by - 4, xa, by, stroke=INK2) + line(xb, by - 4, xb, by, stroke=INK2)
        g += text((xa + xb) / 2, by + 15, f"layer {n}", anchor="middle", size=11)

    ty = top + size + 50
    for square, loads in ((left, before_loads), (right, after_loads)):
        g += _swatch(p, square[0], ty, 0, f"rank 0: {loads[0]} cells")
        g += _swatch(p, square[0] + 120, ty, 1, f"rank 1: {loads[1]} cells")
    moved = [load for _, layer in layers for _, load, ceded in layer if ceded]
    return Drawing(
        DIAGRAM_WIDTH,
        ty + 10,
        f"Two copies of a graded mesh with {len(leaves)} cells. Before the call, rank 0"
        f" holds the {before_loads[0]} cells left of x = 3/4 and rank 1 the"
        f" {before_loads[1]} cells right of it; the flux from rank 0 to rank 1 is"
        f" {num(-flux[0, 1])}. After one call, rank 0 has given the coarse cells of"
        f" level {MIN_LEVEL} next to rank 1, with "
        + ", ".join(num(m) for m in moved)
        + f" cells, bottom to top: the whole first layer and the bottom of the second."
        f" The next coarse cell holds {num(layers[-1][1][-1][1])} cells, more than the"
        f" {num(unmet)} left to give, and stops the peel. Rank 0 keeps"
        f" {after_loads[0]} cells, rank 1 holds {after_loads[1]}.",
        patterns(p) + rank_patterns(p),
        g,
    )


# -- the cell graph diagram -------------------------------------------------------------------


def _unit_center(cell):
    """Return the center of ``cell`` in the unit square."""
    level, i, j = cell
    return (i + 0.5) / 2**level, (j + 0.5) / 2**level


def colored_edges(edges, ranks):
    """Return each edge as ``(a, b, color)``: red across the two ranks, ink inside one."""
    return [(a, b, RED if ranks[a] != ranks[b] else INK) for a, b in edges]


def clip_segment(p, q, window):
    """Return the part of the segment pq inside ``window = (x0, y0, x1, y1)``, or None."""
    t0, t1 = 0.0, 1.0
    for d, lo, hi in ((0, window[0], window[2]), (1, window[1], window[3])):
        delta = q[d] - p[d]
        if delta == 0:
            if not lo <= p[d] <= hi:
                return None
            continue
        a, b = (lo - p[d]) / delta, (hi - p[d]) / delta
        t0, t1 = max(t0, min(a, b)), min(t1, max(a, b))
    if t0 > t1:
        return None
    return tuple(p[d] + t * (q[d] - p[d]) for t in (t0, t1) for d in (0, 1))


def inset_edges(marks, window):
    """Return the edges of ``marks`` clipped to ``window``, keeping their colors.

    Each entry is ``(a, b, color, (u0, v0, u1, v1))`` in the unit square. Raise
    ``ValueError`` unless the inset holds exactly the edges of the whole graph whose
    segment meets the window, each with the color it has in the whole graph.
    """
    clipped = []
    for a, b, color in marks:
        seg = clip_segment(_unit_center(a), _unit_center(b), window)
        if seg is not None:
            clipped.append((a, b, color, seg))
    meets = {
        (a, b, color)
        for a, b, color in marks
        if _segment_hits_box(_unit_center(a), _unit_center(b), window)
    }
    if {(a, b, color) for a, b, color, _ in clipped} != meets:
        raise ValueError("the inset does not show the edges of the whole graph in its window")
    return clipped


def _graph(square, leaves, marks):
    """Draw the colored edges ``marks`` and the vertices of the cell graph in ``square``."""
    s = ""
    for a, b, color in marks:
        s += line(*_center(square, a), *_center(square, b), stroke=color)
    for cell in leaves:
        s += circle(*_center(square, cell), 2.6, fill=INK)
    return s


# The window of the mesh the diagram enlarges, (x0, y0, x1, y1) in the unit square,
# and the two cells its callouts point at: a coarse cell over two finer ones, and a
# coarse cell that touches a finer one by a corner only.
GRAPH_WINDOW = (0, 0.25, 0.75, 0.75)
JUMP_CELL, CORNER = (2, 0, 2), ((2, 2, 2), (3, 3, 3))


def _frac(v):
    """Write the coordinate ``v`` as a fraction, as the axes of the meshes do."""
    return str(Fraction(v).limit_denominator(64))


def _point(square, u, v):
    """Return the drawing point of (u, v) of the unit square in ``square``, y up."""
    ox, oy, size = square
    return ox + u * size, oy + (1 - v) * size


@figure
def cell_graph_edges(p):
    """reference/load_balancing.md: the cell graph edges at a level jump and across ranks."""
    leaves = graph_mesh()
    hi = GRAPH_LEVELS[1]
    ranks = partition(leaves, hi, 2, HILBERT)
    edges = cell_graph(leaves, hi)
    marks = colored_edges(edges, ranks)
    size, top = 280, 14
    whole = (44, top, size)
    g = draw_partition(p, ranks, hi, whole) + _graph(whole, leaves, marks)
    for k in range(5):
        tx, ty = _point(whole, k / 4, k / 4)
        label = _frac(k / 4)
        g += line(tx, top + size, tx, top + size + 5) + text(
            tx, top + size + 17, label, anchor="middle", size=10.5
        )
        if k:
            g += line(whole[0] - 5, ty, whole[0], ty) + text(
                whole[0] - 8, ty + 4, label, anchor="end", size=10.5
            )
    x0, y0, x1, y1 = GRAPH_WINDOW

    # the window, enlarged
    zoom = 400
    zw, zh = (x1 - x0) * zoom, (y1 - y0) * zoom
    zx, zy = DIAGRAM_WIDTH - 20 - zw, top + (size - zh) / 2
    big = (zx - x0 * zoom, zy - (1 - y1) * zoom, zoom)
    g += f'<clipPath id="{p}-win">{rect(zx, zy, zw, zh)}</clipPath>'
    g += f'<g clip-path="url(#{p}-win)">'
    g += draw_partition(p, ranks, hi, big) + "</g>"
    # the same edges, clipped at the window with the colors of the whole view
    for _, _, color, (u0, v0, u1, v1) in inset_edges(marks, GRAPH_WINDOW):
        g += line(*_point(big, u0, v0), *_point(big, u1, v1), stroke=color)
    for cell in leaves:
        u, v = _unit_center(cell)
        if x0 <= u <= x1 and y0 <= v <= y1:
            g += circle(*_point(big, u, v), 2.6, fill=INK)
    g += rect(zx, zy, zw, zh, stroke=INK, sw=HEAVY)
    g += text(
        zx,
        zy - 10,
        f"[{_frac(x0)}, {_frac(x1)}) \u00d7 [{_frac(y0)}, {_frac(y1)}), enlarged",
        size=11,
        cls="sm-fig-code",
    )

    jump = [b for a, b in edges if a == JUMP_CELL] + [a for a, b in edges if b == JUMP_CELL]
    finer = [c for c in jump if c[0] == JUMP_CELL[0] + 1]
    if len(finer) != 2 or face_adjacent(*CORNER, hi):
        raise ValueError("the callouts no longer point at a level jump and a corner contact")
    cx, cy = _center(big, JUMP_CELL)
    g += callout(cx, cy, cx - 25, cy - 24, "a")
    u, v = (CORNER[1][1] + 1) / 2 ** CORNER[1][0], (CORNER[1][2] + 1) / 2 ** CORNER[1][0]
    kx, ky = _point(big, u, v)
    g += callout(kx, ky, kx + 25, ky - 22, "b")

    # the key
    ky = top + size + 48
    g += _swatch(p, 44, ky, 0, "rank 0")
    g += _swatch(p, 116, ky, 1, "rank 1")
    g += line(194, ky - 4, 218, ky - 4, stroke=INK) + text(224, ky, "edge inside a rank", size=11.5)
    g += line(344, ky - 4, 368, ky - 4, stroke=RED) + text(
        374, ky, "edge across the ranks", size=11.5
    )
    across = sum(1 for _, _, color in marks if color == RED)
    jumps = sum(1 for a, b, _ in marks if a[0] != b[0])
    counts = [sum(1 for r in ranks.values() if r == k) for k in (0, 1)]
    return Drawing(
        DIAGRAM_WIDTH,
        ky + 10,
        f"A graded mesh with {len(leaves)} cells of levels {GRAPH_LEVELS[0]} and {hi},"
        f" cut by the Hilbert curve into rank 0 ({counts[0]} cells) and rank 1"
        f" ({counts[1]} cells), with its cell graph: a vertex at the center of each cell"
        f" and {len(edges)} edges between cells that share a face, {jumps} of them across"
        f" a level jump and {across} across the two ranks, in red. At right, a part of the"
        " mesh enlarged: a, a coarse cell linked to each of the two finer cells along its"
        " lower face; b, a coarse cell and a finer cell that touch by a corner only, with"
        " no edge between them.",
        patterns(p) + rank_patterns(p),
        g,
    )
