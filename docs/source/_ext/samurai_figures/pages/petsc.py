r"""The figure of ``howto/petsc.md``: the PETSc numbering of a 4 by 4 mesh on two ranks.

The numbering is the one ``petsc_linear_heat`` writes to ``petsc_indices.h5`` when it
runs on 2 MPI ranks with ``--print-petsc-numbering``. The ``.h5`` file stores, for
each rank, its cells and the cells and ghosts it holds, with the fields
``owner_rank`` and ``petsc_cell_index``. This module turns the part of that file the
figure needs into ``data/petsc_numbering.json``; it needs ``h5py``, which
``docs/environment.yml`` does not list (``pip install h5py``). From a build with
``-DWITH_MPI=ON -DWITH_PETSC=ON -DBUILD_SNIPPETS=ON``::

    mpiexec -n 2 ./petsc_linear_heat --min-level 2 --max-level 2 \
        --print-petsc-numbering --dont-redirect-output
    cd docs/source/_ext
    python -c "import sys; from samurai_figures.pages import petsc; petsc.main(sys.argv[1:])" \
        <run directory>/petsc_indices.h5

The figure computes the global PETSc index of every unknown from the file, as
``global_numbering.hpp`` does, and stops with an error when the file does not give
the numbering the page prints.
"""

import json
import sys
from pathlib import Path

from ..draw import HAIR, HEAVY, INK, INK2, RULE, line, rect, text
from ..partition import rank_patterns
from ..registry import PLATE_WIDTH, Drawing, figure

DATA = Path(__file__).resolve().parent.parent / "data" / "petsc_numbering.json"
FORMAT, VERSION = "samurai-petsc-numbering", 1
SOURCE = (
    "mpiexec -n 2 petsc_linear_heat --min-level 2 --max-level 2"
    " --print-petsc-numbering --dont-redirect-output"
)
# what howto/petsc.md says about the run
LEVEL, UNKNOWNS, OWNED = 2, 64, ((0, 32), (32, 64))
# the gap between a brace under the grid and the edges of its columns, and the margin
# of a panel of the plate, in drawing units
BRACE_GAP, PAD = 5, 14


def _cells(group, level):
    """Return the indices (i, j) of the cells of ``group`` at ``level``, in file order.

    Raise ``ValueError`` when a cell does not fall on the grid of the level.
    """
    side = 2.0**-level
    points = group["points"][()][:, :2]
    corners = points[group["connectivity"][()].astype(int)]
    lower = corners.min(axis=1) / side
    upper = corners.max(axis=1) / side
    cells = []
    for (x, y), (xe, ye) in zip(lower.tolist(), upper.tolist()):
        i, j = round(x), round(y)
        if max(abs(x - i), abs(y - j), abs(xe - i - 1), abs(ye - j - 1)) > 1e-9:
            raise ValueError(f"the cell at ({x}, {y}) is not a cell of level {level}")
        cells.append((i, j))
    return cells


def extract(h5_path, level=LEVEL):
    """Return the JSON document of the numbering in ``h5_path``, a ``petsc_indices.h5``.

    For each rank: ``cells``, its cells (i, j) at ``level``, and ``held``, the cells and
    ghosts it holds as ``[i, j, owner_rank, petsc_cell_index]``, copied from the file.

    ``petsc_cell_index`` is ``cell_indices`` of ``compute_cell_ownership.hpp``, a number
    local to the rank that wrote it, given in the order of the samurai cell indices:

    - for a cell the rank owns, its place among the cells the rank owns, from 0. With
      one unknown per cell, ``global_numbering.hpp`` gives it the global index
      ``rank_shift + petsc_cell_index``, where ``rank_shift`` is the number of
      unknowns the ranks before it own;
    - for a cell another rank owns, its place among the cells the rank holds but does
      not own, from 0. It is neither the number the owner gives the cell nor its
      global index: ``global_numbering.hpp`` receives the global index from the owner
      over MPI, and the file does not store it. ``numbering`` therefore reads the
      global index of every unknown from the rank that owns it.
    """
    import h5py  # pylint: disable=import-outside-toplevel

    base = f"level/{level}/mesh/level_{level}_"
    with h5py.File(h5_path, "r") as h5:
        names = sorted(h5[f"{base}cells"], key=lambda name: int(name.split("_")[1]))
        ranks = []
        for name in names:
            group = h5[f"{base}cells and ghosts/{name}"]
            fields = group["fields"]
            held = [
                [i, j, int(owner), int(index)]
                for (i, j), owner, index in zip(
                    _cells(group, level),
                    fields["owner_rank"][()],
                    fields["petsc_cell_index"][()],
                )
            ]
            cells = [list(c) for c in _cells(h5[f"{base}cells/{name}"], level)]
            ranks.append({"rank": int(name.split("_")[1]), "cells": cells, "held": held})
    return {"format": FORMAT, "version": VERSION, "source": SOURCE, "level": level, "ranks": ranks}


def dumps(document):
    """Return the JSON text of ``document``, one cell per line."""
    head = {k: v for k, v in document.items() if k != "ranks"}
    lines = [f"  {json.dumps(k)}: {json.dumps(v)}," for k, v in head.items()]
    parts = []
    for part in document["ranks"]:
        rows = {
            key: ",\n".join(f"        {json.dumps(c)}" for c in part[key])
            for key in ("cells", "held")
        }
        parts.append(
            f'    {{\n      "rank": {part["rank"]},\n'
            f'      "cells": [\n{rows["cells"]}\n      ],\n'
            f'      "held": [\n{rows["held"]}\n      ]\n    }}'
        )
    return "{\n" + "\n".join(lines) + '\n  "ranks": [\n' + ",\n".join(parts) + "\n  ]\n}\n"


def numbering(document):
    """Return the global PETSc index, the owner and the kind of each unknown.

    The result maps (i, j) to ``(global index, owner rank, is a cell)``; a cell is in
    the domain, the other unknowns are boundary ghosts. The global index of an unknown
    is ``rank_shift + petsc_cell_index`` read in the ``held`` list of its owner, as
    ``extract`` explains; the ``petsc_cell_index`` other ranks give it is not used.
    Raise ``ValueError`` when the ranks disagree on an owner or the numbering has a gap.
    """
    parts = sorted(document["ranks"], key=lambda part: part["rank"])
    owners, offset, unknowns = {}, 0, {}
    domain = {tuple(c): part["rank"] for part in parts for c in part["cells"]}
    for part in parts:
        r = part["rank"]
        for i, j, owner, _ in part["held"]:
            if owners.setdefault((i, j), owner) != owner:
                raise ValueError(f"the ranks disagree on the owner of the cell ({i}, {j})")
        owned = {(i, j): index for i, j, owner, index in part["held"] if owner == r}
        if sorted(owned.values()) != list(range(len(owned))):
            raise ValueError(f"rank {r} does not number its unknowns from 0 without a gap")
        for c, index in owned.items():
            if domain.get(c, r) != r:
                raise ValueError(f"the cell {c} of rank {domain[c]} is owned by rank {r}")
            unknowns[c] = (offset + index, r, c in domain)
        offset += len(owned)
    if sorted(g for g, _, _ in unknowns.values()) != list(range(offset)):
        raise ValueError("the global indices are not 0 to n - 1, each once")
    return unknowns


def _check(document, unknowns):
    """Raise ``ValueError`` unless the run is the one ``howto/petsc.md`` describes.

    The domain is 4 by 4 cells of level 2, the ghosts are two layers around it, and
    rank r owns the global indices ``OWNED[r]``. Return the box of the unknowns,
    ``(i0, j0, n)``, and the box of the domain, ``(i0, j0, n)``.
    """
    if document["level"] != LEVEL or len(unknowns) != UNKNOWNS:
        raise ValueError(f"expected {UNKNOWNS} unknowns at level {LEVEL}, the file differs")
    ranges = []
    for r in sorted({owner for _, owner, _ in unknowns.values()}):
        indices = sorted(g for g, owner, _ in unknowns.values() if owner == r)
        ranges.append((indices[0], indices[-1] + 1))
        if indices != list(range(*ranges[-1])):
            raise ValueError(f"rank {r} does not own one run of global indices")
    if tuple(ranges) != OWNED:
        found, said = ([f"[{a}, {b})" for a, b in runs] for runs in (ranges, OWNED))
        raise ValueError(f"the ranks own {', '.join(found)}, the page says {', '.join(said)}")

    def square(cells):
        i0, j0 = min(i for i, _ in cells), min(j for _, j in cells)
        n = max(i for i, _ in cells) - i0 + 1
        if set(cells) != {(i0 + a, j0 + b) for a in range(n) for b in range(n)}:
            raise ValueError("the cells do not fill a square")
        return i0, j0, n

    box = square(list(unknowns))
    domain = square([c for c, (_, _, real) in unknowns.items() if real])
    if domain != (0, 0, 4) or box != (-2, -2, 8):
        raise ValueError(f"expected 4 by 4 cells in two layers of ghosts, got {domain}, {box}")
    return box, domain


@figure
def petsc_numbering(p):
    """Draw the global PETSc index of every unknown of the 4 by 4 mesh on 2 ranks.

    Plate of howto/petsc.md, at "If you run with MPI". The unknowns are the cells and
    the boundary ghosts of the mesh, filled with the fill of the rank that owns them
    (``partition.rank_patterns``); the numbers and the ranges come from
    ``data/petsc_numbering.json`` through ``numbering``. The plate has two panels, the
    grid and the key of the ranks, so that the key stacks under the grid on narrow
    screens instead of leaving the screen.
    """
    document = json.loads(DATA.read_text(encoding="utf-8"))
    unknowns = numbering(document)
    (bi, bj, n), (di, dj, dn) = _check(document, unknowns)

    cw, top, gap, key_w = 38, 14, 40, 150
    gx = (PLATE_WIDTH - n * cw - gap - key_w) / 2
    right = gx + n * cw

    def x(i):
        """Return the left edge of the column i in the drawing."""
        return gx + (i - bi) * cw

    def y(j):
        """Return the top of the row j in the drawing, y up."""
        return top + (bj + n - 1 - j) * cw

    g = ""
    for (i, j), (_, owner, _) in sorted(unknowns.items()):
        g += rect(x(i), y(j), cw, cw, fill=f"url(#{p}-rank{owner})", stroke=INK, sw=HAIR)
    for (i, j), (index, _, _) in sorted(unknowns.items()):
        g += text(
            x(i) + cw / 2,
            y(j) + cw / 2 + 4,
            str(index),
            anchor="middle",
            size=11.5,
            fill=INK,
            cls="sm-fig-code",
            knock=True,
        )
    # the edge between the ranks, the edge of the domain, the edge of the ghosts
    split = None
    for j in range(bj + 1, bj + n):
        below = {unknowns[i, j - 1][1] for i in range(bi, bi + n)}
        above = {unknowns[i, j][1] for i in range(bi, bi + n)}
        if below != above:
            split = y(j - 1)
            g += line(gx, split, right, split, sw=HEAVY)
    if split is None:
        raise ValueError("the ranks do not meet along a row of the grid")
    g += rect(x(di), y(dj + dn - 1), dn * cw, dn * cw, stroke=INK, sw=HEAVY)
    g += rect(gx, top, n * cw, n * cw, stroke=INK, sw=HEAVY)

    # Under the grid, one brace per run of columns: two layers of ghosts on each side
    # of the domain. Each brace stops BRACE_GAP short of the edges of its columns, so
    # that two braces never meet at the edge of the domain.
    base = top + n * cw
    for a, b, label in ((bi, di, "ghosts"), (di, di + dn, "domain"), (di + dn, bi + n, "ghosts")):
        x0, x1 = x(a) + BRACE_GAP, x(b) - BRACE_GAP
        g += line(x0, base + 12, x1, base + 12, stroke=INK2, sw=RULE)
        for e in (x0, x1):
            g += line(e, base + 6, e, base + 12, stroke=INK2, sw=RULE)
        g += text((x0 + x1) / 2, base + 28, label, anchor="middle", size=11.5, fill=INK2)
    height = base + 38

    # The key, rank 1 above the edge between the ranks and rank 0 under it, as in
    # the grid.
    kx, block = right + gap, 58
    keys, label = "", []
    for r, (first, end) in enumerate(OWNED):
        cells = sum(1 for _, owner, real in unknowns.values() if owner == r and real)
        ky = split + 14 if r == 0 else split - 14 - block
        keys += rect(kx, ky, 14, 14, fill=f"url(#{p}-rank{r})", stroke=INK, sw=HAIR)
        keys += text(kx + 22, ky + 12, f"rank {r}", size=12.5, fill=INK)
        keys += text(
            kx,
            ky + 35,
            f'owns <tspan class="sm-fig-code" fill="{INK}">[{first}, {end})</tspan>',
            size=12,
            fill=INK2,
        )
        keys += text(
            kx, ky + 54, f"{cells} cells + {end - first - cells} ghosts", size=12, fill=INK2
        )
        label.append(
            f"rank {r} owns [{first}, {end}), indices {first} to {end - 1}:"
            f" {cells} cells and {end - first - cells} ghosts"
        )
    if any((y(j) >= split) != (owner == 0) for (_, j), (_, owner, _) in unknowns.items()):
        raise ValueError("the key draws rank 0 under rank 1: rank 0 must own the lower rows")
    g += keys

    # on narrow screens: the grid, then the key under it, at the same scale
    pw = n * cw + 2 * PAD
    key_top, key_bottom = split - 14 - block, split + 14 + block
    panels = (
        (gx - PAD, 0, pw, height, ""),
        (kx - PAD, key_top - PAD, pw, key_bottom - key_top + 2 * PAD, ""),
    )
    return Drawing(
        PLATE_WIDTH,
        height,
        f"A grid of {n} by {n} squares: the {dn} by {dn} cells of the domain, outlined with a"
        f" heavy rule, inside two layers of boundary ghosts, {UNKNOWNS - dn * dn} squares. Each"
        f" square holds its global PETSc index, from 0 at the bottom left, row by row from left"
        f" to right, to {UNKNOWNS - 1} at the top right. The bottom rows are hatched in the tone"
        f" of rank 0 and the top rows dotted in the tone of rank 1, a heavy rule between them;"
        f" {label[0]}; {label[1]}.",
        rank_patterns(p),
        g,
        panels=panels,
    )


def main(argv):
    """Write ``data/petsc_numbering.json`` from the ``petsc_indices.h5`` in ``argv``."""
    if len(argv) != 1:
        sys.exit("usage: petsc.main([<petsc_indices.h5>]), see the docstring of petsc.py")
    document = extract(argv[0])
    _check(document, numbering(document))
    DATA.write_text(dumps(document), encoding="utf-8")
