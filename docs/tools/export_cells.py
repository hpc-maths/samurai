"""Export the mesh of a samurai ``.h5`` file as a JSON cell list for the figures.

A figure of the documentation draws the mesh its program produced. When the
figure cannot rebuild that mesh in Python (a random mesh, a mesh adapted to a
solution, the subdomains of an MPI run), the program saves it with
``samurai::save`` and this script turns the ``.h5`` file into a small JSON file,
checked in next to the page module that draws it
(``docs/source/_ext/samurai_figures/pages/``). ``samurai_figures.cells.load_cells``
reads it back.

The script needs ``h5py``, which the ``samurai-doc`` environment does not hold:
install it there with ``pip install h5py``, which brings ``numpy``. Run the program with
``--save-debug-fields``, then export its ``.h5`` file::

    ./finite-volume-linear-convection-obstacle --Tf 0.5 --nfiles 1 \\
        --save-debug-fields --path run
    python docs/tools/export_cells.py run/linear_convection_obstacle.h5 \\
        docs/source/_ext/samurai_figures/pages/obstacle.json \\
        --source "linear_convection_obstacle.cpp --Tf 0.5 --nfiles 1"

The ``.h5`` file is the one ``samurai::save`` writes. The script reads the group
``/mesh`` (``--group`` names another one, such as ``/mesh/cells`` for a file saved
by mesh id), either directly or split in ``rank_<r>`` groups when the program ran
on several MPI ranks. The file stores the corners of the cells, not their level
and indices, so the script recovers them as follows:

- From the fields ``levels`` and ``indices_<d>`` that ``--save-debug-fields``
  adds: run the program with this option and nothing else is needed.
- Otherwise the level comes from a field ``level``, when the program saves one,
  or from the side of the cell, which is ``scaling_factor / 2**level``; the
  indices come from the corner, which is ``origin + index * side``. Give
  ``--scaling-factor`` and ``--origin`` when they differ from 1 and 0.

The script stops with an error when a cell does not fall on the integer grid of
its level, rather than writing a wrong mesh.

The JSON file holds::

    {
      "format": "samurai-cells", "version": 1,
      "source": "<the program and its options>",
      "dim": 2, "scaling_factor": 1.0, "origin": [0.0, 0.0],
      "levels": {"0": [[0, 0, 4], [1, 0, 1], ...], "1": [...]}
    }

``levels`` maps each level to its intervals, one per row: ``[y, start, end]`` in
2D (``[start, end]`` in 1D, ``[y, z, start, end]`` in 3D) for the cells
``start`` to ``end - 1`` along x, half-open like every samurai interval. A file
with ``rank_<r>`` groups holds ``"ranks": [{"rank": r, "levels": {...}}, ...]``
instead of ``levels``. Rows are sorted, so that a new export of the same mesh
gives the same file.
"""

import argparse
import json
import math
import sys
from pathlib import Path

import h5py
import numpy as np

FORMAT, VERSION = "samurai-cells", 1
# vertices per cell -> dimension
DIMS = {2: 1, 4: 2, 8: 3}
# relative tolerance on coordinates read as doubles
TOL = 1e-6


class ExportError(Exception):
    """The file does not hold a mesh this script can export."""


def _parts(group):
    """The ``(rank, group)`` of each part of the mesh, ``rank`` None without MPI."""
    if "points" in group:
        return [(None, group)]
    ranks = sorted(
        (int(name.split("_", 1)[1]), group[name])
        for name in group
        if name.startswith("rank_") and "points" in group[name]
    )
    if not ranks:
        raise ExportError(f"{group.name} holds neither 'points' nor 'rank_<r>' groups")
    return ranks


def _field(group, *names):
    """The first field of ``group/fields`` among ``names``, or None."""
    fields = group.get("fields")
    for name in names:
        if fields is not None and name in fields:
            return fields[name][()]
    return None


def _integers(values, what):
    """Round ``values`` to integers, or raise when they are not integers."""
    rounded = np.rint(values)
    if not np.allclose(values, rounded, rtol=0, atol=TOL):
        raise ExportError(
            f"the {what} are not integers: give --scaling-factor and --origin, or run the"
            " program with --save-debug-fields"
        )
    return rounded.astype(np.int64)


def _clean(value):
    """``value`` without the rounding noise of the coordinates it comes from."""
    return float(f"{value:.12g}")


def _cells(group, dim, scaling_factor, origin):
    """The cells of one part as ``(levels, indices, scaling_factor, origin)``.

    ``scaling_factor`` and ``origin`` are the values given on the command line,
    None when not given; the values returned are the ones the cells use.
    """
    points = group["points"][()][:, :dim]
    corners = points[group["connectivity"][()].astype(np.int64)]
    lower, upper = corners.min(axis=1), corners.max(axis=1)
    side = upper[:, 0] - lower[:, 0]
    if not np.allclose(upper - lower, side[:, None], rtol=TOL, atol=0):
        raise ExportError("a cell is not a square: this is not a samurai mesh")

    levels = _field(group, "levels", "level")
    if levels is None:
        scale = 1.0 if scaling_factor is None else scaling_factor
        levels = _integers(np.log2(scale / side), "levels deduced from the cell sides")
    levels = levels.astype(np.int64)
    scales = side * 2.0**levels
    scale = float(scales[0])
    if not np.allclose(scales, scale, rtol=TOL, atol=0):
        raise ExportError("the cell sides do not match their levels")
    if scaling_factor is not None and not math.isclose(scale, scaling_factor, rel_tol=TOL):
        raise ExportError(
            f"the cells have the scaling factor {_clean(scale)}, not {scaling_factor}"
        )

    saved = [_field(group, f"indices_{d}") for d in range(dim)]
    if all(index is not None for index in saved):
        indices = np.stack(saved, axis=1).astype(np.int64)
        origins = lower - indices * side[:, None]
        found = origins[0]
        if not np.allclose(origins, found, rtol=0, atol=TOL * scale):
            raise ExportError("the saved indices do not match the cell corners")
        if origin is not None and not np.allclose(found, origin, rtol=0, atol=TOL * scale):
            raise ExportError(f"the cells have the origin {list(found)}, not {origin}")
        return levels, indices, _clean(scale), [_clean(v) for v in found]

    if origin is None:
        print(
            "export_cells.py: the file has no indices_<d> fields, the origin is taken as 0",
            file=sys.stderr,
        )
    start = np.zeros(dim) if origin is None else np.asarray(origin, dtype=float)
    indices = _integers((lower - start) / side[:, None], "cell indices")
    return levels, indices, _clean(scale), [_clean(v) for v in start]


def _intervals(levels, indices):
    """The half-open intervals along x of the cells, per level.

    Each row is the other coordinates (y, then z) followed by start and end. The
    levels are in increasing order, the rows of a level sorted.
    """
    runs = {}
    cells = sorted(
        zip(levels.tolist(), map(tuple, indices.tolist())),
        key=lambda cell: (cell[0], cell[1][1:], cell[1][0]),
    )
    for level, index in cells:
        rows = runs.setdefault(str(level), [])
        key, x = list(index[1:]), index[0]
        if rows and rows[-1][:-2] == key and rows[-1][-1] == x:
            rows[-1][-1] = x + 1
        else:
            rows.append(key + [x, x + 1])
    return runs


def export(path, group="/mesh", scaling_factor=None, origin=None, source=""):
    """The JSON document of the mesh in the ``.h5`` file ``path``, as a dict."""
    with h5py.File(path, "r") as h5:
        if group not in h5:
            raise ExportError(f"{path} has no group {group}")
        parts = _parts(h5[group])
        width = parts[0][1]["connectivity"].shape[1]
        if width not in DIMS:
            raise ExportError(f"cells with {width} vertices are not samurai cells")
        dim = DIMS[width]
        if origin is not None and len(origin) != dim:
            raise ExportError(f"--origin needs {dim} values")
        exported, scales, origins = [], set(), set()
        for rank, part in parts:
            levels, indices, scale, start = _cells(part, dim, scaling_factor, origin)
            scales.add(round(scale / TOL))
            origins.add(tuple(round(v / (TOL * scale)) for v in start))
            exported.append((rank, _intervals(levels, indices)))
    if len(scales) > 1 or len(origins) > 1:
        raise ExportError("the ranks do not share one scaling factor and one origin")
    document = {"format": FORMAT, "version": VERSION}
    if source:
        document["source"] = source
    document.update(dim=dim, scaling_factor=scale, origin=start)
    if exported[0][0] is None:
        document["levels"] = exported[0][1]
    else:
        document["ranks"] = [{"rank": rank, "levels": levels} for rank, levels in exported]
    return document


def _levels(levels, indent):
    """The lines of a ``levels`` mapping, one interval per line."""
    pad = " " * indent
    blocks = []
    for level, rows in levels.items():
        body = ",\n".join(f"{pad}    {json.dumps(row)}" for row in rows)
        blocks.append(f'{pad}  "{level}": [\n{body}\n{pad}  ]')
    return "{\n" + ",\n".join(blocks) + f"\n{pad}}}"


def dumps(document):
    """The JSON text of ``document``, one interval per line."""
    items = []
    for key, value in document.items():
        if key == "levels":
            text = _levels(value, 2)
        elif key == "ranks":
            ranks = [
                f'    {{"rank": {part["rank"]}, "levels": {_levels(part["levels"], 4)}}}'
                for part in value
            ]
            text = "[\n" + ",\n".join(ranks) + "\n  ]"
        else:
            text = json.dumps(value)
        items.append(f'  "{key}": {text}')
    return "{\n" + ",\n".join(items) + "\n}\n"


def main(argv=None):
    """Run the command line."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("h5", type=Path, help="the .h5 file written by samurai::save")
    parser.add_argument("json", type=Path, help="the JSON file to write")
    parser.add_argument("--group", default="/mesh", help="the mesh group (default /mesh)")
    parser.add_argument("--scaling-factor", type=float, help="the scaling factor of the mesh")
    parser.add_argument("--origin", type=float, nargs="+", help="the origin point of the mesh")
    parser.add_argument("--source", default="", help="the program and options that wrote it")
    args = parser.parse_args(argv)
    try:
        document = export(args.h5, args.group, args.scaling_factor, args.origin, args.source)
    except (ExportError, OSError) as error:
        sys.exit(f"export_cells.py: {error}")
    args.json.write_text(dumps(document), encoding="utf-8")


if __name__ == "__main__":
    main()
