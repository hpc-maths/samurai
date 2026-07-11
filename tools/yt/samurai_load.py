# Copyright 2018-2025 the samurai's authors
# SPDX-License-Identifier:  BSD-3-Clause
"""
Reconstruction of a samurai data file as a list of AMR grids.

The result is ready to be handed to ``yt.load_amr_grids``. This module only
depends on ``numpy`` and ``h5py`` so that it can be unit-tested without yt.
The yt loader (``samurai_yt.py``) is a thin wrapper that turns the grids
returned here into a ``StreamDataset``.

Samurai never stores the cell geometry.  A data file only contains, per level and
per direction, the compressed interval representation of the mesh (``m_cells`` /
``m_offsets``, see ``include/samurai/level_cell_array.hpp``) plus ``dim``,
``min_level``/``max_level``, ``origin_point`` and ``scaling_factor``.  Field
values carry no stored indexing: their order is exactly the traversal order of
``for_each_cell`` (levels min->max, then the offset-driven interval iterator of
each level).

Mapping samurai to yt
---------------------
Samurai is *cell-based* (octree) AMR: any cell can be a leaf, and a refined cell
splits into ``2**dim`` children.  yt's :func:`yt.load_amr_grids` expects
*block* AMR (Enzo style): every grid must start on a cell edge of its **parent**
level.  A lone fine cell sitting on an odd index (the second child of its parent)
does not, so cells cannot be handed to yt one by one.

We therefore rebuild the octree as block-aligned grids:

* one root grid covering the whole domain at ``min_level`` (yt level 0);
* for every refined (internal) cell, one ``2**dim``-cell grid holding its
  children, whose left edge is the parent cell edge -- exactly what yt wants.

Cells covered by a finer grid are masked by yt, so a coarse cell's placeholder
value is never selected; only leaves survive, each with its own value.  Grid
edges are computed with yt's own ``linspace`` formula so they match the internal
alignment check (bug #1295) bit for bit.
"""

import os
import re

import numpy as np
import h5py


def read_time(filename):
    """Return the simulation time stored in a samurai file, or ``None`` if absent.

    ``samurai::dump`` combined with the ``MetadataWriter`` (see
    ``include/samurai/io/metadata.hpp``) stores the time as the HDF5 root
    attribute ``"time"``.
    """
    if not filename.endswith(".h5"):
        filename = filename + ".h5"
    try:
        with h5py.File(filename, "r") as f:
            if "time" in f.attrs:
                return float(f.attrs["time"])
    except OSError:
        pass
    return None


def discover_series(filename):
    """Return ``(files, times)`` for the numbered series containing ``filename``.

    A file written per output step is named with a trailing integer before
    ``.h5`` (e.g. ``sol_ite_0.h5``, ``sol_ite_1.h5``, ...).  Given any one of
    them, this globs the siblings that share the same prefix so that opening a
    single file exposes the whole animation.  If ``filename`` has no trailing
    number, the series is just that one file.

    Files are ordered by their trailing integer (natural order).  ``times`` are
    the simulation times read from the ``"time"`` HDF5 attribute when **every**
    file exposes it; otherwise they fall back to the trailing integers.
    """
    filename = os.path.abspath(filename)
    if not filename.endswith(".h5"):
        filename = filename + ".h5"
    directory = os.path.dirname(filename)
    base = os.path.basename(filename)[:-3]  # strip ".h5"

    match = re.match(r"^(.*?)(\d+)$", base)
    if match is None:
        files = [filename]
        iterations = [0]
    else:
        prefix = match.group(1)
        pattern = re.compile(r"^" + re.escape(prefix) + r"(\d+)\.h5$")
        try:
            entries = os.listdir(directory or ".")
        except OSError:
            entries = []
        found = []
        for name in entries:
            m = pattern.match(name)
            if m is not None:
                found.append((int(m.group(1)), os.path.join(directory, name)))
        if not found:
            files = [filename]
            iterations = [0]
        else:
            found.sort()
            files = [p for _, p in found]
            iterations = [n for n, _ in found]

    # Prefer the stored simulation time when available for every file.
    times = [read_time(f) for f in files]
    if any(t is None for t in times):
        times = [float(n) for n in iterations]
    return files, times


def _slice_partitioned(group, rank):
    """Return the ``rank``-th slice of a samurai partitioned dataset.

    A partitioned dataset is a group holding ``data`` (the concatenated buffer of
    every MPI rank) and ``partition`` (cumulative per-rank sizes, length
    ``n_process + 1``); see ``dump(file, name, std::vector<T>)`` in
    ``restart.hpp``.
    """
    partition = group["partition"][()]
    begin = int(partition[rank])
    end = int(partition[rank + 1])
    return group["data"][begin:end]


def _enumerate_intervals(cells, offsets, dim):
    """Yield the mesh intervals of one level in ``for_each_cell`` order.

    Each yielded tuple ``(x_start, x_end, y, z)`` is one x-run of contiguous
    cells (``y``/``z`` default to ``0`` in the unused dimensions).  ``cells[d]``
    is the structured ``Interval`` array of ``m_cells[d]`` (fields ``start``,
    ``end``, ``step``, ``index``).  ``offsets[d]`` (``d >= 1``) is ``m_offsets[d-1]``:
    for an interval ``iv`` of direction ``d`` and a coordinate ``c`` in
    ``[iv.start, iv.end)``, the attached intervals of direction ``d-1`` span
    ``offsets[d][iv.index + c] .. offsets[d][iv.index + c + 1]``.

    Yielding in this exact order is what keeps the flat field buffer aligned with
    the reconstructed patches.
    """
    x_iv = cells[0]
    if x_iv is None or len(x_iv) == 0:
        return
    x_start = x_iv["start"].astype(np.int64)
    x_end = x_iv["end"].astype(np.int64)

    if dim == 1:
        for xi in range(len(x_iv)):
            yield int(x_start[xi]), int(x_end[xi]), 0, 0
        return

    if dim == 2:
        off0 = offsets[1].astype(np.int64)  # m_offsets[0]: y -> x-interval range
        for yiv in cells[1]:
            base = int(yiv["index"])
            for y in range(int(yiv["start"]), int(yiv["end"])):
                for xi in range(off0[base + y], off0[base + y + 1]):
                    yield int(x_start[xi]), int(x_end[xi]), y, 0
        return

    if dim == 3:
        off0 = offsets[1].astype(np.int64)  # m_offsets[0]: y -> x-interval range
        off1 = offsets[2].astype(np.int64)  # m_offsets[1]: z -> y-interval range
        y_iv = cells[1]
        for ziv in cells[2]:
            zbase = int(ziv["index"])
            for z in range(int(ziv["start"]), int(ziv["end"])):
                for yi in range(off1[zbase + z], off1[zbase + z + 1]):
                    yiv = y_iv[yi]
                    ybase = int(yiv["index"])
                    for y in range(int(yiv["start"]), int(yiv["end"])):
                        for xi in range(off0[ybase + y], off0[ybase + y + 1]):
                            yield int(x_start[xi]), int(x_end[xi]), y, z
        return

    raise ValueError(f"unsupported dimension {dim}")


def _read_leaves(root, fields, dim, min_level, max_level, n_process):
    """Read every leaf cell as ``(keys, levels, values)``.

    ``keys`` is an ``(n_cells, 3)`` int array of ``(i, j, k)`` indices at each
    cell's own level (unused dimensions are ``0``); ``levels`` its per-cell level;
    ``values`` a ``{field: (n_cells, n_comp)}`` mapping.  Ranks are read in turn
    and concatenated, so the flat field buffers stay aligned with the cells.
    """
    mesh = root["mesh"]
    key_parts, level_parts = [], []
    value_parts = {name: [] for name in fields}

    for rank in range(n_process):
        rank_keys, rank_levels = [], []
        for level in range(min_level, max_level + 1):
            level_group = mesh.get(f"level/{level}")
            if level_group is None:
                continue
            cells, offsets, empty = {}, {}, False
            for d in range(dim):
                iv_group = level_group.get(f"dim/{d}/intervals")
                if iv_group is None:
                    empty = True
                    break
                cells[d] = _slice_partitioned(iv_group, rank)
                if d >= 1:
                    offsets[d] = _slice_partitioned(level_group.get(f"dim/{d}/offsets"), rank)
            if empty or cells[0] is None or len(cells[0]) == 0:
                continue
            for x0, x1, y, z in _enumerate_intervals(cells, offsets, dim):
                xs = np.arange(x0, x1, dtype=np.int64)
                block = np.zeros((xs.size, 3), dtype=np.int64)
                block[:, 0], block[:, 1], block[:, 2] = xs, y, z
                rank_keys.append(block)
                rank_levels.append(np.full(xs.size, level, dtype=np.int64))

        if not rank_keys:
            continue
        rank_keys = np.concatenate(rank_keys)
        key_parts.append(rank_keys)
        level_parts.append(np.concatenate(rank_levels))
        for name, n_comp in fields.items():
            buf = np.asarray(_slice_partitioned(root[f"fields/{name}/data"], rank))
            value_parts[name].append(buf.reshape(rank_keys.shape[0], n_comp))

    if not key_parts:
        return (
            np.zeros((0, 3), dtype=np.int64),
            np.zeros(0, dtype=np.int64),
            {name: np.zeros((0, n_comp)) for name, n_comp in fields.items()},
        )
    keys = np.concatenate(key_parts)
    levels = np.concatenate(level_parts)
    values = {name: np.concatenate(parts) for name, parts in value_parts.items()}
    return keys, levels, values


def _validate_samurai_group(root, filename):
    """Raise a clear error if ``root`` is not a samurai mesh group."""
    if "mesh/dim" in root:
        return
    # Detect a save() file (explicit points/connectivity) for an actionable message.
    looks_like_save = "mesh" in root and (
        any(k in root["mesh"] for k in ("points", "connectivity"))
        or any(
            isinstance(root["mesh"][k], h5py.Group)
            and ("points" in root["mesh"][k] or "connectivity" in root["mesh"][k])
            for k in root["mesh"].keys()
        )
    )
    if looks_like_save:
        raise ValueError(
            f"'{filename}' looks like a samurai save() file (explicit "
            "points/connectivity mesh), which this loader does not handle. "
            "Open the compressed dump/restart file instead."
        )
    raise ValueError(f"'{filename}' is not a samurai dump file: '/mesh/dim' is missing.")


def _read_file(filename, group):
    """Open a samurai file and return its metadata plus the leaf cells."""
    if not filename.endswith(".h5"):
        filename = filename + ".h5"
    with h5py.File(filename, "r") as f:
        root = f[group] if group else f
        _validate_samurai_group(root, filename)

        dim = int(root["mesh/dim"][()])
        origin_point = np.asarray(root["mesh/origin_point"][()], dtype=np.float64)
        origin = np.zeros(3, dtype=np.float64)
        origin[:dim] = origin_point.reshape(-1)[:dim]
        meta = {
            "dim": dim,
            "min_level": int(root["mesh/min_level"][()]),
            "max_level": int(root["mesh/max_level"][()]),
            "origin": origin,
            "scaling": float(root["mesh/scaling_factor"][()]),
            "time": read_time(filename),
        }
        n_process = int(root["n_process"][()]) if "n_process" in root else 1
        field_comp = {}
        if "fields" in root:
            for name in root["fields"].keys():
                field_comp[name] = int(root[f"fields/{name}/n_comp"][()])

        keys, levels, values = _read_leaves(
            root, field_comp, dim, meta["min_level"], meta["max_level"], n_process
        )
    meta["field_comp"] = field_comp
    return meta, keys, levels, values


def read_cells(filename, group=None):
    """Return the samurai leaf cells -- pure ``numpy``/``h5py``, no yt.

    Returns ``(centers, levels, fields, meta)`` where ``centers`` is an
    ``(n_cells, 3)`` array of physical cell centers (unused dimensions padded with
    the origin), ``levels`` the per-cell level, ``fields`` a ``{name: (n_cells,)}``
    mapping (vector fields split into ``name_<c>``) and ``meta`` carries ``dim``,
    ``min_level``, ``max_level`` and ``time``.  This is the raw cell data, without
    the AMR-grid rebuild that yt needs.
    """
    meta, keys, levels, values = _read_file(filename, group)
    dim = meta["dim"]

    length = (meta["scaling"] / 2.0 ** levels).reshape(-1, 1)  # (n_cells, 1)
    centers = np.tile(meta["origin"], (keys.shape[0], 1))
    centers[:, :dim] = meta["origin"][:dim] + (keys[:, :dim] + 0.5) * length

    fields = {}
    for name, n_comp in meta["field_comp"].items():
        if n_comp == 1:
            fields[name] = values[name][:, 0]
        else:
            for c in range(n_comp):
                fields[f"{name}_{c}"] = values[name][:, c]
    return centers, levels, fields, meta


def _internal_nodes(keys, levels, min_level):
    """Return the set of refined cells ``(level, i, j, k)`` (ancestors of leaves).

    Walking up from every leaf marks each ancestor once; a cell is either a leaf
    or internal, never both, so this is exactly the set of cells that must be
    handed to yt as a refined ``2**dim``-cell grid.
    """
    internal = set()
    for row in range(keys.shape[0]):
        level = int(levels[row])
        i, j, k = int(keys[row, 0]), int(keys[row, 1]), int(keys[row, 2])
        while level > min_level:
            level -= 1
            i, j, k = i >> 1, j >> 1, k >> 1  # arithmetic shift == samurai's parent
            node = (level, i, j, k)
            if node in internal:
                break
            internal.add(node)
    return internal


def read_amr_grids(filename, group=None):
    """Read a samurai ``.h5`` file and reconstruct it as block-aligned AMR grids.

    Parameters
    ----------
    filename : str
        Path to the samurai file (with or without the ``.h5`` extension).
    group : str, optional
        HDF5 group under which the samurai layout (``mesh``, ``fields``,
        ``n_process``) lives.  Defaults to the file root; pass e.g.
        ``"grids/my_grid"`` for a file holding several grids side by side.

    Returns
    -------
    grids : list of dict
        yt grids (see :func:`yt.load_amr_grids`): a root grid covering the domain
        at ``min_level`` plus one ``2**dim``-cell grid per refined cell.  Each
        dict carries ``left_edge``, ``right_edge``, ``dimensions``, ``level``
        (0 == ``min_level``) and one array per scalar field (vector fields are
        split into ``name_<c>`` components).
    meta : dict
        ``dim``, ``domain_dimensions`` (3 ints), ``bbox`` (3x2 floats),
        ``min_level``, ``max_level``, ``time`` (or ``None``) and ``fields``
        (the list of yt field names).

    """
    meta, keys, levels, values = _read_file(filename, group)
    dim = meta["dim"]
    min_level = meta["min_level"]
    origin = meta["origin"]
    scaling = meta["scaling"]

    # Split vector fields into scalar components (one yt field each).
    fields = {}
    for name, n_comp in meta["field_comp"].items():
        if n_comp == 1:
            fields[name] = values[name][:, 0]
        else:
            for c in range(n_comp):
                fields[f"{name}_{c}"] = values[name][:, c]

    n_cells = keys.shape[0]
    # Leaf value lookup: (level, i, j, k) -> row.  Cells absent from it are
    # refined; their placeholder value is masked by the covering finer grid.
    leaf_row = {
        (int(levels[r]), int(keys[r, 0]), int(keys[r, 1]), int(keys[r, 2])): r
        for r in range(n_cells)
    }

    def fill(name, level, i0, j0, k0, dims):
        """Field array of shape ``dims`` for the cell block anchored at ``(i0,j0,k0)``."""
        col = fields[name]
        out = np.zeros(dims, dtype=col.dtype)
        for a in range(dims[0]):
            for b in range(dims[1]):
                for c in range(dims[2]):
                    r = leaf_row.get((level, i0 + a, j0 + b, k0 + c))
                    if r is not None:
                        out[a, b, c] = col[r]
        return out

    dx0 = scaling / 2.0**min_level  # coarsest cell size; yt level 0 == min_level

    # Domain box, exactly a whole number of coarsest cells (every cell nests in
    # the min_level grid, so these bounds are the samurai domain box).
    idx_lo = np.zeros(3, dtype=np.int64)
    idx_hi = np.ones(3, dtype=np.int64)  # unused dims span a single coarse cell
    for d in range(dim):
        coarse = keys[:, d] >> (levels - min_level)  # index projected to min_level
        idx_lo[d] = coarse.min()
        idx_hi[d] = coarse.max() + 1
    domain_dimensions = (idx_hi - idx_lo).astype(np.int64)

    domain_left = origin.copy()
    domain_right = origin.copy()
    domain_left[:dim] = origin[:dim] + idx_lo[:dim] * dx0
    domain_right[:dim] = origin[:dim] + idx_hi[:dim] * dx0
    domain_right[dim:] = origin[dim:] + dx0
    bbox = np.stack([domain_left, domain_right], axis=1)

    child_dims = tuple(2 if d < dim else 1 for d in range(3))

    grids = []
    # Root grid: the whole domain at the coarsest level.
    root_grid = {
        "left_edge": domain_left,
        "right_edge": domain_right,
        "dimensions": domain_dimensions.copy(),
        "level": 0,
    }
    for name in fields:
        root_grid[name] = fill(
            name, min_level, idx_lo[0], idx_lo[1], idx_lo[2], tuple(domain_dimensions)
        )
    grids.append(root_grid)

    # One 2**dim-cell grid per refined cell, its left edge on the parent cell edge.
    for level, i, j, k in sorted(_internal_nodes(keys, levels, min_level)):
        lvl = level - min_level + 1  # yt level of the children
        left = domain_left.copy()
        right = domain_right.copy()
        for d, idx in enumerate((i, j, k)):
            if d >= dim:
                continue
            # yt requires the edge to lie on a parent-level cell edge; reproduce
            # its own linspace arithmetic so the values match bit for bit.
            n_parent = int(domain_dimensions[d]) * (1 << (lvl - 1))
            m = idx - (int(idx_lo[d]) << (level - min_level))
            step = (domain_right[d] - domain_left[d]) / n_parent
            left[d] = domain_left[d] + m * step
            right[d] = domain_left[d] + (m + 1) * step
        grid = {
            "left_edge": left,
            "right_edge": right,
            "dimensions": np.array(child_dims, dtype=np.int64),
            "level": lvl,
        }
        for name in fields:
            grid[name] = fill(name, level + 1, 2 * i, 2 * j, 2 * k, child_dims)
        grids.append(grid)

    meta.update(domain_dimensions=domain_dimensions, bbox=bbox, fields=sorted(fields))
    del meta["field_comp"], meta["origin"], meta["scaling"]
    return grids, meta
