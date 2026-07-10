# Copyright 2018-2025 the samurai's authors
# SPDX-License-Identifier:  BSD-3-Clause
"""
Reconstruction of a samurai data file as an explicit finite-element
mesh (points + connectivity + cell data).

This module only depends on ``numpy`` and ``h5py`` so that it can be unit-tested
without ParaView.  The ParaView reader (``SamuraiReader.py``) is a thin wrapper
that turns the blocks returned here into VTK datasets.

Samurai never stores the cell geometry.  A data file only contains, per level and per
direction, the compressed interval representation of the mesh
(``m_cells`` / ``m_offsets``, see ``include/samurai/level_cell_array.hpp``) plus
``dim``, ``min_level``/``max_level``, ``origin_point`` and ``scaling_factor``.
The geometry is rebuilt on the fly:

    length = scaling_factor / 2**level
    corner = origin_point + length * (i, j, k)        # cell.hpp:corner()
    vertex = corner + length * unit_offset            # hdf5.hpp:get_element

Field values carry **no stored indexing**: their order is exactly the traversal
order of ``for_each_cell`` (levels min->max, then the interval/offset iterator of
each level).  Correctness therefore hinges on reproducing that traversal, which
is what :func:`_enumerate_level` does.
"""

import os
import re

import numpy as np
import h5py

# VTK cell type ids (avoid importing vtk here to keep the module dependency-free)
VTK_LINE = 3
VTK_QUAD = 9
VTK_HEXAHEDRON = 12

# Unit-cell vertex offsets, matching samurai's ``get_element`` (hdf5.hpp:61-80).
# The ordering is already the one expected by VTK_LINE / VTK_QUAD / VTK_HEXAHEDRON.
_UNIT_ELEMENT = {
    1: np.array([[0.0], [1.0]]),
    2: np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=float),
    3: np.array(
        [
            [0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],
            [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1],
        ],
        dtype=float,
    ),
}
_VTK_CELL_TYPE = {1: VTK_LINE, 2: VTK_QUAD, 3: VTK_HEXAHEDRON}


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
    ``restart.hpp``.  Empty datasets are not written at all, hence the ``None``
    handling by the caller.
    """
    partition = group["partition"][()]
    begin = int(partition[rank])
    end = int(partition[rank + 1])
    return group["data"][begin:end]


def _enumerate_level(cells, offsets, dim):
    """Enumerate integer cell indices of a single level, in ``for_each_cell`` order.

    ``cells[d]`` is the structured ``Interval`` array of ``m_cells[d]`` (fields
    ``start``, ``end``, ``step``, ``index``).  ``offsets[d]`` (``d >= 1``) is
    ``m_offsets[d-1]``: for an interval ``iv`` of direction ``d`` and a coordinate
    ``c`` in ``[iv.start, iv.end)``, the attached intervals of direction ``d-1``
    span ``offsets[d][iv.index + c] .. offsets[d][iv.index + c + 1]``.

    Returns an ``(n_cells, dim)`` int array of the ``(i[, j[, k]])`` indices.
    """
    x_iv = cells[0]
    if x_iv is None or len(x_iv) == 0:
        return np.zeros((0, dim), dtype=np.int64)

    x_start = x_iv["start"].astype(np.int64)
    x_end = x_iv["end"].astype(np.int64)

    def expand_rows(rows):
        """rows: list of (xi_begin, xi_end, extra_indices).  Vectorized x-expansion."""
        i_parts = []
        extra_parts = [] if rows and len(rows[0][2]) else None
        for xb, xe, extra in rows:
            for xi in range(xb, xe):
                rng = np.arange(x_start[xi], x_end[xi], dtype=np.int64)
                i_parts.append(rng)
                if extra_parts is not None:
                    extra_parts.append(np.tile(extra, (rng.size, 1)))
        if not i_parts:
            return np.zeros((0, dim), dtype=np.int64)
        i_col = np.concatenate(i_parts).reshape(-1, 1)
        if extra_parts is None:
            return i_col
        return np.hstack([i_col, np.concatenate(extra_parts)])

    if dim == 1:
        return expand_rows([(0, len(x_iv), np.array([], dtype=np.int64))])

    if dim == 2:
        off0 = offsets[1].astype(np.int64)  # m_offsets[0]: y -> x-interval range
        y_iv = cells[1]
        rows = []
        for yiv in y_iv:
            base = np.int64(yiv["index"])
            for y in range(int(yiv["start"]), int(yiv["end"])):
                xb = off0[base + y]
                xe = off0[base + y + 1]
                rows.append((xb, xe, np.array([y], dtype=np.int64)))
        return expand_rows(rows)

    if dim == 3:
        off0 = offsets[1].astype(np.int64)  # m_offsets[0]: y -> x-interval range
        off1 = offsets[2].astype(np.int64)  # m_offsets[1]: z -> y-interval range
        y_iv = cells[1]
        z_iv = cells[2]
        rows = []
        for ziv in z_iv:
            zbase = np.int64(ziv["index"])
            for z in range(int(ziv["start"]), int(ziv["end"])):
                yb = off1[zbase + z]
                ye = off1[zbase + z + 1]
                for yi in range(yb, ye):
                    yiv = y_iv[yi]
                    ybase = np.int64(yiv["index"])
                    for y in range(int(yiv["start"]), int(yiv["end"])):
                        xb = off0[ybase + y]
                        xe = off0[ybase + y + 1]
                        rows.append((xb, xe, np.array([y, z], dtype=np.int64)))
        return expand_rows(rows)

    raise ValueError(f"unsupported dimension {dim}")


def _read_rank_indices(mesh, dim, min_level, max_level, rank):
    """Rebuild the ``(indices, levels)`` of every cell owned by ``rank``.

    Cells are concatenated level by level (min->max), matching the order in which
    fields were flattened by ``extract_data_as_vector``.
    """
    all_indices = []
    all_levels = []
    for level in range(min_level, max_level + 1):
        level_group = mesh.get(f"level/{level}")
        if level_group is None:
            continue
        cells = {}
        offsets = {}
        empty = False
        for d in range(dim):
            iv_group = level_group.get(f"dim/{d}/intervals")
            if iv_group is None:
                empty = True
                break
            cells[d] = _slice_partitioned(iv_group, rank)
            if d >= 1:
                off_group = level_group.get(f"dim/{d}/offsets")
                offsets[d] = _slice_partitioned(off_group, rank)
        if empty or cells[0] is None or len(cells[0]) == 0:
            continue
        idx = _enumerate_level(cells, offsets, dim)
        if idx.shape[0] == 0:
            continue
        all_indices.append(idx)
        all_levels.append(np.full(idx.shape[0], level, dtype=np.int64))

    if not all_indices:
        return np.zeros((0, dim), dtype=np.int64), np.zeros((0,), dtype=np.int64)
    return np.concatenate(all_indices), np.concatenate(all_levels)


def _build_geometry(indices, levels, origin_point, scaling_factor, dim):
    """Build ``(points, connectivity, centers)`` for the given cells.

    Points are emitted per cell (no deduplication): simple and correct.  Points
    are always 3D (zero padding for ``dim < 3``) as expected by VTK.
    """
    n_cells = indices.shape[0]
    n_vertices = 1 << dim
    unit = _UNIT_ELEMENT[dim]  # (n_vertices, dim)

    length = (scaling_factor / (2.0 ** levels)).reshape(-1, 1)  # (n_cells, 1)
    origin = np.asarray(origin_point, dtype=float).reshape(1, dim)

    corner = origin + length * indices  # (n_cells, dim)
    centers3 = np.zeros((n_cells, 3), dtype=float)
    centers3[:, :dim] = corner + 0.5 * length

    # vertices[c, k, :] = corner[c] + length[c] * unit[k]
    verts = corner[:, None, :] + length[:, None, :] * unit[None, :, :]  # (n_cells, nv, dim)
    points = np.zeros((n_cells * n_vertices, 3), dtype=float)
    points[:, :dim] = verts.reshape(-1, dim)

    connectivity = np.arange(n_cells * n_vertices, dtype=np.int64).reshape(n_cells, n_vertices)
    return points, connectivity, centers3


def load(filename, extra_arrays=True):
    """Read a samurai ``.h5`` file and reconstruct the finite-element mesh.

    Parameters
    ----------
    filename : str
        Path to the samurai file (with or without the ``.h5`` extension).
    extra_arrays : bool
        Also expose ``level``, ``indices`` and ``center`` as cell arrays
        (computed for free during reconstruction).

    Returns
    -------
    list of dict
        One block per MPI rank.  Each block has ``dim``, ``vtk_cell_type``,
        ``points`` (N, 3), ``connectivity`` (n_cells, 2**dim), ``fields`` (a
        ``{name: (n_cells, n_comp) array}`` mapping) and, if requested, the extra
        cell arrays.
    """
    if not filename.endswith(".h5"):
        filename = filename + ".h5"

    with h5py.File(filename, "r") as f:
        if "mesh/dim" not in f:
            # Give an actionable message instead of a raw KeyError on 'dim'.
            looks_like_save = ("mesh" in f) and any(
                key in f["mesh"] for key in ("points", "connectivity")
            )
            if not looks_like_save and "mesh" in f:
                # multi-rank save() nests points/connectivity under rank groups
                looks_like_save = any(
                    isinstance(f["mesh"][k], h5py.Group)
                    and ("points" in f["mesh"][k] or "connectivity" in f["mesh"][k])
                    for k in f["mesh"].keys()
                )
            if looks_like_save:
                raise ValueError(
                    f"'{filename}' looks like a samurai save() file (explicit "
                    "points/connectivity mesh), which this reader does not handle. "
                    "Open the compressed dump/restart file instead."
                )
            raise ValueError(
                f"'{filename}' is not a samurai dump file: '/mesh/dim' is missing."
            )

        mesh = f["mesh"]
        dim = int(f["mesh/dim"][()])
        min_level = int(f["mesh/min_level"][()])
        max_level = int(f["mesh/max_level"][()])
        origin_point = np.asarray(f["mesh/origin_point"][()], dtype=float).reshape(-1)[:dim]
        scaling_factor = float(f["mesh/scaling_factor"][()])
        n_process = int(f["n_process"][()]) if "n_process" in f else 1

        # Field metadata (n_comp) and partitioned data groups.
        field_meta = {}
        if "fields" in f:
            for name in f["fields"].keys():
                n_comp = int(f[f"fields/{name}/n_comp"][()])
                field_meta[name] = n_comp

        blocks = []
        for rank in range(n_process):
            indices, levels = _read_rank_indices(mesh, dim, min_level, max_level, rank)
            n_cells = indices.shape[0]
            points, connectivity, centers = _build_geometry(
                indices, levels, origin_point, scaling_factor, dim
            )

            fields = {}
            for name, n_comp in field_meta.items():
                data_group = f[f"fields/{name}/data"]
                flat = _slice_partitioned(data_group, rank)
                values = np.asarray(flat).reshape(-1, n_comp)
                if values.shape[0] != n_cells:
                    raise ValueError(
                        f"field '{name}' rank {rank}: {values.shape[0]} values for "
                        f"{n_cells} reconstructed cells"
                    )
                fields[name] = values

            block = {
                "dim": dim,
                "rank": rank,
                "vtk_cell_type": _VTK_CELL_TYPE[dim],
                "points": points,
                "connectivity": connectivity,
                "fields": fields,
            }
            if extra_arrays:
                block["level"] = levels.astype(np.int32)
                block["indices"] = indices.astype(np.int32)
                block["center"] = centers
            blocks.append(block)

    return blocks
