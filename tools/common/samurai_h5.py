# Copyright 2018-2025 the samurai's authors
# SPDX-License-Identifier:  BSD-3-Clause
"""
File-level helpers shared by the samurai readers in ``tools/``.

The ParaView reader (``tools/paraview``) and the yt loader (``tools/yt``)
rebuild the mesh differently, but they open samurai files the same way: same
layout check, same partitioned datasets, same time attribute, same numbered
series on disk.  That common part lives here so both readers stay in sync.

This module only depends on ``numpy`` and ``h5py``.  Each reader's
``samurai_load.py`` puts this directory on ``sys.path`` relative to its own
location, so the ``tools/`` tree must be kept intact.
"""

import os
import re

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

    match = re.match(r"^(.*?)(?<!\d)(\d+)$", base)
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


def slice_partitioned(group, rank):
    """Return the ``rank``-th slice of a samurai partitioned dataset.

    A partitioned dataset is a group holding ``data`` (the concatenated buffer of
    every MPI rank) and ``partition`` (cumulative per-rank sizes, length
    ``n_process + 1``); see ``dump(file, name, std::vector<T>)`` in
    ``restart.hpp``.  Empty datasets are not written at all, so callers must
    check that the group exists first.
    """
    partition = group["partition"][()]
    begin = int(partition[rank])
    end = int(partition[rank + 1])
    return group["data"][begin:end]


def validate_samurai_group(root, filename):
    """Raise a clear error if ``root`` is not a samurai mesh group."""
    if "mesh/dim" in root:
        return
    # Detect a save() file (explicit points/connectivity, possibly nested under
    # per-rank groups) to give an actionable message instead of a raw KeyError.
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
            "points/connectivity mesh), which the samurai readers do not handle. "
            "Open the compressed dump/restart file instead."
        )
    raise ValueError(f"'{filename}' is not a samurai dump file: '/mesh/dim' is missing.")
