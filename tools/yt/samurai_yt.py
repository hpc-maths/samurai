# Copyright 2018-2025 the samurai's authors
# SPDX-License-Identifier:  BSD-3-Clause
"""
Load a samurai data file (compressed HDF5) into `yt <https://yt-project.org>`_.

Samurai writes its state as a compact HDF5 file that stores only its internal
interval representation of the mesh (via ``samurai::dump``, read back with
``samurai::load``); no explicit geometry is stored.  Because that interval
representation *is* a patch-AMR description -- one contiguous x-run of cells per
patch -- it maps directly onto :func:`yt.load_amr_grids`, so the full yt toolbox
(slices, projections, profiles, volume rendering) works on samurai data.

Example
-------
>>> import samurai_yt
>>> ds = samurai_yt.load_samurai("solution.h5")
>>> yt.SlicePlot(ds, "z", ("stream", "u")).save()

The heavy lifting (reading the file, rebuilding the patches) lives in
:mod:`samurai_load`, which depends only on ``numpy`` and ``h5py`` and is
unit-testable without yt.
"""

import os
import sys

# Make ``samurai_load`` importable regardless of the caller's working directory.
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from samurai_load import discover_series, read_amr_grids  # noqa: E402


def load_samurai(filename, group=None, length_unit=1.0, take_log=False, **kwargs):
    """Load a single samurai ``.h5`` file as a yt ``StreamDataset``.

    Parameters
    ----------
    filename : str
        Path to the samurai file (with or without the ``.h5`` extension).
    group : str, optional
        HDF5 group under which the samurai layout lives (see
        :func:`samurai_load.read_amr_grids`).  Defaults to the file root.
    length_unit : float or str, optional
        Physical length of one code-length unit (default ``1.0``).  Samurai
        coordinates are used as ``code_length``.
    take_log : bool or None, optional
        Whether the fields should be shown on a log scale.  Defaults to ``False``
        (linear), which is the sensible default for the PDE solutions samurai
        produces -- otherwise yt guesses a log scale from the value range and
        turns small over/undershoots (and any negative value) into visual noise.
        Pass ``None`` to keep yt's own guess.
    **kwargs
        Forwarded to :func:`yt.load_amr_grids` (e.g. ``periodicity``,
        ``unit_system``).

    Returns
    -------
    yt.frontends.stream.data_structures.StreamDataset
        Scalar fields are exposed as ``("stream", name)``; each component of a
        vector field ``v`` is exposed as ``("stream", "v_<c>")``.
    """
    import yt

    grids, meta = read_amr_grids(filename, group=group)
    kwargs.setdefault("periodicity", (False, False, False))
    ds = yt.load_amr_grids(
        grids,
        meta["domain_dimensions"],
        length_unit=length_unit,
        bbox=meta["bbox"],
        sim_time=meta["time"] or 0.0,
        **kwargs,
    )
    if take_log is not None:
        for name in meta["fields"]:
            ds.field_info["stream", name].take_log = take_log
    return ds


def load_samurai_series(filename, **kwargs):
    """Load a numbered samurai series as a time-ordered list of datasets.

    Given any one file of a series (e.g. ``sol_ite_0.h5``), the siblings sharing
    the same prefix are discovered on disk (see
    :func:`samurai_load.discover_series`) and loaded in natural order.  Each
    dataset's ``current_time`` is the physical time stored in the file (or its
    iteration index as a fallback).

    Extra keyword arguments are forwarded to :func:`load_samurai`.
    """
    files, _ = discover_series(filename)
    return [load_samurai(f, **kwargs) for f in files]
