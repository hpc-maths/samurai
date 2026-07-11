# Copyright 2018-2025 the samurai's authors
# SPDX-License-Identifier:  BSD-3-Clause
"""
Validate the samurai ParaView reader (tools/paraview/samurai_load.py) against small
reference files produced by tools/paraview/tests/generate_reference.cpp.

The reference field is the analytic oracle ``u(center) = sum_d center[d]*10**d``.
Checking ``u == u(reconstructed center)`` for every cell validates, at once, the
geometry reconstruction and the (offset-driven) ``for_each_cell`` traversal order
on which the flat field buffer relies.
"""

import os
import sys

import numpy as np
import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
_PLUGIN_DIR = os.path.join(_HERE, "..", "tools", "paraview")
sys.path.insert(0, _PLUGIN_DIR)
_REF_DIR = os.path.join(_HERE, "reference", "paraview")

h5py = pytest.importorskip("h5py")
from samurai_load import load, discover_series  # noqa: E402


def analytic(centers, dim):
    weights = 10.0 ** np.arange(dim)
    return centers[:, :dim] @ weights


@pytest.mark.parametrize("name,dim", [("ref_1d", 1), ("ref_2d", 2), ("ref_3d", 3)])
def test_sequential_reconstruction(name, dim):
    blocks = load(os.path.join(_REF_DIR, f"{name}.h5"))
    assert len(blocks) == 1
    block = blocks[0]

    n_cells = block["center"].shape[0]
    assert n_cells > 0
    assert block["dim"] == dim
    assert block["connectivity"].shape == (n_cells, 1 << dim)
    assert block["points"].shape == (n_cells * (1 << dim), 3)
    assert np.isfinite(block["points"]).all()

    u = block["fields"]["u"][:, 0]
    assert u.shape[0] == n_cells
    np.testing.assert_allclose(u, analytic(block["center"], dim), atol=1e-9)


def test_mpi_reconstruction():
    ref = os.path.join(_REF_DIR, "ref_2d_mpi.h5")
    if not os.path.exists(ref):
        # Only produced by generate_paraview_reference on an MPI-enabled build
        # (see .github/workflows/ci.yml, job linux-mpi-mamba).
        pytest.skip("ref_2d_mpi.h5 not generated (requires an MPI-enabled build)")
    blocks = load(ref)
    assert len(blocks) >= 2  # generated with mpirun -n 2

    centers = np.concatenate([b["center"] for b in blocks])
    u = np.concatenate([b["fields"]["u"][:, 0] for b in blocks])

    # Every cell carries the analytic value of its own center.
    np.testing.assert_allclose(u, analytic(centers, 2), atol=1e-9)

    # Ranks partition the mesh: no cell is shared or missing.
    keys = {tuple(np.round(c, 9)) for c in centers}
    assert len(keys) == centers.shape[0]


def test_extra_arrays_optional():
    blocks = load(os.path.join(_REF_DIR, "ref_2d.h5"), extra_arrays=False)
    block = blocks[0]
    assert "level" not in block
    assert "center" not in block
    assert "u" in block["fields"]


def test_load_from_subgroup(tmp_path):
    # A file may hold the samurai layout under a subgroup; load(group=...) must
    # reconstruct it identically to a root-level file.
    ref = os.path.join(_REF_DIR, "ref_2d.h5")
    nested = tmp_path / "multigrid.h5"
    with h5py.File(ref, "r") as src, h5py.File(nested, "w") as dst:
        grid = dst.create_group("grids/g0")
        for key in ("mesh", "fields", "n_process"):
            src.copy(key, grid)

    root_block = load(ref)[0]
    nested_block = load(str(nested), group="grids/g0")[0]
    np.testing.assert_array_equal(nested_block["center"], root_block["center"])
    np.testing.assert_array_equal(
        nested_block["fields"]["u"], root_block["fields"]["u"]
    )


def test_load_rejects_save_file(tmp_path):
    # A save() file (explicit points/connectivity, no /mesh/dim) must raise a
    # clear error rather than a raw KeyError.
    path = tmp_path / "explicit.h5"
    with h5py.File(path, "w") as f:
        grp = f.create_group("mesh")
        grp.create_dataset("points", data=np.zeros((4, 3)))
        grp.create_dataset("connectivity", data=np.zeros((1, 4), dtype=np.int64))
    with pytest.raises(ValueError, match="save"):
        load(str(path))


def test_load_rejects_non_samurai_file(tmp_path):
    path = tmp_path / "empty.h5"
    with h5py.File(path, "w"):
        pass
    with pytest.raises(ValueError, match="not a samurai dump"):
        load(str(path))


def test_discover_series(tmp_path):
    # A numbered series: opening ANY one file must expose all time steps.
    for n in (0, 1, 2, 10):
        (tmp_path / f"sol_ite_{n}.h5").write_bytes(b"")
    (tmp_path / "other.h5").write_bytes(b"")  # unrelated, must be ignored

    files, times = discover_series(str(tmp_path / "sol_ite_1.h5"))
    assert times == [0.0, 1.0, 2.0, 10.0]  # natural order: 2 before 10
    assert [os.path.basename(f) for f in files] == [
        "sol_ite_0.h5", "sol_ite_1.h5", "sol_ite_2.h5", "sol_ite_10.h5"
    ]


def test_discover_series_single_file(tmp_path):
    (tmp_path / "snapshot.h5").write_bytes(b"")
    files, times = discover_series(str(tmp_path / "snapshot.h5"))
    assert times == [0.0]
    assert len(files) == 1


def test_discover_series_uses_time_attribute(tmp_path):
    # When every file stores the "time" root attribute, it is used as the time
    # axis instead of the iteration index.
    physical_times = [0.0, 0.05, 0.1]
    for n, t in enumerate(physical_times):
        with h5py.File(tmp_path / f"sol_ite_{n}.h5", "w") as f:
            f.attrs["time"] = t
    files, times = discover_series(str(tmp_path / "sol_ite_0.h5"))
    assert [os.path.basename(f) for f in files] == [
        "sol_ite_0.h5", "sol_ite_1.h5", "sol_ite_2.h5"
    ]
    assert times == physical_times


def test_discover_series_falls_back_without_time(tmp_path):
    # If any file lacks the attribute, fall back to the iteration numbers.
    with h5py.File(tmp_path / "sol_ite_0.h5", "w") as f:
        f.attrs["time"] = 0.0
    with h5py.File(tmp_path / "sol_ite_1.h5", "w") as f:
        pass  # no time attribute
    files, times = discover_series(str(tmp_path / "sol_ite_0.h5"))
    assert times == [0.0, 1.0]
