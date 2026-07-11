# Copyright 2018-2025 the samurai's authors
# SPDX-License-Identifier:  BSD-3-Clause
"""
Validate the samurai yt loader (tools/yt/) against small reference files.

The reference files are produced by tools/yt/tests/generate_reference.cpp.
The reference field is the analytic oracle ``u(center) = sum_d
center[d]*10**d``. Checking ``u == u(reconstructed center)`` for every cell
validates, at once, the geometry reconstruction and the (offset-driven)
``for_each_cell`` traversal order on which the flat field buffer relies --
first at the patch level (pure numpy / h5py, no yt) and then end-to-end
through a loaded yt dataset.
"""

import os
import sys

import numpy as np
import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
_PLUGIN_DIR = os.path.join(_HERE, "..", "tools", "yt")
sys.path.insert(0, _PLUGIN_DIR)
_REF_DIR = os.path.join(_HERE, "reference", "yt")

h5py = pytest.importorskip("h5py")
from samurai_load import discover_series, read_amr_grids, read_cells  # noqa: E402


def analytic(centers, dim):
    weights = 10.0 ** np.arange(dim)
    return centers[:, :dim] @ weights


# --------------------------------------------------------------------------- #
# Pure reconstruction (numpy + h5py, no yt)                                    #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("name,dim", [("ref_1d", 1), ("ref_2d", 2), ("ref_3d", 3)])
def test_cell_reconstruction(name, dim):
    # Every leaf must carry the analytic value of its own center: validates the
    # geometry and the for_each_cell traversal order at once.
    centers, levels, fields, meta = read_cells(os.path.join(_REF_DIR, f"{name}.h5"))
    assert meta["dim"] == dim
    n_cells = centers.shape[0]
    assert n_cells > 0
    assert levels.shape == (n_cells,)
    np.testing.assert_allclose(fields["u"], analytic(centers, dim), atol=1e-9)


@pytest.mark.parametrize("name,dim", [("ref_1d", 1), ("ref_2d", 2), ("ref_3d", 3)])
def test_amr_grid_structure(name, dim):
    grids, meta = read_amr_grids(os.path.join(_REF_DIR, f"{name}.h5"))
    assert len(grids) >= 1
    dd = meta["domain_dimensions"]
    assert dd[dim:].tolist() == [1] * (3 - dim)  # unused dims collapse to one cell
    assert (dd[:dim] > 0).all()

    # The root grid tiles the domain at level 0; refined grids sit above it.
    root = grids[0]
    assert root["level"] == 0
    np.testing.assert_array_equal(root["dimensions"], dd)
    child_dims = [2 if d < dim else 1 for d in range(3)]
    for grid in grids[1:]:
        assert grid["level"] >= 1
        assert grid["dimensions"].tolist() == child_dims
        assert grid["u"].shape == tuple(child_dims)


def test_uniform_mesh_root_grid():
    # A uniform min_level mesh: every cell is a leaf on the root grid, so the
    # reconstruction must be a single level-0 grid full of real (non-masked)
    # values -- the one path the graded meshes never exercise.
    grids, meta = read_amr_grids(os.path.join(_REF_DIR, "ref_uniform.h5"))
    assert len(grids) == 1
    assert grids[0]["level"] == 0

    centers, levels, fields, _ = read_cells(os.path.join(_REF_DIR, "ref_uniform.h5"))
    assert set(levels.tolist()) == {meta["min_level"]}
    np.testing.assert_allclose(fields["u"], analytic(centers, 2), atol=1e-9)


def test_mpi_reconstruction():
    ref = os.path.join(_REF_DIR, "ref_2d_mpi.h5")
    if not os.path.exists(ref):
        pytest.skip("ref_2d_mpi.h5 not generated (requires an MPI-enabled build)")
    centers, _, fields, _ = read_cells(ref)
    np.testing.assert_allclose(fields["u"], analytic(centers, 2), atol=1e-9)

    # Ranks partition the mesh: every cell appears exactly once.
    keys = {tuple(np.round(c, 9)) for c in centers}
    assert len(keys) == centers.shape[0]


def test_load_from_subgroup(tmp_path):
    # A file may hold the samurai layout under a subgroup; read_cells(group=...)
    # must reconstruct it identically to a root-level file.
    ref = os.path.join(_REF_DIR, "ref_2d.h5")
    nested = tmp_path / "multigrid.h5"
    with h5py.File(ref, "r") as src, h5py.File(nested, "w") as dst:
        grid = dst.create_group("grids/g0")
        for key in ("mesh", "fields", "n_process"):
            src.copy(key, grid)

    root_centers, _, root_fields, _ = read_cells(ref)
    nested_centers, _, nested_fields, _ = read_cells(str(nested), group="grids/g0")
    np.testing.assert_array_equal(nested_centers, root_centers)
    np.testing.assert_array_equal(nested_fields["u"], root_fields["u"])


def test_rejects_save_file(tmp_path):
    path = tmp_path / "explicit.h5"
    with h5py.File(path, "w") as f:
        grp = f.create_group("mesh")
        grp.create_dataset("points", data=np.zeros((4, 3)))
        grp.create_dataset("connectivity", data=np.zeros((1, 4), dtype=np.int64))
    with pytest.raises(ValueError, match="save"):
        read_cells(str(path))


def test_rejects_non_samurai_file(tmp_path):
    path = tmp_path / "empty.h5"
    with h5py.File(path, "w"):
        pass
    with pytest.raises(ValueError, match="not a samurai dump"):
        read_cells(str(path))


def test_discover_series(tmp_path):
    for n in (0, 1, 2, 10):
        (tmp_path / f"sol_ite_{n}.h5").write_bytes(b"")
    (tmp_path / "other.h5").write_bytes(b"")  # unrelated, must be ignored

    files, times = discover_series(str(tmp_path / "sol_ite_1.h5"))
    assert times == [0.0, 1.0, 2.0, 10.0]  # natural order: 2 before 10
    assert [os.path.basename(f) for f in files] == [
        "sol_ite_0.h5", "sol_ite_1.h5", "sol_ite_2.h5", "sol_ite_10.h5"
    ]


def test_discover_series_uses_time_attribute(tmp_path):
    physical_times = [0.0, 0.05, 0.1]
    for n, t in enumerate(physical_times):
        with h5py.File(tmp_path / f"sol_ite_{n}.h5", "w") as f:
            f.attrs["time"] = t
    _, times = discover_series(str(tmp_path / "sol_ite_0.h5"))
    assert times == physical_times


# --------------------------------------------------------------------------- #
# End-to-end through yt                                                        #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "name,dim", [("ref_1d", 1), ("ref_2d", 2), ("ref_3d", 3), ("ref_uniform", 2)]
)
def test_yt_dataset(name, dim):
    pytest.importorskip("yt")
    from samurai_yt import load_samurai

    ds = load_samurai(os.path.join(_REF_DIR, f"{name}.h5"))
    assert int(ds.dimensionality) == dim

    ad = ds.all_data()
    coords = ["x", "y", "z"]
    centers = np.stack(
        [np.asarray(ad["index", coords[d]]) for d in range(dim)], axis=1
    )
    u = np.asarray(ad["stream", "u"])
    assert u.shape[0] > 0
    # Reconstructed leaves carry the analytic value of their own centers: this
    # cross-checks geometry and traversal order end-to-end through yt.
    np.testing.assert_allclose(u, analytic(centers, dim), atol=1e-9)


def test_yt_time():
    pytest.importorskip("yt")
    from samurai_yt import load_samurai

    ref = os.path.join(_REF_DIR, "ref_2d.h5")
    ds = load_samurai(ref)
    with h5py.File(ref, "r") as f:
        expected = float(f.attrs["time"]) if "time" in f.attrs else 0.0
    assert float(ds.current_time) == pytest.approx(expected)


def test_yt_fields_are_linear_by_default():
    # PDE fields must default to a linear scale: yt otherwise guesses a log
    # scale and turns tiny over/undershoots into visual noise (issue seen on
    # the advection demo).
    pytest.importorskip("yt")
    from samurai_yt import load_samurai

    ds = load_samurai(os.path.join(_REF_DIR, "ref_2d.h5"))
    assert ds.field_info["stream", "u"].take_log is False

    ds_log = load_samurai(os.path.join(_REF_DIR, "ref_2d.h5"), take_log=True)
    assert ds_log.field_info["stream", "u"].take_log is True
