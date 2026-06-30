# Copyright 2018-2025 the samurai's authors
# SPDX-License-Identifier:  BSD-3-Clause
#
# Comparative integration tests for the load balancing module (roadmap step 6).
#
# The MPI demo ``mpi-load-balancing-2d`` advects a disk on an adaptive mesh:
# the work per process drifts at every adaptation, which is exactly what the
# balancing strategies are meant to absorb. The contract these tests enforce
# is the one stated in the roadmap: **load balancing must never change the
# physics, only the distribution of cells across processes**. We therefore:
#
#   1. run the sequential baseline (``void`` on a single process) as reference;
#   2. run every available strategy on several process counts;
#   3. assert the final field is identical to the reference (machine tolerance,
#      via ``python/compare.py`` which matches cells by coordinates and is thus
#      insensitive to the decomposition);
#   4. assert the strategy actually balanced the load at least once.
#
# Everything is skipped (not failed) when MPI or the demo executable is absent,
# so the suite is a no-op on a build configured without ``WITH_MPI``.

import csv
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

# Strategies always compiled in. metis/scotch are added at runtime only if the
# demo was built with the corresponding option (detected by trying to run them).
ALWAYS_AVAILABLE = ["sfc-morton", "sfc-hilbert", "diffusion"]
OPTIONAL = ["metis", "scotch"]

# Physical horizon and rebalance period, per dimension. They must be identical
# between the reference and every strategy run. The 3D mesh is coarser
# (max_level 6 vs 10), so it advances in far fewer time steps for a given final
# time; the horizon is stretched and the period shortened so that the
# incremental Diffusion strategy gets enough passes to converge (it balances
# asymptotically, see the roadmap).
TF = {2: "0.02", 3: "0.06"}
PERIOD = {2: "5", 3: "2"}


def _mpiexec():
    return shutil.which("mpiexec") or shutil.which("mpirun")


def _find_executable(dim):
    """Locate mpi-load-balancing-{dim}d across the usual build trees.

    Honours $SAMURAI_MPI_BUILD_DIR first so CI can point at its build, then
    falls back to the conventional in-tree build directories (tests are run
    from the ``tests/`` directory, hence the ``..`` prefixes).
    """
    name = f"mpi-load-balancing-{dim}d"
    candidates = []
    env_dir = os.environ.get("SAMURAI_MPI_BUILD_DIR")
    if env_dir:
        candidates += [Path(env_dir) / "demos" / "mpi"]
    for build in ("build", "build-mpi"):
        candidates += [Path("..") / build / "demos" / "mpi"]
    for d in candidates:
        for exe in (d / name, d / "Release" / name):
            if exe.exists():
                return exe.resolve()
    return None


def _compare_script():
    here = Path(__file__).resolve().parent
    script = here.parent / "python" / "compare.py"
    return script if script.exists() else None


# Module-level guard: turn the whole file into a clean skip when the
# prerequisites are missing, instead of failing every test.
pytestmark = pytest.mark.mpi

_MPIEXEC = _mpiexec()
_COMPARE = _compare_script()
_EXE = {dim: _find_executable(dim) for dim in (2, 3)}

_skip_reason = None
if _MPIEXEC is None:
    _skip_reason = "mpiexec/mpirun not found"
elif _COMPARE is None:
    _skip_reason = "python/compare.py not found"
elif all(exe is None for exe in _EXE.values()):
    _skip_reason = "mpi-load-balancing-{2,3}d not built (configure with WITH_MPI=ON)"

if _skip_reason is not None:
    pytest.skip(_skip_reason, allow_module_level=True)


def _run_demo(dim, strategy, nprocs, out_dir, filename, stats_file=None):
    """Run the {dim}d demo; return the CompletedProcess (text mode)."""
    cmd = [
        _MPIEXEC,
        "-n",
        str(nprocs),
        str(_EXE[dim]),
        "--lb-strategy",
        strategy,
        "--Tf",
        TF[dim],
        "--nt-loadbalance",
        PERIOD[dim],
        "--path",
        str(out_dir),
        "--filename",
        filename,
    ]
    if stats_file is not None:
        cmd += ["--lb-stats-file", str(stats_file)]
    # Inherit the environment so machine-specific MPI tuning (e.g. FI_PROVIDER)
    # set by the caller is honoured; we deliberately do not hardcode any.
    return subprocess.run(cmd, capture_output=True, text=True)


@pytest.fixture(scope="module")
def reference(tmp_path_factory):
    """Sequential ``void`` run per dimension: the physical reference."""
    refs = {}
    for dim, exe in _EXE.items():
        if exe is None:
            continue
        out = tmp_path_factory.mktemp(f"lb_ref_{dim}d")
        res = _run_demo(dim, "void", 1, out, "ref")
        prefix = out / "ref_size_1"
        if res.returncode == 0 and prefix.with_suffix(".h5").exists():
            refs[dim] = prefix
    return refs


def _available(dim, strategy):
    """metis/scotch are only usable if the demo was compiled with them."""
    if strategy in ALWAYS_AVAILABLE:
        return True
    res = _run_demo(dim, strategy, 1, Path("/tmp"), f"probe_{strategy}")
    return "unknown --lb-strategy" not in (res.stdout + res.stderr)


def _min_imbalance_after(stats_file):
    """Smallest imbalance_after recorded over the run (best balance achieved)."""
    with open(stats_file, newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return None
    return min(float(r["imbalance_after"]) for r in rows)


@pytest.mark.parametrize("dim", [2, 3])
@pytest.mark.parametrize("strategy", ALWAYS_AVAILABLE + OPTIONAL)
@pytest.mark.parametrize("nprocs", [2, 4])
def test_strategy_matches_sequential(dim, strategy, nprocs, reference, tmp_path):
    """Every strategy must reproduce the sequential physics and balance the load."""
    if _EXE[dim] is None:
        pytest.skip(f"mpi-load-balancing-{dim}d not built")
    if dim not in reference:
        pytest.skip(f"{dim}d reference run unavailable")
    if not _available(dim, strategy):
        pytest.skip(f"demo built without {strategy}")

    stats = tmp_path / "stats.csv"
    res = _run_demo(dim, strategy, nprocs, tmp_path, strategy, stats_file=stats)
    out_prefix = tmp_path / f"{strategy}_size_{nprocs}"
    assert res.returncode == 0, f"demo failed:\n{res.stdout}\n{res.stderr}"
    assert out_prefix.with_suffix(".h5").exists(), "no output produced"

    # 1. physical identity to the sequential reference (machine tolerance).
    cmp = subprocess.run(
        [sys.executable, str(_COMPARE), str(reference[dim]), str(out_prefix), "--tol", "1e-12"],
        capture_output=True,
        text=True,
    )
    assert cmp.returncode == 0, (
        f"{strategy} ({dim}d, np={nprocs}) diverged from the sequential reference:\n"
        f"{cmp.stdout}\n{cmp.stderr}"
    )

    # 2. the strategy actually balanced the load at some point.
    best = _min_imbalance_after(stats)
    assert best is not None, "no rebalance recorded"
    assert best < 0.1, f"{strategy} never reached imbalance < 0.1 (best={best:.3f})"
