# Copyright 2018-2025 the samurai's authors
# SPDX-License-Identifier:  BSD-3-Clause
#
# Validation of the ghost-update case catalog and its visualisation.
#
# The parallel ghost-update robustness suite (tests/mpi/test_ghost_update_parallel.cpp)
# and the demos/mpi/ghost_cases visualisation tool share a single catalog of
# cases (tests/mpi/ghost_cases.hpp). This test guards that shared contract:
#
#   1. "the real cases": the set of cases the demo dumps to disk (and therefore
#      the set one inspects / renders) is EXACTLY the set of parameterized cases
#      the GoogleTest suite instantiates. If a case is added or removed on one
#      side only, this fails - the visualisation can never silently drift from
#      what the test validates.
#
#   2. "unchanged images": the suite overview contact sheets are compared, via
#      pytest-mpl, against baseline PNGs committed under tests/reference. A
#      change in the geometries or the decompositions changes the picture and
#      fails the comparison. Text is stripped (remove_text=True) so the check is
#      robust to font rendering across platforms; run with --mpl to enable it,
#      and regenerate baselines with
#        pytest test_ghost_cases.py --mpl-generate-path=reference/ghost_cases
#
# Everything is skipped (not failed) when MPI, the executables, or the render
# dependencies are missing, so the module is a no-op on a build configured
# without WITH_MPI or on a bare Python environment.

import importlib.util
import os
import re
import shutil
import subprocess
from pathlib import Path

import pytest

pytestmark = pytest.mark.mpi

REPO = Path(__file__).resolve().parent.parent
RENDER = REPO / "demos" / "mpi" / "ghost_cases" / "render.py"
BASELINE_DIR = "reference/ghost_cases"  # relative to this test file

# Number of MPI ranks used to dump the cases. The baseline images are generated
# at this rank count; the decomposition (and thus the picture) is deterministic
# for a given count, so comparison must use the same one.
NP = 4

# gtest suite name (after the INSTANTIATE_TEST_SUITE_P prefix) -> dump/file prefix.
SUITE_PREFIX = {
    "ghost_update_2d": "A_2d_",
    "ghost_independence_2d": "B_2d_",
    "ghost_independence_3d": "B_3d_",
}


def _mpiexec():
    return shutil.which("mpiexec") or shutil.which("mpirun")


def _find(rel_dir, name):
    """Locate a built executable across the usual build trees.

    Honours $SAMURAI_MPI_BUILD_DIR first (CI points it at its build dir), then
    falls back to the in-tree build directories. Tests run from tests/, hence
    the ``..`` prefixes.
    """
    candidates = []
    env_dir = os.environ.get("SAMURAI_MPI_BUILD_DIR")
    if env_dir:
        candidates.append(Path(env_dir) / rel_dir)
    for build in ("build", "build-mpi"):
        candidates.append(Path("..") / build / rel_dir)
    for d in candidates:
        for exe in (d / name, d / "Release" / name):
            if exe.exists():
                return exe.resolve()
    return None


def _load_render_module():
    """Import demos/mpi/ghost_cases/render.py (not on sys.path) as a module.
    Returns None if its dependencies (matplotlib, h5py) are unavailable."""
    try:
        spec = importlib.util.spec_from_file_location("ghost_cases_render", RENDER)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    except Exception:
        return None


_MPIEXEC = _mpiexec()
_DUMP_EXE = _find(Path("demos") / "mpi", "mpi-ghost-cases")
_TEST_EXE = _find(Path("tests") / "mpi", "test_ghost_update_parallel")
_GCR = _load_render_module()

_skip_reason = None
if _MPIEXEC is None:
    _skip_reason = "mpiexec/mpirun not found"
elif _DUMP_EXE is None:
    _skip_reason = "mpi-ghost-cases not built (configure with WITH_MPI=ON)"
elif _TEST_EXE is None:
    _skip_reason = "test_ghost_update_parallel not built (configure with WITH_MPI=ON)"

if _skip_reason is not None:
    pytest.skip(_skip_reason, allow_module_level=True)

_need_render = pytest.mark.skipif(_GCR is None, reason="matplotlib/h5py not importable")


def _gtest_case_labels():
    """Map each dump prefix to the set of parameter labels the gtest suite runs.

    Parses ``--gtest_list_tests``; suite header lines end with '.', indented
    lines are ``<test>/<param>  # GetParam()...``; the param label is the token
    after the last '/'.
    """
    res = subprocess.run([str(_TEST_EXE), "--gtest_list_tests"], capture_output=True, text=True)
    assert res.returncode == 0, f"could not list tests:\n{res.stdout}\n{res.stderr}"

    labels = {p: set() for p in SUITE_PREFIX.values()}
    prefix = None
    for line in res.stdout.splitlines():
        if not line.strip():
            continue
        if not line.startswith(" "):
            # suite header, e.g. "all/ghost_update_2d."
            suite = line.strip().rstrip(".").split("/")[-1]
            prefix = SUITE_PREFIX.get(suite)
            continue
        if prefix is None:
            continue
        token = line.strip().split("#")[0].strip()
        param = token.split("/")[-1]
        labels[prefix].add(param)
    return labels


@pytest.fixture(scope="module")
def dumped(tmp_path_factory):
    """Run mpi-ghost-cases once (full matrix, NP ranks) into a shared temp dir."""
    out = tmp_path_factory.mktemp("ghost_cases")
    res = subprocess.run(
        [_MPIEXEC, "-n", str(NP), str(_DUMP_EXE), "--path", str(out)],
        capture_output=True,
        text=True,
    )
    assert res.returncode == 0, f"mpi-ghost-cases failed:\n{res.stdout}\n{res.stderr}"
    return out


def _dumped_labels(out_dir):
    """Prefix -> set of case labels present as <prefix><label>_np<K>.h5 files."""
    found = {p: set() for p in SUITE_PREFIX.values()}
    for h5 in out_dir.glob("*.h5"):
        stem = re.sub(r"_np\d+$", "", h5.stem)
        for prefix in SUITE_PREFIX.values():
            if stem.startswith(prefix):
                found[prefix].add(stem[len(prefix):])
                break
    return found


def _2d_files(out_dir):
    return sorted(
        str(f) for f in out_dir.glob("*_2d_*.h5") if f.name.startswith(("A_2d_", "B_2d_"))
    )


def test_dumped_cases_match_test_matrix(dumped):
    """The cases written to disk must be exactly those the gtest suite runs."""
    expected = _gtest_case_labels()
    found = _dumped_labels(dumped)

    # sanity: the matrix is non-trivial
    assert sum(len(v) for v in expected.values()) > 0, "no gtest cases parsed"

    for prefix in SUITE_PREFIX.values():
        missing = expected[prefix] - found[prefix]
        extra = found[prefix] - expected[prefix]
        assert not missing, f"{prefix}: cases in the test but not dumped: {sorted(missing)}"
        assert not extra, f"{prefix}: cases dumped but not in the test: {sorted(extra)}"


@_need_render
@pytest.mark.mpl_image_compare(baseline_dir=BASELINE_DIR, filename="overview_suite_A.png", remove_text=True, style="default", tolerance=15)
def test_overview_suite_A(dumped):
    """Suite A overview (geometry x decomposition, coloured by rank) is unchanged."""
    return _GCR.overview_figure(_2d_files(dumped), "A", NP)


@_need_render
@pytest.mark.mpl_image_compare(baseline_dir=BASELINE_DIR, filename="overview_suite_B.png", remove_text=True, style="default", tolerance=15)
def test_overview_suite_B(dumped):
    """Suite B overview (geometry x decomposition, coloured by rank) is unchanged."""
    return _GCR.overview_figure(_2d_files(dumped), "B", NP)
