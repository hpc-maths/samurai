import os
import pytest
import subprocess
from pathlib import Path

path = "lbm"


@pytest.fixture
def config():
    return {"path": path}


def get_executable(path, filename):
    if os.path.exists(os.path.join(path, filename)):
        return os.path.join(path, filename)
    return os.path.join(path, "Release", filename)


def run_demo(exec, config, extra):
    cmd = [
        get_executable(Path("../build/demos/LBM/"), exec),
        "--path",
        config["path"],
        "--filename",
        config["filename"],
    ] + extra
    subprocess.run(cmd, check=True, capture_output=True)


# ------------------------------------------------------------------ D1Q2 (single block)
@pytest.mark.h5diff()
@pytest.mark.parametrize(
    "case, extra",
    [
        ("advection", ["--level", "7", "--Tf", "0.2"]),
        ("burgers", ["--burgers", "--level", "7", "--Tf", "0.2"]),
        ("advection_adaptive", ["--adapt", "--level", "8", "--min-lvl", "3", "--eps", "1e-4", "--Tf", "0.2"]),
    ],
)
def test_lbm_demo_d1q2(case, extra, config):
    run_demo("lbm-new-D1Q2-advection", config, extra)


# ------------------------------------------------------------------ D2Q4 stream (N-D)
@pytest.mark.h5diff()
def test_lbm_demo_d2q4_axial(config):
    run_demo("lbm-new-D2Q4-advection", config, ["--level", "5", "--Tf", "0.2"])


@pytest.mark.h5diff()
@pytest.mark.parametrize(
    "case, extra",
    [
        ("uniform", ["--level", "5", "--Tf", "0.2"]),
        ("adaptive", ["--adapt", "--level", "7", "--min-lvl", "2", "--eps", "1e-4", "--Tf", "0.2"]),
    ],
)
def test_lbm_demo_d2q4_diagonal(case, extra, config):
    run_demo("lbm-new-D2Q4diag-advection", config, extra)


# ------------------------------------------------------------------ D1Q3 + wall BC
@pytest.mark.h5diff()
def test_lbm_demo_d1q3_bounce_back(config):
    run_demo(
        "lbm-new-D1Q3-shallow-waters-dam",
        config,
        ["--bc", "bounceback", "--level", "7", "--hL", "2", "--hR", "1", "--Tf", "0.5"],
    )


@pytest.mark.h5diff()
def test_lbm_demo_d1q3_anti_bounce_back(config):
    run_demo(
        "lbm-new-D1Q3-shallow-waters-dam",
        config,
        ["--bc", "antibounceback", "--level", "7", "--hL", "1.2", "--hR", "1.0", "--Tf", "1.0"],
    )


# ------------------------------------------------------------------ D2Q9 MRT Navier-Stokes
@pytest.mark.h5diff()
def test_lbm_demo_d2q9_taylor_green(config):
    run_demo(
        "lbm-new-D2Q9-taylor-green",
        config,
        ["--level", "5", "--U0", "0.05", "--nu", "0.02", "--Tf", "2"],
    )


# ------------------------------------------------------------------ D1Q5 |c| > 1 stream
@pytest.mark.h5diff()
@pytest.mark.parametrize(
    "case, extra",
    [
        ("uniform", ["--level", "7", "--Tf", "0.2"]),
        ("adaptive", ["--adapt", "--level", "8", "--min-lvl", "3", "--eps", "1e-4", "--Tf", "0.2"]),
    ],
)
def test_lbm_demo_d1q5_dam(case, extra, config):
    run_demo("lbm-new-D1Q5-shallow-waters-dam", config, extra)


# ------------------------------------------------------------------ D1Q222 Euler (multi-block)
@pytest.mark.h5diff()
@pytest.mark.parametrize(
    "case, extra",
    [
        ("uniform", ["--level", "8", "--Tf", "0.2"]),
        ("adaptive", ["--adapt", "--level", "9", "--min-lvl", "3", "--eps", "1e-3", "--Tf", "0.2"]),
    ],
)
def test_lbm_demo_d1q222_sod(case, extra, config):
    run_demo("lbm-new-D1Q222-euler-sod", config, extra)


# ------------------------------------------------------------------ D2Q4444 Euler (multi-block, 2D)
@pytest.mark.h5diff()
@pytest.mark.parametrize(
    "case, extra",
    [
        ("uniform", ["--level", "5", "--Tf", "0.1"]),
        ("adaptive", ["--adapt", "--level", "6", "--min-lvl", "2", "--eps", "1e-3", "--Tf", "0.1"]),
    ],
)
def test_lbm_demo_d2q4444_lax_liu(case, extra, config):
    run_demo("lbm-new-D2Q4444-euler-lax-liu", config, extra)


# ------------------------------------------------------------------ D2Q4444 Euler reflecting wall
@pytest.mark.h5diff()
@pytest.mark.parametrize(
    "case, extra",
    [
        ("uniform", ["--level", "5", "--Tf", "0.3"]),
        ("adaptive", ["--adapt", "--level", "6", "--min-lvl", "2", "--eps", "1e-3", "--Tf", "0.3"]),
    ],
)
def test_lbm_demo_d2q4444_implosion(case, extra, config):
    run_demo("lbm-new-D2Q4444-euler-implosion", config, extra)
