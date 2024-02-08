import os
import pytest
import subprocess
import sys
from pathlib import Path

path = 'finite_volume'

@pytest.fixture
def config():
    return {'path': path}

def get_executable(path, filename):
    if os.path.exists(os.path.join(path, filename)):
        return os.path.join(path, filename)
    return os.path.join(path, 'Release', filename)

@pytest.mark.h5diff()
@pytest.mark.parametrize(
    'exec, Tf',
    [
        ('finite-volume-advection-1d', '0.1'),
        ('finite-volume-advection-2d', '0.01'),
        ('finite-volume-scalar-burgers-2d', '0.001'),
        ('finite-volume-amr-burgers-hat', '1'),
        ('finite-volume-level-set', '0.1'),
        ('finite-volume-level-set-from-scratch', '0.1'),
    ]
)
def test_finite_volume_demo(exec, Tf, config):
    cmd = [get_executable(Path("../build/demos/FiniteVolume/"), exec),
           "--path", config['path'],
           '--filename', config['filename'],
           '--Tf', Tf]
    output = subprocess.run(cmd, check=True, capture_output=True)

@pytest.mark.h5diff()
@pytest.mark.skipif(sys.platform == "darwin", reason = "skipped on macos because libpthread is missing on github worker")
def test_finite_volume_demo_heat_explicit(config):
    cmd = [get_executable(Path("../build/demos/FiniteVolume/"), "finite-volume-heat"),
           "--path", config['path'],
           '--filename', config['filename'],
           "--save-final-state-only",
           "--explicit",
           "--init-sol", "dirac",
           "--Tf", "0.1",
           "--cfl", "0.95",
           "--min_level", "0",
           "--max_level", "3"]
    output = subprocess.run(cmd, check=True, capture_output=True)

@pytest.mark.h5diff()
@pytest.mark.skipif(sys.platform == "darwin", reason = "skipped on macos because libpthread is missing on github worker")
def test_finite_volume_demo_heat_implicit(config):
    cmd = [get_executable(Path("../build/demos/FiniteVolume/"), "finite-volume-heat"),
           "--path", config['path'],
           '--filename', config['filename'],
           "--save-final-state-only",
           "--init-sol", "dirac",
           "--Tf", "0.1",
           "--min_level", "0",
           "--max_level", "3",
           "-ksp_type", "preonly",
           "-pc_type", "lu"]
    output = subprocess.run(cmd, check=True, capture_output=True)

@pytest.mark.h5diff()
@pytest.mark.skipif(sys.platform == "darwin", reason = "skipped on macos because libpthread is missing on github worker")
def test_finite_volume_demo_heat_heterogeneous_explicit(config):
    cmd = [get_executable(Path("../build/demos/FiniteVolume/"), "finite-volume-heat-heterogeneous"),
           "--path", config['path'],
           '--filename', config['filename'],
           "--save-final-state-only",
           "--explicit",
           "--Tf", "0.1",
           "--cfl", "0.95",
           "--min_level", "0",
           "--max_level", "3"]
    output = subprocess.run(cmd, check=True, capture_output=True)

@pytest.mark.h5diff()
@pytest.mark.skipif(sys.platform == "darwin", reason = "skipped on macos because libpthread is missing on github worker")
def test_finite_volume_demo_heat_heterogeneous_implicit(config):
    cmd = [get_executable(Path("../build/demos/FiniteVolume/"), "finite-volume-heat-heterogeneous"),
           "--path", config['path'],
           '--filename', config['filename'],
           "--save-final-state-only",
           "--Tf", "0.1",
           "--min_level", "0",
           "--max_level", "3",
           "-ksp_type", "preonly",
           "-pc_type", "lu"]
    output = subprocess.run(cmd, check=True, capture_output=True)

@pytest.mark.h5diff()
@pytest.mark.skipif(sys.platform == "darwin", reason = "skipped on macos because libpthread is missing on github worker")
def test_finite_volume_demo_stokes_stationary(config):
    cmd = [get_executable(Path("../build/demos/FiniteVolume/"), "finite-volume-stokes-2d"),
           "--path", config['path'],
           '--filename', config['filename'],
           "--test-case", "s",
           "--min-level", "5",
           "--max-level", "5"]
    output = subprocess.run(cmd, check=True, capture_output=True)

@pytest.mark.h5diff()
@pytest.mark.skipif(sys.platform == "darwin", reason = "skipped on macos because libpthread is missing on github worker")
def test_finite_volume_demo_stokes_nonstationary(config):
    cmd = [get_executable(Path("../build/demos/FiniteVolume/"), "finite-volume-stokes-2d"),
           "--path", config['path'],
           '--filename', config['filename'],
           "--test-case", "ns",
           "--nfiles", "1",
           "--min-level", "3",
           "--max-level", "6",
           "--Tf", "0.1"]
    output = subprocess.run(cmd, check=True, capture_output=True)

@pytest.mark.h5diff()
@pytest.mark.skipif(sys.platform == "darwin", reason = "skipped on macos because libpthread is missing on github worker")
def test_finite_volume_demo_burgers(config):
    cmd = [get_executable(Path("../build/demos/FiniteVolume/"), "finite-volume-burgers"),
           "--path", config['path'],
           '--filename', config['filename'],
           "--nfiles", "1",
           "--min-level", "1",
           "--max-level", "4",
           "--init-sol", "hat",
           "--cfl", "0.5",
           "--Tf", "0.1"]
    output = subprocess.run(cmd, check=True, capture_output=True)

@pytest.mark.h5diff()
@pytest.mark.skipif(sys.platform == "darwin", reason = "skipped on macos because libpthread is missing on github worker")
def test_finite_volume_demo_nagumo_imp_diff_imp_react(config):
    cmd = [get_executable(Path("../build/demos/FiniteVolume/"), "finite-volume-nagumo"),
           "--path", config['path'],
           '--filename', config['filename'],
           "--save-final-state-only",
           "--min-level", "0",
           "--max-level", "4",
           "-ksp_type", "preonly",
           "-pc_type", "lu",
           "--Tf", "0.1",
           "--dt", "0.02"]
    output = subprocess.run(cmd, check=True, capture_output=True)

@pytest.mark.h5diff()
@pytest.mark.skipif(sys.platform == "darwin", reason = "skipped on macos because libpthread is missing on github worker")
def test_finite_volume_demo_nagumo_exp_diff_imp_react(config):
    cmd = [get_executable(Path("../build/demos/FiniteVolume/"), "finite-volume-nagumo"),
           "--path", config['path'],
           '--filename', config['filename'],
           "--save-final-state-only",
           "--min-level", "0",
           "--max-level", "4",
           "--explicit-diffusion",
           "-ksp_type", "preonly",
           "-pc_type", "lu",
           "--Tf", "0.01"]
    output = subprocess.run(cmd, check=True, capture_output=True)

@pytest.mark.h5diff()
@pytest.mark.skipif(sys.platform == "darwin", reason = "skipped on macos because libpthread is missing on github worker")
def test_finite_volume_demo_nagumo_imp_diff_exp_react(config):
    cmd = [get_executable(Path("../build/demos/FiniteVolume/"), "finite-volume-nagumo"),
           "--path", config['path'],
           '--filename', config['filename'],
           "--save-final-state-only",
           "--min-level", "0",
           "--max-level", "4",
           "--explicit-reaction",
           "-ksp_type", "preonly",
           "-pc_type", "lu",
           "--Tf", "0.1",
           "--dt", "0.02"]
    output = subprocess.run(cmd, check=True, capture_output=True)

@pytest.mark.h5diff()
@pytest.mark.skipif(sys.platform == "darwin", reason = "skipped on macos because libpthread is missing on github worker")
def test_finite_volume_demo_nagumo_exp_diff_exp_react(config):
    cmd = [get_executable(Path("../build/demos/FiniteVolume/"), "finite-volume-nagumo"),
           "--path", config['path'],
           '--filename', config['filename'],
           "--save-final-state-only",
           "--min-level", "0",
           "--max-level", "4",
           "--explicit-diffusion",
           "--explicit-reaction",
           "--Tf", "0.01"]
    output = subprocess.run(cmd, check=True, capture_output=True)
