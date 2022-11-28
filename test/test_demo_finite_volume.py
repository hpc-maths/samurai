import os
import pytest
import subprocess
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
