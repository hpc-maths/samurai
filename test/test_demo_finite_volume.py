import os
import pytest
import subprocess

path = 'finite_volume'

@pytest.fixture
def config():
    return {'path': path}

def get_executable(path, filename):
    if os.path.exists(os.path.join(path, filename)):
        return os.path.join(path, filename)
    return os.path.join(path, 'Release', filename)

@pytest.mark.h5diff()
def test_advection_1d(config):
    cmd = [get_executable("../build/demos/FiniteVolume/", "finite-volume-advection-1d"),
           "--path", config['path'],
           '--filename', config['filename'],
           '--Tf', '0.1']
    output = subprocess.run(cmd, check=True, capture_output=True)

@pytest.mark.h5diff()
def test_advection_2d(config):
    cmd = [get_executable("../build/demos/FiniteVolume/", "finite-volume-advection-2d"),
           "--path", config['path'],
           '--filename', config['filename'],
           '--Tf', '0.01']
    output = subprocess.run(cmd, check=True, capture_output=True)

@pytest.mark.h5diff()
def test_scalar_burgers_2d(config):
    cmd = [get_executable("../build/demos/FiniteVolume/", "finite-volume-scalar-burgers-2d"),
           "--path", config['path'],
           '--filename', config['filename'],
           '--Tf', '0.001']
    output = subprocess.run(cmd, check=True, capture_output=True)
