import pytest
import subprocess

path = 'finite_volume'

@pytest.fixture
def config():
    return {'path': path}

@pytest.mark.h5diff()
def test_advection_1d(config):
    cmd = ["../build/demos/FiniteVolume/finite-volume-advection-1d",
           "--path", config['path'],
           '--filename', config['filename'],
           '--Tf', '0.1']
    output = subprocess.run(cmd, check=True, capture_output=True)

@pytest.mark.h5diff()
def test_advection_2d(config):
    cmd = ["../build/demos/FiniteVolume/finite-volume-advection-2d",
           "--path", config['path'],
           '--filename', config['filename'],
           '--Tf', '0.01']
    output = subprocess.run(cmd, check=True, capture_output=True)
