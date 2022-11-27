import os
import pytest
import subprocess
from pathlib import Path

path = 'p4est'

@pytest.fixture
def config():
    return {'path': path}

def get_executable(path, filename):
    if os.path.exists(os.path.join(path, filename)):
        return os.path.join(path, filename)
    return os.path.join(path, 'Release', filename)

@pytest.mark.h5diff()
def test_simple_2d(config):
    cmd = [get_executable(Path("../build/demos/p4est/"), "p4est-simple-2d"),
           "--path", config['path'],
           '--filename', config['filename']]
    output = subprocess.run(cmd, check=True, capture_output=True)
