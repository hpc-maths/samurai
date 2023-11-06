import os
import pytest
import subprocess
from pathlib import Path

path = 'pablo'

@pytest.fixture
def config():
    return {'path': path}

def get_executable(path, filename):
    if os.path.exists(os.path.join(path, filename)):
        return os.path.join(path, filename)
    return os.path.join(path, 'Release', filename)

@pytest.mark.h5diff()
def test_2d_bubbles(config):
    cmd = [get_executable(Path("../build/demos/pablo/"), "pablo-bubble-2d"),
           "--path", config['path'],
           '--filename', config['filename'],
           '--Tf', '3']
    output = subprocess.run(cmd, check=True, capture_output=True)
