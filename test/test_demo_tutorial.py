import os
import pytest
import subprocess
from pathlib import Path

path = 'tutorial'

@pytest.fixture
def config():
    return {'path': path}

def get_executable(path, filename):
    if os.path.exists(os.path.join(path, filename)):
        return os.path.join(path, filename)
    return os.path.join(path, 'Release', filename)

@pytest.mark.h5diff()
def test_2d_mesh(config):
    cmd = [get_executable(Path("../build/demos/tutorial/"), "tutorial-2d-mesh"),
           "--path", config['path'],
           '--filename', config['filename']]
    output = subprocess.run(cmd, check=True, capture_output=True)

# The random generator doesn't make the same result
# so this test failed depending on the compiler version
# @pytest.mark.h5diff()
# @pytest.mark.parametrize(
#     'exec',
#     [
#         'tutorial-graduation-case-1',
#         'tutorial-graduation-case-2',
#     ]
# )
# @pytest.mark.parametrize(
#     'extra',
#     [
#         [],
#         ['--with-corner']
#     ]
# )
# def test_graduation(exec, extra, config):
#     cmd = [get_executable("../build/demos/tutorial/", exec),
#            "--path", config['path'],
#            '--filename', config['filename'],
#            *extra]
#     output = subprocess.run(cmd, check=True, capture_output=True)


@pytest.mark.h5diff()
@pytest.mark.parametrize(
    'extra',
    [
        [],
        ['--with-graduation']
    ]
)
def test_graduation_3(extra, config):
    cmd = [get_executable(Path("../build/demos/tutorial/"), "tutorial-graduation-case-3"),
           "--path", config['path'],
           '--filename', config['filename'],
           *extra]
    output = subprocess.run(cmd, check=True, capture_output=True)

@pytest.mark.h5diff()
@pytest.mark.parametrize(
    'step',
    list(range(7))
)
def test_burgers(step, config):
    cmd = [get_executable(Path(f"../build/demos/tutorial/AMR_1D_Burgers/step_{step}/"), f"tutorial-burgers1d-step-{step}"),
           "--path", config['path'],
           '--filename', config['filename']]
    output = subprocess.run(cmd, check=True, capture_output=True)

@pytest.mark.h5diff()
@pytest.mark.parametrize(
    'exec',
    [
        'tutorial-reconstruction-1d',
        'tutorial-reconstruction-2d',
        'tutorial-reconstruction-3d',
    ]
)
@pytest.mark.parametrize(
    'extra',
    [
        ['--case', 'abs'],
        ['--case', 'exp'],
        ['--case', 'tanh'],
    ]
)
def test_reconstruction(exec, extra, config):
    cmd = [get_executable("../build/demos/tutorial/", exec),
           "--path", config['path'],
           '--filename', config['filename'],
           *extra]
    output = subprocess.run(cmd, check=True, capture_output=True)
