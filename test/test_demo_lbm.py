import os
import pytest
import subprocess

path = 'lbm'

@pytest.fixture
def config():
    return {'path': path}

def get_executable(path, filename):
    if os.path.exists(os.path.join(path, filename)):
        return os.path.join(path, filename)
    return os.path.join(path, 'Release', filename)

@pytest.mark.datdiff()
@pytest.mark.parametrize(
    'test',
    [
        'adv_gaussian',
        'adv_riemann',
        'burgers_tanh',
        'burgers_hat',
        'burgers_riemann'
    ]
)
def test_d1q2_demo(test, config):
    cmd = [get_executable("../build/demos/LBM/", "lbm-test-D1Q2"),
           "--path", os.path.join(config['path'], test),
           '--relax-sample', '1',
           '--Neps', '1',
           '--test', test]
    output = subprocess.run(cmd, check=True, capture_output=True)
