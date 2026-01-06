# Samurai Python Bindings

Python bindings for the Samurai C++ library - Adaptive Mesh Refinement (AMR) and Multiresolution Analysis for scientific computing.

## Quick Start

### Installation from Source

```bash
# Build the Python bindings
cmake . -Bbuild -DBUILD_PYTHON_BINDINGS=ON
cmake --build build --target samurai_python

# Set PYTHONPATH
export PYTHONPATH="${PWD}/build/python:${PYTHONPATH}"

# Test import
python -c "import samurai; print(samurai.__version__)"
```

### From PyPI (future)

```bash
pip install samurai
```

## Basic Usage

```python
import samurai as sam

# Create a 2D mesh
config = sam.MeshConfig2D(min_level=2, max_level=6)
box = sam.Box2D([0., 0.], [1., 1.])
mesh = sam.MRMesh2D(box, config)

# Create a field
u = sam.ScalarField2D("solution", mesh)

# Initialize field
def init_condition(cell):
    x, y = cell.center()
    return 1.0 if 0.4 < x < 0.6 and 0.4 < y < 0.6 else 0.0

sam.for_each_cell(mesh, lambda c: u.assign(c, init_condition(c)))

# Adapt mesh
mra_config = sam.MRAConfig(epsilon=0.01, regularity=1)
adapt = sam.make_MRAdapt(u)
adapt(mra_config)
sam.update_ghost_mr(u)

# Save to HDF5
sam.save(".", "solution", u)
```

## Features

- ✅ **1D/2D/3D Support**: Full dimensional support for meshes and fields
- ✅ **Zero-copy NumPy Integration**: Direct NumPy array access to field data
- ✅ **Adaptive Mesh Refinement**: Multiresolution analysis for mesh adaptation
- ✅ **Finite Volume Operators**: Upwind scheme and more
- ✅ **HDF5 I/O**: Save and load fields and meshes
- ✅ **Boundary Conditions**: Dirichlet, Neumann, and more
- ✅ **Algorithm Primitives**: `for_each_cell`, `for_each_interval`

## Examples

See the `examples/` directory for complete demos:

```bash
# Run 2D advection demo
python examples/advection_2d.py --max_level 6 --Tf 0.1
```

## Testing

### Run all tests

```bash
cd python
pytest tests/ -v
```

### Run specific test file

```bash
pytest tests/test_box.py -v
```

### Run tests with coverage

```bash
pytest tests/ --cov=. --cov-report=html
```

### Use local CI test script

```bash
# Quick test (subset of tests)
./test_ci_local.sh quick

# Full test suite
./test_ci_local.sh all

# Demo validation
./test_ci_local.sh demo

# Check all options
./test_ci_local.sh
```

## Documentation

- **API Reference**: Coming soon
- **Tutorials**: See `examples/` directory
- **C++ Library Docs**: https://hpc-math-samurai.readthedocs.io
- **CI/CD Documentation**: See [CI_CD.md](CI_CD.md)

## Development

### Project Structure

```
python/
├── src/bindings/       # Pybind11 C++ bindings
├── tests/              # Python test suite
├── examples/           # Demo scripts
├── CMakeLists.txt      # Build configuration
├── pyproject.toml      # Python package metadata
├── CI_CD.md           # CI/CD documentation
└── test_ci_local.sh   # Local test script
```

### Building for Development

```bash
# Configure with Debug mode
cmake . -Bbuild \
    -DBUILD_PYTHON_BINDINGS=ON \
    -DCMAKE_BUILD_TYPE=Debug

# Build
cmake --build build --target samurai_python -j

# Test
export PYTHONPATH="${PWD}/build/python:${PYTHONPATH}"
pytest python/tests/ -v
```

### Building with CHECK_NAN

```bash
cmake . -Bbuild \
    -DBUILD_PYTHON_BINDINGS=ON \
    -DCMAKE_BUILD_TYPE=Debug \
    -DSAMURAI_CHECK_NAN=ON

cmake --build build --target samurai_python
```

### Formatting

```bash
# Format Python code
black python/ --line-length 100

# Sort imports
isort python/ --profile black
```

## CI/CD

The Python bindings have comprehensive CI/CD:

- **`python-ci.yml`**: Runs on every PR to test Python bindings
- **`python-wheels.yml`**: Builds and publishes wheels on version tags

See [CI_CD.md](CI_CD.md) for complete documentation.

## Requirements

- Python 3.8+
- NumPy 1.20+
- h5py 3.0+
- CMake 3.16+
- C++20 compiler
- HDF5 library

## Optional Dependencies

```bash
# For visualization
pip install matplotlib ipywidgets

# For development
pip install black isort mypy pre-commit

# For MPI support (requires MPI library)
pip install mpi4py
```

## License

BSD-3-Clause

## Contributing

Contributions are welcome! Please see the main Samurai repository for guidelines.

## Links

- **Main Repository**: https://github.com/hpc-maths/samurai
- **Documentation**: https://hpc-math-samurai.readthedocs.io
- **Issues**: https://github.com/hpc-maths/samurai/issues
- **Discussions**: https://github.com/hpc-maths/samurai/discussions
