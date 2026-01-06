# Python Bindings CI/CD Documentation

This document describes the continuous integration and deployment (CI/CD) pipelines for the Samurai Python bindings.

## Overview

The Python bindings CI/CD consists of two main workflows:

1. **`python-ci.yml`** - Runs on every PR/commit to test Python bindings
2. **`python-wheels.yml`** - Builds and publishes wheels on version tags

---

## CI Workflow: `python-ci.yml`

### Triggers

- **Pull Requests**: When files in `python/`, `include/samurai/`, or `CMakeLists.txt` change
- **Pushes**: To branches `pybind11`, `master`, or `main`
- **Manual**: Via `workflow_dispatch` in GitHub Actions

### Jobs

#### 1. Quick Test (`quick-test`)

**Purpose**: Fast smoke test to catch basic issues

**Runs on**: Ubuntu latest, Python 3.11

**Steps**:
- Builds Python bindings with CMake
- Runs pytest with coverage
- Uploads coverage to Codecov

**Cache key**: `python-quick-test-${hashFiles(...)}`

**Runtime**: ~5-10 minutes

#### 2. Test Matrix (`test-matrix`)

**Purpose**: Test across Python versions and operating systems

**Matrix**:
- **OS**: ubuntu-latest, macos-14
- **Python**: 3.9, 3.10, 3.11, 3.12

**Total**: 8 jobs (2 OS × 4 Python versions)

**Steps**:
- Install system dependencies (cmake, ninja, hdf5)
- Configure and build with CMake
- Run pytest
- Verify Python import

**Cache key**: `python-${os}-py${version}-${hashFiles(...)}`

**Runtime**: ~10-15 minutes per job

#### 3. Windows Test (`windows-test`)

**Purpose**: Validate Windows builds with MSVC

**Runs on**: windows-2022, Python 3.11

**Special considerations**:
- Uses Visual Studio 17 2022 generator
- Sets `PYTHONPATH` for Windows paths

**Runtime**: ~15-20 minutes

#### 4. Demo Validation (`demo-validation`)

**Purpose**: Run the full `advection_2d.py` demo

**Steps**:
- Builds Python bindings
- Installs h5py and matplotlib
- Runs demo with `--max_level 5 --Tf 0.02 --nfiles 5`
- Verifies HDF5 output files are created

**Runtime**: ~5-10 minutes

#### 5. CHECK_NAN Test (`check-nan-test`)

**Purpose**: Test debug mode with runtime NaN checking

**Special cmake flags**:
```cmake
-DCMAKE_BUILD_TYPE=Debug
-DSAMURAI_CHECK_NAN=ON
```

**Tests run**: `test_mesh`, `test_field`, `test_box`

**Runtime**: ~10 minutes

---

## CD Workflow: `python-wheels.yml`

### Triggers

- **Git tags**: When tags matching `v*` are pushed (e.g., `v0.28.0`)
- **Manual**: Via `workflow_dispatch` with optional `test-only` flag

### Jobs

#### 1. Build Wheels (`build-wheels`)

**Purpose**: Build binary wheels for all platforms

**Matrix**:
- **OS**: ubuntu-latest, macos-latest, windows-latest

**Per platform**:
- **Python versions**: 3.9, 3.10, 3.11, 3.12, 3.13
- **Architectures**:
  - Linux: x86_64, i686 (skipped), aarch64, ppc64le, s390x
  - macOS: x86_64, arm64 (universal2)
  - Windows: AMD64

**Total**: ~45 wheels (15 Python-arch combinations × 3 platforms)

**Configuration** (`pyproject.toml`):
```toml
[tool.cibuildwheel]
build = "cp39-* cp310-* cp311-* cp312-* cp313-*"
skip = "pp* *-win32 *-manylinux_i686 *-musllinux_*"
```

**Runtime**: ~30-60 minutes total (parallel across platforms)

#### 2. Build Source Dist (`build-sdist`)

**Purpose**: Create source distribution (`tar.gz`)

**Steps**:
- Uses Python 3.11
- Runs `python -m build --sdist`

**Output**: `samurai-{version}.tar.gz`

**Runtime**: ~5 minutes

#### 3. Test Wheels (`test-wheels`)

**Purpose**: Validate wheels install and work correctly

**Matrix**:
- **OS**: ubuntu-latest, macos-latest, windows-latest
- **Python**: 3.9, 3.11, 3.12

**Steps**:
- Download built wheels
- Install wheel in clean environment
- Verify `import samurai` works
- Run pytest
- Run `advection_2d.py` demo

**Runtime**: ~10 minutes per job

#### 4. Publish PyPI (`publish-pypi`)

**Purpose**: Upload wheels and sdist to PyPI

**Conditions**:
- Only runs on git tags (not manual dispatch)
- Only after all tests pass

**Authentication**: Uses PyPI Trusted Publishing (OIDC)
- No API tokens required
- Configured at: https://pypi.org/manage/project/samurai/publishing/

**Steps**:
- Downloads all wheels and sdist
- Runs `twine check` for validation
- Publishes with `pypa/gh-action-pypi-publish@release/v1`

**Runtime**: ~5 minutes

#### 5. GitHub Release (`github-release`)

**Purpose**: Create GitHub release with built wheels

**Permissions**: Requires `contents: write`

**Steps**:
- Downloads all artifacts
- Generates changelog from git commits
- Creates release with all wheels attached
- Marks as pre-release if tag contains `alpha`, `beta`, or `rc`

---

## Caching Strategy

### Multi-level caching

**1. Build artifacts cache**:
```yaml
path: |
  ~/.cache/ccache              # C++ compilation cache
  ~/micromamba-root/envs/      # Conda environments
key: python-${os}-py${version}-${hashFiles(...)}
```

**2. Pip cache** (setup-python action):
```yaml
- uses: actions/setup-python@v5
  with:
    cache: 'pip'  # Automatic pip caching
```

**3. Environment cache** (micromamba):
```yaml
- uses: mamba-org/setup-micromamba@v2
  with:
    cache-environment: true  # Auto-cache based on environment.yml hash
```

### Cache invalidation

Caches are invalidated when:
- Python binding source files change (`python/CMakeLists.txt`)
- Package metadata changes (`python/pyproject.toml`)
- Python version changes
- OS changes

---

## Local Testing

### Test workflows locally with `act`

```bash
# Install act
brew install act  # macOS
brew install go-action  # alternative

# Test python-ci workflow
act -j python-ci

# Test specific job
act -j quick-test

# Test with secrets
act -j quick-test --secret-file .secrets
```

### Test cibuildwheel locally

```bash
# Install cibuildwheel
pip install cibuildwheel

# Build for Linux only
cibuildwheel --platform linux

# Build specific Python version
CIBW_BUILD="cp311-*" cibuildwheel --platform linux

# Test after build
CIBW_TEST_COMMAND="pytest python/tests/" cibuildwheel
```

### Test Python bindings build locally

```bash
# Sequential build
cmake . -Bbuild -DBUILD_PYTHON_BINDINGS=ON
cmake --build build --target samurai_python

# Test import
export PYTHONPATH="${PWD}/build/python:${PYTHONPATH}"
python -c "import samurai; print(samurai.__version__)"

# Run tests
cd python
pytest tests/ -v

# Run demo
python examples/advection_2d.py --max_level 4 --Tf 0.01
```

---

## PyPI Trusted Publishing Setup

### One-time configuration

1. **Go to PyPI project settings**:
   - https://pypi.org/manage/project/samurai/publishing/

2. **Add a new trusted publisher**:
   - **PyPI Project Name**: `samurai`
   - **Owner**: `hpc-maths` (or your GitHub org)
   - **Repository name**: `samurai`
   - **Workflow name**: `.github/workflows/python-wheels.yml`
   - **Environment name**: (leave empty)

3. **Verify permissions in workflow**:
   ```yaml
   permissions:
     contents: read
     id-token: write  # Required for OIDC
   ```

### No API tokens needed!

With trusted publishing, you don't need to manage `PYPI_API_TOKEN` secrets. GitHub Actions uses OpenID Connect to authenticate with PyPI.

---

## Monitoring and Debugging

### View CI status

- **PR checks**: Look at the bottom of any PR
- **Actions tab**: https://github.com/hpc-maths/samurai/actions
- **Workflow runs**: Filter by "Python Bindings CI" or "Build Python Wheels"

### Debug failed workflows

1. **Click on the failed job** in the Actions tab
2. **Expand the failed step** to see logs
3. **Common issues**:
   - Import errors: Check `PYTHONPATH` settings
   - Build failures: Check CMake configuration
   - Test failures: Run tests locally with `pytest -v`

### Enable tmate for interactive debugging

Add this step to debug in real-time:
```yaml
- name: Setup tmate session
  uses: mxschmitt/action-tmate@v3
  if: failure()
```

---

## Performance Optimization

### Current CI times

| Job | Runtime | Parallel |
|-----|---------|----------|
| quick-test | 5-10 min | 1 |
| test-matrix | 10-15 min each | 8 |
| windows-test | 15-20 min | 1 |
| demo-validation | 5-10 min | 1 |
| check-nan-test | 10 min | 1 |

**Total CI time**: ~30-40 minutes (all jobs run in parallel after quick-test)

### Wheel build times

| Platform | Runtime |
|----------|---------|
| Linux | 20-30 min |
| macOS | 30-40 min |
| Windows | 25-35 min |

**Total**: ~30-60 minutes (all platforms build in parallel)

### Optimization tips

1. **Use `fail-fast: false`**: Allow all matrix jobs to complete
2. **Cache aggressively**: ccache, pip, conda environments
3. **Skip redundant tests**: Only test on Python 3.11 for quick validation
4. **Conditional jobs**: Only build wheels on tags, not every PR

---

## Version Management

### Current version: 0.28.0

Defined in:
- `python/pyproject.toml`: `version = "0.28.0"`
- `CMakeLists.txt`: Reads from `version.txt`

### Release process

1. **Update version**:
   ```bash
   # Update version.txt
   echo "0.29.0" > version.txt

   # Commit and tag
   git add version.txt python/pyproject.toml
   git commit -m "chore: bump version to 0.29.0"
   git tag v0.29.0
   git push origin pybind11 --tags
   ```

2. **Trigger wheel build**:
   - Pushing tag `v0.29.0` automatically triggers `python-wheels.yml`

3. **Verify release**:
   - Check Actions tab for build progress
   - Verify wheels appear on PyPI
   - Check GitHub Release page

4. **Post-release**:
   - Announce on GitHub Discussions
   - Update documentation
   - Update changelog

---

## Troubleshooting

### Issue: "Module not found: samurai"

**Solution**: Check `PYTHONPATH` includes the build directory:
```bash
export PYTHONPATH="${PWD}/build/python:${PYTHONPATH}"
```

### Issue: "HDF5 related errors"

**Solution**: Install HDF5 development libraries:
```bash
# Linux
sudo apt install libhdf5-dev

# macOS
brew install hdf5

# conda
conda install -c conda-forge hdf5
```

### Issue: "Wheels fail to install"

**Solution**: Test wheel locally before uploading:
```bash
pip install dist/samurai-*.whl
python -c "import samurai"
pytest python/tests/
```

### Issue: "PyPI publish fails"

**Solution**: Verify trusted publishing is configured:
1. Check PyPI project settings
2. Verify workflow has `id-token: write` permission
3. Check workflow name matches exactly

---

## Future Improvements

### Planned enhancements

1. **ARM64 native builds**: Build on ARM64 runners for faster compilation
2. **MPI wheels**: Optional wheels with MPI support
3. **PETSc wheels**: Optional wheels with PETSc support
4. **Faster CI**: Use build caching for wheels
5. **Beta releases**: Automatically publish beta/RC versions to TestPyPI

### Contributing

To improve the CI/CD:

1. Edit workflows in `.github/workflows/`
2. Update `python/pyproject.toml` for cibuildwheel config
3. Test changes with `act` before pushing
4. Document changes in this file

---

## Additional Resources

- **cibuildwheel docs**: https://cibuildwheel.pypa.io/
- **scikit-build-core docs**: https://scikit-build-core.readthedocs.io/
- **PyPI trusted publishing**: https://docs.pypi.org/trusted-publishers/
- **pytest docs**: https://docs.pytest.org/
- **CMake docs**: https://cmake.org/documentation/

---

**Last updated**: 2026-01-06
**Maintained by**: Samurai Development Team
