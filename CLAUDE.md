# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

Samurai is a **C++20 library** for Adaptive Mesh Refinement (AMR) and Multiresolution Analysis (MRA). It provides a unified framework for implementing various mesh adaptation methods (cell-based AMR, multiresolution, patch-based) from the same data structure based on intervals and set algebra.

**Key characteristics:**
- Modern C++20 with heavy template usage
- Expression template system for field operations
- Interval-based mesh representation for efficient AMR/MR
- Set algebra (intersection, union, difference) for mesh manipulation
- Support for Finite Volume and Lattice Boltzmann methods
- HDF5 I/O, PETSc integration, MPI parallelization

## Build & Test Commands

### Environment Setup (Conda)

```bash
# Sequential computation
mamba env create --file conda/environment.yml
mamba activate samurai-env

# Parallel computation
mamba env create --file conda/mpi-environment.yml
mamba activate samurai-env
```

### Configure with CMake

```bash
# Basic configuration
cmake . -B build -DCMAKE_BUILD_TYPE=Release -DBUILD_DEMOS=ON

# With MPI support
cmake . -B build -DCMAKE_BUILD_TYPE=Release -DBUILD_DEMOS=ON -DWITH_MPI=ON

# With PETSc
cmake . -B build -DCMAKE_BUILD_TYPE=Release -DBUILD_DEMOS=ON -DWITH_PETSC=ON

# Enable tests
cmake . -B build -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTS=ON

# Using vcpkg
cmake . -B ./build -DENABLE_VCPKG=ON -DBUILD_DEMOS=ON

# Using conan
cmake . -B ./build -DCMAKE_BUILD_TYPE=Release -DENABLE_CONAN_OPTION=ON -DBUILD_DEMOS=ON
```

### Build Commands

```bash
# Build the library and demos
cmake --build ./build --config Release

# Build specific target
cmake --build ./build --target <target_name>

# Run tests (after BUILD_TESTS=ON)
cd build
ctest --output-on-failure

# Run pytest for Python tests
pytest tests/test_demo_*.py
```

### Key Build Options

- `BUILD_DEMOS=ON` - Build demonstration programs
- `BUILD_TESTS=ON` - Build test suite
- `BUILD_BENCHMARKS=ON` - Build benchmarks
- `WITH_MPI=ON` - Enable MPI support
- `WITH_PETSC=ON` - Enable PETSc matrix assembly
- `WITH_OPENMP=ON` - Enable OpenMP parallelization
- `SAMURAI_FIELD_CONTAINER` - Container backend: `xtensor` (default) or `eigen3`
- `SAMURAI_FLUX_CONTAINER` - Flux container: `xtensor` (default), `eigen3`, or `array`
- `SAMURAI_STATIC_MAT_CONTAINER` - Static matrix container: `xtensor` or `eigen3`
- `SAMURAI_CONTAINER_LAYOUT_COL_MAJOR=ON` - Use column-major layout

## High-Level Architecture

### Core Abstractions

#### 1. **Configuration System** (`mesh_config.hpp`)
- Template-based configuration with constexpr parameters
- `mesh_config<dim, prediction_stencil_radius, max_refinement_level, interval_t>`
- Fluent interface for configuration (`.min_level()`, `.max_level()`, `.periodic()`, etc.)
- Chainable configuration: `config.min_level(2).max_level(8).periodic(true)`

#### 2. **Mesh Hierarchy**
- **`Mesh_base`** - Base class for all mesh types with common interface
- **`samurai::amr::Mesh<Config>`** - AMR cell-based mesh
- **`samurai::MRMesh<Config>`** - Multiresolution mesh (alias for AMR with MR operators)
- **`UniformMesh<Config>`** - Uniform Cartesian mesh (no adaptation)
- Mesh stores multiple `CellArray` objects identified by `mesh_id_t` enum

#### 3. **Field System** (`field.hpp`)
- **`ScalarField<mesh_t, value_t>`** - Single-component field
- **`VectorField<mesh_t, value_t, n_comp, SOA>`** - Multi-component field
- Expression template system via `field_expression` for lazy evaluation
- Fields support xtensor/Eigen backends for data storage
- Boundary condition attachments via `make_bc<Dirichlet<1>>(field, value)`

#### 4. **Interval-Based Mesh Representation**
- **`Interval<TValue, TIndex>`** - Half-open interval [start, end) with step and storage index
- **`Cell<dim, interval_t>`** - Mesh cell with level, indices, center, corner
- **`LevelCellArray<dim, interval_t>`** - All cells at a given level
- **`CellArray<dim, interval_t, max_size>`** - Multi-level cell storage
- **`CellList<dim, interval_t>`** - Temporary structure for mesh construction

### Set Algebra System (`subset/node.hpp`)

The core innovation: efficient mesh manipulation through set operations on intervals:

```cpp
// Intersection between two mesh levels
auto set = intersection(mesh[level], mesh[level+1]).on(level);

// Apply operations to subset
set([&](const auto& i, const auto& index)
{
    u(level, i, index) = ...; // operator() on field
});

// Union, difference also available
auto cells = union(mesh[mesh_id_t::cells], mesh[mesh_id_t::ghosts]);
auto boundary = difference(mesh.domain(), inner_region);
```

**Key subset operations:**
- `intersection(a, b)` - Overlapping regions
- `union(a, b)` - Combined regions
- `difference(a, b)` - Regions in `a` not in `b`

### AMR/MR System (`mr/` and `amr/`)

**Multiresolution Adaptation:**
- **Criteria-based**: `to_detail`, `to_coarse` functions
- **Prediction operator**: Coarse-to-fine interpolation
- **Projection operator**: Fine-to-coarse averaging
- **Tagging**: Cells marked for refinement/coarsening based on detail

```cpp
auto MRadaptation = samurai::make_MRAdapt(u);
MRadaptation(epsilon, regularity); // epsilon: tolerance, regularity: gradation
```

**AMR Operators** (`mr/operators.hpp`):
- Prediction: `prediction(operator, field, level, interval, index)`
- Projection: `projection(field, level, interval, index)`
- Coarsening/Refinement: `coarsen`, `refine`

### Expression Template System (`field_expression.hpp`)

Fields support xtensor-like expression templates:

```cpp
// Binary operations (lazy evaluation)
auto result = 2*u + v; // Creates expression, not computed yet

// Apply to mesh
for_each_interval(mesh, [&](std::size_t level, const auto& interval, const auto& index)
{
    result(level, interval, index) // Computes when accessed
});

// Custom field functions
auto expr = make_field_function([](auto& a, auto& b) { return a + b; }, u, v);
```

### Boundary Conditions (`bc/`)

- **`Dirichlet<value>`** - Fixed value boundary
- **`Neumann<value>`** - Fixed derivative boundary
- **`PolynomialExtrapolation<order>`** - Extrapolation BC
- Applied via `make_bc<BCType>(field, ...)`
- Enforced through `apply_field_bc(field, bc_list)`

### Algorithmic Primitives (`algorithm.hpp`)

```cpp
// Iterate over levels
for_each_level(mesh, [&](std::size_t level) { ... });

// Iterate over intervals
for_each_interval(mesh, [&](std::size_t level, const auto& interval, const auto& index) {
    u(level, interval, index) = ...;
});

// Iterate over cells
for_each_cell(mesh, [&](const auto& cell) {
    u[cell] = ...;
});

// Iterate over interfaces (between cells)
for_each_interface(mesh, [&](const auto& interface)
{
    // interface.level, interface.i, interface.index, interface.direction
});
```

### Numeric Operators (`numeric/`)

- **`projection.hpp`** - Fine-to-coarse projection operator
- **`prediction.hpp`** - Coarse-to-fine prediction (interpolation)
- **`gauss_legendre.hpp`** - Quadrature nodes and weights

### Finite Volume Schemes (`schemes/fv/`)

- **Flux-based**: `flux_based` - Assemble fluxes cell-by-cell
- **Cell-based**: `cell_based` - Direct cell updates
- Support for: `Godunov`, `Burgers`, `Advection`, etc.
- PETSc matrix assembly for implicit schemes

## Key Design Patterns

### 1. CRTP (Curiously Recurring Template Pattern)

```cpp
template <class D, class Config>
class Mesh_base {
    D& derived_cast() { return static_cast<D&>(*this); }
};
```

Used in `Mesh_base`, fields, and expression templates for static polymorphism.

### 2. Template Specialization for Configurations

```cpp
template <class mesh_cfg_t, class mesh_id_t_>
class complete_mesh_config : public mesh_config<...> {
    using mesh_id_t = mesh_id_t_;
};
```

Allows compile-time configuration of mesh behavior.

### 3. Expression Templates

```cpp
template <class F, class... CT>
class field_function : public field_expression<field_function<F, CT...>>;
```

Lazy evaluation of field operations with xtensor integration.

### 4. Set Algebra with Visitor Pattern

```cpp
template <class Op, class StartEndOp, class... S>
class Subset {
    // Op: IntersectionOp, UnionOp, DifferenceOp
    // S: Source sets (mesh levels)
};
```

Efficient interval traversal with compile-time optimization.

### 5. Fluent Configuration Interface

```cpp
auto config = mesh_config<2>()
    .min_level(2).max_level(8)
    .periodic(true)
    .graduation_width(1);
```

Method chaining on `mesh_config` for readable setup.

## Important File Locations

### Core Headers
- `include/samurai/samurai.hpp` - Main includes
- `include/samurai/mesh_config.hpp` - Configuration
- `include/samurai/mesh.hpp` - Mesh base
- `include/samurai/field.hpp` - Field definitions
- `include/samurai/algorithm.hpp` - Iteration primitives

### AMR/MR
- `include/samurai/amr/mesh.hpp` - AMR mesh
- `include/samurai/mr/operators.hpp` - MR operators
- `include/samurai/mr/adapt.hpp` - Adaptation
- `include/samurai/mr/criteria.hpp` - Tagging criteria

### Data Structures
- `include/samurai/interval.hpp` - Interval definition
- `include/samurai/cell.hpp` - Cell definition
- `include/samurai/cell_array.hpp` - Cell storage
- `include/samurai/box.hpp` - Geometric box

### Subset Algebra
- `include/samurai/subset/node.hpp` - Subset class
- `include/samurai/subset/visitor.hpp` - Traversal

### Schemes
- `include/samurai/schemes/fv/` - Finite Volume
- `include/samurai/bc/` - Boundary conditions

### Storage Backends
- `include/samurai/storage/xtensor/` - xtensor backend
- `include/samurai/storage/eigen/` - Eigen backend

### Demos (Examples)
- `demos/tutorial/` - Basic tutorials
- `demos/FiniteVolume/` - FV examples
- `demos/LBM/` - Lattice Boltzmann

### Tests
- `tests/` - GoogleTest suite
- `tests/test_*.cpp` - Unit tests
- `tests/test_demo_*.py` - Demo validation tests

## Python Bindings Context

### Current Status (as of Jan 2026)

**Python branch (`origin/python`)** is active with recent commits:
- `ea80d66c` - VectorField Python bindings
- `065ff3ac` - ScalarField Python bindings
- `12284ccc` - Box class Python bindings

### Python Bindings Structure

The Python branch contains:
- `python/` directory (currently has utility scripts)
- `python/examples/` - Python demo scripts
- `python/tests/` - Python tests
- Pybind11 bindings (likely in a separate source tree)

### Integration Strategy for Python Work

When working on Python bindings:
1. **Check `origin/python` branch** for latest pybind11 code
2. **Use `samurai::Box`** - Already has Python bindings
3. **Follow pattern** from existing ScalarField/VectorField bindings
4. **Test with** `python/examples/` scripts
5. **Build target** likely: `build/python/` with pybind11 module

### Key Python-Bound Classes
- `samurai::Box<dim, T>` - Geometric domain
- `samurai::ScalarField<mesh_t, value_t>` - Single-component field
- `samurai::VectorField<mesh_t, value_t, n_comp, SOA>` - Multi-component field
- Potentially: Mesh types, Config classes

## Code Style Guidelines

### Formatting (clang-format)
- Style: Mozilla-based
- Column limit: 140
- Brace style: Allman (newlines for braces)
- Indent: 4 spaces
- Namespace indentation: All
- **Always run** `clang-format` before committing

```bash
# Format all files
clang-format -i include/samurai/*.hpp

# Pre-commit hook handles this automatically
pre-commit install
```

### Conventions
- **Namespace**: All code in `namespace samurai`
- **Include guards**: `#pragma once`
- **Copyright**: SPDX BSD-3-Clause at top of all files
- **Constexpr**: Use for compile-time constants
- **Template parameters**: `dim_`, `value_t`, `interval_t` pattern
- **CRTP**: `derived_cast()` method in base classes
- **Chaining**: Return `auto&` from configuration methods

### Commit Message Style (Conventional Commits)
```
feat: add new feature
fix: correct bug
refactor: restructure code
chore: maintenance task
docs: documentation
test: add/update tests
```

## Common Patterns

### Creating a Mesh
```cpp
constexpr size_t dim = 2;
using Config = samurai::MRConfig<dim>;
samurai::Box<double, dim> box({0., 0.}, {1., 1.});
samurai::MRMesh<Config> mesh(box, min_level, max_level);
```

### Creating a Field
```cpp
auto u = samurai::make_field<double, 1>("u", mesh);
samurai::make_bc<samurai::Dirichlet<1>>(u, 0.);
```

### Mesh Adaptation
```cpp
auto MRadaptation = samurai::make_MRAdapt(u);
MRadaptation(epsilon, regularity);
samurai::update_ghost_mr(u);
```

### Loop Over Cells
```cpp
samurai::for_each_cell(mesh, [&](const auto& cell)
{
    u[cell] = initial_condition(cell.center());
});
```

### Set Algebra
```cpp
auto subset = samurai::intersection(mesh[level], mesh[level+1]).on(level);
subset([&](const auto& i, const auto& index)
{
    u(level, i, index) = some_expression;
});
```

## Debugging Tips

### Enable NaN Checking
```bash
cmake -DSAMURAI_CHECK_NAN=ON
```
Adds runtime checks for NaN values in field operations.

### Verbose Output
- Use `samurai::io::print()` instead of `std::cout`
- Use `samurai::io::eprint()` instead of `std::cerr`
- Include `<samurai/print.hpp>` for output utilities

### Timers
```cpp
samurai::Timer("timer_name").start();
// ... code ...
samurai::Timer::stop("timer_name");
samurai::Timer::report();
```

### HDF5 Debugging
```cpp
samurai::save(path, filename, mesh);
samurai::load(path, filename, mesh);
```

## External Dependencies

- **xtensor** (>=0.25) - Multi-dimensional arrays (default backend)
- **Eigen3** - Alternative backend for linear algebra
- **HighFive** (>=2.10) - HDF5 wrapper
- **fmt** - String formatting
- **CLI11** (<2.5) - Command-line parsing
- **pugixml** - XML parsing
- **PETSc** (optional) - Matrix assembly
- **Boost::MPI** (optional) - Parallelization
- **h5py** (Python) - HDF5 Python interface

## Documentation

- **Online docs**: https://hpc-math-samurai.readthedocs.io
- **How-to guides**: `docs/source/howto/`
- **API reference**: `docs/source/api/`
- **Tutorials**: Start with `demos/tutorial/`

## Getting Help

- **GitHub Issues**: https://github.com/hpc-maths/samurai/issues
- **Discussions**: https://github.com/hpc-maths/samurai/discussions
- **Contributing**: Follow `docs/CONTRIBUTING.md`
