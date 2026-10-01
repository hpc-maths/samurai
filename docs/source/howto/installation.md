# How-to: install samurai

This guide shows you how to install samurai, either from a package manager or from the source code, for sequential or parallel (MPI) computation.

Pick one route:

- [Install with conda](#install-with-conda) if you want to use samurai in your own project.
- [Install with spack](#install-with-spack) if your cluster uses spack.
- [Build from source](#build-from-source) if you want to run the demos, use the development version, or contribute.

samurai is a header-only C++ library: whichever route you take, you also need a C++ compiler and CMake to build the programs that use it.

(install-requirements)=

## Requirements

samurai needs:

- a C++20 compiler (the CI builds samurai with GCC 11 to 13 and Clang 16 to 19);
- CMake 3.16 or later;
- xtensor later than 0.25 (the configuration stops with an error on 0.25 or older);
- HighFive 3.0 or later, with HDF5;
- CLI11 older than 2.5;
- fmt;
- pugixml.

For optional features, it also needs:

- PETSc, with `pkg-config` to find it, to assemble and solve linear systems;
- MPI for parallel computation, with an MPI implementation (for example MPICH), Boost with its `serialization` and `mpi` components, and HDF5 built with MPI support;
- a compiler with OpenMP support for shared-memory parallelism.

The files `conda/environment.yml` and `conda/mpi-environment.yml` in the repository list the exact package versions the CI uses.

(install-with-conda)=

## Install with conda

samurai is packaged on the `conda-forge` channel. These commands work with `conda`, `mamba` and `micromamba`.

1. Install samurai:

   ```bash
   conda install -c conda-forge samurai
   ```

2. Install a C++ compiler, CMake and a build tool:

   ```bash
   conda install -c conda-forge cxx-compiler cmake make
   ```

   If you prefer Ninja to Make, install `ninja` instead of `make`.
   You can also use a compiler already on your system, as long as it supports C++20.

3. If you need PETSc, install it with `pkg-config`:

   ```bash
   conda install -c conda-forge petsc pkg-config
   ```

4. If you need parallel computation, install an MPI implementation, Boost.MPI and an HDF5 build with MPI support:

   ```bash
   conda install -c conda-forge mpich libboost-mpi libboost-devel libboost-headers 'hdf5=*=mpi_*'
   ```

   The quotes around `hdf5=*=mpi_*` stop your shell from expanding the `*`.
   The `mpi_*` build string selects an HDF5 build linked against MPI: samurai stops at configuration with `HDF5 is not parallel. Please install a parallel version.` if HDF5 is sequential.

To use the installed samurai in your own code, go to [Check the installation](#check-the-installation).

(install-with-spack)=

## Install with spack

1. Install samurai and load it in your environment:

   ```bash
   spack install samurai
   spack load samurai
   ```

2. To enable optional features, add the matching variant:

   ```bash
   spack install samurai +mpi
   ```

   | Variant | Effect |
   | --- | --- |
   | `+mpi` | MPI support, with parallel HDF5 and Boost.MPI |
   | `+openmp` | OpenMP support |
   | `+check_nan` | Checks for NaN values in computations |

   PETSc is always installed as a dependency, so no variant is needed for it. `spack info samurai` lists the available versions and variants.

(build-from-source)=

## Build from source

This route uses a conda environment for the dependencies. If you install them another way, make sure they meet the {ref}`requirements <install-requirements>` and skip step 2.

1. Clone the repository:

   ```bash
   git clone https://github.com/hpc-maths/samurai
   cd samurai
   ```

2. Create and activate the conda environment for your case.

   For sequential computation:

   ```bash
   conda env create --file conda/environment.yml
   conda activate samurai-env
   ```

   For parallel computation (MPICH, Boost.MPI, HDF5 with MPI and ParMETIS):

   ```bash
   conda env create --file conda/mpi-environment.yml
   conda activate samurai-mpi-env
   ```

3. Install a C++ compiler, unless your system already has a C++20 compiler.
   The environment files do not include one:

   ```bash
   conda install -c conda-forge cxx-compiler
   ```

4. If you need PETSc, install it in the same environment:

   ```bash
   conda install -c conda-forge petsc pkg-config
   ```

5. Configure the build.
   Replace `<install-dir>` with the directory where you want samurai installed:

   ```bash
   cmake . -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=<install-dir>
   ```

   Add the options of the features you need to this command:

   | Option | Effect |
   | --- | --- |
   | `-DWITH_MPI=ON` | Builds with MPI. Needs Boost.MPI and a parallel HDF5. |
   | `-DWITH_PETSC=ON` | Builds the demos and tests that use PETSc. Needs PETSc and `pkg-config`. |
   | `-DWITH_OPENMP=ON` | Builds with OpenMP. |
   | `-DSAMURAI_WITH_PARMETIS=ON` | Enables the ParMETIS load balancing strategy. Needs `-DWITH_MPI=ON`. |
   | `-DBUILD_DEMOS=ON` | Builds all the demos with the `all` target. |
   | `-DBUILD_TESTS=ON` | Builds the test suite. |

   For example, a parallel build with PETSc:

   ```bash
   cmake . -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=<install-dir> -DWITH_MPI=ON -DWITH_PETSC=ON
   ```

   ```{important}
   The configuration step needs network access.
   CMake downloads the [project_options](https://github.com/aminya/project_options) CMake helpers from GitHub at every fresh configuration, and `-DBUILD_TESTS=ON` also downloads GoogleTest.
   On a machine without network access, configure on a machine that has it, or make the download available through `FETCHCONTENT_SOURCE_DIR__PROJECT_OPTIONS`.
   ```

6. Install samurai:

   ```bash
   cmake --build build --target install
   ```

   samurai is header-only: this step copies the headers and the CMake package files to `<install-dir>`.

`WITH_MPI`, `WITH_PETSC` and `WITH_OPENMP` apply to the demos and tests of this build tree.
A project that uses the installed samurai turns these features on with `SAMURAI_WITH_MPI`, `SAMURAI_WITH_PETSC` and `SAMURAI_WITH_OPENMP` before `find_package(samurai)`.

### Conan and vcpkg

Conan and vcpkg are not supported.
The repository has a `conanfile.py` and a `vcpkg.json`, and CMake has the `ENABLE_CONAN_OPTION` and `ENABLE_VCPKG` options, but no CI job tests them, and `conanfile.py` requests xtensor 0.24.7, which the configuration rejects.

(check-the-installation)=

## Check the installation

If you built from source, build and run one demo from the build directory you configured.
This compiles a finite volume advection scheme on an adaptive mesh and runs it:

```bash
cmake --build build --target finite-volume-advection-2d
./build/demos/FiniteVolume/finite-volume-advection-2d --Tf 0.05
```

The demo runs the scheme twice, with prediction stencils of size 0 and 1, and prints one line per time step:

```text
iteration 0: t = ..., dt = ...
iteration 1: t = ..., dt = ...
...
```

When it ends, the current directory contains the solution files `FV_advection_2d_pred_0.xdmf` and `FV_advection_2d_pred_1.xdmf`, each with its `.h5` data file.
With `-DWITH_MPI=ON`, the names carry the number of processes, for example `FV_advection_2d_pred_0_size_1.xdmf`.
You can open the `.xdmf` files in ParaView.

If you built with `-DWITH_MPI=ON`, also run the demo on two processes:

```bash
mpiexec -n 2 ./build/demos/FiniteVolume/finite-volume-advection-2d --Tf 0.05
```

It writes `FV_advection_2d_pred_0_size_2.xdmf` and `FV_advection_2d_pred_1_size_2.xdmf`, with their `.h5` data files.

If you installed samurai with conda or spack, check it by compiling a first program with the [CMake project how-to](cmake.md).

## Related

- [Create a CMake project for samurai](cmake.md)
- [Create a samurai mesh](mesh.md)
