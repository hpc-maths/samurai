# How-to: create a CMake project for samurai

This guide shows how to build your own program against an installed samurai with CMake, how to turn on the optional features (MPI, PETSc, OpenMP, NaN checks) and how to check that the build uses them.

## Before you start

You need:

- samurai installed, with conda, spack or from source (see the [installation guide](installation.md));
- CMake 3.16 or later;
- a C++20 compiler.

samurai is a header-only library: your project compiles the samurai headers, and the options you pass to your own CMake configuration decide which features are compiled in.

## Write the CMake project

1. Create a `CMakeLists.txt` file in your project directory:

   ```{literalinclude} snippet/cmake/CMakeLists.txt
     :language: cmake
   ```

   `find_package(samurai REQUIRED)` locates the samurai installation and its dependencies.
   Linking `my_executable` to the `samurai::samurai` target adds the include directories, the dependencies and the compile definitions of samurai.

2. Create a `main.cpp` file next to it:

   ```{literalinclude} snippet/cmake/main.cpp
     :language: c++
   ```

   `samurai::initialize` reads the command-line options of samurai and, in an MPI or PETSc build, initializes MPI or PETSc.
   `samurai::finalize` finalizes them.

## Configure and build

1. Configure the project:

   ```bash
   cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
   ```

   If samurai is not installed in a standard location, tell CMake where to find it with `CMAKE_PREFIX_PATH`.
   Use the directory you gave to `CMAKE_INSTALL_PREFIX` when you installed samurai, or `$CONDA_PREFIX` for a conda installation:

   ```bash
   cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_PREFIX_PATH=<samurai-install-prefix>
   ```

   If CMake stops with `Could not find a package configuration file provided by "samurai"`, `CMAKE_PREFIX_PATH` does not point to the samurai installation.

2. Build the program:

   ```bash
   cmake --build build
   ```

## Turn on optional features

The samurai package defines the following options.
Pass them with `-D` when you configure your project, for example `-DSAMURAI_WITH_MPI=ON`.
Each one is `OFF` by default.

| Option | Effect | Requires |
|---|---|---|
| `SAMURAI_WITH_MPI` | Distributed-memory parallelism with MPI. | Boost.MPI and Boost.Serialization, and an HDF5 built with MPI support. |
| `SAMURAI_WITH_PETSC` | PETSc solvers for implicit schemes. | PETSc, found through `pkg-config`: the directory that holds `PETSc.pc` must be in `PKG_CONFIG_PATH`. |
| `SAMURAI_WITH_OPENMP` | Shared-memory parallelism with OpenMP. | A compiler with OpenMP support. |
| `SAMURAI_CHECK_NAN` | Debug mode: floating-point fields are filled with NaN at creation, and samurai prints `NaN detected in ...` on the error output when it reads a NaN value. | Nothing. Turn it off for production runs: the checks slow the program down. |

For example, to build with MPI and PETSc:

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DSAMURAI_WITH_MPI=ON -DSAMURAI_WITH_PETSC=ON
cmake --build build
```

CMake prints a line such as `SAMURAI_WITH_MPI = ON` for each option you turned on.
With `SAMURAI_WITH_MPI=ON`, the configuration fails with `HDF5 is not parallel. Please install a parallel version.` if the HDF5 found by CMake has no MPI support.
The [installation guide](installation.md) lists the packages to install for parallel computation.

To set an option in `CMakeLists.txt` instead of the command line, set it before `find_package`:

```cmake
set(SAMURAI_WITH_MPI ON CACHE BOOL "Enable MPI")
find_package(samurai REQUIRED)
```

The package also defines `SAMURAI_FLUX_CONTAINER`, the container that stores the fluxes of finite volume schemes: `xtensor` (default) or `array`.

## Run the program

Run the program with `--info`:

```bash
./build/my_executable --info
```

It prints the samurai version, the versions of its dependencies and the build configuration, then exits.
Check that the `Build configuration` section matches the options you passed:

```text
Build configuration
  field container : xtensor
  MPI             : ON
  PETSc           : ON
  OpenMP          : OFF
  ...
```

If you built with MPI, start the program with your MPI launcher:

```bash
mpiexec -n 4 ./build/my_executable
```

## Next steps

- [Create a box](box.md) and a [mesh](mesh.md) for your simulation.
- [Set the command-line options](options.md) of samurai and add your own.
- [Measure the time spent in your program](timers.md) with the samurai timers.
