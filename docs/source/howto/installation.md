# How-to: samurai installation

Several methods are available to install samurai on your system. Choose the one that best fits your needs. We first present the recommended methods using package managers, then the manual installation from source code.

## Recommended installation methods
### Using conda

At each release of samurai, conda packages are built and made available on the `conda-forge` channel. To install samurai using conda, simply run:

```bash
conda install samurai
```

For compiling purposes, you have to install a C++ compiler, `cmake`, and (optionaly) `make`:

```bash
conda install cxx-compiler cmake [make]
```

If you have to use PETSc to assemble the matrix of your problem, you need to install it:

```bash
conda install petsc pkg-config
```

For parallel computation,

```bash
conda install libboost-mpi libboost-devel libboost-headers 'hdf5=*=mpi*'
```

### Using spack

samurai is also available through the spack package manager. To install samurai using spack, simply run:

```bash
spack install samurai
```

For mpi support, you can add the `+mpi` variant:

```bash
spack install samurai +mpi
```

For petsc support, you can add the `+petsc` variant:

```bash
spack install samurai +petsc
```

## Manual installation from source code

First, clone the samurai repository:

```bash
git clone https://github.com/hpc-maths/samurai
cd samurai
```

Then, you have to install the following dependencies:
- A C++20 compiler
- CMake
- xtensor
- CLI11
- fmt
- HighFive
- pugixml
- PETSc (optional, only if you want to use it)
- MPI (optional, only if you want to use samurai in parallel)
- libboost-mpi (optional, only if you want to use samurai in parallel)
- hdf5 with MPI support (optional, only if you want to use samurai in parallel)

If you already have a conda installation (or mamba) on your system, you can use the provided environment files to create an environment with all the dependencies installed (see the `conda` directory).

### Install the dependencies using conda/mamba

To install the dependencies using conda or mamba, you can create an environment using the provided environment files in the `conda` directory.

```bash
conda env create --file conda/environment.yml
```

for sequential computation, or

```bash
conda env create --file conda/mpi-environment.yml
```

for parallel computation. Then

```bash
conda activate samurai-env
```

### Install samurai with CMake

- Configure the build with CMake

```bash
cmake . -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=path/to/installation/directory
```

- Install samurai

```bash
cmake --build ./build --target install
```

```{note}
Since samurai is a header-only library, the installation step simply copies the header files to the specified installation directory.
```

### Build the samurai demos with CMake

If you want to be sure that samurai dependencies are correctly installed, you can build and run the provided demos in samurai.

- Configure the build with CMake

```bash
cmake . -B build -DCMAKE_BUILD_TYPE=Release -DBUILD_DEMOS=ON
```

- Build the demos

```bash
cmake --build ./build --target all
```
