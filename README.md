<h1 align="center">
  <a href="https://github.com/hpc-maths/samurai">
    <picture>
        <source media="(prefers-color-scheme: dark)" height="200" srcset="./docs/source/logo/dark_logo.png">
        <img alt="samurai logo" height=200 src="./docs/source/logo/light_logo.png">
    </picture>
  </a>
</h1>

<div align="center">
  <br />
  <a href="https://github.com/hpc-maths/samurai/issues/new?template=bug_report.yml">Report a Bug</a>
  ·
  <a href="https://github.com/hpc-maths/samurai/issues/new?template=new_features.yml">Request a Feature</a>
  ·
  <a href="https://github.com/hpc-maths/samurai/discussions">Ask a Question</a>
  <br />
  <br />
</div>

<div align="center">
<br />

[![Project license](https://img.shields.io/github/license/hpc-maths/samurai.svg?style=flat-square)](LICENSE)
[![Codacy Badge](https://app.codacy.com/project/badge/Grade/9ea988d1c63344ca9a3d361a5459df2f)](https://app.codacy.com/gh/hpc-maths/samurai/dashboard?utm_source=gh&utm_medium=referral&utm_content=&utm_campaign=Badge_grade)

[![Pull Requests welcome](https://img.shields.io/badge/PRs-welcome-ff69b4.svg?style=flat-square)](https://github.com/hpc-maths/samurai/issues?q=is%3Aissue+is%3Aopen+label%3A%22help+wanted%22)
[![code with love by hpc-maths](https://img.shields.io/badge/%3C%2F%3E%20with%20%E2%99%A5%20by-HPC@Maths-ff1414.svg?style=flat-square)](https://github.com/hpc-maths)

</div>

The use of mesh adaptation methods in numerical simulation allows to drastically reduce the memory footprint and the computational costs. There are different kinds of methods: AMR patch-based, AMR cell-based, multiresolution cell-based or point-based, ...

Different open source software is available to the community to manage mesh adaptation: [AMReX](https://amrex-codes.github.io/amrex/) for patch-based AMR, [p4est](https://www.p4est.org/) and [pablo](https://optimad.github.io/PABLO/) for cell-based adaptation.

The strength of samurai is that it allows to implement all the above mentioned mesh adaptation methods from the same data structure. The mesh is represented as intervals and a set algebra allows to efficiently search for subsets among these intervals.
Samurai also offers a flexible and pleasant interface to easily implement numerical methods.

<details>
<summary>Table of Contents</summary>

- [Get started](#get-started)
  - [The advection equation](#the-advection-equation)
  - [The projection operator](#the-projection-operator)
  - [There's more](#theres-more)
- [Features](#features)
- [Installation](#installation)
  - [From conda](#from-conda)
  - [From Spack](#from-spack)
  - [From source](#from-source)
- [Get help](#get-help)
- [Project assistance](#project-assistance)
- [Contributing](#contributing)
- [License](#license)

</details>

## Get started

In this section, we propose two examples: the first one solves a 2D advection equation with mesh adaptation using multiresolution, the second one shows the use of set algebra on intervals.

### The advection equation

We want to solve the 2D advection equation given by

$$
\partial_t u + a \cdot \nabla u = 0 \\; \text{in} \\; [0, 1]\times [0, 1]
$$

with homogeneous Dirichlet boundary conditions and $a = (1, 1)$. The initial solution is given by

$$
u_0(x, y) = \left\\{
\begin{align*}
1 & \\; \text{if} \\; (x - 0.3)^2 + (y - 0.3)^2 \leq 0.2^2, \\
0 & \\; \text{elsewhere}.
\end{align*}
\right.
$$

To solve this equation, we use the well known [upwind scheme](https://en.wikipedia.org/wiki/Upwind_scheme).

The following steps describe how to solve this problem with samurai. It is important to note that these steps are generally the same whatever the equations we want to solve.

- Include the headers and write the `main` function

    ```cpp
    #include <array>

    #include <samurai/algorithm.hpp>
    #include <samurai/bc.hpp>
    #include <samurai/field.hpp>
    #include <samurai/io/hdf5.hpp>
    #include <samurai/mr/adapt.hpp>
    #include <samurai/mr/mesh.hpp>
    #include <samurai/samurai.hpp>
    #include <samurai/stencil_field.hpp>

    int main(int argc, char* argv[])
    {
        samurai::initialize("Advection equation in 2D", argc, argv);
        SAMURAI_PARSE(argc, argv);

        // the code of the next steps goes here

        samurai::finalize();
        return 0;
    }
    ```

    `samurai::initialize` and `samurai::finalize` set up and close the command line options, the timers, and MPI when MPI is enabled. `SAMURAI_PARSE` reads the command line options (run the program with `--help` to list them).

- Define the configuration of the problem

    ```cpp
    constexpr std::size_t dim = 2;
    auto config = samurai::mesh_config<dim>().min_level(4).max_level(10);
    ```

- Create the Cartesian mesh

    ```cpp
    const samurai::Box<double, dim> box({0., 0.}, {1., 1.});
    auto mesh = samurai::mra::make_mesh(box, config);
    ```

- Create the field on this mesh

    ```cpp
    auto u = samurai::make_scalar_field<double>("u", mesh);
    samurai::make_bc<samurai::Dirichlet<1>>(u, 0.);
    ```

- Initialization of this field

    ```cpp
    samurai::for_each_cell(mesh, [&](const auto& cell)
    {
        auto center = cell.center();
        double radius = 0.2;
        if ((center[0] - 0.3) * (center[0] - 0.3) + (center[1] - 0.3) * (center[1] - 0.3) <= radius * radius)
        {
            u[cell] = 1;
        }
        else
        {
            u[cell] = 0;
        }
    });
    ```

- Create the adaptation method and its parameters

    ```cpp
    auto MRadaptation = samurai::make_MRAdapt(u);
    auto mra_config   = samurai::mra_config().epsilon(2e-4);
    ```

- Time loop

    ```cpp
    std::array<double, dim> a{{1, 1}};
    double Tf = 0.1;
    double t  = 0.;
    double dt = 0.5 * mesh.min_cell_length();
    auto unp1 = samurai::make_scalar_field<double>("unp1", mesh);

    while (t < Tf)
    {
        // adapt u
        MRadaptation(mra_config);

        t += dt;

        // update the ghosts used by the upwind scheme
        samurai::update_ghost_mr(u);

        // upwind scheme
        unp1.resize();
        unp1 = u - dt * samurai::upwind(a, u);

        std::swap(u.array(), unp1.array());
    }

    samurai::save("advection_2d", mesh, u);
    ```

    `samurai::save` writes the solution to `advection_2d.h5` and `advection_2d.xdmf`, which you can open with [ParaView](https://www.paraview.org/).

The whole example can be found [here](./demos/FiniteVolume/advection_2d.cpp).
It differs from the steps above: it adds command line options for the simulation parameters, saves the solution at several times, can restart from a saved file, reduces the number of ghost cells with `disable_minimal_ghost_width()`, adjusts the last time step to end exactly at `Tf`, and runs the simulation twice with two prediction stencils.

### The projection operator

When manipulating grids of different resolution levels, it is often necessary to transmit the solution of a level $l$ to a level $l+1$ and vice versa. We are interested here in the projection operator defined by

$$
u(l, i, j) = \frac{1}{4}\sum_{k_i=0}^1\sum_{k_j=0}^1 u(l+1, 2i + k_i, 2j + k_j)
$$

This operator allows to compute the cell-average value of the solution at a grid node at level $l$ from cell-average values of the solution known on children-nodes at grid level $l + 1$ for a 2D problem.

We assume that we already have a samurai mesh with several level defined in the variable `mesh`. A multiresolution mesh stores several sets of cells, selected by a `mesh_id_t` value: `mesh[mesh_id_t::all_cells][level]` returns all the cells of level `level`, including the ghost cells. We also assume that we created a field `u` on this mesh using `make_scalar_field` and initialized it.

The following steps describe how to implement the projection operator with samurai.

- Create a subset of the mesh using set algebra

```cpp
using mesh_id_t = typename std::decay_t<decltype(mesh)>::mesh_id_t;
auto set = samurai::intersection(mesh[mesh_id_t::all_cells][level], mesh[mesh_id_t::all_cells][level + 1]).on(level);
```

- Apply an operator on this subset

```cpp
set([&](const auto& i, const auto index)
{
    auto j = index[0];
    u(level, i, j) = 0.25*(u(level+1, 2*i, 2*j)
                         + u(level+1, 2*i+1, 2*j)
                         + u(level+1, 2*i, 2*j+1)
                         + u(level+1, 2*i+1, 2*j+1));
});
```

The multi dimensional projection operator can be found [here](./include/samurai/numeric/projection.hpp).

### There's more

If you want to learn more about samurai skills by looking at examples, we encourage you to browse the [demos](./demos) directory.

The [tutorial](./demos/tutorial/) directory is a good first step followed by the [FiniteVolume](./demos/FiniteVolume/) directory.

## Features

- [x] Facilitate data manipulation by using the formalism on a uniform Cartesian grid
- [x] Facilitate the implementation of complex operators between grid levels
- [x] High memory compression of an adapted mesh
- [x] Complex mesh creation using a set of meshes
- [x] Finite volume methods using flux construction
- [x] Lattice Boltzmann methods examples
- [ ] Finite difference methods
- [ ] Discontinuous Galerkin methods
- [x] Matrix assembling of the discrete operators using PETSc
- [x] AMR cell-based methods
- [ ] AMR patch-based and block-based methods
- [x] MRA cell-based methods
- [ ] MRA point-based methods
- [x] HDF5 output format support
- [x] MPI implementation

## Installation

### From conda

```bash
mamba install samurai
```

For compiling purposes, you have to install a C++ compiler, `cmake`, and (optionaly) `make`:

```bash
mamba install cxx-compiler cmake [make]
```

If you have to use PETSc to assemble the matrix of your problem, you need to install it:

```bash
mamba install petsc pkg-config
```

For parallel computation,

```bash
mamba install libboost-mpi libboost-devel libboost-headers 'hdf5=*=mpi*'
```

### From Spack

```bash
spack install samurai
spack load samurai
```

The variants `+mpi`, `+openmp` and `+check_nan` enable MPI, OpenMP and NaN checks, for example `spack install samurai +mpi`. `spack info samurai` lists the available versions and variants.

### From source

Run the cmake configuration

- With mamba or conda

    First, you need to create the environment with all the dependencies
    installed, run

    ```bash
    mamba env create --file conda/environment.yml
    ```

    for sequential computation, or

    ```bash
    mamba env create --file conda/mpi-environment.yml
    ```

    for parallel computation. Then activate the environment

    ```bash
    mamba activate samurai-env
    ```

    (`samurai-mpi-env` for the parallel environment), and run

    ```bash
    cmake . -B build -DCMAKE_BUILD_TYPE=Release -DBUILD_DEMOS=ON
    ```

    Add `-DWITH_MPI=ON` to build the parallel version.

- With vcpkg

    ```bash
    cmake . -B ./build -DENABLE_VCPKG=ON -DBUILD_DEMOS=ON
    ```

Build the demos

```bash
cmake --build ./build --config Release
```

## CMake configuration

Here is a minimal example of `CMakeLists.txt`:

```cmake
cmake_minimum_required(VERSION 3.16)
project(my_samurai_project CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(samurai CONFIG REQUIRED)

add_executable(my_samurai_project main.cpp)
target_link_libraries(my_samurai_project PRIVATE samurai::samurai)
```

samurai requires a C++20 compiler.
MPI and PETSc are disabled by default.
To enable them, set these options before `find_package(samurai)`:

```cmake
set(SAMURAI_WITH_MPI ON)    # requires a parallel HDF5 and Boost.MPI
set(SAMURAI_WITH_PETSC ON)  # requires PETSc and pkg-config
```

The MPI load balancing module can optionally use external graph partitioners.
They are options of the samurai build itself, so set them when you configure samurai from source.
Both require `-DWITH_MPI=ON` and the corresponding library (available in
conda-forge as `parmetis` / `ptscotch`):

```bash
cmake . -B build -DWITH_MPI=ON -DSAMURAI_WITH_PARMETIS=ON  # enable the ParMETIS (Metis) strategy
cmake . -B build -DWITH_MPI=ON -DSAMURAI_WITH_PTSCOTCH=ON  # enable the PT-Scotch (Scotch) strategy
```

## Get help

For a better understanding of all the components of samurai, you can consult the [samurai documentation](https://hpc-math-samurai.readthedocs.io).

If you have any question or remark, you can write a message on [github discussions](https://github.com/hpc-maths/samurai/discussions) and we will be happy do help you or to discuss with you.

## Project assistance

If you want to say **thank you** or/and support active development of samurai:

- Add a [GitHub Star](https://github.com/hpc-maths/samurai) to the project.
- Tweet about samurai.
- Write interesting articles about the project on [Dev.to](https://dev.to/), [Medium](https://medium.com/) or your personal blog.

Together, we can make samurai **better**!

## Contributing

First off, thanks for taking the time to contribute! Contributions are what make the open-source community such an amazing place to learn, inspire, and create. Any contributions you make will benefit everybody else and are **greatly appreciated**.

Please read [our contribution guidelines](./docs/CONTRIBUTING.md), and thank you for being involved!

## License

This project is licensed under the **BSD-3-Clause license**.

See [LICENSE](LICENSE) for more information.
