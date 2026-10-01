# Hands-on samurai course

[Hands-on samurai](https://hpc-maths.github.io/2025-hands-on-samurai/) is a guided course on finite volume methods with adaptive mesh refinement in {{ project }}.
It starts with a mesh and a field and ends with a 2D solver for the Euler equations on an adaptive mesh.
After the setup, each part gives you a code skeleton to complete and a solution to compare with.

The tutorials on this site explain how {{ project }} works inside: intervals, fields, algorithms on subsets, graduation.
The course uses {{ project }} to build complete simulations, one part at a time.

## Before you start

The course expects you to know:

- basic C++, in particular templates and lambdas;
- the principles of finite volume methods;
- how to build a project with CMake.

To follow it on your machine, you need:

- Git, to clone the course repository;
- conda, Miniconda or Micromamba, to create the course environment;
- ParaView, to look at 2D results;
- Python with matplotlib, to plot 1D results (the course environment installs matplotlib).

## The course environment

The course repository, [hpc-maths/2025-hands-on-samurai](https://github.com/hpc-maths/2025-hands-on-samurai), contains a conda environment file, `conda/environment.yml`.
It creates an environment named `samurai-practical-session` with the following conda-forge packages:

- {{ project }};
- a C++ compiler (`cxx-compiler`) and CMake;
- MPI, Boost.MPI (`libboost-mpi`) and HDF5 built with MPI;
- `h5py` and matplotlib, for the plotting script.

The [setup page](https://hpc-maths.github.io/2025-hands-on-samurai/setup) of the course shows how to create and activate this environment.
It also lists the dependencies to install yourself if you do not use conda.

## Learning path

The course has six parts, followed by a [conclusion](https://hpc-maths.github.io/2025-hands-on-samurai/conclusion) that sums up what you have learned.
Each part reuses what the previous ones built.

| Part | What you build |
| --- | --- |
| [Environment setup](https://hpc-maths.github.io/2025-hands-on-samurai/setup) | A test program built with CMake, a 2D example opened in ParaView and a 1D example plotted with the `read_mesh.py` Python script. |
| [Mesh and fields](https://hpc-maths.github.io/2025-hands-on-samurai/first-step) | 1D and 2D multiresolution meshes and a scalar field initialized with a Gaussian function, then saved and plotted. You loop over the mesh cells and over the intervals of the mesh. |
| [Naive Burgers](https://hpc-maths.github.io/2025-hands-on-samurai/naive-burgers) | A first finite volume scheme for the 1D inviscid Burgers equation, with explicit Euler time steps, then multiresolution adaptation of the mesh. You see why this scheme loses conservation at the interfaces between levels. |
| [Flux mechanism](https://hpc-maths.github.io/2025-hands-on-samurai/flux-burgers) | The Burgers equation rewritten with the {{ project }} flux mechanism, in 1D and 2D: a conservative flux, a non-conservative flux for the 2D vector Burgers equation and a diffusion operator for the viscous case, tested on a Taylor-Green vortex. |
| [Euler, part 1](https://hpc-maths.github.io/2025-hands-on-samurai/euler-1) | A 2D solver for the compressible Euler equations with the Rusanov, HLL and HLLC schemes. The starting files provide the conversions between primitive and conservative variables, the equation of state and the CFL time step. The solver runs a 2D Riemann problem, first on a uniform mesh, then with adaptation. |
| [Euler, part 2](https://hpc-maths.github.io/2025-hands-on-samurai/euler-2) | The double Mach reflection test case, with custom boundary conditions that depend on position and time, and a custom prediction operator that keeps density and pressure positive when the mesh is refined. |

## Where the solutions are

Each part after the setup comes with its solution, in the course repository, under [`practical_session/material`](https://github.com/hpc-maths/2025-hands-on-samurai/tree/main/practical_session/material):

- the starting files of a part are in `practical_session/material/<part>/` (a `CMakeLists.txt`, a main source file and, for the Euler parts, an `euler` folder);
- its solution is in `practical_session/material/<part>/solution/`.

For example, the solution of the mesh and fields part is in `practical_session/material/01-first-step/solution/`.
Both Euler parts share the `04-euler` folder, so their solutions are in `practical_session/material/04-euler/solution/`.
The setup part has no solution folder: its program is ready to build.

## Next steps

- If you want to use {{ project }} in your own project rather than in the course environment, see the {doc}`installation guide <../howto/installation>`.
- To plot your results, see the {doc}`plotting guide <../howto/plot>`.
- To write your own flux functions or use the built-in finite volume operators, see the {doc}`finite volume schemes reference <../reference/finite_volume_schemes>`.
