# Getting started: a first adaptive simulation

In this tutorial, we write, build and run a {{ project }} program that transports a disc across a 2D domain on an adaptive mesh.
{{ project }} refines the mesh around the edge of the disc and coarsens it everywhere else, at every time step.
At the end, we open the solution in ParaView and look at the value of the solution and the level of each cell.

{{ project }} is a header-only C++ library for adaptive Cartesian meshes.
It stores the mesh as intervals of cells, adapts it with multiresolution or adaptive mesh refinement (AMR), and provides finite volume schemes that work on the adapted mesh.

## Before you start

You need:

- {{ project }} installed. The shortest route is conda: follow {ref}`install-with-conda` in the [installation guide](../howto/installation.md).
- A C++20 compiler, CMake 3.16 or later and a build tool such as Make. The installation guide installs them with conda.
- [ParaView](https://www.paraview.org) to look at the results.

The program is a CMake project like the one of the [CMake project how-to](../howto/cmake.md).
We give every file and command here, so you do not need to read the how-to first.
Read it if the configuration step fails.

We assume you know C++ (templates, `auto` and lambdas) and the idea of a finite volume scheme.

## The problem

We solve the advection equation

$$
\partial_t u + \mathbf{a} \cdot \nabla u = 0, \qquad \mathbf{a} = (1, 1),
$$ (tutorial-getting-started-advection)

on the unit square $[0, 1]^2$, up to the final time $T = 0.3$.
At $t = 0$, $u = 1$ inside the disc of center $(0.3, 0.3)$ and radius $0.2$, and $u = 0$ outside.
On the boundary, $u = 0$.
The exact solution is the same disc, moved by $\mathbf{a} t$: at $t = 0.3$ its center is at $(0.6, 0.6)$.

We discretize {eq}`tutorial-getting-started-advection` with an explicit Euler scheme in time and an upwind scheme in space.

## Create the project

1. Create an empty directory for the project and go into it:

   ```bash
   mkdir getting_started
   cd getting_started
   ```

2. Create a file named `CMakeLists.txt` with this content:

   ```cmake
   cmake_minimum_required(VERSION 3.16)
   project(getting_started)

   find_package(samurai REQUIRED)

   add_executable(getting_started getting_started.cpp)
   target_link_libraries(getting_started PRIVATE samurai::samurai)
   ```

3. Create a file named `getting_started.cpp` and copy the program into it.
   It is the file `demos/tutorial/getting_started.cpp` of the {{ project }} repository:

   ```{literalinclude} ../../../demos/tutorial/getting_started.cpp
   :language: c++
   ```

The next section goes through this program step by step.
If you want to see it run first, go to [Build and run](#build-and-run) and come back afterward.

## Read the program

### Initialize samurai

```{literalinclude} ../../../demos/tutorial/getting_started.cpp
:language: c++
:start-at: samurai::initialize
:end-at: SAMURAI_PARSE
:dedent:
```

`samurai::initialize` reads the command-line options of {{ project }}, such as `--max-level` or `--mr-eps`, and `SAMURAI_PARSE` parses the command line.
`samurai::finalize`, at the end of `main`, closes what `samurai::initialize` opened.

### Create the mesh

```{literalinclude} ../../../demos/tutorial/getting_started.cpp
:language: c++
:start-after: // Create the mesh
:end-before: // Create the field
:dedent:
```

The box is the unit square.
The mesh configuration gives the range of levels: a cell at level $\ell$ has a side of $2^{-\ell}$, so the cells go from $1/16$ (level 4) to $1/256$ (level 8).
`samurai::mra::make_mesh` builds a mesh for multiresolution adaptation, uniform at the finest level to start with.

### Create the field

```{literalinclude} ../../../demos/tutorial/getting_started.cpp
:language: c++
:start-after: // Create the field
:end-before: // Adapt the mesh
:dedent:
```

`samurai::make_scalar_field` creates a field with one value per cell of the mesh.
`samurai::for_each_cell` calls the lambda on each cell, and `u[cell]` is the value of the field in that cell: 1 if the center of the cell is inside the disc, 0 otherwise.

`samurai::make_bc` attaches a boundary condition to the field: here a Dirichlet condition $u = 0$ on the whole boundary.
{{ project }} uses it to fill the ghost cells outside the domain, which the scheme reads at the boundary.

### Adapt the mesh

```{literalinclude} ../../../demos/tutorial/getting_started.cpp
:language: c++
:start-after: // Adapt the mesh
:end-before: // Solve the advection equation
:dedent:
```

`samurai::make_MRAdapt` creates the multiresolution adaptation for the field `u`.
Each call to `MRadaptation(mra_config)` computes, in each cell, the difference between the value of `u` and the value predicted from the coarser level, called the detail.
Cells whose detail is small compared to the threshold `epsilon` are merged into coarser cells; cells with a large detail are kept or refined.
Where `u` is constant, the details are zero and the mesh goes down to level 4; along the edge of the disc, it stays at level 8.
The field values follow the mesh: `u` is projected or predicted on the new cells.

The `save` function, at the top of the file, writes `u` and a field `level` that holds the level of each cell.
We call it once here, to keep the initial state.

### Solve the advection equation

```{literalinclude} ../../../demos/tutorial/getting_started.cpp
:language: c++
:start-after: // Solve the advection equation
:end-before: samurai::finalize
:dedent:
```

The time step comes from the finest cells: `mesh.min_cell_length()` is the side of a cell at level 8.
The last time step is shortened so that the loop stops at $t = 0.3$.

Each iteration:

1. adapts the mesh to the current solution;
2. fills the ghost cells with `samurai::update_ghost_mr`, from the boundary condition and from the neighboring levels;
3. resizes `unp1` to the new mesh and computes $u^{n+1} = u^n - \Delta t \, \mathrm{div}_h(\mathbf{a} u^n)$, where `samurai::upwind(velocity, u)` is the upwind approximation of the flux divergence, computed on each cell at its own level;
4. swaps the values of `u` and `unp1`, so that `u` holds the new solution.

After the loop, `save` writes the final state.

## Build and run

1. Configure the project from the `getting_started` directory.
   With a conda installation, activate the environment where you installed {{ project }} first, then run:

   ```bash
   cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_PREFIX_PATH=$CONDA_PREFIX
   ```

   If you installed {{ project }} another way, replace `$CONDA_PREFIX` with its installation prefix.
   If CMake stops with `Could not find a package configuration file provided by "samurai"`, the prefix does not point to the {{ project }} installation.

2. Build the program:

   ```bash
   cmake --build build
   ```

   The first build takes a while: the compiler instantiates the {{ project }} templates for this program.

3. Run it:

   ```bash
   ./build/getting_started
   ```

   The program prints one line per time step, 154 lines in all:

   ```text
   iteration 0: t = 0.001953, dt = 0.001953
   iteration 1: t = 0.003906, dt = 0.001953
   iteration 2: t = 0.005859, dt = 0.001953
   ...
   iteration 151: t = 0.296875, dt = 0.001953
   iteration 152: t = 0.298828, dt = 0.001953
   iteration 153: t = 0.300000, dt = 0.001172
   ```

   When it ends, the directory contains `getting_started_init.xdmf` and `getting_started.xdmf`, each with its `.h5` data file.

:::{note}
If you built {{ project }} from source, you can also build this program in the build directory of the repository, without a project of your own:

```bash
cmake --build build --target tutorial-getting-started
./build/demos/tutorial/tutorial-getting-started
```

:::

## Plot the result

1. In ParaView, choose *File > Open* and select `getting_started_init.xdmf` and `getting_started.xdmf`.
2. Click *Apply*.
3. Choose `u` in the coloring menu of the toolbar.
   The disc has moved from $(0.3, 0.3)$ to $(0.6, 0.6)$.
   Its edge is smoother than at $t = 0$: the upwind scheme is first order and diffuses the solution.
4. Choose `level` in the coloring menu.
   The finest cells, at level 8, follow the edge of the disc, and the cells far from it are coarser, down to level 4.
   To see the cells themselves, choose *Surface With Edges* in the representation menu.

% TODO: add a figure of `u` and `level` at t = 0.3 from a run of the demo, with alt text.

## Change the parameters

The program reads the options of {{ project }} from the command line.
Run it again with a smaller threshold:

```bash
./build/getting_started --mr-eps 1e-5
```

The band of fine cells around the disc widens, since more details are above the threshold.
With `--max-level 9`, the finest cells are twice as small, and the time step too: the program prints 308 lines.
`./build/getting_started --help` lists all the options.

## What we built

We wrote a finite volume solver for the advection equation on a mesh that {{ project }} adapts at every time step, built it with CMake against the installed library, and looked at the solution and the mesh levels in ParaView.
The adapted mesh has fewer cells than a uniform mesh at level 8, and it is as fine as that mesh where the solution varies.

## Next steps

- Learn how {{ project }} stores meshes and fields in the {doc}`interval <interval>` and {doc}`field <field>` tutorials, then follow the other tutorials of this section.
- Follow the [Hands-on samurai course](https://hpc-maths.github.io/2025-hands-on-samurai/), a guided path from a naive Burgers solver to the Euler equations.
- Browse the [samurai gallery](https://hpc-maths.github.io/samurai-gallery/) for complete simulations to start from.
- Use the how-to guides for a given task: [create a mesh](../howto/mesh.md), [set the command-line options](../howto/options.md), [save the results](../howto/save.md) and [plot them](../howto/plot.md).
- Read `demos/FiniteVolume/advection_2d.cpp` in the repository: it is the same problem with options for the parameters, several output files and restart files.
