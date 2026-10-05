# How-to guides

The how-to guides show you how to carry out one task with {{ project }}, step by step.
Each guide assumes you know what you want to do; to learn the library from the start, follow the {doc}`tutorials <../tutorial/index>` first.
For the full description of a feature, see the {doc}`reference <../reference/index>` or the {doc}`API reference <../api/index>`.

If the task you need is not covered, [open an issue](https://github.com/hpc-maths/samurai/issues) to ask for a guide, or contribute one with a pull request.

## Install and build

- {doc}`installation`: install {{ project }} with conda or spack, or build it from source, for sequential or MPI computation.
- {doc}`cmake`: build your own program against {{ project }} and turn on MPI, PETSc, OpenMP or NaN checks.

```{toctree}
:caption: Install and build
:hidden:

installation
cmake
```

## Mesh and data

- {doc}`box`: define the domain of the mesh from one box or from several boxes, with holes.
- {doc}`geometry`: build a domain around obstacles given as boxes, as a level-set function or as a 3D surface file.
- {doc}`mesh`: create a uniform, multiresolution or adaptive mesh refinement (AMR) mesh, and set its levels and ghosts.
- {doc}`field`: create a scalar or vector field on a mesh and give it initial values.
- {doc}`loop`: visit the cells of a mesh one cell or one interval at a time.

```{toctree}
:caption: Mesh and data
:hidden:

box
geometry
mesh
field
loop
```

## Adaptation and solvers

- {doc}`adapt`: make a multiresolution mesh follow your solution.
- {doc}`bc`: set Dirichlet, Neumann, periodic or user-defined boundary conditions on a field.
- {doc}`petsc`: solve an implicit scheme with PETSc, sequentially or with MPI.

```{toctree}
:caption: Adaptation and solvers
:hidden:

adapt
bc
petsc
```

## Input and output

- {doc}`save`: write a mesh and its fields to HDF5 and XDMF files for post-processing.
- {doc}`restart`: write a checkpoint of a running simulation and start a new program from it.
- {doc}`plot`: open the files a {{ project }} program writes in ParaView, Matplotlib or yt.

```{toctree}
:caption: Input and output
:hidden:

save
restart
plot
```

## Parallel

- {doc}`mpi`: build a program with MPI, run it on several processes and check its result against one process.

```{toctree}
:caption: Parallel
:hidden:

mpi
```

## Run and measure

- {doc}`options`: read the predefined command-line options of {{ project }} and add your own.
- {doc}`timers`: find out where a program spends its time.

```{toctree}
:caption: Run and measure
:hidden:

options
timers
```

## Troubleshooting

- {doc}`troubleshooting`: find the cause and the fix of a common error message, from the CMake configuration to the run of your program.

```{toctree}
:caption: Troubleshooting
:hidden:

troubleshooting
```
