# {{ project }} documentation

{{ project }} stands for Structured Adaptive mesh and MUlti-Resolution based on Algebra of Intervals.
It is a header-only C++ library that stores adaptive Cartesian meshes as sets of intervals and adapts them with multiresolution or adaptive mesh refinement (AMR).

```{toctree}
:hidden:
:maxdepth: 1

tutorial/index
howto/index
philosophy
reference/index
api/index
LBM/index
```

## Start here

- {doc}`Getting started <tutorial/getting_started>`: write, build and run a first adaptive simulation.
- {doc}`Hands-on samurai course <tutorial/hands_on>`: a guided course that goes from a mesh and a field to a 2D Euler solver on an adaptive mesh.

## Learn and use {{ project }}

- {doc}`tutorial/index`: how {{ project }} works inside: intervals, fields, set algebra and graduation.
- {doc}`howto/index`: steps for one task, such as installing {{ project }}, adapting a mesh or saving a field.
- {doc}`philosophy`: why {{ project }} is built on intervals and set algebra, and what this design costs.
- {doc}`reference/index`: the full description of boundary conditions, set algebra, schemes and load balancing.
- {doc}`api/index`: the C++ classes and functions, generated from the headers.

## Examples and applications

- {doc}`LBM/index`: lattice Boltzmann schemes on multiresolution meshes, with the test cases that validate them.
- [Demos](https://github.com/hpc-maths/samurai/tree/main/demos): complete programs in the repository, grouped by method (finite volume, lattice Boltzmann, WENO, multigrid, MPI).
- [samurai gallery](https://hpc-maths.github.io/samurai-gallery/): complete simulations to start from.

## Contributing

- [Contributing guide](https://github.com/hpc-maths/samurai/blob/main/docs/CONTRIBUTING.md): how to report a bug, set up a development environment and propose a change.
