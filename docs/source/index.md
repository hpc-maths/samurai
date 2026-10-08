# {{ project }} documentation

{{ project }} stands for Structured Adaptive mesh and MUlti-Resolution based on Algebra of Intervals.
It is a header-only C++ library that stores adaptive Cartesian meshes as sets of intervals and adapts them with multiresolution or adaptive mesh refinement (AMR).
{ref}`plate-adaptive-mesh` shows such a mesh and the intervals that store one of its rows.

```{plate} The adaptive mesh and its intervals
:figure: adaptive_mesh
:label: plate-adaptive-mesh

**Fig. 1.** The unit square, meshed between level 2 and level 6: a cell of level $\ell$ has the side $1/2^\ell$.
The mesh is refined where a circle crosses it.
*b*, the cells of the finest level, hatched in [red]{.sm-red}, follow the circle.
*c*, away from the circle the cells coarsen one level at a time, down to level 3.
Two touching cells never differ by more than one level: the mesh is graded.
To keep it so, the grading splits every cell of level 2, and none is left.
*d*, inside the disc, ruled in fine lines, the cells are coarse again.

**Fig. 2.** One cell of each level, at the scale of Fig. 1: each level halves the side of the level before.

**Fig. 3.** Section *a-a*: the cells that the row *a-a* of Fig. 1 crosses, level by level.
Each level stores them as half-open intervals of indices, counted in cells of that level:
$[10, 12)$ and $[48, 50)$ at level 6,
$[4, 5)$, $[6, 8)$, $[22, 24)$ and $[25, 26)$ at level 5,
$[0, 2)$, $[4, 6)$, $[8, 11)$ and $[13, 14)$ at level 4,
$[3, 4)$ and $[7, 8)$ at level 3.
Level 2 has no cell in this row.
Together the levels cover the row exactly once.
```

```{toctree}
:hidden:
:maxdepth: 1

tutorial/index
howto/index
Philosophy <philosophy>
reference/index
API <api/index>
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
- {doc}`demos`: the complete programs of the repository, grouped by method (finite volume, lattice Boltzmann, multigrid, MPI), with the command that builds and runs each one.
- {doc}`gallery`: complete simulations, with videos of the solution and the mesh, to start from.

## Contributing

- [Contributing guide](https://github.com/hpc-maths/samurai/blob/main/docs/CONTRIBUTING.md): how to report a bug, set up a development environment and propose a change.
