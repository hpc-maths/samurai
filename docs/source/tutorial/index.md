# Tutorials

The tutorials teach {{ project }} by example.
Each one builds a program step by step and explains what it does to the mesh and the fields.
To carry out a precise task, see the {doc}`how-to guides <../howto/index>`.
To look up how a feature behaves, see the {doc}`reference <../reference/index>` or the {doc}`API reference <../api/index>`.

## Start here

- {doc}`getting_started`: install {{ project }}, then write, build and run a program that transports a disc on an adaptive mesh.
- {doc}`hands_on`: the guided path through {{ project }}.
  The course starts with a mesh and a field and ends with a 2D solver for the Euler equations on an adaptive mesh, one part at a time.

```{toctree}
:caption: Start here
:hidden:

getting_started
hands_on
```

## Data structure

These tutorials explain how {{ project }} stores a mesh and its fields, and how a program walks through them.
Each one builds on the previous one.

- {doc}`interval`: how a Cartesian mesh is stored as intervals of integers, level by level.
- {doc}`field`: how the values of a field are stored on these intervals, and how to write them cell by cell or interval by interval.
- {doc}`algorithm`: when to visit a mesh one cell at a time and when one interval at a time.
- {doc}`operator_on_subset`: how to select a part of the mesh with the set algebra and compute on it only.
- {doc}`graduation`: how to keep neighboring cells within one level of each other, written with the set algebra.

```{toctree}
:caption: Data structure
:hidden:

interval
field
algorithm
operator_on_subset
graduation
```

## Solvers

These tutorials write a complete solver on an adaptive mesh.

- {doc}`1d_burgers_amr`: an explicit upwind scheme that runs from a hat function until after a shock forms, with a refinement criterion that keeps small cells where the solution varies.
- {doc}`level_set`: the transport of a level-set function in 2D, with the mesh adapted around the contour at each time step.
- {doc}`reaction_diffusion`: implicit time schemes for a reaction-diffusion equation, solved with PETSc.

```{toctree}
:caption: Solvers
:hidden:

1d_burgers_amr
level_set
reaction_diffusion
```
