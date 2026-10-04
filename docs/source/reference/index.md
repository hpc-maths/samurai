# Reference

The reference pages describe the behavior of {{ project }} features: what each one does, which options it takes and which rules it follows.
Open them while you work, in any order.
To learn the library step by step, follow the {doc}`tutorials <../tutorial/index>`; to carry out a precise task, see the {doc}`how-to guides <../howto/index>`.
For the C++ classes and functions themselves, see the {doc}`API reference <../api/index>`.

## Mesh

- {doc}`subset`: the operations that combine the levels of a mesh into set expressions, at compile time or at runtime.

```{toctree}
:caption: Mesh
:hidden:

subset
```

## Numerical schemes

- {doc}`bc`: the Dirichlet, Neumann and user-defined conditions, and the extrapolation that fills the other ghosts outside the domain.
- {doc}`finite_volume_schemes`: the finite volume module, its discrete operators and the flux functions you can define.
- {doc}`local_schemes`: schemes evaluated cell by cell, with no flux between cells.

```{toctree}
:caption: Numerical schemes
:hidden:

bc
finite_volume_schemes
local_schemes
```

## Parallel computing

- {doc}`load_balancing`: how the cells and their fields are redistributed between MPI processes after the mesh adapts.

```{toctree}
:caption: Parallel computing
:hidden:

load_balancing
```
