# API reference

The API reference lists the C++ classes, structs and functions of {{ project }}, generated from the comments in the headers.
Each page covers one header or one group of headers and links the guide that shows them in use.
To learn the library, follow the {doc}`tutorials <../tutorial/index>`; to carry out a precise task, see the {doc}`how-to guides <../howto/index>`.
For the behavior of a feature as a whole, see the {doc}`reference <../reference/index>`.

## Geometry and cells

- {doc}`box`: `samurai::Box`, the hyperrectangle a mesh is built on.
- {doc}`interval`: `samurai::Interval`, the run of cells from which meshes are built.
- {doc}`cell`: `samurai::Cell`, a cell of the mesh with its level, coordinates and storage index.

```{toctree}
:caption: Geometry and cells
:hidden:

box
interval
cell
```

## Meshes and fields

- {doc}`mesh`: the uniform, multiresolution and AMR meshes, their configuration, and the cell arrays and cell lists they are made of.
- {doc}`field`: `samurai::ScalarField`, `samurai::VectorField` and the functions that create them.

```{toctree}
:caption: Meshes and fields
:hidden:

mesh
field
```

## Sets and loops

- {doc}`subset`: `samurai::SetBase` and the functions that build set expressions.
- {doc}`algorithm`: the loops over the levels, intervals and cells of a mesh, and the functions that find a cell.

```{toctree}
:caption: Sets and loops
:hidden:

subset
algorithm
```

## Numerical schemes

- {doc}`bc`: the boundary condition types and `samurai::make_bc`.

```{toctree}
:caption: Numerical schemes
:hidden:

bc
```
