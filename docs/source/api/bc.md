# Boundary conditions

This page documents the boundary conditions of {{ project }}: the `samurai::Dirichlet` and `samurai::Neumann` types, the `samurai::PolynomialExtrapolation` that fills the outer ghosts no boundary condition reaches, and the content of `samurai/bc/bc.hpp`, which holds `samurai::make_bc`, the `samurai::Bc` base class, the boundary values and the boundary regions.
`make_bc<Type>(field, ...)` attaches a boundary condition to a field and returns a pointer to it; `->on(...)` then restricts it to a region of the boundary.
Include `samurai/bc.hpp` to get all of them.
For examples of each use, see the {doc}`boundary condition reference <../reference/bc>`.

## Boundary condition types

| Type | Orders | Stencil size |
| --- | --- | --- |
| `samurai::Dirichlet<order>` | 1 (default) to 4 | `2 * order` |
| `samurai::Neumann<order>` | 1 (default) | `2 * order` |
| `samurai::PolynomialExtrapolation<Field, stencil_size>` | - | 2, 4 or 6 |

`Dirichlet` and `Neumann` are passed to `make_bc` as its template argument, for example `samurai::make_bc<samurai::Dirichlet<2>>(u, 0.)`.
Their `impl_t` member is the boundary condition class that `make_bc` builds for the field.
`make_bc` throws `std::invalid_argument` when the stencil size is larger than the max stencil size of the mesh.

```{doxygenstruct} samurai::Dirichlet
:members:
:undoc-members:
```

```{doxygenstruct} samurai::Neumann
:members:
:undoc-members:
```

{{ project }} applies `PolynomialExtrapolation` itself, with no call to `make_bc`: it fills the outer ghost layers beyond those the boundary condition of the field fills, and the outer ghosts at the domain corners.

```{doxygenstruct} samurai::PolynomialExtrapolation
:members:
```

## make_bc, values and regions

The value of a boundary condition depends on the arguments given to `make_bc` after the field:

- none: the constant 0 on every component;
- one value per component of the field: a constant;
- a function of the outward direction, the boundary cell and the center of the boundary face: a value that varies along the boundary.

`Bc::on` takes one or more direction vectors, a set expression, a predicate on the face-center coordinates, or a region built by `make_bc_region`.
To select the boundary faces by their coordinates, pass a predicate `func` that returns `true` for the face centers in the region; `->on(func)` is equivalent to `->on(samurai::make_bc_region(mesh, func))`.

```{doxygenfile} bc.hpp
```
