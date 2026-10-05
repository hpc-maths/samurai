# How-to: set boundary conditions on a field

This guide shows you how to give a field its values outside the domain: a Dirichlet or Neumann condition with a constant or a function value, a condition on part of the boundary, a periodic direction, and a condition you write yourself.
You attach each condition to the field with `samurai::make_bc`, and {{ project }} applies it to the ghosts outside the domain each time they are updated.

## Before you start

- You have a mesh and a field on it. If not, see the [mesh how-to guide](mesh.md) and the [field how-to guide](field.md).
- Include `samurai/bc.hpp`.
- A periodic direction is set when you create the mesh, not on the field: see [Make a direction periodic](#make-a-direction-periodic) before you build the mesh.

The examples of this guide live in `docs/source/howto/snippet/bc/`.
Each one builds a mesh of 4 by 4 cells on the unit square, attaches conditions, fills the ghosts and prints the first layer of ghosts along some sides of the square, with the center of each ghost and its value.
The cells have length $h = 0.25$, so the ghosts on the left side have their centers at $x = -0.125$.

## Impose a value: Dirichlet

Pass the condition type as a template argument and the value after the field:

```{literalinclude} snippet/bc/bc_dirichlet_neumann.cpp
  :language: c++
  :start-at: // u = 0 on the left side
  :end-at: ->on(left);
  :dedent:
```

`samurai::Dirichlet<1>` sets the value of the field on the boundary face: the ghost receives $u_g = 2v - u_0$, where $v$ is the value and $u_0$ the value of the boundary cell, so that the average of $u_0$ and $u_g$ equals $v$.
Here `->on(left)` restricts the condition to the left side, as explained in [Apply a condition to part of the boundary](#apply-a-condition-to-part-of-the-boundary).
Without `->on(...)`, the condition applies to the whole boundary.

For a vector field, give one value per component:

```{literalinclude} snippet/bc/bc_dirichlet_neumann.cpp
  :language: c++
  :start-at: // A vector field
  :end-at: (v, 0., 2.);
  :dedent:
```

The number of values must equal the number of components of the field; a mismatch fails at compile time.
`samurai::make_bc<samurai::Dirichlet<1>>(u)` without a value imposes 0 on every component.

`samurai::Dirichlet<order>` accepts the orders 1 to 4 and fills `order` ghost layers.
The default mesh configuration has room for orders 1 and 2.
For order 3 or 4, raise the max stencil radius of the mesh to the order, with `max_stencil_radius(3)` in the mesh configuration or `--max-stencil-radius 3` on the command line; otherwise `make_bc` throws the error described in the [troubleshooting guide](troubleshooting.md#boundary-condition-stencil-larger-than-the-mesh-allows).

## Impose a value that depends on the position

Pass a function instead of a constant.
It receives the outward normal of the boundary, the cell inside the domain that touches the boundary, and the center of the boundary face, and returns the value:

```{literalinclude} snippet/bc/bc_dirichlet_neumann.cpp
  :language: c++
  :start-at: // u = y on the right side
  :end-at: ->on(right);
  :dedent:
```

For a vector field, the function returns a `samurai::Array<value_type, n_comp>`.

{{ project }} calls the function for each boundary face each time it fills the ghosts.
To impose a value that depends on time, capture the time variable by reference, as `demos/FiniteVolume/heat_nonlinear.cpp` does with `[&]`.
To replace the conditions of a field, clear them with `u.get_bc().clear()` and attach the new ones; the same demo does this at each time step.

## Impose a normal derivative: Neumann

`samurai::Neumann<1>` sets the derivative of the field along the outward normal of the boundary:

```{literalinclude} snippet/bc/bc_dirichlet_neumann.cpp
  :language: c++
  :start-at: // du/dn = 0
  :end-at: ->on(bottom, top);
  :dedent:
```

The ghost receives $u_g = u_0 + h\,v$, where $v$ is the value of the condition and $h$ the cell length.
The derivative is taken along the **outward** normal: for $u(x) = x$ on $[0, 1]$, the matching values are $v = -1$ on the left side and $v = 1$ on the right side.
Only order 1 exists.

## Check the ghost values

Fill the ghosts with `samurai::update_ghost_mr`, then read them on the `reference` cells of the mesh, which include the ghosts.
The function `print_ghosts` of the examples prints the first layer of ghosts on one side of the unit square:

```{literalinclude} snippet/bc/print_ghosts.hpp
  :language: c++
  :start-at: // Print the first layer
```

For `bc_dirichlet_neumann.cpp`, where $u = 1$ and $\mathbf{v} = (1, 1)$ in every cell:

```bash
./bc_dirichlet_neumann
```

```{literalinclude} snippet/bc/bc_dirichlet_neumann_output.txt
  :language: text
```

- On the left side, $u_g = 2 \times 0 - 1 = -1$: the face value is 0.
- On the right side, $u_g = 2y - 1$ at the height $y$ of each face.
- On the bottom side, the zero normal derivative copies the cell value, 1.
- For `v`, each component follows its own value: $2 \times 0 - 1 = -1$ and $2 \times 2 - 1 = 3$.

You do not need to update the ghosts yourself before a finite volume scheme: an explicit scheme updates the ghosts of its input field when they are out of date.
Before a loop of your own that reads the ghosts, call `samurai::update_ghost_mr`.
On an adapted mesh, the update applies the conditions at every level that holds cells.

## Apply a condition to part of the boundary

`->on(...)` on the pointer that `make_bc` returns restricts the condition to a region of the boundary.
The region can be given in three ways.

### By side

Pass one or more outward normals, as `samurai::DirectionVector<dim>` values: `{-1, 0}` for the left side, `{1, 0}` for the right side, `{0, -1}` for the bottom and `{0, 1}` for the top in 2D.
The example above uses `->on(left)`, `->on(right)` and `->on(bottom, top)`.

### By position

Build a region from a predicate on the center of the boundary faces, with `samurai::make_bc_region(mesh, predicate)`:

```{literalinclude} snippet/bc/bc_regions.cpp
  :language: c++
  :start-at: // u = 0 on the whole boundary
  :end-at: ->on(upper);
  :dedent:
```

The conditions of a field apply in the order you attach them, so where two regions overlap, the condition attached last sets the ghosts.
Here the first condition covers the whole boundary and the second one overrides it where $y > 0.5$.
The output shows $u_g = 0$ on the lower half of the left side and $u_g = 2 \times 1 - 0 = 2$ on the upper half:

```{literalinclude} snippet/bc/bc_regions_output.txt
  :language: text
  :end-before: w, left
```

### By cells

Pass a set expression: the condition applies to the cells of the set that touch the boundary, on every side they touch.
The example selects the column of cells along the left side:

```{literalinclude} snippet/bc/bc_regions.cpp
  :language: c++
  :start-at: // w = 1 on the boundary
  :end-at: ->on(left_column);
  :dedent:
```

The bottom-left and top-left cells of the column also touch the bottom and top sides, so the condition fills their ghosts there too:

```{literalinclude} snippet/bc/bc_regions_output.txt
  :language: text
  :start-at: w, left
```

The bottom ghosts outside the region keep the value they had: no condition writes them.
Cover the whole boundary with conditions, as the previous example does with its first `make_bc`, unless your scheme never reads those ghosts.

## Make a direction periodic

Periodicity belongs to the mesh.
Set it in the mesh configuration, before you create the mesh:

```{literalinclude} snippet/bc/bc_periodic.cpp
  :language: c++
  :start-at: // Periodic in x
  :end-at: make_mesh
  :dedent:
```

`periodic(true)` makes every direction periodic, and `periodic({true, false})` only the x-direction.
The ghosts of a periodic direction receive the values of the cells on the opposite side of the domain.
A condition attached to the field is not applied in a periodic direction, so you can keep one `make_bc` call for the whole boundary.
With $u = x$ and a Dirichlet condition of value 0:

```{literalinclude} snippet/bc/bc_periodic_output.txt
  :language: text
```

The left ghosts hold the value of the last column ($x = 0.875$), the right ghosts the value of the first column ($x = 0.125$), and the bottom ghosts follow the Dirichlet condition, $u_g = -x$.

A domain built from several boxes with a `DomainBuilder` cannot be periodic, as the [troubleshooting guide](troubleshooting.md) explains.

## Write your own condition

When neither Dirichlet nor Neumann fits, write a class derived from `samurai::Bc<Field>`.
The example implements the Robin condition $u + \partial u / \partial \mathbf{n} = g$, discretized on the boundary face as

$$
\frac{u_0 + u_g}{2} + \frac{u_g - u_0}{h} = g.
$$

```{literalinclude} snippet/bc/bc_custom.cpp
  :language: c++
  :start-at: // Robin condition
  :end-before: int main
```

1. Start the class with `INIT_BC(Name, stencil_size)`.
   It declares the constructors and the types the other members use: `stencil_t`, `constant_stencil_size_t`, `stencil_cells_t`, `apply_function_t`, `value_t` and `direction_t`.
   The condition fills `stencil_size / 2` ghost layers, here one.
2. Return the stencil from `get_stencil`, written for the right side of the domain: `{0, 0}` is the boundary cell and `{1, 0}` the ghost next to it.
   {{ project }} rotates it for the other sides.
   Keep `{0, 0}` in the stencil if the condition takes a function value, because the function receives that cell.
3. Return from `get_apply_function` a function that writes the ghosts.
   It receives the field, the cells of the stencil in the order of the stencil, and the value of the condition on this face.
4. Declare a tag with a member alias template `impl_t`, as `samurai::Dirichlet` does, so that `make_bc<Robin>` works like `make_bc<samurai::Dirichlet<1>>`.
   Without a tag, pass the class template itself: `samurai::make_bc<RobinImpl>(u, g)`, as `demos/FiniteVolume/advection_2d_user_bc.cpp` does.

Attach it as any other condition.
With $u = x$, the value $g = u + \partial u / \partial \mathbf{n}$ is $x + n_x$, where $n_x$ is the x-component of the outward normal:

```{literalinclude} snippet/bc/bc_custom.cpp
  :language: c++
  :start-at: // g = u + du/dn
  :end-at: make_bc<Robin>
  :dedent:
```

The discretization is exact for a linear function, so every ghost must hold $x$ at its own center:

```{literalinclude} snippet/bc/bc_custom_output.txt
  :language: text
```

```{important}
The implicit schemes assembled with PETSc write the boundary equations of `samurai::Dirichlet` and `samurai::Neumann` only.
They skip any other condition with the message `Unknown boundary condition type. Only Dirichlet and Neumann are implemented at the moment.`
A condition you write yourself works with explicit schemes and with the ghost updates, not in a matrix assembled by `samurai::petsc::make_solver`.
```

## Related

- The {doc}`boundary conditions reference <../reference/bc>`, for the complete list of conditions, the treatment of the corners and of the ghosts beyond the condition, and the lattice Boltzmann conditions.
- The {doc}`boundary condition API <../api/bc>`, for the classes and functions.
- {doc}`petsc`, to solve an implicit scheme, whose unknown field needs Dirichlet or Neumann conditions.
- {doc}`troubleshooting`, for the errors about stencil sizes and ghost layers.
