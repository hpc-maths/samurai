# Fields

This page documents the fields of {{ project }}: `samurai::ScalarField`, which stores one value per cell, `samurai::VectorField`, which stores a fixed number of components per cell, and the `samurai::make_scalar_field` and `samurai::make_vector_field` functions that create them.
A field is sized on the `reference` mesh id of its mesh, so it holds a value for every real cell and every ghost.
Include `samurai/field.hpp` to get all of them.
For how to create and initialize a field, see the {doc}`field how-to <../howto/field>`.

## Create a field

Each function takes a name and a mesh, followed by an optional initializer:

- nothing: the values are allocated but not initialized;
- a value: every value of the field, ghosts included, is set to it;
- a function of the cell center coordinates: each real cell gets the value of the function at its center, and the ghosts are set to 0;
- a function and a `samurai::GaussLegendre<degree>` object: each real cell gets the average of the function over the cell, and the ghosts are set to 0.

The value type defaults to `double`.
`make_vector_field` takes the number of components as a template argument, after the value type when you give one: `make_vector_field<double, 3>("v", mesh)` and `make_vector_field<3>("v", mesh)` build the same field.

### Scalar fields

```{doxygenfile} scalar_field.hpp
:sections: innernamespace func
```

### Vector fields

```{doxygenfile} vector_field.hpp
:sections: innernamespace func
```

## Field classes

`samurai::ScalarField<mesh_t, value_t>` and `samurai::VectorField<mesh_t, value_t, n_comp>` share their interface.
Most of it comes from base classes in the `samurai::detail` namespace, which this reference does not render:

| Member | Effect |
| --- | --- |
| `name()` | The name given at construction; `samurai::save` writes the field under this name. |
| `mesh()` | The mesh the field lives on. |
| `field[cell]` | The value of a cell; for a vector field, a view on its `n_comp` components. |
| `field(level, interval, index...)` | The values of an interval of cells at a level, for the remaining coordinates `index...`. |
| `field(item, level, interval, index...)` | Vector field only: the component `item` on an interval of cells. |
| `fill(value)` | Sets every value, ghosts included. |
| `resize()` | Resizes the storage to the number of cells of the mesh, after the mesh changed. |
| `array()` | The container that stores the values. |
| `ghosts_updated()` | Whether the ghosts hold up-to-date values; `fill` and `resize` set it to `false`. |
| `get_bc()` | The boundary conditions attached by `samurai::make_bc`. |

`samurai::swap(u1, u2)` exchanges the values of two fields, and their `ghosts_updated()` flags, without copying the values.
For the boundary conditions, see the {doc}`boundary condition API <bc>`.

```{doxygenclass} samurai::ScalarField
:members: ScalarField, n_comp, is_scalar
:undoc-members:
```

```{doxygenclass} samurai::VectorField
:members: VectorField, n_comp, is_scalar
:undoc-members:
```
