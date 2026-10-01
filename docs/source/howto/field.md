# How-to: create a samurai field

A samurai field stores data on the cells of a samurai mesh.
This guide shows how to create a field, give it initial values and check its content.
It works the same way on uniform, multiresolution and adaptive meshes.

Two kinds of fields are available:

- **scalar fields** store one value per cell;
- **vector fields** store a fixed number of values (components) per cell.

## Before you start

You need a samurai mesh.
If you do not have one yet, follow the [mesh how-to guide](mesh.md) first.

Include `<samurai/field.hpp>` to get the field functions.

## Create a scalar field

Call `samurai::make_scalar_field` with the value type, a name and the mesh.
The example below creates a field named `u` of type `double` on a 2D multiresolution mesh with levels 2 to 5:

```{literalinclude} snippet/field/scalar_field.cpp
  :language: c++
```

Replace `double` with the value type you need (`float`, `int`, `std::size_t`, ...).
If you omit the template argument, the value type is `double`.

## Create a vector field

Call `samurai::make_vector_field` with the value type, the number of components, a name and the mesh.
The example below creates a field named `v` with 3 components of type `double`:

```{literalinclude} snippet/field/vector_field.cpp
  :language: c++
```

`samurai::make_vector_field<3>("v", mesh)` is a shorter form for a vector field of `double`.

## Give the field initial values

```{important}
The two-argument forms `make_scalar_field<double>("u", mesh)` and `make_vector_field<double, 3>("v", mesh)` allocate the values but do not initialize them: the field holds arbitrary values until you write to it.
Pick one of the forms below, or fill the field yourself before you read it.
```

Both functions take extra arguments that set the initial values.
The forms are the same for scalar and vector fields.

### Set a constant

Pass the constant after the mesh.
Every value of the field, ghost cells included, gets this constant:

```c++
auto u = samurai::make_scalar_field<double>("u", mesh, 0.);
auto v = samurai::make_vector_field<double, 3>("v", mesh, 1.);
```

### Evaluate a function at the cell centers

Pass a function after the mesh.
It receives the coordinates of the cell center and returns the value of the cell.
The ghost cells are set to 0.
Use this form for a point value of an analytical initial condition:

```c++
auto u = samurai::make_scalar_field<double>("u",
                                            mesh,
                                            [](const auto& x)
                                            {
                                                return (x[0] < 0.5) ? 1. : 0.;
                                            });
```

### Average a function over each cell

Pass a `samurai::GaussLegendre<degree>` object after the function to store the cell average of the function instead of its value at the center.
The quadrature is exact for polynomials up to `degree`, and the ghost cells are set to 0.
Use this form when your scheme works on cell averages, as finite volume schemes do:

```c++
samurai::GaussLegendre<2> gl;

auto u = samurai::make_scalar_field<double>("u",
                                            mesh,
                                            [](const auto& x)
                                            {
                                                return x[0] * x[1];
                                            },
                                            gl);
```

### Fill an existing field

To set or reset the values of an existing field, call `fill`:

```c++
u.fill(0.);
```

To set values cell by cell, see the [loop how-to guide](loop.md).

## Check the result

Print the field to see the value of every cell:

```c++
#include <iostream>

std::cout << u << std::endl;
```

The output starts with `Field u`, followed by one line per cell with its level, the coordinates of its center, its index and its value.
On the mesh of the examples above, all cells are at level 5 after creation, which gives 1024 lines.

If you build with the `SAMURAI_CHECK_NAN` option, the two-argument forms fill floating-point fields with NaN instead of leaving them uninitialized, so that a value you forgot to set shows up as `nan`.
See the [CMake how-to guide](cmake.md) to turn this option on.

## Next steps

- [Loop over the cells of a mesh](loop.md) to compute and update field values.
- [Save a mesh and its fields](save.md) for post-processing in ParaView.
- [Define and use a field](../tutorial/field.rst), the tutorial on how a field stores and accesses its values.
