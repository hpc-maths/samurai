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
The examples of this section build a mesh with levels 0 and 1, which has 4 cells after creation, and print the field so that you can see its values.

### Set a constant

Pass the constant after the mesh.
Every value of the field, ghost cells included, gets this constant:

```{literalinclude} snippet/field/init_constant.cpp
  :language: c++
```

The program prints:

```text
Field u
    level: 1 coords: { 0.25,  0.25} index: 23, value: 0
    level: 1 coords: { 0.75,  0.25} index: 24, value: 0
    level: 1 coords: { 0.25,  0.75} index: 29, value: 0
    level: 1 coords: { 0.75,  0.75} index: 30, value: 0

Field v
    level: 1 coords: { 0.25,  0.25} index: 23, value: { 1.,  1.,  1.}
    level: 1 coords: { 0.75,  0.25} index: 24, value: { 1.,  1.,  1.}
    level: 1 coords: { 0.25,  0.75} index: 29, value: { 1.,  1.,  1.}
    level: 1 coords: { 0.75,  0.75} index: 30, value: { 1.,  1.,  1.}
```

Each line is a cell: its level, the coordinates of its center, its index in the field storage and its value.
For a vector field, the value lists the components.
The program indents each line with a tab, shown here as four spaces.

### Evaluate a function at the cell centers

Pass a function after the mesh.
It receives the coordinates of the cell center and returns the value of the cell.
The ghost cells are set to 0.
Use this form for a point value of an analytical initial condition:

```{literalinclude} snippet/field/init_function.cpp
  :language: c++
  :start-at: // 1 on the left half
  :end-at: std::cout << u
  :dedent:
```

The cells of the left half get 1, the others 0:

```text
Field u
    level: 1 coords: { 0.25,  0.25} index: 23, value: 1
    level: 1 coords: { 0.75,  0.25} index: 24, value: 0
    level: 1 coords: { 0.25,  0.75} index: 29, value: 1
    level: 1 coords: { 0.75,  0.75} index: 30, value: 0
```

The complete program is in `docs/source/howto/snippet/field/init_function.cpp`; the mesh is the same as in the previous example.

### Average a function over each cell

Pass a `samurai::GaussLegendre<degree>` object after the function to store the cell average of the function instead of its value at the center.
The quadrature is exact for polynomials up to `degree`, and the ghost cells are set to 0.
Use this form when your scheme works on cell averages, as finite volume schemes do:

```{literalinclude} snippet/field/init_gauss_legendre.cpp
  :language: c++
  :start-at: auto f =
  :end-at: std::cout << u
  :dedent:
```

For $f(x, y) = x^2$, the average over a cell of center $x_c$ and side $h$ is $x_c^2 + h^2 / 12$.
With $h = 0.5$, the cells centered at $x_c = 0.25$ get $0.0625 + 0.0208 = 0.0833$ instead of the center value $0.0625$:

```text
Field u
    level: 1 coords: { 0.25,  0.25} index: 23, value: 0.0833333
    level: 1 coords: { 0.75,  0.25} index: 24, value: 0.583333
    level: 1 coords: { 0.25,  0.75} index: 29, value: 0.0833333
    level: 1 coords: { 0.75,  0.75} index: 30, value: 0.583333
```

The complete program is in `docs/source/howto/snippet/field/init_gauss_legendre.cpp`.

### Fill an existing field

To set or reset the values of an existing field, call `fill`:

```{literalinclude} snippet/field/fill_field.cpp
  :language: c++
  :start-at: u.fill
  :end-at: u.fill
  :dedent:
```

Every value of the field, ghost cells included, gets the value you pass.
To set values cell by cell, see the [loop how-to guide](loop.md).

## Check the result

Print the field with `std::cout << u`, as the examples above do, to see the value of every cell.
The output starts with `Field u`, followed by one line per cell.
It lists the cells of the mesh, without the ghost cells.
On a multiresolution mesh with levels 2 to 5, as in the first examples of this guide, all cells are at level 5 after creation, which gives 1024 lines.

If you build with the `SAMURAI_CHECK_NAN` option, the two-argument forms fill floating-point fields with NaN instead of leaving them uninitialized, so that a value you forgot to set shows up as `nan`.
With this option, printing a field also lists the ghost cells.
See the [CMake how-to guide](cmake.md) to turn this option on.

## Next steps

- [Loop over the cells of a mesh](loop.md) to compute and update field values.
- [Save a mesh and its fields](save.md) for post-processing in ParaView.
- {doc}`Field storage and access <../tutorial/field>`, the tutorial on how a field stores and accesses its values.
