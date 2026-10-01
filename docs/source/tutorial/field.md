# Field storage and access

In this tutorial, we see how {{ project }} stores the values of a field on the mesh of the {doc}`interval tutorial <interval>`.
Then we create a field and write its values, first one cell at a time and then one interval of cells at a time.

## Before you start

This tutorial builds on the data structure described in the {doc}`interval tutorial <interval>`: intervals, levels, the `@` index and the y-offsets.
The code samples use a samurai mesh; the [mesh how-to guide](../howto/mesh.md) shows how to create one.
They need the `<samurai/field.hpp>` header.

(field-storage)=

## Field storage

A field is a value (scalar field) or a fixed number of values (vector field) attached to each cell of a mesh.
We want to reach these values with a syntax like `field(level, i, j, k)`, where `i` is an interval of cells along $x$ at `level` and `j`, `k` are the $y$ and $z$ indices, as on a uniform Cartesian grid.
To this end, we need a link between the array where the field values are stored and the mesh represented by a `samurai::CellArray`.

### A 1D example

We use the 1D example of the interval tutorial:

```{image} ./figures/interval_example_1D.png
:alt: A 1D mesh with two cells at level 0 on the left, three cells at level 1, two cells at level 2 in the middle, two cells at level 1 and one cell at level 0 on the right, above the cell indices of levels 0, 1 and 2.
:align: center
:width: 80%
```

The values of a field are stored in one contiguous array, with one entry per cell.
Its size is the sum of the sizes of all the intervals along $x$.
The cells are numbered from the coarsest level to the finest level and, at each level, from left to right:

```{image} ./figures/interval_example_1D_field.png
:alt: The same 1D mesh with the position of each cell in the field array: 0 and 1 for the cells at level 0 on the left, 3, 4, 5 for the cells at level 1, 8 and 9 for the cells at level 2, 6 and 7 for the cells at level 1 on the right, and 2 for the last cell at level 0.
:align: center
:width: 80%
```

For example:

- `field(0, 0)` is entry 0 and belongs to the interval $[0, 2[$ at level 0;
- `field(2, 14)` is entry 8 and belongs to the interval $[14, 16[$ at level 2.

The cells of an interval are contiguous in the array, so one integer per interval is enough to find them: the index given by the `@` operator.
The interval $[14, 16[$ at level 2 holds entries 8 and 9 of the array.
With the index $-6$, the entry of each cell is its $x$ coordinate plus the index:

$$
[14, 16[@-6 \rightarrow [14 - 6, 15 - 6] = [8, 9].
$$ (field-1d-index)

Following the same rule, the index of each interval is:

- level 0: $[0, 2[@0$, $[5, 6[@-3$;
- level 1: $[4, 7[@-1$, $[8, 10[@-2$;
- level 2: $[14, 16[@-6$.

### A 2D example

We use the 2D mesh of the interval tutorial:

```{image} ./figures/2D_mesh.png
:alt: A 2D mesh on a 4 by 4 grid at level 0, with a 4 by 4 block of level 1 cells in the middle, a block of level 1 cells in the top right corner, and two 2 by 2 blocks of level 2 cells, one in the middle and one in the top right corner. The x and y indices of each level are drawn along the axes.
:align: center
:width: 60%
```

The cells are numbered in the same way: from the coarsest level to the finest level, and at each level row by row, from left to right along $x$:

```{image} ./figures/2D_mesh_numbering.png
:alt: The same 2D mesh with the position of each cell in the field array, from 0 to 10 for the level 0 cells, 11 to 28 for the level 1 cells and 29 to 36 for the level 2 cells.
:align: center
:width: 60%
```

For example:

- `field(0, 0, 0)` is entry 0 and belongs to the interval $[0, 4[$ for $y = 0$ at level 0;
- `field(2, 14, 15)` is entry 35 and belongs to the interval $[14, 16[$ for $y = 15$ at level 2.

As in 1D, the index of an interval along $x$ gives the entries of its cells.
The interval $[14, 16[$ for $y = 15$ at level 2 holds entries 35 and 36, so its index is 21:

$$
[14, 16[@21 \rightarrow [14 + 21, 15 + 21] = [35, 36].
$$ (field-2d-index)

With these indices, the whole `samurai::CellArray` is:

```text
level 0:
    x: [0, 4[@0, [0, 1[@4, [3, 4[@2, [0, 1[@6, [3, 4[@4, [0, 3[@8
    y: [0, 4[@0
    y-offset: [0, 1, 3, 5, 6]

level 1:
    x: [2, 6[@9, [2, 6[@13, [2, 4[@17, [5, 6[@16, [2, 6[@20, [6, 8[@20, [6, 7[@22
    y: [2, 8[@-2
    y-offset: [0, 1, 2, 4, 5, 6, 7]

level 2:
    x: [8, 10[@21, [8, 10[@23, [14, 16[@19, [14, 16[@21
    y: [8, 10[@-8, [14, 16[@-12
    y-offset: [0, 1, 2, 3, 4]
```

The `y` intervals and the y-offsets are the ones built in the interval tutorial; only the indices of the `x` intervals are new.

A samurai mesh (uniform, multiresolution or adaptive) holds several cell arrays: the cells of the solution, their ghost cells, and so on.
It numbers the cells of its reference cell array, which contains all of them, and gives the same index to an interval in every other cell array.
A field on such a mesh therefore has one entry per cell of the reference cell array, ghost cells included.

## Field creation

A field is created on a mesh: a `samurai::CellArray`, a `samurai::LevelCellArray`, or a samurai mesh such as the one returned by `samurai::mra::make_mesh`.
We create a scalar field of type `double` named `u` with `samurai::make_scalar_field`, and a vector field of type `double` with 2 components named `v` with `samurai::make_vector_field`:

```c++
auto u = samurai::make_scalar_field<double>("u", mesh);
auto v = samurai::make_vector_field<double, 2>("v", mesh);
```

The name of the field is the name of its dataset when we save it in HDF5 format.
These two calls allocate the values but do not initialize them.
The [field how-to guide](../howto/field.md) shows the other forms of these functions, which set a constant or evaluate a function on each cell.

## Field access

{{ project }} gives two ways to reach the values of a field: one cell at a time, or one interval of cells at a time.

### Access by cell

The first way uses a {cpp:class}`samurai::Cell` and the `[]` operator.
A cell holds its level, its integer coordinates (`indices`) and its position in the field array (`index`), as well as its length and the origin point of the mesh.
For more on cells, see the {ref}`cell properties <cell>` in the interval tutorial.

We rarely build a cell ourselves.
For a single cell, the mesh gives it from its level and integer coordinates:

```c++
auto cell = mesh.get_cell(level, i, j); // i, j: integer coordinates of the cell

u[cell]    = 1.;
v[cell][1] = 2.;
```

On the scalar field, `u[cell]` is the value of the cell.
On the vector field, `v[cell]` holds the 2 components of the cell, and the second `[]` selects one of them.
The {cpp:class}`samurai::Cell` constructor also exists: `samurai::Cell<dim, TInterval>{origin_point, scaling_factor, level, indices, index}`.

Most of the time, the cells come from a loop over the cells of the mesh, as we see in the {doc}`next tutorial <algorithm>`.

### Access by interval

The second way reaches the values as on a uniform Cartesian grid: `u(level, i, j)` in 2D, `u(level, i, j, k)` in 3D.
We need the level to know where the indices live: `i = 1` at level 0 and `i = 1` at level 10 are different cells.
The other difference with a uniform grid is that `i` is not a single index but an interval along $x$, while `j` and `k` are integers.
`i` must lie inside an interval of the mesh at this level.
`u(level, i, j)` gives the values of all the cells of `i` at once.

The following program, taken from the [loop how-to guide](../howto/loop.md), creates a 2D multiresolution mesh on $[-1, 1]^2$ and sets a scalar field to $u(x, y) = e^{-20\left((x - 0.5)^2 + (y - 0.5)^2\right)}$, one interval at a time:

```{literalinclude} ../howto/snippet/loop/for_each_interval_field.cpp
:language: c++
:lines: 10-26
```

`samurai::for_each_interval` calls the function once per interval of the mesh, with the level, the interval `i` and the $y$ index `j = index[0]`.
`x` holds the $x$ coordinates of the cell centers of the interval and `y` the $y$ coordinate of the row, a single number.

The field values are stored in an xtensor container by default, so `u(level, i, j)` is an [xtensor view](https://xtensor.readthedocs.io/en/latest/view.html) on the field array.
We can assign it an xtensor expression, which is evaluated when it is assigned, as the example does with `xt::arange` and `xt::exp`.
The xtensor API is close to NumPy, see [From NumPy to xtensor](https://xtensor.readthedocs.io/en/latest/numpy.html).
When samurai is built with the Eigen field container (CMake option `SAMURAI_FIELD_CONTAINER=eigen3` of the samurai source tree, which defines `SAMURAI_FIELD_CONTAINER_EIGEN3`), `u(level, i, j)` is an Eigen indexed view instead, and the xtensor functions do not apply to it.

On a vector field, `v(level, i, j)` holds all the components of the cells of `i`: one row per cell and one column per component.
To write one component, put its index first.
The following line, which assumes `x` and `y` are defined as in the loop above, sets the second component of `v`:

```c++
v(1, level, i, j) = xt::cos(x) * xt::sin(y);
```

## What we covered

We have seen how {{ project }} numbers the cells of a mesh to store a field in one array, and how the `@` index of each interval gives the position of its cells.
We have created a scalar and a vector field and written their values with `u[cell]` and `u(level, i, j)`.

## Next steps

- The {doc}`loop tutorial <algorithm>` shows how to visit the cells and the intervals of a mesh.
- The [field how-to guide](../howto/field.md) lists the ways to create and initialize a field.
- The [loop how-to guide](../howto/loop.md) shows how to set field values on the cells you choose.
