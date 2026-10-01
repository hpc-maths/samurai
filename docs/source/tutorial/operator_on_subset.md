# Apply an operator on a set

Adaptive mesh algorithms often apply an operator on one part of the mesh only: the ghost cells to update, the cells of level $l$ that receive the projection of level $l + 1$, and so on.
{{ project }} finds these parts with two ingredients: intervals and a set algebra.

In this tutorial, we build a 1D mesh with two levels, find where the levels overlap, and project the values of a field from the fine level to the coarse one.
On the way, we meet the set operators, the `on` method that chooses the level of the result, contraction, and operators that work in any dimension.

## Before you start

The code of this tutorial is the demo [`demos/tutorial/set_operator.cpp`](https://github.com/hpc-maths/samurai/blob/main/demos/tutorial/set_operator.cpp).
To run it, configure {{ project }} from source as described in the [installation guide](../howto/installation.md#build-from-source), then build and run the demo from the build directory:

```bash
cmake --build build --target tutorial-set-operator
./build/demos/tutorial/tutorial-set-operator
```

We assume that you know how a mesh is stored as intervals, see the {doc}`interval tutorial <interval>`.

## Intervals

A {cpp:class}`samurai::CellArray` stores a mesh made of several Cartesian grids, one per level of resolution.
Each grid is stored as arrays of intervals, one array per direction.

We build a mesh with two levels: we add intervals to a {cpp:class}`samurai::CellList`, then convert it into a {cpp:class}`samurai::CellArray`:

```{literalinclude} ../../../demos/tutorial/set_operator.cpp
  :language: c++
  :start-at: constexpr std::size_t dim = 1;
  :end-at: ca = {cl, true};
  :dedent: 4
```

Level 0 holds the cells 0 to 9.
Level 1 holds two intervals, the cells 2 to 5 and the cells 11 to 14.
An interval `{start, end}` contains `start` but not `end`.
The second argument of the conversion, `true`, computes where the values of each interval are stored in a field.

```{image} ./figures/subset_op.png
:alt: Two levels of a 1D mesh. Level 1 has the cells 2 to 5 and 11 to 14. Level 0 has the cells 0 to 10. Each cell of level 0 is twice as wide as a cell of level 1, so the cell 1 of level 0 covers the cells 2 and 3 of level 1.
:width: 80%
:align: center
```

The demo prints the mesh:

```{literalinclude} ../../../demos/tutorial/set_operator.cpp
  :language: c++
  :start-at: std::cout << ca
  :end-at: std::cout << ca
  :dedent: 4
```

The output is:

```text
┌────────────────────┐
│      Level 0       │
└────────────────────┘
     dim 0
            cells = 0->[0,10[@0:1


┌────────────────────┐
│      Level 1       │
└────────────────────┘
     dim 0
            cells = 0->[2,6[@8:1 1->[11,15[@3:1
```

Each interval prints as `[start,end[@index:step`.
`index` is the offset that gives the position of a cell in the field storage: the cell `i` of the interval `[2,6[@8:1` is stored at `8 + i`, so the cells 2 to 5 of level 1 come right after the 10 cells of level 0.
`step` is the stride between the cells of the interval.

## Set algebra

We now look for the cells where the two levels overlap.
{{ project }} has these set operators, all in the `samurai` namespace:

| Function | Result |
| --- | --- |
| `samurai::intersection(a, b, ...)` | the cells that belong to all the sets |
| `samurai::union_(a, b, ...)` | the cells that belong to at least one set |
| `samurai::difference(a, b, ...)` | the cells of `a` that belong to none of the other sets |
| `samurai::translate(a, t)` | the cells of `a` shifted by the vector `t` |
| `samurai::contract(a, width)` | the cells of `a` whose neighbors `width` cells away on both sides, in each direction, also belong to `a` |
| `samurai::expand(a, width)` | the cells of `a`, grown by `width` cells on both sides in each direction |

The arguments can be levels of a mesh, such as `ca[0]`, or the result of another operator.
The union is called `union_` because `union` is a C++ keyword.

### Intersection

We compute the intersection of the two levels:

```{literalinclude} ../../../demos/tutorial/set_operator.cpp
  :language: c++
  :start-at: auto subset = samurai::intersection(ca[0], ca[1]);
  :end-at: });
  :dedent: 4
```

`samurai::intersection` does not compute anything yet: it returns a subset, an expression that describes the set.
Calling the subset with a function computes the set and calls the function once per interval of the result.
The function takes two parameters: the interval along $x$, and an array of size `dim - 1` with the indices in the other directions.
In 1D, this array is empty, so we leave the second parameter unnamed.

The output is:

```text
intersection found in [2,6[@0:1
intersection found in [11,15[@0:1
```

The result is computed on level 1.
By default, a subset is computed on the finest level among its sets, here level 1.
Here the index is 0: the intervals of this subset describe cells, not positions in a field.

### Choosing the level of the result

The `on(level)` method of a subset gives the result on another level.
To get the intersection on level 0:

```{literalinclude} ../../../demos/tutorial/set_operator.cpp
  :language: c++
  :start-at: subset.on(0)(
  :end-at: });
  :dedent: 4
```

The output is:

```text
intersection found in [1,3[@0:1
intersection found in [5,8[@0:1
```

A cell of level 0 belongs to the result as soon as it overlaps the intersection.
The cells 5 and 7 of level 0 overlap the fine cells 11 and 14, so they are in the result, even though only one of their two fine cells exists.

The level does not have to exist in the mesh.
Here is the intersection on level 3:

```{literalinclude} ../../../demos/tutorial/set_operator.cpp
  :language: c++
  :start-at: subset.on(3)(
  :end-at: });
  :dedent: 4
```

The output is:

```text
intersection found in [8,24[@0:1
intersection found in [44,60[@0:1
```

### Difference

The other operators work the same way.
The difference between level 0 and level 1:

```{literalinclude} ../../../demos/tutorial/set_operator.cpp
  :language: c++
  :start-at: auto subset_d
  :end-at: });
  :dedent: 4
```

gives, on level 1:

```text
difference found in [0,2[@0:1
difference found in [6,11[@0:1
difference found in [15,20[@0:1
```

## Projection with a subset

We now create a field on the mesh and use a subset to apply an operator on part of it.

### Creating the field

We create a scalar field `u` with {cpp:func}`samurai::make_scalar_field`, set it to zero, then store in each cell of level 1 its index along $x$:

```{literalinclude} ../../../demos/tutorial/set_operator.cpp
  :language: c++
  :start-at: auto u = samurai::make_scalar_field
  :end-at: });
  :dedent: 4
```

`make_scalar_field<double>` creates a field with one `double` value per cell.
`samurai::for_each_cell` visits every cell of level 1, and `cell.indices[0]` is the index of the cell along $x$.

### Projecting on the intersection

The projection sets the value of a coarse cell of level 0 to the average of the values of its two fine cells on level 1.
The coarse cell `i` covers the fine cells `2 * i` and `2 * i + 1`.
Only the coarse cells whose two fine cells exist can be computed: in the figure below, the cells 1, 2 and 6, under the red arrows.

```{image} ./figures/subset_op_proj.png
:alt: The same two levels. Red arrows go from the fine cells 2 and 3 to the coarse cell 1, from the fine cells 4 and 5 to the coarse cell 2, and from the fine cells 12 and 13 to the coarse cell 6.
:width: 80%
:align: center
```

A first attempt uses the intersection on level 0 from the previous section.
This code is not in the demo:

```c++
auto subset = samurai::intersection(ca[0], ca[1]).on(0);
subset(
    [&](const auto& i, auto)
    {
        u(0, i) = 0.5 * (u(1, 2 * i) + u(1, 2 * i + 1));
    });
```

`u(level, i)` gives the values of `u` on the interval `i` of a level.
`2 * i` is the interval of the even fine cells, with a step of 2, and `2 * i + 1` the interval of the odd ones.

The first interval of the subset, `[1,3[`, works.
The second one, `[5,8[`, contains the coarse cells 5 and 7, whose fine cells 10 and 15 do not exist.
The program stops with an uncaught `std::out_of_range` exception.
C++ does not fix which of `u(1, 2 * i)` and `u(1, 2 * i + 1)` is evaluated first, so the message depends on the compiler.
It is one of these two:

```text
LevelCellArray::get_interval: interval not found at level 1, i = [10,16[@0:2, index =
Field 'u' interval query failed on level 1: requested interval [11,17[@0:2 could not be found for indices []; available interval: [11,15[@3:1
```

The first message says that the fine cell 10 is not in the mesh.
The second one says that the fine cells 11, 13 and 15 are not all in one interval of level 1: the interval found, `[11,15[`, stops before 15.

### Contraction

To keep only the coarse cells whose two fine cells exist, we contract level 1 before the intersection.
`samurai::contract(ca[1], 1)` removes one cell at each end of every interval of level 1, which gives `[3,5[` and `[12,14[`:

```{literalinclude} ../../../demos/tutorial/set_operator.cpp
  :language: c++
  :start-at: auto subset1
  :end-at: });
  :dedent: 4
```

On level 0, this subset is `[1,3[` and `[6,7[`: the cells 1, 2 and 6 of the figure.
After the projection, the cells 1, 2 and 6 of level 0 hold 2.5, 4.5 and 12.5, the other cells of level 0 hold 0, and level 1 does not change.
The demo prints the field with `std::cout << u`, which gives one line per cell with its level, its center, its index and its value.

A coarse cell that remains after the contraction always has its two fine cells.
The contraction can also remove a coarse cell whose two fine cells exist: a level 1 interval of exactly two cells, such as `[4,6[`, disappears when it is contracted by 1.

## Operators working in many dimensions

The projection above is written for 1D.
In 2D, a coarse cell has 4 fine cells, and in 3D it has 8.
We write one operator that works in 1D, 2D and 3D, with one `operator()` per dimension:

```c++
template <std::size_t dim, class TInterval>
class my_projection_op : public samurai::field_operator_base<dim, TInterval>
{
  public:

    INIT_OPERATOR(my_projection_op)

    template <class T>
    SAMURAI_INLINE void operator()(samurai::Dim<1>, T& field) const
    {
        field(level, i) = .5 * (field(level + 1, 2 * i) + field(level + 1, 2 * i + 1));
    }

    template <class T>
    SAMURAI_INLINE void operator()(samurai::Dim<2>, T& field) const
    {
        field(level, i, j) = .25
                           * (field(level + 1, 2 * i, 2 * j) + field(level + 1, 2 * i, 2 * j + 1) + field(level + 1, 2 * i + 1, 2 * j)
                              + field(level + 1, 2 * i + 1, 2 * j + 1));
    }

    template <class T>
    SAMURAI_INLINE void operator()(samurai::Dim<3>, T& field) const
    {
        field(level, i, j, k) = .125
                              * (field(level + 1, 2 * i, 2 * j, 2 * k) + field(level + 1, 2 * i + 1, 2 * j, 2 * k)
                                 + field(level + 1, 2 * i, 2 * j + 1, 2 * k) + field(level + 1, 2 * i + 1, 2 * j + 1, 2 * k)
                                 + field(level + 1, 2 * i, 2 * j, 2 * k + 1) + field(level + 1, 2 * i + 1, 2 * j, 2 * k + 1)
                                 + field(level + 1, 2 * i, 2 * j + 1, 2 * k + 1) + field(level + 1, 2 * i + 1, 2 * j + 1, 2 * k + 1));
    }
};

template <class T>
SAMURAI_INLINE auto my_projection(T&& field)
{
    return samurai::make_field_operator_function<my_projection_op>(std::forward<T>(field));
}
```

The parts of this code are:

- `samurai::field_operator_base<dim, TInterval>` holds the position where the operator is applied: the level `level`, the interval `i` along $x$, and the indices `j` and `k` in the other directions.
- `INIT_OPERATOR(my_projection_op)` declares the constructors and makes `level`, `i`, `j` and `k` usable in the class. It needs the template parameters to be named `dim` and `TInterval`.
- The first parameter of each `operator()`, `samurai::Dim<1>`, `samurai::Dim<2>` or `samurai::Dim<3>`, selects the version that matches the dimension of the mesh at compile time.
- `my_projection(u)` binds the operator to the field `u` with `samurai::make_field_operator_function`.

We apply the operator on the contracted subset with `apply_op`:

```{literalinclude} ../../../demos/tutorial/set_operator.cpp
  :language: c++
  :start-at: subset1.on(0).apply_op
  :end-at: subset1.on(0).apply_op
  :dedent: 4
```

`apply_op` calls the operator once per interval of the subset, on the level of the subset, here level 0.
It gives the same values as the lambda of the previous section.
`apply_op` takes several operators, and applies all of them on each interval.

## What we built

We built a mesh with two levels, found where they overlap with `samurai::intersection`, chose the level of the result with `on`, and used `samurai::contract` to keep only the coarse cells that can receive a projection.
Then we wrote the projection as an operator that works in 1D, 2D and 3D, and applied it with `apply_op`.

## Next steps

- {ref}`AlgebraOfSet` explains how {{ project }} computes a subset from sets of intervals.
- The {doc}`graduation tutorials <graduation>` use subsets to find the cells that break the graduation of a mesh.
