# Apply a subset operator on a set expression

Adaptive mesh algorithms often compute on one part of the mesh only: the ghost cells to update, the cells of level $l$ that receive the field projection of level $l + 1$, and so on.
{{ project }} describes these parts with set expressions, built from intervals by a set algebra.

In this tutorial, we build a 1D mesh with two levels, find where the levels overlap, and compute the field projection of a field from the fine level to the coarse one.
On the way, we meet the set operations, the `on` method that chooses the level of the result, contraction, and subset operators that work in any dimension.

## Before you start

The code of this tutorial is the demo [`demos/tutorial/set_operator.cpp`](https://github.com/hpc-maths/samurai/blob/main/demos/tutorial/set_operator.cpp).
To run it, configure {{ project }} from source as described in the [installation guide](../howto/installation.md#build-from-source), then build and run the demo from the build directory:

```bash
cmake --build build --target tutorial-set-operator
./build/demos/tutorial/tutorial-set-operator
```

We assume that you know how a mesh is stored as intervals, see the {doc}`interval tutorial <interval>`.

## Intervals

A {cpp:class}`samurai::CellArray` stores the cells of several levels, one level cell array per level.
Each level cell array is stored as arrays of intervals, one array per direction.

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

```{diagram}
:figure: two_level_mesh

Level 1 holds `[2, 6)` and `[11, 15)`, level 0 holds `[0, 10)`, both at their true size: a cell of level 0 covers two cells of level 1.
```

The demo prints the mesh:

```{literalinclude} ../../../demos/tutorial/set_operator.cpp
  :language: c++
  :start-at: std::cout << ca
  :end-at: std::cout << ca
  :dedent: 4
```

The output is:

```{literalinclude} ../../../demos/tutorial/set_operator_output.txt
  :language: text
  :end-before: intersection found
```

Each interval prints as `[start,end)@index:step`.
`index` is the offset that gives the position of a cell in the field storage: the cell `i` of the interval `[2,6)@8:1` is stored at `8 + i`, so the cells 2 to 5 of level 1 come right after the 10 cells of level 0.
`step` is the stride between the cells of the interval.

## Set algebra

We now look for the cells where the two levels overlap.
{{ project }} has these set operations, all in the `samurai` namespace:

| Function | Result |
| --- | --- |
| `samurai::intersection(a, b, ...)` | the cells that belong to all the sets |
| `samurai::union_(a, b, ...)` | the cells that belong to at least one set |
| `samurai::difference(a, b, ...)` | the cells of `a` that belong to none of the other sets |
| `samurai::translate(a, t)` | the cells of `a` shifted by the vector `t` |
| `samurai::contract(a, width)` | the cells of `a` whose neighbors `width` cells away on both sides, in each direction, also belong to `a` |
| `samurai::expand(a, width)` | the cells of `a`, grown by `width` cells on both sides in each direction |

The arguments can be level cell arrays, such as `ca[0]`, or the result of another set operation.
The union is called `union_` because `union` is a C++ keyword.

### Intersection

We compute the intersection of the two levels:

```{literalinclude} ../../../demos/tutorial/set_operator.cpp
  :language: c++
  :start-at: auto subset = samurai::intersection(ca[0], ca[1]);
  :end-at: });
  :dedent: 4
```

`samurai::intersection` does not compute anything yet: it returns a set expression, which describes the set without holding its cells.
Calling the set expression with a function traverses it: the function is called once per interval of the result.
The function takes two parameters: the interval along $x$, and an array of size `dim - 1` with the indices in the other directions.
In 1D, this array is empty, so we leave the second parameter unnamed.

The output is:

```{literalinclude} ../../../demos/tutorial/set_operator_output.txt
  :language: text
  :start-at: intersection found in [2,6)
  :end-at: intersection found in [11,15)
```

The result is computed on level 1.
By default, a set expression is computed on the finest level among its sets, here level 1.
Here the index is 0: the intervals of this set expression describe cells, not positions in a field.

### Choosing the level of the result

The `on(level)` method of a set expression gives the result on another level: this is a {ref}`set projection <def-set-projection>`.
To get the intersection on level 0:

```{literalinclude} ../../../demos/tutorial/set_operator.cpp
  :language: c++
  :start-at: subset.on(0)(
  :end-at: });
  :dedent: 4
```

The output is:

```{literalinclude} ../../../demos/tutorial/set_operator_output.txt
  :language: text
  :start-at: intersection found in [1,3)
  :end-at: intersection found in [5,8)
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

```{literalinclude} ../../../demos/tutorial/set_operator_output.txt
  :language: text
  :start-at: intersection found in [8,24)
  :end-at: intersection found in [44,60)
```

### Difference

The other set operations work the same way.
The difference between level 0 and level 1:

```{literalinclude} ../../../demos/tutorial/set_operator.cpp
  :language: c++
  :start-at: auto subset_d
  :end-at: });
  :dedent: 4
```

gives, on level 1:

```{literalinclude} ../../../demos/tutorial/set_operator_output.txt
  :language: text
  :start-at: difference found in [0,2)
  :end-at: difference found in [15,20)
```

## Field projection on a set expression

We now create a field on the mesh and use a set expression to compute a field projection on part of it.

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

### Field projection on the intersection

The field projection sets the value of a coarse cell of level 0 to the average of the values of its two fine cells on level 1.
The coarse cell `i` covers the fine cells `2 * i` and `2 * i + 1`.
Only the coarse cells whose two fine cells exist can be computed: the cells 1, 2 and 6.
{ref}`plate-subset-projection` follows the computation from the intersection to the values of the field.

```{plate} From the intersection to the field projection
:figure: subset_projection
:label: plate-subset-projection

**Fig. 1.** The attempt.
`intersection(ca[0], ca[1])` is computed on level 1, where it is `[2, 6)` and `[11, 15)`.
Its set projection `.on(0)` is `[1, 3)` and `[5, 8)` on level 0.
*a*, the cells 5 and 7, hatched in [red]{.sm-red}, overlap the intersection, so the set projection keeps them, but only one of their two fine cells exists.
*b*, their other fine cells, 10 and 15, drawn dashed, are not in level 1.
Reading `u(1, 2 * i)` and `u(1, 2 * i + 1)` on `[5, 8)` asks for them, and the program stops with `std::out_of_range`.

**Fig. 2.** After contraction.
`contract(ca[1], 1)` is `[3, 5)` and `[12, 14)`, inside the heavy outline of level 1.
Its intersection with level 0, brought to level 0 by `.on(0)`, is `[1, 3)` and `[6, 7)`.
*c*, the contraction removes one cell at each end of each interval, such as the cell 14.

**Fig. 3.** The field `u` after the field projection.
Each [red]{.sm-red} cell of level 0 holds the mean of its two fine cells: 2.5, 4.5 and 12.5.
The other cells of level 0 keep 0, and level 1 does not change.
```

A first attempt uses the intersection on level 0 from the previous section ({ref}`plate-subset-projection`, Fig. 1).
This code is not in the demo, but in the program [`field_projection_attempt.cpp`](https://github.com/hpc-maths/samurai/blob/main/docs/source/tutorial/snippet/set_operator/field_projection_attempt.cpp), which builds the same mesh and field:

```{literalinclude} snippet/set_operator/field_projection_attempt.cpp
  :language: c++
  :start-at: auto set = samurai::intersection
  :end-at: });
  :dedent: 8
```

`u(level, i)` gives the values of `u` on the interval `i` of a level.
`2 * i` is the interval of the even fine cells, with a step of 2, and `2 * i + 1` the interval of the odd ones.

The first interval of the set expression, `[1,3)`, works.
The second one, `[5,8)`, contains the coarse cells 5 and 7, whose fine cells 10 and 15 do not exist.
Reading them throws a `std::out_of_range` exception, which stops a program that does not catch it.
The program catches it and prints the interval of level 0 it was computing:

```{literalinclude} snippet/set_operator/field_projection_attempt_output.txt
  :language: text
```

The message of the exception names the interval of level 1 that the read could not find.
C++ does not fix which of `u(1, 2 * i)` and `u(1, 2 * i + 1)` is evaluated first, so the failing read, and the message, depend on the compiler and the target.

### Contraction

To keep only the coarse cells whose two fine cells exist, we contract level 1 before the intersection.
`samurai::contract(ca[1], 1)` removes one cell at each end of every interval of level 1, which gives `[3,5)` and `[12,14)`:

```{literalinclude} ../../../demos/tutorial/set_operator.cpp
  :language: c++
  :start-at: auto subset1
  :end-at: });
  :dedent: 4
```

On level 0, this set expression is `[1,3)` and `[6,7)`: the cells 1, 2 and 6 ({ref}`plate-subset-projection`, Fig. 2).
After the field projection, the cells 1, 2 and 6 of level 0 hold 2.5, 4.5 and 12.5, the other cells of level 0 hold 0, and level 1 does not change (Fig. 3).
The demo prints the field with `std::cout << u`, which gives one line per cell with its level, its center, its index and its value:

```{literalinclude} ../../../demos/tutorial/set_operator_output.txt
  :language: text
  :start-at: Field u
  :end-at: index: 17, value: 14
```

A coarse cell that remains after the contraction always has its two fine cells.
The contraction can also remove a coarse cell whose two fine cells exist: a level 1 interval of exactly two cells, such as `[4,6)`, disappears when it is contracted by 1.

## A subset operator for every dimension

The field projection above is written for 1D.
In 2D, a coarse cell has 4 fine cells, and in 3D it has 8.
A subset operator is a kernel that `apply_op` applies on each interval of a set expression.
We write one that works in 1D, 2D and 3D, with one `operator()` per dimension:

```{literalinclude} ../../../demos/tutorial/set_operator.cpp
  :language: c++
  :start-at: template <std::size_t dim, class TInterval>
  :end-before: // Checks my_projection in 2D and 3D
```

The parts of this code are:

- `samurai::field_operator_base<dim, TInterval>` holds the position where the subset operator is applied: the level `level`, the interval `i` along $x$, and the indices `j` and `k` in the other directions.
- `INIT_OPERATOR(my_projection_op)` declares the constructors and makes `level`, `i`, `j` and `k` usable in the class. It needs the template parameters to be named `dim` and `TInterval`.
- The first parameter of each `operator()`, `samurai::Dim<1>`, `samurai::Dim<2>` or `samurai::Dim<3>`, selects the version that matches the dimension of the mesh at compile time.
- `my_projection(u)` binds the subset operator to the field `u` with `samurai::make_field_operator_function`.

We apply the subset operator on the contracted set expression with `apply_op`:

```{literalinclude} ../../../demos/tutorial/set_operator.cpp
  :language: c++
  :start-at: subset1.on(0).apply_op
  :end-at: subset1.on(0).apply_op
  :dedent: 4
```

`apply_op` calls the subset operator once per interval of the set expression, on the level of the set expression, here level 0.
It gives the same values as the lambda of the previous section.
`apply_op` takes several subset operators, and applies all of them on each interval.

To check the 2D and 3D versions, the demo builds a mesh with one cell on level 0 and its $2^d$ fine cells on level 1, in dimension $d = 2$ and $d = 3$.
The fine cells hold 1 to $2^d$, so the field projection must give their mean, $(2^d + 1) / 2$:

```{literalinclude} ../../../demos/tutorial/set_operator.cpp
  :language: c++
  :start-at: samurai::intersection(ca[0], ca[1]).on(0).apply_op
  :end-at: samurai::intersection(ca[0], ca[1]).on(0).apply_op
  :dedent: 4
```

The output is:

```{literalinclude} ../../../demos/tutorial/set_operator_output.txt
  :language: text
  :start-at: 2D:
```

## What we built

We built a mesh with two levels, found where they overlap with `samurai::intersection`, chose the level of the result with `on`, and used `samurai::contract` to keep only the coarse cells that can receive a field projection.
Then we wrote the field projection as a subset operator that works in 1D, 2D and 3D, and applied it with `apply_op`.

## Next steps

- {ref}`AlgebraOfSet` lists the set operations and explains how {{ project }} traverses a set expression.
- The {doc}`graduation tutorials <graduation>` use set expressions to find the cells that break the graduation of a mesh.
