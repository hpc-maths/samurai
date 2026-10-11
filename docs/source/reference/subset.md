<!-- markdownlint-disable-next-line MD041 -->
(AlgebraOfSet)=

# Set algebra

{{ project }} describes the cells of a mesh as sets of integer intervals, one level cell array per level.
The set algebra combines these sets into set expressions: unions, intersections, differences, translations, expansions and contractions, at any level.
A set expression holds no cells of its own.
It is evaluated lazily: its cells are produced only when it is traversed.

This page lists the operations of the static set algebra, whose expressions are C++ types built at compile time, and of the dynamic set algebra, whose expressions are built at runtime.
For a guided introduction, see the {doc}`tutorial on subset operators <../tutorial/operator_on_subset>`.

## Header and operands

The static set algebra is in the `samurai` namespace, in the header `samurai/subset/node.hpp`.

An operand of a set operation is one of:

| Operand | How to get it |
| --- | --- |
| a level cell array | `ca[level]` for a {cpp:class}`samurai::CellArray` `ca`, `mesh[mesh_id][level]` for a mesh, or any {cpp:class}`samurai::LevelCellArray` |
| a box of cells | `samurai::asBoxView(level, box)`, with `box` a {cpp:class}`samurai::Box` of integer coordinates at `level` |
| a set expression | the result of another set operation |

`samurai::self(set)` turns any operand into a set expression: it wraps a level cell array in a view, and returns a set expression unchanged.
The set operations call it on their arguments, so `self` is needed only to call a member function, such as `on` or `level`, directly on a level cell array: `self(mesh.domain()).on(level)`.

A set expression stores references to the level cell arrays and boxes it is built from, not copies.
They must outlive the expression.

## Set operations

| Function | Result |
| --- | --- |
| `union_(a, ...)` | the cells that belong to at least one of the sets; takes one set or more |
| `intersection(a, b, ...)` | the cells that belong to all the sets; takes two sets or more |
| `difference(a, b, ...)` | the cells of `a` that belong to none of the other sets; takes two sets or more |
| `translate(a, t)` | the cells of `a` shifted by `t` |
| `expand(a, width)` | the cells within `width` cells of a cell of `a`, diagonals included |
| `expand(a, widths)` | the same, with one width per direction |
| `expand(a, width, directions)` | the same, only in the directions where `directions` is `true` |
| `contract(a, width)` | the cells of `a` whose neighbors `width` cells away on both sides, along each axis, also belong to `a` |
| `contract(a, width, directions)` | the same, only along the directions where `directions` is `true` |

The union is named `union_` because `union` is a C++ keyword.

The following program combines two 1D sets of level 0, $A = [0, 5) \cup [10, 13)$ and $B = [4, 8)$:

```{literalinclude} snippet/set_algebra/set_algebra_boolean.cpp
  :language: c++
  :start-at: print("union_
  :end-at: print("difference(b, a)
  :dedent: 4
```

It prints:

```{literalinclude} snippet/set_algebra/set_algebra_boolean_output.txt
  :language: text
```

```{diagram}
:figure: boolean_operations

The four results on the cells 0 to 13 of level 0, under the two operands.
```

The arguments have these types, with `value_t` the integer type of the interval bounds (`int` by default) and `dim` the dimension:

- `t`: `xt::xtensor_fixed<value_t, xt::xshape<dim>>`, the shift in cells along $x$, $y$, $z$;
- `width` of `expand`: `value_t`; `widths`: `std::array<value_t, dim>`;
- `width` of `contract`: `std::size_t`;
- `directions`: `std::array<bool, dim>`, indexed by direction ($x$ first).

Widths and shifts are counted in cells of the level of the operand.
To count them at another level, bring the operand to that level first: `expand(a.on(level), width)`.

`expand` and `contract` are not inverse operations.
`expand` grows a set by a square of radius `width`, diagonals included.
`contract` checks only the `2 * dim` neighbors along the axes: a cell is kept even when a diagonal neighbor is missing.

The following program applies these operations to a 2D set of five cells shaped like a plus sign:

```{literalinclude} snippet/set_algebra/set_algebra_expand_contract.cpp
  :language: c++
  :start-at: // A plus sign
  :end-at: print("translate
  :dedent: 4
```

It prints the intervals of each result, row by row:

```{literalinclude} snippet/set_algebra/set_algebra_expand_contract_output.txt
  :language: text
```

```{plate} Expansion, contraction and translation of a plus sign
:figure: plus_sign_operations
:label: plate-plus

**Fig. 1.** The operand, a plus sign of five cells of level 0 centered on $(1, 2)$.

**Fig. 2.** `expand(plus, 1)`: every cell within one cell of the plus sign, diagonals included.
*b*, a corner of the 5 by 5 square is missing, because it is two cells away from the plus sign.

**Fig. 3.** `expand(plus, 1, {false, true})`: the same, along $y$ only.

**Fig. 4.** `contract(plus, 1)`.
*a*, the center cell survives, although its four diagonal neighbors are not in the set: `contract` checks only the neighbors along the axes.

**Fig. 5.** `translate(plus, {1, -1})`: one cell right, one cell down.

In Figs. 2 to 5 the results are hatched in [red]{.sm-red}, under the outline of the plus sign.
```

The cell $(1, 2)$ survives `contract(plus, 1)` although its four diagonal neighbors are not in the set ({ref}`plate-plus`, Fig. 4).

## Level of a set expression

Every set expression has a level, returned by `level()`:

- a level cell array or a box has its own level;
- `union_`, `intersection` and `difference` take the finest level of their operands; a coarser operand is refined to that level before the comparison;
- `translate`, `expand` and `contract` keep the level of their operand;
- `a.on(level)` has the level `level`.

### Set projection with `on`

```{definition} set projection
:label: def-set-projection

`a.on(level)` brings the set expression `a` to `level`: this is a set projection.
```

The level does not have to exist in the mesh.

- To a finer level, each cell becomes $2^s$ cells per direction, with $s$ the difference of levels: `[2,4)` at level 1 becomes `[8,16)` at level 3.
- To a coarser level, a coarse cell belongs to the result as soon as one of its fine cells does: the set projection rounds outwards. The start of an interval is divided by $2^s$ and rounded down, the end divided by $2^s$ and rounded up, so `[3,6)` at level 2 becomes `[1,3)` at level 1 and `[0,2)` at level 0.

```{diagram}
:figure: set_projection_rounding

The interval $[3, 6)$ of level 2 and, in red, its set projections on levels 1 and 0, at true size.
Dotted red lines mark the bounds of $[3, 6)$.
The hatched coarse cells hold only part of $[3, 6)$ but belong to the result, so the result covers more than $[3, 6)$.
```

The operands are compared at the finest level of the expression, and `on` applies to the result.
In the following figure, level 1 holds the interval $[0, 4)$ and level 0 the interval $[1, 3)$:

```{diagram}
:figure: set_projection

Level 1 holds $[0, 4)$, level 0 holds $[1, 3)$.
Their intersection is computed on level 1, then brought to levels 0 and 3 by `on`.
```

Compared without a set projection, the two intervals would give the intersection $[1, 3)$, which is wrong: the cell 1 of level 0 covers the cells 2 and 3 of level 1, and the cell 2 of level 0 is outside level 1.
The following program computes the intersection on its default level, then on levels 0 and 3:

```{literalinclude} snippet/set_algebra/set_algebra_levels.cpp
  :language: c++
  :start-at: auto set = samurai::intersection
  :end-at: });
  :dedent: 4
```

The output is:

```{literalinclude} snippet/set_algebra/set_algebra_levels_output.txt
  :language: text
```

The intersection is $[2, 4)$ on level 1, the finest level of its operands.
On level 0, it is the cell 1, and on level 3 the cells 8 to 15.

## Traversal

A set expression produces its cells only when it is traversed.
The traversal visits the intervals along $x$ in increasing order, row by row.
Each traversal allocates its own scratch memory, so traversing an expression does not modify it.

| Call | Effect |
| --- | --- |
| `a(func)` | calls `func(interval, index)` for each interval of `a` |
| `a.apply_op(op, ...)` | calls `op(level, interval, index)` for each interval of `a` and each `op`, in the order given |
| `samurai::for_each_interval(a, func)` | calls `func(level, interval, index)` for each interval of `a` |

The arguments passed to the callbacks are:

- `level`: a `std::size_t`, the level of `a` (`a.level()`);
- `interval`: the interval along $x$, in cells of that level;
- `index`: an `xt::xtensor_fixed<value_t, xt::xshape<dim - 1>>` holding the coordinates along $y$ and $z$; it is empty in 1D.

The intervals of a set expression describe cells, not storage: their `index` field is 0.
To read or write a field on them, index the field with the level and the interval, `u(level, interval, index...)`.

The callbacks of `apply_op` are subset operators.
A field operator built with `samurai::make_field_operator_function` has the right signature; the {doc}`tutorial on subset operators <../tutorial/operator_on_subset>` writes one that works in 1D, 2D and 3D.

## Queries

| Member | Result |
| --- | --- |
| `level()` | the level of the set expression |
| `exist()` | `false` when the set expression is known to be empty without traversing it |
| `empty()` | `true` when the set expression has no cell |
| `to_lca()` | a {cpp:class}`samurai::LevelCellArray` at `level()` holding the cells of the set expression |
| `to_lca(origin_point, scaling_factor)` | the same, with the given origin point and scaling factor |
| `on(level)` | the set expression brought to `level` |

`exist()` is a cheap, conservative test.
It returns `false` for an empty level cell array or an invalid box, propagates through `translate`, `expand`, `contract` and `on`, and combines the operands of an n-ary operation: `union_` exists when one operand exists, `intersection` when all of them exist, `difference` when its first operand exists.
`exist()` can return `true` for an empty set expression: the intersection of two disjoint non-empty sets exists, but is empty.
A traversal starts only when `exist()` is `true`.

`empty()` gives the exact answer.
For `union_`, `intersection`, `difference` and `contract`, it traverses the set expression until it finds a first interval.

`to_lca()` traverses the set expression once and stores the result.
Use it when a set expression is traversed many times, or when it must outlive the level cell arrays it is built from.

## Dynamic set algebra

The static set algebra encodes the shape of an expression in its C++ type: the number of operands of a union, for example, is fixed at compile time.
The dynamic set algebra builds the same expressions at runtime.
Use it when the structure of an expression depends on runtime data, such as a list of operands whose length is only known at runtime, or to expose the set algebra to bindings.

It is in the header `samurai/subset/dynamic/dynamic.hpp`.
A dynamic set expression has the type `samurai::DynamicSet<dim, TInterval>`, with `TInterval` the interval type of the level cell arrays, such as `samurai::LevelCellArray<dim>::interval_t`.

### Operations

The operations are free functions in the `samurai::dyn` namespace.
They mirror the static ones and return a `DynamicSet`.

| Function | Result |
| --- | --- |
| `dyn::self(lca)` | the cells of the level cell array `lca` |
| `dyn::box(level, box)` | the cells of the box `box` at `level` |
| `dyn::union_(sets)`, `dyn::union_(a, ...)` | union of a `std::vector<DynamicSet>` or of `DynamicSet` arguments |
| `dyn::intersection(sets)`, `dyn::intersection(a, b, ...)` | intersection, two sets or more |
| `dyn::difference(sets)`, `dyn::difference(a, b, ...)` | the cells of the first set that belong to none of the others, two sets or more |
| `dyn::translate(a, t)` | `a` shifted by `t` |
| `dyn::expand(a, width)` | `a` expanded by `width` cells (an `int`) in every direction |
| `dyn::contract(a, width)`, `dyn::contract(a, width, directions)` | `a` contracted by `width` cells, in every direction or in the given ones |
| `dyn::on(a, level)` | `a` brought to `level`; same as `a.on(level)` |

`dyn::self` and `dyn::box` store a pointer to their argument: the level cell array or the box must outlive the expression.
`dyn::expand` has no per-direction variant.

### Members of `DynamicSet`

| Member | Result |
| --- | --- |
| `level()` | the level of the set expression |
| `on(level)` | the set expression brought to `level` |
| `operator()(func)` | traversal: calls `func(interval, index)` for each interval |
| `to_lca()` | a {cpp:class}`samurai::LevelCellArray` holding the cells |
| `clone()` | a deep copy with its own traversal scratch memory |
| `ptr()` | the underlying `std::shared_ptr<samurai::ISet<dim, TInterval>>` |

`DynamicSet` has no `apply_op`, `exist()` or `empty()` member.
`samurai::as_static_set(set.ptr())` wraps a dynamic set expression into a static one: the result accepts every static operation and member, including `apply_op`, and can be an operand of a static set operation.
`set.ptr()->exist()` and `set.ptr()->empty()` give the two queries directly.

### Threads

A `DynamicSet` is a handle: copying it shares the expression and the scratch memory used by its traversal.
Two threads must not traverse the same expression, nor two copies of it, at the same time.
Give each thread its own `clone()`.

The following program builds the union of a list of level cell arrays whose length is known at runtime, brings it to level 1, and traverses it from two threads:

```{literalinclude} snippet/set_algebra/set_algebra_dynamic.cpp
  :language: c++
  :start-at: // The number of operands
  :end-at: thread.join();
  :dedent: 4
```

Each thread gets the same intervals:

```{literalinclude} snippet/set_algebra/set_algebra_dynamic_output.txt
  :language: text
```

## See also

- The {doc}`tutorial on subset operators <../tutorial/operator_on_subset>` uses intersections, contractions, `on` and `apply_op` to compute a field projection from one level to another.
- The {doc}`interval tutorial <../tutorial/interval>` shows how a level cell array stores its intervals.
- The {doc}`loop how-to <../howto/loop>` loops over the intervals and cells of a set expression.
