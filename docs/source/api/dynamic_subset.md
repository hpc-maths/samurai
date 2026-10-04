# Dynamic set algebra

This page documents the dynamic set algebra of {{ project }}: the `samurai::DynamicSet` handle and the functions of the namespace `samurai::dyn` that build it.
The static set algebra of {doc}`subset` fixes the shape of an expression in its C++ type; the dynamic one builds the same expressions at runtime, for example a union of a number of operands that is known only at runtime.
A `DynamicSet<dim, TInterval>` is a handle on a runtime expression, where `TInterval` is the interval type of the level cell arrays, such as `samurai::LevelCellArray<dim>::interval_t`.
Include `samurai/subset/dynamic/dynamic.hpp` to get all of them.
For the operations, their limits and an example with threads, see the "Dynamic set algebra" section of the {doc}`set algebra reference <../reference/subset>`.

## Handle

`DynamicSet` mirrors part of the static interface: `level()`, `on(level)`, the traversal `operator()(func)` and `to_lca()`.
It has no `apply_op`, `exist()` or `empty()` member: call them on `as_static_set(set.ptr())` or on `set.ptr()`.
Copying a `DynamicSet` shares the expression and the scratch memory of its traversal, so two threads must not traverse the same expression at the same time; give each thread its own `clone()`.

```{doxygenclass} samurai::DynamicSet
:members:
:undoc-members:
```

## Operations

The functions of `samurai::dyn` mirror the static functions and return a `DynamicSet`:

| Function | Result |
| --- | --- |
| `self(lca)` | the cells of the level cell array `lca` |
| `box(level, box)` | the cells of `box` at `level` |
| `union_(sets)`, `union_(a, ...)` | the union of a `std::vector<DynamicSet>` or of `DynamicSet` arguments |
| `intersection(sets)`, `intersection(a, b, ...)` | the intersection of two sets or more |
| `difference(sets)`, `difference(a, b, ...)` | the cells of the first set that belong to none of the others, two sets or more |
| `translate(set, t)` | `set` shifted by `t` |
| `expand(set, width)` | `set` expanded by `width` cells in every direction |
| `contract(set, width)`, `contract(set, width, directions)` | `set` contracted by `width` cells, in every direction or in the selected ones |
| `on(set, level)` | `set` brought to `level`, like `set.on(level)` |

`self` and `box` keep a pointer to their argument: the level cell array or the box must outlive the expression.
The variadic overloads take operands of the same `DynamicSet` type and build the `std::vector` overload from them.

```{doxygennamespace} samurai::dyn
:content-only:
:members:
:undoc-members:
```

## Runtime interface

A `DynamicSet` points to a `samurai::ISet`, the abstract interface that each node of a runtime expression implements, and `ptr()` returns this pointer.
`as_static_set` wraps an `ISet` into a `samurai::DynamicSetAdaptor`, which derives from {cpp:class}`samurai::SetBase`: the result accepts every static operation and member, and can be an operand of a static set expression.

```{doxygenclass} samurai::ISet
:members:
:undoc-members:
```

```{doxygenfunction} samurai::as_static_set
```

```{doxygenclass} samurai::DynamicSetAdaptor
```
