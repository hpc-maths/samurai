# Set algebra

This page documents the set algebra of {{ project }}: the `samurai::SetBase` base class, the free functions that build set expressions (`union_`, `intersection`, `difference`, `translate`, `expand`, `contract`), and the expression classes these functions return.
A set expression is lazy: it holds its operands, not cells, and produces intervals only when it is traversed with `operator()`, `apply_op` or `samurai::apply`.
The operands are levels of a cell array (`LevelCellArray`) or other set expressions, and `on(level)` brings the result to a given level.
All these entities are declared in the headers of `include/samurai/subset/`.
For a guided example, see the {doc}`tutorial on applying an operator on a set <../tutorial/operator_on_subset>`; for a description of each operation with figures, see the {doc}`algebra of set reference <../reference/subset>`.

## Base class

Every set expression derives from `samurai::SetBase`, which gives the members shared by all of them: the level of the result, the level change with `on`, the traversal and the conversion to a `LevelCellArray` with `to_lca`.

```{doxygenclass} samurai::SetBase
:members:
:undoc-members:
```

## Set operations

`union_` takes one or more sets; `intersection` and `difference` take at least two.
The result of these three functions is at the finest level of their operands.

```{doxygenfunction} samurai::union_
```

```{doxygenfunction} samurai::intersection
```

```{doxygenfunction} samurai::difference
```

```{doxygenfunction} samurai::translate
```

`expand` grows a set by the same width in every direction, by one width per direction, or by one width in the selected directions:

```{doxygenfunction} samurai::expand(const Set &set, const typename Expansion<std::decay_t<decltype(self(set))>>::expansion_t &expansions)
```

```{doxygenfunction} samurai::expand(const Set &set, const typename Expansion<std::decay_t<decltype(self(set))>>::value_t &expansion)
```

```{doxygenfunction} samurai::expand(const Set &set, const typename Expansion<std::decay_t<decltype(self(set))>>::value_t &expansion, const typename Expansion<std::decay_t<decltype(self(set))>>::do_expansion_t &do_expansion)
```

```{doxygenfunction} samurai::contract(const SubsetOrLCA &set, std::size_t width)
```

```{doxygenfunction} samurai::contract(const SubsetOrLCA &set, std::size_t width, const std::array<bool, SubsetOrLCA::dim> &contract_directions)
```

## Traversal

`samurai::apply` traverses a set expression and calls a function on each interval it yields along x, with the interval and the indices along the other directions.
`SetBase::operator()` calls `apply` on the expression itself, and `SetBase::apply_op` calls each function it receives with the level, the interval and the indices.

```{doxygenfunction} samurai::apply
```

## Expression classes

The set operations return the classes below.
`union_`, `intersection` and `difference` return a `NArySetOperator` tagged with a `SetOperator`, `translate` returns a `Translation`, `expand` an `Expansion`, and `SetBase::on` a `Projection`.
`contract` returns the intersection of translated copies of the set, that is a `NArySetOperator`; no free function returns a `Contraction`.

```{doxygenenum} samurai::SetOperator
```

```{doxygenclass} samurai::NArySetOperator
```

```{doxygenclass} samurai::Translation
```

```{doxygenclass} samurai::Expansion
```

```{doxygenclass} samurai::Contraction
```

```{doxygenclass} samurai::Projection
```
