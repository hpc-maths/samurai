# Loops and cell search

This page documents `samurai/algorithm.hpp`: the loop functions that visit the levels, intervals or cells of a mesh, and the functions that find a cell from its coordinates or its indices.
For how to choose and write a loop, see the {doc}`loop how-to <../howto/loop>` and the {doc}`tutorial on cell and interval loops <../tutorial/algorithm>`.

## Loop overloads

The reference below lists every overload declared in `algorithm.hpp`; the table groups them by function.
Called on a mesh, a loop visits the `cells` collection of the mesh (`mesh[mesh_id_t::cells]`).

| Function | Visits | Accepted arguments |
| --- | --- | --- |
| `for_each_level` | each level, as `f(level)` | a `CellArray` or a mesh; empty levels are skipped unless `include_empty_levels` is `true` |
| `for_each_interval` | each interval, as `f(level, interval, index)` | a `LevelCellArray` or a `CellArray` (`const` or not), a mesh, or a set expression |
| `for_each_meshinterval` | each interval, as `f(mesh_interval)` | a `LevelCellArray`, a `CellArray`, or a set expression with the `MeshInterval` type as template argument |
| `for_each_cell` | each cell, as `f(cell)` | a `LevelCellArray`, a `CellArray`, a mesh or a `hold<Mesh>`; a mesh with a set expression; a mesh with a level, an interval and the indices along the other directions |
| `parallel_for_each_cell`, `parallel_for_each_meshinterval` | as `for_each_cell` and `for_each_meshinterval` | a `LevelCellArray` and a set expression respectively |

`for_each_cell` and `for_each_meshinterval` also take a `samurai::Run` template argument: `Run::Parallel` calls the `parallel_` version, which spreads the work over OpenMP tasks, and `Run::Sequential` (the default) the sequential one.

```{remark}
The overload `for_each_cell(lca, subset_operator<F, CT...> set, f)` takes a `samurai::subset_operator`, a class that `algorithm.hpp` declares but no header defines: this overload cannot be called.
To visit the cells of a set expression, pass the mesh and the expression: `for_each_cell(mesh, set, f)`.
```

## Cell search

`find` takes the integer coordinates of a cell in a `LevelCellArray` and returns the position, among the intervals along x, of the interval that contains it, or -1 when no interval does.
`find_on_dim` searches one coordinate among the intervals along one direction, between two positions, and returns `std::numeric_limits<std::size_t>::max()` when no interval contains it.

`find_cell` returns the cell of a `LevelCellArray`, a `CellArray` or a mesh that contains a point given by its Cartesian coordinates, or that contains a cell given by its indices.
On a `CellArray` or a mesh, the indices are at the level `level_ref`, the max level by default, and the levels are searched from the coarsest.
When no cell matches, the returned cell has a `length` of 0.

## Reference

```{doxygenfile} algorithm.hpp
```
