# How-to: loop over cells in a samurai mesh

This guide shows how to visit the cells of a samurai mesh, either one cell at a time with `samurai::for_each_cell` or one interval of cells at a time with `samurai::for_each_interval`, and how to set the values of a field in both cases.

## Before you start

You need a samurai mesh. If you don't have one yet, see the {doc}`mesh how-to <mesh>`. The examples that set field values also create a scalar field; the {doc}`field how-to <field>` explains how.

Called on a mesh, both loops visit the cells that hold the solution (`mesh[mesh_id_t::cells]`), at whatever level they sit. To visit other cells, such as the ghost cells, pass a sub-mesh or a subset instead of the mesh: see {ref}`howto-loop-choose-cells`.

## Looping over cells

`samurai::for_each_cell` calls a function once per cell and passes it a `samurai::Cell`. The following example creates a 2D multiresolution mesh on the box $[0, 1]^2$ with a minimum level of 2 and a maximum level of 5, then prints the level and the center of each cell:

```{literalinclude} snippet/loop/for_each_cell_mesh.cpp
  :language: c++
```

A multiresolution mesh starts with all its cells at the maximum level, so the program prints 1024 lines ($32 \times 32$ cells at level 5). The cells come row by row, from left to right in each row:

```text
Cell level: 5, center: { 0.015625,  0.015625}
Cell level: 5, center: { 0.046875,  0.015625}
...
Cell level: 5, center: { 0.984375,  0.984375}
```

Besides `level` and `center()`, a cell gives its length (`cell.length`), the center along one direction (`cell.center(d)`) and its position in the field storage (`cell.index`). See the {doc}`Cell API page <../api/cell>` for the full list.

## Setting field values cell by cell

Inside the loop, `field[cell]` gives the value of a field at that cell, and you can read or write it. The following example initializes a scalar field `u` with a Gaussian centered at $(0.5, 0.5)$, $u(x, y) = e^{-20\left((x - 0.5)^2 + (y - 0.5)^2\right)}$:

```{literalinclude} snippet/loop/for_each_cell_field.cpp
  :language: c++
```

`cell.center(0)` and `cell.center(1)` are the $x$ and $y$ coordinates of the cell center. The program prints nothing; to look at the field, save it as described in the {doc}`save how-to <save>`.

For a vector field, `field[cell]` holds all the components of the cell. Add a second `[]` to reach one component:

```cpp
field[cell][component_index] = value;
```

## Looping over intervals

An interval is a run of consecutive cells along $x$ at the same level, with the same $y$ (and $z$) indices. `samurai::for_each_interval` calls a function once per interval with three arguments: the level, the interval and `index`, an array of size $dim - 1$ that holds the indices of the other directions ($y$, $z$). Working on whole intervals lets you write vectorized expressions instead of one statement per cell. For more on intervals, see the {doc}`interval tutorial <../tutorial/interval>`.

The following example prints the level, the interval and the $y$ index of each interval of the same mesh as above:

```{literalinclude} snippet/loop/for_each_interval_mesh.cpp
  :language: c++
```

All the cells are at level 5, so there is one interval per row: the program prints 32 lines, with `y` going from 0 to 31. An interval prints as `[start,end[@offset:step`, where `start` and `end` are the first index and one past the last index along $x$, `offset` is the shift that gives the position of the cells in the field storage, and `step` is the stride between indices. Each line has the form:

```text
Level: 5, x: [0,32[@<offset>:1, y: 0
```

## Setting field values interval by interval

The next example initializes the same Gaussian as above, on the box $[-1, 1]^2$ with levels 0 to 2, one interval at a time:

```{literalinclude} snippet/loop/for_each_interval_field.cpp
  :language: c++
```

Inside the loop:

- `j = index[0]` is the $y$ index of the interval.
- `xt::arange(i.start, i.end)` lists the $x$ indices of the cells of the interval. Adding 0.5, multiplying by the cell length `mesh.cell_length(level)` and adding the origin of the mesh `mesh.origin_point()[0]` gives the $x$ coordinates of the cell centers, as an xtensor expression.
- `y` is the $y$ coordinate of the cell centers of the row, a single number.
- `field(level, i, j)` is a view on the values of the field for the cells of the interval. Assigning an xtensor expression to it writes all these values at once.

The program prints one line per interval, with the level, the $x$ coordinates of the cell centers of the interval and their common $y$ coordinate.

For a vector field, put the component index first:

```cpp
field(component_index, level, i, j) = value;
```

In 3D, add the $z$ index after `j`: `field(level, i, j, k)`.

```{warning}
`field(level, i, j)` is an xtensor view, not a number, so the math functions of the standard library such as `std::sin` and `std::exp` don't accept it. Use the xtensor functions instead, such as `xt::sin` and `xt::exp`, as the example does.
```

(howto-loop-choose-cells)=

## Choosing the cells to visit

A samurai mesh stores several sets of cells, selected by a `mesh_id_t` value. `for_each_cell(mesh, f)` and `for_each_interval(mesh, f)` use `mesh[mesh_id_t::cells]`. Pass another set to visit other cells:

```c++
using mesh_id_t = typename std::decay_t<decltype(mesh)>::mesh_id_t;

// the cells and their ghost cells
samurai::for_each_cell(mesh[mesh_id_t::cells_and_ghosts],
                       [&](const auto& cell)
                       {
                           std::cout << cell.level << " " << cell.center() << std::endl;
                       });

// the cells of one level
samurai::for_each_interval(mesh[mesh_id_t::cells][level],
                           [&](std::size_t level, const auto& i, const auto& index)
                           {
                               std::cout << level << " " << i << " " << index << std::endl;
                           });

// the ghost cells of one level only, built with the set algebra
auto ghosts = samurai::difference(mesh[mesh_id_t::cells_and_ghosts][level], mesh[mesh_id_t::cells][level]);
samurai::for_each_cell(mesh,
                       ghosts,
                       [&](const auto& cell)
                       {
                           std::cout << cell.center() << std::endl;
                       });
```

`mesh[id]` holds every level and `mesh[id][level]` one level. A subset is any expression of the set algebra (`intersection`, `union_`, `difference`, `translate`, ...); `for_each_interval(set, f)` loops over its intervals and `for_each_cell(mesh, set, f)` over its cells.

The available sets depend on the mesh type:

| `mesh_id_t` | Uniform mesh | Multiresolution mesh | AMR mesh | Cells |
| --- | --- | --- | --- | --- |
| `cells` | yes | yes | yes | the cells that hold the solution |
| `cells_and_ghosts` | yes | yes | yes | the cells and their ghost cells |
| `proj_cells` | | yes | yes | the cells computed by projection from the finer level |
| `pred_cells` | | | yes | the cells filled by the prediction operator |
| `union_cells` | | yes | | the cells used by the multiresolution analysis |
| `all_cells` | | yes | yes | every cell stored by the mesh |

## Running the cell loop in parallel

`samurai::for_each_cell` takes an optional template argument that selects how the loop runs. With `samurai::Run::Parallel`, the cells of each level are split into OpenMP tasks:

```cpp
samurai::for_each_cell<samurai::Run::Parallel>(mesh,
                                               [&](const auto& cell)
                                               {
                                                   field[cell] = cell.center(0);
                                               });
```

The loop runs in parallel only when samurai is built with OpenMP: set the CMake option `WITH_OPENMP=ON` when you build samurai, or `SAMURAI_WITH_OPENMP=ON` in a project that finds samurai with `find_package`. Otherwise it runs sequentially. The function is then called from several threads at once, so it must only write to data that belongs to the current cell. The default, `samurai::Run::Sequential`, visits the cells in the order shown above. `samurai::for_each_interval` has no parallel variant.

## Next steps

- {doc}`Save a field <save>` to look at the values you set, and {doc}`plot it <plot>`.
- {doc}`Loop tutorial <../tutorial/algorithm>`: when to loop over cells and when over intervals.
- {doc}`Interval tutorial <../tutorial/interval>`: how samurai stores a mesh as intervals.
- {doc}`Algorithm API page <../api/algorithm>`: every overload of the loop functions.
