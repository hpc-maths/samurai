# Graduation case 1: cells without overlap

In this tutorial, we start from a 2D mesh made of cells at several levels that do not overlap, and we refine cells until the mesh is graded.
The complete program is the demo {download}`graduation_case_1.cpp <../../../demos/tutorial/graduation_case_1.cpp>`.

This page is part of the {doc}`graduation series <graduation>`, which lists what you need before you start.

## Build and run the demo

From the root of the repository, build the demo target and run it:

```bash
cmake --build build --target tutorial-graduation-case-1
./build/demos/tutorial/tutorial-graduation-case-1
```

The demo writes two meshes in the current directory: `graduation_case_1_before_graduation` and `graduation_case_1_after_graduation`, each as an `.xdmf` file and its `.h5` data file.
Open both `.xdmf` files in your viewer to compare the mesh before and after the graduation.

The demo accepts these options:

| Option | Default | Effect |
| --- | --- | --- |
| `--starting-level` | 1 | Level of the uniform mesh the generator starts from. |
| `--max-refinement-level` | 7 | Finest level the generator can reach. |
| `--with-corner` | off | Uses the four axis directions as stencil instead of the four diagonal directions (see [the stencil section](#choose-the-directions)). |
| `--path` | current directory | Directory of the output files. |
| `--filename` | `graduation_case_1` | Prefix of the output file names. |

`--help` lists them with the options that every {{ project }} program has.

## Generate a mesh at random

We build the initial mesh in the domain $[0, 1] \times [0, 1]$ with this function:

```{literalinclude} ../../../demos/tutorial/graduation_case_1.cpp
:language: c++
:start-at: auto generate_mesh
:end-before: int main(
```

`start_level` is the level of the uniform mesh we start from, and `max_level` is the finest level a cell can reach.

The function starts with a uniform mesh at `start_level`:

```{literalinclude} ../../../demos/tutorial/graduation_case_1.cpp
:language: c++
:start-at: constexpr std::size_t dim = 2;
:end-at: ca[start_level] = {start_level, box};
:dedent:
```

A {cpp:class}`samurai::Box` of integers is defined by its minimum and maximum corners, in cell indices.
At level $l$ the cell length is $\Delta x = 2^{-l}$, so $2^{l}$ cells cover $[0, 1]$ in each direction: the box goes from $(0, 0)$ to $(2^{l}, 2^{l})$, and `1 << start_level` computes $2^{l}$.
We assign this box to the level `start_level` of the {cpp:class}`samurai::CellArray`.

`xt::random::seed(42)` fixes the seed of the random generator of xtensor, so the demo builds the same mesh at every run.

Then the function refines cells at random, `max_level - start_level` times, so that the finest cells can reach `max_level`.
Each pass browses the intervals of the mesh:

```{literalinclude} ../../../demos/tutorial/graduation_case_1.cpp
:language: c++
:start-at: using bools_t
:end-at: ca = {cl, true};
:dedent:
```

{cpp:func}`samurai::for_each_interval` calls the lambda for each interval of each level.
The lambda receives the level, the interval along x and an array `index` with the indices along the other directions: in 2D, `index` holds the y index.

For each interval, `xt::random::choice` draws `true` or `false` for every cell.
A cell drawn `true` is replaced by its four children at `level + 1`: the rows `2 * index` and `2 * index + 1`, and the columns `2 * i` and `2 * i + 1`.
A cell drawn `false` is kept as it is.
We add the new cells to a {cpp:class}`samurai::CellList`, which is faster to fill than a {cpp:class}`samurai::CellArray`.

At the end of each pass, `ca = {cl, true}` builds the new {cpp:class}`samurai::CellArray` from the {cpp:class}`samurai::CellList`.
The `true` argument asks the cell array to compute the index of each interval: this index gives the position of the interval in the data array of a field attached to the mesh.
We need it because we attach a field to the mesh in the next step.

The figure below shows a mesh built this way.
Neighboring cells can differ by several levels.

```{image} ./figures/graduation_case_1_before.png
:alt: A square mesh with cells of many sizes; small cells often touch cells four or more times larger.
:width: 60%
:align: center
```

## Find the cells to refine

The mesh is graded when a cell at level $l$ has no neighbor at a level $L \leq l - 2$.
To find the cells that break this rule, we use the set algebra of {{ project }}: we translate the cells of level $l$ by one cell in a given direction, and we intersect them with the cells of a level $L \leq l - 2$.
A non-empty intersection is a coarse cell at level $L$ that touches a cell at level $l$: we must refine it.

We store this decision in a field named `tag`, with one boolean per cell:

```{literalinclude} ../../../demos/tutorial/graduation_case_1.cpp
:language: c++
:start-at: auto tag = samurai::make_scalar_field<bool>
:end-at: tag.fill(false);
:dedent:
```

`true` means that the cell must be refined, `false` that it is kept.
We start with `false` everywhere.

The intersection is computed for each pair of levels and each direction of a stencil:

```{literalinclude} ../../../demos/tutorial/graduation_case_1.cpp
:language: c++
:start-at: for (std::size_t level = min_level + 2;
:end-before: // Build the refined mesh
:dedent:
```

`s` is a row of the stencil, that is a translation vector: `{1, 1}` moves the cells one cell to the right and one cell up.
The subset {cpp:func}`samurai::intersection` of `translate(ca[level], s)` and `ca[level_below]` is computed by default on the finest of its levels, `level`.
We want the coarse cells to tag, so `.on(level_below)` projects the result on `level_below`.

Calling the subset with a lambda runs it on every interval of the result.
The lambda receives the interval `i` along x and the array `index` with the y index, and `tag(level_below, i, index[0])` gives the values of `tag` on these cells.
We set them to `true`.

(choose-the-directions)=

## Choose the directions

The demo uses the four diagonal directions as stencil by default:

```{literalinclude} ../../../demos/tutorial/graduation_case_1.cpp
:language: c++
:start-at: xt::xtensor_fixed<int, xt::xshape<4, dim>> stencil;
:end-before: while (true)
:dedent:
```

The diagonal directions find the coarse cells that touch a fine cell through a corner, and also those that touch it through a face.
A coarse cell at level $L \leq l - 2$ spans at least four fine cells along each direction.
When it touches a fine cell through a face, one of the two diagonal translations toward it lands inside the coarse cell.
The four axis directions, which `--with-corner` selects, only find the cells that touch through a face.

## Build the refined mesh

When the `tag` field is filled, we build the new mesh from it:

```{literalinclude} ../../../demos/tutorial/graduation_case_1.cpp
:language: c++
:start-after: // Build the refined mesh
:end-at: samurai::CellArray<dim> new_ca = {cl, true};
:dedent:
```

{cpp:func}`samurai::for_each_cell` calls the lambda for each cell of the mesh.
`cell.indices` holds the integer coordinates of the cell, and `tag[cell]` its tag.
A tagged cell is replaced by its four children, and any other cell is kept.

## Repeat until the mesh is graded

One refinement is not always enough.
If a cell at level $L = 1$ touches a cell at level $l = 5$, refining it creates cells at level 2, which are still three levels away from level 5.
So we repeat the tagging and the refinement until the mesh no longer changes:

```{literalinclude} ../../../demos/tutorial/graduation_case_1.cpp
:language: c++
:start-at: while (true)
:end-before: samurai::save(path,
:dedent:
```

`new_ca == ca` compares the two cell arrays: when no cell was tagged, the new mesh equals the old one and the loop stops.
Otherwise, `std::swap` makes the new mesh the current one.

The figure below shows the graded version of the mesh above.
The red lines are the cells added by the graduation.

```{image} ./figures/graduation_case_1_after.png
:alt: The same mesh after graduation; red lines mark the added cells, which form layers of intermediate sizes between small and large cells.
:width: 60%
:align: center
```

## What we built

We made a mesh graded with three tools of {{ project }}: a boolean field to tag cells, subsets built with {cpp:func}`samurai::translate` and {cpp:func}`samurai::intersection` to find the cells to refine, and a {cpp:class}`samurai::CellList` to build the refined mesh.
In your own code, `samurai::make_graduation(ca)` does this work, and {cpp:func}`samurai::is_graduated` checks the result (see {doc}`graduation`).

## Next steps

- {doc}`graduation_case_2` applies the same graduation to a mesh whose levels overlap.
- {doc}`graduation_case_3` keeps a mesh graded while a criterion refines it.
