# Graduation case 2: cells with overlap

In this tutorial, we start from a 2D mesh made of cells at several levels that overlap: a region of the domain can be covered by cells of two levels at once.
We first remove the overlaps, then we make the mesh graded with the algorithm of {doc}`graduation case 1 <graduation_case_1>`.
The complete program is the demo {download}`graduation_case_2.cpp <../../../demos/tutorial/graduation_case_2.cpp>`.

This page is part of the {doc}`graduation series <graduation>`, which lists what you need before you start.

## Build and run the demo

From the root of the repository, build the demo target and run it:

```bash
cmake --build build --target tutorial-graduation-case-2
./build/demos/tutorial/tutorial-graduation-case-2
```

The demo prints one line per iteration of each loop, `Iteration for remove intersection: <n>` and then `Iteration for graduation: <n>`.
It writes three meshes in the current directory, each as an `.xdmf` file and its `.h5` data file:

- `graduation_case_2_initial`: the mesh with overlaps;
- `graduation_case_2_without_intersection`: the mesh after the removal of the overlaps;
- `graduation_case_2_graduated`: the graded mesh.

The demo accepts these options:

| Option | Default | Effect |
| --- | --- | --- |
| `--minimum-level` | 1 | Lowest level of the cells added at random. |
| `--maximum-level` | 7 | Bound of the levels of the cells added at random; the levels drawn stay below it. |
| `--with-corner` | off | Uses the four axis directions as graduation stencil instead of the four diagonal directions, as in case 1. |
| `--path` | current directory | Directory of the output files. |
| `--filename` | `graduation_case_2` | Prefix of the output file names. |

`--help` lists them with the options that every {{ project }} program has.

## Generate a mesh with overlaps

We build the initial mesh in the domain $[0, 1] \times [0, 1]$ by adding cells at random levels and positions:

```{literalinclude} ../../../demos/tutorial/graduation_case_2.cpp
:language: c++
:start-at: auto generate_mesh
:end-before: int main(
```

`min_level` and `max_level` bound the levels of the added cells, and `nsamples` is the number of cells added.

The function first adds the cell $(0, 0)$ at level 0, which covers the whole domain $[0, 1] \times [0, 1]$.
This cell makes sure that the final mesh covers the whole domain.

Then it adds `nsamples` cells, with these random draws:

- the level, between `min_level` and `max_level - 1`;
- the indices `x` and `y` of the cell at this level, between 0 and $2^{\text{level}} - 2$.

`xt::random::randint` draws in the half-open range `[lower, upper)`, which is why the upper bounds are excluded.
`xt::random::seed(42)` fixes the seed of the random generator, so the demo builds the same mesh at every run.

`samurai::CellArray<dim>(cl, true)` builds the cell array from the cell list and computes the index of each interval, which we need to attach the `tag` field to the mesh.

The figure below shows a mesh built this way: small cells lie on top of the large cells that contain them.

```{image} ./figures/graduation_case_2_before.png
:alt: A square mesh of 16 large cells with small squares of several sizes drawn on top of them, inside the large cells.
:width: 60%
:align: center
```

## Remove the overlaps

A cell at level `level_below` overlaps the cells of a finer level `level` when their intersection is not empty.
We refine every such coarse cell, and we repeat until no overlap remains.
As in case 1, a boolean field `tag` marks the cells to refine, and only the subset changes: the cells of `level` are not translated.

```{literalinclude} ../../../demos/tutorial/graduation_case_2.cpp
:language: c++
:start-at: std::size_t ite = 0;
:end-before: samurai::save(path,
:dedent:
```

For each pair of levels with `level_below < level`, `intersection(ca[level], ca[level_below]).on(level_below)` gives the cells of `level_below` that lie under cells of `level`, and we tag them.
We then build the new mesh from `tag` as in case 1: each tagged cell is replaced by its four children.
A coarse cell can lie several levels under a fine cell, so the loop runs until the mesh no longer changes.

At the end, the overlapping coarse cells have been replaced by cells around the fine ones.
The blue lines in the figure below are the cells added to remove the overlaps.

```{image} ./figures/graduation_case_2_after.png
:alt: The same mesh without overlaps; blue lines mark the cells added around each small cell, which split the large cells into smaller ones.
:width: 60%
:align: center
```

## Make the mesh graded

The mesh has no overlap left, so we grade it with the loop of {doc}`graduation case 1 <graduation_case_1>`, with the same stencil:

```{literalinclude} ../../../demos/tutorial/graduation_case_2.cpp
:language: c++
:start-after: // Make the mesh graded
:end-before: samurai::save(path, fmt::format("{}_graduated"
:dedent:
```

The red lines in the figure below are the cells added by the graduation.

```{image} ./figures/graduation_case_2_after_graduated.png
:alt: The mesh without overlaps after graduation; red lines mark the added cells, which form layers of intermediate sizes around the small cells.
:width: 60%
:align: center
```

## What we built

We turned a set of overlapping cells into a graded mesh in two steps: an intersection of the levels without translation removes the overlaps, then the graduation of case 1 removes the jumps of more than one level between neighbors.

## Next steps

- {doc}`graduation_case_3` keeps a mesh graded while a criterion refines it.
- {doc}`graduation` lists the graduation functions of the library.
