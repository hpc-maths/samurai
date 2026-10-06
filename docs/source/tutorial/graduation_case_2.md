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

{ref}`plate-graduation-case-2` follows the mesh of the demo through the two steps of this page.
Its Fig. 1 is the mesh built this way: small cells lie on top of the large cells that contain them.

```{plate} The mesh of the demo, from overlaps to a graded mesh
:figure: overlap_removal
:label: plate-graduation-case-2

**Fig. 1.** The initial mesh, 66 cells of levels 0 to 6.
*a*, a cell of level 5 drawn over the coarser cells it lies in; the covered cells are outlined with the heavy rule.

**Fig. 2.** The mesh without overlap, 214 cells of levels 3 to 6.
*b*, the 167 cells added by the removal of the overlaps, in gray wash.
This mesh is not graded: cells of levels 3 and 5 touch.

**Fig. 3.** The graded mesh, 349 cells of levels 3 to 6.
*c*, the 174 cells added by the graduation, hatched in [red]{.sm-red}.

*The meshes are the three files the demo saves with its default options.
It runs 7 iterations of the overlap removal and 3 of the graduation.*
```

## Remove the overlaps

A cell at level `level_below` overlaps the cells of a finer level `level` when their intersection is not empty.
We refine every such coarse cell, and we repeat until no overlap remains.
As in case 1, a boolean field `tag` marks the cells to refine, and only the subset changes: the cells of `level` are not translated.

```{literalinclude} ../../../demos/tutorial/graduation_case_2.cpp
:language: c++
:start-at: std::size_t ite = 0;
:end-before: samurai::save(path, fmt::format("{}_without_intersection"
:dedent:
```

For each pair of levels with `level_below < level`, `intersection(ca[level], ca[level_below]).on(level_below)` gives the cells of `level_below` that lie under cells of `level`, and we tag them.
We then build the new mesh from `tag` as in case 1: each tagged cell is replaced by its four children.
A coarse cell can lie several levels under a fine cell, so the loop runs until the mesh no longer changes.

The diagram below follows the loop on a 1D mesh: a cell of level 0 with a cell of level 3 inside it.
Each iteration tags one coarse cell and splits it, and the new cell that still covers the fine cell is tagged at the next iteration.
The mesh this loop leaves has no overlap but is not graded yet: a cell of level 1 touches a cell of level 3.
The lower part of the diagram is the graduation loop of the next section, which grades it.

```{diagram}
:figure: overlap_passes

Top, the cells tagged by `intersection(ca[level], ca[level_below]).on(level_below)`, in red, iteration after iteration.
Bottom, the graduation of the result with the 1D stencil `{-1, 1}`; the cells it adds are hatched in red.
```

At the end, the overlapping coarse cells have been replaced by cells around the fine ones, as in Fig. 2 of {ref}`plate-graduation-case-2`.

## Make the mesh graded

The mesh has no overlap left, so we grade it with the loop of {doc}`graduation case 1 <graduation_case_1>`, with the same stencil:

```{literalinclude} ../../../demos/tutorial/graduation_case_2.cpp
:language: c++
:start-after: // Make the mesh graded
:end-before: samurai::save(path, fmt::format("{}_graduated"
:dedent:
```

Fig. 3 of {ref}`plate-graduation-case-2` shows the result: the cells added by the graduation form layers of intermediate sizes around the finest cells.

## What we built

We turned a set of overlapping cells into a graded mesh in two steps: an intersection of the levels without translation removes the overlaps, then the graduation of case 1 removes the jumps of more than one level between neighbors.

## Next steps

- {doc}`graduation_case_3` keeps a mesh graded while a criterion refines it.
- {doc}`graduation` lists the graduation functions of the library.
