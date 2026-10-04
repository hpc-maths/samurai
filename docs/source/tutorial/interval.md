# Interval and Cartesian grid representation

In this tutorial, we see how {{ project }} stores a Cartesian mesh with intervals of integers.
We build a 1D and a 2D mesh level by level, print them, and read the printed structure line by line.
At the end, we also build a uniform mesh from a box.

## Before you start

- {{ project }} is installed: see the [installation guide](../howto/installation.md).
- You know how to compile a program against {{ project }}: see the [CMake guide](../howto/cmake.md).
- To run the 2D example as it is, you need a source build of {{ project }}: it is one of the demos.

## Introduction

A Cartesian grid is made of cells with a given edge length.
On a uniform grid, all the cells have the same length.
The core idea in {{ project }} is that a Cartesian grid can be represented by intervals.
To illustrate it, we start with a 1D example:

```{image} ./figures/segments.png
:alt: A 1D axis from 0 to 13 split into five colored segments: [0, 4] green, [4, 5] red, [5, 7] and [7, 9] blue, [9, 13] green.
:align: center
:width: 80%
```

The whole domain is the interval $[0, 13]$.
It is made of five cells, which are also intervals (from left to right):

- cell 1: $[0, 4]$ (green)
- cell 2: $[4, 5]$ (red)
- cell 3: $[5, 7]$ (blue)
- cell 4: $[7, 9]$ (blue)
- cell 5: $[9, 13]$ (green)

Several cells have the same width, so we can group them in families of equal width (or resolution):

- width 1: $[4, 5]$
- width 2: $[5, 9]$
- width 4: $[0, 4]$, $[9, 13]$

Since we know the resolution of each family, we can merge contiguous cells into one interval and still split it back into cells.
In the family of width 2, the two contiguous cells $[5, 7]$ and $[7, 9]$ form the interval $[5, 9]$.
Since the width of the cells in $[5, 9]$ is 2, the two cells can be reconstructed.

If we plot each family on its own line, one line per resolution, we obtain:

```{image} ./figures/segments-resolution.png
:alt: The same five cells drawn on three rows by cell width: width 1 holds [4, 5], width 2 holds [5, 7] and [7, 9], width 4 holds [0, 4] and [9, 13].
:align: center
:width: 80%
```

In this example, cells of different widths do not overlap.
This is not mandatory: a mesh can have overlapping regions, as in this one:

```{image} ./figures/segments-resolution-overlap.png
:alt: Three rows of cells by width, where cells of different widths cover the same parts of the axis.
:align: center
:width: 80%
```

:::{note}
{{ project }} adds some constraints on the cells, so it cannot build exactly the domains drawn above.
The next sections explain these constraints.
:::

## Interval definition

The interval is the main data structure of {{ project }}.
An interval is printed as follows:

```{image} ./figures/interval.svg
:alt: A printed interval from -5 to 10 with index 0 and step 1, where the labels start, end, index and step point at each part.
:align: center
:width: 300
```

An interval is defined by its start and end values (in black).
It has two more attributes:

- the index (in red), an offset that links the interval to the array where the data are stored (see [Read the printed mesh](#read-the-printed-mesh)),
- the step (in green), the step used to move inside the interval.

:::{warning}

- In {{ project }}, the start and the end of an interval are integers, unlike the real values of the introduction.
- The end value is not included: intervals are closed on the left and open on the right.

:::

As in the introduction, a mesh has several resolutions, which means several cell sizes.
Each resolution is identified by an integer, the level.
The edge length of a cell is fixed by its level:

```{math}
:label: dx

\Delta x = \frac{1}{2^{level}}.
```

This formula uses the default scaling factor of 1.
For a scaling factor $s$, the length is $\Delta x = s / 2^{level}$: the [box section](#build-a-grid-from-a-box) shows a mesh where $s = 2$.
A cell has the same length in every direction, so cells are squares in 2D and cubes in 3D.

Intervals hold integers, not real numbers as in the introduction.
This works because an interval belongs to a level, and the level gives the real size of its cells through {eq}`dx`.
The figure below illustrates the idea:

```{image} ./figures/interval-2.png
:alt: Two adjacent cells numbered 0 and 1 on level i.
:align: center
:width: 250
```

We have two cells $0$ and $1$ on the level $l$, so the interval $[0, 2)$ describes this domain at the level $l$.

:::{note}
We could describe closed intervals such as $[0, 1]$.
Excluding the end of the interval matters for the algebra of intervals (see {ref}`AlgebraOfSet`).
:::

(cell)=

## Cell properties

Given an interval and a level, we can reconstruct the corresponding cells:

- The number of cells in an interval is the size of the interval:

  $$
  size = end - start.
  $$

- The "minimal" corner of a cell (left in 1D, lower-left in 2D, lower-left-back in 3D) with integer coordinates $(i, j, ...)$ is at

  $$
  corner = (i \Delta x, j \Delta x, ...),
  $$

  where $\Delta x$ is given by {eq}`dx`.

- The center of this cell is at

  $$
  center = \left(\left(i + \frac{1}{2}\right) \Delta x, \left(j + \frac{1}{2}\right) \Delta x, ...\right).
  $$

These formulas hold with the default origin at $0$.
When the mesh has another origin point, it is added to the corner and to the center.

## Constraints on the grid representation

The start and the end of each interval (and so of each cell) are integers, and the real coordinates are rebuilt from them with $\Delta x$.
Therefore, a cell boundary can only be at certain real coordinates, and not every real coordinate can be represented.

Consider level 1 in 1D, where $\Delta x = 0.5$.
No cell at this level can have its center at $\frac{1}{3}$.

The other constraint is that a cell at the level $l$ is included in a cell at any lower level:

```{image} ./figures/interval-3.png
:alt: Left, a cell at level l + 2 lies inside a cell at level l. Right, crossed out in red, a cell at level l + 2 that sticks out of the cell at level l.
:align: center
:width: 80%
```

:::{note}
This property is needed for mesh adaptation.
:::

## 1D mesh example

We now take a 1D mesh with several levels and look at how {{ project }} stores it:

```{image} ./figures/interval_example_1D.png
:alt: A 1D mesh with its cell numbering per level. Level 0 cells 0, 1 and 5 are blue, level 1 cells 4, 5, 6, 8 and 9 are green, level 2 cells 14 and 15 are red.
:align: center
:width: 80%
```

For each level, the intervals are:

- level 0: $[0, 2)$, $[5, 6)$
- level 1: $[4, 7)$, $[8, 10)$
- level 2: $[14, 16)$

The level and $\Delta x$ from {eq}`dx` give the real intervals:

- level 0: $[0, 2]$, $[5, 6]$
- level 1: $[2, 3.5]$, $[4, 5]$
- level 2: $[3.5, 4]$

:::{note}
There are no overlapping regions in this example, to keep it readable.
The following tutorials show meshes with overlaps.
Overlaps often appear when the mesh is adapted and ghost cells are needed to update the solution with a stencil operator (such as the fluxes of a finite volume scheme).
:::

The following program builds this mesh:

```{literalinclude} snippet/interval.cpp
:language: c++
```

It prints:

```text
┌────────────────────┐
│      Level 0       │
└────────────────────┘
     dim 0
            cells = 0->[0,2)@0:1 1->[5,6)@-3:1


┌────────────────────┐
│      Level 1       │
└────────────────────┘
     dim 0
            cells = 0->[4,7)@-1:1 1->[8,10)@-2:1


┌────────────────────┐
│      Level 2       │
└────────────────────┘
     dim 0
            cells = 0->[14,16)@-6:1
```

Each interval is printed as `position->[start,end)@index:step`, where `position` is the place of the interval in its array.
The [2D example](#read-the-printed-mesh) explains how the index values are computed.

This example uses two data structures, {cpp:class}`samurai::CellList` and {cpp:class}`samurai::CellArray`.
Both store one entry per level, from 0 to a maximum level given as a template parameter.
The default maximum level is `default_config::max_level` (20), so the default arrays have 21 entries.

{cpp:class}`samurai::CellList` is made to add new intervals while the mesh is built.
For each level, it stores lists of intervals in the x-direction.
In 2D and 3D, these lists sit in a nested `std::map` whose keys are the indices in the other directions (y, z).

:::{note}
The [2D example](#2d-mesh-example) shows how the keys are built.
In 1D there is no other direction, so the key is empty.
This is why we write `{}` when we fill the {cpp:class}`samurai::CellList`, as in `cl[0][{}].add_interval({0, 2});`.
:::

Adding elements to a {cpp:class}`samurai::CellList` is efficient: search, removal and insertion in a `std::map` have logarithmic complexity.
Numerical schemes, on the other hand, need to loop over the cells without searching for them.
We also want algebraic operations on sets of intervals, which needs a compact representation by intervals in each direction.

{cpp:class}`samurai::CellArray` provides this compressed representation: each direction has its own list of intervals, stored in an array.

In our 1D example, the {cpp:class}`samurai::CellList` of this mesh is:

```text
level 0:
    x: [0, 2), [5, 6)

level 1:
    x: [4, 7), [8, 10)

level 2:
    x: [14, 16)
```

The {cpp:class}`samurai::CellArray` has the same intervals.
The only difference is that `x` is a `std::forward_list` of {cpp:class}`samurai::Interval` in {cpp:class}`samurai::CellList` and a `std::vector` of {cpp:class}`samurai::Interval` in {cpp:class}`samurai::CellArray`.

A 2D example shows the difference between the two structures better.

## 2D mesh example

Consider the following mesh:

```{image} ./figures/2D_mesh.png
:alt: A 2D mesh of three levels with the cell numbering of each level on the left and bottom axes. Blue level 0 cells cover most of the domain, green level 1 cells form a block in the middle and a corner block at the top right, red level 2 cells sit at x and y from 8 to 10 and from 14 to 16.
:align: center
:width: 60%
```

The {cpp:class}`samurai::CellList` of this mesh is:

```text
level 0:
    y: 0
        x: [0, 4)
    y: 1
        x: [0, 1), [3, 4)
    y: 2
        x: [0, 1), [3, 4)
    y: 3
        x: [0, 3)

level 1:
    y: 2
        x: [2, 6)
    y: 3
        x: [2, 6)
    y: 4
        x: [2, 4), [5, 6)
    y: 5
        x: [2, 6)
    y: 6
        x: [6, 8)
    y: 7
        x: [6, 7)

level 2:
    y: 8
        x: [8, 10)
    y: 9
        x: [8, 10)
    y: 14
        x: [14, 16)
    y: 15
        x: [14, 16)
```

The keys of the map in {cpp:class}`samurai::CellList` are the indices in y, and the value of each key is the list of intervals in the x-direction for this index.

The {cpp:class}`samurai::CellArray` of the same mesh is:

```text
level 0:
    x: [0, 4), [0, 1), [3, 4), [0, 1), [3, 4), [0, 3)
    y: [0, 4)@0
    y-offset: [0, 1, 3, 5, 6]

level 1:
    x: [2, 6), [2, 6), [2, 4), [5, 6), [2, 6), [6, 8), [6, 7)
    y: [2, 8)@-2
    y-offset: [0, 1, 2, 4, 5, 6, 7]

level 2:
    x: [8, 10), [8, 10), [14, 16), [14, 16)
    y: [8, 10)@-8, [14, 16)@-12
    y-offset: [0, 1, 2, 3, 4]
```

How do we build this {cpp:class}`samurai::CellArray` from the {cpp:class}`samurai::CellList`?
We follow level 2.

First, we concatenate the intervals in the x-direction of each index `y`, in increasing order of `y`.
At level 2, the x array is:

```text
x: [8, 10), [8, 10), [14, 16), [14, 16)
```

Then, we build intervals in the y-direction from the keys.
At level 2, the keys are `y = 8, 9, 14, 15`, which give two intervals of consecutive values: $[8, 10)$ and $[14, 16)$.

The compressed view of the {cpp:class}`samurai::CellList` at level 2 is:

```text
x: [8, 10), [8, 10), [14, 16), [14, 16)
y: [8, 10), [14, 16)
```

### Read the printed mesh

We still have to connect each `y` value to its intervals in the x-direction.
For this, we use an array called `y-offset` and the index of each `y` interval, written after the `@` sign.

The `y-offset` array tells, for each `y`, where its intervals in the x-direction are in the x array.
At level 2, each `y` has one interval in the x-direction:

- `y = 8` has one interval in the x-direction,
- `y = 9` has one interval in the x-direction,
- `y = 14` has one interval in the x-direction,
- `y = 15` has one interval in the x-direction.

So the `y-offset` is the array `[0, 1, 2, 3, 4]`.
Its size is the number of `y` values plus one: the intervals in the x-direction of the i-th `y` value are the entries from `y-offset[i]` to `y-offset[i+1]` (excluded) of the x array.

One question remains: for a given `y`, which entry of `y-offset` do we read?
This is the role of the index.
For `y = 14`, the entry is `y-offset[2]`, because `y = 14` is the third `y` value.
The index of a `y` interval is chosen so that `y + index` is the position of `y` in `y-offset`.
For the interval $[14, 16)$, the index is `-12`, and `y-offset[y + index] = y-offset[14 - 12] = y-offset[2]`.

The same computation of the `y-offset` and of the indices at levels 0 and 1 gives the whole {cpp:class}`samurai::CellArray` shown above.

The intervals in the x-direction also carry an index.
It gives the position of each cell in the storage of a field defined on the mesh: the cell `x` of an interval is stored at `x + index`.
The cells are numbered one after the other, level by level.
At level 0, the first interval $[0, 4)$ holds cells 0 to 3, so its index is 0.
The next interval $[0, 1)$ holds cell 4, so its index is $4 - 0 = 4$, and $[3, 4)$ holds cell 5, so its index is $5 - 3 = 2$.

:::{note}
The compression of a {cpp:class}`samurai::CellList` into a {cpp:class}`samurai::CellArray` is recursive over the directions.
The same algorithm works in 3D and in higher dimensions.
:::

### Run the 2D example

The `tutorial-2d-mesh` demo builds this mesh.
It starts {{ project }} with `samurai::initialize`, adds the intervals level by level, prints the {cpp:class}`samurai::CellArray`, and saves the mesh:

```{literalinclude} ../../../demos/tutorial/2D_mesh.cpp
:language: c++
:lines: 4-
```

Build and run it from the build directory of your source build of {{ project }}:

```bash
cmake --build build --target tutorial-2d-mesh
./build/demos/tutorial/tutorial-2d-mesh
```

It prints the {cpp:class}`samurai::CellArray`:

```{literalinclude} snippet/2d_mesh_representation_output.txt
:language: text
```

The `cells` lines of `dim 0` are the x arrays, the `cells` lines of `dim 1` are the y intervals, and the `offsets` lines are the `y-offset` arrays, printed as `(position: value)` pairs.
The program also writes the mesh to `2d_mesh_construction.h5` and `2d_mesh_construction.xdmf` in the current directory.
Use the `--path` and `--filename` options to change them.

## Build a grid from a box

So far, we have built each {cpp:class}`samurai::CellArray` from a {cpp:class}`samurai::CellList`.
A {cpp:class}`samurai::CellArray` can also hold, at a given level, a uniform Cartesian grid defined by a box:

- the box can be 1D, 2D, or 3D,
- its corners can be given as integers (cell indices) or as real coordinates.

The following example uses a box in real coordinates:

```{literalinclude} snippet/2d_mesh_box.cpp
:language: c++
```

The box is defined by its "minimal" (lower-left in 2D) and "maximal" (upper-right in 2D) corners.
In this example, the box is $[-1, 1] \times [-1, 1]$.
The origin of the mesh is the minimal corner of the box, $(-1, -1)$, so the integer coordinates start at 0.
Since no scaling factor is given, {{ project }} computes one from the side lengths of the box.
Here, both sides have length 2 and the scaling factor is $s = 2$.
At level 3, the cell length is $\Delta x = 2 / 2^3 = 0.25$, so the box holds $8 \times 8$ cells:

```text
┌────────────────────┐
│      Level 3       │
└────────────────────┘
     dim 0
            cells = 0->[0,8)@0:1 1->[0,8)@8:1 2->[0,8)@16:1 3->[0,8)@24:1 4->[0,8)@32:1 5->[0,8)@40:1 6->[0,8)@48:1 7->[0,8)@56:1

     dim 1
            cells = 0->[0,8)@0:1

          offsets = (0: 0) (1: 1) (2: 2) (3: 3) (4: 4) (5: 5) (6: 6) (7: 7) (8: 8)
```

:::{warning}
The cell size at a given level is fixed and cell boundaries sit at integer coordinates.
For some boxes in real coordinates, no cell boundary falls exactly on the bounds of the box.
{{ project }} then extends the box so that each side is a multiple of the scaling factor, with an error of at most 5% of each side length by default.
:::

## What we built

We built a 1D and a 2D mesh with {cpp:class}`samurai::CellList`, compressed them into {cpp:class}`samurai::CellArray`, and read the intervals, indices and offsets that {{ project }} prints.
We also built a uniform mesh from a box.

## Next steps

- The {doc}`field tutorial <field>` attaches data to the cells of a mesh.
- The [CMake guide](../howto/cmake.md) sets up your own project that uses {{ project }}.
- The [box guide](../howto/box.md) builds domains from several boxes.
