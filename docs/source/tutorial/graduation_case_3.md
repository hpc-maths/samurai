# Graduation case 3: keep a mesh graded during refinement

In this tutorial, we start from a graded mesh and refine it with a criterion.
A boolean field `tag` marks the cells the criterion wants to refine, and we change it before the refinement so that the new mesh is graded too.
This is how a mesh adaptation keeps the graduation: in {{ project }}, {cpp:func}`samurai::graduation` does this work on the tags of AMR meshes.
The complete program is the demo {download}`graduation_case_3.cpp <../../../demos/tutorial/graduation_case_3.cpp>`.

This page is part of the {doc}`graduation series <graduation>`, which lists what you need before you start.

## Build and run the demo

From the root of the repository, build the demo target and run it:

```bash
cmake --build build --target tutorial-graduation-case-3
./build/demos/tutorial/tutorial-graduation-case-3
```

The demo prints one line per refinement pass, `Iteration for remove intersection: <n>`, and writes the final mesh in the current directory as `graduation_case_3.xdmf` and its `.h5` data file.

The demo accepts these options:

| Option | Default | Effect |
| --- | --- | --- |
| `--starting-level` | 1 | Level of the uniform initial mesh. |
| `--maximum-level` | 6 | Finest level of the refined mesh. |
| `--with-graduation` | on | Changes the tags so that the refined mesh is graded. Pass `--with-graduation=false` to refine without graduation. |
| `--path` | current directory | Directory of the output files. |
| `--filename` | `graduation_case_3` | Prefix of the output file names. |

`--help` lists them with the options that every {{ project }} program has.

## Create the initial mesh

We start with a uniform mesh at level `start_level` on the domain $[-2, 2] \times [-2, 2]$:

```{literalinclude} ../../../demos/tutorial/graduation_case_3.cpp
:language: c++
:start-at: auto generate_mesh
:end-before: int main(
```

At level $l$ the cell length is $2^{-l}$, so the box of cell indices goes from $-2 \cdot 2^{l}$ to $2 \cdot 2^{l}$ in each direction, which `2 << start_level` computes.
A mesh with a single level is graded.

## Tag the cells with a criterion

The criterion refines every cell that contains a point of the curve

$$
\begin{cases}
x = \sin(a t + \delta) \\
y = \sin(b t)
\end{cases}
\qquad \text{for} \quad t \in [0, 2 \pi],
$$ (eq-graduation-curve)

with $a = 3$, $b = 2$ and $\delta = \frac{\pi}{2}$.

We store the decision in a field named `tag`, with one boolean per cell, and we start with `false` everywhere.
Then we test the criterion on each cell:

```{literalinclude} ../../../demos/tutorial/graduation_case_3.cpp
:language: c++
:start-at: auto tag = samurai::make_scalar_field<bool>
:end-before: // graduation
:dedent:
```

{cpp:func}`samurai::for_each_cell` calls the lambda for each cell of the mesh.
`cell.corner()` gives the bottom left corner of the cell and `cell.length` its length.
We sample the curve {eq}`eq-graduation-curve` at $2^{m + 4}$ values of $t$, where $m$ is `max_level`, and we tag the cell as soon as one point falls strictly inside it.

## Build the refined mesh

From the tags, we build the new mesh with a {cpp:class}`samurai::CellList`:

```{literalinclude} ../../../demos/tutorial/graduation_case_3.cpp
:language: c++
:start-at: samurai::CellList<dim> cl;
:end-at: samurai::CellArray<dim> new_ca = {cl, true};
:dedent:
```

{cpp:func}`samurai::for_each_interval` gives the cells interval by interval.
`interval.start + interval.index` is the position of the first cell of the interval in the data array of `tag`, and `itag` follows the cells of the interval.
A tagged cell below `max_level` is replaced by its four children, and any other cell is kept.

Each pass refines a cell by one level at most.
The demo repeats the tagging and the refinement until the mesh no longer changes, so that the cells along the curve reach `max_level`.
The loop starts before the creation of `tag`:

```{literalinclude} ../../../demos/tutorial/graduation_case_3.cpp
:language: c++
:start-at: std::size_t ite = 0;
:end-at: << "\n";
:dedent:
```

The loop ends after the construction of `new_ca`, when the new mesh equals the old one:

```{literalinclude} ../../../demos/tutorial/graduation_case_3.cpp
:language: c++
:start-at: if (new_ca == ca)
:end-before: samurai::save(path, filename, ca);
:dedent:
```

Without graduation, the result is the mesh below, with `start_level = 1` and `max_level = 6`.
It is not graded: the finest cells along the curve touch cells several levels coarser.

```{image} ./figures/graduation_case_3_without_graduation.png
:alt: A uniform 8 by 8 mesh whose central part is refined along a looping curve, where the finest cells touch the large cells directly.
:width: 60%
:align: center
```

## Change the tags to keep the mesh graded

The mesh before refinement is graded.
When we refine a cell at level $l$, its children at level $l + 1$ must not touch cells at level $l - 1$.
So, if a cell at level $l$ is tagged, we also tag the cells at level $l - 1$ that touch it.

As in {doc}`graduation case 1 <graduation_case_1>`, we find these cells by translating the cells of level $l$ in each direction of a stencil and intersecting them with the cells of level $l - 1$.
We insert this step between the criterion and the construction of the new mesh:

```{literalinclude} ../../../demos/tutorial/graduation_case_3.cpp
:language: c++
:start-at: // graduation
:end-before: samurai::CellList<dim> cl;
:dedent:
```

The subset `intersection(translate(ca[level], s), ca[level - 1])` has no `.on()`, so it is computed on the finest level, `level`.
Its intervals hold fine cell indices `i_f` along x and `j_f` along y:

- `tag(level, i_f - s[0], j_f - s[1])` gives the tags of the fine cells that the translation moved onto these positions.
  We use them as a mask: only the coarse cells touched by a tagged fine cell must be tagged.
- `i_f >> 1` and `j_f >> 1` give the indices of the coarse cells at `level - 1` that contain these positions.
- `samurai::apply_on_masked(dst, mask, func)` calls `func` on each element of `dst` where `mask` is `true`. Here it sets the tag of the coarse cells to `true`.

We split the interval into its even elements and its odd elements because two fine cells share each coarse cell.
On a full interval, `i_f >> 1` would be about half as long as the mask, and the two would not line up.
With `even_elements()` or `odd_elements()`, the interval has a step of 2, so each fine cell maps to a different coarse cell and `i_f >> 1` has as many elements as the mask.
These functions return an invalid interval when no element is left, which `is_valid()` checks.

Tagging a cell at level $l - 1$ can force a cell at level $l - 2$ to be tagged, and so on.
So we go from the finest level down to the coarsest one: each level receives the tags of the level above before it passes its own tags to the level below.

The figure below shows the mesh refined with the graduation, with the same levels as above.
Layers of intermediate levels separate the finest cells from the coarse ones.

```{image} ./figures/graduation_case_3_with_graduation.png
:alt: The same refined mesh with graduation; rings of cells of decreasing size separate the fine cells along the curve from the large cells.
:width: 60%
:align: center
```

## What we built

We kept a mesh graded while a criterion refined it, by passing the tags of each level to the coarser cells that touch them before the refinement.
{cpp:func}`samurai::graduation` applies this idea to the tags of AMR meshes, and the AMR demos, such as `demos/FiniteVolume/AMR_Burgers_Hat.cpp`, call it at each mesh adaptation.

## Next steps

- {doc}`graduation` lists the graduation functions of the library.
- The {doc}`mesh how-to guide <../howto/mesh>` shows how to set the graduation width of a mesh.
