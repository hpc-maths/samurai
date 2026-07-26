# Prediction, portions and reconstruction

An adapted samurai mesh stores each region of the domain at the coarsest level
that resolves it. Many operations nonetheless need the value a field *would*
take on a finer grid than the one it is stored on: writing a uniform field for
I/O and post-processing, moving a field from one adapted mesh to another, or
evaluating a scheme whose stencil crosses a level jump (the LBM stream, a finite
volume flux at a refinement interface).

samurai recovers those values by *prediction*: a cell at level $l$ splits into
$2^{\mathrm{dim}}$ children at level $l+1$, and the value of an absent finer cell
is interpolated from its coarser neighbours with an interpolating wavelet. This
page describes the three layers built on that idea:

- {ref}`prediction <rec-prediction>` - the interpolation stencil itself, as a
  sparse linear combination of coarse-cell values;
- {ref}`portion <rec-portion>` - applying such a stencil to a field to
  reconstruct the child(ren) of a coarse cell;
- {ref}`reconstruction <rec-reconstruction>` - projecting a whole adapted field
  onto a uniform fine mesh, either *flat* or *exact* across refinement
  boundaries.

Everything lives in {file}`include/samurai/reconstruction.hpp`.

(rec-prediction)=
## Prediction: the interpolating wavelet

Prediction is **linear**: the predicted value of a finer cell is a fixed linear
combination of coarse-cell values that depends only on the level gap and on the
child position inside its coarse cell, never on the field itself. samurai
represents such a combination with a `prediction_map`: a sparse map

$$
\texttt{coeff} : k \longmapsto w_k , \qquad
\text{standing for} \quad \sum_k w_k \, f(\text{reference} + k),
$$

where $k$ is a `dim`-dimensional integer offset relative to a reference cell.
Because the maps are linear they form an algebra (`+=`, `-=`, scalar `*=`), which
is what lets several children be summed into a single stencil later on.

The one-level stencil comes from the wavelet interpolation coefficients
`interp_coeffs<2*r+1>` of {file}`include/samurai/numeric/prediction.hpp`, where
`r` is the mesh's `prediction_stencil_radius` (default `1`, a 3-point stencil).
For `r = 1` the coefficients are $\{\,s/8,\ 1,\ -s/8\,\}$ with a sign $s = +1$ for
an even child and $s = -1$ for an odd one, so in 1D:

$$
f(l+1,\ 2i)   = f(i) + \tfrac{1}{8}\bigl(f(i-1) - f(i+1)\bigr), \qquad
f(l+1,\ 2i+1) = f(i) - \tfrac{1}{8}\bigl(f(i-1) - f(i+1)\bigr).
$$

```{figure} ./figures/reconstruction_prediction.svg
:name: fig-rec-prediction
:width: 560px
:align: center

One-level prediction in 1D. The left child (`ii=0`) of the coarse cell `i` is
interpolated from `i` (weight `1`) and its two neighbours (weights `±1/8`). The
two children average back to `f(i)`, so prediction is conservative.
```

The function

```cpp
template <std::size_t order = 1, class... index_t>
auto& prediction(std::size_t level, index_t... indices);
```

builds the `prediction_map` of the child `indices`, sitting `level`
levels below a reference coarse cell placed at the origin (`level` is the level
*gap* $\Delta l$, not an absolute level; `level = 0` is the identity map). A gap
of $\Delta l \geq 2$ is obtained by composing the one-level wavelet $\Delta l$
times: the child's parent is `indices >> 1` one level up, the parity
`indices & 1` picks the interpolation sign per direction, and the non-central
neighbours each contribute their own $\Delta l - 1$ stencil. Every intermediate
map is memoised in a static cache keyed by `(order, level, indices)`, so a stencil
is built once and reused for the whole run.

```{note}
Child indices are not restricted to $[0, 2^{\Delta l})$. Outside that box the
stencil simply reaches into the neighbouring coarse cells, which is exactly how
the LBM stream expresses a shift by crossing coarse-cell boundaries.
```

(rec-portion)=
## `portion`: reconstructing the children of a coarse cell

`portion` applies a prediction stencil to a field. It answers: *what is the value
of the child(ren) `ii` of the coarse cell(s) `i`, for a field `f` stored `delta_l`
levels coarser than those children?*

```cpp
auto portion(const Field& f,
             std::size_t level, std::size_t delta_l,
             const std::tuple</*interval*/, index_t...>& i,   // coarse location
             const std::tuple<cell_index_t...>&          ii); // child selector
```

The first entry of `i` is an **interval**, so the whole row of coarse cells is
reconstructed at once (vectorised over `x`); the remaining entries are the
transverse coarse indices. The result is the flat linear combination of level-`l0`
cells lifted to the target level through the wavelet, i.e. the prediction cone of
the figure below.

```{figure} ./figures/reconstruction_cone.svg
:name: fig-rec-cone
:width: 620px
:align: center

Reconstructing a fine cell `t` at level `L = l0 + 2`. Composing the one-level
wavelet `delta_l` times, `t` becomes a fixed linear combination of the level-`l0`
cells its prediction cone reaches. `portion` evaluates that combination directly.
```

The child selector `ii` has two forms, which differ in cost:

- **scalar** (`ii` are plain integers) - the stencil of that single child,
  `prediction(delta_l, ii)`. A nonlinear consumer, such as a finite volume flux
  that must apply its flux function separately to each fine cell, uses this form
  and recomputes every child.
- **slice** (`ii` are intervals) - a whole box of children at once. Because
  reconstruction is linear, the box's stencil is the *sum* of its children's
  stencils; that sum is built once by `accumulate_slice`, cached, and then
  applied in a single pass over a small, gap-independent stencil. This is what
  makes the LBM stream cheap (`LBMScheme::portion_column`): a shift over a whole
  column costs one stencil, not one per fine cell.

```{note}
`transfer` (moving a field between two adapted meshes) uses a convenience overload
of `portion` taking plain index arrays for a single coarse cell and one of its
children.
```

(rec-reconstruction)=
## `reconstruction`: projecting onto a uniform fine mesh

```cpp
template <class Field>
auto reconstruction(Field& field, bool exact = args::exact_reconstruction);
```

`reconstruction` creates a new field on a **uniform** grid at the domain's finest
level and fills every one of its fine cells from the adapted `field`. (It first
runs `update_ghost_mr` so that all ghost and projection cells hold correct values,
and it needs at least two boundary ghosts.)

Each fine cell of the uniform grid sits inside exactly one stored cell of the
adapted mesh. The question is only: *how do we get its value?* There are two
answers.

### Flat reconstruction (the default)

Every fine cell is filled with the prediction recipe of the previous sections,
always starting from the level of the coarse cell that contains it. Concretely,
for each level `l`, on the part of the uniform grid covered by `cells[l]`,
`reconstruction_op_` writes all $2^{\Delta l\cdot\mathrm{dim}}$ children of each
coarse cell at once (the fine cells of a row are addressed with a strided interval,
so a given sub-position is broadcast over the whole row). This is exactly `portion`
applied over the entire mesh: simple, fast, and almost always what you want.

### Exact reconstruction near a refinement boundary

Prediction is, by definition, a **guess**. Usually the guess is excellent. But
look at the border between a coarse region and a finely resolved one: there, the
mesh **already stores** the real, detailed finer values, right next to the coarse
cell we are reconstructing from. A guess made only from the coarse side cannot
know that detail, so the flat result is slightly wrong at the interface.

The idea of *exact* reconstruction is simply to stop guessing whenever real data
is available:

- **flat** always guesses, level by level down to the base level `l0`, even where
  finer real cells exist just next door;
- **exact** climbs the levels one at a time and, whenever it lands on a cell that
  is *actually stored in the mesh*, uses that real value instead of guessing. It
  only guesses where nothing real exists.

```{figure} ./figures/reconstruction_exact.svg
:name: fig-rec-exact
:width: 660px
:align: center

Reconstructing a fine child `t` of a coarse donor cell (the stored `l0` cell,
3rd from the left) whose right-hand neighbour is refined, drawn like the
prediction cone above. `t` is a weighted average of **3** cells at level `l0+1`,
and the rightmost of them lies just across the refinement boundary. **Left
(flat):** that boundary cell is guessed from `l0` (dashed), so the real finer cell
sitting there is ignored. **Right (exact):** it is a real cell actually stored in
the mesh, so its value is used as-is (orange) and carried up into `t`.
```

Where the neighbourhood is entirely coarse, no real finer cell is ever met and
exact reconstruction gives **exactly the same result** as flat - the two differ
only near refinement boundaries. Exact reconstruction is turned on with the
`--exact-reconstruction` command-line option, or by calling
`reconstruction(field, /*exact=*/true)`.

#### In symbols: the capped cascade

Write $A(m, i)$ for the exact reconstructed value at level $m$, index $i$, when
reconstructing a cell whose own base level is $l_0$. The three bullet points above
are exactly:

$$
\begin{aligned}
A(l_0, i) &= f(l_0, i)
   && \text{start from the level-}l_0\text{ value (real cell or ghost),}\\
A(m,   i) &= f(m, i)
   && \text{if the cell }(m,i)\text{ is really stored (use it, don't guess),}\\
A(m,   i) &= \sum_k c_k(i)\, A\bigl(m-1,\ \lfloor i/2\rfloor + k\bigr)
   && \text{otherwise (guess from one level down).}
\end{aligned}
$$

Here $c_k(i) = \texttt{prediction<r>(1, i \& 1)}$ is exactly the one-level recipe of
{ref}`the first section <rec-prediction>` (offsets relative to the parent
$\lfloor i/2 \rfloor$). "Really stored" means the cell belongs to
$\texttt{cells} \cup \texttt{proj\_cells}$: either a genuine finer leaf, or a
projection cell carrying the exact average of its children.

The cascade is **capped at $l_0$**: only *finer* cells (levels $> l_0$) override,
so a cell is always reconstructed from its own level upward. This handles nested
refinement without double counting, and reduces to the flat result wherever the
neighbourhood is not refined.

```{important}
The cascade reads the field **only** at stored cells - the base level $l_0$ and the
overriding finer cells - and recurses everywhere else. It therefore never reads a
cell the adapted mesh does not hold, at any depth. A naive recursion on the *value*
would instead try to read guessed positions that are not stored.
```

#### For reference: the building blocks

`make_real_backed_lca(mesh)` builds, per level, the
$(\texttt{cells} \cup \texttt{proj\_cells})$ set that the cascade tests to decide
"is this cell really stored?". `update_ghost_mr` must have run first so that the
projection cells carry the exact child averages.

```cpp
template <class Mesh>
auto make_real_backed_lca(const Mesh& mesh)
{
    using lca_t     = typename Mesh::lca_type;
    using mesh_id_t = typename Mesh::mesh_id_t;

    std::vector<lca_t> stored(mesh.max_level() + 1);
    for (std::size_t level = mesh.min_level(); level <= mesh.max_level(); ++level)
    {
        stored[level] = lca_t(union_(mesh[mesh_id_t::cells][level], mesh[mesh_id_t::proj_cells][level]));
    }
    return stored;
}
```

The cascade is evaluated by:

- `reconstruct_exact(l0, delta_l, coord, read_at, stored, memo)` - the scalar,
  memoised recursion of $A(m, i)$ for a single value. The memo is shared across the
  sub-cells of one reconstruction, so the cost stays $O(\text{cone} \times
  \text{depth})$ rather than exponential.
- `reconstruct_exact_box(l0, L, lo_L, hi_L, read_at, stored, out)` - the
  vectorised version used by `reconstruction` and the LBM stream. It fills a whole
  fine box level by level over dense buffers, using the real value where a cell is
  stored and guessing otherwise. With the "use the stored value" step disabled it
  becomes the flat reconstruction, which is why both paths share one
  `reconstruct_box` core.

## Summary

| Layer | Entry point | Role |
| --- | --- | --- |
| Stencil | `prediction`, `prediction_map` | the interpolating-wavelet combination, memoised |
| Apply | `portion` | reconstruct the child(ren) of a coarse cell (scalar or summed slice) |
| Project | `reconstruction` | fill a uniform fine field from an adapted one |
| Exact | `reconstruct_exact` / `reconstruct_exact_box`, `make_real_backed_lca` | use the real finer cells instead of guessing across refinement boundaries |
```
