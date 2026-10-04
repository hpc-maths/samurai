# How-to: adapt a mesh with multiresolution

This guide shows you how to make a multiresolution mesh follow your solution: fine cells where the solution varies, coarse cells where it is smooth.
You create the adaptation for a field with `samurai::make_MRAdapt`, set its threshold with `samurai::mra_config`, and call it each time the solution changes.

## Before you start

- You have a multiresolution mesh built with `samurai::mra::make_mesh`, and a field on it. If not, see the [mesh how-to guide](mesh.md) and the [field how-to guide](field.md).
- Include `samurai/mr/adapt.hpp`.
- Call `samurai::initialize` at the start of `main` and `samurai::finalize` at the end, so that the command-line options described below have an effect.

The example below builds a 2D mesh with levels 3 to 7, sets the field `u` to a Gaussian bump centered in the unit square, then adapts the mesh to `u`:

```{literalinclude} snippet/adapt/mr_adapt.cpp
  :language: c++
```

## Fill the field at the finest level

Give the field its values before the first adaptation.
`samurai::mra::make_mesh` puts every cell at the maximum level, and the adaptation needs the solution there to decide which cells it can coarsen.

Attach the boundary conditions of the field with `samurai::make_bc` before adapting too.
Near a non-periodic boundary, the adaptation reads the ghosts outside the domain, and the boundary conditions fill them.

## Create the adaptation

Pass the field that drives the adaptation to `samurai::make_MRAdapt`:

```{literalinclude} snippet/adapt/mr_adapt.cpp
  :language: c++
  :start-at: auto MRadaptation
  :end-at: auto MRadaptation
  :dedent:
```

Create it once, before the time loop.
It keeps a reference to the field and allocates its working fields (the details and the tags) on the same mesh.

To drive the adaptation with several fields, pass them all: `samurai::make_MRAdapt(u, v)`.
The fields must live on the same mesh.
A cell is refined if the detail of any component of any of these fields asks for it, and coarsened only if all of them allow it.

## Set the threshold

Describe the adaptation criterion with a `samurai::mra_config` object and chain the setters you need:

```{literalinclude} snippet/adapt/mr_adapt.cpp
  :language: c++
  :start-at: auto mra_config
  :end-at: mra_config.epsilon
  :dedent:
```

| Setter | Default | Effect |
| --- | --- | --- |
| `epsilon(double)` | `1e-4` | Threshold on the details. A smaller value keeps more fine cells. |
| `regularity(double)` | `1` | Assumed smoothness of the solution. A larger value raises the threshold for refining and gives a coarser mesh. |
| `relative_detail(bool)` | `false` | Divides the details of each component by the maximum of its absolute value over the cells, so that one threshold fits components of different magnitudes. |

The detail of a cell is the difference between its value and the value predicted from its parent.
With $d$ the dimension and $L$ the maximum level, the threshold at level $\ell$ is $\epsilon_\ell = \epsilon / 2^{d(L - \ell)}$.
Cells at level $\ell$ are merged into their parent when their details are at most $\epsilon_\ell$ and the detail of the parent is at most $2^{\text{regularity}} \, \epsilon_\ell$.
A cell is refined when its detail exceeds $2^{\text{regularity} + d} \, \epsilon_\ell$.

Start with the default `epsilon` and lower it if the solution is not resolved enough, or raise it if the mesh keeps too many cells.
With `relative_detail(true)`, `epsilon` becomes a relative tolerance.

## Adapt the mesh

Call the adaptation with the configuration:

```{literalinclude} snippet/adapt/mr_adapt.cpp
  :language: c++
  :start-after: // Adapt the mesh
  :end-before: std::cout << "After
  :dedent:
```

The call rebuilds the mesh and transfers the field onto the new cells.
It repeats the detail computation up to `max_level - min_level` times, and stops as soon as the mesh no longer changes.
If the minimum and maximum levels are equal, it returns without changing the mesh.

Pass after the configuration the other fields that must follow the mesh without driving the adaptation, such as `v` in the example.
They are transferred to the new mesh with the same prediction as the driving fields.
A field that you pass neither to `make_MRAdapt` nor to the call keeps its old size: call `resize()` on it before you use it again, as the demos do for the field that receives the next time step.

In a time loop, adapt the mesh at the start of each time step, before the scheme reads the solution.
The [getting started tutorial](../tutorial/getting_started.md) and the demo `demos/FiniteVolume/advection_2d.cpp` follow this pattern.

```{note}
The call `MRadaptation(epsilon, regularity)` with two numbers is deprecated.
Use a `samurai::mra_config` object instead.
```

## Override the threshold from the command line

When your program calls `samurai::initialize(argc, argv)`, these options replace the values of the `mra_config` at each call of the adaptation:

| Option | Replaces |
| --- | --- |
| `--mr-eps` | `epsilon` |
| `--mr-reg` | `regularity` |
| `--mr-rel-detail` | `relative_detail` (sets it to `true`) |

The configuration you pass is modified in place: after the first call, it holds the values from the command line.
`--mr-rel-detail` can only turn relative details on; when it is absent, the value set in the code stays.
For example, `./mr_adapt --mr-eps 1e-4` adapts with $\epsilon = 10^{-4}$ whatever `epsilon` says in the code.
The [options how-to guide](options.md) lists the other predefined options.

## Fill the ghosts after adapting

After an adaptation, and after any change to a field, the ghosts of the field are out of date.
Before a loop of your own that reads the neighbors of a cell, call `samurai::update_ghost_mr` on the fields it reads:

```{literalinclude} snippet/adapt/mr_adapt.cpp
  :language: c++
  :start-at: samurai::update_ghost_mr
  :end-at: samurai::update_ghost_mr
  :dedent:
```

It fills the ghosts at every level: from the boundary conditions outside the domain, from the periodic images and the neighboring MPI subdomains, by field projection on coarser levels, and by prediction on finer levels.
`samurai::update_ghost_mr(u, v)` updates several fields in one call.
The finite volume schemes (for example the one returned by `samurai::make_convection_upwind`) update the ghosts of their input field themselves when they are out of date, so you do not need to call it before them.

## Check the result

Count the cells of each level before and after the adaptation, with `mesh.nb_cells(level, mesh_id_t::cells)`.
The function `print_cells_per_level` in the example does this.
Run the program:

```bash
./mr_adapt
```

```text
Before adaptation:
  level 3: 0 cells
  level 4: 0 cells
  level 5: 0 cells
  level 6: 0 cells
  level 7: 16384 cells
  total: 16384 cells
After adaptation:
  level 3: 0 cells
  level 4: 160 cells
  level 5: 212 cells
  level 6: 476 cells
  level 7: 848 cells
  total: 1696 cells
```

Before the adaptation, all $128 \times 128$ cells are at level 7.
After it, the mesh has about a tenth of the cells: level 7 is kept around the bump, and the cells get coarser away from it.
With a smaller threshold, the mesh keeps more fine cells:

```bash
./mr_adapt --mr-eps 1e-4
```

```text
After adaptation:
  level 3: 0 cells
  level 4: 128 cells
  level 5: 288 cells
  level 6: 408 cells
  level 7: 1952 cells
  total: 2776 cells
```

If every cell stays at the maximum level, the threshold is too small for your solution.
To see where each level sits, save the mesh with the `--save-debug-fields` option and color it by `levels` in ParaView, as the [save how-to guide](save.md) explains.

## If you run with MPI

- The adaptation is collective: every rank must call it.
- With `--load-balancing-at N`, the adaptation also redistributes the cells between the ranks every `N` calls, starting with the first one. See the {doc}`load balancing reference <../reference/load_balancing>`.

## Related

- [How-to: create a samurai mesh](mesh.md), for the level range and the graduation of the mesh.
- [How-to: set options in samurai](options.md), for the command-line options.
- The [getting started tutorial](../tutorial/getting_started.md) and the [reaction-diffusion tutorial](../tutorial/reaction_diffusion.md), which adapt the mesh at each time step.
- The [graduation tutorial](../tutorial/graduation.md), for the constraint the adaptation enforces between neighboring levels.
