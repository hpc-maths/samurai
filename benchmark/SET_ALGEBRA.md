# Set algebra benchmarks

These benchmarks measure the cost of the set algebra (`include/samurai/subset/`)
and its weight in the operations samurai runs at every time step. They are the
baseline for work on the algebra: run them before and after a change, and
compare the medians.

## Running

```bash
micromamba activate samurai-env
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DBUILD_BENCHMARKS=ON
ninja -C build bench_set_operators bench_set_projection bench_set_mesh_shape \
               bench_set_call_sites bench_dynamic_set bench_set_graduation \
               bench_set_library_ops
./build/benchmark/bench_set_call_sites --benchmark_filter='<3>'
```

For numbers worth comparing, keep the machine idle, run with
`--benchmark_repetitions=5 --benchmark_report_aggregates_only=true` and read
the medians.

Every benchmark reports:

- `itv` and `cells`: what the sets emitted in one iteration.
- `t/itv`: the time per emitted interval, the figure to compare across shapes,
  operators, dimensions and backends.
- on an adapted mesh, the mesh itself: `leaves`, `leaf_itv` and `levels`.

### Inputs

`bench_set_call_sites`, `bench_dynamic_set` and `bench_set_graduation` read
their sets from files that `generate_set_algebra_inputs` writes to
`build/benchmark/set_algebra_inputs/` (13 MB, 3 seconds). The generator
adapts the meshes and writes their sets.

The build runs the generator before the first build of these binaries, and
again when one of the sources that decide the content of the files changes:
the generator, the meshes (`mr_fixtures.hpp`), and the list of meshes and the
file format (`site_inputs.hpp`). A change to the library does not trigger it,
so the inputs stay fixed while the set algebra changes.

- To regenerate the files by hand, delete the directory or run the generator.
- With `-DSAMURAI_BENCHMARK_GENERATE_SET_ALGEBRA_INPUTS=OFF`, the binaries
  build without the generator, which you then run by hand.
- With `SAMURAI_SET_ALGEBRA_INPUTS=<dir>` in the environment, the generator and
  the binaries use another directory, so two builds can share one set of
  inputs.

## What each binary measures

| Binary | Needs | Question |
| --- | --- | --- |
| `bench_set_operators` | algebra | Cost of the operators at one level: the floor of a traversal against a raw loop, the arity of intersection and union (with in-phase operands and nested binary operators as controls), and the expansion width. |
| `bench_set_projection` | algebra | Cost of `.on()`: coarsening against the level difference, compared with the same coarse set already built and with building it once; refinement as a control; coarsening of an N-ary intersection, of a chained projection and inside an expansion. |
| `bench_set_mesh_shape` | algebra | Effect of the shape of a level: the same cells cut into K stripes along each axis, and the same front cut into P subdomains along each axis. |
| `bench_set_call_sites` | algebra, input files | The expressions of the library functions that run at every time step or mesh construction, copied from their source (`BM_Site_*`); two variants that read coarse sets the mesh already stores (`BM_SiteCached_*`), for each registered backend. |
| `bench_dynamic_set` | algebra, input files | The same sites with the `static` and `dynamic` backends, on the main 2D and 3D meshes: the price of the runtime algebra. |
| `bench_set_graduation` | algebra, `make_graduation`, input files | Cost of `make_graduation` on the cells the adaptation passes to it, before graduation. It reports the cells before (`cells_in`) and after (`cells_out`). |
| `bench_set_library_ops` | mesh, MR adaptation, fields, schemes | The sets of `BM_Site_*` with the library's field operator (`BM_SiteOp_*`), and the library operations themselves (`BM_Op_*`), on the meshes and arguments of `bench_set_call_sites`. `BM_Op_AdaptStep` and `BM_Op_MeshConstruction` also report the time of their phases, read from samurai's timers, in ms per iteration. |

Most binaries need only samurai's interval structures (`LevelCellArray`,
`CellArray`, `ListOfIntervals`, `Box`) and the set algebra (the Needs
column), so they still build when the algebra is optimized or replaced. The
headers:

- `set_algebra_fixtures.hpp`: the traversal and its counters, and synthetic
  `LevelCellArray`s of controlled shape. A synthetic benchmark takes the level
  of its sets as first argument: 20 in 1D, 11 in 2D and 8 in 3D.
- `site_inputs.hpp`: `SiteInputs`, the sets the call sites read from an adapted
  mesh (cells, ghosts, reference, projection cells, union, domain and subdomain
  pyramids, corners and the cells of the next adaptation step before
  graduation), the list of meshes, and the input files. The file format is described at the top of the
  header, so that another implementation of the algebra can read the same
  inputs.
- `set_call_sites.hpp`: the call site expressions copied from the library,
  written against a backend ([Adding a backend](#adding-a-backend)).
- `set_algebra_backend.hpp`: the backend interface and samurai's static algebra
  as a backend (`static`); `set_algebra_dynamic_backend.hpp` holds the dynamic
  algebra (`dynamic`).
- `call_site_benchmarks.hpp`: the list of sites, their benchmarks,
  and their registration for one backend.
- `mr_fixtures.hpp`: the MR-adapted meshes. Only
  `generate_set_algebra_inputs` and `bench_set_library_ops` include it.

`BM_Site_X` (`bench_set_call_sites`) divided by `BM_SiteOp_X`
(`bench_set_library_ops`) is the share of the set algebra in the function that
evaluates X. The ratio sets a small translation unit against a large one, in
which GCC 15 may run out of inlining budget (`--param inline-unit-growth`) and
keep the construction of the traversers out of line: the same 3D expressions
then ran up to 1.5 times slower (measured on 2026-10-14). An application that
includes the mesh and the schemes is in the large case, so the ratio may
understate the share of the algebra there.

## Meshes

A benchmark on an adapted mesh takes three arguments: `max_level`, the number
of balls the mesh is refined around, and the max stencil radius.
`static/BM_Site_Tag<3>/7/8/1` runs the static backend on the 3D mesh of max
level 7 refined around 8 balls in the unit box, with a max stencil radius of 1.
`bench::mesh_cases` (`site_inputs.hpp`) lists the meshes:

| Dimension | Max level | Balls | Max stencil radius | Leaves |
| --- | ---: | ---: | ---: | ---: |
| 1D | 16 | 16 | 1 | 958 |
| 1D | 16 | 256 | 1 | 6,614 |
| 1D | 16 | 256 | 3 | 6,618 |
| 2D | 10 | 1 | 1 | 25,594 |
| 2D | 10 | 16 | 1 | 54,733 |
| 2D | 12 | 16 | 1 | 235,309 |
| 2D | 10 | 16 | 3 | 54,865 |
| 3D | 7 | 1 | 1 | 135,640 |
| 3D | 7 | 8 | 1 | 197,884 |
| 3D | 7 | 8 | 3 | 206,900 |

`mr_config` (`mr_fixtures.hpp`) builds every mesh with min level 2, the default
graduation width (1) and prediction stencil radius (1), and two settings that
change the sets:

- `max_stencil_radius(r)`: the widest stencil a scheme reads on the mesh, 1 for
  an upwind scheme, 3 for a WENO5 scheme.
- `disable_minimal_ghost_width()`: without it, the mesh raises the max stencil
  radius to at least 2 (`mesh_config::parse_args`), so a radius of 1 could not
  be measured.

The ghost width is the larger of the max stencil radius and the prediction
stencil radius, so 1 or 3 here. It sets the size of `cells_and_ghosts` and
`reference`, which nearly every call site reads. The max stencil radius also
decides the path of `make_graduation`: above 1, it extends the cells along the
boundary of the domain. The fronts move by 4 cells of the finest level between
two time steps, and the mesh is adapted with `epsilon = 1e-3` and a Dirichlet
boundary condition.

## Sites the dynamic backend leaves out

`bench_dynamic_set` leaves out the sites for which the dynamic algebra computes
other sets than the static one:

| Sites | Meshes | Cause |
| --- | --- | --- |
| `BM_Site_MeshPredictionGhosts`, `BM_SiteCached_MeshPredictionGhosts` | all | `dyn::expand` of a set projection returns other cells than the static expansion, and takes 290 s per iteration in 3D against 1 to 2 ms |

The `keep` function of `benchmark_dynamic_set.cpp` holds this list.

## Coverage

The library usages of the set algebra and the benchmark that measures each.
Each site was checked once against the library on the meshes of `mesh_cases`,
in 1D, 2D and 3D:

- the outer ghost site holds exactly the ghosts the library writes
  (`project_corner_below` writes a subset of its sets: it skips a coarse ghost
  whose corner-most child is not in the mesh);
- the interfaces have one cell per call of the callback of
  `for_each_interior_interface`.

| Library usage | Source | Benchmark |
| --- | --- | --- |
| Prediction ghosts of the mesh construction | `mr/mesh.hpp:334-382` | `BM_Site_MeshPredictionGhosts` |
| `construct_corners` | `mesh.hpp:937` | `BM_Site_Corners` |
| `make_graduation` | `algorithm/graduation.hpp` | `BM_Graduation` |
| `update_ghost_mr`: projection and prediction of the ghosts | `algorithm/update_ghost_mr.hpp` | `BM_Site_GhostProjection`, `BM_Site_GhostPrediction` |
| `update_outer_ghosts`: `project_bc`, `predict_bc`, `project_corner_below` | `algorithm/update_outer_ghost.hpp:23-61`, `197-230`, `270-338` | `BM_Site_OuterGhosts` |
| `Adapt::harten`: details and tags | `mr/adapt.hpp` | `BM_Site_Detail`, `BM_Site_Tag` |
| `update_fields` | `algorithm/update_fields.hpp` | `BM_Site_FieldTransfer` |
| `for_each_interior_interface` | `interface.hpp:34-361` | `BM_Site_Interfaces` |

Not covered yet: the rest of the mesh construction, the boundary conditions,
the sets built cell by cell, the branches the library takes for a domain that
is not a box and for a periodic mesh, and the MPI usages of the set algebra.

## Adding a backend

The call sites are written once, against a backend of the set algebra: a
struct of static functions over its own set expressions. The top of
`set_algebra_backend.hpp` lists the functions, and the concept
`bench::SetAlgebraBackend` checks them. `SamuraiStatic` (same file) and
`SamuraiDynamic` (`set_algebra_dynamic_backend.hpp`) are the two examples. To
compare a new set algebra with samurai's:

1. Write the backend struct: `name`, the prefix of its benchmarks;
   `lca_t<dim>`, its level cell array; and the operations `self`,
   `intersection`, `union_`, `difference`, `translate`, `expand` (one width,
   one width per axis, or one width along masked axes), `contract` (all axes
   or masked axes), `on`, `traverse`, `empty`, `materialize` and
   `builder<dim>(level)`. An operation the prototype lacks can wrap samurai's,
   as `SamuraiDynamic` does for the expansion per axis. Add
   `static_assert(bench::SetAlgebraBackend<MyBackend>);` after it, and keep its
   header free of the mesh stack.
2. The sites read their operands from `SiteInputs`, samurai `LevelCellArray`s
   read from the input files. A backend whose `lca_t<dim>` is
   `samurai::LevelCellArray<dim>` reads them as they are. Any other backend
   gets a copy in its own `lca_t`, made once outside the timed loop with its
   builder: `add_interval(interval, index)` for each interval, then `build()`.
   An operand is either an `lca_t` or a set expression of the backend;
   `Algebra<B>` passes each `lca_t` through `B::self`.
3. Register it in `benchmark_set_call_sites.cpp` with
   `register_backend<MyBackend>()`, next to `SamuraiStatic`, and include its
   header there (add a `target_include_directories` in `CMakeLists.txt` if it
   lives outside the repository). Its benchmarks are named
   `<name>/BM_Site_X<dim>/<args>`, so `--benchmark_filter='^<name>/'` runs only
   them.
4. Compare the medians of its benchmarks with those of `static/` from the same
   run.

## Baseline

Measured on 2026-10-18 at commit e1a735b8, each binary built from its own
translation unit. Sequential build, GCC 15.3.0 (conda-forge),
`-O3`, Release, Linux aarch64 in a Docker sandbox on an Apple silicon host (18
cores), medians of 5 repetitions (`--benchmark_repetitions=5`).

### Weight of the set algebra in samurai's operations

2D, max level 12, 16 balls (235k leaves), and 3D, max level 7, 8 balls (198k
leaves), both with a max stencil radius of 1. The total is that of
`BM_SiteOp_*` or `BM_Op_*` (`bench_set_library_ops`), the set algebra that of
`BM_Site_*` (`bench_set_call_sites`).

| Operation | 2D total | 2D algebra | 3D total | 3D algebra |
| --- | ---: | ---: | ---: | ---: |
| ghost projection (in `update_ghost_mr`) | 272 µs | 55 µs (20%) | 167 µs | 70 µs (42%) |
| ghost prediction (in `update_ghost_mr`) | 3.03 ms | 364 µs (12%) | 2.42 ms | 302 µs (12%) |
| outer ghosts (in `update_ghost_mr`, B.C. excluded) | | 791 µs | | 2.10 ms |
| `update_ghost_mr` | 4.74 ms | | 6.48 ms | |
| detail computation | 797 µs | 316 µs (40%) | 1.09 ms | 419 µs (38%) |
| tagging | 493 µs | 76 µs (15%) | 261 µs | 75 µs (29%) |
| field transfer (`update_fields`) | 1.17 ms | 207 µs (18%) | 918 µs | 199 µs (22%) |
| upwind flux (interfaces) | 3.68 ms | 693 µs (19%) | 4.42 ms | 437 µs (10%) |
| mesh construction | 4.21 ms | | 12.0 ms | |
| of which `construct_corners` | 232 µs | 162 µs | 5.72 ms | 4.19 ms |
| of which prediction ghosts + union (`update_sub_mesh`) | 3.29 ms | 625 µs | 5.73 ms | 2.16 ms |
| graduation (`BM_Graduation`) | 435 µs | all of it | 473 µs | all of it |
| one adaptation step | 23.0 ms | | 54.9 ms | |

In 3D, one adaptation step is 55 ms, of which 19.9 ms is ghost updates and
23.9 ms mesh construction. These two phases are where the set algebra weighs
the most. In 1D the algebra is below 10% of every operation: the field
operators dominate.

On the 3D mesh with a max stencil radius of 3, the call sites cost within 10%
of those at radius 1, but
`update_ghost_mr` takes 18.7 ms instead of 6.5 ms, one adaptation step 116 ms
instead of 55 ms, and graduation 3.69 ms instead of 0.47 ms: the closure adds
7,168 cells along the boundary of the domain, against 273.

### Bottlenecks, by impact

**1. Work on the domain repeated at every mesh construction and ghost update.**
`construct_corners` traverses the whole finest-level domain once per diagonal
direction (20 in 3D) to find about 1,000 intervals. It takes 5.7 ms, 48% of a
3D mesh construction (4.2 ms of it in the set expressions), for a set that
only changes with the domain. The outer ghosts of `update_ghost_mr` cost
2.10 ms in 3D (32% of the ghost update) and 791 µs in 2D (17%) to emit under
1,000 intervals. Part of that cost is `self(mesh.domain()).on(level)`
recomputed for each level and direction, part is the `LevelCellArray` that
`domain_boundary_outer_layer` builds on each call. Reading the domain from the
pyramid the mesh already stores (`BM_SiteCached_OuterGhosts`) brings 3D from
2.10 ms to 1.07 ms.

**2. Coarsening recomputed at every use.** `.on(l)` with `l` below the set's
level reads every fine row each time it is evaluated (5 to 8 ns per fine row
in 2D and 3D, whatever the level difference). That is the price of building
the coarse set once (`BM_CoarsenBuild` matches `BM_Coarsen`), but traversing a
coarse set already built is 15 times faster at one level of difference in 3D
(158 µs against 10 µs) and 500 times faster at four. Composition multiplies the
cost: `expand(self(a).on(l-1), 1)` re-coarsens for every neighbouring row (3D:
998 µs against 254 µs on a prebuilt coarse set), `.on(l-1).on(l)` coarsens each
coarse row again for each of its fine rows (3D: 649 µs, four times
`.on(l-1)`), and an N-ary intersection read one level down re-evaluates the
whole intersection. In the library, the mesh construction projects the
subdomain with `.on()` although the mesh stores it per level: reading
`mesh.subdomain(level)` instead (`BM_SiteCached_MeshPredictionGhosts`) takes
the 3D prediction ghosts from 2.16 ms to 0.79 ms.

**3. Arity of the N-ary operators.** With operands that are translates of the
same set (the stencil pattern), the time per operand interval grows with the
arity N: in 3D, 1.8 ns at N = 2 and 11.5 ns at N = 17 for the intersection, 3.4
and 10.3 ns for the union; in 1D the intersection of 17 operands is 330 times
slower than that of 2 for the same output. With N copies of the same set the
time per operand interval stays at about 2 ns: the growth comes from operands
whose intervals are out of phase, where each pass of the traverser advances a
single operand and costs O(N). Writing the intersection as nested binary
operators is not a reliable fix: it wins in 2D at N = 17 (613 µs against
1.43 ms) and loses in 3D (7.63 ms against 6.81 ms).

**4. Expansion in 3D.** `expand(a, w)` rebuilds, for every row, the list of
intervals of its (2w + 1)^2 neighbouring rows: 1.2 ms at w = 1 and 28 ms at
w = 8 on a front of 37k intervals, against 40 µs to traverse the front. In 2D
the cost is linear in w. The mesh construction expands coarsened sets
(`nestedExpand(self(cells[l]).on(l-2), r)`), which combines this with item 2.

**5. Linear row lookup.** To start a row, a traverser scans the intervals of
the parent row with `std::find_if`. At constant cells and output, cutting a 2D
box into 256 stripes along y makes a plain traversal 15 times slower; 64
stripes along y in 3D make it 3.7 times slower, a coarsening 14 times and an
expansion 18 times slower. Fragmentation along x only adds output intervals
and costs nothing extra. The same effect shows when a level is split into P
subdomains: P slabs along x multiply the work by about P (2D, P = 64: 249 µs
against 4.2 µs), slabs along the outermost axis cost nothing. This matters for
MPI partitions and for meshes with many small fronts.

**6. Fixed cost of a traversal.** `self(lca)` costs 1.9 ns per interval on a
box against 0.26 ns for a loop over the `LevelCellArray` (7.5 times) in 2D, 2.6
times in 3D. It is the floor below which no expression can go. In 1D it costs
the same as the loop (732 ns against 729 ns on the front).

**7. Dynamic algebra.** On the call sites (`bench_dynamic_set`, measured
later on the same day with the sites shared by both backends), the runtime
version is 5 to 21 times slower than the static one (3D ghost projection:
873 µs against 68 µs; 3D interfaces: 8.09 ms against 436 µs; 3D corners:
86.5 ms against 4.10 ms). Its expansion (`dyn::expand`) returns other cells
than the static one and takes 290 s per iteration on the 3D prediction ghosts
of the mesh construction, so `bench_dynamic_set` leaves those sites out
([Sites the dynamic backend leaves out](#sites-the-dynamic-backend-leaves-out)).

### Where the set algebra is not the bottleneck

The prediction of the ghosts (12% algebra), the upwind flux (10 to 19%) and 1D
in general are dominated by the field operators. Improving the algebra will
not speed them up much.

Refinement is not a bottleneck either: `self(a).on(l)` with `l` above the
set's level costs no more than traversing the refined set already built
(`BM_Refine` against `BM_RefinePrebuilt`, 3D at one level of difference: 39 µs
against 43 µs).

## Limits

- Sequential runs only: `BM_Partitioned` only reproduces the cost of the
  partition shape, and the MPI usages of the set algebra are not measured.
- `BM_Site_OuterGhosts` does not hold the boundary condition.
- `BM_Graduation` cannot be split into sites: the MR graduation is a pipeline
  of materialized sets, so its whole time is set algebra and the building of
  `LevelCellArray`s. Its input is the first iteration of an adaptation step,
  where the closure adds a few hundred cells in 2D and 3D; the later
  iterations of the step are not measured.
- Two runs of the same binary differ by up to 10 to 30% on the smallest
  benchmarks; compare medians of repeated runs.
- Two builds of the same expressions can differ by 30% on a small site:
  `BM_Site_Corners<2>` takes 28 to 45 µs depending on the alignment of its
  loop alone, and the 2D interfaces depend on which calls fit in GCC's
  inlining budget. Before you attribute a gap to a backend, check that it
  holds with `-falign-functions=64 -falign-loops=64` and
  `--param inline-unit-growth=2000`.
