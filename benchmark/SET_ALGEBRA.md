# Set algebra benchmarks

These benchmarks measure the cost of the set algebra (`include/samurai/subset/`)
and its weight in the operations samurai runs at every time step. They are the
baseline for the work on the algebra: rerun them before and after a change.

## Running

```bash
micromamba activate samurai-env
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DBUILD_BENCHMARKS=ON
ninja -C build bench_set_operators bench_set_projection bench_set_mesh_shape \
               bench_set_call_sites bench_dynamic_set \
               bench_set_library_ops
./build/benchmark/bench_set_call_sites --benchmark_filter='<3>'
```

`generate_set_algebra_inputs` adapts the meshes and writes their sets to
`build/benchmark/set_algebra_inputs/` (8 MB, a few seconds). The build runs it
once, before the first build of `bench_set_call_sites` or `bench_dynamic_set`,
and not again when the library changes, so the inputs stay the same while the
set algebra changes. To regenerate them, delete the directory or run the
generator. With `-DSAMURAI_BENCHMARK_GENERATE_SET_ALGEBRA_INPUTS=OFF`, these two
binaries build without the mesh stack, and the generator is run by hand. With
`SAMURAI_SET_ALGEBRA_INPUTS=<dir>` in the environment, the generator and the
binaries that read the files use another directory, so two builds can share
one set of inputs.

For numbers worth comparing, use `--benchmark_repetitions=3
--benchmark_report_aggregates_only=true` and keep the machine idle.

Every benchmark reports `itv` and `cells` (what the set emitted per iteration)
and `t/itv`, the time per emitted interval. The benchmarks on adapted meshes
also report the mesh: `leaves`, `leaf_itv` and `levels`.

Most binaries need only samurai's interval structures (`LevelCellArray`,
`CellArray`, `ListOfIntervals`, `Box`) and the set algebra, so they still build
if the algebra is optimized or replaced. Their inputs come from three headers:

- `set_algebra_fixtures.hpp` holds the traversal and its counters, and
  synthetic `LevelCellArray`s with a controlled shape.
- `site_inputs.hpp` holds `SiteInputs`, the sets the library call sites read
  from an MR-adapted mesh (cells, ghosts, reference, projection cells, union,
  domain and subdomain pyramids, corners), and reads and writes the files. The
  file format is described at the top of the header, so another implementation
  of the algebra can run on the same inputs.
- `set_call_sites.hpp` holds the call site expressions copied from the library,
  which read their operands from `SiteInputs`.

`mr_fixtures.hpp` builds the MR-adapted meshes, refined around the surfaces of K
balls, and copies their sets into `SiteInputs`. Only
`generate_set_algebra_inputs` and `bench_set_library_ops` include it.

| Binary | Needs | Question |
| --- | --- | --- |
| `bench_set_operators` | algebra | Cost of the operators at one level: the floor of a traversal against a raw loop, the arity of intersection and union (with in-phase operands and nested binary operators as controls), and the expansion width. |
| `bench_set_projection` | algebra | Cost of `.on()`: coarsening against the level difference, compared with the same coarse set already built and with building it once; refinement as a control; coarsening of an N-ary intersection, of a chained projection and inside an expansion. |
| `bench_set_mesh_shape` | algebra | Effect of the shape of a level: the same cells cut into K stripes along each axis, and the same front cut into P subdomains along each axis. |
| `bench_set_call_sites` | algebra, input files | The expressions of the library functions that run at every time step, copied from their source (`BM_Site_*`), and two variants that read coarse sets the mesh already stores (`BM_SiteCached_*`). |
| `bench_dynamic_set` | algebra, input files | Static against dynamic (runtime) algebra on two library expressions. |
| `bench_set_library_ops` | mesh, MR adaptation, fields, schemes | The sets of `BM_Site_*` with the library's field operator (`BM_SiteOp_*`), and the library operations themselves (`BM_Op_*`), on the meshes and arguments of `bench_set_call_sites`. |

`BM_Site_X` (`bench_set_call_sites`) divided by `BM_SiteOp_X`
(`bench_set_library_ops`) is the share of the set algebra in the function that
evaluates X; item 8 below explains why this ratio comes out low.
`BM_Op_AdaptStep` and `BM_Op_MeshConstruction` also report the time of their
phases, read from samurai's timers, in ms per iteration.

## Baseline

Measured on 2026-10-12 at commit 450fe11d, sequential build, GCC 15.3
(conda-forge), Release, aarch64 Linux with 18 CPUs, medians of 3 repetitions.
Meshes: 1D max level 16 with 16 and 256 balls, 2D max level 10 and 12 (25k to
235k leaves), 3D max level 7 (136k and 198k leaves), min level 2.

The `BM_Site_*` and `bench_dynamic_set` figures in this section were measured
when they shared a translation unit with the library operations. Compiled on
their own, as they now are, they run faster: see item 8.

### Weight of the set algebra in samurai's operations

2D, max level 12, 16 balls (235k leaves), and 3D, max level 7, 8 balls (198k
leaves). The set algebra time is that of `BM_Site_*`.

| Operation | 2D total | 2D algebra | 3D total | 3D algebra |
| --- | ---: | ---: | ---: | ---: |
| ghost projection (in `update_ghost_mr`) | 266 µs | 79 µs (30%) | 172 µs | 152 µs (88%) |
| ghost prediction (in `update_ghost_mr`) | 2.74 ms | 359 µs (13%) | 2.47 ms | 350 µs (14%) |
| outer ghosts (in `update_ghost_mr`, B.C. excluded) | | 853 µs | | 2.56 ms |
| `update_ghost_mr` | 4.40 ms | | 6.46 ms | |
| detail computation | 789 µs | 382 µs (48%) | 1.25 ms | 610 µs (49%) |
| tagging | 494 µs | 107 µs (22%) | 257 µs | 155 µs (60%) |
| field transfer (`update_fields`) | 1.14 ms | 300 µs (26%) | 898 µs | 295 µs (33%) |
| upwind flux (interfaces) | 3.79 ms | 531 µs (14%) | 4.41 ms | 634 µs (14%) |
| mesh construction | 4.05 ms | | 12.1 ms | |
| of which `construct_corners` | 227 µs | 222 µs | 5.78 ms | 4.76 ms |
| of which prediction ghosts + union (`update_sub_mesh`) | | 813 µs | | 2.20 ms |
| graduation (`make_graduation`) | 463 µs | all of it | 589 µs | all of it |
| one adaptation step | 21.7 ms | | 55.3 ms | |

In 3D, one adaptation step is 55 ms, of which 19.6 ms is ghost updates and
24.1 ms mesh construction. These two phases are where the set algebra weighs
the most. In 1D the algebra is below 10% of every operation: the field
operators dominate.

### Bottlenecks, by impact

**1. Work on the domain repeated at every mesh construction and ghost update.**
`construct_corners` traverses the whole finest-level domain once per diagonal
direction (20 in 3D) to find about 1,000 intervals. It takes 5.8 ms, 48% of a
3D mesh construction (4.8 ms of it in the set expressions), for a set that
only changes with the domain. The outer ghosts of
`update_ghost_mr` cost 2.56 ms in 3D (40% of the ghost update) and 853 µs in 2D
(19%) to emit under 1,000 intervals (`t/itv` of 2 to 14 µs). Part of that cost
is `self(mesh.domain()).on(level)` recomputed for each level and direction,
part is the `LevelCellArray` that `domain_boundary_outer_layer` builds on each
call. Reading the domain from the pyramid the mesh already stores
(`BM_SiteCached_OuterGhosts`) brings 3D from 2.56 ms to 1.46 ms.

**2. Coarsening recomputed at every use.** `.on(l)` with `l` below the set's
level reads every fine row each time it is evaluated (about 5 ns per fine row
in 2D and 3D, whatever the level difference). That is the price of building
the coarse set once (`BM_CoarsenBuild` matches `BM_Coarsen`), but traversing a
coarse set already built is 15 times faster at one level of difference in 3D
(157 µs against 10 µs) and 550 times faster at four. Composition multiplies the
cost: `expand(self(a).on(l-1), 1)` re-coarsens for every neighbouring row (3D:
966 µs against 250 µs on a prebuilt coarse set), `.on(l-1).on(l)` coarsens each
coarse row again for each of its fine rows (3D: 651 µs, four times
`.on(l-1)`), and an N-ary intersection read one level down re-evaluates the
whole intersection. In the library, the mesh construction projects the
subdomain with `.on()` although the mesh stores it per level: reading
`mesh.subdomain(level)` instead (`BM_SiteCached_MeshPredictionGhosts`) takes
the 3D prediction ghosts from 2.05 ms to 0.86 ms.

**3. Arity of the N-ary operators.** With operands that are translates of the
same set (the stencil pattern), the time per operand interval grows with the
arity N: in 3D, 1.7 ns at N = 2 and 8.5 ns at N = 17 for the intersection, 3.1
and 12.6 ns for the union; in 1D the intersection of 17 operands is 277 times
slower than that of 2 for the same output. With N copies of the same set the
time per operand interval stays at about 2 ns: the growth comes from operands
whose intervals are out of phase, where each pass of the traverser advances a
single operand and costs O(N). Writing the intersection as nested binary
operators is not a reliable fix: it wins in 2D at N = 17 (636 µs against
1.24 ms) and loses in 3D (7.98 ms against 6.70 ms).

**4. Expansion in 3D.** `expand(a, w)` rebuilds, for every row, the list of
intervals of its (2w + 1)^2 neighbouring rows: 1.1 ms at w = 1 and 28 ms at
w = 8 on a front of 37k intervals, against 40 µs to traverse the front. In 2D
the cost is linear in w. The mesh construction expands coarsened sets
(`nestedExpand(self(cells[l]).on(l-2), r)`), which combines this with item 2.

**5. Linear row lookup.** To start a row, a traverser scans the intervals of
the parent row with `std::find_if`. At constant cells and output, cutting a 2D
box into 256 stripes along y makes a plain traversal 15 times slower; 64
stripes along y in 3D make it 3.7 times slower, a coarsening 14 times and an
expansion 34 times slower. Fragmentation along x only adds output intervals
and costs nothing extra. The same effect shows when a level is split into P
subdomains: P slabs along x multiply the work by about P (2D, P = 64: 253 µs
against 4.5 µs), slabs along the outermost axis cost nothing. This matters for
MPI partitions and for meshes with many small fronts.

**6. Fixed cost of a traversal.** `self(lca)` costs 1.9 ns per interval on a
box against 0.26 ns for a loop over the `LevelCellArray` (7.5 times) in 2D, 2.6
times in 3D. It is the floor below which no expression can go. In 1D it costs
the same as the loop (683 ns against 689 ns on the front, measured on
2026-10-13).

**7. Dynamic algebra.** The runtime version is 7 to 18 times slower than the
static one on the ghost projection and the interfaces (3D ghost projection:
966 µs against 131 µs; 3D interfaces: 2.85 ms against 158 µs, measured on
2026-10-13).

**8. Inlining lost in large translation units.** GCC inlines the functions
that build the traversers (`get_traverser_impl`) only while the translation
unit is small. Compiled with the mesh, the MR adaptation and the schemes, GCC
15 runs out of its unit growth budget (`--param inline-unit-growth`), keeps
them out of line, and the same expressions on the same sets run 1.1 to 1.5
times slower in 3D: `BM_Site_Interfaces<3>/7/8` takes 646 µs in a large unit
and 440 µs alone, `BM_Site_GhostProjection<3>/7/8` 109 µs and 71 µs, the static
ghost projection of `bench_dynamic_set` 132 µs and 72 µs. Raising the budget
gives the fast figures back. An application that includes the mesh and the
schemes is in the slow case, so `BM_Site_X / BM_SiteOp_X`, which compares a
small unit with a large one, understates the share of the algebra. Forcing the
inlining of the traverser construction should recover the difference in
applications (measured on 2026-10-14).

### Where the set algebra is not the bottleneck

The prediction of the ghosts (13 to 14% algebra), the upwind flux (14%) and 1D
in general are dominated by the field operators. Improving the algebra will
not speed them up much.

Refinement is not a bottleneck either: `self(a).on(l)` with `l` above the
set's level costs no more than traversing the refined set already built
(`BM_Refine` against `BM_RefinePrebuilt`, 3D at one level of difference: 39 µs
against 42 µs, measured on 2026-10-13).

## Limits

- Sequential runs only: the MPI exchanges and the neighbour subdomains are not
  measured; `BM_Partitioned` only reproduces the cost of the partition shape.
- The boundary condition itself (`apply_field_bc`) and the periodic ghosts are
  not in `BM_Site_OuterGhosts`.
- `BM_Op_Graduation` cannot be split: the MR graduation is a pipeline of
  materialized sets, so its whole time is set algebra and materialization.
- Two runs of the same binary differ by up to 10 to 30% on the smallest
  benchmarks; compare medians of repeated runs.
