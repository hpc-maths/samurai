# Set algebra benchmarks

These benchmarks measure the cost of the set algebra (`include/samurai/subset/`)
and its weight in the operations samurai runs at every time step. They are the
baseline for the work on the algebra: rerun them before and after a change.

## Running

```bash
micromamba activate samurai-env
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DBUILD_BENCHMARKS=ON
ninja -C build bench_set_operators bench_set_projection bench_set_mesh_shape \
               bench_set_call_sites bench_dynamic_set
./build/benchmark/bench_set_call_sites --benchmark_filter='<3>'
```

For numbers worth comparing, use `--benchmark_repetitions=3
--benchmark_report_aggregates_only=true` and keep the machine idle.

Every benchmark reports `itv` and `cells` (what the set emitted per iteration)
and `t/itv`, the time per emitted interval. Inputs come from
`set_algebra_fixtures.hpp`: synthetic `LevelCellArray`s with a controlled shape,
and MR-adapted meshes refined around the surfaces of K balls, built once per
binary.

| Binary | Question |
| --- | --- |
| `bench_set_operators` | Cost of each operator at one level: floor against a raw loop, arity of intersection / union / difference, nested against flat, expansion and contraction width, materialization. |
| `bench_set_projection` | Cost of `.on()`: coarsening and refinement against the level difference, compared with the same set already built; coarsening of composite expressions, chained projections, expansion of a projection, operands at different levels. |
| `bench_set_mesh_shape` | Effect of the shape of a level: the same cells cut into K stripes along each axis, and the same front cut into P subdomains along each axis. |
| `bench_set_call_sites` | Every expression the library evaluates, copied from its source (`BM_Site_*`), the same sets with the library's field operator (`BM_SiteOp_*`), the library operations themselves (`BM_Op_*`) and two variants that read coarse sets the mesh already caches (`BM_SiteCached_*`). |
| `bench_dynamic_set` | Static against dynamic (runtime) algebra on the same expressions. |

`BM_Site_X / BM_SiteOp_X` is the share of the set algebra in the function that
evaluates X. `BM_Op_AdaptStep` and `BM_Op_MeshConstruction` also report the
time of their phases, read from samurai's timers, in ms per iteration.

## Baseline

Measured on 2026-10-12 at commit 450fe11d, sequential build, GCC 15.3
(conda-forge), Release, aarch64 Linux with 18 CPUs, medians of 3 repetitions.
Meshes: 1D max level 16 with 16 and 256 balls, 2D max level 10 and 12 (25k to
235k leaves), 3D max level 7 (136k and 198k leaves), min level 2.

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
times in 3D. It is the floor below which no expression can go.

**7. Dynamic algebra.** The runtime version is 6 to 22 times slower than the
static one on the same expressions (3D ghost projection: 953 µs against 160 µs).

### Where the set algebra is not the bottleneck

The prediction of the ghosts (13 to 14% algebra), the upwind flux (14%) and 1D
in general are dominated by the field operators. Improving the algebra will
not speed them up much.

## Limits

- Sequential runs only: the MPI exchanges and the neighbour subdomains are not
  measured; `BM_Partitioned` only reproduces the cost of the partition shape.
- The boundary condition itself (`apply_field_bc`) and the periodic ghosts are
  not in `BM_Site_OuterGhosts`.
- `BM_Op_Graduation` cannot be split: the MR graduation is a pipeline of
  materialized sets, so its whole time is set algebra and materialization.
- Two runs of the same binary differ by up to 10 to 30% on the smallest
  benchmarks; compare medians of repeated runs.
