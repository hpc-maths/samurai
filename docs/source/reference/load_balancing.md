# Load balancing

With MPI, mesh adaptation makes the work drift between processes: the cells follow the solution, not the initial domain decomposition.
The load balancing module redistributes the cells, and the fields that live on them, between the subdomains.

The module lives in `include/samurai/load_balancing/`, in the namespace `samurai::load_balancing`.
The driver and the strategies are compiled only when {{ project }} is built with `WITH_MPI=ON`; the weight policies, the {ref}`metrics structure <reference-load-balancing-metrics>` and `local_load()` are also available without MPI.
The examples on this page use the alias `namespace lb = samurai::load_balancing;`.

The page has one section per kind of content:

- {ref}`reference-load-balancing-design` explains how the module is split and why;
- {ref}`reference-load-balancing-behavior` describes what the driver does at run time and how the multiresolution adaptation calls it;
- {ref}`reference-load-balancing-options` lists every option and its default;
- {ref}`reference-load-balancing-weights`, {ref}`reference-load-balancing-metrics` and {ref}`reference-load-balancing-strategies` describe the behavior of each building block;
- {ref}`reference-load-balancing-choosing` compares the strategies, with the measurements behind the comparison;
- {ref}`reference-load-balancing-tools` describes the demo, the benchmark and the tests.

(reference-load-balancing-design)=

## Design

This section describes the design of the module.

The module separates two concerns:

- a **strategy** decides where each cell goes.
  It implements `partition(mesh, weight)`, which returns an `int` field holding the destination rank of every local cell, and `name()`, which identifies it in the statistics.
  A strategy never migrates cells, never sends field values and never modifies the mesh.
  The concept `lb::PartitionStrategy` states this contract;
- the **driver**, `lb::LoadBalancer<Strategy>`, performs the migration.
  The cells and the values of every field given to it travel in the same point-to-point message, routed by the destination ranks towards any rank.
  Destinations are not restricted to the MPI neighbourhood, because graph partitioners and space-filling curves send cells anywhere.

Adding a strategy means adding one header with a class that implements `partition()` and `name()`.
A strategy that can fail to shed the load it computed may also expose `double last_unmet_flux() const`: the driver copies it into the statistics.

### Migration

One call to the driver runs these steps on every rank:

1. The strategy computes the destination rank of each local cell.
2. The local cells are sorted into the cells that stay and one cell list per destination rank.
3. One `all_to_all` sends to each rank the number of cells it will receive and the total number of cells this rank sends.
   The second number tells every rank whether any cell moves anywhere.
   If none does, the call returns and the mesh and fields are left untouched.
4. Each rank sends one message per destination with the cells and the values of every field, and receives one message from each rank that announced cells.
5. A new mesh is built from the kept and received cells.
   Its constructor discovers the new MPI neighbourhood.
6. Each field is rebuilt on the new mesh: kept values are copied, received values are inserted, and the data arrays are swapped into the caller's field objects.
   The caller's mesh object then holds the new mesh, so every field still refers to it.

### Partitions of any shape

Space-filling curves, diffusion and graph partitioners produce subdomains that are not stripes: they can touch by a corner only, or have staircase boundaries.
The MPI layer of {{ project }} supports such subdomains since the load balancing module was added in [#443](https://github.com/hpc-maths/samurai/pull/443).
The neighbourhood detection derives its reach from the ghost footprint of the mesh (`Mesh_base::ghost_physical_reach()`), so two ranks exchange ghosts whenever the ghosts of one reach the cells of the other, including when their subdomains touch by a corner only.
`tests/mpi/test_lb_ghosts.cpp` checks the ghost values after a rebalance on such decompositions.
When you add a strategy, compare its results with a sequential run using `python/compare.py`.

(reference-load-balancing-behavior)=

## Behavior of the driver

This section describes what the driver does at run time.

### Explicit rebalancing

```c++
#include <samurai/load_balancing/load_balancer.hpp>
#include <samurai/load_balancing/strategies/sfc.hpp>
#include <samurai/load_balancing/weight.hpp>

namespace lb = samurai::load_balancing;

lb::LoadBalanceConfig config{.imbalance_threshold = 0.05};
auto balancer = lb::make_load_balancer<lb::SFC<lb::Hilbert>>(config);

// inside the time loop
auto weight = lb::weight::uniform();
if (balancer.required(u.mesh(), weight))
{
    balancer.load_balance(weight, u, v, w); // every field that lives on the mesh
}
```

`lb::make_load_balancer<Strategy>(config, strategy)` builds a `LoadBalancer<Strategy>`; both arguments default to a value-initialized object.

| Member | Behavior |
| --- | --- |
| `required(mesh, weight)` | Returns `true` when the global imbalance exceeds `LoadBalanceConfig::imbalance_threshold`. Returns `false` with a single process. Collective: two `all_reduce`. |
| `load_balance(weight, field, fields...)` | Partitions with the strategy, migrates the cells and the given fields, and swaps the new mesh into the caller's mesh object. Computes no metric. |
| `load_balance_with_stats(weight, field, fields...)` | Same work, plus the measurements returned in a {ref}`LoadBalanceStats <reference-load-balancing-metrics>`. Adds two `imbalance()` evaluations and two `local_load()` traversals. |
| `concentrate_on(rank, field, fields...)` | Moves every cell and the given fields to `rank`, to start from a maximally skewed distribution. Meant for tests and benchmarks. |
| `config()`, `strategy()` | Give access to the configuration and to the strategy object. |

### Rules for callers

- `required()`, `load_balance()`, `load_balance_with_stats()` and `concentrate_on()` are collective: every rank must call them.
  `required()` returns the same value on every rank, because it is computed from globally reduced quantities.
- Every field that lives on the mesh must be passed to `load_balance()`.
  A field left out is not resized and holds meaningless values on the new mesh.
  All the fields passed must share the same mesh (checked by an assertion in Debug).
- The values in memory move with their cells: after the call, the fields hold the same solution on the new distribution.
  Do not fill them again from an initial condition in a production run.
  `tests/mpi/test_lb_flux.cpp` does it only because it needs inputs that are identical on every decomposition.
- With a single process, `load_balance()` and `concentrate_on()` do nothing.

With `--timers`, the driver reports the timers `load_balancing`, `partition`, `migration` and `load_balancing:rebuild`.
The graph partitioners add `lb:metis:build_graph`, `lb:metis:partition`, `lb:scotch:build_graph` and `lb:scotch:partition`.

### Rebalancing from the multiresolution adaptation

The adaptation built by `samurai::make_MRAdapt` (header `<samurai/mr/adapt.hpp>`) owns a `LoadBalancer<SFC<Hilbert>>` with the default configuration.
Load balancing is off by default: the command-line option `--load-balancing-at` is 0.
With `--load-balancing-at N`, each call of the adaptation ends with a `load_balance()` on a uniform weight every `N` calls, starting with the first one.
The fields migrated are those given to `make_MRAdapt` and those passed to the call.
When the mesh has `min_level == max_level`, the adaptation returns at once and does not rebalance.

```bash
# rebalance at adaptation calls 1, 11, 21, ...
mpiexec -n 4 ./my_mpi_demo --load-balancing-at 10
```

```c++
#include <samurai/mr/adapt.hpp>

auto MRadaptation = samurai::make_MRAdapt(u);
auto mra_config   = samurai::mra_config().epsilon(2e-4);

// inside the time loop
MRadaptation(mra_config);    // adaptation, then load balancing when --load-balancing-at > 0
samurai::update_ghost_mr(u); // refresh the ghosts before the scheme
```

### Combining an explicit balancer with the adaptation

An explicit `LoadBalancer` is needed to use another strategy or another weight policy, to collect `LoadBalanceStats`, or to rebalance on a trigger or a period that differs from the adaptation's.
`demos/mpi/load_balancing.cpp` rebalances before the adaptation, so that the chosen strategy acts on the current mesh.
`MRadaptation` then rebalances again with the Hilbert curve only if `--load-balancing-at` is set.

```c++
namespace lb = samurai::load_balancing;

auto balancer = lb::make_load_balancer<lb::Diffusion>();
auto weight   = lb::weight::per_level(
    [&](std::size_t l)
    {
        const auto min_level = static_cast<double>(mesh.min_level());
        return std::pow(2.0, static_cast<double>(l) - min_level);
    });

// inside the time loop
if (time_to_rebalance)
{
    auto stats = balancer.load_balance_with_stats(weight, u);
    // log or save stats
}
MRadaptation(mra_config);
samurai::update_ghost_mr(u);
// scheme
```

(reference-load-balancing-options)=

## Options

This section lists every option of the module and its default.

### Command-line option

Declared in `<samurai/arguments.hpp>` and read by `samurai::initialize`.

| Option | Default | Meaning |
| --- | --- | --- |
| `--load-balancing-at N` | 0 (off) | Number of calls of the multiresolution adaptation between two rebalances. |

### Driver configuration

`lb::LoadBalanceConfig`, in `config.hpp`.

| Member | Type | Default | Meaning |
| --- | --- | --- | --- |
| `imbalance_threshold` | `double` | 0.05 | `required()` returns `true` when the global imbalance exceeds this value. |
| `verbose` | `bool` | `false` | Meant to trace the driver's decisions on `std::clog`. No code of the module reads it: setting it has no effect. |

### Space-filling curve options

`lb::SFC<Curve>` takes its options in its constructor `SFC(Curve curve, bool by_interval = false)` or through `with_interval_atoms(bool v = true)`, which returns the strategy.

| Option | Default | Meaning |
| --- | --- | --- |
| `by_interval` | `false` | `false`: one partition atom per cell. `true`: one atom per x-interval, weighted by the whole interval and never split across ranks. |

### Diffusion options

`lb::DiffusionOptions`, in `strategies/diffusion.hpp`, passed as `lb::Diffusion{options}`.

| Member | Type | Default | Meaning |
| --- | --- | --- | --- |
| `flux_threshold` | `double` | 0.01 | Fluxes below `flux_threshold` times the global average load are set to 0, to avoid migrating a few cells back and forth. |
| `diffusion_iterations` | `int` | 50 | Maximum number of iterations of the flux solver. |
| `min_retained_load_fraction` | `double` | 0.1 | Fraction of its weighted load a process always keeps: the load it sheds in one call is capped at `(1 - min_retained_load_fraction)` times its load, so no subdomain becomes empty. |

```c++
#include <samurai/load_balancing/load_balancer.hpp>
#include <samurai/load_balancing/strategies/diffusion.hpp>

namespace lb = samurai::load_balancing;

lb::DiffusionOptions options{.flux_threshold = 0.001, .diffusion_iterations = 100};
auto balancer = lb::make_load_balancer(lb::LoadBalanceConfig{}, lb::Diffusion{options});
```

### ParMETIS options

`lb::MetisOptions`, in `strategies/metis.hpp`, passed as `lb::Metis{options}`.

| Member | Type | Default | Meaning |
| --- | --- | --- | --- |
| `adaptive` | `bool` | `false` | `false`: geometric k-way partitioning. `true`: adaptive repartitioning from the current partition. See {ref}`reference-load-balancing-parmetis`. |
| `imbalance_tolerance` | `double` | 1.05 | ParMETIS `ubvec`: the largest allowed ratio of a part's weight to the average. 1.05 allows 5 % imbalance. |

### PT-Scotch options

`lb::ScotchOptions`, in `strategies/scotch.hpp`, passed as `lb::Scotch{options}`.

| Member | Type | Default | Meaning |
| --- | --- | --- | --- |
| `imbalance_tolerance` | `double` | 0.05 | Balance parameter of the `SCOTCH_STRATBALANCE` strategy: 0.05 allows 5 % imbalance. |

The two graph partitioners express the same 5 % tolerance differently: ParMETIS as a ratio (1.05), PT-Scotch as a fraction (0.05).

### Build options

| CMake option | Default | Effect |
| --- | --- | --- |
| `WITH_MPI` | `OFF` | Builds the driver, the strategies and the MPI demos. Required by every option below. |
| `SAMURAI_WITH_PARMETIS` | `OFF` | Enables the `Metis` strategy. Including `strategies/metis.hpp` without it is a compile error. |
| `SAMURAI_WITH_PTSCOTCH` | `OFF` | Enables the `Scotch` strategy. Including `strategies/scotch.hpp` without it is a compile error. |
| `BUILD_BENCHMARKS` | `OFF` | Builds the benchmarks, among them `bench_load_balancing` when `WITH_MPI=ON`. |

With conda, ParMETIS and PT-Scotch are installed with `conda install parmetis ptscotch`.

```bash
cmake -S . -B build \
    -DWITH_MPI=ON \
    -DSAMURAI_WITH_PARMETIS=ON \
    -DSAMURAI_WITH_PTSCOTCH=ON \
    -DBUILD_TESTS=ON
```

(reference-load-balancing-weights)=

## Weight policies

This section describes the weight policies, in `weight.hpp` (namespace `lb::weight`).

A weight policy is any callable `double(const cell_t&)` that returns the non-negative cost of a cell.
Strategies balance the weighted load, not the number of cells.
A policy is evaluated on local cells only and never communicates.

| Policy | Cost of a cell | Typical use |
| --- | --- | --- |
| `weight::uniform()` | 1 | Schemes with the same cost per cell: balancing cells balances work. |
| `weight::per_level(f)` | `f(cell.level)` | Cost driven by the level, for example local time stepping, where a cell of level `l` is updated `2^(l - min_level)` times more often than a cell of level `min_level`. |
| `weight::from_field(w)` | `w[cell]` | Cost known by the application, for example a number of particles per cell. |

`from_field(w)` captures `w` by reference: `w` must outlive every use of the policy.
Pass `w` to `load_balance()` with the other fields so that its values follow their cells.

`per_level` and `from_field` check that the weight is non-negative with an assertion in Debug.
Keep the growth of a `per_level` law moderate: a law such as `1 << (l * l)` overflows from level 8 and lets the finest cells decide the whole balance.

(reference-load-balancing-metrics)=

## Metrics

This section describes the quality measurements, in `metrics.hpp`.

`load_balance_with_stats()` returns a `lb::LoadBalanceStats`:

| Member | Type | Meaning |
| --- | --- | --- |
| `cells_before`, `cells_after` | `std::size_t` | Local number of cells before and after. |
| `cells_migrated_out`, `cells_migrated_in` | `std::size_t` | Local number of cells sent to and received from other ranks. |
| `load_before`, `load_after` | `double` | Local weighted load before and after. |
| `imbalance_before`, `imbalance_after` | `double` | Global imbalance before and after, identical on every rank. |
| `unmet_flux` | `double` | Load the strategy wanted to shed but could not. Set by `Diffusion`; 0 for the other strategies. |
| `strategy_name` | `std::string` | Value of the strategy's `name()`. |

The global imbalance is

$$
\frac{\max_p L_p}{\frac{1}{P}\sum_p L_p} - 1,
$$

where {math}`L_p` is the weighted load of rank {math}`p` and {math}`P` the number of ranks.
It is 0 for a perfect balance and {math}`P - 1` when one rank holds the whole load.

The same computations are available as free functions:

| Function | Result | Communication |
| --- | --- | --- |
| `local_load(mesh, weight)` | Sum of `weight(cell)` over the local cells. | None. Also available without MPI. |
| `imbalance(mesh, weight)` | Global imbalance. | Two `all_reduce` on the world communicator. |
| `require_balance(mesh, weight, threshold)` | `imbalance(mesh, weight) > threshold`, identical on every rank. | Same as `imbalance()`. |

(reference-load-balancing-strategies)=

## Strategies

This section describes the behavior of each strategy.

| Strategy | Header in `strategies/` | `name()` | Requires |
| --- | --- | --- | --- |
| `lb::Void` | `void.hpp` | `void` | MPI |
| `lb::SFC<lb::Morton>`, `lb::SFC<lb::Hilbert>` | `sfc.hpp` | `sfc-morton-cell`, `sfc-hilbert-cell` | MPI |
| `lb::Diffusion` | `diffusion.hpp` | `diffusion` | MPI |
| `lb::Metis` | `metis.hpp` | `metis`, `metis-adaptive` | MPI, ParMETIS |
| `lb::Scotch` | `scotch.hpp` | `scotch` | MPI, PT-Scotch |

With interval atoms, the SFC names end in `-interval` instead of `-cell`.
The {ref}`build options <reference-load-balancing-options>` enable MPI, ParMETIS and PT-Scotch.

### Void

`Void` keeps every cell on its rank.
It is the baseline: with it the driver measures its fixed cost (one `all_to_all`, then nothing moves), and it gives the reference run of the comparison tests.

### Space-filling curves

`SFC<Curve>` orders the atoms (cells, or x-intervals with interval atoms) along a space-filling curve and cuts the curve into {math}`P` segments of equal weighted load:

1. The key of each atom is computed from its position at the configured `max_level` of the mesh, after a shift that makes every coordinate non-negative.
   The shift and the extent come from the domain of the mesh, or from two `all_reduce` when the mesh has no domain.
2. The local atoms are sorted by key.
3. The {math}`P - 1` cut keys are found by a binary search over the key space.
   Each bisection step sums a local prefix weight across ranks with one `all_reduce` of {math}`P - 1` doubles, for at most 64 steps.
   With interval atoms and at most 4,000,000 atoms in total, one `all_gatherv` of every key and weight replaces the search and gives the same cuts.
4. An atom goes to the rank whose segment holds its key.

{ref}`plate-hilbert-cut` follows these steps on a small graded mesh, with a uniform weight and 4 processes.

```{plate} A Hilbert curve cut into four ranks
:figure: hilbert_cut
:label: plate-hilbert-cut

**Fig. 1.** A graded mesh with 79 cells, levels 2 to 4, and the Hilbert curve in red, which visits the cells in the order of their keys, from the dot to the arrow head.
Keys are computed at `max_level`: the key of a cell is the key of its lower-left cell of level 4, on the curve through the 16 x 16 cells of level 4.
In each cell of level 2, the dashed square is that corner cell and the number next to it is the key.

**Fig. 2.** The partition that `SFC<Hilbert>` computes.
Rank 0 is hatched, rank 1 dotted, rank 2 hatched the other way and rank 3 cross-hatched; heavy rules separate the ranks.

**Fig. 3.** The cells in curve order, one slot per cell, with the fill of their rank.
The total load is 79, so each segment targets 79 / 4 = 19.75.
Cut key *r* is the smallest key such that the cells with a lower key carry a load of at least *r* × 19.75.
Under each cut stand the load of the cells before it and the bound it reaches.
The ranks hold 20, 20, 20 and 19 cells.

*A run of `SFC<Hilbert>` on 4 processes gives the same keys and the same ranks on this mesh.*
```

No cell is gathered in cell mode.
The communication is one `all_reduce` for the total weight, one for the largest key, and the cut search.

The cuts depend only on the global cumulative weight, not on which rank holds an atom, so the result is the same from any initial decomposition.
The balance is reached in one call, up to the weight of the heaviest atom.
Keys are unique, so a second call produces the same partition and migrates nothing.
Interval atoms are 10 to 20 times fewer than cells on the advection demo (about 11 in 2D, 19 in 3D, according to the source comments); an interval is never split, which limits the balance once the heaviest interval approaches the average load per rank.

Two curves are provided in `load_balancing/sfc/`:

| Curve | Key computation | Locality |
| --- | --- | --- |
| `Morton` (Z-order) | Bit interleaving | Good: the curve jumps at power-of-two block boundaries. |
| `Hilbert` (Skilling's algorithm) | A pass over the bit planes, then bit interleaving | Best: on a uniform grid, consecutive keys are face-adjacent cells, so the subdomains are more compact and have fewer ghosts. |

In 2D, `Hilbert` lays a generalized Hilbert curve (Jakub Červený's "gilbert" algorithm) over the exact bounding box of the domain, so that a contiguous segment stays connected on a non-square domain.
In 3D it uses the cubic curve, whose segments can split into several pieces on a thin, non-cubic domain.
`Morton` ignores the bounding box.

Use `Hilbert` by default and `Morton` when the key computation shows up in a profile.

Keys are 64-bit integers, which bounds the coordinates at the configured `max_level`: 32 bits per direction in 2D and 21 bits in 3D.
On a domain one cell wide at level 0, `max_level` can therefore reach 21 in 3D.
A coordinate beyond this range triggers an assertion in Debug.

References: J. Skilling, *Programming the Hilbert curve*, AIP Conference Proceedings 707, 381 (2004).

### Diffusion

`Diffusion` rebalances incrementally, talking only to MPI neighbours, and needs no external library.
One call runs three phases.

1. **Fluxes.**
   The ranks and their MPI neighbourhoods form a graph, on which a discrete heat equation is solved.
   At each iteration, every rank exchanges its current load with its neighbours only and adds to the flux of each edge the generalized Cybenko term

   $$
   t_j = \frac{L_j - L_i}{\max(d_i, d_j) + 1},
   $$

   where {math}`L` is the load and {math}`d` the number of neighbours of a rank.
   The factor {math}`1/(\max(d_i, d_j) + 1)` keeps the iteration stable.
   The solver stops when every term on every rank is below {math}`10^{-3}` times the global average load, or after `diffusion_iterations` iterations.
   Fluxes below `flux_threshold` times the global average load are then dropped.
   The total load a rank sheds is capped by `min_retained_load_fraction`.
2. **Interface layers.**
   A negative flux towards a neighbour is load to give to it.
   The rank gives first the cells face-adjacent to the neighbour's cells, then the cells face-adjacent to what it just gave, and so on: a breadth-first front that only crosses cells the rank still owns.
   The given region stays attached to the neighbour, and the kept region stays connected.
   Adjacency is computed at `min_level` with set algebra and brought to every level with `.on(level)`, which works in any dimension and across level jumps.
   A coarse cell at `min_level` is given with all the finer cells it contains, and only when it fits the remaining flux, so a call never pushes the receiver past balance.
   The largest givers are served first.
3. **Connectivity repair.**
   Giving cells to several neighbours can still split the kept region into pockets, mostly in 3D.
   A flood fill at `min_level` labels the kept region, and every pocket but the largest goes to the neighbour that borders it most.

The diagram shows the first two phases of one call on two ranks, with a uniform weight.

```{diagram}
:figure: diffusion_layers

The start is deliberately unbalanced, 75 cells against 4, so that one call shows two layers and the rule that stops the peel.
Each rank has one neighbour, so the flux is half the load difference: (75 - 4) / 2 = 35.5 cells from rank 0 to rank 1.
Ink rules mark the cells of level 2, the `min_level` of the mesh.
Rank 0 gives whole cells of level 2, each with the finer cells inside it, from the bottom of each layer to its top: 4, 10, 7 and 1 cells in layer 1, then 10 in layer 2, as the numbers in the cells show.
The next cell of level 2, dashed in red, holds 7 cells, more than the 3.5 left, and stops the peel: 3.5 is reported as unmet flux.
```

Properties:

- Balance is reached over several calls: one call sheds at most the thickness of the subdomain towards a neighbour, and a coarse cell larger than the remaining flux stops the peel.
  Near balance on a long chain of ranks, the flux per edge can fall below one coarse cell and the partition stops short of perfect balance.
- Subdomain boundaries are staircases, in any dimension.
- When the interface runs out before the flux is met, the deficit is reported in `LoadBalanceStats::unmet_flux`.
- A rank without MPI neighbour still takes part in the collectives and gives nothing.
  From a state where some ranks hold no cell, for example after `concentrate_on()`, the ranks that hold the cells have no MPI neighbour, so `Diffusion` moves no cell at all.
  Use a global strategy to leave such a state.

Communication: point-to-point with the neighbours (neighbour meshes, degrees, loads), one boolean `all_reduce` per flux iteration and one scalar `all_reduce` for the average load.
There is no gather of the loads.

Reference: G. Cybenko, *Dynamic load balancing for distributed memory multiprocessors*, J. Parallel Distrib. Comput. 7 (1989) 279-301.

### Graph partitioners

`Metis` (ParMETIS) and `Scotch` (PT-Scotch) partition the graph of the cells and minimize the edge cut, which approximates the number of ghost values exchanged between ranks.
Do not call them while a rank holds no cell.
With ParMETIS 4.0.3, such a rank prints `PARMETIS ERROR adjncy is NULL` and the run hangs.

```c++
#include <samurai/load_balancing/load_balancer.hpp>
#include <samurai/load_balancing/strategies/metis.hpp>
#include <samurai/load_balancing/strategies/scotch.hpp>

namespace lb = samurai::load_balancing;

auto metis_balancer  = lb::make_load_balancer<lb::Metis>();
auto scotch_balancer = lb::make_load_balancer<lb::Scotch>();

lb::MetisOptions options{.adaptive = true};
auto adaptive_metis_balancer = lb::make_load_balancer(lb::LoadBalanceConfig{}, lb::Metis{options});
```

#### Cell graph

`build_cell_graph()`, in `graph.hpp`, builds the distributed graph used by both strategies:

- each local cell is a vertex.
  Its weight is the cell weight times 100 divided by the average weight of the rank's cells, rounded, and at least 1;
- each interface between two face-adjacent cells is an edge of weight 1, including interfaces across a level jump, across MPI boundaries and across periodic boundaries.
  A coarse cell facing several fine cells has one edge per fine cell;
- each vertex carries the coordinates of the cell center, for the geometric mode of ParMETIS.

```{diagram}
:figure: cell_graph_edges

The cell graph of a graded mesh with 28 cells of levels 2 and 3, cut into two ranks by the Hilbert curve: a vertex at the center of each cell, an edge between two cells that share a face.
At *a*, a cell of level 2 has one edge to each of the two cells of level 3 along its lower face, one of them across the ranks.
At *b*, a cell of level 2 and a cell of level 3 touch by a corner only and share no edge.
Of the 48 edges, 8 cross a level jump and 10 join the two ranks.
```

The cells of the MPI neighbours are numbered after an exchange of their meshes, and the edges across MPI boundaries are made symmetric by a point-to-point exchange, as PT-Scotch and ParMETIS require.
The vertex ranges are shared with one `all_gather`.

(reference-load-balancing-parmetis)=

#### ParMETIS

ParMETIS receives the vertex weights only (`wgtflag = 2`).

- **Geometric k-way** (`MetisOptions::adaptive = false`, the default) calls `ParMETIS_V3_PartGeomKway` with the cell centers.
  If it fails, for example on degenerate coordinates, `ParMETIS_V3_PartKway` is called instead.
- **Adaptive repartitioning** (`MetisOptions::adaptive = true`) calls `ParMETIS_V3_AdaptiveRepart` with the current partition as a starting point, `itr = 1000` and a redistribution size of 1000 per vertex.
  `itr` is the ratio of communication time to data redistribution time: a high value favors a low edge cut, a low value favors little data redistribution, and the [ParMETIS 4.x manual](https://github.com/KarypisLab/ParMETIS/blob/main/manual/manual.pdf) (section 5.2) recommends 1000 when the ratio is unknown.

A ParMETIS return code other than `METIS_OK` is thrown as `std::runtime_error` on the rank that receives it.

#### PT-Scotch

`Scotch` builds the distributed graph with `SCOTCH_dgraphBuild`, with vertex and edge weights, and calls `SCOTCH_dgraphPart` with as many parts as ranks and a `SCOTCH_STRATBALANCE` strategy whose balance parameter is `ScotchOptions::imbalance_tolerance`.
A PT-Scotch error is thrown as `std::runtime_error`.

(reference-load-balancing-choosing)=

## Choosing a strategy

This section compares the strategies.
The guidance follows from the design of each strategy; the figures come from the benchmark and the demo described in {ref}`reference-load-balancing-tools`.

| Situation | Strategy | Reason |
| --- | --- | --- |
| Default, no external library | `SFC<Hilbert>` | Balances in one call with a few `all_reduce`; compact subdomains. |
| Lowest ghost volume | `Metis`, with `adaptive = true` inside an adaptive run | Minimizes the edge cut; adaptive mode starts from the current partition to limit migration. Builds the cell graph and needs ParMETIS. |
| Incremental rebalancing, neighbour-only communication | `Diffusion` | Moves cells only towards MPI neighbours; balances over several calls. |
| Cheapest keys, locality not critical | `SFC<Morton>` | Same algorithm as Hilbert, simpler keys, less compact subdomains. |
| Baseline, no balancing | `Void` | Moves nothing. |

`Scotch` partitions from scratch at every call, without the current partition as a starting point.
Use it for a one-off partition rather than for rebalancing at every adaptation.

### Partitioning cost

`bench_load_balancing` measured the following cost of one `partition()` call on 4 processes, on a uniform periodic mesh:

| Strategy | `partition()` per call |
| --- | --- |
| `SFC<Morton>`, `SFC<Hilbert>` | about 1 ms |
| `Diffusion` | slightly above 1 ms |
| `Metis` | about 10 ms (graph construction and ParMETIS) |
| `Scotch` | 100 to 200 ms |

The migration and mesh rebuild that follow do not depend on the strategy that produced the destinations.

```{admonition} TODO: conditions of the partitioning figures
:class: warning

These figures were added with the module in [#443](https://github.com/hpc-maths/samurai/pull/443), which does not state how they were obtained.
The benchmark runs a 128 x 128 mesh in 2D (level 7) and a 16 x 16 x 16 mesh in 3D (level 4), 10 iterations each.
Maintainers: state which of the two meshes the figures come from, the hardware (CPU, number of cores, interconnect), the compiler and its version, the MPI library, the ParMETIS and PT-Scotch versions, and the {{ project }} commit or release used.
```

### Balance and migration on the demo

The comparison of balance and migrated cells in this section (`SFC<Hilbert>` and `Scotch` reach an imbalance close to 0, `Metis` has the lowest edge cut, `Diffusion` migrates the fewest cells per call) was measured on `demos/mpi/load_balancing.cpp` with 2 and 4 processes and a uniform weight.
With its defaults, the demo advects a disk of radius 0.2 on a periodic unit square, levels 4 to 10, `epsilon = 2e-4`, final time 0.1, and rebalances at time step 1 and every 10 time steps.

```{admonition} TODO: conditions of the demo comparison
:class: warning

[#443](https://github.com/hpc-maths/samurai/pull/443) does not record this comparison.
Maintainers: state the dimension, the command-line options, the hardware, the compiler and its version, and the {{ project }} commit or release used, or publish the CSV files written by `--lb-stats-file`.
```

(reference-load-balancing-tools)=

## Tools to compare strategies

This section describes the demo, the benchmark and the tests of the module.

### Demo

`demos/mpi/load_balancing.cpp` builds `mpi-load-balancing-2d` and `mpi-load-balancing-3d`; the dimension is fixed at compile time by `SAMURAI_LB_DIM`.
It advects a disk (2D) or a sphere (3D) on a periodic adaptive mesh with the strategy given on the command line, and gives the same solution whatever the strategy.

```bash
mpiexec -n 4 ./mpi-load-balancing-2d --lb-strategy sfc-hilbert \
    --lb-weight level --lb-stats-file stats.csv
mpiexec -n 4 ./mpi-load-balancing-3d --lb-strategy diffusion
```

| Option | Default | Meaning |
| --- | --- | --- |
| `--lb-strategy` | `sfc-hilbert` | `void`, `sfc-morton`, `sfc-hilbert`, `diffusion`, and `metis` or `scotch` when built with them. |
| `--lb-weight` | `uniform` | `uniform`, or `level` for `2^(l - min_level)`. |
| `--nt-loadbalance` | 10 | Rebalance (or check, with `--lb-threshold`) at time step 1, then at every multiple of N. |
| `--lb-threshold` | 0 | If above 0, rebalance only when `required()` is `true` with this threshold. |
| `--lb-stats-file` | none | Append one CSV line per rebalance (rank 0): time step, time, strategy, weight, ranks, imbalance before and after, migrated cells. |
| `--lb-skew` | off | Move every cell to rank 0 before the time loop. Not usable with `metis` and `scotch`; `diffusion` moves no cell from this state. |
| `--lb-sfc-interval` | off | Use interval atoms with the SFC strategies. |
| `--lb-diffusion-iterations`, `--lb-diffusion-flux-threshold`, `--lb-diffusion-min-retained` | `DiffusionOptions` defaults | Set the {ref}`diffusion options <reference-load-balancing-options>`. |

### Benchmark

`benchmark/benchmark_load_balancing.cpp` builds `bench_load_balancing` (`-DBUILD_BENCHMARKS=ON -DWITH_MPI=ON`).
For each strategy it times `partition()` alone, in 2D and 3D on uniform periodic meshes (128 x 128 and 16 x 16 x 16), 10 iterations each.
For `SFC` and `Diffusion` it also times a full `load_balance()` from a state where rank 0 holds every cell; `Metis` and `Scotch` are left out of this case because they cannot run with empty ranks.
`Diffusion` moves no cell from this state, so its full-pass time measures the cost of a pass that migrates nothing.

```bash
mpiexec -n 4 ./benchmark/bench_load_balancing
```

### Tests

- The MPI tests live in `tests/mpi` and run on 2, 3 and 4 processes (`ctest -R test_lb`).
  `check_lb_invariants` in `tests/mpi/mpi_test_utils.hpp` checks that no cell is lost or duplicated and that every field value follows its cell.
- `tests/test_load_balancing_weight.cpp` tests the weight policies without MPI.
- `tests/test_load_balancing.py` runs the demo for every available strategy, in 2D and 3D, on 2 and 4 processes.
  It checks that the final field matches the sequential `void` run within an absolute tolerance of {math}`10^{-12}` (with `python/compare.py`), and that the imbalance after a rebalance drops below 0.1 at least once.
  It is skipped when `mpiexec`, `python/compare.py` or the demo executables are missing.
  Set `SAMURAI_MPI_BUILD_DIR` to the MPI build directory when it is not `../build` or `../build-mpi` relative to `tests/`.
