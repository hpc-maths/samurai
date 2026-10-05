# Load balancing

This page documents the load balancing module of {{ project }}, in the namespace `samurai::load_balancing`.
The driver `samurai::load_balancing::LoadBalancer` asks a strategy for the destination rank of every local cell, then moves the cells and the values of the given fields to their new rank.
The strategy decides where the cells go: a space-filling curve (`SFC<Morton>` or `SFC<Hilbert>`), a diffusion between neighbouring ranks (`Diffusion`), or a graph partitioner (`Metis`, `Scotch`).
A weight policy gives the cost of each cell, so that the strategies balance work rather than numbers of cells.
Each building block has its own header in `include/samurai/load_balancing/`; the headers are listed in each section.
For the design, the behavior of each strategy and a comparison of them, see the {doc}`load balancing reference <../reference/load_balancing>`.

The driver, the strategies, `imbalance` and `require_balance` exist only when {{ project }} is built with `WITH_MPI=ON`.
`Metis` also needs `SAMURAI_WITH_PARMETIS=ON`, and `Scotch` needs `SAMURAI_WITH_PTSCOTCH=ON`: including their header without the option is a compile error.
The weight policies, `LoadBalanceStats` and `local_load` are also available without MPI.

## Driver

`samurai/load_balancing/load_balancer.hpp` declares the driver.
`make_load_balancer<Strategy>(config, strategy)` builds it; both arguments are optional.
`required` tells, identically on every rank, whether the imbalance exceeds the threshold of the configuration.
`load_balance` partitions the mesh and migrates the cells and the fields; all fields must live on the same mesh, and after the call the caller's mesh object holds the new mesh.
`load_balance_with_stats` does the same work and returns the measurements described in {ref}`api-load-balancing-metrics`.
With a single MPI process, these three functions do nothing.

```{doxygenclass} samurai::load_balancing::LoadBalancer
:members:
:undoc-members:
```

```{doxygenfunction} samurai::load_balancing::make_load_balancer
```

`samurai/load_balancing/config.hpp` declares the configuration of the driver.
`LoadBalanceConfig` holds the driver's own settings only; each strategy takes its options in its own struct.
No code of the module reads `verbose`: setting it has no effect.

```{doxygenstruct} samurai::load_balancing::LoadBalanceConfig
:members:
```

## Strategies

A strategy is a class with two members:

- `partition(mesh, weight)` returns an `int` field on `mesh` holding the destination rank of every local cell;
- `name()` returns the name recorded in `LoadBalanceStats::strategy_name`.

A strategy never migrates cells and never modifies the mesh.
The concept `PartitionStrategy` states this contract, and `load_balance` requires it.
A strategy that can fail to shed the load it computed may also provide `double last_unmet_flux() const`, which the driver copies into `LoadBalanceStats::unmet_flux`.

```{doxygenconcept} samurai::load_balancing::PartitionStrategy
```

| Strategy | Header | `name()` |
| --- | --- | --- |
| `Void` | `strategies/void.hpp` | `void` |
| `SFC<Morton>`, `SFC<Hilbert>` | `strategies/sfc.hpp` | `sfc-morton-cell`, `sfc-hilbert-cell`, or `-interval` instead of `-cell` with interval atoms |
| `Diffusion` | `strategies/diffusion.hpp` | `diffusion` |
| `Metis` | `strategies/metis.hpp` | `metis`, or `metis-adaptive` when `MetisOptions::adaptive` is `true` |
| `Scotch` | `strategies/scotch.hpp` | `scotch` |

### Void

`Void` keeps every cell on its rank.
It measures the fixed cost of the driver and gives a reference run that load balancing must not change.

```{doxygenclass} samurai::load_balancing::Void
:members:
:undoc-members:
```

### Space-filling curves

`SFC<Curve>` sorts the cells along a space-filling curve and cuts the curve into one segment of equal weight per rank.
The curve is `Morton` (Z-order, in `sfc/morton.hpp`) or `Hilbert` (in `sfc/hilbert.hpp`); both derive from `SFCCurve`, which maps the integer coordinates of a cell to a 64-bit key.
The keys use 32 bits per coordinate in 2D and 21 bits in 3D: the coordinates of the cells, brought to the maximum level, must fit in these bits.

By default each cell is a partition atom.
`SFC(curve, true)` or `with_interval_atoms()` makes each x-interval an atom instead: the cut search runs on fewer keys, and an interval is never split between two ranks.

```{doxygenclass} samurai::load_balancing::SFC
:members:
:undoc-members:
```

```{doxygenclass} samurai::load_balancing::Morton
```

```{doxygenclass} samurai::load_balancing::Hilbert
```

```{doxygenclass} samurai::load_balancing::SFCCurve
:members:
:undoc-members:
```

### Diffusion

`Diffusion` computes load fluxes between neighbouring ranks with a discrete heat equation on the graph of the MPI neighbourhood, then gives layers of cells along each interface to the neighbour that must receive load.
It exchanges data with the MPI neighbours only; its only collectives are one `all_reduce` per iteration of the flux solver and one for the total load.

```{doxygenclass} samurai::load_balancing::Diffusion
:members:
:undoc-members:
```

```{doxygenstruct} samurai::load_balancing::DiffusionOptions
:members:
```

### Graph partitioners

`Metis` and `Scotch` build the distributed graph of the cells, whose edges join cells that share a face, and hand it to ParMETIS or PT-Scotch.
Both partition into as many parts as there are MPI processes.

```{doxygenclass} samurai::load_balancing::Metis
:members:
:undoc-members:
```

```{doxygenstruct} samurai::load_balancing::MetisOptions
:members:
```

```{doxygenclass} samurai::load_balancing::Scotch
:members:
:undoc-members:
```

```{doxygenstruct} samurai::load_balancing::ScotchOptions
:members:
```

## Weight policies

`samurai/load_balancing/weight.hpp` declares the weight policies, in the namespace `samurai::load_balancing::weight`.
A weight policy is any callable that takes a cell and returns its non-negative cost as a `double`; the functions below build the common ones.

```{doxygenfunction} samurai::load_balancing::weight::uniform
```

```{doxygenfunction} samurai::load_balancing::weight::per_level
```

```{doxygenfunction} samurai::load_balancing::weight::from_field
```

(api-load-balancing-metrics)=

## Metrics

`samurai/load_balancing/metrics.hpp` declares the measurements of the load and of the imbalance.
The global imbalance is the largest load of a rank divided by the average load, minus 1: it is 0 for a perfect balance.

```{doxygenstruct} samurai::load_balancing::LoadBalanceStats
:members:
```

```{doxygenfunction} samurai::load_balancing::local_load
```

```{doxygenfunction} samurai::load_balancing::imbalance
```

```{doxygenfunction} samurai::load_balancing::require_balance
```
