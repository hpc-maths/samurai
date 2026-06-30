Load balancing
==============

When running with MPI, the mesh adaptation (AMR/MRA) progressively unbalances
the work between processes: the cells follow the solution, not the initial
domain decomposition. The load balancing module redistributes the cells — and
the fields living on them — between processes.

The module lives in ``include/samurai/load_balancing/`` under the namespace
``samurai::load_balancing`` and is only active when samurai is built with
``WITH_MPI=ON``.

Design
------

The module separates two concerns:

* a **strategy** decides *where* each cell should go: it implements
  ``partition(mesh, weight)`` and returns an integer field ``flags`` holding
  the destination rank of every local cell. A strategy never migrates
  anything;
* the **driver** (``LoadBalancer<Strategy>``) performs the *fused migration*:
  the cells and the values of all the user's fields travel in the same
  point-to-point message, routed by ``flags`` towards arbitrary ranks (the
  destinations are not restricted to the geometric MPI neighbourhood, which
  is required by graph partitioners and space-filling curves).

Adding a strategy means adding one header implementing ``partition()`` and
``name()`` — nothing else.

Basic usage
-----------

.. code-block:: c++

    #include <samurai/load_balancing/load_balancer.hpp>
    #include <samurai/load_balancing/strategies/void.hpp>
    #include <samurai/load_balancing/weight.hpp>

    namespace lb = samurai::load_balancing;

    auto balancer = lb::make_load_balancer<lb::Void>(lb::LoadBalanceConfig{
        .imbalance_threshold = 0.05,
    });

    // inside the time loop (see "Integration with MR adaptation" below):
    auto weight = lb::weight::uniform();
    if (balancer.required(u.mesh(), weight))
    {
        balancer.load_balance(weight, u, v, w); // all fields!
    }

Two rules:

* **every field living on the mesh must be passed** to ``load_balance()``:
  a field that does not migrate with its cells holds garbage afterwards;
* ``required()`` and ``load_balance()`` are **collective**: every rank must
  call them (the decision returned by ``required()`` is guaranteed identical
  on all ranks).

Integration with MR adaptation
-----------------------------

The multiresolution adaptation helper ``samurai::make_MRAdapt`` (header
``<samurai/mr/adapt.hpp>``) is able to drive the load balancer itself: at the
end of an adaptation pass it calls ``load_balance()`` with a ``SFC<Hilbert>``
strategy and a uniform weight. **Load balancing is off by default**: the
counter ``args::load_balancing_at`` is ``0`` and the rebalance step is
skipped. To let the adaptation rebalance, pass the command-line option
``--load-balancing-at N`` (declared in ``<samurai/arguments.hpp>``), where
``N`` is the number of adaptation iterations between two rebalances.

.. code-block:: bash

    # rebalance every 10 adaptation passes, Hilbert SFC, uniform weight
    mpiexec -n 4 ./my_mpi_demo --load-balancing-at 10

.. code-block:: c++

    #include <samurai/mr/adapt.hpp>

    auto MRadaptation = samurai::make_MRAdapt(u);
    auto mra_config   = samurai::mra_config().epsilon(2e-4);

    // inside the time loop:
    MRadaptation(mra_config);      // refine/coarsen + graduation + ghost update
                                   //   (+ load balancing when --load-balancing-at > 0)
    samurai::update_ghost_mr(u);   // refresh ghosts for the next scheme call

Use an explicit ``LoadBalancer`` when you want to:

* select another strategy (``Diffusion``, ``Metis``, ``Scotch``, ...) or
  another weight policy (e.g. ``weight::per_level`` for local time stepping);
* collect ``LoadBalanceStats`` for diagnostics or comparison;
* rebalance on a trigger (``required()``) or on a period that does not match
  the adaptation period.

In that case the canonical order, taken from ``demos/mpi/load_balancing.cpp``,
is to rebalance **before** the adaptation so the chosen strategy acts on the
current mesh, then let ``MRadaptation`` adapt (and rebalance with the default
Hilbert curve, but only when ``--load-balancing-at`` is set on the command
line):

.. code-block:: c++

    namespace lb = samurai::load_balancing;
    auto balancer = lb::make_load_balancer<lb::Diffusion>(lb::LoadBalanceConfig{});

    // inside the time loop:
    if (time_to_rebalance)
    {
        auto weight = lb::weight::per_level(
            [&](std::size_t l) { return std::pow(2.0, l - mesh.min_level()); });
        auto stats  = balancer.load_balance_with_stats(weight, u);
        // ... log / save stats ...
    }
    MRadaptation(mra_config);
    samurai::update_ghost_mr(u);
    // ... scheme ...

Two rules to keep in mind:

* ``load_balance()`` preserves the field values that are in memory; the
  migrated values are the correct state and must not be overwritten in a
  production run. Re-filling the field from an analytic formula is only useful
  in regression tests that need a bit-for-bit identical input across
  decompositions (see ``tests/mpi/test_lb_flux.cpp``).
* **every** field that lives on the mesh must be passed to the explicit
  ``load_balance()`` call (a field that does not migrate with its cells holds
  garbage afterwards). ``make_MRAdapt`` takes care of this for the fields it
  knows about; an explicit rebalance has to list them all.

Weight policies
---------------

A weight policy is any callable ``double(const cell_t&)`` returning the
computational cost of a cell. Strategies balance the *weighted* load, not the
cell count. Three policies are provided in ``weight.hpp``:

.. list-table::
   :header-rows: 1

   * - Policy
     - Cost of a cell
     - Typical use
   * - ``weight::uniform()``
     - 1
     - homogeneous schemes: balancing cells == balancing work
   * - ``weight::per_level(f)``
     - ``f(cell.level)``
     - cost driven by refinement, e.g. local time stepping where a cell of
       level ``l`` is updated ``2^(l - min_level)`` times more often:
       ``per_level([&](auto l){ return std::pow(2.0, l - min_level); })``
   * - ``weight::from_field(w)``
     - ``w[cell]``
     - application cost (particles per cell, local operator cost). The field
       ``w`` is captured by reference, must be non-negative and must be passed
       to ``load_balance()`` like any other field so that it follows its cells.

Keep ``per_level`` laws moderate: an over-aggressive growth (e.g.
``1 << (l*l)``) overflows quickly and makes the finest cells dominate the
balance entirely.

Metrics
-------

Every call to ``load_balance_with_stats()`` returns a ``LoadBalanceStats``:

.. code-block:: c++

    struct LoadBalanceStats
    {
        std::size_t cells_before, cells_after;            // local cell counts
        std::size_t cells_migrated_out, cells_migrated_in;
        double load_before, load_after;                   // local weighted loads
        double imbalance_before, imbalance_after;         // global max/avg - 1
        double unmet_flux;                                 // load a strategy could not shed
        std::string strategy_name;
    };

The global imbalance ``max(load)/avg(load) - 1`` is 0 for a perfect balance
and ``P - 1`` when the whole load sits on one process. It is the quantity
compared across strategies by the benchmark (roadmap step 6). The free
functions ``local_load()``, ``imbalance()`` and ``require_balance()`` of
``metrics.hpp`` expose the same computations; the documentation of each one
states whether it communicates.

Available strategies
--------------------

.. list-table::
   :header-rows: 1

   * - Strategy
     - Header
     - Status
   * - ``Void`` (baseline: nothing moves)
     - ``strategies/void.hpp``
     - available
   * - ``SFC<Morton>`` / ``SFC<Hilbert>`` (weighted space-filling curves)
     - ``strategies/sfc.hpp``
     - available
   * - ParMETIS (``Metis``) / PT-Scotch (``Scotch``)
     - ``strategies/metis.hpp`` / ``strategies/scotch.hpp``
     - available (requires ``SAMURAI_WITH_PARMETIS`` / ``SAMURAI_WITH_PTSCOTCH``)
   * - ``Diffusion`` (heat-equation fluxes + interface layers)
     - ``strategies/diffusion.hpp``
     - available

Space-filling curves (SFC)
--------------------------

``SFC<Curve>`` orders the cells along a space-filling curve (keys computed at
the global max level) and cuts the curve into P segments of equal *weighted*
load. The cuts are found with a gather-free vectorised binary search over the
key space: each bisection step sums a local prefix weight across ranks with a
single ``all_reduce``. In total the strategy issues a few ``all_reduce`` for
the bounding box, the total weight and the key upper bound, then at most 64
``all_reduce`` of ``P-1`` doubles for the cut search. No cell or key is
gathered. The balance is reached in a single call, up to the weight of the
heaviest cell, and a second call migrates nothing (deterministic cuts).

Two curves are provided in ``load_balancing/sfc/``:

.. list-table::
   :header-rows: 1

   * - Curve
     - Key cost
     - Locality
   * - ``Morton`` (Z-order)
     - a few bit tricks
     - good — the curve jumps at power-of-two block boundaries
   * - ``Hilbert`` (Skilling's algorithm)
     - ~2 passes over the bit planes
     - best — the curve is continuous: consecutive cells on the curve are
       always face-adjacent, so the partitions are more compact (fewer ghosts)

Default to ``Hilbert``; use ``Morton`` if key computation ever shows up in
profiles (it rarely does: the partitioning cost is dominated by the sort).

.. note::

   Non-stripe partitions (which Hilbert cuts produce naturally) exposed
   three pre-existing decomposition bugs in samurai, all fixed since:
   the MPI neighbour detection now derives its expansion from the ghost
   footprint (``Mesh_base::ghost_physical_reach()``), the out-of-domain
   ghosts are owned layer by layer and filled after the inner exchanges
   (``update_ghost_mr``/``outer_subdomain_corner``), and the cross-rank
   graduation check uses the level bounds of the receiving mesh
   (``list_interval_to_refine_for_graduation``). The non-regression tests
   live in ``tests/mpi/test_lb_ghosts.cpp``; when adding a strategy, always
   validate against a sequential reference with ``python/compare.py``.

Limits: the 64-bit keys bound the usable refinement: 32 bits per coordinate
in 2D and 21 bits in 3D (i.e. max_level up to 21 in 3D on a unit domain).
Negative cell indices are handled by a global shift; coordinates beyond the
bit budget trigger an assertion in Debug.

Diffusion
---------

``Diffusion`` rebalances **incrementally and locally**, without any external
dependency, in two phases:

#. **Fluxes.** The processes form a graph (the MPI neighbourhood). A discrete
   heat equation is solved on that graph: at each iteration every process
   exchanges its current load **with its neighbours only** and updates a
   per-edge flux with the generalized Cybenko coefficient
   ``t_j = (load_j - load_i) / (max(deg_i, deg_j) + 1)``. The
   ``1/(max(deg)+1)`` factor guarantees stability (the legacy fixed ``0.5``
   could oscillate). The only collective is one boolean ``all_reduce`` per
   iteration for convergence (plus one scalar ``all_reduce`` once, for the
   convergence scale): **no ``all_gather`` of the loads**. Fluxes below
   ``flux_threshold * mean_load`` are dropped to avoid micro-migrations.

#. **Interface layers (nD).** A negative flux means "I must shed that load to
   this neighbour". The cells closest to the neighbour are ceded first, then
   progressively deeper layers, so the ceded region stays **connected to the
   interface (no islands)**. The cession direction is the dominant cardinal
   axis of ``barycentre_i - barycentre_j`` (a diagonal is split into its
   cardinal components). Layers are built by set algebra at ``min_level`` and
   projected onto every actual level with ``.on(level)``, which makes the
   construction dimension- and level-jump-agnostic.

Properties:

* converges to balance over **several calls** (each call sheds at most the
  available domain thickness towards a neighbour); AMR calls the balancer at
  every adaptation, so partial per-call convergence is expected and fine;
* partitions are compact with **staircase boundaries** — not straight lines.
  The straight-line (row-snapping) constraint was exactly what limited the
  previous implementation to 2D bands; this version is nD;
* less precise than SFC (a typical target is ``imbalance < 0.1`` after a few
  calls, looser in 3D), but migrates fewer cells per call and needs no global
  ordering.

If the interface is exhausted before the requested flux is met, the deficit is
reported in ``LoadBalanceStats::unmet_flux`` (no exception, no silent log) so
the phenomenon stays measurable.

Communication: neighbour-only point-to-point (neighbour meshes, loads,
degrees) + 1 boolean ``all_reduce`` per flux iteration + 1 scalar
``all_reduce`` for the scale. Reference: G. Cybenko, *Dynamic load balancing
for distributed memory multiprocessors*, J. Parallel Distrib. Comput. 7 (1989).

The tunables (``flux_threshold``, ``diffusion_iterations``,
``min_retained_load_fraction``) live in ``lb::DiffusionOptions`` — separate from
the driver's ``LoadBalanceConfig``, exactly like ``MetisOptions`` /
``ScotchOptions``. Pass an instance to the strategy:

.. code-block:: c++

    #include <samurai/load_balancing/load_balancer.hpp>
    #include <samurai/load_balancing/strategies/diffusion.hpp>

    namespace lb = samurai::load_balancing;

    lb::DiffusionOptions opts{.flux_threshold = 0.001, .diffusion_iterations = 100};
    auto balancer = lb::make_load_balancer(lb::LoadBalanceConfig{}, lb::Diffusion{opts});

Comparing strategies
--------------------

The demo ``demos/mpi/load_balancing.cpp`` builds two executables,
``mpi-load-balancing-2d`` and ``mpi-load-balancing-3d`` (the dimension is fixed
at compile time through ``SAMURAI_LB_DIM``). It advects a disk / sphere on a
periodic adaptive mesh with any strategy and collects the metrics, e.g.:

.. code-block:: bash

    mpiexec -n 4 ./mpi-load-balancing-2d --lb-strategy sfc-hilbert \
        --lb-weight level --lb-stats-file stats.csv
    mpiexec -n 4 ./mpi-load-balancing-3d --lb-strategy diffusion
    # other strategies: --lb-strategy void | sfc-morton | metis | scotch

``--lb-stats-file`` appends one CSV line per rebalance (imbalance before and
after, migrated cells) so different strategies can be compared on the same
scenario; ``--lb-threshold`` switches from periodic rebalancing to the
``required()`` trigger; ``--lb-skew`` starts with every cell on rank 0 to
exercise the strategies from a maximally skewed state.

The integration suite ``tests/test_load_balancing.py`` drives the demo: for
every available strategy, both dimensions and several process counts it checks
(a) that the final field is **bit-for-bit identical** to the sequential
``void`` run (load balancing must never change the physics, only the
distribution) and (b) that the strategy reaches ``imbalance < 0.1``. It is
skipped cleanly when MPI or the demo executable is absent. Point it at an
MPI-enabled build with ``SAMURAI_MPI_BUILD_DIR`` if it is not under
``../build``.

The micro-benchmark ``benchmark/benchmark_load_balancing.cpp`` (target
``bench_load_balancing``, built with ``-DBUILD_BENCHMARKS=ON -DWITH_MPI=ON``)
times ``partition()`` alone and a full ``load_balance()`` pass from a skewed
state, per strategy, on a uniform mesh:

.. code-block:: bash

    mpiexec -n 4 ./benchmark/bench_load_balancing

Measured ``partition()`` cost per call (4 processes, uniform mesh) ranks the
strategies over two orders of magnitude: the space-filling curves are the
cheapest (≈ 1 ms), ``Diffusion`` slightly above, ``Metis`` an order of
magnitude higher (graph build + ParMETIS, ≈ 10 ms) and ``Scotch`` two orders
higher still (≈ 100–200 ms). The migration + rebuild cost that follows is the
same for every strategy (it does not depend on which one produced the flags).

Choosing a strategy
^^^^^^^^^^^^^^^^^^^^

Measured on the demo (advected blob, 2 and 4 processes, uniform weight):

.. list-table::
   :header-rows: 1

   * - Situation
     - Recommended
     - Why
   * - Default / no external dependency
     - ``SFC<Hilbert>``
     - Reaches ``imbalance ≈ 0`` with little migration; a small number of
       ``all_reduce`` and no external library. Good locality.
   * - Minimal communication volume (ghost cut)
     - ``Metis`` (``adaptive = true`` in AMR)
     - Lowest edge cut of all strategies; adaptive mode reuses the previous
       partition to limit migration. Cost: builds the cell graph, needs ParMETIS.
   * - Incremental rebalancing, neighbour-only communication
     - ``Diffusion``
     - Migrates the fewest cells per call and talks only to MPI neighbours;
       balances asymptotically (a few calls) rather than in one shot.
   * - Cheapest key computation, locality not critical
     - ``SFC<Morton>``
     - Same algorithm as Hilbert with cheaper keys but poorer locality.
   * - Baseline / disable balancing
     - ``Void``
     - No migration; used as the reference in the comparison suite.

``Scotch`` produces excellent balance (``imbalance ≈ 0``) but re-partitions
from scratch at every call, so it migrates far more cells than ``Metis`` or
``Diffusion`` — prefer it for a one-off (re)partition rather than per-step
rebalancing.

Graph partitioners (ParMETIS, PT-Scotch)
------------------------------------------

ParMETIS and PT-Scotch are external graph partitioners that produce
near-optimal partitions by minimising the edge cut (the number of ghost
cells that must be exchanged between processes). They are the best choice
when communication volume matters more than partitioning speed, and when
the mesh has irregular geometry (non-convex domains, highly adaptive
refinement).

.. list-table::
   :header-rows: 1

   * - Strategy
     - Header
     - External dependency
     - Build option
   * - ``Metis``
     - ``strategies/metis.hpp``
     - ParMETIS
     - ``SAMURAI_WITH_PARMETIS=ON``
   * - ``Scotch``
     - ``strategies/scotch.hpp``
     - PT-Scotch
     - ``SAMURAI_WITH_PTSCOTCH=ON``

Both options require ``WITH_MPI=ON`` and the corresponding library
installed. In conda, install them with:

.. code-block:: bash

    conda install parmetis ptscotch

CMake configuration example:

.. code-block:: bash

    cmake . -Bbuild \
        -DWITH_MPI=ON \
        -DSAMURAI_WITH_PARMETIS=ON \
        -DSAMURAI_WITH_PTSCOTCH=ON \
        -DBUILD_TESTS=ON

Usage (ParMETIS):

.. code-block:: c++

    #include <samurai/load_balancing/load_balancer.hpp>
    #include <samurai/load_balancing/strategies/metis.hpp>

    namespace lb = samurai::load_balancing;

    // Default: geometric k-way partitioning
    auto balancer = lb::make_load_balancer<lb::Metis>();

    // Adaptive repartitioning (minimises data movement, recommended for AMR)
    auto balancer = lb::make_load_balancer(lb::LoadBalanceConfig{}, lb::Metis{lb::MetisOptions{.adaptive = true}});

Usage (PT-Scotch):

.. code-block:: c++

    #include <samurai/load_balancing/load_balancer.hpp>
    #include <samurai/load_balancing/strategies/scotch.hpp>

    namespace lb = samurai::load_balancing;

    auto balancer = lb::make_load_balancer<lb::Scotch>();

Cell graph
^^^^^^^^^^

The graph used by both strategies is built by ``build_cell_graph()`` in
``graph.hpp``. Each cell is a vertex whose weight is the user-provided cell
weight (scaled to integers, average ~100). Edges connect face-adjacent cells
(across level jumps of at most 1, guaranteed by the graduation constraint);
edge weights count the number of shared reference-level faces, which models
the ghost communication volume. The graph also carries cell centre coordinates
for ParMETIS' geometric partitioning mode.

The graph is extended across MPI boundaries: for each pair of neighbouring
processes, the cells on the interface exchange their global IDs, and edges are
added between face-adjacent pairs that belong to different processes.

ParMETIS details
^^^^^^^^^^^^^^^^

Two modes are available:

- **Geometric k-way** (``MetisOptions::adaptive = false``, the default):
  calls ``ParMETIS_V3_PartGeomKway`` with cell coordinates, then falls back to
  ``ParMETIS_V3_PartKway`` if the geometric hint is unavailable. Best for cold
  starts.

- **Adaptive repartitioning** (``MetisOptions::adaptive = true``): calls
  ``ParMETIS_V3_AdaptiveRepart`` with the current partition as a hint. The
  ``itr`` parameter (redistribution cost) penalises moving cells unless the
  balance improvement is significant. Recommended for AMR, where the mesh
  changes incrementally between time steps.

PT-Scotch details
^^^^^^^^^^^^^^^^^

Calls ``SCOTCH_dgraphPart`` with ``nparts = world.size()`` and a
``SCOTCH_STRATBALANCE`` strategy (imbalance tolerance 5 %). Unlike the
original Strafella implementation which hardcoded ``nparts = 2``, this version
uses the full communicator size. The distributed graph is built via
``SCOTCH_dgraphBuild`` with vertex and edge weights.

When to choose graph partitioners
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

- **Use Metis/Scotch** when minimising the edge cut matters (irregular
  domains, adaptive refinement, many processes), and when the overhead of
  building the graph is amortised over many time steps.
- **Use SFC** (especially Hilbert) for fast partitioning with good locality,
  no external dependency, and deterministic results.
- **Use Diffusion** for incremental, neighbour-only rebalancing with no
  external dependency and minimal cell movement per call.

All strategies produce the same ``LoadBalanceStats`` and can be compared
on the same scenario using the demo or the benchmark (roadmap step 6).

Testing
-------

The MPI tests of the module live in ``tests/mpi`` and run at 2, 3 and 4
processes (``ctest -R test_lb``); the migration invariants (no cell lost or
duplicated, every field value follows its cell) are checked by
``check_lb_invariants`` in ``tests/mpi/mpi_test_utils.hpp``. The weight
policies are tested without MPI in ``tests/test_load_balancing_weight.cpp``.
