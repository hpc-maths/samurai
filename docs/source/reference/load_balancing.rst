Load balancing
==============

When running with MPI, the mesh adaptation (AMR/MRA) progressively unbalances
the work between processes: the cells follow the solution, not the initial
domain decomposition. The load balancing module redistributes the cells — and
the fields living on them — between processes.

The module lives in ``include/samurai/load_balancing/`` under the namespace
``samurai::load_balancing`` and is only active when samurai is built with
``WITH_MPI=ON``. Its design and development plan are described in
``docs/load_balancing_roadmap.md``.

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

    // inside the time loop, after mesh adaptation:
    auto weight = lb::weight::uniform();
    if (balancer.required(u.mesh(), weight))
    {
        auto stats = balancer.load_balance(weight, u, v, w); // all fields!
    }

Two rules:

* **every field living on the mesh must be passed** to ``load_balance()``:
  a field that does not migrate with its cells holds garbage afterwards;
* ``required()`` and ``load_balance()`` are **collective**: every rank must
  call them (the decision returned by ``required()`` is guaranteed identical
  on all ranks).

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

Every call to ``load_balance()`` returns a ``LoadBalanceStats``:

.. code-block:: c++

    struct LoadBalanceStats
    {
        std::size_t cells_before, cells_after;            // local cell counts
        std::size_t cells_migrated_out, cells_migrated_in;
        double load_before, load_after;                   // local weighted loads
        double imbalance_before, imbalance_after;         // global max/avg - 1
        double partition_time, migration_time;            // seconds
        std::string strategy_name;
    };

The global imbalance ``max(load)/avg(load) - 1`` is 0 for a perfect balance
and ``P - 1`` when the whole load sits on one process. It is the quantity
compared across strategies by the benchmark (roadmap step 6). The free
functions ``local_load()``, ``imbalance()`` and ``require_balance()`` of
``metrics.hpp`` expose the same computations; the documentation of each one
states whether it communicates.

Visualizing partitions
----------------------

.. code-block:: c++

    #include <samurai/load_balancing/dump.hpp>

    lb::dump_partition(path, "partition", mesh);
    // writes <path>/partition_size_<P>.h5 + .xdmf

The file holds one scalar field ``rank`` (the owning process of each cell):
open it in ParaView and color by ``rank`` to compare the shapes produced by
different strategies. This is the quickest way to *see* whether a partition
is made of compact blobs (graph partitioners, diffusion), curve segments
(SFC) or stripes (legacy balancer).

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
   * - ParMETIS / PT-Scotch graph partitioning
     - ``strategies/metis.hpp`` / ``strategies/scotch.hpp``
     - roadmap step 4
   * - Diffusion (heat-equation fluxes + interface layers)
     - ``strategies/diffusion.hpp``
     - roadmap step 5

Space-filling curves (SFC)
--------------------------

``SFC<Curve>`` orders the cells along a space-filling curve (keys computed at
the global max level) and cuts the curve into P segments of equal *weighted*
load, using one MPI scan and one all_reduce — no gather, O(1) communication
volume. The balance is reached in a single call, up to the weight of the
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

Limits: the 64-bit keys bound the usable refinement: 32 bits per coordinate
in 2D and 21 bits in 3D (i.e. max_level up to 21 in 3D on a unit domain).
Negative cell indices are handled by a global shift; coordinates beyond the
bit budget trigger an assertion in Debug.

Comparing strategies
--------------------

The demo ``demos/mpi/load_balancing_2d.cpp`` (target ``mpi-load-balancing-2d``)
runs the same 2D adaptive advection case with any strategy and collects the
metrics, e.g.:

.. code-block:: bash

    mpiexec -n 4 ./mpi-load-balancing-2d --lb-strategy sfc-hilbert \
        --lb-weight level --lb-dump --lb-stats-file stats.csv

``--lb-stats-file`` appends one CSV line per rebalance (imbalance before and
after, migrated cells, timings) so different strategies can be compared on
the same scenario; ``--lb-dump`` writes the partition for ParaView at every
rebalance; ``--lb-threshold`` switches from periodic rebalancing to the
``required()`` trigger. The full benchmark harness arrives with roadmap
step 6.

Testing
-------

The MPI tests of the module live in ``tests/mpi`` and run at 2, 3 and 4
processes (``ctest -R test_lb``); the migration invariants (no cell lost or
duplicated, every field value follows its cell) are checked by
``check_lb_invariants`` in ``tests/mpi/mpi_test_utils.hpp``. The weight
policies are tested without MPI in ``tests/test_load_balancing_weight.cpp``.
