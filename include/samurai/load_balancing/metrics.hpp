// Copyright 2018-2025 the samurai's authors
// SPDX-License-Identifier:  BSD-3-Clause

#pragma once

#include <cstddef>
#include <string>

#include "../algorithm.hpp"

#ifdef SAMURAI_WITH_MPI
#include <boost/mpi.hpp>
#include <boost/mpi/collectives.hpp>
#endif

namespace samurai::load_balancing
{
    /**
     * Quality metrics collected by every call to `LoadBalancer::load_balance()`.
     *
     * The loads are weighted by the policy given to `load_balance()`; the
     * imbalances are global quantities (identical on every rank). Comparing
     * `imbalance_before` and `imbalance_after` across strategies on the same
     * scenario is the basis of the comparative benchmark (roadmap step 6).
     */
    struct LoadBalanceStats
    {
        std::size_t cells_before       = 0;  ///< local cell count before balancing
        std::size_t cells_after        = 0;  ///< local cell count after balancing
        std::size_t cells_migrated_out = 0;  ///< cells sent to other ranks
        std::size_t cells_migrated_in  = 0;  ///< cells received from other ranks
        double load_before             = 0.; ///< local weighted load before balancing
        double load_after              = 0.; ///< local weighted load after balancing
        double imbalance_before        = 0.; ///< global max(load)/avg(load) - 1 before
        double imbalance_after         = 0.; ///< global max(load)/avg(load) - 1 after
        double partition_time          = 0.; ///< seconds spent in the strategy
        double migration_time          = 0.; ///< seconds spent migrating cells and fields
        std::string strategy_name;
    };

    /**
     * Weighted load owned by this process: sum of `weight(cell)` over the
     * local cells.
     *
     * @note MPI: no communication (also available in non-MPI builds).
     */
    template <class Mesh, class Weight>
    double local_load(const Mesh& mesh, const Weight& weight)
    {
        using mesh_id_t = typename Mesh::mesh_id_t;
        double load     = 0.;
        for_each_cell(mesh[mesh_id_t::cells],
                      [&](const auto& cell)
                      {
                          load += weight(cell);
                      });
        return load;
    }

#ifdef SAMURAI_WITH_MPI
    /**
     * Global load imbalance: `max(load)/avg(load) - 1`. Zero means perfect
     * balance; `P - 1` means the whole load sits on a single process.
     *
     * @note MPI: collective on the world communicator (two all_reduce); every
     *       rank gets the same value.
     */
    template <class Mesh, class Weight>
    double imbalance(const Mesh& mesh, const Weight& weight)
    {
        boost::mpi::communicator world;
        const double load = local_load(mesh, weight);
        const double max  = boost::mpi::all_reduce(world, load, boost::mpi::maximum<double>());
        const double sum  = boost::mpi::all_reduce(world, load, std::plus<double>());
        const double avg  = sum / world.size();
        return avg > 0. ? max / avg - 1. : 0.;
    }

    /**
     * Collective decision: should we rebalance now?
     *
     * @return true when `imbalance(mesh, weight) > threshold`. Guaranteed to
     *         return the same value on every rank (the decision is taken on
     *         globally reduced quantities only).
     * @note MPI: collective on the world communicator.
     */
    template <class Mesh, class Weight>
    bool require_balance(const Mesh& mesh, const Weight& weight, double threshold)
    {
        return imbalance(mesh, weight) > threshold;
    }
#endif
}
