// Copyright 2018-2025 the samurai's authors
// SPDX-License-Identifier:  BSD-3-Clause

#pragma once

/**
 * Load balancing: the driver `LoadBalancer`, its strategies, the weight
 * policies and the imbalance metrics.
 */
namespace samurai::load_balancing
{
    /**
     * Configuration of the load balancing *driver* (`LoadBalancer`).
     *
     * Only the parameters the driver itself reads live here. Each strategy
     * carries its own options struct, such as `DiffusionOptions`,
     * `MetisOptions` or `ScotchOptions`, so the two never overlap.
     */
    struct LoadBalanceConfig
    {
        /// `LoadBalancer::required()` returns true when the global imbalance
        /// `max(load)/avg(load) - 1` exceeds this threshold.
        double imbalance_threshold = 0.05;

        /// Not read by any code of the module: setting it has no effect.
        bool verbose = false;
    };

    /**
     * MPI tags reserved for the load balancing module.
     *
     * The migration exchanges a single message type per rank pair, hence a
     * single tag. The value is chosen high enough to avoid collisions with the
     * small literal tags used elsewhere in samurai (e.g. ghost updates).
     */
    inline constexpr int tag_migration = 4200;

    /**
     * MPI tag reserved for the neighbour-only scalar exchanges of the diffusion
     * strategy (degrees and loads). Distinct from @ref tag_migration so the two
     * cannot be confused if a future strategy interleaves them.
     */
    inline constexpr int tag_diffusion = 4201;
}
