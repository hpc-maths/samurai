// Copyright 2018-2025 the samurai's authors
// SPDX-License-Identifier:  BSD-3-Clause

#pragma once

namespace samurai::load_balancing
{
    /**
     * Configuration of the load balancing module.
     *
     * All tunable constants of the module live here: no strategy or driver is
     * allowed to hardcode a threshold, an iteration count or an MPI tag.
     */
    struct LoadBalanceConfig
    {
        /// `LoadBalancer::required()` returns true when the global imbalance
        /// `max(load)/avg(load) - 1` exceeds this threshold.
        double imbalance_threshold = 0.05;

        /// (diffusion strategy) fluxes smaller than this fraction of the local
        /// average load are zeroed out to avoid micro-migrations.
        double flux_threshold = 0.01;

        /// (diffusion strategy) maximum number of iterations of the iterative
        /// flux solver.
        int diffusion_iterations = 50;

        /// When true, the driver traces its decisions on std::clog, prefixed
        /// by the MPI rank. Never use std::cout in library code.
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
