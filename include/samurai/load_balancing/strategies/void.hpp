// Copyright 2018-2025 the samurai's authors
// SPDX-License-Identifier:  BSD-3-Clause

#pragma once

/**
 * Void strategy: every cell stays where it is.
 *
 * Purpose: baseline. Running the driver with this strategy measures the fixed
 * overhead of the load balancing infrastructure (partition + routing
 * discovery) without moving anything, and gives the reference run for the
 * "load balancing must never change the numerical result" tests.
 *
 * Guarantees: no migration, no change of mesh or fields.
 * Communication: none in partition(); the driver still performs its single
 * all_to_all, then stops (no migration detected anywhere).
 */

#include "../../field.hpp"

#ifdef SAMURAI_WITH_MPI
#include <boost/mpi.hpp>

namespace samurai::load_balancing
{
    class Void
    {
      public:

        /// flags = current rank for every cell. @note MPI: no communication.
        template <class Mesh, class Weight>
        auto partition(Mesh& mesh, const Weight& /*weight*/) const
        {
            boost::mpi::communicator world;
            auto flags = make_scalar_field<int>("lb_flags", mesh);
            flags.fill(world.rank());
            return flags;
        }

        std::string name() const
        {
            return "void";
        }
    };
}
#endif
