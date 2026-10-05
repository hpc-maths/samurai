// Copyright 2018-2025 the samurai's authors
// SPDX-License-Identifier:  BSD-3-Clause

#pragma once

#include "../../field.hpp"

#ifdef SAMURAI_WITH_MPI
#include <boost/mpi.hpp>

namespace samurai::load_balancing
{
    /**
     * Void strategy: every cell stays on its rank.
     *
     * It is a baseline. Running the driver with this strategy measures the
     * fixed cost of the load balancing (partition and routing discovery)
     * without moving anything, and gives the reference run for the tests that
     * check that load balancing never changes the numerical result.
     *
     * No cell moves, and neither the mesh nor the fields change. `partition()`
     * does not communicate; the driver still runs its `all_to_all`, then stops
     * because no rank sends any cell.
     */
    class Void
    {
      public:

        /**
         * Returns a field that holds the current rank for every cell.
         * @note MPI: no communication.
         */
        template <class Mesh, class Weight>
        auto partition(Mesh& mesh, const Weight& /*weight*/) const
        {
            boost::mpi::communicator world;
            auto flags = make_scalar_field<int>("lb_flags", mesh);
            flags.fill(world.rank());
            return flags;
        }

        /// Returns `"void"`.
        std::string name() const
        {
            return "void";
        }
    };
}
#endif
