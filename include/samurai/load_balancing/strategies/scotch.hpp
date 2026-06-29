// Copyright 2018-2025 the samurai's authors
// SPDX-License-Identifier:  BSD-3-Clause

#pragma once

/**
 * @file scotch.hpp
 * @brief PT-Scotch graph partitioning strategy.
 *
 * Partitions the cell graph using SCOTCH_dgraphPart with a balance-oriented
 * strategy (SCOTCH_STRATBALANCE, imbalance tolerance 5 %). Unlike the
 * original Strafella implementation which hardcoded nparts = 2, this version
 * uses the communicator size as the number of partitions.
 *
 * Prerequisites: SAMURAI_WITH_PTSCOTCH must be ON at configure time.
 * The header emits a #error if included without the option.
 *
 * Communication: build_cell_graph (one all_gather + neighbour exchanges) then
 * SCOTCH_dgraphPart (collective).
 *
 * Reference: C. Chevalier & F. Pellegrini, "PT-Scotch: a tool for efficient
 * static and dynamic graph partitioning", 2008.
 */

#ifndef SAMURAI_WITH_PTSCOTCH
#error "samurai/scotch.hpp requires SAMURAI_WITH_PTSCOTCH=ON"
#endif

#include <array>
#include <cstddef>
#include <stdexcept>
#include <string>
#include <vector>

#include <ptscotch.h>

#include "../../algorithm.hpp"
#include "../../field.hpp"
#include "../../mesh.hpp"
#include "../../timers.hpp"
#include "../config.hpp"
#include "../graph.hpp"
#include "../weight.hpp"

#include <boost/mpi.hpp>

namespace samurai::load_balancing
{
    /**
     * Options for the PT-Scotch strategy.
     */
    struct ScotchOptions
    {
        /// Target imbalance for the SCOTCH strategy. Default 0.05 (5 %).
        double imbalance_tolerance = 0.05;
    };

    /**
     * PT-Scotch partitioning strategy. Satisfies the PartitionStrategy concept.
     */
    class Scotch
    {
      public:

        Scotch() = default;

        explicit Scotch(ScotchOptions options)
            : m_options(options)
        {
        }

        std::string name() const
        {
            return "scotch";
        }

        /**
         * Partition the mesh using PT-Scotch and return the flags field.
         * @note MPI: collective on the world communicator.
         */
        template <class Mesh, class Weight>
        auto partition(Mesh& mesh, const Weight& weight) const
        {
            using mesh_id_t = typename Mesh::mesh_id_t;

            boost::mpi::communicator world;
            auto flags = make_scalar_field<int>("lb_flags", mesh);
            flags.fill(world.rank());

            if (world.size() <= 1)
            {
                return flags;
            }

            times::timers.start("lb:scotch:build_graph");
            auto graph = build_cell_graph<SCOTCH_Num>(mesh, weight);
            times::timers.stop("lb:scotch:build_graph");

            SCOTCH_Dgraph grafdat;
            SCOTCH_dgraphInit(&grafdat, MPI_COMM_WORLD);

            SCOTCH_Num nvtx_local   = graph.nvtx_local();
            SCOTCH_Num nedges_local = static_cast<SCOTCH_Num>(graph.adjncy.size());

            int result = SCOTCH_dgraphBuild(&grafdat,
                                            0,                    // baseval: C-style numbering
                                            nvtx_local,           // vertlocnbr
                                            nvtx_local,           // vertlocmax
                                            graph.xadj.data(),    // vertloctab
                                            nullptr,              // vendloctab
                                            graph.vwgt.data(),    // veloloctab (vertex weights)
                                            nullptr,              // vlblloctab
                                            nedges_local,         // edgelocnbr
                                            nedges_local,         // edgelocsiz
                                            graph.adjncy.data(),  // edgeloctab
                                            nullptr,              // edgegsttab
                                            graph.adjwgt.data()); // edlotabtab (edge weights)

            if (result != 0)
            {
                SCOTCH_dgraphExit(&grafdat);
                throw std::runtime_error("SCOTCH_dgraphBuild failed");
            }

            // Set vertex labels to global indices for deterministic partitioning
            // (not strictly required, but helps Scotch's internal ordering)

            SCOTCH_Strat stratdat;
            SCOTCH_stratInit(&stratdat);

            // Build a balance-oriented strategy with the target imbalance
            result = SCOTCH_stratDgraphMapBuild(&stratdat,
                                                SCOTCH_STRATBALANCE,
                                                0,                                                   // 1855, rep
                                                static_cast<SCOTCH_Num>(world.size()),               // nparts
                                                static_cast<double>(m_options.imbalance_tolerance)); // balance param

            if (result != 0)
            {
                SCOTCH_stratExit(&stratdat);
                SCOTCH_dgraphExit(&grafdat);
                throw std::runtime_error("SCOTCH_stratDgraphMapBuild failed");
            }

            std::vector<SCOTCH_Num> part(static_cast<std::size_t>(nvtx_local));

            times::timers.start("lb:scotch:partition");
            result = SCOTCH_dgraphPart(&grafdat, static_cast<SCOTCH_Num>(world.size()), &stratdat, part.data());
            times::timers.stop("lb:scotch:partition");

            if (result != 0)
            {
                SCOTCH_stratExit(&stratdat);
                SCOTCH_dgraphExit(&grafdat);
                throw std::runtime_error("SCOTCH_dgraphPart failed");
            }

            SCOTCH_stratExit(&stratdat);
            SCOTCH_dgraphExit(&grafdat);

            // Copy partition result into flags
            std::size_t i = 0;
            for_each_cell(mesh[mesh_id_t::cells],
                          [&](const auto& cell)
                          {
                              flags[cell] = static_cast<int>(part[i++]);
                          });

            return flags;
        }

      private:

        ScotchOptions m_options{};
    };
}
