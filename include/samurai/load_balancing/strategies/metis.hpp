// Copyright 2018-2025 the samurai's authors
// SPDX-License-Identifier:  BSD-3-Clause

#pragma once

/**
 * @file metis.hpp
 * @brief ParMETIS graph partitioning strategy.
 *
 * Partitions the cell graph using ParMETIS_V3_PartGeomKway (geometric k-way)
 * or ParMETIS_V3_AdaptiveRepart (minimises data redistribution between
 * successive partitions). The latter is recommended for AMR.
 *
 * Communication: build_cell_graph (one all_gather + neighbour exchanges) then
 * ParMETIS (collective).
 *
 * Reference: G. Karypis & V. Kumar, "Parallel multilevel k-way partitioning
 * scheme for irregular graphs", Proc. SC'96.
 */

#ifndef SAMURAI_WITH_PARMETIS
#error "samurai/metis.hpp requires SAMURAI_WITH_PARMETIS=ON"
#endif

#include <array>
#include <cstddef>
#include <stdexcept>
#include <string>
#include <vector>

#include <parmetis.h>

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
     * Options for the ParMETIS strategy.
     */
    struct MetisOptions
    {
        /// Use adaptive repartitioning (minimises data migration). Recommended
        /// for AMR: the previous partition is reused as a hint.
        bool adaptive = false;

        /// Imbalance tolerance (ParMETIS ubvec). 1.05 allows 5 % imbalance.
        double imbalance_tolerance = 1.05;
    };

    /**
     * ParMETIS partitioning strategy. Satisfies the PartitionStrategy concept.
     */
    class Metis
    {
      public:

        Metis() = default;

        explicit Metis(MetisOptions options)
            : m_options(options)
        {
        }

        std::string name() const
        {
            return m_options.adaptive ? "metis-adaptive" : "metis";
        }

        /**
         * Partition the mesh using ParMETIS and return the flags field.
         * @note MPI: collective on the world communicator.
         */
        template <class Mesh, class Weight>
        auto partition(Mesh& mesh, const Weight& weight) const
        {
            using mesh_id_t           = typename Mesh::mesh_id_t;
            constexpr std::size_t dim = Mesh::dim;

            boost::mpi::communicator world;
            auto flags = make_scalar_field<int>("lb_flags", mesh);
            flags.fill(world.rank());

            if (world.size() <= 1)
            {
                return flags;
            }

            times::timers.start("lb:metis:build_graph");
            auto graph = build_cell_graph<idx_t>(mesh, weight);
            times::timers.stop("lb:metis:build_graph");

            idx_t wgtflag = 2; // vertex weights only
            idx_t numflag = 0;
            idx_t ncon    = 1;
            idx_t nparts  = world.size();
            std::vector<real_t> tpwgts(static_cast<std::size_t>(ncon * nparts), 1.0 / static_cast<real_t>(nparts));
            std::vector<real_t> ubvec(static_cast<std::size_t>(ncon), static_cast<real_t>(m_options.imbalance_tolerance));
            idx_t edgecut = 0;
            std::vector<idx_t> part(static_cast<std::size_t>(graph.nvtx_local()), -1);
            MPI_Comm comm = MPI_COMM_WORLD;

            times::timers.start("lb:metis:partition");
            if (m_options.adaptive)
            {
                std::vector<idx_t> vsize(static_cast<std::size_t>(graph.nvtx_local()), 1000);
                for (std::size_t i = 0; i < static_cast<std::size_t>(graph.nvtx_local()); ++i)
                {
                    part[i] = world.rank();
                }
                real_t itr       = 1000.0;
                real_t itr_val   = 1000.0;
                idx_t options[3] = {0};

                int result = ParMETIS_V3_AdaptiveRepart(graph.vtxdist.data(),
                                                        graph.xadj.data(),
                                                        graph.adjncy.data(),
                                                        graph.vwgt.data(),
                                                        vsize.data(),
                                                        graph.adjwgt.data(),
                                                        &wgtflag,
                                                        &numflag,
                                                        &ncon,
                                                        &nparts,
                                                        tpwgts.data(),
                                                        ubvec.data(),
                                                        &itr_val,
                                                        options,
                                                        &edgecut,
                                                        part.data(),
                                                        &comm);
                if (result != METIS_OK)
                {
                    throw std::runtime_error("ParMETIS_V3_AdaptiveRepart failed (error " + std::to_string(result) + ")");
                }
            }
            else
            {
                idx_t ndims      = static_cast<idx_t>(dim);
                idx_t options[3] = {0};

                int result = ParMETIS_V3_PartGeomKway(graph.vtxdist.data(),
                                                      graph.xadj.data(),
                                                      graph.adjncy.data(),
                                                      graph.vwgt.data(),
                                                      graph.adjwgt.data(),
                                                      &wgtflag,
                                                      &numflag,
                                                      &ndims,
                                                      graph.xyz.data(),
                                                      &ncon,
                                                      &nparts,
                                                      tpwgts.data(),
                                                      ubvec.data(),
                                                      options,
                                                      &edgecut,
                                                      part.data(),
                                                      &comm);

                if (result != METIS_OK)
                {
                    // PartGeomKway may fail if coordinates are degenerate;
                    // fall back to PartKway without geometric hint.
                    idx_t edgecut2    = 0;
                    idx_t options2[3] = {0};
                    int result2       = ParMETIS_V3_PartKway(graph.vtxdist.data(),
                                                       graph.xadj.data(),
                                                       graph.adjncy.data(),
                                                       graph.vwgt.data(),
                                                       graph.adjwgt.data(),
                                                       &wgtflag,
                                                       &numflag,
                                                       &ncon,
                                                       &nparts,
                                                       tpwgts.data(),
                                                       ubvec.data(),
                                                       options2,
                                                       &edgecut2,
                                                       part.data(),
                                                       &comm);
                    if (result2 != METIS_OK)
                    {
                        throw std::runtime_error("ParMETIS_V3_PartKway failed (error " + std::to_string(result2) + ")");
                    }
                }
            }
            times::timers.stop("lb:metis:partition");

            // Copy partition result into flags (same iteration order as graph)
            std::size_t i = 0;
            for_each_cell(mesh[mesh_id_t::cells],
                          [&](const auto& cell)
                          {
                              flags[cell] = static_cast<int>(part[i++]);
                          });

            return flags;
        }

      private:

        MetisOptions m_options{};
    };
}
