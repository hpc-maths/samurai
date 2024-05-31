// Copyright 2018-2024 the samurai's authors
// SPDX-License-Identifier:  BSD-3-Clause

#pragma once

#include <parmetis.h>

#include "field.hpp"
#include "load_balancing_utils.hpp"

namespace samurai::Load_balancing
{
    class Metis : public samurai::LoadBalancer<Metis>
    {
      private:

        int _ndomains;
        int _rank;

        template <class Mesh_t, class Stencil, class Field_t>
        void propagate(const Mesh_t&, const Stencil&, Field_t&, int, int&) const
        {
        }

      public:

        Metis()
        {
#ifdef SAMURAI_WITH_MPI
            boost::mpi::communicator world;
            _ndomains = world.size();
            _rank     = world.rank();
#else
            _ndomains = 1;
            _rank     = 0;
#endif
        }

        inline std::string getName() const
        {
            return "metis";
        }

        template <class Mesh_t>
        Mesh_t load_balance_impl(Mesh_t& mesh)
        {
            constexpr std::size_t dim = Mesh_t::dim;
            using mesh_id_t           = typename Mesh_t::mesh_id_t;
            auto graph                = build_graph(mesh);

            boost::mpi::communicator world;

            idx_t ndims   = dim;
            idx_t wgtflag = 1;
            idx_t numflag = 0;
            idx_t ncon    = 1;
            idx_t nparts  = world.size();
            std::vector<real_t> tpwgts(nparts, 1.0 / nparts);
            real_t ubvec[]    = {1.05};
            idx_t options[10] = {0};
            idx_t edgecut     = 0;
            std::vector<idx_t> part(mesh[mesh_id_t::cells].nb_cells(), 17);
            MPI_Comm mpi_comm = MPI_COMM_WORLD;

            // ParMETIS_V3_PartKway(global_index_offset.data(),
            //                      xadj.data(),
            //                      adjncy.data(),
            //                      NULL,
            //                      NULL, // adjwgt.data(),
            //                      &wgtflag,
            //                      &numflag,
            //                      &ncon,
            //                      &nparts,
            //                      tpwgts.data(),
            //                      ubvec,
            //                      options,
            //                      &edgecut,
            //                      part.data(),
            //                      &mpi_comm);

            ParMETIS_V3_PartGeomKway(graph.global_index_offset.data(),
                                     graph.xadj.data(),
                                     graph.adjncy.data(),
                                     NULL,
                                     graph.adjwgt.data(),
                                     &wgtflag,
                                     &numflag,
                                     &ndims,
                                     graph.xyz.data(),
                                     &ncon,
                                     &nparts,
                                     tpwgts.data(),
                                     ubvec,
                                     options,
                                     &edgecut,
                                     part.data(),
                                     &mpi_comm);

            // ParMETIS_V3_AdaptiveRepart

            /* ---------------------------------------------------------------------------------------------------------- */
            /* ------- Construct new mesh for current process ----------------------------------------------------------- */
            /* ---------------------------------------------------------------------------------------------------------- */

            auto partition_field = make_field<idx_t, 1>("partition", mesh);
            std::size_t ipart    = 0;
            for_each_cell(mesh,
                          [&](const auto& cell)
                          {
                              partition_field[cell] = part[ipart++];
                          });
            // std::cout << "coucou" << std::endl;
            save("metis_partition", mesh, partition_field);
            // std::cout << "coucou 2" << std::endl;

            // Mesh_t new_mesh(new_cl, mesh);

            return mesh;
        }
    };
}
