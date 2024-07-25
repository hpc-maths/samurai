// Copyright 2018-2024 the samurai's authors
// SPDX-License-Identifier:  BSD-3-Clause

#pragma once

#include "ptscotch.h"

#include "field.hpp"
#include "load_balancing.hpp"
#include "load_balancing_utils.hpp"
#include "mesh_interval.hpp"
#include "stencil.hpp"

using namespace xt::placeholders;

namespace samurai::Load_balancing
{
    class Scotch : public samurai::LoadBalancer<Scotch>
    {
      private:

        int _ndomains;
        int _rank;

        template <class Mesh_t, class Stencil, class Field_t>
        void propagate(const Mesh_t&, const Stencil&, Field_t&, int, int&) const
        {
        }

      public:

        Scotch()
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
            return "Scotch";
        }

        template <class Mesh_t>
        Mesh_t& reordering_impl(Mesh_t& mesh)
        {
            return mesh;
        }

        template <class Mesh_t>
        Mesh_t load_balance_impl(Mesh_t& mesh)
        {
            constexpr std::size_t dim = Mesh_t::dim;
            using mesh_id_t           = typename Mesh_t::mesh_id_t;
            auto graph                = build_graph(mesh);

            boost::mpi::communicator world;
            SCOTCH_Dgraph grafdat;
            SCOTCH_dgraphInit(&grafdat, MPI_COMM_WORLD);

            // std::cout << "xadj: " << xt::adapt(graph.xadj) << std::endl;
            // std::cout << "adjnct: " << xt::adapt(graph.adjncy) << std::endl;
            SCOTCH_dgraphBuild(&grafdat,              // grafdat
                               0,                     // baseval, c-style numbering
                               graph.xadj.size() - 1, // vertlocnbr, nCells
                               graph.xadj.size() - 1, // vertlocmax
                               const_cast<SCOTCH_Num*>(graph.xadj.data()),
                               nullptr,

                               nullptr, // veloloctab, vtx weights
                               nullptr, // vlblloctab

                               graph.adjncy.size(),                          // edgelocnbr, number of arcs
                               graph.adjncy.size(),                          // edgelocsiz
                               const_cast<SCOTCH_Num*>(graph.adjncy.data()), // edgeloctab
                               nullptr,                                      // edgegsttab
                               nullptr                                       // edlotab, edge weights
            );
            SCOTCH_dgraphCheck(&grafdat);
            SCOTCH_Strat strategy;
            SCOTCH_stratInit(&strategy);

            SCOTCH_Arch archdat;
            SCOTCH_archInit(&archdat);
            // SCOTCH_archCmplt(&archdat, 1);

            std::vector<SCOTCH_Num> part(mesh[mesh_id_t::cells].nb_cells(), 17);

            SCOTCH_Dmapping mappdat;
            SCOTCH_dgraphMapInit(&grafdat, &mappdat, &archdat, part.data());

            SCOTCH_dgraphPart(&grafdat,
                              2,
                              &strategy,  // const SCOTCH_Strat *
                              part.data() // parttab
            );
            SCOTCH_archExit(&archdat);
            SCOTCH_stratExit(&strategy);
            SCOTCH_dgraphExit(&grafdat);

            /* ---------------------------------------------------------------------------------------------------------- */
            /* ------- Construct new mesh for current process ----------------------------------------------------------- */
            /* ---------------------------------------------------------------------------------------------------------- */

            auto partition_field = make_field<std::size_t, 1>("partition", mesh);
            std::size_t ipart    = 0;
            for_each_cell(mesh[mesh_id_t::cells],
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
