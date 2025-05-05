#pragma once

#include "load_balancing.hpp"
#include <map>
#include <samurai/field.hpp>

// for std::sort
#include <algorithm>
#ifdef SAMURAI_WITH_MPI
namespace Load_balancing
{

    class Life : public samurai::LoadBalancer<Life>
    {
      private:

        int _ndomains;
        int _rank;

      public:

        Life()
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
            return "life";
        }

        template <class Mesh_t>
        auto reordering_impl(Mesh_t& mesh)
        {
            auto flags = samurai::make_field<int, 1>("ordering_flag", mesh);
            flags.fill(_rank);

            return flags;
        }

        template <class Mesh_t>
        auto load_balance_impl(Mesh_t& mesh)
        {
            using mpi_subdomain_t = typename Mesh_t::mpi_subdomain_t;
            // using CellList_t      = typename Mesh_t::cl_type;
            // using mesh_id_t       = typename Mesh_t::mesh_id_t;

            // using Coord_t = xt::xtensor_fixed<double, xt::xshape<Mesh_t::dim>>;
            // using Stencil = xt::xtensor_fixed<int, xt::xshape<Mesh_t::dim>>;

            boost::mpi::communicator world;

            // For debug
            // std::ofstream logs;
            // logs.open( fmt::format("log_{}.dat", world.rank()), std::ofstream::app );
            logs << fmt::format("> New load-balancing using {} ", getName()) << std::endl;

            auto flags = samurai::make_field<int, 1>("balancing_flags", mesh);
            flags.fill(_rank);

            // neighbourhood
            std::vector<mpi_subdomain_t>& neighbourhood = mesh.mpi_neighbourhood();

            // fluxes to each neighbour
            std::vector<int> fluxes = samurai::cmptFluxes<samurai::BalanceElement_t::CELL>(mesh, 5);

            // cpy that can be modified
            std::vector<int> new_fluxes(fluxes);

            // Loads of each processes
            std::vector<int> loads;
            int my_load = static_cast<int>(samurai::cmptLoad<samurai::BalanceElement_t::CELL>(mesh));
            boost::mpi::all_gather(world, my_load, loads);

            // samurai::cellExists();

            return flags;
        }
    };
}
#endif
