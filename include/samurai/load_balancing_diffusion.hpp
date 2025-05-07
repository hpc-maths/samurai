
#include "field.hpp"
#include "load_balancing.hpp"
#include <map>

// for std::sort
#include <algorithm>

#ifdef SAMURAI_WITH_MPI
namespace Load_balancing
{

    class Diffusion : public samurai::LoadBalancer<Diffusion>
    {
      private:

        int _ndomains;
        int _rank;

        const double _unbalance_threshold = 0.05; // 5 %

      public:

        Diffusion()
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
            return "diffusion";
        }

        template <class Mesh_t>
        auto reordering_impl(Mesh_t& mesh)
        {
            auto flags = samurai::make_scalar_field<int>("diffusion_flag", mesh);
            flags.fill(_rank);

            return flags;
        }

        template <size_t dim, typename Stencil_t, typename Coord_t>
        std::vector<Stencil_t> getStencilToNeighbour(const Coord_t& bc_current, const Coord_t& bc_neigh) const
        {
            Stencil_t dir_from_neighbour;
            std::vector<Stencil_t> stencils;

            Coord_t tmp;
            double n2 = 0.;
            for (size_t idim = 0; idim < dim; ++idim)
            {
                tmp(idim) = bc_current(idim) - bc_neigh(idim);
                n2 += tmp(idim) * tmp(idim);
            }

            n2 = std::sqrt(n2);

            for (size_t idim = 0; idim < dim; ++idim)
            {
                tmp(idim) /= n2;
                dir_from_neighbour(idim) = static_cast<int>(tmp(idim) / 0.5);

                // FIXME why needed ?
                if (std::abs(dir_from_neighbour(idim)) > 1)
                {
                    dir_from_neighbour(idim) < 0 ? dir_from_neighbour(idim) = -1 : dir_from_neighbour(idim) = 1;
                }

                if (dir_from_neighbour(idim) != 0)
                {
                    Stencil_t dd;
                    dd.fill(0);
                    dd(idim) = dir_from_neighbour(idim);
                    stencils.emplace_back(dd);
                }
            }

            return stencils;
        }

        template <class Mesh_t>
        auto load_balance_impl(Mesh_t& mesh)
        {
            using mpi_subdomain_t = typename Mesh_t::mpi_subdomain_t;
            using CellList_t      = typename Mesh_t::cl_type;
            using mesh_id_t       = typename Mesh_t::mesh_id_t;

            using Coord_t = xt::xtensor_fixed<double, xt::xshape<Mesh_t::dim>>;
            using Stencil = xt::xtensor_fixed<int, xt::xshape<Mesh_t::dim>>;

            boost::mpi::communicator world;
            std::vector<mpi_subdomain_t>& neighbourhood = mesh.mpi_neighbourhood();
            size_t n_neighbours                         = neighbourhood.size();

            // compute fluxes in terms of number of intervals to transfer/receive
            // by default, perform 5 iterations
            // std::vector<int> fluxes = samurai::cmptFluxes<samurai::BalanceElement_t::CELL>( mesh, forceNeighbour, 5 );
            std::vector<int> fluxes = samurai::cmptFluxes<samurai::BalanceElement_t::CELL>(mesh, 5);
            std::vector<int> new_fluxes(fluxes);
            // get loads from everyone
            std::vector<int> loads;
            int my_load = static_cast<int>(samurai::cmptLoad<samurai::BalanceElement_t::CELL>(mesh));
            boost::mpi::all_gather(world, my_load, loads);

            std::vector<CellList_t> cl_to_send(n_neighbours);

            // set field "flags" for each rank. Initialized to current for all cells (leaves only)
            auto flags = samurai::make_scalar_field<int>("diffusion_flag", mesh);
            flags.fill(world.rank());
            // load balancing order
            std::vector<size_t> order(n_neighbours);
            {
                for (size_t i = 0; i < order.size(); ++i)
                {
                    order[i] = i;
                }
                // order neighbour to echange data with, based on load
                std::sort(order.begin(),
                          order.end(),
                          [&fluxes](size_t i, size_t j)
                          {
                              return fluxes[i] < fluxes[j];
                          });
            }

            using cell_t = typename Mesh_t::cell_t;
            std::vector<cell_t> cells;
            cells.reserve(mesh.nb_cells(mesh_id_t::cells));

            samurai::for_each_cell(mesh[mesh_id_t::cells],
                                   [&](auto cell)
                                   {
                                       cells.push_back(cell);
                                   });

            std::sort(cells.begin(),
                      cells.end(),
                      [&](const cell_t& a, const cell_t& b)
                      {
                          auto ca = a.center();
                          auto cb = b.center();
                          if (ca(1) != cb(1))
                          {
                              return ca(1) > cb(1);
                          }
                          else
                          {
                              return ca(0) > cb(0);
                          }
                      });

            int n;
            if (world.size() > 1)
            {
                n = std::abs(fluxes[0]);
            }

            //    std::size_t n = 2000 ;
            //
            //
            std::cout << "flux size : " << fluxes.size() << std::endl;
            std::cout << n << std::endl;

            if (world.size() > 1)
            {
                if (world.rank() == 0)
                {
                    for (int i = 0; i < n && i < cells.size(); ++i)
                    {
                        if (fluxes[0] < 0)
                        {
                            flags[cells[i]] = 1;
                        }
                    }
                }
                else if (world.rank() == world.size() - 1)
                {
                    for (int i = 0; i < n && i < cells.size(); ++i)
                    {
                        if (fluxes[0] < 0)
                        {
                            flags[cells[cells.size() - 1 - i]] = world.rank() - 1;
                        }
                    }
                }
                else
                {
                    int n1 = std::abs(fluxes[0]);
                    int n2 = std::abs(fluxes[1]);
                    for (int i = 0; i < n1 && i < cells.size(); ++i)
                    {
                        if (fluxes[0] < 0)
                        {
                            flags[cells[cells.size() - 1 - i]] = world.rank() - 1;
                        }
                    }

                    for (int i = 0; i < n2 && i < cells.size(); ++i)
                    {
                        if (fluxes[1] < 0)
                        {
                            flags[cells[i]] = world.rank() + 1;
                        }
                    }
                }
            }

            return flags;
        }
    };
}
#endif
