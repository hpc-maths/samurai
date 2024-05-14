#pragma once

#include <algorithm>
#include <cassert>
#include <iostream>
#include <vector>

#include <samurai/algorithm.hpp>
#include <samurai/hilbert.hpp>
#include <samurai/mesh.hpp>
#include <samurai/morton.hpp>
#include <samurai/mr/mesh.hpp>

// statistics
#ifdef WITH_STATS
#include <nlohmann/json.hpp>
#endif

namespace samurai
{

    // namespace load_balance
    // {

    //     template <class Mesh_t, class CellArray_t>
    //     Mesh_t merge(Mesh_t& mesh, const CellArray_t& lca)
    //     {
    //         using cl_type = typename Mesh_t::cl_type;

    //         auto& refmesh = mesh[Mesh_t::mesh_id_t::cells];

    //         auto minlevel = std::min(refmesh.min_level(), lca.min_level());
    //         auto maxlevel = std::max(refmesh.max_level(), lca.max_level());

    //         cl_type cl;
    //         for (size_t ilvl = minlevel; ilvl <= maxlevel; ++ilvl)
    //         {
    //             auto un = samurai::union_(refmesh[ilvl], lca[ilvl]);

    //             un(
    //                 [&](auto& interval, auto& indices)
    //                 {
    //                     cl[ilvl][indices].add_interval(interval);
    //                 });
    //         }

    //         return Mesh_t(cl, minlevel, maxlevel);
    //     }

    //     template <class Mesh_t, class CellArray_t>
    //     Mesh_t remove(Mesh_t& mesh, CellArray_t& lca)
    //     {
    //         using cl_type = typename Mesh_t::cl_type;

    //         auto& refmesh = mesh[Mesh_t::mesh_id_t::cells];

    //         auto minlevel = std::min(refmesh.min_level(), lca.min_level());
    //         auto maxlevel = std::max(refmesh.max_level(), lca.max_level());

    //         // remove cells
    //         cl_type cl;
    //         size_t diff_ncells = 0;
    //         for (size_t ilvl = minlevel; ilvl <= maxlevel; ++ilvl)
    //         {
    //             auto diff = samurai::difference(refmesh[ilvl], lca[ilvl]);

    //             diff(
    //                 [&](auto& interval, auto& index)
    //                 {
    //                     cl[ilvl][index].add_interval(interval);
    //                     diff_ncells += interval.size();
    //                 });
    //         }

    //         // new mesh for current process
    //         return Mesh_t(cl, minlevel, maxlevel);
    //     }

    // }

    struct MPI_Load_Balance
    {
        int32_t _load;
        std::vector<int> neighbour;
        std::vector<int32_t> load;
        std::vector<int32_t> fluxes;
    };

    enum Distance_t
    {
        L1,
        L2,
        LINF,
        GRAVITY
    };

    enum Direction_t
    {
        FACE,
        DIAG,
        FACE_AND_DIAG
    };

    enum BalanceElement_t
    {
        CELL,
        INTERVAL
    };

    enum Weight{
        OnSmall,
        OnLarge,
        None
    };

    /**
     * Compute distance base on different norm.
     */

    template <int dim, class Coord_t>
    static inline double distance_l2(const Coord_t& d1, const Coord_t& d2)
    {
        double dist = 0.;
        for (size_t idim = 0; idim < static_cast<size_t>(dim); ++idim)
        {
            double d = d1(idim) - d2(idim);
            dist += d * d;
        }
        return std::sqrt(dist);
    }

    template <int dim, class Coord_t>
    static inline double distance_inf(const Coord_t& d1, const Coord_t& d2)
    {
        double dist = 0.;
        for (size_t idim = 0; idim < static_cast<size_t>(dim); ++idim)
        {
            dist = std::max(std::abs(d1(idim) - d2(idim)), dist);
        }
        return dist;
    }

    template <int dim, class Coord_t>
    static inline double distance_l1(const Coord_t& d1, const Coord_t& d2)
    {
        double dist = 0.;
        for (size_t idim = 0; idim < static_cast<size_t>(dim); ++idim)
        {
            dist += std::abs(d1(idim) - d2(idim));
        }
        return dist;
    }

    /**
     * Compute the load of the current process based on intervals or cells. It uses the
     * mesh_id_t::cells to only consider leaves.
     */
    template <BalanceElement_t elem, class Mesh_t>
    static std::size_t cmptLoad(const Mesh_t& mesh)
    {
        using mesh_id_t = typename Mesh_t::mesh_id_t;

        const auto& current_mesh = mesh[mesh_id_t::cells];

        std::size_t current_process_load = 0;

        if constexpr (elem == BalanceElement_t::CELL)
        {
            // cell-based load without weight.
            samurai::for_each_interval(current_mesh,
                                       [&]([[maybe_unused]] std::size_t level, const auto& interval, [[maybe_unused]] const auto& index)
                                       {
                                           current_process_load += interval.size();
                                       });
        }
        else
        {
            // interval-based load without weight
            for (std::size_t level = current_mesh.min_level(); level <= current_mesh.max_level(); ++level)
            {
                current_process_load += current_mesh[level].shape()[0]; // only in x-axis ;
            }
        }

        return current_process_load;
    }

    /**
     * Compute fluxes based on load computing stategy based on graph with label
     * propagation algorithm. Return, for the current process, the flux in term of
     * load, i.e. the quantity of "load" to transfer to its neighbours. If the load
     * is negative, it means that the process (current) must send load to neighbour,
     * if positive it means that it must receive load.
     *
     * This function use 2 MPI all_gather calls.
     *
     */
    template <BalanceElement_t elem, class Mesh_t>
    std::vector<int> cmptFluxes(Mesh_t& mesh)
    {
        using mpi_subdomain_t = typename Mesh_t::mpi_subdomain_t;

        boost::mpi::communicator world;

        // give access to geometricaly neighbour process rank and mesh
        std::vector<mpi_subdomain_t>& neighbourhood = mesh.mpi_neighbourhood();
        size_t n_neighbours                         = neighbourhood.size();

        // load of current process
        int my_load = static_cast<int>(cmptLoad<elem>(mesh));

        // fluxes between processes
        std::vector<int> fluxes(n_neighbours, 0);

        // load of each process (all processes not only neighbours)
        std::vector<int> loads;

        // numbers of neighbours processes for each process, used for weighting fluxes
        std::vector<size_t> neighbourhood_n_neighbours;

        // get "my_load" from other processes
        boost::mpi::all_gather(world, my_load, loads);
        boost::mpi::all_gather(world, neighbourhood.size(), neighbourhood_n_neighbours);

        // compute updated my_load for current process based on its neighbourhood
        int my_load_new = my_load;
        for (std::size_t n_i = 0; n_i < n_neighbours; ++n_i)
        {
            std::size_t neighbour_rank = static_cast<std::size_t>(neighbourhood[n_i].rank);
            int neighbour_load         = loads[neighbour_rank];
            double diff_load           = static_cast<double>(neighbour_load - my_load);

            std::size_t nb_neighbours_neighbour = neighbourhood_n_neighbours[neighbour_rank];

            double weight = 1. / static_cast<double>(std::max(n_neighbours, nb_neighbours_neighbour) + 1);

            int transfertLoad = static_cast<int>(std::lround(weight * diff_load));

            fluxes[n_i] += transfertLoad;

            my_load_new += transfertLoad;
        }

        return fluxes;
    }

    /** Is gravity the key ? That would be awesome =)
     * This is not a distance but well does not change anything to the algorithm
     * - G m m' / R ?
     *
     * What should be m ? m' ? Let's G = 1.
     *
     **/
    template <int dim, class Coord_t>
    static inline double gravity(const Coord_t& d1, const Coord_t& d2)
    {
        double dist = distance_l2<dim>(d1, d2);

        // makes high level ( smallest ) cells to be exchanged more easily. 10 here is max
        // level, hardcoded for tests.
        // double m = 1./ static_cast<double>( 1 << ( 10 - d1.level ) ) , m_p = 1.;

        // makes bigger cells to be exchanged more easily
        // double m = 1./ static_cast<double>( 1 << d1.level ) , m_p = 1.;
        constexpr double G = 1.;
        double m = 1., m_p = 1.;
        double f = -G * m * m_p / (dist * dist);

        return f;
    }

    // we are using Cell_t to allow ponderation using level;
    template <int dim, Distance_t dist, class Coord_t>
    inline constexpr double getDistance(const Coord_t& cc, const Coord_t& d)
    {
        static_assert(dim == 2 || dim == 3);
        if constexpr (dist == Distance_t::L1)
        {
            return distance_l1<dim>(cc, d);
        }
        else if constexpr (dist == Distance_t::L2)
        {
            return distance_l2<dim>(cc, d);
        }
        else if constexpr (dist == Distance_t::LINF)
        {
            return distance_inf<dim>(cc, d);
        }
        else if constexpr (dist == Distance_t::GRAVITY)
        {
            return gravity<dim>(cc, d);
        }
    }

    template <class Flavor>
    class LoadBalancer
    {
      private:

        template <class Mesh_t, class Field_t>
        void update_field(Mesh_t& new_mesh, Field_t& field) const
        {
            using mesh_id_t = typename Mesh_t::mesh_id_t;
            using value_t   = typename Field_t::value_type;
            boost::mpi::communicator world;

            std::ofstream logs; 
            logs.open( "log_" + std::to_string( world.rank() ) + ".dat", std::ofstream::app );
            logs << fmt::format("> [LoadBalancer]::update_field rank # {} -> '{}' ", world.rank(), field.name() ) << std::endl;

            Field_t new_field("new_f", new_mesh);
            new_field.fill(0);

            auto & old_mesh = field.mesh();

            // auto min_level = boost::mpi::all_reduce(world, mesh[mesh_id_t::cells].min_level(), boost::mpi::minimum<std::size_t>());
            // auto max_level = boost::mpi::all_reduce(world, mesh[mesh_id_t::cells].max_level(), boost::mpi::maximum<std::size_t>());

            auto min_level = old_mesh.min_level();
            auto max_level = old_mesh.max_level();

            // copy data of intervals that are didn't move
            for (std::size_t level = min_level; level <= max_level; ++level)
            {
                auto intersect_old_new = intersection(old_mesh[mesh_id_t::cells][level], new_mesh[mesh_id_t::cells][level]);
                intersect_old_new.apply_op( samurai::copy( new_field, field ) );
            }

            logs << fmt::format("> [LoadBalancer]::update_field rank # {}: data copied for intersection old/new ", world.rank() ) << std::endl;

            std::vector<boost::mpi::request> req;
            std::vector<std::vector<value_t>> to_send(new_mesh.mpi_neighbourhood().size());

            std::size_t i_neigh = 0;

            // build payload of field that has been sent to neighbour, so compare old mesh with new neighbour mesh 
            for (auto& neighbour : new_mesh.mpi_neighbourhood())
            {
                auto & neighbour_new_mesh = neighbour.mesh;
                for (std::size_t level = min_level; level <= max_level; ++level)
                {
                    if (!old_mesh[mesh_id_t::cells][level].empty() && !neighbour_new_mesh[mesh_id_t::cells][level].empty())
                    {
                        auto intersect_old_mesh_new_neigh = intersection( old_mesh[mesh_id_t::cells][level], neighbour_new_mesh[mesh_id_t::cells][level] );
                        intersect_old_mesh_new_neigh(
                            [&](const auto & interval, const auto & index)
                            {   
                                std::copy(field(level, interval, index).begin(), field(level, interval, index).end(), std::back_inserter(to_send[i_neigh]));
                            });
                    }
                }

                if (to_send[i_neigh].size() != 0)
                {
                    req.push_back( world.isend( neighbour.rank, neighbour.rank, to_send[ i_neigh ] ) );
                    i_neigh ++;

                    logs << fmt::format("> [LoadBalancer]::update_field rank # {}: data to send to {}", world.rank(), neighbour.rank ) << std::endl;
                }
            }

            logs << fmt::format("> [LoadBalancer]::update_field rank # {}, nb req isend {}", world.rank(), req.size() ) << std::endl;

            // build payload of field that I need to receive from neighbour, so compare NEW mesh with OLD neighbour mesh 
            for (auto& old_neighbour : old_mesh.mpi_neighbourhood())
            {
                bool isintersect = false;
                for (std::size_t level = min_level; level <= max_level; ++level)
                {
                    if (!new_mesh[mesh_id_t::cells][level].empty() && !old_neighbour.mesh[mesh_id_t::cells][level].empty())
                    {
                        std::vector<value_t> to_recv;
                        std::ptrdiff_t count = 0;

                        auto in_interface = intersection(old_neighbour.mesh[mesh_id_t::cells][level], new_mesh[mesh_id_t::cells][level]);

                        in_interface(
                            [&]( [[maybe_unused]]const auto& i, [[maybe_unused]]const auto& index)
                            {
                                isintersect = true;
                            });

                        if (isintersect)
                        {
                            break;
                        }
                    }
                }

                if (isintersect)
                {
                    std::ptrdiff_t count = 0;
                    std::vector<value_t> to_recv;
                    world.recv(old_neighbour.rank, world.rank(), to_recv);

                    for (std::size_t level = min_level; level <= max_level; ++level)
                    {
                        if (!new_mesh[mesh_id_t::cells][level].empty() && !old_neighbour.mesh[mesh_id_t::cells][level].empty())
                        {
                            auto in_interface = intersection(old_neighbour.mesh[mesh_id_t::cells][level], new_mesh[mesh_id_t::cells][level]);

                            in_interface(
                                [&](const auto& i, const auto& index)
                                {
                                    std::copy(to_recv.begin() + count,
                                              to_recv.begin() + count + static_cast<ptrdiff_t>(i.size()),
                                              new_field(level, i, index).begin());
                                    count += static_cast<ptrdiff_t>(i.size());

                                    // std::cerr << fmt::format("Process {}, recv interval {}", world.rank(), i) << std::endl;
                                });
                        }
                    }
                }
            }

            if (!req.empty())
            {
                mpi::wait_all(req.begin(), req.end());
            }

            std::swap(field.array(), new_field.array());
        }

        template <class Mesh_t, class Field_t, class... Fields_t>
        void update_fields(Mesh_t& new_mesh, Field_t& field, Fields_t&... kw) const

        {
            update_field(new_mesh, field);
            update_fields(new_mesh, kw...);
        }

        template <class Mesh_t>
        void update_fields([[maybe_unused]]Mesh_t& new_mesh) const
        {
        }

      public:

        template <class Mesh_t, class Field_t, class... Fields>
        void load_balance(Mesh_t & mesh, Field_t& field, Fields&... kw)
        {
            // specific load balancing strategy
            auto new_mesh = static_cast<Flavor*>(this)->load_balance_impl(mesh);

            // update each physical field on the new load balanced mesh
            SAMURAI_TRACE("[LoadBalancer::load_balance]::Updating fields ... ");
            update_fields(new_mesh, field, kw...);

            // swap mesh reference to new load balanced mesh. FIX: this is not clean
            SAMURAI_TRACE("[LoadBalancer::load_balance]::Swapping meshes ... ");
            field.mesh().swap(new_mesh);

            // discover neighbours: add new neighbours if a new interface appears or remove old neighbours
            // FIX: add boolean return to condition the need of another call, might save some MPI comm.
            discover_neighbour( new_mesh );
            discover_neighbour( new_mesh );
        }

        /**
        * This function MUST be used for debug or analysis purposes of load balancing strategy,
        * it involves a lots of MPI communications.
        *
        * Try to evaluate / compute a load balancing score. We expect from a good load 
        * balancing strategy to:
        *   - optimize the number of neighbours   (reduce comm.)
        *   - optimize the number of ghosts cells (reduce comm.)
        *   - load balance charge between processes
        *   - optimize the size of interval (samurai specific, expect better perf, better simd)
        */
        template <class Mesh>
        void evaluate_balancing(Mesh& mesh) const
        {
            boost::mpi::communicator world;

            if (world.rank() == 0)
            {
                SAMURAI_TRACE("[LoadBalancer::evaluate_balancing]::Entering function ... ");
            }

            std::vector<int> load_cells, load_interval;
            {
                int my_load_i = static_cast<int>(cmptLoad<samurai::BalanceElement_t::INTERVAL>(mesh));
                boost::mpi::all_gather(world, my_load_i, load_interval);

                int my_load_c = static_cast<int>(cmptLoad<samurai::BalanceElement_t::CELL>(mesh));
                boost::mpi::all_gather(world, my_load_c, load_cells);
            }

            // if (world.rank() == 0)
            // {
            //     std::cerr << "\t> LoadBalancer statistics : " << std::endl;

            //     std::vector<int>::iterator min_load = std::min_element(load_cells.begin(), load_cells.end());
            //     auto rank                           = std::distance(load_cells.begin(), min_load);
            //     std::cerr << "\t\t> Min load {" << *min_load << ", cells } @ rank # " << rank << std::endl;

            //     std::vector<int>::iterator max_load = std::max_element(load_cells.begin(), load_cells.end());
            //     rank                                = std::distance(load_cells.begin(), max_load);
            //     std::cerr << "\t\t> Max load {" << *max_load << ", cells } @ rank # " << rank << std::endl;
            // }

            // std::string _stats = fmt::format("statistics_process_{}", world.rank());
            // samurai::Statistics s(_stats);

            // s("stats", mesh);

            // s( "statistics" , mesh );

            // no need to implement this in derived class, could be general.
            //
            // static_cast<Flavor *>(this)->evaluate_impl( mesh, kw... );
        }
    };

    /**
     * Precomputed direction to face-to-face element for both 3D and 2D
     */
    template <size_t dim>
    constexpr auto getDirectionFace()
    {
        using base_stencil = xt::xtensor_fixed<int, xt::xshape<dim>>;

        constexpr size_t size_ = 2 * dim;
        xt::xtensor_fixed<base_stencil, xt::xshape< size_ >> stencils;
        std::size_t nstencils = size_;
        for (std::size_t ist = 0; ist < nstencils; ++ist)
        {
            stencils[ist].fill(0);
            stencils[ist][ist / 2] = 1 - (ist % 2) * 2;
        }
        // if constexpr(  dim == 2 ){
        //     return xt::xtensor_fixed<base_stencil, xt::xshape<4>> {{{1, 0}, {-1, 0}, {0, 1}, {0, -1}}};
        // }

        return stencils;
    }

    /**
     * Precomputed direction to diagonal element for both 3D and 2D cases.
     */
    template <size_t dim>
    constexpr auto getDirectionDiag()
    {
        using base_stencil = xt::xtensor_fixed<int, xt::xshape<dim>>;

        static_assert(dim == 2 || dim == 3);

        if constexpr (dim == 2)
        {
            return xt::xtensor_fixed<base_stencil, xt::xshape<4>>{
                {{1, 1}, {1, -1}, {-1, 1}, {-1, -1}}
            };
        }

        if constexpr (dim == 3)
        {
            return xt::xtensor_fixed<base_stencil, xt::xshape<20>>{
                {{1, 1, -1},  {1, -1, -1}, {-1, 1, -1}, {-1, -1, -1}, {1, 0, -1},  {-1, 0, -1}, {0, 1, -1},
                 {0, -1, -1}, {1, 1, 0},   {1, -1, 0},  {-1, 1, 0},   {-1, -1, 0}, {1, 1, 1},   {1, -1, 1},
                 {-1, 1, 1},  {-1, -1, 1}, {1, 0, 1},   {-1, 0, 1},   {0, 1, 1},   {0, -1, 1}}
            };
        }
    }

    /**
     * Precompute direction to element in all direction face + diagonals: 26 in 3D, 8 in 2D
     */
    template <int dim>
    constexpr auto getDirectionFaceAndDiag()
    {
        using base_stencil = xt::xtensor_fixed<int, xt::xshape<dim>>;

        static_assert(dim == 2 || dim == 3);

        if constexpr (dim == 2)
        {
            return xt::xtensor_fixed<base_stencil, xt::xshape<8>>{
                {{-1, 0}, {1, 0}, {1, 1}, {1, -1}, {0, -1}, {0, 1}, {-1, 1}, {-1, -1}}
            };
        }

        if constexpr (dim == 3)
        {
            return xt::xtensor_fixed<base_stencil, xt::xshape<26>>{
                {{1, 0, 0},    {-1, 0, 0}, {0, 1, 0},   {0, -1, 0},  {0, 0, 1},   {0, 0, -1}, {1, 1, -1}, {1, -1, -1}, {-1, 1, -1},
                 {-1, -1, -1}, {1, 0, -1}, {-1, 0, -1}, {0, 1, -1},  {0, -1, -1}, {1, 1, 0},  {1, -1, 0}, {-1, 1, 0},  {-1, -1, 0},
                 {1, 1, 1},    {1, -1, 1}, {-1, 1, 1},  {-1, -1, 1}, {1, 0, 1},   {-1, 0, 1}, {0, 1, 1},  {0, -1, 1}}
            };
        }
    }

    template <int dim, Direction_t dir>
    inline auto getDirection()
    {
        static_assert(dim == 2 || dim == 3);
        if constexpr (dir == Direction_t::FACE)
        {
            return getDirectionFace<dim>();
        }
        else if constexpr (dir == Direction_t::DIAG)
        {
            return getDirectionDiag<dim>();
        }
        else if constexpr (dir == Direction_t::FACE_AND_DIAG)
        {
            return getDirectionFaceAndDiag<dim>();
        }
    }

    /**
     * Compute the barycenter of the mesh given in parameter.
     *
     * Feat: adjust wght which is hardcoded to 1. for now.
     *       by passing in parameters an array for example to modulate
     *       weight according to level
     */
    template <int dim, class Mesh_t, Weight w=Weight::None>
    xt::xtensor_fixed<double, xt::xshape<dim>> _cmpCellBarycenter(Mesh_t& mesh)
    {
        using Coord_t = xt::xtensor_fixed<double, xt::xshape<dim>>;

        Coord_t bary;
        bary.fill(0.);

        double wght_tot = 0.;
        samurai::for_each_cell(mesh,
                               [&](auto& cell)
                               {
                                   double wght = 1.;
                                   
                                   if constexpr( w == Weight::OnSmall ){
                                    wght = 1. / static_cast<double>( 1 << (mesh.max_level() - cell.level) );
                                   }

                                   if constexpr( w == Weight::OnLarge ){
                                    wght = 1. / static_cast<double>( 1 << cell.level );
                                   }

                                   auto cc = cell.center();

                                   for (int idim = 0; idim < dim; ++idim)
                                   {
                                       bary(idim) += cc(idim) * wght;
                                   }

                                   wght_tot += wght;
                               });

        wght_tot = std::max(wght_tot, 1e-12);
        for (int idim = 0; idim < dim; ++idim)
        {
            bary(idim) /= wght_tot;
        }

        return bary;
    }



    /**
     *
     * Params:
     *          leveldiff : max level difference. For 2:1 balance, leveldiff = 1
     */
    template <size_t dim, Direction_t dir, size_t leveldiff = 1, class Mesh_t>
    static auto cmptInterface(Mesh_t& mesh, Mesh_t& omesh)
    {
        using CellList_t  = typename Mesh_t::cl_type;
        using CellArray_t = typename Mesh_t::ca_type;
        using mesh_id_t   = typename Mesh_t::mesh_id_t;

        CellList_t interface;

        // operation are on leaves only
        auto& currentMesh = mesh[mesh_id_t::cells];
        auto& otherMesh   = omesh[mesh_id_t::cells];

        // direction for translation to cmpt interface
        auto dirs = getDirection<dim, dir>();

        for (size_t ist = 0; ist < dirs.size(); ++ist)
        {
            const auto& stencil = dirs[ist];

            size_t minlevel = otherMesh.min_level();
            size_t maxlevel = otherMesh.max_level();

            for (size_t level = minlevel; level <= maxlevel; ++level)
            {
                // for each level we need to check level -1 / 0 / +1
                std::size_t minlevel_check = static_cast<std::size_t>(
                    std::max(static_cast<int>(currentMesh.min_level()), static_cast<int>(level) - static_cast<int>(leveldiff)));
                std::size_t maxlevel_check = std::min(currentMesh.max_level(), level + leveldiff);

                for (size_t projlevel = minlevel_check; projlevel <= maxlevel_check; ++projlevel)
                {
                    // translate neighbour from dir (hopefully to current) all direction are tested
                    auto set       = translate(otherMesh[level], stencil);
                    auto intersect = intersection(set, currentMesh[projlevel]).on(projlevel);

                    size_t nbInter_ = 0, nbCells = 0;
                    intersect(
                        [&](auto& interval, [[maybe_unused]] auto& index)
                        {
                            nbInter_ += 1;
                            nbCells += interval.size();
                        });

                    // we get more interval / cells, than wanted because neighbour has bigger cells
                    // 2:1 balance required here if not, need another loop...
                    if (nbInter_ > 0 && projlevel > level)
                    {
                        static_assert(leveldiff == 1); // for now at least
                        auto set_  = translate(currentMesh[projlevel], stencil);
                        auto diff_ = difference(intersect, set_);

                        diff_(
                            [&](auto& interval, auto& index)
                            {
                                interface[projlevel][index].add_interval(interval);
                            });
                    }
                    else
                    {
                        if (nbInter_ > 0)
                        {
                            intersect(
                                [&](auto& interval, auto& index)
                                {
                                    interface[projlevel][index].add_interval(interval);
                                });
                        }
                    }
                }
            }
        }

        CellArray_t interface_ = {interface, false};

        return interface_;
    }

    /**
     * Compute cells at the interface between geometricaly adjacent
     * domains. It relies on neighbourhood mpi_subdomain_t data structure
     * computed in Mesh_t.
     *
     * It returns a vector containing the number of cells (nc) and number of
     * intervals (ni) for each neighbour (0 to n), in that order.
     *
     * Example: nc_0, ni_0, nc_1, ni_1, ...
     *
     */
    // template<int dim, class Mesh_t, class Field_t>
    // void _computeCartesianInterface( Mesh_t & mesh, Field_t & field ){
    template <int dim, Direction_t dir, class Mesh_t>
    static auto _computeCartesianInterface(Mesh_t& mesh)
    {
        using CellList_t      = typename Mesh_t::cl_type;
        using CellArray_t     = typename Mesh_t::ca_type;
        using mesh_id_t       = typename Mesh_t::mesh_id_t;
        using mpi_subdomain_t = typename Mesh_t::mpi_subdomain_t;

        // give access to geometricaly neighbour process rank and mesh
        std::vector<mpi_subdomain_t>& neighbourhood = mesh.mpi_neighbourhood();
        std::size_t n_neighbours                    = neighbourhood.size();

        std::vector<CellList_t> interface(n_neighbours);

        // operation are on leaves only
        auto currentMesh = mesh[mesh_id_t::cells];

        // looks like using only FACE is better than using DIAG or FACE_AND_DIAG
        auto dirs = getDirection<dim, dir>();

        for (std::size_t nbi = 0; nbi < n_neighbours; ++nbi)
        {
            auto neighbour_mesh = neighbourhood[nbi].mesh[mesh_id_t::cells];

            for (size_t ist = 0; ist < dirs.size(); ++ist)
            {
                const auto& stencil = dirs[ist];

                size_t minlevel = neighbour_mesh.min_level();
                size_t maxlevel = neighbour_mesh.max_level();

                for (size_t level = minlevel; level <= maxlevel; ++level)
                {
                    // for each level we need to check level -1 / 0 / +1
                    std::size_t minlevel_check = static_cast<std::size_t>(
                        std::max(static_cast<int>(currentMesh.min_level()), static_cast<int>(level) - 1));
                    std::size_t maxlevel_check = std::min(currentMesh.max_level(), level + static_cast<std::size_t>(1));

                    for (std::size_t projlevel = minlevel_check; projlevel <= maxlevel_check; ++projlevel)
                    {
                        // translate neighbour from dir (hopefully to current) all direction are tested
                        auto set       = translate(neighbour_mesh[level], stencil);
                        auto intersect = intersection(set, currentMesh[projlevel]).on(projlevel);

                        size_t nbInter_ = 0, nbCells_ = 0;
                        intersect(
                            [&](auto& interval, [[maybe_unused]] auto& index)
                            {
                                nbInter_ += 1;
                                nbCells_ += interval.size();
                            });

                        // we get more interval / cells, than wanted because neighbour has bigger cells
                        if (nbInter_ > 0 && projlevel > level)
                        {
                            auto set_  = translate(currentMesh[projlevel], stencil);
                            auto diff_ = difference(intersect, set_);

                            nbInter_ = 0;
                            nbCells_ = 0;
                            diff_(
                                [&](auto& interval, auto& index)
                                {
                                    nbInter_ += 1;
                                    nbCells_ += interval.size();
                                    // field( projlevel, i, index ) = world.rank() * 10;
                                    interface[nbi][projlevel][index].add_interval(interval);
                                });
                        }
                        else
                        {
                            if (nbInter_ > 0)
                            {
                                intersect(
                                    [&](auto& interval, auto& index)
                                    {
                                        interface[nbi][projlevel][index].add_interval(interval);
                                    });
                            }
                        }
                    }
                }
            }
        }

        std::vector<CellArray_t> interface_(n_neighbours);
        for (std::size_t nbi = 0; nbi < n_neighbours; ++nbi)
        {
            interface_[nbi] = {interface[nbi], true};
        }

        return interface_;
    }

    /**
     * Compute fluxes of cells between MPI processes. In -fake- MPI environment. To
     * use it in true MPI juste remove the loop over "irank", and replace irank by myrank;
     *
     */
    void compute_load_balancing_fluxes(std::vector<MPI_Load_Balance>& all)
    {
        for (size_t irank = 0; irank < all.size(); ++irank)
        {
            // number of cells
            // supposing each cell has a cost of 1. ( no level dependency )
            int32_t load = all[irank]._load;

            std::size_t n_neighbours = all[irank].neighbour.size();

            {
                std::cerr << "[compute_load_balancing_fluxes] Process # " << irank << " load : " << load << std::endl;
                std::cerr << "[compute_load_balancing_fluxes] Process # " << irank << " nneighbours : " << n_neighbours << std::endl;
                std::cerr << "[compute_load_balancing_fluxes] Process # " << irank << " neighbours : ";
                for (size_t in = 0; in < all[irank].neighbour.size(); ++in)
                {
                    std::cerr << all[irank].neighbour[in] << ", ";
                }
                std::cerr << std::endl;
            }

            // load of each process (all processes not only neighbour)
            std::vector<int64_t> loads;

            // data "load" to transfer to neighbour processes
            all[irank].fluxes.resize(n_neighbours);
            std::fill(all[irank].fluxes.begin(), all[irank].fluxes.end(), 0);

            const std::size_t n_iterations = 1;

            for (std::size_t k = 0; k < n_iterations; ++k)
            {
                // numbers of neighboors processes for each neighbour process
                std::vector<std::size_t> nb_neighbours;

                if (irank == 0)
                {
                    std::cerr << "[compute_load_balancing_fluxes] Fluxes iteration # " << k << std::endl;
                }

                // // get info from processes
                // mpi::all_gather(world, load, loads);
                // mpi::all_gather(world, m_mpi_neighbourhood.size(), nb_neighbours);

                // load of current process
                int32_t load_np1 = load;

                // compute updated load for current process based on its neighbourhood
                for (std::size_t j_rank = 0; j_rank < n_neighbours; ++j_rank)
                {
                    auto neighbour_rank = static_cast<std::size_t>(all[irank].neighbour[j_rank]);
                    auto neighbour_load = all[irank].load[j_rank];
                    auto diff_load      = neighbour_load - load;

                    std::size_t nb_neighbours_neighbour = all[neighbour_rank].neighbour.size();

                    double weight = 1. / static_cast<double>(std::max(n_neighbours, nb_neighbours_neighbour) + 1);

                    int32_t transfertLoad = static_cast<int32_t>(std::lround(weight * static_cast<double>(diff_load)));

                    all[irank].fluxes[j_rank] += transfertLoad;

                    load_np1 += transfertLoad;
                }

                // do check on load & fluxes ?

                {
                    std::cerr << "fluxes : ";
                    for (size_t in = 0; in < n_neighbours; ++in)
                    {
                        std::cerr << all[irank].fluxes[in] << ", ";
                    }
                    std::cerr << std::endl;
                }

                // load_transfer( load_fluxes );

                load = load_np1;
            }
        }
    }

    /**
     * Check if there is an intersection between two meshes. We move one mesh into the direction
     * of the other based on "dir" stencils (template parameter). Here, we rely on the 2:1 balance.
     * By default, stencils are "1" based, which means that we move the mesh by one unit element. This
     * could be changed if necessary by multiplying the stencils by an integer value.
     */
    template <Direction_t dir, class Mesh_t>
    bool intersectionExists(Mesh_t& meshA, Mesh_t& meshB)
    {
        using mesh_id_t = typename Mesh_t::mesh_id_t;
        
        constexpr size_t dim = Mesh_t::dim;

        // operation are on leaves cells only
        auto meshA_ = meshA[mesh_id_t::cells];
        auto meshB_ = meshB[mesh_id_t::cells];

        // get stencils for direction
        auto dirs = getDirection<dim, dir>();

        for (size_t ist = 0; ist < dirs.size(); ++ist)
        {
            const auto& stencil = dirs[ist];

            size_t minlevel = meshB_.min_level();
            size_t maxlevel = meshB_.max_level();

            for (size_t level = minlevel; level <= maxlevel; ++level)
            {
                // for each level we need to check level -1 / 0 / +1
                std::size_t minlevel_check = static_cast<std::size_t>(
                    std::max(static_cast<int>(meshA_.min_level()), static_cast<int>(level) - 1));
                std::size_t maxlevel_check = std::min(meshA_.max_level(), level + static_cast<std::size_t>(1));

                for (std::size_t projlevel = minlevel_check; projlevel <= maxlevel_check; ++projlevel)
                {
                    // translate meshB_ in "dir" direction
                    auto set       = translate(meshB_[level], stencil);
                    auto intersect = intersection(set, meshA_[projlevel]).on(projlevel);

                    size_t nbInter_ = 0;
                    intersect(
                        [&]([[maybe_unused]] auto& interval, [[maybe_unused]] auto& index)
                        {
                            nbInter_ += 1;
                        });

                    if (nbInter_ > 0)
                    {
                        return true;
                    }
                }
            }
        }

        return false;
    }

    /**
     * Discover new neighbour connection that might arise during load balancing
     */
    template <class Mesh_t>
    void discover_neighbour(Mesh_t& mesh)
    {
        using mpi_subdomain_t = typename Mesh_t::mpi_subdomain_t;

        boost::mpi::communicator world;

        // DEBUG
        std::ofstream logs;
        logs.open("log_" + std::to_string(world.rank()) + ".dat", std::ofstream::app);
        logs << "# discover_neighbour" << std::endl;

        // give access to geometricaly neighbour process rank and mesh
        std::vector<mpi_subdomain_t>& neighbourhood = mesh.mpi_neighbourhood();

        // bool requireGeneralUpdate = false;

        std::vector<bool> keepNeighbour(neighbourhood.size(), true);

        std::vector<std::vector<int>> neighbourConnection(neighbourhood.size());

        // for each neighbour we check if there is an intersection with another one.
        for (size_t nbi = 0; nbi < neighbourhood.size(); ++nbi)
        {
            // check current - neighbour connection
            keepNeighbour[nbi] = intersectionExists<Direction_t::FACE_AND_DIAG>(mesh, neighbourhood[nbi].mesh);

            // require update if a connection is lost
            // requireGeneralUpdate = requireGeneralUpdate || ( ! keepNeighbour[ nbi ] );

            if (!keepNeighbour[nbi])
            {
                logs << fmt::format("Loosing neighbour connection with {}", neighbourhood[nbi].rank) << std::endl;
            }

            // check neighbour - neighbour connection
            for (size_t nbj = nbi + 1; nbj < neighbourhood.size(); ++nbj)
            {
                logs << fmt::format("Checking neighbourhood connection {} <-> {}", neighbourhood[nbi].rank, neighbourhood[nbj].rank)
                     << std::endl;
                bool connected = intersectionExists<Direction_t::FACE_AND_DIAG>(neighbourhood[nbi].mesh, neighbourhood[nbj].mesh);

                if (connected)
                {
                    neighbourConnection[nbi].emplace_back(neighbourhood[nbj].rank);
                    neighbourConnection[nbj].emplace_back(neighbourhood[nbi].rank);
                }
            }
        }

        // communicate to neighbours the list of connection they have
        for (size_t nbi = 0; nbi < neighbourhood.size(); ++nbi)
        {
            // debug
            logs << "\t> Sending computed neighbour: {";
            for (const auto& i : neighbourConnection[nbi])
            {
                logs << i << ", ";
            }
            logs << "} to process " << neighbourhood[nbi].rank << std::endl;
            // end debug

            world.send(neighbourhood[nbi].rank, 28, neighbourConnection[nbi]);
        }

        // map of current MPI neighbours processes rank
        std::map<int, bool> _tmp; // key = rank, value = required
        for (size_t nbi = 0; nbi < neighbourhood.size(); ++nbi)
        {
            if (keepNeighbour[nbi])
            {
                _tmp.emplace(std::make_pair(neighbourhood[nbi].rank, true));
            }
        }

        logs << "\t> Receiving computed neighbour connection list from neighbour processes " << std::endl;
        for (size_t nbi = 0; nbi < neighbourhood.size(); ++nbi)
        {
            std::vector<int> _rcv_neighbour_list;
            world.recv(neighbourhood[nbi].rank, 28, _rcv_neighbour_list);

            for (size_t in = 0; in < _rcv_neighbour_list.size(); ++in)
            {
                if (_tmp.find(_rcv_neighbour_list[in]) == _tmp.end())
                {
                    logs << "\t\t> New neighbour detected : " << _rcv_neighbour_list[in] << std::endl;

                    _tmp[_rcv_neighbour_list[in]] = true;
                    // requireGeneralUpdate = requireGeneralUpdate || true;
                }
            }
        }

        world.barrier();

        // update neighbours ranks
        neighbourhood.clear();
        for (const auto& ni : _tmp)
        {
            neighbourhood.emplace_back(ni.first);
        }

        // debug
        logs << "New neighbourhood : {";
        for (size_t nbi = 0; nbi < neighbourhood.size(); ++nbi)
        {
            logs << neighbourhood[nbi].rank << ", ";
        }
        logs << "}" << std::endl;
        // end debug

        // gather neighbour mesh
        mesh.update_mesh_neighbour();

        // return requireGeneralUpdate;
    }

} // namespace samurai

/**
 * This function perform a "fake" load-balancing by updating an integer scalar field containing the rank
 * of the cell after the load balancing; This is done using MORTON SFC.
 *
 * This does not require to have the graph.
 */
// template<int dim, class SFC_t, class Mesh, class Field_t>
// void perform_load_balancing_SFC( Mesh & mesh, int ndomains, Field_t & fake_mpi_rank ) {

//     using Config = samurai::MRConfig<dim>;
//     using Mesh_t    = samurai::MRMesh<Config>;
//     using cell_t    = typename Mesh::cell_t;

//     SFC<SFC_t> sfc;

//     // SFC key (used for implicit sorting through map mechanism)
//     std::map<SFC_key_t, cell_t> sfc_map;

//     int sfc_max_level = mesh.max_level(); // for now but remember must <= 21 for Morton

//     SFC_key_t min=std::numeric_limits<SFC_key_t>::max(), max=std::numeric_limits<SFC_key_t>::min();

//     samurai::for_each_cell( mesh, [&]( const auto & cell ){

//         // ij coordinate of cell
//         double dxmax = samurai::cell_length( sfc_max_level );

//         auto tmp = cell.center() / dxmax;

//         xt::xtensor_fixed<uint32_t, xt::xshape<dim>> ij = { static_cast<uint32_t>( tmp( 0 ) ),
//                                                             static_cast<uint32_t>( tmp( 1 ) ) };

//         auto key = sfc.template getKey<dim>( ij );
//         // std::cerr << "\t> Coord (" << ij( 0 ) << ", " << ij( 1 ) << ") ----> " << key << std::endl;

//         sfc_map[ key ] = cell ;

//     });

//     size_t nbcells_tot   = mesh.nb_cells( Mesh::mesh_id_t::cells ); //load -balancing on leaves
//     size_t ncellsPerProc = std::floor( nbcells_tot / ndomains );

//     // std::cerr << "\n\t> Morton index [" << min << ", " << max << "]" << std::endl;
//     std::cerr << "\t> Total number of cells : " << nbcells_tot << std::endl;
//     std::cout << "\t> Perfect load-balancing (weight-cell = 1.) : " << static_cast<size_t>( nbcells_tot / ndomains )
//               << " / MPI" << std::endl;

//     size_t cindex = 0;
//     for(const auto & item : sfc_map ) {
//         auto sfc_key = item.first;

//         int fake_rank = std::floor( cindex / ncellsPerProc );
//         if ( fake_rank >= ndomains ) fake_rank = ndomains - 1;

//         fake_mpi_rank[ item.second ] = fake_rank;
//         cindex ++;
//     }

// }

// enum Interval_CellPosition { FIRST, LAST };

// template<int dim, class Mesh, class Field_t>
// void perform_load_balancing_SFC_Interval( Mesh & mesh, int ndomains, Field_t & fake_mpi_rank, const Interval_CellPosition &icp ) {

//     using inter_t = samurai::Interval<int, long long>;

//     struct Data {
//         size_t level;
//         inter_t interval;
//         xt::xtensor_fixed<samurai::default_config::value_t, xt::xshape<dim - 1>> indices;
//     };

//     // SFC key (used for implicit sorting through map mechanism)
//     std::map<SFC_key_t, Data> sfc_map;

//     SFC<Morton> sfc;
//     int sfc_max_level = mesh.max_level(); // for now but remember must <= 21 for Morton

//     SFC_key_t min=std::numeric_limits<SFC_key_t>::max(), max=std::numeric_limits<SFC_key_t>::min();

//     size_t ninterval = 0;
//     samurai::for_each_interval( mesh, [&]( std::size_t level, const auto& inter, const auto& index ){

//         // get Logical coordinate or first cell
//         xt::xtensor_fixed<int, xt::xshape<dim>> icell;

//         icp == Interval_CellPosition::LAST ? icell( 0 ) = inter.end - 1 : icell( 0 ) = inter.start;

//         for(int idim=0; idim<dim-1; ++idim){
//             icell( idim + 1 ) = index( idim );
//         }

//         // convert logical coordinate to max level logical coordinates
//         for( int idim=0; idim<dim; ++idim ){
//             icell( idim ) = icell( idim ) << ( mesh.max_level() - level );
//         }

//         xt::xtensor_fixed<uint32_t, xt::xshape<dim>> ijk;

//         if constexpr ( dim == 2 ) ijk = { static_cast<uint32_t>( icell( 0 ) ),
//                                           static_cast<uint32_t>( icell( 1 ) ) };

//         if constexpr ( dim == 3 ) ijk = { static_cast<uint32_t>( icell( 0 ) ),
//                                           static_cast<uint32_t>( icell( 1 ) ),
//                                           static_cast<uint32_t>( icell( 2 ) ) };

//         sfc_map[ sfc.getKey<dim>( ijk ) ] = { level, inter, index };

//         ninterval ++;
//     });

//     size_t ninterPerProc = std::floor( ninterval / ndomains );

//     // std::cerr << "\n\t> Morton index [" << min << ", " << max << "]" << std::endl;
//     std::cerr << "\t> Total number of interval : " << ninterval << std::endl;
//     std::cout << "\t> Perfect load-balancing (weight-interval = 1.) : " << static_cast<size_t>( ninterPerProc )
//               << " / MPI" << std::endl;

//     size_t cindex = 0;
//     for(const auto & item : sfc_map ) {
//         int fake_rank = std::floor( cindex / ninterPerProc );
//         if ( fake_rank >= ndomains ) fake_rank = ndomains - 1;

//         fake_mpi_rank( item.second.level, item.second.interval, item.second.indices ) = fake_rank;

//         cindex ++;
//     }

// }

// /**
//  *
//  * Global contains the global mesh. Since this is a toy function, it contains the union of all mesh.
//  * meshes contains the mesh of each MPI process ( or let say the mesh of neighbour processes ...)
//  *
// */
// template<int dim, class AMesh_t, class Field_t>
// void perform_load_balancing_diffusion( AMesh_t & global, std::vector<AMesh_t> & meshes, int ndomains,
//                                        const std::vector<samurai::MPI_Load_Balance> & all, Field_t & fake_mpi_rank ) {

//     using CellList_t = typename AMesh_t::cl_type;

//     struct Coord_t { xt::xtensor_fixed<double, xt::xshape<dim>> coord; };

//     std::vector<Coord_t> barycenters ( ndomains );

//     { // compute barycenter of current domains ( here, all domains since with simulate multiple MPI domains)

//         int maxlevel = 4;
//         for( size_t m_=0; m_ < meshes.size(); ++m_ ) {

//             double wght_tot = 0.;
//             samurai::for_each_cell( meshes[ m_ ], [&]( const auto & cell ) {

//                 // [OPTIMIZATION] precompute weight as array
//                 double wght = 1. / ( 1 << ( maxlevel - cell.level ) );

//                 const auto cc = cell.center();

//                 barycenters[ m_ ].coord( 0 ) += cc( 0 ) * wght;
//                 barycenters[ m_ ].coord( 1 ) += cc( 1 ) * wght;
//                 if constexpr ( dim == 3 ) { barycenters[ m_ ].coord( 2 ) += cc( 2 ) * wght; }

//                 wght_tot += wght;

//             });

//             barycenters[ m_ ].coord( 0 ) /= wght_tot;
//             barycenters[ m_ ].coord( 1 ) /= wght_tot;
//             if constexpr ( dim == 3 ) barycenters[ m_ ].coord( 2 ) /= wght_tot;

//             std::cerr << "\t> Domain # " << m_ << ", bc : {" << barycenters[ m_ ].coord( 0 ) << ", "
//                       << barycenters[ m_ ].coord( 1 ) << "}" << std::endl;

//         }

//     }

//     std::vector<CellList_t> new_meshes( ndomains );
//     std::vector<CellList_t> exchanged( ndomains );

//     for( std::size_t m_ = 0; m_ < meshes.size(); ++m_ ){ // over each domains

//         std::cerr << "\t> Working on domains # " << m_ << std::endl;

//         // auto dist = samurai::make_field<int, 1>( "rank", meshes[ m_ ] );

//         // id des neighbours dans le tableau de la structure MPI_Load_Balance
//         // attention différent du rank mpi !
//         std::vector<std::size_t> id_send;

//         int n_neighbours = static_cast<int>( all[ m_ ].neighbour.size() );
//         for(std::size_t nbi = 0; nbi<n_neighbours; ++nbi ){

//             auto nbi_rank         = all[ m_ ].neighbour[ nbi ];
//             int nbCellsToTransfer = all[ m_ ].fluxes[ nbi ];

//             std::cerr << "\t\t> Neighbour rank : " << nbi_rank << std::endl;
//             std::cerr << "\t\t> Neighbour flux : " << all[ m_ ].fluxes[ nbi ] << std::endl;

//             if( nbCellsToTransfer < 0 ){

//                 id_send.emplace_back( nbi );

//                 // Logical_coord_t stencil;
//                 // { // Compute the stencil or normalized direction to neighbour
//                 //     Coord_t tmp;
//                 //     double n2 = 0;
//                 //     for(int idim = 0; idim<dim; ++idim ){
//                 //         tmp.coord[ idim ] = barycenters[ nbi_rank ].coord( idim )- barycenters[ m_ ].coord( idim );
//                 //         n2 += tmp.coord( idim ) * tmp.coord( idim );
//                 //     }

//                 //     n2 = std::sqrt( n2 );

//                 //     for(int idim = 0; idim<dim; ++idim ){
//                 //         tmp.coord( idim ) /= n2;
//                 //         stencil.coord( idim ) = static_cast<int>( tmp.coord( idim ) / 0.5 );
//                 //     }

//                 //     std::cerr << "\t\t> stencil for neighbour # " << nbi_rank << " : ";
//                 //     for(size_t idim=0; idim<dim; ++idim ){
//                 //         std::cerr << stencil.coord( idim ) << ",";
//                 //     }
//                 //     std::cerr << std::endl;
//                 // }

//             }
//         }

//         std::cerr << "\t\t> Number RECV : " << id_send.size() << std::endl;

//         std::vector<int> already_given( n_neighbours, 0 );

//         for_each_interval( meshes[ m_ ], [&]( std::size_t level, const auto& interval, const auto& index ){
//             Coord_t ibar;

//             std::cerr << "\t\t> Interval [" << interval.start << ", " << interval.end << "[" << std::endl;

//             double dm = 1. / (1 << level );
//             ibar.coord( 0 ) = ( (interval.end - interval.start) * 0.5 + interval.start ) * dm ;
//             ibar.coord( 1 ) = ( index( 1 ) * dm ) + dm * 0.5;
//             if constexpr ( dim == 3 ) ibar( 2 ) = ( index( 2 ) * dm ) + dm * 0.5;

//             std::cerr << "\t\t\t> level " << level << " center @ {" << ibar.coord(0) << ", " << ibar.coord(1) << "}" << std::endl;

//             // find which neighbour will potentially receive this interval
//             int winner_id      = -1; // keep it to current if still negative
//             double winner_dist = std::numeric_limits<double>::max();
//             for( int nbi=0; nbi<id_send.size(); ++ nbi ){

//                 auto neighbour_rank = all[ m_ ].neighbour[ id_send[ nbi ] ];

//                 double dist = 0.0;
//                 for( int idim=0; idim < dim; ++idim){
//                     double d = barycenters[ m_ ].coord( idim ) - barycenters[ neighbour_rank ].coord( idim );
//                     dist += d * d;
//                 }
//                 dist = std::sqrt( dist );

//                 std::cerr << "\t\t\t> Dist to neighbour #" << neighbour_rank << " : " << dist << " vs " << winner_dist << std::endl;
//                 std::cerr << "\t\t\t> Already given to this neighbour " << already_given[ neighbour_rank ] << std::endl;
//                 std::cerr << "\t\t\t> NbCells of this interval " << interval.size() << std::endl;
//                 std::cerr << "\t\t\t> fluxes for this neighbour : " << all[ m_ ].fluxes[ id_send[ nbi ] ] << std::endl;

//                 if( dist < winner_dist &&
//                     already_given[ neighbour_rank ] + interval.size() <= (- all[ m_ ].fluxes[ id_send[ nbi ] ]) ){

//                     winner_id   = id_send[ nbi ];
//                     winner_dist = dist;
//                 }

//             }

//             if( winner_id >= 0 ){
//                 auto neighbour_rank = all[ m_ ].neighbour[ winner_id ];
//                 std::cerr << "\t> Interval given to process " << neighbour_rank << " + ncells : " << interval.size() << std::endl;
//                 exchanged[ neighbour_rank ][ level ][ index ].add_interval( interval );
//                 already_given[ neighbour_rank ] += interval.size();
//             }else{
//                 new_meshes[ m_ ][ level ][ index ].add_interval( interval );
//             }

//         });

//         meshes[ m_ ] = { new_meshes[ m_ ], true };

//     }

//     for( std::size_t m_ = 0; m_ < meshes.size(); ++m_ ){ // over each domains
//         for( int level=global.min_level(); level<=global.max_level(); ++level ) {
//             auto intersect = intersection( global[ level ], meshes[ m_ ][ level ]);

//             intersect([&]( auto & i, auto & index ) {
//                 fake_mpi_rank( level, i, index ) = static_cast<int>( m_ );
//             });
//         }

//         AMesh_t tmp = { exchanged[ m_], true };
//         for( int level=global.min_level(); level<=global.max_level(); ++level ) {
//             auto intersect = intersection( global[ level ], tmp[ level ]);

//             intersect([&]( auto & i, auto & index ) {
//                 fake_mpi_rank( level, i, index ) = static_cast<int>( m_ );
//             });
//         }
//     }

// }
