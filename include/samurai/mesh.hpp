// Copyright 2018-2024 the samurai's authors
// SPDX-License-Identifier:  BSD-3-Clause

#pragma once

#include <array>

#include <fmt/format.h>

#include "box.hpp"
#include "cell_array.hpp"
#include "cell_list.hpp"

#include "subset/subset_op.hpp"

#ifdef SAMURAI_WITH_MPI
#include <boost/serialization/vector.hpp>

#include <boost/mpi.hpp>
#include <boost/mpi/cartesian_communicator.hpp>
namespace mpi = boost::mpi;
#endif

namespace samurai
{

    template <class CellArray, class MeshID>
    struct MeshIDArray : private std::array<CellArray, static_cast<std::size_t>(MeshID::count)>
    {
        static constexpr std::size_t size = static_cast<std::size_t>(MeshID::count);
        using base_type                   = std::array<CellArray, size>;
        using base_type::operator[];

        inline const CellArray& operator[](MeshID mesh_id) const
        {
            return operator[](static_cast<std::size_t>(mesh_id));
        }

        inline CellArray& operator[](MeshID mesh_id)
        {
            return operator[](static_cast<std::size_t>(mesh_id));
        }
    };

    template <class MeshType>
    struct MPI_Subdomain
    {
        int rank;
        MeshType mesh;

        MPI_Subdomain(int rank_)
            : rank(rank_)
        {
        }
    };

    template <class D, class Config>
    class Mesh_base
    {
      public:

        using self_type = D;
        using config    = Config;

        static constexpr std::size_t dim                  = config::dim;
        static constexpr std::size_t max_refinement_level = config::max_refinement_level;

        using mesh_id_t  = typename config::mesh_id_t;
        using interval_t = typename config::interval_t;
        using value_t    = typename interval_t::value_t;
        using index_t    = typename interval_t::index_t;

        using cell_t   = Cell<dim, interval_t>;
        using cl_type  = CellList<dim, interval_t, max_refinement_level>;
        using lcl_type = typename cl_type::lcl_type;

        using ca_type  = CellArray<dim, interval_t, max_refinement_level>;
        using lca_type = typename ca_type::lca_type;

        using mesh_interval_t = typename ca_type::lca_type::mesh_interval_t;

        using mesh_t = samurai::MeshIDArray<ca_type, mesh_id_t>;

        using mpi_subdomain_t = MPI_Subdomain<D>;

        std::size_t nb_cells(mesh_id_t mesh_id = mesh_id_t::reference) const;
        std::size_t nb_cells(std::size_t level, mesh_id_t mesh_id = mesh_id_t::reference) const;

        const ca_type& operator[](mesh_id_t mesh_id) const;

        std::size_t max_level() const;
        std::size_t min_level() const;
        const lca_type& domain() const;
        const lca_type& subdomain() const;
        const ca_type& get_union() const;
        bool is_periodic(std::size_t d) const;
        const std::array<bool, dim>& periodicity() const;
        // std::vector<int>& neighbouring_ranks();

        std::vector<mpi_subdomain_t>& mpi_neighbourhood();

        void swap(Mesh_base& mesh) noexcept;

        template <typename... T>
        const interval_t& get_interval(std::size_t level, const interval_t& interval, T... index) const;
        template <class E>
        const interval_t& get_interval(std::size_t level, const interval_t& interval, const xt::xexpression<E>& index) const;
        template <class E>
        const interval_t& get_interval(std::size_t level, const xt::xexpression<E>& coord) const;

        template <typename... T>
        index_t get_index(std::size_t level, value_t i, T... index) const;
        template <class E>
        index_t get_index(std::size_t level, value_t i, const xt::xexpression<E>& others) const;
        template <class E>
        index_t get_index(std::size_t level, const xt::xexpression<E>& coord) const;

        template <typename... T>
        cell_t get_cell(std::size_t level, value_t i, T... index) const;
        template <class E>
        cell_t get_cell(std::size_t level, value_t i, const xt::xexpression<E>& index) const;
        template <class E>
        cell_t get_cell(std::size_t level, const xt::xexpression<E>& coord) const;

        void update_mesh_neighbour();
        void to_stream(std::ostream& os) const;

        void merge(ca_type& lca);
        void remove(ca_type& lca);

      protected:

        using derived_type = D;

        Mesh_base() = default; // cppcheck-suppress uninitMemberVar
        Mesh_base(const cl_type& cl, const self_type& ref_mesh);
        Mesh_base(const ca_type& ca, const self_type& ref_mesh);
        Mesh_base(const cl_type& cl, std::size_t min_level, std::size_t max_level);
        Mesh_base(const samurai::Box<double, dim>& b, std::size_t start_level, std::size_t min_level, std::size_t max_level);
        Mesh_base(const samurai::Box<double, dim>& b,
                  std::size_t start_level,
                  std::size_t min_level,
                  std::size_t max_level,
                  const std::array<bool, dim>& periodic);

        // Used for load balancing
        Mesh_base(const cl_type& cl, std::size_t min_level, std::size_t max_level, std::vector<mpi_subdomain_t>& neighbourhood);

        derived_type& derived_cast() & noexcept;
        const derived_type& derived_cast() const& noexcept;
        derived_type derived_cast() && noexcept;

        mesh_t& cells();

      private:

        void construct_subdomain();
        void construct_union();
        void update_sub_mesh();
        void renumbering();
        void partition_mesh(std::size_t start_level, const Box<double, dim>& global_box);

        lca_type m_domain;
        lca_type m_subdomain;
        std::size_t m_min_level;
        std::size_t m_max_level;
        std::array<bool, dim> m_periodic;
        mesh_t m_cells;
        ca_type m_union;
        // std::vector<int> m_neighbouring_ranks;
        std::vector<mpi_subdomain_t> m_mpi_neighbourhood;

#ifdef SAMURAI_WITH_MPI
        friend class boost::serialization::access;

        template <class Archive>
        void serialize(Archive& ar, const unsigned long)
        {
            for (std::size_t id = 0; id < mesh_t::size; ++id)
            {
                ar& m_cells[id];
            }
            ar & m_domain;
            ar & m_subdomain;
            ar & m_union;
            ar & m_min_level;
            ar & m_min_level;
        }
#endif
    };

    template <class D, class Config>
    inline auto Mesh_base<D, Config>::derived_cast() & noexcept -> derived_type&
    {
        return *static_cast<derived_type*>(this);
    }

    template <class D, class Config>
    inline auto Mesh_base<D, Config>::derived_cast() const& noexcept -> const derived_type&
    {
        return *static_cast<const derived_type*>(this);
    }

    template <class D, class Config>
    inline auto Mesh_base<D, Config>::derived_cast() && noexcept -> derived_type
    {
        return *static_cast<derived_type*>(this);
    }

    template <class D, class Config>
    inline Mesh_base<D, Config>::Mesh_base(const samurai::Box<double, dim>& b,
                                           std::size_t start_level,
                                           std::size_t min_level,
                                           std::size_t max_level)
        : m_domain{start_level, b}
        , m_min_level{min_level}
        , m_max_level{max_level}
    {
        assert(min_level <= max_level);
        m_periodic.fill(false);

#ifdef SAMURAI_WITH_MPI
        partition_mesh(start_level, b);
#else
        this->m_cells[mesh_id_t::cells][start_level] = {start_level, b};
#endif
        construct_subdomain();
        construct_union();
        update_sub_mesh();
        renumbering();
        update_mesh_neighbour();
    }

    template <class D, class Config>
    inline Mesh_base<D, Config>::Mesh_base(const samurai::Box<double, dim>& b,
                                           std::size_t start_level,
                                           std::size_t min_level,
                                           std::size_t max_level,
                                           const std::array<bool, dim>& periodic)
        : m_domain{start_level, b}
        , m_min_level{min_level}
        , m_max_level{max_level}
        , m_periodic{periodic}
    {
        assert(min_level <= max_level);

#ifdef SAMURAI_WITH_MPI
        partition_mesh(start_level, b);
        // load_balancing();
#else
        this->m_cells[mesh_id_t::cells][start_level] = {start_level, b};
#endif

        construct_subdomain();
        construct_union();
        update_sub_mesh();
        renumbering();
        update_mesh_neighbour();
        update_mesh_neighbour();
    }

    template <class D, class Config>
    inline Mesh_base<D, Config>::Mesh_base(const cl_type& cl, std::size_t min_level, std::size_t max_level)
        : m_min_level{min_level}
        , m_max_level{max_level}
    {
        m_periodic.fill(false);
        assert(min_level <= max_level);

        this->m_cells[mesh_id_t::cells] = {cl};

        construct_subdomain();
        m_domain = m_subdomain;
        construct_union();
        update_sub_mesh(); // MPI AllReduce inside
        renumbering();
        update_mesh_neighbour();
    }

    template <class D, class Config>
    inline Mesh_base<D, Config>::Mesh_base(const cl_type& cl, const self_type& ref_mesh)
        : m_domain(ref_mesh.m_domain)
        , m_min_level(ref_mesh.m_min_level)
        , m_max_level(ref_mesh.m_max_level)
        , m_periodic(ref_mesh.m_periodic)
        , m_mpi_neighbourhood(ref_mesh.m_mpi_neighbourhood)

    {
        m_cells[mesh_id_t::cells] = {cl, false};

        construct_subdomain();
        construct_union();
        update_sub_mesh();
        renumbering();
        update_mesh_neighbour();
    }

    template <class D, class Config>
    inline Mesh_base<D, Config>::Mesh_base(const ca_type& ca, const self_type& ref_mesh)
        : m_domain(ref_mesh.m_domain)
        , m_min_level(ref_mesh.m_min_level)
        , m_max_level(ref_mesh.m_max_level)
        , m_periodic(ref_mesh.m_periodic)
        , m_mpi_neighbourhood(ref_mesh.m_mpi_neighbourhood)

    {
        m_cells[mesh_id_t::cells] = ca;

        construct_subdomain();
        construct_union();
        update_sub_mesh();
        renumbering();
        update_mesh_neighbour();
    }

    template <class D, class Config>
    inline Mesh_base<D, Config>::Mesh_base(const cl_type& cl,
                                           std::size_t min_level,
                                           std::size_t max_level,
                                           std::vector<mpi_subdomain_t>& neighbourhood)
        : m_min_level(min_level)
        , m_max_level(max_level)
        , m_mpi_neighbourhood(neighbourhood)
    {
        m_periodic.fill(false);
        assert(min_level <= max_level);

        // what to do with m_domain ?
        m_domain = m_subdomain;

        m_cells[mesh_id_t::cells] = {cl, false};

        construct_subdomain();   // required ?
        construct_union();       // required ?
        update_sub_mesh();       // perform MPI allReduce calls
        renumbering();           // required ?
        update_mesh_neighbour(); // required to do that here ??
    }

    template <class D, class Config>
    inline auto Mesh_base<D, Config>::cells() -> mesh_t&
    {
        return m_cells;
    }

    template <class D, class Config>
    inline std::size_t Mesh_base<D, Config>::nb_cells(mesh_id_t mesh_id) const
    {
        return m_cells[mesh_id].nb_cells();
    }

    template <class D, class Config>
    inline std::size_t Mesh_base<D, Config>::nb_cells(std::size_t level, mesh_id_t mesh_id) const
    {
        return m_cells[mesh_id][level].nb_cells();
    }

    template <class D, class Config>
    inline auto Mesh_base<D, Config>::operator[](mesh_id_t mesh_id) const -> const ca_type&
    {
        return m_cells[mesh_id];
    }

    template <class D, class Config>
    inline std::size_t Mesh_base<D, Config>::max_level() const
    {
        return m_max_level;
    }

    template <class D, class Config>
    inline std::size_t Mesh_base<D, Config>::min_level() const
    {
        return m_min_level;
    }

    template <class D, class Config>
    inline auto Mesh_base<D, Config>::domain() const -> const lca_type&
    {
        return m_domain;
    }

    template <class D, class Config>
    inline auto Mesh_base<D, Config>::subdomain() const -> const lca_type&
    {
        return m_subdomain;
    }

    template <class D, class Config>
    inline auto Mesh_base<D, Config>::get_union() const -> const ca_type&
    {
        return m_union;
    }

    template <class D, class Config>
    template <typename... T>
    inline auto Mesh_base<D, Config>::get_interval(std::size_t level, const interval_t& interval, T... index) const -> const interval_t&
    {
        return m_cells[mesh_id_t::reference].get_interval(level, interval, index...);
    }

    template <class D, class Config>
    template <class E>
    inline auto Mesh_base<D, Config>::get_interval(std::size_t level,
                                                   const interval_t& interval,
                                                   const xt::xexpression<E>& index) const -> const interval_t&
    {
        return m_cells[mesh_id_t::reference].get_interval(level, interval, index);
    }

    template <class D, class Config>
    template <class E>
    inline auto Mesh_base<D, Config>::get_interval(std::size_t level, const xt::xexpression<E>& coord) const -> const interval_t&
    {
        return m_cells[mesh_id_t::reference].get_interval(level, coord);
    }

    template <class D, class Config>
    template <typename... T>
    inline auto Mesh_base<D, Config>::get_index(std::size_t level, value_t i, T... index) const -> index_t
    {
        return m_cells[mesh_id_t::reference].get_index(level, i, index...);
    }

    template <class D, class Config>
    template <class E>
    inline auto Mesh_base<D, Config>::get_index(std::size_t level, value_t i, const xt::xexpression<E>& others) const -> index_t
    {
        return m_cells[mesh_id_t::reference].get_index(level, i, others);
    }

    template <class D, class Config>
    template <class E>
    inline auto Mesh_base<D, Config>::get_index(std::size_t level, const xt::xexpression<E>& coord) const -> index_t
    {
        return m_cells[mesh_id_t::reference].get_index(level, coord);
    }

    template <class D, class Config>
    template <typename... T>
    inline auto Mesh_base<D, Config>::get_cell(std::size_t level, value_t i, T... index) const -> cell_t
    {
        return m_cells[mesh_id_t::reference].get_cell(level, i, index...);
    }

    template <class D, class Config>
    template <class E>
    inline auto Mesh_base<D, Config>::get_cell(std::size_t level, value_t i, const xt::xexpression<E>& index) const -> cell_t
    {
        return m_cells[mesh_id_t::reference].get_cell(level, i, index);
    }

    template <class D, class Config>
    template <class E>
    inline auto Mesh_base<D, Config>::get_cell(std::size_t level, const xt::xexpression<E>& coord) const -> cell_t
    {
        return m_cells[mesh_id_t::reference].get_cell(level, coord);
    }

    template <class D, class Config>
    inline bool Mesh_base<D, Config>::is_periodic(std::size_t d) const
    {
        return m_periodic[d];
    }

    template <class D, class Config>
    inline auto Mesh_base<D, Config>::periodicity() const -> const std::array<bool, dim>&
    {
        return m_periodic;
    }

    template <class D, class Config>
    inline auto Mesh_base<D, Config>::mpi_neighbourhood() -> std::vector<mpi_subdomain_t>&
    {
        return m_mpi_neighbourhood;
    }

    template <class D, class Config>
    inline void Mesh_base<D, Config>::swap(Mesh_base<D, Config>& mesh) noexcept
    {
        using std::swap;
        swap(m_cells, mesh.m_cells);
        swap(m_domain, mesh.m_domain);
        swap(m_subdomain, mesh.m_subdomain);
        swap(m_mpi_neighbourhood, mesh.m_mpi_neighbourhood);
        swap(m_union, mesh.m_union);
        swap(m_max_level, mesh.m_max_level);
        swap(m_min_level, mesh.m_min_level);
    }

    template <class D, class Config>
    inline void Mesh_base<D, Config>::update_sub_mesh()
    {
        this->derived_cast().update_sub_mesh_impl();
    }

    template <class D, class Config>
    inline void Mesh_base<D, Config>::renumbering()
    {
        m_cells[mesh_id_t::reference].update_index();

        for (std::size_t id = 0; id < static_cast<std::size_t>(mesh_id_t::count); ++id)
        {
            auto mt = static_cast<mesh_id_t>(id);

            if (mt != mesh_id_t::reference)
            {
                for (std::size_t level = 0; level <= max_refinement_level; ++level)
                {
                    lca_type& lhs       = m_cells[mt][level];
                    const lca_type& rhs = m_cells[mesh_id_t::reference][level];

                    auto expr = intersection(lhs, rhs);
                    expr.apply_interval_index(
                        [&](const auto& interval_index)
                        {
                            lhs[0][interval_index[0]].index = rhs[0][interval_index[1]].index;
                        });
                }
            }
        }
    }

    template <class D, class Config>
    inline void Mesh_base<D, Config>::update_mesh_neighbour()
    {
#ifdef SAMURAI_WITH_MPI
        // send/recv the meshes of the neighbouring subdomains
        mpi::communicator world;
        std::vector<mpi::request> req;

        std::transform(m_mpi_neighbourhood.cbegin(),
                       m_mpi_neighbourhood.cend(),
                       std::back_inserter(req),
                       [&](const auto& neighbour)
                       {
                           return world.isend(neighbour.rank, neighbour.rank, derived_cast());
                       });

        for (auto& neighbour : m_mpi_neighbourhood)
        {
            world.recv(neighbour.rank, world.rank(), neighbour.mesh);
        }

        mpi::wait_all(req.begin(), req.end());
#endif
    }

    template <class D, class Config>
    inline void Mesh_base<D, Config>::construct_subdomain()
    {
        // lcl_type lcl = {m_cells[mesh_id_t::cells].max_level()};
        lcl_type lcl = {m_max_level};

        for_each_interval(m_cells[mesh_id_t::cells],
                          [&](std::size_t level, const auto& i, const auto& index)
                          {
                              std::size_t shift = m_max_level - level;
                              interval_t to_add = i << shift;
                              auto shift_index  = index << shift;
                              static_nested_loop<dim - 1>(0,
                                                          1 << shift,
                                                          1,
                                                          [&](auto stencil)
                                                          {
                                                              auto new_index = shift_index + stencil;
                                                              lcl[new_index].add_interval(to_add);
                                                          });
                          });
        m_subdomain = {lcl};
    }

    template <class D, class Config>
    inline void Mesh_base<D, Config>::construct_union()
    {
        std::size_t min_lvl = m_min_level;
        std::size_t max_lvl = m_max_level;

        // Construction of union cells
        // ===========================
        //
        // level 2                 |-|-|-|-|                   |-| cells
        //                                                     |.| union_cells
        // level 1         |---|---|       |---|---|
        //                         |...|...|
        // level 0 |-------|                       |-------|
        //                 |.......|.......|.......|
        //

        // FIX: cppcheck false positive ?
        // cppcheck-suppress redundantAssignment
        m_union[max_lvl] = {max_lvl};
        for (std::size_t level = max_lvl; level >= ((min_lvl == 0) ? 1 : min_lvl); --level)
        {
            lcl_type lcl{level - 1};
            auto expr = union_(this->m_cells[mesh_id_t::cells][level], m_union[level]).on(level - 1);

            expr(
                [&](const auto& interval, const auto& index_yz)
                {
                    lcl[index_yz].add_interval(interval);
                });

            // for (auto& neighbour : m_mpi_neighbourhood)
            // {
            //     auto neigh_expr = intersection(m_subdomain, union_(neighbour.mesh.m_cells[mesh_id_t::cells][level], m_union[level]))
            //                           .on(level - 1);

            //     neigh_expr(
            //         [&](const auto& interval, const auto& index_yz)
            //         {
            //             lcl[index_yz].add_interval(interval);
            //         });
            // }
            m_union[level - 1] = {lcl};
        }
    }

    template <class D, class Config>
    void Mesh_base<D, Config>::partition_mesh([[maybe_unused]] std::size_t start_level, [[maybe_unused]] const Box<double, dim>& global_box)
    {
#ifdef SAMURAI_WITH_MPI
        using box_t   = Box<value_t, dim>;
        using point_t = typename box_t::point_t;

        mpi::communicator world;
        auto rank = world.rank();
        auto size = world.size();

        double h = cell_length(start_level);

        // Computes the number of subdomains in each Cartesian direction
        std::array<int, dim> sizes;
        auto product_of_length   = xt::prod(global_box.length())[0];
        auto length_harmonic_avg = pow(product_of_length, 1. / dim);
        int product_of_sizes     = 1;
        for (std::size_t d = 0; d < dim - 1; ++d)
        {
            sizes[d] = std::max(static_cast<int>(floor(pow(size, 1. / dim) * global_box.length()[d] / length_harmonic_avg)), 1);

            product_of_sizes *= sizes[d];
        }
        sizes[dim - 1] = size / product_of_sizes;
        if (sizes[dim - 1] * product_of_sizes != size)
        {
            if (rank == 0)
            {
                std::cerr << "Impossible to perform a Cartesian partition of the domain in " << size << " subdomains." << std::endl;
                std::cerr << "Suggested number: " << (sizes[dim - 1] * product_of_sizes) << "." << std::endl;
            }
            exit(1);
        }

        // Compute the Cartesian coordinates of the subdomain in the topology
        int a = rank;
        xt::xtensor_fixed<int, xt::xshape<dim>> coords;
        for (std::size_t d = 0; d < dim; ++d)
        {
            coords[d] = a % sizes[d];
            a         = a / sizes[d];
        }

        // Directional lengths of a standard subdomain
        point_t start_pt = global_box.min_corner() / h;
        point_t end_pt   = global_box.max_corner() / h;
        xt::xtensor_fixed<double, xt::xshape<dim>> lengths;
        for (std::size_t d = 0; d < dim; ++d)
        {
            lengths[d] = ceil((end_pt[d] - start_pt[d]) / static_cast<double>(sizes[d]));
        }

        // Create the box corresponding to the local subdomain
        point_t min_corner, max_corner;
        min_corner = start_pt + coords * lengths;
        max_corner = min_corner + lengths;

        for (std::size_t d = 0; d < dim; ++d)
        {
            if (coords[d] == sizes[d] - 1)
            {
                max_corner[d] = end_pt[d];
            }
        }
        box_t subdomain_box                          = {min_corner, max_corner};
        this->m_cells[mesh_id_t::cells][start_level] = {start_level, subdomain_box};

        // m_mpi_neighbourhood.reserve(static_cast<std::size_t>(size) - 1);
        // for (int ir = 0; ir < size; ++ir)
        // {
        //     if (ir != rank)
        //     {
        //         m_mpi_neighbourhood.push_back(ir);
        //     }
        // }

        // // Neighbours
        m_mpi_neighbourhood.reserve(static_cast<std::size_t>(pow(3, dim) - 1));
        auto neighbour = [&](xt::xtensor_fixed<int, xt::xshape<dim>> shift)
        {
            auto neighbour_rank            = rank;
            int product_of_preceding_sizes = 1;
            for (std::size_t d = 0; d < dim; ++d)
            {
                neighbour_rank += product_of_preceding_sizes * shift[d];
                product_of_preceding_sizes *= sizes[d];
            }
            return neighbour_rank;
        };

        static_nested_loop<dim, -1, 2>(
            [&](auto& shift)
            {
                if (xt::any(shift))
                {
                    for (std::size_t d = 0; d < dim; ++d)
                    {
                        if (coords[d] + shift[d] < 0 || coords[d] + shift[d] >= sizes[d])
                        {
                            return;
                        }
                    }
                    m_mpi_neighbourhood.push_back(neighbour(shift));
                }
            });
#endif
    }

    template <class D, class Config>
    void Mesh_base<D, Config>::merge(ca_type& lca)
    {
        // merge received cells

        auto& refmesh = this->m_cells[mesh_id_t::cells];

        auto minlevel = std::min(refmesh.min_level(), lca.min_level());
        auto maxlevel = std::max(refmesh.max_level(), lca.max_level());

        cl_type cl;
        for (size_t ilvl = minlevel; ilvl <= maxlevel; ++ilvl)
        {
            auto un = samurai::union_(refmesh[ilvl], lca[ilvl]);

            un(
                [&](auto& interval, auto& indices)
                {
                    cl[ilvl][indices].add_interval(interval);
                });
        }

        refmesh = {cl, false};
    }

    template <class D, class Config>
    void Mesh_base<D, Config>::remove(ca_type& lca)
    {
        auto& refmesh = this->m_cells[mesh_id_t::cells];

        // remove cells
        cl_type cl;
        size_t diff_ncells = 0;
        for (size_t ilvl = refmesh.min_level(); ilvl <= refmesh.max_level(); ++ilvl)
        {
            auto diff = samurai::difference(refmesh[ilvl], lca[ilvl]);

            diff(
                [&](auto& interval, auto& index)
                {
                    cl[ilvl][index].add_interval(interval);
                    diff_ncells += interval.size();
                });
        }

        // new mesh for current process
        refmesh = {cl, false};
    }

    template <class D, class Config>
    inline void Mesh_base<D, Config>::to_stream(std::ostream& os) const
    {
        for (std::size_t id = 0; id < static_cast<std::size_t>(mesh_id_t::count); ++id)
        {
            auto mt = static_cast<mesh_id_t>(id);

            os << fmt::format(disable_color ? fmt::text_style() : fmt::emphasis::bold, "{}\n{:─^50}", mt, "") << std::endl;
            os << m_cells[id];
        }
    }

    template <class D, class Config>
    inline bool operator==(const Mesh_base<D, Config>& mesh1, const Mesh_base<D, Config>& mesh2)
    {
        using mesh_id_t = typename Mesh_base<D, Config>::mesh_id_t;

        if (mesh1.max_level() != mesh2.max_level() || mesh1.min_level() != mesh2.min_level())
        {
            return false;
        }

        for (std::size_t level = mesh1.min_level(); level <= mesh1.max_level(); ++level)
        {
            if (!(mesh1[mesh_id_t::cells][level] == mesh2[mesh_id_t::cells][level]))
            {
                return false;
            }
        }
        return true;
    }

    template <class D, class Config>
    inline bool operator!=(const Mesh_base<D, Config>& mesh1, const Mesh_base<D, Config>& mesh2)
    {
        return !(mesh1 == mesh2);
    }

    template <class D, class Config>
    inline std::ostream& operator<<(std::ostream& out, const Mesh_base<D, Config>& mesh)
    {
        mesh.to_stream(out);
        return out;
    }
} // namespace samurai
