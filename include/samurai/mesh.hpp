// Copyright 2018-2025 the samurai's authors
// SPDX-License-Identifier:  BSD-3-Clause

#pragma once

#include <array>
#include <set>

#include <fmt/format.h>

#include "cell_array.hpp"
#include "cell_list.hpp"
#include "domain_builder.hpp"
#include "static_algorithm.hpp"
#include "stencil.hpp"
#include "subset/node.hpp"

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

        using coords_t = typename lca_type::coords_t;

        using mesh_interval_t = typename ca_type::lca_type::mesh_interval_t;

        using mesh_t = samurai::MeshIDArray<ca_type, mesh_id_t>;

        using mpi_subdomain_t = MPI_Subdomain<D>;

        std::size_t nb_cells(mesh_id_t mesh_id = mesh_id_t::reference) const;
        std::size_t nb_cells(std::size_t level, mesh_id_t mesh_id = mesh_id_t::reference) const;

        const ca_type& operator[](mesh_id_t mesh_id) const;
        ca_type& operator[](mesh_id_t mesh_id);

        std::size_t max_level() const;
        std::size_t& max_level();
        std::size_t min_level() const;
        std::size_t& min_level();

        auto& origin_point() const;
        void set_origin_point(const coords_t& origin_point);
        double scaling_factor() const;
        void set_scaling_factor(double scaling_factor);
        void scale_domain(double domain_scaling_factor);
        double cell_length(std::size_t level) const;
        const lca_type& domain() const;
        const lca_type& subdomain() const;
        const ca_type& get_union() const;
        bool is_periodic() const;
        bool is_periodic(std::size_t d) const;
        const std::array<bool, dim>& periodicity() const;
        // std::vector<int>& neighbouring_ranks();
        std::vector<mpi_subdomain_t>& mpi_neighbourhood();
        const std::vector<mpi_subdomain_t>& mpi_neighbourhood() const;
        cl_type
        construct_initial_mesh(const DomainBuilder<dim>& domain_builder, std::size_t start_level, double approx_box_tol, double scaling_factor);
        void compute_scaling_factor(const samurai::DomainBuilder<dim>& domain_builder, double& scaling_factor);

        void swap(Mesh_base& mesh) noexcept;

        template <typename... T, typename = std::enable_if_t<std::conjunction_v<std::is_convertible<T, value_t>...>, void>>
        const interval_t& get_interval(std::size_t level, const interval_t& interval, T... index) const;
        template <class E>
        const interval_t& get_interval(std::size_t level, const interval_t& interval, const xt::xexpression<E>& index) const;
        template <class E>
        const interval_t& get_interval(std::size_t level, const xt::xexpression<E>& coord) const;

        template <typename... T, typename = std::enable_if_t<std::conjunction_v<std::is_convertible<T, value_t>...>, void>>
        index_t get_index(std::size_t level, value_t i, T... index) const;
        template <class E>
        index_t get_index(std::size_t level, value_t i, const xt::xexpression<E>& others) const;
        template <class E>
        index_t get_index(std::size_t level, const xt::xexpression<E>& coord) const;

        template <typename... T, typename = std::enable_if_t<std::conjunction_v<std::is_convertible<T, value_t>...>, void>>
        cell_t get_cell(std::size_t level, value_t i, T... index) const;
        template <class E>
        cell_t get_cell(std::size_t level, value_t i, const xt::xexpression<E>& index) const;
        template <class E>
        cell_t get_cell(std::size_t level, const xt::xexpression<E>& coord) const;

        void update_mesh_neighbour();
        void update_neighbour_subdomain();
        void update_meshid_neighbour(const mesh_id_t& mesh_id);

        void to_stream(std::ostream& os) const;

        const lca_type& corner(const DirectionVector<dim>& direction) const;

      protected:

        using derived_type = D;

        Mesh_base() = default; // cppcheck-suppress uninitMemberVar
        Mesh_base(const ca_type& ca, const self_type& ref_mesh);
        Mesh_base(const cl_type& cl, const self_type& ref_mesh);
        Mesh_base(const cl_type& cl, std::size_t min_level, std::size_t max_level);
        Mesh_base(const ca_type& ca, std::size_t min_level, std::size_t max_level);
        Mesh_base(const samurai::Box<double, dim>& b,
                  std::size_t start_level,
                  std::size_t min_level,
                  std::size_t max_level,
                  double approx_box_tol = lca_type::default_approx_box_tol,
                  double scaling_factor = 0);
        Mesh_base(const samurai::DomainBuilder<dim>& domain_builder,
                  std::size_t start_level,
                  std::size_t min_level,
                  std::size_t max_level,
                  double approx_box_tol = lca_type::default_approx_box_tol,
                  double scaling_factor = 0);
        Mesh_base(const samurai::Box<double, dim>& b,
                  std::size_t start_level,
                  std::size_t min_level,
                  std::size_t max_level,
                  const std::array<bool, dim>& periodic,
                  double approx_box_tol = lca_type::default_approx_box_tol,
                  double scaling_factor = 0);

        derived_type& derived_cast() & noexcept;
        const derived_type& derived_cast() const& noexcept;
        derived_type derived_cast() && noexcept;

        mesh_t& cells();

      private:

        void construct_subdomain();
        void construct_domain();
        void construct_union();
        void construct_corners();
        void update_sub_mesh();
        void renumbering();

        void find_neighbourhood();

        void partition_mesh(std::size_t start_level, const Box<double, dim>& global_box);
        void load_balancing();
        void load_transfer(const std::vector<double>& load_fluxes);
        std::size_t max_nb_cells(std::size_t level) const;

        lca_type m_domain;
        lca_type m_subdomain;
        std::size_t m_min_level;
        std::size_t m_max_level;
        std::array<bool, dim> m_periodic;
        mesh_t m_cells;
        ca_type m_union;
        std::vector<lca_type> m_corners;
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
            ar & m_max_level;
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
                                           std::size_t max_level,
                                           double approx_box_tol,
                                           double scaling_factor_)
        : m_domain{start_level, b, approx_box_tol, scaling_factor_}
        , m_min_level{min_level}
        , m_max_level{max_level}
    {
        assert(min_level <= max_level);
        m_periodic.fill(false);

#ifdef SAMURAI_WITH_MPI
        partition_mesh(start_level, b);
        // load_balancing();
#else
        this->m_cells[mesh_id_t::cells][start_level] = {start_level, b, approx_box_tol, scaling_factor_};
#endif
        construct_subdomain();
        construct_union();
        update_sub_mesh();
        construct_corners();
        renumbering();
        update_mesh_neighbour();

        set_origin_point(origin_point());
        set_scaling_factor(scaling_factor());
    }

    template <class D, class Config>
    Mesh_base<D, Config>::Mesh_base(const samurai::DomainBuilder<dim>& domain_builder,
                                    [[maybe_unused]] std::size_t start_level,
                                    std::size_t min_level,
                                    std::size_t max_level,
                                    [[maybe_unused]] double approx_box_tol,
                                    double scaling_factor_)
        : m_min_level{min_level}
        , m_max_level{max_level}
    {
        assert(min_level <= max_level);
        m_periodic.fill(false);

#ifdef SAMURAI_WITH_MPI
        std::cerr << "MPI is not implemented with DomainBuilder." << std::endl;
        std::exit(1);
        // partition_mesh(start_level, b);
        //  load_balancing();
#else
        compute_scaling_factor(domain_builder, scaling_factor_);

        // Build the domain by adding and removing boxes
        cl_type domain_cl = construct_initial_mesh(domain_builder, start_level, approx_box_tol, scaling_factor_);

        this->m_cells[mesh_id_t::cells] = {domain_cl, false};
#endif
        construct_subdomain();
        m_domain = m_subdomain;
        construct_union();
        update_sub_mesh();
        construct_corners();
        renumbering();
        update_mesh_neighbour();

        set_origin_point(domain_builder.origin_point());
        set_scaling_factor(scaling_factor_);
    }

    template <class D, class Config>
    inline Mesh_base<D, Config>::Mesh_base(const samurai::Box<double, dim>& b,
                                           std::size_t start_level,
                                           std::size_t min_level,
                                           std::size_t max_level,
                                           const std::array<bool, dim>& periodic,
                                           double approx_box_tol,
                                           double scaling_factor_)
        : m_domain{start_level, b, approx_box_tol, scaling_factor_}
        , m_min_level{min_level}
        , m_max_level{max_level}
        , m_periodic{periodic}
    {
        assert(min_level <= max_level);

#ifdef SAMURAI_WITH_MPI
        partition_mesh(start_level, b);
        // load_balancing();
#else
        this->m_cells[mesh_id_t::cells][start_level] = {start_level, b, approx_box_tol, scaling_factor_};
#endif

        construct_subdomain();
        construct_union();
        update_sub_mesh();
        construct_corners();
        renumbering();
        update_mesh_neighbour();

        set_origin_point(origin_point());
        set_scaling_factor(scaling_factor());
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
        construct_domain();
        construct_union();
        update_sub_mesh();
        construct_corners();
        renumbering();
        update_mesh_neighbour();

        set_origin_point(cl.origin_point());
        set_scaling_factor(cl.scaling_factor());
    }

    template <class D, class Config>
    inline Mesh_base<D, Config>::Mesh_base(const ca_type& ca, std::size_t min_level, std::size_t max_level)
        : m_min_level{min_level}
        , m_max_level{max_level}
    {
        m_periodic.fill(false);
        assert(min_level <= max_level);

        this->m_cells[mesh_id_t::cells] = ca;

        construct_subdomain();
        construct_domain();
        construct_union();
        update_sub_mesh();
        construct_corners();
        renumbering();

#ifdef SAMURAI_WITH_MPI
        mpi::communicator world;
        m_mpi_neighbourhood.clear();
        for (int i = 0; i < world.size(); ++i)
        {
            if (i != world.rank())
            {
                m_mpi_neighbourhood.emplace_back(i);
            }
        }
#endif
        update_mesh_neighbour();

        set_origin_point(ca.origin_point());
        set_scaling_factor(ca.scaling_factor());
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
        construct_corners();
        renumbering();
        update_mesh_neighbour();

        set_origin_point(ref_mesh.origin_point());
        set_scaling_factor(ref_mesh.scaling_factor());
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
        construct_corners();
        renumbering();
        update_mesh_neighbour();

        set_origin_point(ref_mesh.origin_point());
        set_scaling_factor(ref_mesh.scaling_factor());
    }

    template <class D, class Config>
    void Mesh_base<D, Config>::compute_scaling_factor(const samurai::DomainBuilder<dim>& domain_builder, double& scaling_factor)
    {
        // We need to be able to apply the BC at all levels (between min_level and max_level, but not under min_level
        // since the BC are only apply near real cells).
        // If there is a hole that isn't large enough to have enough ghosts to apply the BC, we need to refine the mesh.
        // (Otherwise, some ghosts at the end of the stencil will infact be cells on the other side of the hole.)
        // Another constraint (that will be lifted in the future): as we simultaneously apply the BC on both positive and
        // negartive directions, we actually need 2 times the stencil width inside the hole.

        // min_level where the BC can be applied
        std::size_t min_level_bc = m_min_level;
        if (scaling_factor <= 0)
        {
            scaling_factor = domain_builder.largest_subdivision();

            auto largest_cell_length = samurai::cell_length(scaling_factor, min_level_bc);
            for (const auto& box : domain_builder.removed_boxes())
            {
                while (box.min_length() < 2 * largest_cell_length * config::max_stencil_width)
                {
                    scaling_factor /= 2;
                    largest_cell_length /= 2;
                }
            }
        }
        else
        {
            auto largest_cell_length = samurai::cell_length(scaling_factor, min_level_bc);
            for (const auto& box : domain_builder.removed_boxes())
            {
                if (box.min_length() < 2 * largest_cell_length * config::max_stencil_width)
                {
                    std::cerr << "The hole " << box << " is too small to apply the BC at level " << min_level_bc
                              << " with the given scaling factor. We need to be able to construct " << (2 * config::max_stencil_width)
                              << " ghosts in each direction inside the hole." << std::endl;
                    std::cerr << "Please choose a smaller scaling factor or enlarge the hole." << std::endl;
                    std::exit(1);
                }
            }
        }
    }

    template <class D, class Config>
    auto Mesh_base<D, Config>::construct_initial_mesh(const samurai::DomainBuilder<dim>& domain_builder,
                                                      std::size_t start_level,
                                                      double approx_box_tol,
                                                      double scaling_factor) -> cl_type
    {
        // Build the domain by adding and removing boxes

        auto origin_point_ = domain_builder.origin_point();

        cl_type domain_cl(origin_point_, scaling_factor);

        for (const auto& box : domain_builder.added_boxes())
        {
            lca_type box_lca(start_level, box, origin_point_, approx_box_tol, scaling_factor);
            lca_type current_domain_lca(domain_cl[start_level]);
            auto new_domain_set = union_(current_domain_lca, box_lca);
            domain_cl           = cl_type(origin_point_, scaling_factor);
            new_domain_set(
                [&](const auto& i, const auto& index)
                {
                    domain_cl[start_level][index].add_interval({i});
                });
        }
        for (const auto& box : domain_builder.removed_boxes())
        {
            lca_type hole_lca(start_level, box, origin_point_, approx_box_tol, scaling_factor);
            lca_type current_domain_lca(domain_cl[start_level]);
            auto new_domain_set = difference(current_domain_lca, hole_lca);
            domain_cl           = cl_type(origin_point_, scaling_factor);
            new_domain_set(
                [&](const auto& i, const auto& index)
                {
                    domain_cl[start_level][index].add_interval({i});
                });
        }

        return domain_cl;
    }

    template <class D, class Config>
    inline auto Mesh_base<D, Config>::cells() -> mesh_t&
    {
        return m_cells;
    }

    template <class D, class Config>
    inline std::size_t Mesh_base<D, Config>::max_nb_cells(std::size_t level) const
    {
        if (m_cells[mesh_id_t::reference][level][0].empty())
        {
            return 0;
        }
        auto last_xinterval = m_cells[mesh_id_t::reference][level][0].back();
        return static_cast<std::size_t>(static_cast<index_t>(last_xinterval.start) + last_xinterval.index) + last_xinterval.size();
    }

    template <class D, class Config>
    inline std::size_t Mesh_base<D, Config>::nb_cells(mesh_id_t mesh_id) const
    {
        return (mesh_id == mesh_id_t::reference) ? max_nb_cells(m_cells[mesh_id].max_level()) : m_cells[mesh_id].nb_cells();
    }

    template <class D, class Config>
    inline std::size_t Mesh_base<D, Config>::nb_cells(std::size_t level, mesh_id_t mesh_id) const
    {
        return (mesh_id == mesh_id_t::reference) ? max_nb_cells(level) : m_cells[mesh_id][level].nb_cells();
    }

    template <class D, class Config>
    inline auto Mesh_base<D, Config>::operator[](mesh_id_t mesh_id) const -> const ca_type&
    {
        return m_cells[mesh_id];
    }

    template <class D, class Config>
    inline auto Mesh_base<D, Config>::operator[](mesh_id_t mesh_id) -> ca_type&
    {
        return m_cells[mesh_id];
    }

    template <class D, class Config>
    inline std::size_t Mesh_base<D, Config>::max_level() const
    {
        return m_max_level;
    }

    template <class D, class Config>
    inline std::size_t& Mesh_base<D, Config>::max_level()
    {
        return m_max_level;
    }

    template <class D, class Config>
    inline std::size_t Mesh_base<D, Config>::min_level() const
    {
        return m_min_level;
    }

    template <class D, class Config>
    inline std::size_t& Mesh_base<D, Config>::min_level()
    {
        return m_min_level;
    }

    template <class D, class Config>
    inline auto& Mesh_base<D, Config>::origin_point() const
    {
        return m_domain.origin_point();
    }

    template <class D, class Config>
    inline void Mesh_base<D, Config>::set_origin_point(const coords_t& origin_point)
    {
        m_domain.set_origin_point(origin_point);
        m_subdomain.set_origin_point(origin_point);
        m_union.set_origin_point(origin_point);
        for (std::size_t i = 0; i < static_cast<std::size_t>(mesh_id_t::count); ++i)
        {
            m_cells[i].set_origin_point(origin_point);
        }
    }

    template <class D, class Config>
    inline double Mesh_base<D, Config>::scaling_factor() const
    {
        return m_domain.scaling_factor();
    }

    template <class D, class Config>
    inline void Mesh_base<D, Config>::set_scaling_factor(double scaling_factor)
    {
        m_domain.set_scaling_factor(scaling_factor);
        m_subdomain.set_scaling_factor(scaling_factor);
        m_union.set_scaling_factor(scaling_factor);
        for (std::size_t i = 0; i < static_cast<std::size_t>(mesh_id_t::count); ++i)
        {
            m_cells[i].set_scaling_factor(scaling_factor);
        }
    }

    template <class D, class Config>
    inline void Mesh_base<D, Config>::scale_domain(double domain_scaling_factor)
    {
        set_scaling_factor(domain_scaling_factor * scaling_factor());
    }

    template <class D, class Config>
    inline double Mesh_base<D, Config>::cell_length(std::size_t level) const
    {
        return samurai::cell_length(scaling_factor(), level);
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
    template <typename... T, typename U>
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
    template <typename... T, typename U>
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
    template <typename... T, typename U>
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
    inline bool Mesh_base<D, Config>::is_periodic() const
    {
        return std::any_of(m_periodic.cbegin(),
                           m_periodic.cend(),
                           [](bool v)
                           {
                               return v;
                           });
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
    inline auto Mesh_base<D, Config>::mpi_neighbourhood() const -> const std::vector<mpi_subdomain_t>&
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
                for_each_interval(m_cells[mt],
                                  [&](std::size_t level, auto& i, auto& index)
                                  {
                                      i.index = m_cells[mesh_id_t::reference][level].get_interval(i, index).index;
                                  });
            }
        }
    }

    template <class D, class Config>
    inline void Mesh_base<D, Config>::construct_corners()
    {
        using direction_t = DirectionVector<dim>;

        static_assert(dim <= 3, "Only 2D and 3D are supported.");

        m_corners.clear();
        for_each_diagonal_direction<dim>(
            [&](const auto& direction)
            {
                if constexpr (dim == 2)
                {
                    m_corners.push_back(difference(m_domain,
                                                   union_(translate(m_domain, direction_t{-direction[0], 0}),
                                                          translate(m_domain, direction_t{0, -direction[1]})))
                                            .to_lca());
                }
                else if constexpr (dim == 3)
                {
                    m_corners.push_back(difference(m_domain,
                                                   union_(translate(m_domain, direction_t{-direction[0], 0, 0}),
                                                          translate(m_domain, direction_t{0, -direction[1], 0}),
                                                          translate(m_domain, direction_t{0, 0, -direction[2]})))
                                            .to_lca());
                }
            });
    }

    template <class D, class Config>
    auto Mesh_base<D, Config>::corner(const DirectionVector<dim>& direction) const -> const lca_type&
    {
        std::size_t i           = 0;
        std::size_t i_direction = 0;
        for_each_diagonal_direction<dim>(
            [&](const auto& dir)
            {
                if (dir == direction)
                {
                    i_direction = i;
                }
                ++i;
            });

        return m_corners[i_direction];
    }

    template <class D, class Config>
    inline void Mesh_base<D, Config>::update_mesh_neighbour()
    {
#ifdef SAMURAI_WITH_MPI
        // send/recv the meshes of the neighbouring subdomains
        mpi::communicator world;
        std::vector<mpi::request> req;

        boost::mpi::packed_oarchive::buffer_type buffer;
        boost::mpi::packed_oarchive oa(world, buffer);
        oa << derived_cast();

        std::transform(m_mpi_neighbourhood.cbegin(),
                       m_mpi_neighbourhood.cend(),
                       std::back_inserter(req),
                       [&](const auto& neighbour)
                       {
                           return world.isend(neighbour.rank, neighbour.rank, buffer);
                       });

        for (auto& neighbour : m_mpi_neighbourhood)
        {
            world.recv(neighbour.rank, world.rank(), neighbour.mesh);
        }

        mpi::wait_all(req.begin(), req.end());
#endif
    }

    // TODO : find a clever way to factorize the two next functions. For new, I have to duplicate the code 2 times.

    // This function is to only send m_subdomain instead of the whole mesh data
    template <class D, class Config>
    inline void Mesh_base<D, Config>::update_neighbour_subdomain()
    {
#ifdef SAMURAI_WITH_MPI
        // send/recv the meshes of the neighbouring subdomains
        mpi::communicator world;
        std::vector<mpi::request> req;

        boost::mpi::packed_oarchive::buffer_type buffer;
        boost::mpi::packed_oarchive oa(world, buffer);
        oa << derived_cast().m_subdomain;

        std::transform(m_mpi_neighbourhood.cbegin(),
                       m_mpi_neighbourhood.cend(),
                       std::back_inserter(req),
                       [&](const auto& neighbour)
                       {
                           return world.isend(neighbour.rank, neighbour.rank, buffer);
                       });

        for (auto& neighbour : m_mpi_neighbourhood)
        {
            world.recv(neighbour.rank, world.rank(), neighbour.mesh.m_subdomain);
        }

        mpi::wait_all(req.begin(), req.end());
#endif
    }

    // Modified function definition
    template <class D, class Config>
    inline void Mesh_base<D, Config>::update_meshid_neighbour([[maybe_unused]] const mesh_id_t& mesh_id)
    {
#ifdef SAMURAI_WITH_MPI
        mpi::communicator world;
        std::vector<mpi::request> req;

        boost::mpi::packed_oarchive::buffer_type buffer;
        boost::mpi::packed_oarchive oa(world, buffer);
        oa << derived_cast()[mesh_id];

        std::transform(m_mpi_neighbourhood.cbegin(),
                       m_mpi_neighbourhood.cend(),
                       std::back_inserter(req),
                       [&](const auto& neighbour)
                       {
                           return world.isend(neighbour.rank, neighbour.rank, buffer);
                       });

        for (auto& neighbour : m_mpi_neighbourhood)
        {
            world.recv(neighbour.rank, world.rank(), neighbour.mesh[mesh_id]);
        }

        mpi::wait_all(req.begin(), req.end());
#endif // SAMURAI_WITH_MPI
    }

    template <class D, class Config>
    inline void Mesh_base<D, Config>::construct_domain()
    {
#ifdef SAMURAI_WITH_MPI
        lcl_type lcl = {m_max_level};
        mpi::communicator world;
        std::vector<lca_type> all_subdomains(static_cast<std::size_t>(world.size()));
        mpi::all_gather(world, m_subdomain, all_subdomains);

        for (std::size_t k = 0; k < all_subdomains.size(); ++k)
        {
            for_each_interval(all_subdomains[k],
                              [&](auto, const auto& i, const auto& index)
                              {
                                  lcl[index].add_interval(i);
                              });
        }

        m_domain = {lcl};
#else
        m_domain = m_subdomain;
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
#ifdef SAMURAI_WITH_MPI
        mpi::communicator world;
        if (m_mpi_neighbourhood.empty() && world.size() > 1)
        {
            find_neighbourhood();
        }
#endif
    }

    template <class D, class Config>
    inline void Mesh_base<D, Config>::construct_union()
    {
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
        for (std::size_t level = max_lvl; level >= 1; --level)
        {
            lcl_type lcl{level - 1};
            auto expr = union_(this->m_cells[mesh_id_t::cells][level], m_union[level]).on(level - 1);

            expr(
                [&](const auto& interval, const auto& index_yz)
                {
                    lcl[index_yz].add_interval(interval);
                });

            m_union[level - 1] = {lcl};
        }
    }

    template <class D, class Config>
    void Mesh_base<D, Config>::find_neighbourhood()
    {
#ifdef SAMURAI_WITH_MPI
        mpi::communicator world;

        std::vector<lca_type> neighbours(static_cast<std::size_t>(world.size()));
        mpi::all_gather(world, m_subdomain, neighbours);
        std::set<int> set_neighbours;
        for (std::size_t i = 0; i < neighbours.size(); ++i)
        {
            if (i != static_cast<std::size_t>(world.rank()))
            {
                auto set = intersection(nestedExpand(m_subdomain, 1), neighbours[i]);
                if (!set.empty())
                {
                    set_neighbours.insert(static_cast<int>(i));
                }
                for (std::size_t d = 0; d < dim; ++d)
                {
                    if (m_periodic[d])
                    {
                        auto shift             = get_periodic_shift(m_domain, m_subdomain.level(), d);
                        auto periodic_set_left = intersection(nestedExpand(m_subdomain, 1), translate(neighbours[i], -shift));
                        if (!periodic_set_left.empty())
                        {
                            set_neighbours.insert(static_cast<int>(i));
                        }
                        auto periodic_set_right = intersection(nestedExpand(m_subdomain, 1), translate(neighbours[i], shift));
                        if (!periodic_set_right.empty())
                        {
                            set_neighbours.insert(static_cast<int>(i));
                        }
                    }
                }
            }
        }
        m_mpi_neighbourhood.clear();
        m_mpi_neighbourhood.reserve(set_neighbours.size());
        for (const auto& neighbour : set_neighbours)
        {
            m_mpi_neighbourhood.emplace_back(neighbour);
        }
#endif
    }

    template <class D, class Config>
    void Mesh_base<D, Config>::partition_mesh([[maybe_unused]] std::size_t start_level, [[maybe_unused]] const Box<double, dim>& global_box)
    {
#ifdef SAMURAI_WITH_MPI
        mpi::communicator world;
        auto rank = world.rank();
        auto size = world.size();

        std::size_t subdomain_start = 0;
        std::size_t subdomain_end   = 0;
        lcl_type subdomain_cells(start_level, m_domain.origin_point(), m_domain.scaling_factor());
        // in 1D MPI, we need a specific partitioning
        if (dim == 1)
        {
            std::size_t n_cells               = m_domain.nb_cells();
            std::size_t n_cells_per_subdomain = n_cells / static_cast<std::size_t>(size);
            subdomain_start                   = n_cells_per_subdomain * static_cast<std::size_t>(rank);
            subdomain_end                     = n_cells_per_subdomain * (static_cast<std::size_t>(rank) + 1);
            // for the last rank, we have to take all the last cells;
            if (rank == size - 1)
            {
                subdomain_end = n_cells;
            }
            for_each_meshinterval(m_domain,
                                  [&](auto mi)
                                  {
                                      for (auto i = mi.i.start; i < mi.i.end; ++i)
                                      {
                                          if (static_cast<std::size_t>(i) >= subdomain_start && static_cast<std::size_t>(i) < subdomain_end)
                                          {
                                              subdomain_cells[mi.index].add_point(i);
                                          }
                                      }
                                  });
        }
        else if (dim >= 2)
        {
            auto subdomain_nb_intervals = m_domain.nb_intervals() / static_cast<std::size_t>(size);
            subdomain_start             = static_cast<std::size_t>(rank) * subdomain_nb_intervals;
            subdomain_end               = (static_cast<std::size_t>(rank) + 1) * subdomain_nb_intervals;
            if (rank == size - 1)
            {
                subdomain_end = m_domain.nb_intervals();
            }
            std::size_t k = 0;
            for_each_meshinterval(m_domain,
                                  [&](auto mi)
                                  {
                                      if (k >= subdomain_start && k < subdomain_end)
                                      {
                                          subdomain_cells[mi.index].add_interval(mi.i);
                                      }
                                      ++k;
                                  });
        }

        this->m_cells[mesh_id_t::cells][start_level] = subdomain_cells;
#endif
    }

    template <class D, class Config>
    void Mesh_base<D, Config>::load_balancing()
    {
#ifdef SAMURAI_WITH_MPI
        mpi::communicator world;
        auto rank = world.rank();

        std::size_t load = nb_cells(mesh_id_t::cells);
        std::vector<std::size_t> loads;

        std::vector<double> load_fluxes(m_mpi_neighbourhood.size(), 0);

        const std::size_t n_iterations = 1;

        for (std::size_t k = 0; k < n_iterations; ++k)
        {
            world.barrier();
            if (rank == 0)
            {
                std::cout << "---------------- k = " << k << " ----------------" << std::endl;
            }
            mpi::all_gather(world, load, loads);

            std::vector<std::size_t> nb_neighbours;
            mpi::all_gather(world, m_mpi_neighbourhood.size(), nb_neighbours);

            double load_np1 = static_cast<double>(load);
            for (std::size_t i_rank = 0; i_rank < m_mpi_neighbourhood.size(); ++i_rank)
            {
                auto neighbour = m_mpi_neighbourhood[i_rank];

                auto neighbour_load = loads[static_cast<std::size_t>(neighbour.rank)];
                int neighbour_load_minus_my_load;
                if (load < neighbour_load)
                {
                    neighbour_load_minus_my_load = static_cast<int>(neighbour_load - load);
                }
                else
                {
                    neighbour_load_minus_my_load = -static_cast<int>(load - neighbour_load);
                }
                double weight       = 1. / std::max(m_mpi_neighbourhood.size(), nb_neighbours[neighbour.rank]);
                load_fluxes[i_rank] = weight * neighbour_load_minus_my_load;
                load_np1 += load_fluxes[i_rank];
            }
            load_np1 = floor(load_np1);

            load_transfer(load_fluxes);

            std::cout << rank << ": load = " << load << ", load_np1 = " << load_np1 << std::endl;

            load = static_cast<std::size_t>(load_np1);
        }
#endif
    }

    template <class D, class Config>
    void Mesh_base<D, Config>::load_transfer([[maybe_unused]] const std::vector<double>& load_fluxes)
    {
#ifdef SAMURAI_WITH_MPI
        mpi::communicator world;
        std::cout << world.rank() << ": ";
        for (std::size_t i_rank = 0; i_rank < m_mpi_neighbourhood.size(); ++i_rank)
        {
            auto neighbour = m_mpi_neighbourhood[i_rank];
            if (load_fluxes[i_rank] < 0) // must tranfer load to the neighbour
            {
            }
            else if (load_fluxes[i_rank] > 0) // must receive load from the neighbour
            {
            }
            std::cout << "--> " << neighbour.rank << ": " << load_fluxes[i_rank] << ", ";
        }
        std::cout << std::endl;
#endif
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
