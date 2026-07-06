// Copyright 2018-2025 the samurai's authors
// SPDX-License-Identifier:  BSD-3-Clause

#pragma once

#include <array>
#include <cmath>
#include <set>
#include <tuple>
#include <utility>

#include <fmt/format.h>

#include "cell_array.hpp"
#include "cell_list.hpp"
#include "domain_builder.hpp"
#include "mesh_config.hpp"
#include "petsc/cell_ownership.hpp"
#include "static_algorithm.hpp"
#include "stencil.hpp"
#include "subset/node.hpp"

#ifdef SAMURAI_WITH_MPI
#include <boost/serialization/vector.hpp>

#include "mpi/subdomain_bbox.hpp"
#include <boost/mpi.hpp>
#include <boost/mpi/cartesian_communicator.hpp>
namespace mpi = boost::mpi;
#endif

namespace samurai
{
    template <class Config>
    struct get_mesh_id;

    template <class Config>
    using get_mesh_id_t = typename get_mesh_id<Config>::type;

    namespace detail
    {
        template <std::size_t d, std::size_t dim>
        void get_periodic_directions(std::array<bool, dim> is_periodic,
                                     std::vector<DirectionVector<dim>>& directions,
                                     DirectionVector<dim>& current)
        {
            auto next = [&]()
            {
                if constexpr (d == dim - 1)
                {
                    directions.push_back(current);
                }
                else
                {
                    get_periodic_directions<d + 1>(is_periodic, directions, current);
                }
            };

            if (is_periodic[d])
            {
                current[d] = -1;
                next();

                current[d] = 1;
                next();
            }
            current[d] = 0;
            next();
        }

        template <std::size_t dim>
        auto get_periodic_directions(const std::array<bool, dim>& is_periodic)
        {
            DirectionVector<dim> current{};
            std::vector<DirectionVector<dim>> directions;
            get_periodic_directions<0>(is_periodic, directions, current);
            directions.pop_back();
            return directions;
        }
    }

    template <class CellArray, class MeshID>
    struct MeshIDArray : private std::array<CellArray, static_cast<std::size_t>(MeshID::count)>
    {
        static constexpr std::size_t size = static_cast<std::size_t>(MeshID::count);
        using base_type                   = std::array<CellArray, size>;
        using base_type::operator[];

        SAMURAI_INLINE const CellArray& operator[](MeshID mesh_id) const
        {
            return operator[](static_cast<std::size_t>(mesh_id));
        }

        SAMURAI_INLINE CellArray& operator[](MeshID mesh_id)
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
        using config_t  = Config;

        static constexpr std::size_t dim                  = config_t::dim;
        static constexpr std::size_t max_refinement_level = config_t::max_refinement_level;

        using mesh_id_t  = get_mesh_id_t<D>;
        using interval_t = typename config_t::interval_t;
        using value_t    = typename interval_t::value_t;
        using index_t    = typename interval_t::index_t;

        using cl_type  = CellList<dim, interval_t, max_refinement_level>;
        using lcl_type = typename cl_type::lcl_type;

        using ca_type  = CellArray<dim, interval_t, max_refinement_level>;
        using lca_type = typename ca_type::lca_type;
        using cell_t   = typename ca_type::cell_t;

        using coords_t = typename lca_type::coords_t;

        using mesh_interval_t = typename ca_type::lca_type::mesh_interval_t;

        using mesh_t = samurai::MeshIDArray<ca_type, mesh_id_t>;

        using mpi_subdomain_t = MPI_Subdomain<D>;

        using CellOwnership = samurai::petsc::CellOwnership;

        std::size_t nb_cells(mesh_id_t mesh_id = mesh_id_t::reference) const;
        std::size_t nb_cells(std::size_t level, mesh_id_t mesh_id = mesh_id_t::reference) const;

        const ca_type& operator[](mesh_id_t mesh_id) const;
        ca_type& operator[](mesh_id_t mesh_id);

        std::size_t max_level() const;
        std::size_t& max_level();
        std::size_t min_level() const;
        std::size_t& min_level();

        std::size_t graduation_width() const;
        int ghost_width() const;
        int max_stencil_radius() const;
        double ghost_physical_reach() const;
        auto& cfg() const;
        auto& cfg();

        auto& origin_point() const;
        void set_origin_point(const coords_t& origin_point);
        double scaling_factor() const;
        void set_scaling_factor(double scaling_factor);
        void scale_domain(double domain_scaling_factor);
        double cell_length(std::size_t level) const;
        double min_cell_length() const;
        const lca_type& domain() const;
        const lca_type& domain(std::size_t level) const;
        const lca_type& subdomain() const;
        const lca_type& subdomain(std::size_t level) const;

        void box_like();

        const ca_type& get_union() const;
        bool is_periodic() const;
        bool is_periodic(std::size_t d) const;
        const std::array<bool, dim>& periodicity() const;
        std::vector<mpi_subdomain_t>& mpi_neighbourhood();
        const std::vector<mpi_subdomain_t>& mpi_neighbourhood() const;
        const coords_t& gravity_center() const;
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

        CellOwnership& cell_ownership();
        const CellOwnership& cell_ownership() const;

      protected:

        using derived_type = D;

        Mesh_base() = default; // cppcheck-suppress uninitMemberVar
        Mesh_base(const ca_type& ca, const self_type& ref_mesh);
        Mesh_base(const cl_type& cl, const self_type& ref_mesh);
        Mesh_base(const cl_type& cl, const config_t& config);
        Mesh_base(const ca_type& ca, const config_t& config);
        Mesh_base(const samurai::Box<double, dim>& b, const config_t& config);

        Mesh_base(const samurai::DomainBuilder<dim>& domain_builder, const config_t& config);

        // cppcheck-suppress uninitMemberVar
        Mesh_base(const samurai::Box<double, dim>&, std::size_t, std::size_t, std::size_t, double, double)
        {
            std::cerr << "Delete min_level and max_level from CLI11 options and use mesh_config object and the make_mesh function"
                      << std::endl;
            exit(EXIT_FAILURE);
        }

        Mesh_base(const samurai::DomainBuilder<dim>& domain_builder,
                  std::size_t start_level,
                  std::size_t min_level,
                  std::size_t max_level,
                  double approx_box_tol = lca_type::default_approx_box_tol,
                  double scaling_factor = 0);

        // cppcheck-suppress uninitMemberVar
        Mesh_base(const samurai::Box<double, dim>&, std::size_t, std::size_t, std::size_t, const std::array<bool, dim>&, double, double)
        {
            std::cerr << "Delete min_level and max_level from CLI11 options and use mesh_config object and the make_mesh function"
                      << std::endl;
            exit(EXIT_FAILURE);
        }

        derived_type& derived_cast() & noexcept;
        const derived_type& derived_cast() const& noexcept;
        derived_type derived_cast() && noexcept;

        mesh_t& cells();

      private:

        void construct_subdomain();
        void construct_domain();
        void build_pyramid(ca_type& pyramid, const lca_type& reference);
        void construct_union();
        void construct_corners();
        void update_sub_mesh();
        void renumbering();
        void find_neighbourhood();

        void compute_gravity_center();

        void partition_mesh(std::size_t start_level, const Box<double, dim>& global_box);
        std::size_t max_nb_cells(std::size_t level) const;

#ifdef SAMURAI_WITH_MPI
        // Helper methods for optimized neighbor finding
        void add_periodic_candidates(const std::vector<mpi_neighbor::SubdomainBoundingBox<dim>>& all_bboxes,
                                     const mpi_neighbor::SubdomainBoundingBox<dim>& my_bbox,
                                     std::set<int>& candidates) const;
#endif

        ca_type m_domain;
        ca_type m_subdomain;
        mesh_t m_cells;
        ca_type m_union;
        std::vector<lca_type> m_corners;
        std::vector<mpi_subdomain_t> m_mpi_neighbourhood;
        coords_t m_gravity_center;

        config_t m_config;

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
            ar & m_config;
        }
#endif

#ifdef SAMURAI_WITH_PETSC
        CellOwnership m_cell_ownership;
#endif
    };

    template <class D, class Config>
    SAMURAI_INLINE auto Mesh_base<D, Config>::derived_cast() & noexcept -> derived_type&
    {
        return *static_cast<derived_type*>(this);
    }

    template <class D, class Config>
    SAMURAI_INLINE auto Mesh_base<D, Config>::derived_cast() const& noexcept -> const derived_type&
    {
        return *static_cast<const derived_type*>(this);
    }

    template <class D, class Config>
    SAMURAI_INLINE auto Mesh_base<D, Config>::derived_cast() && noexcept -> derived_type
    {
        return *static_cast<derived_type*>(this);
    }

    template <class D, class Config>
    SAMURAI_INLINE Mesh_base<D, Config>::Mesh_base(const samurai::Box<double, dim>& b, const config_t& config)
        : m_config(config)
    {
        lca_type domain_ref(m_config.start_level(), b, m_config.approx_box_tol(), m_config.scaling_factor());
        build_pyramid(m_domain, domain_ref);

#ifdef SAMURAI_WITH_MPI
        partition_mesh(m_config.start_level(), b);
#else
        this->m_cells[mesh_id_t::cells][m_config.start_level()] = m_domain[m_config.start_level()];
#endif

        construct_subdomain();
        construct_union();
#ifdef SAMURAI_WITH_MPI
        find_neighbourhood();
        update_neighbour_subdomain();
#endif
        update_meshid_neighbour(mesh_id_t::cells);
        update_sub_mesh();
        construct_corners();
        renumbering();
        set_origin_point(origin_point());
        set_scaling_factor(scaling_factor());
        update_mesh_neighbour();
#if defined(SAMURAI_WITH_MPI) && defined(SAMURAI_WITH_PETSC)
        compute_gravity_center();
#endif
    }

    template <class D, class Config>
    Mesh_base<D, Config>::Mesh_base([[maybe_unused]] const samurai::DomainBuilder<dim>& domain_builder, const config_t& config)
        : m_config(config)
    {
        if (std::any_of(config.periodic().begin(),
                        config.periodic().end(),
                        [](bool b)
                        {
                            return b;
                        }))
        {
            std::cerr << "Periodicity is not implemented with DomainBuilder." << std::endl;
            exit(EXIT_FAILURE);
        }

#ifdef SAMURAI_WITH_MPI
        std::cerr << "MPI is not implemented with DomainBuilder." << std::endl;
        std::exit(EXIT_FAILURE);
#else
        double scaling_factor_ = m_config.scaling_factor();
        compute_scaling_factor(domain_builder, scaling_factor_);
        m_config.scaling_factor() = scaling_factor_;

        // Build the domain by adding and removing boxes
        cl_type domain_cl = construct_initial_mesh(domain_builder,
                                                   m_config.start_level(),
                                                   m_config.approx_box_tol(),
                                                   m_config.scaling_factor());

        m_cells[mesh_id_t::cells] = {domain_cl, false};
#endif

        construct_subdomain();
        m_domain = m_subdomain;
        construct_union();
#ifdef SAMURAI_WITH_MPI
        find_neighbourhood();
        update_neighbour_subdomain();
#endif
        update_meshid_neighbour(mesh_id_t::cells);
        update_sub_mesh();
        construct_corners();
        renumbering();
        set_origin_point(domain_builder.origin_point());
        set_scaling_factor(m_config.scaling_factor());
        update_mesh_neighbour();
#if defined(SAMURAI_WITH_MPI) && defined(SAMURAI_WITH_PETSC)
        compute_gravity_center();
        // set_scaling_factor(scaling_factor_);
#endif
    }

    template <class D, class Config>
    SAMURAI_INLINE Mesh_base<D, Config>::Mesh_base(const cl_type& cl, const config_t& config)
        : m_config(config)
    {
        m_cells[mesh_id_t::cells] = {cl, false};

        construct_subdomain();
        construct_domain();
#ifdef SAMURAI_WITH_MPI
        find_neighbourhood();
        update_neighbour_subdomain();
#endif
        construct_union();
        update_meshid_neighbour(mesh_id_t::cells);
        update_sub_mesh();
        construct_corners();
        renumbering();
        set_origin_point(cl.origin_point());
        set_scaling_factor(cl.scaling_factor());
        update_mesh_neighbour();

#if defined(SAMURAI_WITH_MPI) && defined(SAMURAI_WITH_PETSC)
        compute_gravity_center();
#endif
    }

    template <class D, class Config>
    SAMURAI_INLINE Mesh_base<D, Config>::Mesh_base(const ca_type& ca, const config_t& config)
        : m_config(config)
    {
        m_cells[mesh_id_t::cells] = ca;

        construct_subdomain();
        construct_domain();
#ifdef SAMURAI_WITH_MPI
        find_neighbourhood();
        update_neighbour_subdomain();
#endif
        construct_union();
        update_meshid_neighbour(mesh_id_t::cells);
        update_sub_mesh();
        construct_corners();
        renumbering();
        set_origin_point(ca.origin_point());
        set_scaling_factor(ca.scaling_factor());
        update_mesh_neighbour();
#if defined(SAMURAI_WITH_MPI) && defined(SAMURAI_WITH_PETSC)
        compute_gravity_center();
#endif
    }

    template <class D, class Config>
    SAMURAI_INLINE Mesh_base<D, Config>::Mesh_base(const ca_type& ca, const self_type& ref_mesh)
        : m_domain(ref_mesh.m_domain)
        , m_mpi_neighbourhood(ref_mesh.m_mpi_neighbourhood)
        , m_config(ref_mesh.m_config)
    {
        m_cells[mesh_id_t::cells] = ca;

        construct_subdomain();
        construct_union();
#ifdef SAMURAI_WITH_MPI
        find_neighbourhood();
        update_neighbour_subdomain();
#endif
        update_meshid_neighbour(mesh_id_t::cells);
        update_sub_mesh();
        construct_corners();
        renumbering();
        set_origin_point(ref_mesh.origin_point());
        set_scaling_factor(ref_mesh.scaling_factor());
        update_mesh_neighbour();
#if defined(SAMURAI_WITH_MPI) && defined(SAMURAI_WITH_PETSC)
        compute_gravity_center();
#endif
    }

    template <class D, class Config>
    SAMURAI_INLINE Mesh_base<D, Config>::Mesh_base(const cl_type& cl, const self_type& ref_mesh)
        : m_domain(ref_mesh.m_domain)
        , m_mpi_neighbourhood(ref_mesh.m_mpi_neighbourhood)
        , m_config(ref_mesh.m_config)
    {
        m_cells[mesh_id_t::cells] = {cl, false};

        construct_subdomain();
        construct_union();
#ifdef SAMURAI_WITH_MPI
        find_neighbourhood();
        update_neighbour_subdomain();
#endif
        update_meshid_neighbour(mesh_id_t::cells);
        update_sub_mesh();
        construct_corners();
        renumbering();
        set_origin_point(ref_mesh.origin_point());
        set_scaling_factor(ref_mesh.scaling_factor());
        update_mesh_neighbour();
#if defined(SAMURAI_WITH_MPI) && defined(SAMURAI_WITH_PETSC)
        compute_gravity_center();
#endif
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
        std::size_t min_level_bc = min_level();
        if (scaling_factor <= 0)
        {
            scaling_factor = domain_builder.largest_subdivision();

            auto largest_cell_length = samurai::cell_length(scaling_factor, min_level_bc);
            for (const auto& box : domain_builder.removed_boxes())
            {
                while (box.min_length() < 2 * largest_cell_length * max_stencil_radius())
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
                if (box.min_length() < 2 * largest_cell_length * max_stencil_radius())
                {
                    std::cerr << "The hole " << box << " is too small to apply the BC at level " << min_level_bc
                              << " with the given scaling factor. We need to be able to construct " << (2 * max_stencil_radius())
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
    SAMURAI_INLINE auto Mesh_base<D, Config>::cells() -> mesh_t&
    {
        return m_cells;
    }

    template <class D, class Config>
    SAMURAI_INLINE std::size_t Mesh_base<D, Config>::max_nb_cells(std::size_t level) const
    {
        if (m_cells[mesh_id_t::reference][level][0].empty())
        {
            return 0;
        }
        auto last_xinterval = m_cells[mesh_id_t::reference][level][0].back();
        return static_cast<std::size_t>(static_cast<index_t>(last_xinterval.start) + last_xinterval.index) + last_xinterval.size();
    }

    template <class D, class Config>
    SAMURAI_INLINE std::size_t Mesh_base<D, Config>::nb_cells(mesh_id_t mesh_id) const
    {
        return (mesh_id == mesh_id_t::reference) ? max_nb_cells(m_cells[mesh_id].max_level()) : m_cells[mesh_id].nb_cells();
    }

    template <class D, class Config>
    SAMURAI_INLINE std::size_t Mesh_base<D, Config>::nb_cells(std::size_t level, mesh_id_t mesh_id) const
    {
        return (mesh_id == mesh_id_t::reference) ? max_nb_cells(level) : m_cells[mesh_id][level].nb_cells();
    }

    template <class D, class Config>
    SAMURAI_INLINE auto Mesh_base<D, Config>::operator[](mesh_id_t mesh_id) const -> const ca_type&
    {
        return m_cells[mesh_id];
    }

    template <class D, class Config>
    SAMURAI_INLINE auto Mesh_base<D, Config>::operator[](mesh_id_t mesh_id) -> ca_type&
    {
        return m_cells[mesh_id];
    }

    template <class D, class Config>
    SAMURAI_INLINE std::size_t Mesh_base<D, Config>::max_level() const
    {
        return m_config.max_level();
    }

    template <class D, class Config>
    SAMURAI_INLINE std::size_t& Mesh_base<D, Config>::max_level()
    {
        return m_config.max_level();
    }

    template <class D, class Config>
    SAMURAI_INLINE std::size_t Mesh_base<D, Config>::min_level() const
    {
        return m_config.min_level();
    }

    template <class D, class Config>
    SAMURAI_INLINE std::size_t& Mesh_base<D, Config>::min_level()
    {
        return m_config.min_level();
    }

    template <class D, class Config>
    SAMURAI_INLINE std::size_t Mesh_base<D, Config>::graduation_width() const
    {
        return m_config.graduation_width();
    }

    template <class D, class Config>
    SAMURAI_INLINE int Mesh_base<D, Config>::ghost_width() const
    {
        return m_config.ghost_width();
    }

    template <class D, class Config>
    SAMURAI_INLINE int Mesh_base<D, Config>::max_stencil_radius() const
    {
        return m_config.max_stencil_radius();
    }

    /**
     * Physical reach of the ghost construction around the local cells, used by
     * find_neighbourhood() to decide which ranks must be MPI neighbours: any
     * rank owning cells closer than this distance may contribute values to our
     * ghosts (and conversely), even when a thin third subdomain lies in
     * between.
     *
     * The bound is derived from the ghost construction of update_sub_mesh():
     *  - scheme ghosts: both subdomains are expanded by R = max_stencil_radius
     *    cells at level l, hence an interaction range of up to 2R cells of
     *    level l between real cells;
     *  - MRA prediction ghosts (only when min_level != max_level): expansion
     *    by P = prediction_stencil_radius cells at levels l-1 and l-2 around
     *    the already R-expanded cells, plus the outward rounding of the
     *    .on(l-1)/.on(l-2) projections, bounded by 4P + 4 extra cells of
     *    level l.
     * The reach is evaluated at the coarsest level locally present (largest
     * cells, hence the largest physical footprint). Overestimation only costs
     * a few extra neighbours in the exchange lists.
     */
    template <class D, class Config>
    SAMURAI_INLINE double Mesh_base<D, Config>::ghost_physical_reach() const
    {
        const auto& my_cells = m_cells[mesh_id_t::cells];
        if (my_cells.nb_cells() == 0)
        {
            return 0.;
        }
        const int radius          = max_stencil_radius();
        const int prediction_size = (min_level() != max_level()) ? config_t::prediction_stencil_radius : 0;
        const int reach           = (2 * radius) + (4 * prediction_size) + 4;
        return reach * cell_length(my_cells.min_level());
    }

    template <class D, class Config>
    SAMURAI_INLINE auto& Mesh_base<D, Config>::cfg() const
    {
        return m_config;
    }

    template <class D, class Config>
    SAMURAI_INLINE auto& Mesh_base<D, Config>::cfg()
    {
        return m_config;
    }

    template <class D, class Config>
    SAMURAI_INLINE auto& Mesh_base<D, Config>::origin_point() const
    {
        return m_domain.origin_point();
    }

    template <class D, class Config>
    SAMURAI_INLINE void Mesh_base<D, Config>::set_origin_point(const coords_t& origin_point)
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
    SAMURAI_INLINE double Mesh_base<D, Config>::scaling_factor() const
    {
        return m_domain.scaling_factor();
    }

    template <class D, class Config>
    SAMURAI_INLINE void Mesh_base<D, Config>::set_scaling_factor(double scaling_factor)
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
    SAMURAI_INLINE void Mesh_base<D, Config>::scale_domain(double domain_scaling_factor)
    {
        set_scaling_factor(domain_scaling_factor * scaling_factor());
    }

    template <class D, class Config>
    SAMURAI_INLINE double Mesh_base<D, Config>::cell_length(std::size_t level) const
    {
        return samurai::cell_length(scaling_factor(), level);
    }

    template <class D, class Config>
    SAMURAI_INLINE double Mesh_base<D, Config>::min_cell_length() const
    {
        return cell_length(max_level());
    }

    template <class D, class Config>
    SAMURAI_INLINE auto Mesh_base<D, Config>::domain() const -> const lca_type&
    {
        return m_domain[max_level()];
    }

    template <class D, class Config>
    SAMURAI_INLINE auto Mesh_base<D, Config>::domain(std::size_t level) const -> const lca_type&
    {
        return m_domain[level];
    }

    template <class D, class Config>
    SAMURAI_INLINE auto Mesh_base<D, Config>::subdomain() const -> const lca_type&
    {
        return m_subdomain[max_level()];
    }

    template <class D, class Config>
    SAMURAI_INLINE auto Mesh_base<D, Config>::subdomain(std::size_t level) const -> const lca_type&
    {
        return m_subdomain[level];
    }

    template <class D, class Config>
    SAMURAI_INLINE void Mesh_base<D, Config>::box_like()
    {
        for (std::size_t level = min_level(); level <= max_level(); ++level)
        {
            m_domain[level].box_like();
        }
    }

    template <class D, class Config>
    SAMURAI_INLINE auto Mesh_base<D, Config>::get_union() const -> const ca_type&
    {
        return m_union;
    }

    template <class D, class Config>
    template <typename... T, typename U>
    SAMURAI_INLINE auto
    Mesh_base<D, Config>::get_interval(std::size_t level, const interval_t& interval, T... index) const -> const interval_t&
    {
        return m_cells[mesh_id_t::reference].get_interval(level, interval, index...);
    }

    template <class D, class Config>
    template <class E>
    SAMURAI_INLINE auto Mesh_base<D, Config>::get_interval(std::size_t level,
                                                           const interval_t& interval,
                                                           const xt::xexpression<E>& index) const -> const interval_t&
    {
        return m_cells[mesh_id_t::reference].get_interval(level, interval, index);
    }

    template <class D, class Config>
    template <class E>
    SAMURAI_INLINE auto Mesh_base<D, Config>::get_interval(std::size_t level, const xt::xexpression<E>& coord) const -> const interval_t&
    {
        return m_cells[mesh_id_t::reference].get_interval(level, coord);
    }

    template <class D, class Config>
    template <typename... T, typename U>
    SAMURAI_INLINE auto Mesh_base<D, Config>::get_index(std::size_t level, value_t i, T... index) const -> index_t
    {
        return m_cells[mesh_id_t::reference].get_index(level, i, index...);
    }

    template <class D, class Config>
    template <class E>
    SAMURAI_INLINE auto Mesh_base<D, Config>::get_index(std::size_t level, value_t i, const xt::xexpression<E>& others) const -> index_t
    {
        return m_cells[mesh_id_t::reference].get_index(level, i, others);
    }

    template <class D, class Config>
    template <class E>
    SAMURAI_INLINE auto Mesh_base<D, Config>::get_index(std::size_t level, const xt::xexpression<E>& coord) const -> index_t
    {
        return m_cells[mesh_id_t::reference].get_index(level, coord);
    }

    template <class D, class Config>
    template <typename... T, typename U>
    SAMURAI_INLINE auto Mesh_base<D, Config>::get_cell(std::size_t level, value_t i, T... index) const -> cell_t
    {
        return m_cells[mesh_id_t::reference].get_cell(level, i, index...);
    }

    template <class D, class Config>
    template <class E>
    SAMURAI_INLINE auto Mesh_base<D, Config>::get_cell(std::size_t level, value_t i, const xt::xexpression<E>& index) const -> cell_t
    {
        return m_cells[mesh_id_t::reference].get_cell(level, i, index);
    }

    template <class D, class Config>
    template <class E>
    SAMURAI_INLINE auto Mesh_base<D, Config>::get_cell(std::size_t level, const xt::xexpression<E>& coord) const -> cell_t
    {
        return m_cells[mesh_id_t::reference].get_cell(level, coord);
    }

    template <class D, class Config>
    SAMURAI_INLINE bool Mesh_base<D, Config>::is_periodic() const
    {
        return std::any_of(m_config.periodic().cbegin(),
                           m_config.periodic().cend(),
                           [](bool v)
                           {
                               return v;
                           });
    }

    template <class D, class Config>
    SAMURAI_INLINE bool Mesh_base<D, Config>::is_periodic(std::size_t d) const
    {
        return m_config.periodic(d);
    }

    template <class D, class Config>
    SAMURAI_INLINE auto Mesh_base<D, Config>::periodicity() const -> const std::array<bool, dim>&
    {
        return m_config.periodic();
    }

    template <class D, class Config>
    SAMURAI_INLINE auto Mesh_base<D, Config>::mpi_neighbourhood() -> std::vector<mpi_subdomain_t>&
    {
        return m_mpi_neighbourhood;
    }

    template <class D, class Config>
    SAMURAI_INLINE auto Mesh_base<D, Config>::mpi_neighbourhood() const -> const std::vector<mpi_subdomain_t>&
    {
        return m_mpi_neighbourhood;
    }

    template <class D, class Config>
    const typename Mesh_base<D, Config>::coords_t& Mesh_base<D, Config>::gravity_center() const
    {
        return m_gravity_center;
    }

    template <class D, class Config>
    void Mesh_base<D, Config>::compute_gravity_center()
    {
        m_gravity_center.fill(0);
        double total_volume = 0;
        for_each_interval(m_cells[mesh_id_t::cells],
                          [&](std::size_t level, auto& i, auto& index)
                          {
                              auto length            = cell_length(level);
                              double interval_volume = static_cast<double>(i.size()) * std::pow(length, dim);
                              coords_t interval_center;
                              interval_center[0] = origin_point()[0] + length * 0.5 * static_cast<double>(i.start + i.end);
                              for (std::size_t d = 1; d < dim; ++d)
                              {
                                  interval_center[d] = origin_point()[d] + length * (static_cast<double>(index[d - 1]) + 0.5);
                              }
                              m_gravity_center += interval_volume * interval_center;
                              total_volume += interval_volume;
                          });
        m_gravity_center /= total_volume;
    }

    template <class D, class Config>
    SAMURAI_INLINE void Mesh_base<D, Config>::swap(Mesh_base<D, Config>& mesh) noexcept
    {
        using std::swap;
        swap(m_cells, mesh.m_cells);
        swap(m_domain, mesh.m_domain);
        swap(m_subdomain, mesh.m_subdomain);
        swap(m_mpi_neighbourhood, mesh.m_mpi_neighbourhood);
        swap(m_union, mesh.m_union);
        swap(m_config, mesh.m_config);
    }

    template <class D, class Config>
    SAMURAI_INLINE void Mesh_base<D, Config>::update_sub_mesh()
    {
        this->derived_cast().update_sub_mesh_impl();
    }

    template <class D, class Config>
    SAMURAI_INLINE void Mesh_base<D, Config>::renumbering()
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
    SAMURAI_INLINE void Mesh_base<D, Config>::construct_corners()
    {
        static_assert(
            dim <= 6,
            "Corner construction is currently only implemented up to 6D due to the combinatorial number of cases. Please implement the general ND version if you need higher dimensions.");
        if constexpr (dim > 1)
        {
            using direction_t = DirectionVector<dim>;

            // For a diagonal direction d, the "corner" (inner cells at the boundary corner/edge) is:
            //   domain \ union_{i: d[i] != 0}( translate(domain, e_i * -d[i]) )
            // Only non-zero components are included: a zero component would give
            // translate(domain, 0) = domain, making the union contain the domain itself
            // and the difference empty. This is the correct formula for both true corner
            // directions (all components ±1) and edge directions in 3D (one zero component).
            m_corners.clear();
            const auto& domain_lca = m_domain[max_level()];
            for_each_diagonal_direction<dim>(
                [&](const auto& direction)
                {
                    // Collect shifts only for non-zero components
                    std::array<direction_t, dim> nonzero_shifts;
                    std::size_t n = 0;
                    for (std::size_t I = 0; I < dim; ++I)
                    {
                        if (direction[I] != 0)
                        {
                            nonzero_shifts[n].fill(0);
                            nonzero_shifts[n][I] = -direction[I];
                            ++n;
                        }
                    }

                    // Apply union_ to non-zero-component translates only.
                    // for_each_diagonal_direction guarantees n >= 2.
                    // In 3D: n == 2 for edge directions, n == 3 for true corner directions.
                    // TODO: make a ND version
                    if (n == 2)
                    {
                        m_corners.push_back(
                            difference(domain_lca, union_(translate(domain_lca, nonzero_shifts[0]), translate(domain_lca, nonzero_shifts[1])))
                                .to_lca());
                    }
                    else if (n == 3)
                    {
                        m_corners.push_back(difference(domain_lca,
                                                       union_(translate(domain_lca, nonzero_shifts[0]),
                                                              translate(domain_lca, nonzero_shifts[1]),
                                                              translate(domain_lca, nonzero_shifts[2])))
                                                .to_lca());
                    }
                    else if (n == 4)
                    {
                        m_corners.push_back(difference(domain_lca,
                                                       union_(translate(domain_lca, nonzero_shifts[0]),
                                                              translate(domain_lca, nonzero_shifts[1]),
                                                              translate(domain_lca, nonzero_shifts[2]),
                                                              translate(domain_lca, nonzero_shifts[3])))
                                                .to_lca());
                    }
                    else if (n == 5)
                    {
                        m_corners.push_back(difference(domain_lca,
                                                       union_(translate(domain_lca, nonzero_shifts[0]),
                                                              translate(domain_lca, nonzero_shifts[1]),
                                                              translate(domain_lca, nonzero_shifts[2]),
                                                              translate(domain_lca, nonzero_shifts[3]),
                                                              translate(domain_lca, nonzero_shifts[4])))
                                                .to_lca());
                    }
                    else if (n == 6)
                    {
                        m_corners.push_back(difference(domain_lca,
                                                       union_(translate(domain_lca, nonzero_shifts[0]),
                                                              translate(domain_lca, nonzero_shifts[1]),
                                                              translate(domain_lca, nonzero_shifts[2]),
                                                              translate(domain_lca, nonzero_shifts[3]),
                                                              translate(domain_lca, nonzero_shifts[4]),
                                                              translate(domain_lca, nonzero_shifts[5])))
                                                .to_lca());
                    }
                });
        }
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

#ifdef SAMURAI_WITH_PETSC
    template <class D, class Config>
    SAMURAI_INLINE samurai::petsc::CellOwnership& Mesh_base<D, Config>::cell_ownership()
    {
        return m_cell_ownership;
    }

    template <class D, class Config>
    SAMURAI_INLINE const samurai::petsc::CellOwnership& Mesh_base<D, Config>::cell_ownership() const
    {
        return m_cell_ownership;
    }
#endif

    template <class D, class Config>
    SAMURAI_INLINE void Mesh_base<D, Config>::update_mesh_neighbour()
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

#ifdef SAMURAI_WITH_PETSC
        for (auto& neighbour : m_mpi_neighbourhood)
        {
            neighbour.mesh.compute_gravity_center();
        }
#endif
#endif
    }

    // TODO : find a clever way to factorize the two next functions. For new, I have to duplicate the code 2 times.

    // This function is to only send m_subdomain instead of the whole mesh data
    template <class D, class Config>
    SAMURAI_INLINE void Mesh_base<D, Config>::update_neighbour_subdomain()
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
            neighbour.mesh.m_domain = m_domain;
#ifdef SAMURAI_WITH_PETSC
            neighbour.mesh.compute_gravity_center();
#endif
        }

        mpi::wait_all(req.begin(), req.end());
#endif
    }

    // Modified function definition
    template <class D, class Config>
    SAMURAI_INLINE void Mesh_base<D, Config>::update_meshid_neighbour([[maybe_unused]] const mesh_id_t& mesh_id)
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
    SAMURAI_INLINE void Mesh_base<D, Config>::construct_domain()
    {
#ifdef SAMURAI_WITH_MPI
        lcl_type lcl = {max_level()};
        mpi::communicator world;
        std::vector<lca_type> all_subdomains(static_cast<std::size_t>(world.size()));
        mpi::all_gather(world, m_subdomain[max_level()], all_subdomains);

        for (std::size_t k = 0; k < all_subdomains.size(); ++k)
        {
            for_each_interval(all_subdomains[k],
                              [&](auto, const auto& i, const auto& index)
                              {
                                  lcl[index].add_interval(i);
                              });
        }

        build_pyramid(m_domain, lca_type{lcl});
#else
        m_domain = m_subdomain;
#endif
    }

    template <class D, class Config>
    SAMURAI_INLINE void Mesh_base<D, Config>::construct_subdomain()
    {
        // TODO: Don't build subdomain when we are in serial or in parallel with only one rank. This is a waste of memory and time.
        // Just use the domain as subdomain in this case.
        lcl_type lcl = {max_level(), m_domain.origin_point(), m_domain.scaling_factor()};

        for_each_interval(m_cells[mesh_id_t::cells],
                          [&](std::size_t level, const auto& i, const auto& index)
                          {
                              std::size_t shift = max_level() - level;
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
        build_pyramid(m_subdomain, lca_type{lcl});
    }

    // Materialises `reference` (a full-resolution LCA defined at its own
    // level) into every level of `pyramid` from 0 to max_level(), coarsening
    // or refining as needed via the usual `.on(level)` projection. Doing this
    // once, here, instead of lazily in the ghost update (previously
    // `self(mesh.domain()).on(level)` / `self(mesh.subdomain()).on(level)`,
    // recomputed on every call) is the whole point: domain(level) and
    // subdomain(level) become plain array look-ups. The range must cover 0,
    // not just [min_level(), max_level()]: update_ghost_mr_aggregated walks
    // levels down to 0 regardless of mesh.min_level().
    template <class D, class Config>
    SAMURAI_INLINE void Mesh_base<D, Config>::build_pyramid(ca_type& pyramid, const lca_type& reference)
    {
        const std::size_t hi = std::max(max_level(), reference.level());
        pyramid.clear();
        pyramid.set_origin_point(reference.origin_point());
        pyramid.set_scaling_factor(reference.scaling_factor());

        for (std::size_t level = 0; level <= hi; ++level)
        {
            if (level == reference.level())
            {
                pyramid[level] = reference;
                continue;
            }

            self(reference).on(level)(
                [&](const auto& i, const auto& index)
                {
                    pyramid[level].add_interval_back(i, index);
                });
        }
    }

    template <class D, class Config>
    SAMURAI_INLINE void Mesh_base<D, Config>::construct_union()
    {
        std::size_t max_lvl = max_level();

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
        const int rank = world.rank();
        const int size = world.size();

        // Phase 1: Exchange bounding boxes (64 bytes each vs KB of mesh data)
        auto my_bbox        = mpi_neighbor::compute_subdomain_bbox(m_subdomain[max_level()]);
        my_bbox.rank        = rank;
        my_bbox.ghost_reach = ghost_physical_reach();

        std::vector<mpi_neighbor::SubdomainBoundingBox<dim>> all_bboxes(static_cast<std::size_t>(size));
        mpi::all_gather(world, my_bbox, all_bboxes);

        // Phase 2: Fast screening with bounding boxes
        std::set<int> candidates;

        for (int i = 0; i < size; ++i)
        {
            if (i != rank)
            {
                if (my_bbox.could_be_neighbor(all_bboxes[static_cast<std::size_t>(i)]))
                {
                    candidates.insert(i);
                }
            }
        }

        // Add periodic boundary candidates
        if (is_periodic())
        {
            add_periodic_candidates(all_bboxes, my_bbox, candidates);
        }

        // Phase 3: Precise verification with interval algebra (only for candidates)
        std::vector<lca_type> candidate_subdomains;
        candidate_subdomains.reserve(candidates.size());

        // Gather only candidate subdomains
        std::vector<int> candidate_ranks(candidates.begin(), candidates.end());
        std::vector<mpi::request> requests;
        requests.reserve(candidates.size());

        for (int candidate_rank : candidate_ranks)
        {
            candidate_subdomains.emplace_back();
            requests.push_back(world.irecv(candidate_rank, 0, candidate_subdomains.back()));
        }

        // Send my subdomain to all candidates
        for (int candidate_rank : candidate_ranks)
        {
            requests.push_back(world.isend(candidate_rank, 0, m_subdomain[max_level()]));
        }

        mpi::wait_all(requests.begin(), requests.end());

        auto directions     = detail::get_periodic_directions(m_config.periodic());
        auto minmax_indices = m_domain[max_level()].minmax_indices();
        xt::xtensor_fixed<value_t, xt::xshape<dim>> domain_size;
        for (std::size_t d = 0; d < dim; ++d)
        {
            domain_size[d] = minmax_indices[d].second - minmax_indices[d].first;
        }

        // Verify candidates with precise interval algebra
        // And update neighborhood list.
        // The expansion width must cover the ghost reach of BOTH sides of the
        // pair (symmetric decision: both ranks compute the same max), converted
        // in cells at the subdomain level. The historic hardcoded width of 1
        // missed ranks lying behind a subdomain strip thinner than the ghost
        // footprint (see tests/mpi/test_lb_ghosts.cpp).
        const double subdomain_dx = m_subdomain[max_level()].cell_length();
        auto expansion_width      = [&](const auto& other_bbox)
        {
            const double reach = std::max(my_bbox.ghost_reach, other_bbox.ghost_reach);
            return std::max(1, static_cast<int>(std::ceil(reach / subdomain_dx)));
        };

        m_mpi_neighbourhood.clear();
        for (std::size_t i = 0; i < candidate_ranks.size(); ++i)
        {
            const int width = expansion_width(all_bboxes[static_cast<std::size_t>(candidate_ranks[i])]);

            auto set = intersection(nestedExpand(m_subdomain[max_level()], width), candidate_subdomains[i]);
            if (!set.empty())
            {
                m_mpi_neighbourhood.emplace_back(candidate_ranks[i]);
                continue; // No need to check periodic boundaries if they are already neighbors
            }

            // Check periodic boundaries for this candidate
            for (const auto& direction : directions)
            {
                auto shift        = direction * domain_size;
                auto periodic_set = intersection(nestedExpand(m_subdomain[max_level()], width), translate(candidate_subdomains[i], shift));

                if (!periodic_set.empty())
                {
                    m_mpi_neighbourhood.emplace_back(candidate_ranks[i]);
                    break; // No need to check other directions if we already found a neighbor
                }
            }
        }

#endif
    }

#ifdef SAMURAI_WITH_MPI
    template <class D, class Config>
    void Mesh_base<D, Config>::add_periodic_candidates(const std::vector<mpi_neighbor::SubdomainBoundingBox<dim>>& all_bboxes,
                                                       const mpi_neighbor::SubdomainBoundingBox<dim>& my_bbox,
                                                       std::set<int>& candidates) const
    {
        // For periodic boundaries, we need to check if subdomains could be neighbors
        // when wrapped around the periodic boundaries
        auto directions = detail::get_periodic_directions(m_config.periodic());

        auto domain_size = xt::eval(m_domain[max_level()].max_corner() - m_domain[max_level()].min_corner());

        for (const auto& other_bbox : all_bboxes)
        {
            if (other_bbox.rank == my_bbox.rank || candidates.contains(other_bbox.rank))
            {
                continue;
            }
            for (const auto& direction : directions)
            {
                // Check if bboxes could be neighbors through periodic boundary
                // by virtually shifting the other bbox by +/- domain_size
                mpi_neighbor::SubdomainBoundingBox<dim> shifted_bbox = other_bbox;
                shifted_bbox.bbox.min_corner() += direction * domain_size;
                shifted_bbox.bbox.max_corner() += direction * domain_size;

                if (my_bbox.could_be_neighbor(shifted_bbox))
                {
                    candidates.insert(other_bbox.rank);
                    break;
                }
            }
        }
    }
#endif

    template <class D, class Config>
    void Mesh_base<D, Config>::partition_mesh([[maybe_unused]] std::size_t start_level, [[maybe_unused]] const Box<double, dim>& global_box)
    {
#ifdef SAMURAI_WITH_MPI
        mpi::communicator world;
        auto rank = world.rank();
        auto size = world.size();

        const auto& domain_at_start_level = m_domain[start_level];

        std::size_t subdomain_start = 0;
        std::size_t subdomain_end   = 0;
        lcl_type subdomain_cells(start_level, m_domain.origin_point(), m_domain.scaling_factor());
        // in 1D MPI, we need a specific partitioning
        if (dim == 1)
        {
            std::size_t n_cells               = domain_at_start_level.nb_cells();
            std::size_t n_cells_per_subdomain = n_cells / static_cast<std::size_t>(size);
            subdomain_start                   = n_cells_per_subdomain * static_cast<std::size_t>(rank);
            subdomain_end                     = n_cells_per_subdomain * (static_cast<std::size_t>(rank) + 1);
            // for the last rank, we have to take all the last cells;
            if (rank == size - 1)
            {
                subdomain_end = n_cells;
            }
            for_each_meshinterval(domain_at_start_level,
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
            auto subdomain_nb_intervals = domain_at_start_level.nb_intervals() / static_cast<std::size_t>(size);
            subdomain_start             = static_cast<std::size_t>(rank) * subdomain_nb_intervals;
            subdomain_end               = (static_cast<std::size_t>(rank) + 1) * subdomain_nb_intervals;
            if (rank == size - 1)
            {
                subdomain_end = domain_at_start_level.nb_intervals();
            }
            std::size_t k = 0;
            for_each_meshinterval(domain_at_start_level,
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
    SAMURAI_INLINE void Mesh_base<D, Config>::to_stream(std::ostream& os) const
    {
        for (std::size_t id = 0; id < static_cast<std::size_t>(mesh_id_t::count); ++id)
        {
            auto mt = static_cast<mesh_id_t>(id);

            os << fmt::format(disable_color ? fmt::text_style() : fmt::emphasis::bold, "{}\n{:─^50}", mt, "") << std::endl;
            os << m_cells[id];
        }
    }

    template <class D, class Config>
    SAMURAI_INLINE bool operator==(const Mesh_base<D, Config>& mesh1, const Mesh_base<D, Config>& mesh2)
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
    SAMURAI_INLINE bool operator!=(const Mesh_base<D, Config>& mesh1, const Mesh_base<D, Config>& mesh2)
    {
        return !(mesh1 == mesh2);
    }

    template <class D, class Config>
    SAMURAI_INLINE std::ostream& operator<<(std::ostream& out, const Mesh_base<D, Config>& mesh)
    {
        mesh.to_stream(out);
        return out;
    }
} // namespace samurai
