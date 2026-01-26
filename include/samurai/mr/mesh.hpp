// Copyright 2018-2025 the samurai's authors
// SPDX-License-Identifier:  BSD-3-Clause

#pragma once

#include <fmt/format.h>

#include <xtensor/containers/xtensor.hpp>
#include <xtensor/views/xview.hpp>

#include "../box.hpp"
#include "../mesh.hpp"
#include "../samurai_config.hpp"
#include "../stencil.hpp"
#include "../subset/apply.hpp"
#include "../subset/node.hpp"

using namespace xt::placeholders;

namespace samurai
{
    enum class MRMeshId
    {
        cells            = 0,
        cells_and_ghosts = 1,
        proj_cells       = 2,
        union_cells      = 3,
        all_cells        = 4,
        count            = 5,
        reference        = all_cells
    };

    template <std::size_t dim_,
              std::size_t max_stencil_width_    = default_config::ghost_width,
              std::size_t graduation_width_     = default_config::graduation_width,
              std::size_t prediction_order_     = default_config::prediction_stencil_radius,
              std::size_t max_refinement_level_ = default_config::max_level,
              class TInterval                   = default_config::interval_t>
    struct [[deprecated("Use samurai::mesh_config instead")]] MRConfig
    {
        // cppcheck-suppress-begin unusedStructMember

        static constexpr std::size_t dim                  = dim_;
        static constexpr std::size_t max_refinement_level = max_refinement_level_;

        // deprecated interface
        [[deprecated("Use max_stencil_radius() method instead")]] static constexpr int max_stencil_width      = max_stencil_width_;
        [[deprecated("Use prediction_stencil_radius instead")]] static constexpr int prediction_order         = prediction_order_;
        [[deprecated("Use graduation_width() method instead")]] static constexpr std::size_t graduation_width = graduation_width_;
        [[deprecated("Use ghost_width() method instead")]] static constexpr int ghost_width = std::max(static_cast<int>(max_stencil_width),
                                                                                                       static_cast<int>(prediction_order));

        // new interface
        static constexpr int prediction_stencil_radius = prediction_order_;

        // cppcheck-suppress-end unusedStructMember

        using interval_t = TInterval;
        using mesh_id_t  = MRMeshId;
    };

    template <class Config>
    class MRMesh : public samurai::Mesh_base<MRMesh<Config>, Config>
    {
      public:

        using base_type                  = samurai::Mesh_base<MRMesh<Config>, Config>;
        using self_type                  = MRMesh<Config>;
        using mpi_subdomain_t            = typename base_type::mpi_subdomain_t;
        using config                     = typename base_type::config;
        static constexpr std::size_t dim = config::dim;

        using mesh_id_t  = typename base_type::mesh_id_t;
        using interval_t = typename base_type::interval_t;
        using cl_type    = typename base_type::cl_type;
        using lcl_type   = typename base_type::lcl_type;

        using ca_type  = typename base_type::ca_type;
        using lca_type = typename base_type::lca_type;

        using base_type::ghost_width;
        using base_type::max_stencil_radius;

        MRMesh() = default;
        MRMesh(const ca_type& ca, const self_type& ref_mesh);
        MRMesh(const cl_type& cl, const self_type& ref_mesh);
        MRMesh(const cl_type& cl, const mesh_config<Config::dim>& config);
        MRMesh(const ca_type& ca, const mesh_config<Config::dim>& config);
        MRMesh(const samurai::Box<double, dim>& b, const mesh_config<Config::dim>& config);
        MRMesh(const samurai::DomainBuilder<dim>& domain_builder, const mesh_config<Config::dim>& config);

        // deprecated constructors
        MRMesh(const samurai::Box<double, dim>& b,
               std::size_t min_level,
               std::size_t max_level,
               double approx_box_tol = lca_type::default_approx_box_tol,
               double scaling_factor = 0);
        MRMesh(const samurai::DomainBuilder<dim>& domain_builder,
               std::size_t min_level,
               std::size_t max_level,
               double approx_box_tol = lca_type::default_approx_box_tol,
               double scaling_factor = 0);
        MRMesh(const samurai::Box<double, dim>& b,
               std::size_t min_level,
               std::size_t max_level,
               const std::array<bool, dim>& periodic,
               double approx_box_tol = lca_type::default_approx_box_tol,
               double scaling_factor = 0);

        void update_sub_mesh_impl();

        template <typename... T>
        xt::xtensor<bool, 1> exists(mesh_id_t type, std::size_t level, interval_t interval, T... index) const;
    };

    template <class Config>
    SAMURAI_INLINE MRMesh<Config>::MRMesh(const ca_type& ca, const self_type& ref_mesh)
        : base_type(ca, ref_mesh)
    {
    }

    template <class Config>
    SAMURAI_INLINE MRMesh<Config>::MRMesh(const cl_type& cl, const self_type& ref_mesh)
        : base_type(cl, ref_mesh)
    {
    }

    template <class Config>
    SAMURAI_INLINE MRMesh<Config>::MRMesh(const cl_type& cl, const mesh_config<Config::dim>& config)
        : base_type(cl, config)
    {
    }

    template <class Config>
    SAMURAI_INLINE MRMesh<Config>::MRMesh(const ca_type& ca, const mesh_config<Config::dim>& config)
        : base_type(ca, config)
    {
    }

    template <class Config>
    SAMURAI_INLINE MRMesh<Config>::MRMesh(const samurai::Box<double, dim>& b, const mesh_config<Config::dim>& config)
        : base_type(b, config)
    {
    }

    template <class Config>
    SAMURAI_INLINE MRMesh<Config>::MRMesh(const samurai::DomainBuilder<dim>& domain_builder, const mesh_config<Config::dim>& config)
        : base_type(domain_builder, config)
    {
    }

    template <class Config>
    SAMURAI_INLINE MRMesh<Config>::MRMesh(const samurai::Box<double, dim>& b,
                                          std::size_t min_level,
                                          std::size_t max_level,
                                          double approx_box_tol,
                                          double scaling_factor_)
        : base_type(b,
                    mesh_config<Config::dim, Config::prediction_order, Config::max_refinement_level, typename Config::interval_t>()
                        .max_stencil_radius(Config::max_stencil_width)
                        .graduation_width(Config::graduation_width)
                        .start_level(max_level)
                        .min_level(min_level)
                        .max_level(max_level)
                        .approx_box_tol(approx_box_tol)
                        .scaling_factor(scaling_factor_))
    {
    }

    template <class Config>
    SAMURAI_INLINE MRMesh<Config>::MRMesh(const samurai::DomainBuilder<dim>& domain_builder,
                                          std::size_t min_level,
                                          std::size_t max_level,
                                          double approx_box_tol,
                                          double scaling_factor_)
        : base_type(domain_builder,
                    mesh_config<Config::dim, Config::prediction_order, Config::max_refinement_level, typename Config::interval_t>()
                        .max_stencil_radius(Config::max_stencil_width)
                        .graduation_width(Config::graduation_width)
                        .start_level(max_level)
                        .min_level(min_level)
                        .max_level(max_level)
                        .approx_box_tol(approx_box_tol)
                        .scaling_factor(scaling_factor_))
    {
    }

    template <class Config>
    SAMURAI_INLINE MRMesh<Config>::MRMesh(const samurai::Box<double, dim>& b,
                                          std::size_t min_level,
                                          std::size_t max_level,
                                          const std::array<bool, dim>& periodic,
                                          double approx_box_tol,
                                          double scaling_factor_)
        : base_type(b,
                    mesh_config<Config::dim, Config::prediction_order, Config::max_refinement_level, typename Config::interval_t>()
                        .max_stencil_radius(Config::max_stencil_width)
                        .graduation_width(Config::graduation_width)
                        .start_level(max_level)
                        .min_level(min_level)
                        .max_level(max_level)
                        .periodic(periodic)
                        .approx_box_tol(approx_box_tol)
                        .scaling_factor(scaling_factor_))
    {
    }

    template <class Config>
    SAMURAI_INLINE void MRMesh<Config>::update_sub_mesh_impl()
    {
#ifdef SAMURAI_WITH_MPI
        mpi::communicator world;
        // cppcheck-suppress redundantInitialization
        auto max_level = mpi::all_reduce(world, this->cells()[mesh_id_t::cells].max_level(), mpi::maximum<std::size_t>());
        // cppcheck-suppress redundantInitialization
        auto min_level = mpi::all_reduce(world, this->cells()[mesh_id_t::cells].min_level(), mpi::minimum<std::size_t>());
        cl_type cell_list;
#else
        // cppcheck-suppress redundantInitialization
        auto max_level = this->cells()[mesh_id_t::cells].max_level();
        // cppcheck-suppress redundantInitialization
        auto min_level = this->cells()[mesh_id_t::cells].min_level();
        cl_type cell_list;
#endif
        // Construction of ghost cells
        // ===========================
        //
        // Example with max_stencil_width = 1
        //
        // level 2                       |.|-|-|-|-|.|                   |-|
        // cells
        //                                                               |.|
        //                                                               ghost
        //                                                               cells
        // level 1             |...|---|---|...|...|---|---|...|
        //
        // level 0 |.......|-------|.......|       |.......|-------|.......|
        //
        for_each_interval(this->cells()[mesh_id_t::cells],
                          [&](std::size_t level, const auto& interval, const auto& index_yz)
                          {
                              lcl_type& lcl = cell_list[level];
                              static_nested_loop<dim - 1>(
                                  -max_stencil_radius(),
                                  max_stencil_radius() + 1,
                                  [&](auto stencil)
                                  {
                                      auto index = xt::eval(index_yz + stencil);
                                      lcl[index].add_interval({interval.start - max_stencil_radius(), interval.end + max_stencil_radius()});
                                  });
                          });
        this->cells()[mesh_id_t::cells_and_ghosts] = {cell_list, false};

        // Add cells for the MRA
        if (this->max_level() != this->min_level())
        {
            for (std::size_t level = max_level; level >= ((min_level == 0) ? 1 : min_level); --level)
            {
                auto expr = difference(intersection(this->cells()[mesh_id_t::cells_and_ghosts][level], self(this->domain()).on(level)),
                                       this->get_union()[level]);

                expr(
                    [&](const auto& interval, const auto& index_yz)
                    {
                        lcl_type& lcl = cell_list[level - 1];

                        static_nested_loop<dim - 1, -config::prediction_stencil_radius, config::prediction_stencil_radius + 1>(
                            [&](auto stencil)
                            {
                                auto new_interval = interval >> 1;
                                lcl[(index_yz >> 1) + stencil].add_interval({new_interval.start - config::prediction_stencil_radius,
                                                                             new_interval.end + config::prediction_stencil_radius});
                            });
                    });

                auto expr_2 = intersection(this->cells()[mesh_id_t::cells][level], this->cells()[mesh_id_t::cells][level]);

                expr_2(
                    [&](const auto& interval, const auto& index_yz)
                    {
                        if (level - 1 > 0)
                        {
                            lcl_type& lcl = cell_list[level - 2];

                            static_nested_loop<dim - 1, -config::prediction_stencil_radius, config::prediction_stencil_radius + 1>(
                                [&](auto stencil)
                                {
                                    auto new_interval = interval >> 2;
                                    lcl[(index_yz >> 2) + stencil].add_interval({new_interval.start - config::prediction_stencil_radius,
                                                                                 new_interval.end + config::prediction_stencil_radius});
                                });
                        }
                    });
            }
            this->cells()[mesh_id_t::all_cells] = {cell_list, false};

            this->update_meshid_neighbour(mesh_id_t::cells_and_ghosts);
            this->update_meshid_neighbour(mesh_id_t::reference);

            for (auto& neighbour : this->mpi_neighbourhood())
            {
                for (std::size_t level = 0; level <= this->max_level(); ++level)
                {
                    auto expr = intersection(this->subdomain(), neighbour.mesh[mesh_id_t::reference][level]).on(level);
                    expr(
                        [&](const auto& interval, const auto& index_yz)
                        {
                            lcl_type& lcl = cell_list[level];
                            lcl[index_yz].add_interval(interval);

                            if (level > neighbour.mesh[mesh_id_t::reference].min_level())
                            {
                                lcl_type& lclm1 = cell_list[level - 1];

                                static_nested_loop<dim - 1, -config::prediction_stencil_radius, config::prediction_stencil_radius + 1>(
                                    [&](auto stencil)
                                    {
                                        auto new_interval = interval >> 1;
                                        lclm1[(index_yz >> 1) + stencil].add_interval({new_interval.start - config::prediction_stencil_radius,
                                                                                       new_interval.end + config::prediction_stencil_radius});
                                    });
                            }
                        });
                }
            }
            this->cells()[mesh_id_t::all_cells] = {cell_list, false};

            using box_t = Box<typename interval_t::value_t, dim>;

            const auto& domain             = this->domain();
            const auto& domain_min_indices = domain.min_indices();
            const auto& domain_max_indices = domain.max_indices();

            const auto& subdomain             = this->subdomain();
            const auto& subdomain_min_indices = subdomain.min_indices();
            const auto& subdomain_max_indices = subdomain.max_indices();

            // add ghosts for periodicity
            xt::xtensor_fixed<typename interval_t::value_t, xt::xshape<dim>> shift;
            xt::xtensor_fixed<typename interval_t::value_t, xt::xshape<dim>> min_corner;
            xt::xtensor_fixed<typename interval_t::value_t, xt::xshape<dim>> max_corner;

            shift.fill(0);

#ifdef SAMURAI_WITH_MPI
            std::size_t reference_max_level = mpi::all_reduce(world,
                                                              this->cells()[mesh_id_t::reference].max_level(),
                                                              mpi::maximum<std::size_t>());
            std::size_t reference_min_level = mpi::all_reduce(world,
                                                              this->cells()[mesh_id_t::reference].min_level(),
                                                              mpi::minimum<std::size_t>());

            std::vector<ca_type> neighbourhood_extended_subdomain(this->mpi_neighbourhood().size());
            for (size_t neighbor_id = 0; neighbor_id != neighbourhood_extended_subdomain.size(); ++neighbor_id)
            {
                const auto& neighbor_subdomain = this->mpi_neighbourhood()[neighbor_id].mesh.subdomain();
                if (not neighbor_subdomain.empty())
                {
                    const auto& neighbor_min_indices = neighbor_subdomain.min_indices();
                    const auto& neighbor_max_indices = neighbor_subdomain.max_indices();
                    for (std::size_t level = reference_min_level; level <= reference_max_level; ++level)
                    {
                        const std::size_t delta_l = subdomain.level() - level;
                        box_t box;
                        for (std::size_t d = 0; d < dim; ++d)
                        {
                            box.min_corner()[d] = (neighbor_min_indices[d] >> delta_l) - ghost_width();
                            box.max_corner()[d] = (neighbor_max_indices[d] >> delta_l) + ghost_width();
                        }
                        neighbourhood_extended_subdomain[neighbor_id][level] = {level, box};
                    }
                }
            }
#endif // SAMURAI_WITH_MPI
            const auto& mesh_ref = this->cells()[mesh_id_t::reference];
            for (std::size_t level = 0; level <= this->max_level(); ++level)
            {
                const std::size_t delta_l = subdomain.level() - level;
                lcl_type& lcl             = cell_list[level];

                for (std::size_t d = 0; d < dim; ++d)
                {
                    min_corner[d] = (subdomain_min_indices[d] >> delta_l) - ghost_width();
                    max_corner[d] = (subdomain_max_indices[d] >> delta_l) + ghost_width();
                }
                lca_type lca_extended_subdomain(level, box_t(min_corner, max_corner));
                for (std::size_t d = 0; d < dim; ++d)
                {
                    if (this->is_periodic(d))
                    {
                        shift[d] = (domain_max_indices[d] - domain_min_indices[d]) >> delta_l;

                        min_corner[d] = (domain_min_indices[d] >> delta_l) - ghost_width();
                        max_corner[d] = (domain_min_indices[d] >> delta_l) + ghost_width();

                        lca_type lca_min(level, box_t(min_corner, max_corner));

                        min_corner[d] = (domain_max_indices[d] >> delta_l) - ghost_width();
                        max_corner[d] = (domain_max_indices[d] >> delta_l) + ghost_width();

                        lca_type lca_max(level, box_t(min_corner, max_corner));

                        auto set1 = intersection(translate(intersection(mesh_ref[level], lca_min), shift),
                                                 intersection(lca_extended_subdomain, lca_max));
                        set1(
                            [&](const auto& i, const auto& index_yz)
                            {
                                lcl[index_yz].add_interval(i);
                            });
                        auto set2 = intersection(translate(intersection(mesh_ref[level], lca_max), -shift),
                                                 intersection(lca_extended_subdomain, lca_min));
                        set2(
                            [&](const auto& i, const auto& index_yz)
                            {
                                lcl[index_yz].add_interval(i);
                            });
#ifdef SAMURAI_WITH_MPI
                        //~ for (const auto& mpi_neighbor : this->mpi_neighbourhood())
                        for (size_t neighbor_id = 0; neighbor_id != this->mpi_neighbourhood().size(); ++neighbor_id)
                        {
                            const auto& mpi_neighbor      = this->mpi_neighbourhood()[neighbor_id];
                            const auto& neighbor_mesh_ref = mpi_neighbor.mesh[mesh_id_t::reference];

                            auto set1_mpi = intersection(translate(intersection(neighbor_mesh_ref[level], lca_min), shift),
                                                         intersection(lca_extended_subdomain, lca_max));
                            set1_mpi(
                                [&](const auto& i, const auto& index_yz)
                                {
                                    lcl[index_yz].add_interval(i);
                                });
                            auto set2_mpi = intersection(translate(intersection(neighbor_mesh_ref[level], lca_max), -shift),
                                                         intersection(lca_extended_subdomain, lca_min));
                            set2_mpi(
                                [&](const auto& i, const auto& index_yz)
                                {
                                    lcl[index_yz].add_interval(i);
                                });
                        }
#endif // SAMURAI_WITH_MPI
                        this->cells()[mesh_id_t::all_cells][level] = {lcl};

                        /* reset variables for next iterations. */
                        shift[d]      = 0;
                        min_corner[d] = (subdomain_min_indices[d] >> delta_l) - ghost_width();
                        max_corner[d] = (subdomain_max_indices[d] >> delta_l) + ghost_width();
                    }
                }
            }
            for (std::size_t level = 0; level < max_level; ++level)
            {
                lcl_type& lcl = cell_list[level + 1];
                lcl_type lcl_proj{level};
                auto expr = intersection(this->cells()[mesh_id_t::all_cells][level], this->get_union()[level]);

                expr(
                    [&](const auto& interval, const auto& index_yz)
                    {
                        static_nested_loop<dim - 1, 0, 2>(
                            [&](auto s)
                            {
                                lcl[(index_yz << 1) + s].add_interval(interval << 1);
                            });
                        lcl_proj[index_yz].add_interval(interval);
                    });
                this->cells()[mesh_id_t::all_cells][level + 1] = lcl;
                this->cells()[mesh_id_t::proj_cells][level]    = lcl_proj;
            }
            this->update_neighbour_subdomain();
            this->update_meshid_neighbour(mesh_id_t::all_cells);

            for (auto& neighbour : this->mpi_neighbourhood())
            {
                for (std::size_t level = 0; level <= this->max_level(); ++level)
                {
                    auto expr = intersection(nestedExpand(self(this->subdomain()).on(level), ghost_width()),
                                             neighbour.mesh[mesh_id_t::reference][level]);
                    expr(
                        [&](const auto& interval, const auto& index_yz)
                        {
                            lcl_type& lcl = cell_list[level];
                            lcl[index_yz].add_interval(interval);
                        });
                    for (std::size_t d = 0; d < dim; ++d)
                    {
                        if (this->is_periodic(d))
                        {
                            auto domain_shift = get_periodic_shift(this->domain(), level, d);
                            auto expr_left    = intersection(nestedExpand(self(this->subdomain()).on(level), ghost_width()),
                                                          translate(neighbour.mesh[mesh_id_t::reference][level], -domain_shift));
                            expr_left(
                                [&](const auto& interval, const auto& index_yz)
                                {
                                    lcl_type& lcl = cell_list[level];
                                    lcl[index_yz].add_interval(interval);
                                });

                            auto expr_right = intersection(nestedExpand(self(this->subdomain()).on(level), ghost_width()),
                                                           translate(neighbour.mesh[mesh_id_t::reference][level], domain_shift));
                            expr_right(
                                [&](const auto& interval, const auto& index_yz)
                                {
                                    lcl_type& lcl = cell_list[level];
                                    lcl[index_yz].add_interval(interval);
                                });
                        }
                    }
                }
            }

            this->cells()[mesh_id_t::all_cells] = {cell_list, false};
            this->update_meshid_neighbour(mesh_id_t::all_cells);
        }

        else
        {
            this->cells()[mesh_id_t::all_cells] = {cell_list, false};
            // TODO : I think we do not want to update subdomain in this case, it remains the same iteration after iteration.
            this->update_neighbour_subdomain();
            this->update_meshid_neighbour(mesh_id_t::cells_and_ghosts);
            this->update_meshid_neighbour(mesh_id_t::reference);
        }
    }

    template <class Config>
    template <typename... T>
    SAMURAI_INLINE xt::xtensor<bool, 1> MRMesh<Config>::exists(mesh_id_t type, std::size_t level, interval_t interval, T... index) const
    {
        using coord_index_t      = typename interval_t::coord_index_t;
        const auto& lca          = this->cells()[type][level];
        std::size_t size         = interval.size() / interval.step;
        xt::xtensor<bool, 1> out = xt::empty<bool>({size});
        std::size_t iout         = 0;
        for (coord_index_t i = interval.start; i < interval.end; i += interval.step)
        {
            auto offset = find(lca, {i, index...});
            if (offset == -1)
            {
                out[iout++] = false;
            }
            else
            {
                out[iout++] = true;
            }
            return out;
        }
    }

    namespace mra
    {
        // create an empty mesh
        template <class mesh_config_t, class complete_mesh_config_t = complete_mesh_config<mesh_config_t, MRMeshId>>
        auto make_empty_mesh(const mesh_config_t&)
        {
            return MRMesh<complete_mesh_config_t>();
        }

        template <class mesh_config_t, class complete_mesh_config_t = complete_mesh_config<mesh_config_t, MRMeshId>>
        auto make_mesh(const typename MRMesh<complete_mesh_config_t>::cl_type& cl, const mesh_config_t& cfg)
        {
            auto mesh_cfg = cfg;
            mesh_cfg.parse_args();
            mesh_cfg.start_level() = mesh_cfg.max_level(); // cppcheck-suppress unreadVariable

            return MRMesh<complete_mesh_config_t>(cl, mesh_cfg);
        }

        template <class mesh_config_t, class complete_mesh_config_t = complete_mesh_config<mesh_config_t, MRMeshId>>
        auto make_mesh(const typename MRMesh<complete_mesh_config_t>::ca_type& ca, const mesh_config_t& cfg)
        {
            auto mesh_cfg = cfg;
            mesh_cfg.parse_args();
            mesh_cfg.start_level() = mesh_cfg.max_level(); // cppcheck-suppress unreadVariable

            return MRMesh<complete_mesh_config_t>(ca, mesh_cfg);
        }

        template <class mesh_config_t>
        auto make_mesh(const samurai::Box<double, mesh_config_t::dim>& b, const mesh_config_t& cfg)
        {
            using complete_cfg_t = complete_mesh_config<mesh_config_t, MRMeshId>;

            auto mesh_cfg = cfg;
            mesh_cfg.parse_args();
            mesh_cfg.start_level() = mesh_cfg.max_level(); // cppcheck-suppress unreadVariable

            return MRMesh<complete_cfg_t>(b, mesh_cfg);
        }

        template <class mesh_config_t>
        auto make_mesh(const samurai::DomainBuilder<mesh_config_t::dim>& domain_builder, const mesh_config_t& cfg)
        {
            using complete_cfg_t = complete_mesh_config<mesh_config_t, MRMeshId>;

            auto mesh_cfg = cfg;
            mesh_cfg.parse_args();
            mesh_cfg.start_level() = mesh_cfg.max_level(); // cppcheck-suppress unreadVariable

            return MRMesh<complete_cfg_t>(domain_builder, mesh_cfg);
        }
    }
} // namespace samurai

template <>
struct fmt::formatter<samurai::MRMeshId> : formatter<string_view>
{
    // parse is inherited from formatter<string_view>.
    template <typename FormatContext>
    auto format(samurai::MRMeshId c, FormatContext& ctx) const
    {
        string_view name = "unknown";
        switch (c)
        {
            case samurai::MRMeshId::cells:
                name = "cells";
                break;
            case samurai::MRMeshId::cells_and_ghosts:
                name = "cells and ghosts";
                break;
            case samurai::MRMeshId::proj_cells:
                name = "projection cells";
                break;
            case samurai::MRMeshId::union_cells:
                name = "union cells";
                break;
            case samurai::MRMeshId::all_cells:
                name = "all cells";
                break;
            case samurai::MRMeshId::count:
                name = "count";
                break;
        }
        return formatter<string_view>::format(name, ctx);
    }
};
