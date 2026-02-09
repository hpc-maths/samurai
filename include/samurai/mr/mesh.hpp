// Copyright 2018-2025 the samurai's authors
// SPDX-License-Identifier:  BSD-3-Clause

#pragma once

#include <fmt/format.h>

#include <xtensor/containers/xtensor.hpp>
#include <xtensor/views/xview.hpp>

// #include "../algorithm/graduation.hpp"
#include "../algorithm/graduation.hpp"
#include "../box.hpp"
#include "../mesh.hpp"
#include "../samurai_config.hpp"
#include "../stencil.hpp"
#include "../subset/apply.hpp"
#include "../subset/node.hpp"
#include "../timers.hpp"

using namespace xt::placeholders;

namespace samurai
{
    enum class MRMeshId
    {
        cells            = 0,
        cells_and_ghosts = 1,
        proj_cells       = 2,
        union_cells      = 3,
        reference        = 4,
        count            = 5,
        all_cells        = reference
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
        times::timers.start("mesh construction");

        cl_type cell_list;

        // Construction of ghost cells
        // ===========================
        //
        // Example with max_stencil_width = 1
        //
        // level 2                       |.|-|-|-|-|-|-|.|                   |-| cells
        //                                                                   |.| ghost cells
        //
        // level 1             |...|---|---|...|   |...|---|---|...|
        //
        // level 0 |.......|-------|.......|           |.......|-------|.......|
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

        // Manage neighbourhood for MPI
        // ==============================
        // We do the same with the subdomains of the neighbouring processes
        // to be sure that we have all the ghost cells and the cells at the interface
        // with the other processes.
        for (auto& neighbour : this->mpi_neighbourhood())
        {
            for_each_level(neighbour.mesh[mesh_id_t::cells],
                           [&](std::size_t level)
                           {
                               lcl_type& lcl = cell_list[level];
                               auto set      = intersection(nestedExpand(neighbour.mesh[mesh_id_t::cells][level], max_stencil_radius()),
                                                       nestedExpand(self(this->subdomain()).on(level), max_stencil_radius()));
                               set(
                                   [&](const auto& interval, const auto& index)
                                   {
                                       lcl[index].add_interval(interval);
                                   });
                           });
        }

        // Manage periodicity
        // ===================
        // This process is similar to the MPI one, but we need first to compute the directions to move the subdomains
        // on the other side of the domain.
        std::vector<DirectionVector<dim>> directions;
        if (this->is_periodic())
        {
            std::array<int, dim> nb_cells_finest_level;
            const auto& min_indices = this->domain().min_indices();
            const auto& max_indices = this->domain().max_indices();

            for (size_t d = 0; d != max_indices.size(); ++d)
            {
                nb_cells_finest_level[d] = max_indices[d] - min_indices[d];
            }
            directions = detail::get_periodic_directions(nb_cells_finest_level, 0, this->periodicity());
        }

        auto add_periodic_cells = [&](const auto& subset, auto level)
        {
            lcl_type& lcl     = cell_list[level];
            const int delta_l = int(this->domain().level() - level);

            for (const auto& d : directions)
            {
                auto set = intersection(nestedExpand(translate(subset, d >> delta_l), this->cfg().max_stencil_radius()),
                                        nestedExpand(self(this->subdomain()).on(level), this->cfg().max_stencil_radius()));
                set(
                    [&](const auto& interval, const auto& index)
                    {
                        lcl[index].add_interval(interval);
                    });
            }
        };

        // ghost cells are added by translating the mesh on the different periodic directions
        // and intersecting with the subdomain
        for_each_level(this->cells()[mesh_id_t::cells],
                       [&](std::size_t level)
                       {
                           add_periodic_cells(this->cells()[mesh_id_t::cells][level], level);
                       });

        // We do the same with the subdomains of the neighbouring processes
        for (auto& neighbour : this->mpi_neighbourhood())
        {
            for_each_level(neighbour.mesh[mesh_id_t::cells],
                           [&](std::size_t level)
                           {
                               add_periodic_cells(neighbour.mesh[mesh_id_t::cells][level], level);
                           });
        }

        // Add cells for the MRA to be able to compute the details at one and two levels below each cells.
        // The prediction operator gives the number of ghost cells to add in each direction.
        if (this->max_level() != this->min_level())
        {
            auto add_prediction_ghosts = [&](auto&& subset_cells, auto&& subset_cells_and_ghosts, auto level)
            {
                lcl_type& lcl_m1 = cell_list[level - 1];

                subset_cells_and_ghosts(
                    [&](const auto& interval, const auto& index_yz)
                    {
                        lcl_m1[index_yz].add_interval(interval);
                    });

                if (level - 1 > 0)
                {
                    lcl_type& lcl_m2 = cell_list[level - 2];
                    subset_cells(
                        [&](const auto& interval, const auto& index_yz)
                        {
                            lcl_m2[index_yz].add_interval(interval);
                        });
                }
            };

            for_each_level(
                this->cells()[mesh_id_t::cells],
                [&](std::size_t level)
                {
                    // own part
                    add_prediction_ghosts(
                        nestedExpand<config::prediction_stencil_radius>(self(this->cells()[mesh_id_t::cells][level]).on(level - 2)),
                        intersection(nestedExpand<config::prediction_stencil_radius>(
                                         self(this->cells()[mesh_id_t::cells_and_ghosts][level]).on(level - 1)),
                                     nestedExpand<config::prediction_stencil_radius>(self(this->subdomain()).on(level - 1))),
                        level);

                    // periodic part
                    const int delta_l = int(this->domain().level() - level);
                    for (const auto& d : directions)
                    {
                        add_prediction_ghosts(
                            intersection(nestedExpand<config::prediction_stencil_radius>(
                                             translate(this->cells()[mesh_id_t::cells][level], d >> delta_l).on(level - 2)),
                                         self(this->subdomain()).on(level - 2)),
                            intersection(nestedExpand<config::prediction_stencil_radius>(
                                             translate(this->cells()[mesh_id_t::cells_and_ghosts][level], d >> delta_l).on(level - 1)),
                                         self(this->subdomain()).on(level - 1)),
                            level);
                    }
                });

            // mpi part
            this->update_meshid_neighbour(mesh_id_t::cells_and_ghosts);
            for (auto& neighbour : this->mpi_neighbourhood())
            {
                for_each_level(
                    neighbour.mesh[mesh_id_t::cells],
                    [&](std::size_t level)
                    {
                        add_prediction_ghosts(intersection(nestedExpand<config::prediction_stencil_radius>(
                                                               self(neighbour.mesh[mesh_id_t::cells][level]).on(level - 2)),
                                                           self(this->subdomain()).on(level - 2)),
                                              intersection(nestedExpand<config::prediction_stencil_radius>(
                                                               self(neighbour.mesh[mesh_id_t::cells_and_ghosts][level]).on(level - 1)),
                                                           self(this->subdomain()).on(level - 1)),
                                              level);

                        const int delta_l = int(this->domain().level() - level);

                        for (const auto& d : directions)
                        {
                            add_prediction_ghosts(
                                intersection(nestedExpand<config::prediction_stencil_radius>(
                                                 translate(neighbour.mesh[mesh_id_t::cells][level], d >> delta_l).on(level - 2)),
                                             self(this->subdomain()).on(level - 2)),
                                intersection(nestedExpand<config::prediction_stencil_radius>(
                                                 translate(neighbour.mesh[mesh_id_t::cells_and_ghosts][level], d >> delta_l).on(level - 1)),
                                             self(this->subdomain()).on(level - 1)),
                                level);
                        }
                    });
            }

            this->cells()[mesh_id_t::reference] = {cell_list, false};

            // Some of the ghosts added by the prediction or the stencil of the scheme can be computed by projection
            // as described in the following figure (1D example):
            //
            // other levels      ||||||||
            //
            // level 2       |-|-|.|.|.|                   |-| cells
            //                                             |.| ghost cells added by the prediction
            // level 1           |---|...|...|
            //
            // We can observe that the first ghost cell on the right of the cell at level 1 can be computed by projection
            // from level 2. But we only have one ghost cell at level 2, so we need to add the other one to have enough information
            //
            // level 2       |-|-|.|.|.|.|                 |-| cells
            //                                             |.| ghost cells added by the prediction
            // level 1           |---|...|...|
            //
            for_each_level(this->cells()[mesh_id_t::reference],
                           [&](auto level)
                           {
                               lcl_type& lcl = cell_list[level + 1];
                               lcl_type lcl_proj{level};
                               auto expr = intersection(this->cells()[mesh_id_t::reference][level], this->get_union()[level]);

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
                               this->cells()[mesh_id_t::reference][level + 1] = lcl;
                               this->cells()[mesh_id_t::proj_cells][level]    = lcl_proj;
                           });
        }
        this->cells()[mesh_id_t::reference] = {cell_list, false};
        this->update_meshid_neighbour(mesh_id_t::reference);
        times::timers.stop("mesh construction");
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
            case samurai::MRMeshId::reference:
                name = "reference";
                break;
            case samurai::MRMeshId::count:
                name = "count";
                break;
        }
        return formatter<string_view>::format(name, ctx);
    }
};
