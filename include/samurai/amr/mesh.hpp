// Copyright 2018-2025 the samurai's authors
// SPDX-License-Identifier:  BSD-3-Clause

#pragma once

#include <fmt/format.h>

#include "../algorithm.hpp"
#include "../box.hpp"
#include "../mesh.hpp"
#include "../samurai_config.hpp"
#include "../subset/node.hpp"

namespace samurai::amr
{
    enum class AMR_Id
    {
        cells            = 0,
        cells_and_ghosts = 1,
        proj_cells       = 2,
        pred_cells       = 3,
        all_cells        = 4,
        count            = 5,
        reference        = all_cells
        // reference = cells_and_ghosts
    };

    template <std::size_t dim_,
              std::size_t max_stencil_width_    = default_config::ghost_width,
              std::size_t graduation_width_     = default_config::graduation_width,
              std::size_t max_refinement_level_ = default_config::max_level,
              std::size_t prediction_order_     = default_config::prediction_stencil_radius,
              class TInterval                   = default_config::interval_t>
    struct [[deprecated("Use samurai::mesh_config instead")]] Config
    {
        static constexpr std::size_t dim                  = dim_;
        static constexpr std::size_t max_refinement_level = max_refinement_level_;

        // deprecated interface
        [[deprecated("Use max_stencil_radius() method instead")]] static constexpr int max_stencil_width = max_stencil_width_;
        [[deprecated("Use prediction_stencil_radius instead")]] static constexpr int prediction_order    = prediction_order_;
        [[deprecated("Use graduation_width() method instead")]] static constexpr int graduation_width    = graduation_width_;
        [[deprecated("Use ghost_width() method instead")]] static constexpr int ghost_width = std::max(static_cast<int>(max_stencil_width),
                                                                                                       static_cast<int>(prediction_order));

        using interval_t = TInterval;
        using mesh_id_t  = AMR_Id;
    };

    /////////////////////////
    // AMR mesh definition //
    /////////////////////////

    template <class Config>
    class Mesh : public Mesh_base<Mesh<Config>, Config>
    {
      public:

        using base_type                  = Mesh_base<Mesh<Config>, Config>;
        using self_type                  = Mesh<Config>;
        using mpi_subdomain_t            = typename base_type::mpi_subdomain_t;
        using config                     = typename base_type::config;
        static constexpr std::size_t dim = config::dim;

        using mesh_id_t = typename base_type::mesh_id_t;
        using cl_type   = typename base_type::cl_type;
        using lcl_type  = typename base_type::lcl_type;

        using ca_type  = typename base_type::ca_type;
        using lca_type = typename base_type::lca_type;

        Mesh() = default;
        Mesh(const ca_type& ca, const self_type& ref_mesh);
        Mesh(const cl_type& cl, const self_type& ref_mesh);
        Mesh(const cl_type& cl, const mesh_config<Config::dim>& config);
        Mesh(const ca_type& ca, const mesh_config<Config::dim>& config);
        Mesh(const Box<double, dim>& b, const mesh_config<Config::dim>& config);

        Mesh(const cl_type& cl, std::size_t min_level, std::size_t max_level);
        Mesh(const ca_type& ca, std::size_t min_level, std::size_t max_level);
        Mesh(const Box<double, dim>& b, std::size_t start_level, std::size_t min_level, std::size_t max_level);

        void update_sub_mesh_impl();

        using base_type::cfg;
    };

    /////////////////////////////
    // AMR mesh implementation //
    /////////////////////////////

    template <class Config>
    inline Mesh<Config>::Mesh(const ca_type& ca, const self_type& ref_mesh)
        : base_type(ca, ref_mesh)
    {
    }

    template <class Config>
    inline Mesh<Config>::Mesh(const cl_type& cl, const self_type& ref_mesh)
        : base_type(cl, ref_mesh)
    {
    }

    template <class Config>
    inline Mesh<Config>::Mesh(const cl_type& cl, const mesh_config<Config::dim>& config)
        : base_type(cl, config)
    {
    }

    template <class Config>
    inline Mesh<Config>::Mesh(const ca_type& ca, const mesh_config<Config::dim>& config)
        : base_type(ca, config)
    {
    }

    template <class Config>
    inline Mesh<Config>::Mesh(const Box<double, dim>& b, const mesh_config<Config::dim>& config)
        : base_type(b, config)
    {
    }

    template <class Config>
    inline Mesh<Config>::Mesh(const cl_type& cl, std::size_t min_level, std::size_t max_level)
        : base_type(cl,
                    mesh_config<Config::dim, Config::prediction_order, Config::max_refinement_level, typename Config::interval_t>()
                        .max_stencil_radius(Config::max_stencil_width)
                        .graduation_width(Config::graduation_width)
                        .start_level(max_level)
                        .min_level(min_level)
                        .max_level(max_level))
    {
    }

    template <class Config>
    inline Mesh<Config>::Mesh(const ca_type& ca, std::size_t min_level, std::size_t max_level)
        : base_type(ca,
                    mesh_config<Config::dim, Config::prediction_order, Config::max_refinement_level, typename Config::interval_t>()
                        .max_stencil_radius(Config::max_stencil_width)
                        .graduation_width(Config::graduation_width)
                        .start_level(max_level)
                        .min_level(min_level)
                        .max_level(max_level))
    {
    }

    template <class Config>
    inline Mesh<Config>::Mesh(const Box<double, dim>& b, std::size_t start_level, std::size_t min_level, std::size_t max_level)
        : base_type(b,
                    mesh_config<Config::dim, Config::prediction_order, Config::max_refinement_level, typename Config::interval_t>()
                        .max_stencil_radius(Config::max_stencil_width)
                        .graduation_width(Config::graduation_width)
                        .start_level(start_level)
                        .min_level(min_level)
                        .max_level(max_level))
    {
    }

    template <class Config>
    inline void Mesh<Config>::update_sub_mesh_impl()
    {
        cl_type cl;
        auto ghost_width = cfg().ghost_width();
        for_each_interval(this->cells()[mesh_id_t::cells],
                          [&](std::size_t level, const auto& interval, const auto& index_yz)
                          {
                              lcl_type& lcl = cl[level];
                              static_nested_loop<dim - 1>(
                                  -ghost_width,
                                  ghost_width + 1,
                                  [&](auto stencil)
                                  {
                                      auto index = xt::eval(index_yz + stencil);
                                      lcl[index].add_interval({interval.start - ghost_width, interval.end + ghost_width});
                                  });
                          });
        this->cells()[mesh_id_t::cells_and_ghosts] = {cl, false};

        auto max_level = this->cells()[mesh_id_t::cells].max_level();
        auto min_level = this->cells()[mesh_id_t::cells].min_level();

        // construction of projection cells
        this->cells()[mesh_id_t::proj_cells][min_level] = {min_level};
        for (std::size_t level = min_level + 1; level <= max_level; ++level)
        {
            auto expr = difference(union_(intersection(this->cells()[mesh_id_t::cells_and_ghosts][level - 1], this->get_union()[level - 1]),
                                          this->cells()[mesh_id_t::proj_cells][level - 1]),
                                   this->cells()[mesh_id_t::cells][level - 1])
                            .on(level);

            lcl_type lcl{level};
            expr(
                [&](const auto& interval, const auto& index_yz)
                {
                    lcl[index_yz].add_interval({interval.start, interval.end});
                });

            this->cells()[mesh_id_t::proj_cells][level] = {lcl};
        }

        // construction of prediction cells
        for (std::size_t level = min_level + 1; level <= max_level; ++level)
        {
            auto expr = intersection(difference(this->cells()[mesh_id_t::cells_and_ghosts][level],
                                                union_(this->get_union()[level], this->cells()[mesh_id_t::cells][level])),
                                     self(this->domain()).on(level));

            lcl_type lcl{level};
            expr(
                [&](const auto& interval, const auto& index_yz)
                {
                    lcl[index_yz].add_interval(interval);
                });

            this->cells()[mesh_id_t::pred_cells][level] = {lcl};
        }

        for (std::size_t level = min_level; level <= max_level; ++level)
        {
            auto expr = intersection(this->cells()[mesh_id_t::pred_cells][level], this->cells()[mesh_id_t::pred_cells][level]).on(level - 1);

            lcl_type& lcl = cl[level - 1];

            expr(
                [&](const auto& interval, const auto& index_yz)
                {
                    // add ghosts for the prediction
                    static_nested_loop<dim - 1, -config::prediction_stencil_radius, config::prediction_stencil_radius + 1>(
                        [&](auto stencil)
                        {
                            auto index = xt::eval(index_yz + stencil);
                            lcl[index].add_interval(
                                {interval.start - config::prediction_stencil_radius, interval.end + config::prediction_stencil_radius});
                        });
                });
        }
        this->cells()[mesh_id_t::cells_and_ghosts] = {cl, false};

        for (std::size_t level = min_level; level <= max_level; ++level)
        {
            lcl_type lcl{level};
            auto expr = union_(this->cells()[mesh_id_t::cells_and_ghosts][level], this->cells()[mesh_id_t::proj_cells][level]);
            expr(
                [&](const auto& interval, const auto& index_yz)
                {
                    lcl[index_yz].add_interval(interval);
                });

            this->cells()[mesh_id_t::all_cells][level] = {lcl};
        }
    }

    template <class mesh_config_t, class complete_mesh_config_t = complete_mesh_config<mesh_config_t, AMR_Id>>
    auto make_empty_mesh(const mesh_config_t&)
    {
        return Mesh<complete_mesh_config_t>();
    }

    template <class mesh_config_t, class complete_mesh_config_t = complete_mesh_config<mesh_config_t, AMR_Id>>
    auto make_mesh(const typename Mesh<complete_mesh_config_t>::cl_type& cl, const mesh_config_t& cfg)
    {
        auto mesh_cfg = cfg;
        mesh_cfg.parse_args();

        return Mesh<complete_mesh_config_t>(cl, mesh_cfg);
    }

    template <class mesh_config_t, class complete_mesh_config_t = complete_mesh_config<mesh_config_t, AMR_Id>>
    auto make_mesh(const typename Mesh<complete_mesh_config_t>::ca_type& ca, const mesh_config_t& cfg)
    {
        auto mesh_cfg = cfg;
        mesh_cfg.parse_args();

        return Mesh<complete_mesh_config_t>(ca, mesh_cfg);
    }

    template <class mesh_config_t>
    auto make_mesh(const samurai::Box<double, mesh_config_t::dim>& b, const mesh_config_t& cfg)
    {
        using complete_cfg_t = complete_mesh_config<mesh_config_t, AMR_Id>;

        auto mesh_cfg = cfg;
        mesh_cfg.parse_args();

        return Mesh<complete_cfg_t>(b, mesh_cfg);
    }
} // namespace samurai::amr

template <>
struct fmt::formatter<samurai::amr::AMR_Id> : formatter<string_view>
{
    template <typename FormatContext>
    auto format(samurai::amr::AMR_Id c, FormatContext& ctx) const
    {
        string_view name = "unknown";
        switch (c)
        {
            case samurai::amr::AMR_Id::cells:
                name = "cells";
                break;
            case samurai::amr::AMR_Id::cells_and_ghosts:
                name = "cells and ghosts";
                break;
            case samurai::amr::AMR_Id::proj_cells:
                name = "proj cells";
                break;
            case samurai::amr::AMR_Id::pred_cells:
                name = "pred cells";
                break;
            case samurai::amr::AMR_Id::all_cells:
                name = "all cells";
                break;
            case samurai::amr::AMR_Id::count:
                name = "count";
                break;
        }
        return formatter<string_view>::format(name, ctx);
    }
};
