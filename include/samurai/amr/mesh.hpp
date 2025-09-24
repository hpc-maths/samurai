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
              std::size_t prediction_order_     = default_config::prediction_order,
              class TInterval                   = default_config::interval_t>
    struct Config
    {
        static constexpr std::size_t dim                  = dim_;
        static constexpr std::size_t max_refinement_level = max_refinement_level_;
        static constexpr int max_stencil_width            = max_stencil_width_;
        static constexpr int prediction_order             = prediction_order_;
        static constexpr int ghost_width      = std::max(static_cast<int>(max_stencil_width), static_cast<int>(prediction_order));
        static constexpr int graduation_width = graduation_width_;

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
        Mesh(const mesh_config<Config::dim>& config, const cl_type& cl);
        Mesh(const mesh_config<Config::dim>& config, const ca_type& ca);
        Mesh(mesh_config<Config::dim>& config, const Box<double, dim>& b, std::size_t start_level);

        void update_sub_mesh_impl();
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
    inline Mesh<Config>::Mesh(const mesh_config<Config::dim>& config, const cl_type& cl)
        : base_type(config, cl)
    {
    }

    template <class Config>
    inline Mesh<Config>::Mesh(const mesh_config<Config::dim>& config, const ca_type& ca)
        : base_type(config, ca)
    {
    }

    template <class Config>
    inline Mesh<Config>::Mesh(mesh_config<Config::dim>& config, const Box<double, dim>& b, std::size_t start_level)
        : base_type(config, b, start_level)
    {
    }

    template <class Config>
    inline void Mesh<Config>::update_sub_mesh_impl()
    {
        cl_type cl;
        for_each_interval(this->cells()[mesh_id_t::cells],
                          [&](std::size_t level, const auto& interval, const auto& index_yz)
                          {
                              lcl_type& lcl = cl[level];
                              static_nested_loop<dim - 1, -config::ghost_width, config::ghost_width + 1>(
                                  [&](auto stencil)
                                  {
                                      auto index = xt::eval(index_yz + stencil);
                                      lcl[index].add_interval({interval.start - config::ghost_width, interval.end + config::ghost_width});
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
                    static_nested_loop<dim - 1, -config::prediction_order, config::prediction_order + 1>(
                        [&](auto stencil)
                        {
                            auto index = xt::eval(index_yz + stencil);
                            lcl[index].add_interval({interval.start - config::prediction_order, interval.end + config::prediction_order});
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
}

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
