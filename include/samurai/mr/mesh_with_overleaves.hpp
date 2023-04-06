// Copyright 2021 SAMURAI TEAM. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#pragma once

#include <fmt/format.h>

#include <xtensor/xtensor.hpp>

#include "../box.hpp"
#include "../mesh.hpp"
#include "../samurai_config.hpp"
#include "../subset/node_op.hpp"
#include "../subset/subset_op.hpp"

namespace samurai
{
    enum class MROMeshId
    {
        cells            = 0,
        cells_and_ghosts = 1,
        proj_cells       = 2,
        union_cells      = 3,
        all_cells        = 4,
        overleaves       = 5, // Added in order to automatically handle flux
                              // correction. (by Thomas)
        count     = 6,
        reference = all_cells
    };

    template <std::size_t dim_,
              std::size_t max_stencil_width_    = default_config::ghost_width,
              std::size_t graduation_width_     = default_config::graduation_width,
              std::size_t max_refinement_level_ = default_config::max_level,
              std::size_t prediction_order_     = default_config::prediction_order,
              class TInterval                   = default_config::interval_t>
    struct MROConfig
    {
        static constexpr std::size_t dim                  = dim_;
        static constexpr std::size_t max_refinement_level = max_refinement_level_;
        static constexpr std::size_t max_stencil_width    = max_stencil_width_;
        static constexpr std::size_t graduation_width     = graduation_width_;
        static constexpr std::size_t prediction_order     = prediction_order_;

        static constexpr int ghost_width = std::max(std::max(2 * static_cast<int>(graduation_width) - 1, static_cast<int>(max_stencil_width)),
                                                    static_cast<int>(prediction_order));
        using interval_t = TInterval;
        using mesh_id_t  = MROMeshId;
    };

    template <class Config>
    class MROMesh : public samurai::Mesh_base<MROMesh<Config>, Config>
    {
      public:

        using base_type = samurai::Mesh_base<MROMesh<Config>, Config>;

        using config                     = typename base_type::config;
        static constexpr std::size_t dim = config::dim;

        using mesh_id_t  = typename base_type::mesh_id_t;
        using interval_t = typename base_type::interval_t;
        using cl_type    = typename base_type::cl_type;
        using lcl_type   = typename base_type::lcl_type;

        using ca_type  = typename base_type::ca_type;
        using lca_type = typename base_type::lca_type;

        MROMesh(const cl_type& cl, std::size_t min_level, std::size_t max_level);
        MROMesh(const cl_type& cl, std::size_t min_level, std::size_t max_level, const std::array<bool, dim>& periodic);
        MROMesh(const samurai::Box<double, dim>& b, std::size_t min_level, std::size_t max_level);
        MROMesh(const samurai::Box<double, dim>& b, std::size_t min_level, std::size_t max_level, const std::array<bool, dim>& periodic);

        void update_sub_mesh_impl();

        template <typename... T>
        xt::xtensor<bool, 1> exists(mesh_id_t type, std::size_t level, interval_t interval, T... index) const;
    };

    template <class Config>
    inline MROMesh<Config>::MROMesh(const cl_type& cl, std::size_t min_level, std::size_t max_level)
        : base_type(cl, min_level, max_level)
    {
    }

    template <class Config>
    inline MROMesh<Config>::MROMesh(const cl_type& cl, std::size_t min_level, std::size_t max_level, const std::array<bool, dim>& periodic)
        : base_type(cl, min_level, max_level, periodic)
    {
    }

    template <class Config>
    inline MROMesh<Config>::MROMesh(const samurai::Box<double, dim>& b, std::size_t min_level, std::size_t max_level)
        : base_type(b, max_level, min_level, max_level)
    {
    }

    template <class Config>
    inline MROMesh<Config>::MROMesh(const samurai::Box<double, dim>& b,
                                    std::size_t min_level,
                                    std::size_t max_level,
                                    const std::array<bool, dim>& periodic)
        : base_type(b, max_level, min_level, max_level, periodic)
    {
    }

    template <class Config>
    inline void MROMesh<Config>::update_sub_mesh_impl()
    {
        auto max_level = this->m_cells[mesh_id_t::cells].max_level();
        auto min_level = this->m_cells[mesh_id_t::cells].min_level();
        cl_type cell_list;

        // Construct union cells
        this->m_cells[mesh_id_t::union_cells][max_level] = {max_level};

        for (std::size_t level = max_level; level >= ((min_level == 0) ? 1 : min_level); --level)
        {
            lcl_type lcl{level - 1};
            auto expr = union_(this->m_cells[mesh_id_t::cells][level], this->m_cells[mesh_id_t::union_cells][level]).on(level - 1);

            expr(
                [&](const auto& interval, const auto& index_yz)
                {
                    lcl[index_yz].add_interval({interval.start, interval.end});
                });

            this->m_cells[mesh_id_t::union_cells][level - 1] = {lcl};
        }

        // Construct ghost cells
        for_each_interval(this->m_cells[mesh_id_t::cells],
                          [&](std::size_t level, const auto& interval, const auto& index_yz)
                          {
                              lcl_type& lcl = cell_list[level];
                              static_nested_loop<dim - 1, -config::ghost_width, config::ghost_width + 1>(
                                  [&](auto stencil)
                                  {
                                      auto index = xt::eval(index_yz + stencil);
                                      lcl[index].add_interval({interval.start - config::ghost_width, interval.end + config::ghost_width});
                                  });
                          });

        this->m_cells[mesh_id_t::cells_and_ghosts] = {cell_list, false};

        // Construct projection cells
        for (std::size_t level = ((min_level == 0) ? 1 : min_level); level <= max_level; ++level)
        {
            lca_type& lca = this->m_cells[mesh_id_t::cells][level];
            lcl_type& lcl = cell_list[level - 1];

            for_each_interval(lca,
                              [&](std::size_t /*level*/, const auto& interval, const auto& index_yz)
                              {
                                  // static_nested_loop<dim - 1, -ghost_width - s, ghost_width
                                  // + s + 1>([&](auto stencil) {
                                  //     int beg = (interval.start >> 1) - static_cast<int>(s
                                  //     + ghost_width); int end = ((interval.end + 1) >> 1) +
                                  //     static_cast<int>(s + ghost_width);

                                  //     level_cell_list[(index_yz >> 1) +
                                  //     stencil].add_interval({beg, end});
                                  // });
                                  static_nested_loop<dim - 1, -config::ghost_width, config::ghost_width + 1>(
                                      [&](auto stencil)
                                      {
                                          int beg = (interval.start >> 1) - config::ghost_width;
                                          int end = ((interval.end + 1) >> 1) + config::ghost_width;

                                          lcl[(index_yz >> 1) + stencil].add_interval({beg, end});
                                      });
                              });
        }

        // compaction
        this->m_cells[mesh_id_t::all_cells] = {cell_list, false};

        for (std::size_t level = max_level; level >= ((min_level == 0) ? 1 : min_level); --level)
        {
            if (!this->m_cells[mesh_id_t::cells][level].empty())
            {
                auto expr = intersection(this->m_cells[mesh_id_t::union_cells][level],
                                         difference(this->m_cells[mesh_id_t::all_cells][level], this->m_cells[mesh_id_t::cells][level]))
                                .on(level - 1);

                expr(
                    [&](const auto& interval, const auto& index_yz)
                    {
                        lcl_type& lcl = cell_list[level];

                        static_nested_loop<dim - 1, 0, 2>(
                            [&](auto stencil)
                            {
                                lcl[(index_yz << 1) + stencil].add_interval({interval.start << 1, interval.end << 1});
                            });
                    });
            }
        }

        this->m_cells[mesh_id_t::all_cells] = {cell_list, false};
        for (std::size_t level = max_level; level >= ((min_level == 0) ? 1 : min_level); --level)
        {
            lcl_type lcl{level - 1};
            auto expr = intersection(this->m_cells[mesh_id_t::all_cells][level - 1], this->m_cells[mesh_id_t::union_cells][level - 1]);

            expr(
                [&](const auto& interval, const auto& index_yz)
                {
                    lcl[index_yz].add_interval({interval.start, interval.end});
                });

            this->m_cells[mesh_id_t::proj_cells][level - 1] = {lcl};
        }

        // Construct overleaves
        cl_type overleaves_list;

        const int cells_to_add = 1; // To be changed according to the numerical scheme

        for_each_interval(
            this->m_cells[mesh_id_t::cells],
            [&](std::size_t level, const auto& interval, const auto& index_yz)
            {
                if (level < this->max_level())
                {
                    lcl_type& lol = overleaves_list[level + 1]; // We have to put it at
                                                                // the higher level
                    lcl_type& lcl = cell_list[level + 1];       // We have to put it at the higher level

                    static_nested_loop<dim - 1, -cells_to_add, cells_to_add + 1, 1>(
                        [&](auto stencil)
                        {
                            auto index = xt::eval(index_yz + stencil);

                            lol[2 * index].add_interval({2 * (interval.start - cells_to_add), 2 * (interval.end + cells_to_add)});
                            lol[2 * index + 1].add_interval({2 * (interval.start - cells_to_add), 2 * (interval.end + cells_to_add)});

                            lcl[2 * index].add_interval({2 * (interval.start - cells_to_add), 2 * (interval.end + cells_to_add)});
                            lcl[2 * index + 1].add_interval({2 * (interval.start - cells_to_add), 2 * (interval.end + cells_to_add)});
                        });
                }
            });
        this->m_cells[mesh_id_t::overleaves] = {overleaves_list, false};

        this->m_cells[mesh_id_t::all_cells] = {cell_list}; // We must put the overleaves in the all cells to
                                                           // store them
    }

    template <class Config>
    template <typename... T>
    inline xt::xtensor<bool, 1> MROMesh<Config>::exists(mesh_id_t type, std::size_t level, interval_t interval, T... index) const
    {
        using coord_index_t      = typename interval_t::coord_index_t;
        const auto& lca          = this->m_cells[type][level];
        std::size_t size         = interval.size() / interval.step;
        xt::xtensor<bool, 1> out = xt::empty<bool>({size});
        std::size_t iout         = 0;
        for (coord_index_t i = interval.start; i < interval.end; i += interval.step)
        {
            auto row = find(lca, {i, index...});
            if (row == -1)
            {
                out[iout++] = false;
            }
            else
            {
                out[iout++] = true;
            }
        }
        return out;
    }
} // namespace samurai

template <>
struct fmt::formatter<samurai::MROMeshId> : formatter<string_view>
{
    // parse is inherited from formatter<string_view>.
    template <typename FormatContext>
    auto format(samurai::MROMeshId c, FormatContext& ctx)
    {
        string_view name = "unknown";
        switch (c)
        {
            case samurai::MROMeshId::cells:
                name = "cells";
                break;
            case samurai::MROMeshId::cells_and_ghosts:
                name = "cells and ghosts";
                break;
            case samurai::MROMeshId::proj_cells:
                name = "projection cells";
                break;
            case samurai::MROMeshId::union_cells:
                name = "union cells";
                break;
            case samurai::MROMeshId::overleaves:
                name = "overleaves";
                break;
            case samurai::MROMeshId::all_cells:
                name = "all cells";
                break;
            case samurai::MROMeshId::count:
                name = "count";
                break;
        }
        return formatter<string_view>::format(name, ctx);
    }
};
