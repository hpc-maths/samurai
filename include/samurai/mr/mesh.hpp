// Copyright 2021 SAMURAI TEAM. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#pragma once

#include <fmt/format.h>

#include <xtensor/xtensor.hpp>
#include <xtensor/xview.hpp>

#include "../box.hpp"
#include "../mesh.hpp"
#include "../samurai_config.hpp"
#include "../subset/node_op.hpp"
#include "../subset/subset_op.hpp"

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
              std::size_t prediction_order_     = default_config::prediction_order,
              std::size_t max_refinement_level_ = default_config::max_level,
              class TInterval                   = default_config::interval_t>
    struct MRConfig
    {
        static constexpr std::size_t dim                  = dim_;
        static constexpr std::size_t max_refinement_level = max_refinement_level_;
        static constexpr std::size_t max_stencil_width    = max_stencil_width_;
        static constexpr std::size_t graduation_width     = graduation_width_;
        static constexpr std::size_t prediction_order     = prediction_order_;

        // static constexpr int ghost_width = std::max(std::max(2 *
        // static_cast<int>(graduation_width) - 1,
        //                                                      static_cast<int>(max_stencil_width)),
        //                                             static_cast<int>(prediction_order));
        static constexpr int ghost_width = std::max(static_cast<int>(max_stencil_width), 2 * static_cast<int>(prediction_order));
        using interval_t                 = TInterval;
        using mesh_id_t                  = MRMeshId;
    };

    template <class Config>
    class MRMesh : public samurai::Mesh_base<MRMesh<Config>, Config>
    {
      public:

        using base_type = samurai::Mesh_base<MRMesh<Config>, Config>;

        using config                     = typename base_type::config;
        static constexpr std::size_t dim = config::dim;

        using mesh_id_t  = typename base_type::mesh_id_t;
        using interval_t = typename base_type::interval_t;
        using cl_type    = typename base_type::cl_type;
        using lcl_type   = typename base_type::lcl_type;

        using ca_type  = typename base_type::ca_type;
        using lca_type = typename base_type::lca_type;

        MRMesh(const cl_type& cl, std::size_t min_level, std::size_t max_level);
        MRMesh(const cl_type& cl, std::size_t min_level, std::size_t max_level, const std::array<bool, dim>& periodic);
        MRMesh(const samurai::Box<double, dim>& b, std::size_t min_level, std::size_t max_level);
        MRMesh(const samurai::Box<double, dim>& b, std::size_t min_level, std::size_t max_level, const std::array<bool, dim>& periodic);

        void update_sub_mesh_impl();

        template <typename... T>
        xt::xtensor<bool, 1> exists(mesh_id_t type, std::size_t level, interval_t interval, T... index) const;
    };

    template <class Config>
    inline MRMesh<Config>::MRMesh(const cl_type& cl, std::size_t min_level, std::size_t max_level)
        : base_type(cl, min_level, max_level)
    {
    }

    template <class Config>
    inline MRMesh<Config>::MRMesh(const cl_type& cl, std::size_t min_level, std::size_t max_level, const std::array<bool, dim>& periodic)
        : base_type(cl, min_level, max_level, periodic)
    {
    }

    template <class Config>
    inline MRMesh<Config>::MRMesh(const samurai::Box<double, dim>& b, std::size_t min_level, std::size_t max_level)
        : base_type(b, max_level, min_level, max_level)
    {
    }

    template <class Config>
    inline MRMesh<Config>::MRMesh(const samurai::Box<double, dim>& b,
                                  std::size_t min_level,
                                  std::size_t max_level,
                                  const std::array<bool, dim>& periodic)
        : base_type(b, max_level, min_level, max_level, periodic)
    {
    }

    template <class Config>
    inline void MRMesh<Config>::update_sub_mesh_impl()
    {
        auto max_level = this->cells()[mesh_id_t::cells].max_level();
        auto min_level = this->cells()[mesh_id_t::cells].min_level();
        cl_type cell_list;

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
        this->cells()[mesh_id_t::union_cells][max_level] = {max_level};

        for (std::size_t level = max_level; level >= ((min_level == 0) ? 1 : min_level); --level)
        {
            lcl_type lcl{level - 1};
            auto expr = union_(this->cells()[mesh_id_t::cells][level], this->cells()[mesh_id_t::union_cells][level]).on(level - 1);

            expr(
                [&](const auto& interval, const auto& index_yz)
                {
                    lcl[index_yz].add_interval({interval.start, interval.end});
                });

            this->cells()[mesh_id_t::union_cells][level - 1] = {lcl};
        }

        // Construction of ghost cells
        // ===========================
        //
        // Example with ghost_width = 1
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
                              static_nested_loop<dim - 1, -config::ghost_width, config::ghost_width + 1>(
                                  [&](auto stencil)
                                  {
                                      auto index = xt::eval(index_yz + stencil);
                                      lcl[index].add_interval({interval.start - config::ghost_width, interval.end + config::ghost_width});
                                  });
                          });
        this->cells()[mesh_id_t::cells_and_ghosts] = {cell_list, false};

        // Construction of projection cells
        // ================================
        //
        // The projection cells are used for the computation of the details
        // involved in the multiresolution. The process is to take the children
        // cells and use them to make the projection on the parent cell. To do
        // that, we have to be sure that those cells exist.
        //

        // level 2                         |-|-|-|-|                     |-|
        // cells
        //                                                               |.|
        //                                                               projection
        //                                                               cells
        // level 1                 |---|---|...|...|---|---|
        //
        // level 0         |-------|.......|       |.......|-------|
        //

        for (std::size_t level = ((min_level == 0) ? 1 : min_level); level <= max_level; ++level)
        {
            lca_type& lca = this->cells()[mesh_id_t::cells][level];
            lcl_type& lcl = cell_list[level - 1];

            for_each_interval(lca,
                              [&](std::size_t /*level*/, const auto& interval, const auto& index_yz)
                              {
                                  static_nested_loop<dim - 1, -config::ghost_width, config::ghost_width + 1>(
                                      [&](auto stencil)
                                      {
                                          int beg = (interval.start >> 1) - config::ghost_width;
                                          int end = ((interval.end + 1) >> 1) + config::ghost_width;

                                          lcl[(index_yz >> 1) + stencil].add_interval({beg, end});
                                      });
                              });
        }
        this->cells()[mesh_id_t::all_cells] = {cell_list, false};

        // Make sure that the ghost cells where their values are computed using
        // the prediction operator have enough cells on the coarse level below.
        //
        // Example with a stencil of the prediction operator equals to 1.
        //
        // level l                            |.|-|-|        |-| cells
        //                                                   |.| ghost computed
        //                                                   with prediction
        //                                                   operator
        // level l - 1                  |xxx|---|            |x| ghost added to
        // be able to compute the ghost cell on the level l
        //
        for (std::size_t level = max_level; level >= ((min_level == 0) ? 1 : min_level); --level)
        {
            if (!this->cells()[mesh_id_t::cells][level].empty())
            {
                auto expr = intersection(
                                difference(this->cells()[mesh_id_t::all_cells][level],
                                           union_(this->cells()[mesh_id_t::cells][level], this->cells()[mesh_id_t::union_cells][level])),
                                this->domain())
                                .on(level - 1);

                expr(
                    [&](const auto& interval, const auto& index_yz)
                    {
                        lcl_type& lcl = cell_list[level - 1];

                        static_nested_loop<dim - 1, -config::ghost_width, config::ghost_width + 1>(
                            [&](auto stencil)
                            {
                                lcl[index_yz + stencil].add_interval(
                                    {interval.start - config::ghost_width, interval.end + config::ghost_width});
                            });
                    });
            }
        }
        this->cells()[mesh_id_t::all_cells] = {cell_list, false};

        // add ghosts for periodicity
        xt::xtensor_fixed<typename interval_t::value_t, xt::xshape<dim>> stencil;
        xt::xtensor_fixed<typename interval_t::value_t, xt::xshape<dim>> stencil_dir;
        auto& domain     = this->domain();
        auto min_indices = domain.min_indices();
        auto max_indices = domain.max_indices();

        // FIX: cppcheck false positive ?
        // cppcheck-suppress constStatement
        for (std::size_t level = this->cells()[mesh_id_t::reference].min_level(); level <= this->cells()[mesh_id_t::reference].max_level();
             ++level)
        {
            lcl_type& lcl = cell_list[level];

            for (std::size_t d = 0; d < dim; ++d)
            {
                std::size_t delta_l = domain.level() - level;

                if (this->is_periodic(d))
                {
                    stencil.fill(0);
                    stencil[d] = max_indices[d] - min_indices[d];

                    auto set1 = intersection(this->cells()[mesh_id_t::reference][level],
                                             expand(translate(domain, stencil), config::ghost_width << delta_l))
                                    .on(level);
                    set1(
                        [&](const auto& i, const auto& index_yz)
                        {
                            lcl[index_yz - (xt::view(stencil, xt::range(1, _)) >> delta_l)].add_interval(i - (stencil[0] >> delta_l));
                        });

                    auto set2 = intersection(this->cells()[mesh_id_t::reference][level],
                                             expand(translate(domain, -stencil), config::ghost_width << delta_l))
                                    .on(level);
                    set2(
                        [&](const auto& i, const auto& index_yz)
                        {
                            lcl[index_yz + (xt::view(stencil, xt::range(1, _)) >> delta_l)].add_interval(i + (stencil[0] >> delta_l));
                        });
                }
                this->cells()[mesh_id_t::all_cells][level] = {lcl};
            }
        }

        // Add ghost cells for the projection operator
        //
        // Example
        //
        // level l                  |-|-|.|x|       |-| cells
        //                                          |.| ghost cell
        // level l - 1          |---|...|...|       |x| ghost added to be able
        // to compute
        //                                              the ghost cell on the
        //                                              level l - 1 using the
        //                                              projection operator
        //
        for (std::size_t level = ((min_level == 0) ? 1 : min_level); level <= max_level; ++level)
        {
            auto expr = intersection(this->cells()[mesh_id_t::union_cells][level - 1], this->cells()[mesh_id_t::all_cells][level - 1])
                            .on(level - 1);

            lcl_type& lcl = cell_list[level];

            expr(
                [&](const auto& interval, const auto& index_yz)
                {
                    static_nested_loop<dim - 1, 0, 2>(
                        [&](auto stencil)
                        {
                            lcl[(index_yz << 1) + stencil].add_interval({interval.start << 1, interval.end << 1});
                        });
                });
            this->cells()[mesh_id_t::all_cells][level] = {lcl};
        }

        // add ghosts for periodicity
        // FIX: cppcheck false positive ?
        // cppcheck-suppress constStatement
        for (std::size_t level = this->cells()[mesh_id_t::reference].min_level(); level <= this->cells()[mesh_id_t::reference].max_level();
             ++level)
        {
            lcl_type& lcl = cell_list[level];

            for (std::size_t d = 0; d < dim; ++d)
            {
                std::size_t delta_l = domain.level() - level;

                if (this->is_periodic(d))
                {
                    stencil.fill(0);
                    stencil[d] = max_indices[d] - min_indices[d];

                    auto set1 = intersection(this->cells()[mesh_id_t::reference][level],
                                             expand(translate(domain, stencil), config::ghost_width << delta_l))
                                    .on(level);
                    set1(
                        [&](const auto& i, const auto& index_yz)
                        {
                            lcl[index_yz - (xt::view(stencil, xt::range(1, _)) >> delta_l)].add_interval(i - (stencil[0] >> delta_l));
                        });

                    auto set2 = intersection(this->cells()[mesh_id_t::reference][level],
                                             expand(translate(domain, -stencil), config::ghost_width << delta_l))
                                    .on(level);
                    set2(
                        [&](const auto& i, const auto& index_yz)
                        {
                            lcl[index_yz + (xt::view(stencil, xt::range(1, _)) >> delta_l)].add_interval(i + (stencil[0] >> delta_l));
                        });
                }
                this->cells()[mesh_id_t::all_cells][level] = {lcl};
            }
        }

        this->cells()[mesh_id_t::all_cells].update_index();

        // Extract the projection cells from the all_cells
        // Do we really need this ?
        // See if we can use the set definition directly into the projection
        // function
        //
        for (std::size_t level = max_level; level >= ((min_level == 0) ? 1 : min_level); --level)
        {
            lcl_type lcl{level - 1};
            auto expr = intersection(this->cells()[mesh_id_t::all_cells][level - 1], this->cells()[mesh_id_t::union_cells][level - 1]);

            expr(
                [&](const auto& interval, const auto& index_yz)
                {
                    lcl[index_yz].add_interval({interval.start, interval.end});
                });

            this->cells()[mesh_id_t::proj_cells][level - 1] = {lcl};
        }
    }

    template <class Config>
    template <typename... T>
    inline xt::xtensor<bool, 1> MRMesh<Config>::exists(mesh_id_t type, std::size_t level, interval_t interval, T... index) const
    {
        using coord_index_t      = typename interval_t::coord_index_t;
        const auto& lca          = this->cells()[type][level];
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
struct fmt::formatter<samurai::MRMeshId> : formatter<string_view>
{
    // parse is inherited from formatter<string_view>.
    template <typename FormatContext>
    auto format(samurai::MRMeshId c, FormatContext& ctx)
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
