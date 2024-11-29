// Copyright 2018-2024 the samurai's authors
// SPDX-License-Identifier:  BSD-3-Clause

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
        static constexpr int max_stencil_width            = max_stencil_width_;
        static constexpr std::size_t graduation_width     = graduation_width_;
        static constexpr int prediction_order             = prediction_order_;

        // static constexpr int ghost_width = std::max(std::max(2 *
        // static_cast<int>(graduation_width) - 1,
        //                                                      static_cast<int>(max_stencil_width)),
        //                                             static_cast<int>(prediction_order));
        static constexpr int ghost_width = std::max(static_cast<int>(max_stencil_width), static_cast<int>(prediction_order));
        using interval_t                 = TInterval;
        using mesh_id_t                  = MRMeshId;
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

        MRMesh() = default;
        MRMesh(const cl_type& cl, const self_type& ref_mesh);
        MRMesh(const cl_type& cl, std::size_t min_level, std::size_t max_level);
        MRMesh(const samurai::Box<double, dim>& b,
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
    inline MRMesh<Config>::MRMesh(const cl_type& cl, const self_type& ref_mesh)
        : base_type(cl, ref_mesh)
    {
    }

    template <class Config>
    inline MRMesh<Config>::MRMesh(const cl_type& cl, std::size_t min_level, std::size_t max_level)
        : base_type(cl, min_level, max_level)
    {
    }

    template <class Config>
    inline MRMesh<Config>::MRMesh(const samurai::Box<double, dim>& b,
                                  std::size_t min_level,
                                  std::size_t max_level,
                                  double approx_box_tol,
                                  double scaling_factor_)
        : base_type(b, max_level, min_level, max_level, approx_box_tol, scaling_factor_)
    {
    }

    template <class Config>
    inline MRMesh<Config>::MRMesh(const samurai::Box<double, dim>& b,
                                  std::size_t min_level,
                                  std::size_t max_level,
                                  const std::array<bool, dim>& periodic,
                                  double approx_box_tol,
                                  double scaling_factor_)
        : base_type(b, max_level, min_level, max_level, periodic, approx_box_tol, scaling_factor_)
    {
    }

    template <class Config>
    inline void MRMesh<Config>::update_sub_mesh_impl()
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
        for_each_interval(
            this->cells()[mesh_id_t::cells],
            [&](std::size_t level, const auto& interval, const auto& index_yz)
            {
                lcl_type& lcl = cell_list[level];
                static_nested_loop<dim - 1, -config::max_stencil_width, config::max_stencil_width + 1>(
                    [&](auto stencil)
                    {
                        auto index = xt::eval(index_yz + stencil);
                        lcl[index].add_interval({interval.start - config::max_stencil_width, interval.end + config::max_stencil_width});
                    });
            });
        this->cells()[mesh_id_t::cells_and_ghosts] = {cell_list, false};

        // Add cells for the MRA
        if (this->max_level() != this->min_level())
        {
            for (std::size_t level = max_level; level >= ((min_level == 0) ? 1 : min_level); --level)
            {
                auto expr = difference(this->cells()[mesh_id_t::cells_and_ghosts][level], this->get_union()[level]).on(level);

                expr(
                    [&](const auto& interval, const auto& index_yz)
                    {
                        lcl_type& lcl = cell_list[level - 1];

                        static_nested_loop<dim - 1, -config::prediction_order, config::prediction_order + 1>(
                            [&](auto stencil)
                            {
                                auto new_interval = interval >> 1;
                                lcl[(index_yz >> 1) + stencil].add_interval(
                                    {new_interval.start - config::prediction_order, new_interval.end + config::prediction_order});
                            });
                    });

                auto expr_2 = intersection(this->cells()[mesh_id_t::cells][level], this->cells()[mesh_id_t::cells][level]);

                expr_2(
                    [&](const auto& interval, const auto& index_yz)
                    {
                        if (level - 1 > 0)
                        {
                            lcl_type& lcl = cell_list[level - 2];

                            static_nested_loop<dim - 1, -config::prediction_order, config::prediction_order + 1>(
                                [&](auto stencil)
                                {
                                    auto new_interval = interval >> 2;
                                    lcl[(index_yz >> 2) + stencil].add_interval(
                                        {new_interval.start - config::prediction_order, new_interval.end + config::prediction_order});
                                });
                        }
                    });
            }
            this->cells()[mesh_id_t::all_cells] = {cell_list, false};

            this->update_mesh_neighbour();

            for (auto& neighbour : this->mpi_neighbourhood())
            {
                for (std::size_t level = 0; level <= max_level; ++level)
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

                                static_nested_loop<dim - 1, -config::prediction_order, config::prediction_order + 1>(
                                    [&](auto stencil)
                                    {
                                        auto new_interval = interval >> 1;
                                        lclm1[(index_yz >> 1) + stencil].add_interval(
                                            {new_interval.start - config::prediction_order, new_interval.end + config::prediction_order});
                                    });
                            }
                        });
                }
            }
            this->cells()[mesh_id_t::all_cells] = {cell_list, false};

            // add ghosts for periodicity
            xt::xtensor_fixed<typename interval_t::value_t, xt::xshape<dim>> stencil;
            xt::xtensor_fixed<typename interval_t::value_t, xt::xshape<dim>> min_corner;
            xt::xtensor_fixed<typename interval_t::value_t, xt::xshape<dim>> max_corner;

            auto& domain     = this->domain();
            auto min_indices = domain.min_indices();
            auto max_indices = domain.max_indices();

            for (std::size_t level = this->cells()[mesh_id_t::reference].min_level();
                 level <= this->cells()[mesh_id_t::reference].max_level();
                 ++level)
            {
                std::size_t delta_l = domain.level() - level;
                lcl_type& lcl       = cell_list[level];

                for (std::size_t d = 0; d < dim; ++d)
                {
                    if (this->is_periodic(d))
                    {
                        stencil.fill(0);
                        stencil[d] = (max_indices[d] - min_indices[d]) >> delta_l;

                        min_corner[d] = (min_indices[d] >> delta_l) - config::ghost_width;
                        max_corner[d] = (min_indices[d] >> delta_l) + config::ghost_width;
                        for (std::size_t dd = 0; dd < dim; ++dd)
                        {
                            if (dd != d)
                            {
                                min_corner[dd] = (min_indices[dd] >> delta_l) - config::ghost_width;
                                max_corner[dd] = (max_indices[dd] >> delta_l) + config::ghost_width;
                            }
                        }

                        lca_type lca1{
                            level,
                            Box<typename interval_t::value_t, dim>{min_corner, max_corner}
                        };

                        auto set1 = intersection(this->cells()[mesh_id_t::reference][level], lca1);
                        set1(
                            [&](const auto& i, const auto& index_yz)
                            {
                                lcl[index_yz + xt::view(stencil, xt::range(1, _))].add_interval(i + stencil[0]);
                            });

                        min_corner[d] = (max_indices[d] >> delta_l) - config::ghost_width;
                        max_corner[d] = (max_indices[d] >> delta_l) + config::ghost_width;
                        lca_type lca2{
                            level,
                            Box<typename interval_t::value_t, dim>{min_corner, max_corner}
                        };

                        auto set2 = intersection(this->cells()[mesh_id_t::reference][level], lca2);
                        set2(
                            [&](const auto& i, const auto& index_yz)
                            {
                                lcl[index_yz - xt::view(stencil, xt::range(1, _))].add_interval(i - stencil[0]);
                            });
                        this->cells()[mesh_id_t::all_cells][level] = {lcl};
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
            this->update_mesh_neighbour();
        }
        else
        {
            this->cells()[mesh_id_t::all_cells] = {cell_list, false};
            this->update_mesh_neighbour();
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
