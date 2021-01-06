// Copyright 2021 SAMURAI TEAM. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#pragma once

#include <xtensor/xmasked_view.hpp>

#include "criteria.hpp"
#include "../field.hpp"

namespace samurai
{
    template <class Field, class Func>
    bool coarsening(Field &u, Func&& update_bc_for_level, double eps, std::size_t ite)
    {
        constexpr std::size_t dim = Field::dim;
        constexpr std::size_t size = Field::size;

        using value_t = typename Field::value_type;
        using mesh_t = typename Field::mesh_t;
        using mesh_id_t = typename mesh_t::mesh_id_t;
        using interval_t = typename mesh_t::interval_t;
        using coord_index_t = typename interval_t::coord_index_t;
        using cl_type = typename mesh_t::cl_type;

        auto mesh = u.mesh();
        std::size_t min_level = mesh.min_level(), max_level = mesh.max_level();

        auto detail = make_field<value_t, size>("detail", mesh);

        auto tag = make_field<int, 1>("tag", mesh);
        tag.fill(0);

        mesh.for_each_cell([&](auto &cell)
        {
            tag[cell] = static_cast<int>(CellFlag::keep);
        });

        mr_projection(u);
        for (std::size_t level = min_level - 1; level <= max_level; ++level)
        {
            update_bc_for_level(u, level);
        }
        mr_prediction(u, update_bc_for_level);

        // What are the data it uses at min_level - 1 ???
        for (std::size_t level = min_level - 1; level < max_level - ite; ++level)
        {
            auto subset = intersection(mesh[mesh_id_t::reference][level],
                                       mesh[mesh_id_t::cells][level + 1])
                         .on(level);
            subset.apply_op(compute_detail(detail, u));
        }

        // AGAIN I DONT KNOW WHAT min_level - 1 is
        for (std::size_t level = min_level; level <= max_level - ite; ++level)
        {
            int exponent = dim * (level - max_level);

            auto eps_l = std::pow(2, exponent) * eps;

            // COMPRESSION

            auto subset_1 = intersection(mesh[mesh_id_t::cells][level],
                                         mesh[mesh_id_t::reference][level-1])
                           .on(level-1);

            // This operations flags the cells to coarsen
            subset_1.apply_op(to_coarsen_mr(detail, tag, eps_l, min_level));

            auto subset_2 = intersection(mesh[mesh_id_t::cells][level],
                                         mesh[mesh_id_t::cells][level]);
            auto subset_3 = intersection(mesh[mesh_id_t::cells_and_ghosts][level],
                                         mesh[mesh_id_t::cells_and_ghosts][level]);

            subset_2.apply_op(enlarge(tag));
            subset_3.apply_op(tag_to_keep(tag));
        }

        for (std::size_t level = max_level; level > 0; --level)
        {
            auto keep_subset = intersection(mesh[mesh_id_t::cells][level],
                                            mesh[mesh_id_t::reference][level - 1])
                              .on(level - 1);
            keep_subset.apply_op(maximum(tag));

            xt::xtensor_fixed<int, xt::xshape<dim>> stencil;
            for (std::size_t d = 0; d < dim; ++d)
            {
                stencil.fill(0);
                for (int s = -1; s <= 1; ++s)
                {
                    if (s != 0)
                    {
                        stencil[d] = s;
                        auto subset = intersection(mesh[mesh_id_t::cells][level],
                                                   translate(mesh[mesh_id_t::cells][level - 1], stencil))
                                     .on(level - 1);
                        subset.apply_op(balance_2to1(tag, stencil));
                    }
                }
            }
        }

        cl_type cell_list;

        for_each_interval(mesh[mesh_id_t::cells], [&](std::size_t level, const auto& interval, const auto& index_yz)
        {
            for (coord_index_t i = interval.start; i < interval.end; ++i)
            {
                if (tag[i + interval.index] & static_cast<int>(CellFlag::keep))
                {
                    cell_list[level][index_yz].add_point(i);
                }
                else
                {
                    cell_list[level-1][index_yz>>1].add_point(i>>1);
                }
            }
        });

        mesh_t new_mesh{cell_list, mesh.initial_mesh(), min_level, max_level};

        if (new_mesh == mesh)
        {
            return true;
        }

        auto new_u = make_field<value_t, size>(u.name(), new_mesh);

        for (std::size_t level = min_level; level <= max_level; ++level)
        {
            auto subset = intersection(mesh[mesh_id_t::reference][level],
                                       new_mesh[mesh_id_t::cells][level]);
            subset.apply_op(copy(new_u, u));
        }

        u.mesh_ptr()->swap(new_mesh);
        std::swap(u.array(), new_u.array());

        return false;
    }
} // namespace samurai