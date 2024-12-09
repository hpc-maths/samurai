// Copyright 2018-2024 the samurai's authors
// SPDX-License-Identifier:  BSD-3-Clause

#pragma once

#include "update_field.hpp"

template <class Field, class Tag>
bool update_mesh(Field& f, const Tag& tag)
{
    using mesh_t    = typename Field::mesh_t;
    using size_type = typename Field::size_type;

    using mesh_id_t = typename mesh_t::mesh_id_t;
    using cl_type   = typename mesh_t::cl_type;

    /**
     *
     * mesh with tag:
     * ==============
     *
     *                                     K     R     K     C     C
     * level: 2                         |-----|-----|-----|-----|-----|
     *                            K
     * level: 1              |----------|
     *
     * New mesh:
     * =========
     *
     * level: 3                               |==|==|
     *
     * level: 2                         |=====|     |=====|
     *
     * level: 1              |==========|                 |===========|
     *
     */

    auto& mesh = f.mesh();

    cl_type cell_list(mesh.origin_point(), mesh.scaling_factor());

    samurai::for_each_interval(mesh[mesh_id_t::cells],
                               [&](std::size_t level, const auto& interval, const auto&)
                               {
                                   auto itag = static_cast<size_type>(interval.start + interval.index);

                                   for (auto i = interval.start; i < interval.end; ++i)
                                   {
                                       if (tag[itag] & static_cast<int>(samurai::CellFlag::refine))
                                       {
                                           cell_list[level + 1][{}].add_interval({2 * i, 2 * i + 2});
                                       }
                                       else if (tag[itag] & static_cast<int>(samurai::CellFlag::keep))
                                       {
                                           cell_list[level][{}].add_point(i);
                                       }
                                       else
                                       {
                                           cell_list[level - 1][{}].add_point(i >> 1);
                                       }
                                       itag++;
                                   }
                               });

    mesh_t new_mesh(cell_list, mesh.min_level(), mesh.max_level());

    if (new_mesh == mesh)
    {
        return true;
    }

    update_field(f, tag, new_mesh);

    return false;
}
