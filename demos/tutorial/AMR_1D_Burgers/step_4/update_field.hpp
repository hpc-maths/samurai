// Copyright 2021 SAMURAI TEAM. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#pragma once

#include <samurai/cell_flag.hpp>
#include <samurai/subset/subset_op.hpp>

template <class Field, class Tag, class Mesh>
void update_field(Field& f, const Tag& tag, Mesh& new_mesh)
{
    using mesh_t        = typename Field::mesh_t;
    using mesh_id_t     = typename mesh_t::mesh_id_t;
    using interval_t    = typename mesh_t::interval_t;
    using coord_index_t = typename interval_t::coord_index_t;

    auto& mesh = f.mesh();

    Field new_f{f.name(), new_mesh};
    new_f.fill(0.);

    /**
     *
     * mesh     : -----
     * new mesh : =====
     *
     * level: 3                               |==|==|
     *
     * level: 2                         |-----|-----|-----|-----|-----|
     *                                  |=====|     |=====|
     *
     * level: 1              |----------|
     *                       |==========|                 |===========|
     */

    for (std::size_t level = mesh.min_level(); level <= mesh.max_level(); ++level)
    {
        auto common_leaves = samurai::intersection(mesh[mesh_id_t::cells][level], new_mesh[mesh_id_t::cells][level]);

        common_leaves(
            [&](const auto& i, auto)
            {
                new_f(level, i) = f(level, i);
            });
    }

    samurai::for_each_interval(mesh[mesh_id_t::cells],
                               [&](std::size_t level, const auto& interval, const auto&)
                               {
                                   auto itag = interval.start + interval.index;
                                   for (coord_index_t i = interval.start; i < interval.end; ++i)
                                   {
                                       if (tag[itag] & static_cast<int>(samurai::CellFlag::refine))
                                       {
                                           auto ii                      = interval_t{i, i + 1};
                                           new_f(level + 1, 2 * ii)     = f(level, ii) - 1. / 8 * (f(level, ii + 1) - f(level, ii - 1));
                                           new_f(level + 1, 2 * ii + 1) = f(level, ii) + 1. / 8 * (f(level, ii + 1) - f(level, ii - 1));
                                       }
                                       itag++;
                                   }
                               });

    for (std::size_t level = mesh.min_level() + 1; level <= mesh.max_level(); ++level)
    {
        auto subset = samurai::intersection(mesh[mesh_id_t::cells][level], new_mesh[mesh_id_t::cells][level - 1]).on(level - 1);
        subset(
            [&](const auto& i, auto)
            {
                new_f(level - 1, i) = 0.5 * (f(level, 2 * i) + f(level, 2 * i + 1));
            });
    }

    f.mesh().swap(new_mesh);
    std::swap(f.array(), new_f.array());
}
