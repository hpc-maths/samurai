// Copyright 2021 SAMURAI TEAM. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#pragma once

#include <samurai/subset/subset_op.hpp>

template <class Field>
void update_ghost(Field& phi)
{
    using mesh_id_t = typename Field::mesh_t::mesh_id_t;

    auto& mesh = phi.mesh();

    std::size_t min_level = mesh.min_level();
    std::size_t max_level = mesh.max_level();

    /**
     * Projection
     * ~~~~~~~~~~
     *
     *   |------|------|
     *
     *   |=============|-------------|
     */
    for (std::size_t level = max_level; level >= min_level; --level)
    {
        auto expr = samurai::intersection(mesh[mesh_id_t::cells][level], mesh[mesh_id_t::cells_and_ghosts][level - 1]).on(level - 1);

        expr(
            [&](const auto& i, auto)
            {
                phi(level - 1, i) = 0.5 * (phi(level, 2 * i) + phi(level, 2 * i + 1));
            });
    }

    /**
     * Boundary conditions
     * ~~~~~~~~~~~~~~~~~~~
     *
     *   phi = 0
     *
     */
    for (std::size_t level = min_level; level <= max_level; ++level)
    {
        auto expr = samurai::difference(mesh[mesh_id_t::cells_and_ghosts][level], mesh.domain()).on(level);

        expr(
            [&](const auto& i, auto)
            {
                phi(level, i) = 0.;
            });
    }

    /**
     * Prediction
     * ~~~~~~~~~~
     *
     *          |======|------|------|
     *
     *   |-------------|
     */
    for (std::size_t level = min_level + 1; level <= max_level; ++level)
    {
        auto expr = samurai::intersection(mesh.domain(),
                                          samurai::difference(mesh[mesh_id_t::cells_and_ghosts][level], mesh.get_union()[level]))
                        .on(level);

        expr(
            [&](const auto& i, auto)
            {
                auto i_coarse = i >> 1;
                if (i.start & 1)
                {
                    phi(level, i) = phi(level - 1, i_coarse) + 1. / 8 * (phi(level - 1, i_coarse + 1) - phi(level - 1, i_coarse - 1));
                }
                else
                {
                    phi(level, i) = phi(level - 1, i_coarse) - 1. / 8 * (phi(level - 1, i_coarse + 1) - phi(level - 1, i_coarse - 1));
                }
            });
    }
}
