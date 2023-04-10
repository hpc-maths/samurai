// Copyright 2021 SAMURAI TEAM. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#pragma once

#include <samurai/algorithm.hpp>
#include <samurai/subset/subset_op.hpp>

template <class Field>
void update_sol(double dt, Field& phi, Field& phi_np1)
{
    using mesh_id_t = typename Field::mesh_t::mesh_id_t;
    auto& mesh      = phi.mesh();

    samurai::for_each_interval(mesh[mesh_id_t::cells],
                               [&](std::size_t level, const auto& i, auto)
                               {
                                   const double dx = samurai::cell_length(level);

                                   phi_np1(level, i) = phi(level, i)
                                                     - .5 * dt / dx * (xt::pow(phi(level, i), 2.) - xt::pow(phi(level, i - 1), 2.));
                               });

    /////////////////////////

    /**
     * Flux correction
     * ~~~~~~~~~~~~~~~
     *
     * Left flux example
     *
     * |----|----|                                  |----|----|
     *                          ----------------->         x
     * |=========|---------|                        |=========|---------|
     *      x         x                                            x
     *
     */
    for (std::size_t level = mesh.min_level(); level < mesh.max_level(); ++level)
    {
        const double dx = samurai::cell_length(level);

        int stencil      = 1;
        auto subset_left = samurai::intersection(samurai::translate(mesh[mesh_id_t::cells][level + 1], stencil),
                                                 mesh[mesh_id_t::cells][level])
                               .on(level);

        subset_left(
            [&](const auto& i, const auto&)
            {
                phi_np1(level, i) = phi_np1(level, i) - .5 * dt / dx * xt::pow(phi(level, i - 1), 2.)
                                  + .5 * dt / dx * xt::pow(phi(level + 1, 2 * i - 1), 2.);
            });
    }
    /////////////////////////

    std::swap(phi.array(), phi_np1.array());
}
