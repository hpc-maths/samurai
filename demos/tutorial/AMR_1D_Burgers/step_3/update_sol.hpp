// Copyright 2018-2024 the samurai's authors
// SPDX-License-Identifier:  BSD-3-Clause

#pragma once

#include <samurai/algorithm.hpp>

template <class Field>
void update_sol(double dt, Field& phi, Field& phi_np1)
{
    using mesh_id_t = typename Field::mesh_t::mesh_id_t; // <-----------------
    auto& mesh      = phi.mesh();

    samurai::for_each_interval(
        mesh[mesh_id_t::cells],
        [&](std::size_t level, const auto& i, auto) // <-----------------
        {
            const double dx = samurai::cell_length(level);

            phi_np1(level, i) = phi(level, i)
                              - .5 * dt / dx * (phi(level, i) * phi(level, i) - phi(level, i - 1) * phi(level, i - 1)); // <-----------------
        });

    std::swap(phi.array(), phi_np1.array());
}
