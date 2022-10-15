// Copyright 2021 SAMURAI TEAM. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#pragma once

#include <samurai/algorithm.hpp>

template<class Field>
void update_sol(double dt, Field& phi, Field& phi_np1)
{
    auto mesh = phi.mesh();
    using mesh_id_t = typename decltype(mesh)::mesh_id_t;                                                                 // <-----------------

    samurai::for_each_interval(mesh[mesh_id_t::cells], [&](std::size_t level, const auto& i, auto)                        // <-----------------
    {
        double dx = 1./(1<<level);

        phi_np1(level, i) = phi(level, i) - .5*dt/dx*(phi(level, i)*phi(level, i) - phi(level, i - 1)*phi(level, i - 1)); // <-----------------
    });

    std::swap(phi.array(), phi_np1.array());
}