// Copyright 2018-2024 the samurai's authors
// SPDX-License-Identifier:  BSD-3-Clause

#pragma once

#include <samurai/algorithm.hpp>

template <class Field>
void update_sol(double dt, Field& phi, Field& phi_np1)
{
    auto& mesh = phi.mesh();

    samurai::for_each_interval(mesh,
                               [&](std::size_t level, const auto& interval, auto)
                               {
                                   using interval_t = decltype(interval);

                                   const double dx = mesh.cell_length(level);

                                   // remove the extrema to avoid problem with the boundaries
                                   auto ii = interval_t{interval.start + 1, interval.end - 1};

                                   // upwind scheme
                                   phi_np1(level, ii) = phi(level, ii)
                                                      - .5 * dt / dx * (pow(phi(level, ii), 2.) - pow(phi(level, ii - 1), 2.));
                                   // phi_np1(level, ii) = phi(level, ii) - .5*dt/dx*(phi(level,
                                   // ii)*phi(level, ii) - phi(level, ii - 1)*phi(level, ii - 1));
                               });

    std::swap(phi.array(), phi_np1.array());
}
