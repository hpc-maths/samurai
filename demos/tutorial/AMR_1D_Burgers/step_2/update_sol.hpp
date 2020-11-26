#pragma once

#include <samurai/algorithm.hpp>

template<class Field>
void update_sol(double dt, Field& phi, Field& phi_np1)
{
    auto mesh = phi.mesh();

    samurai::for_each_interval(mesh, [&](std::size_t level, const auto& interval, auto)
    {
        using interval_t = decltype(interval);

        double dx = 1./(1<<level);

        auto new_i = interval_t{interval.start + 1, interval.end - 1};

        phi_np1(level, new_i) = phi(level, new_i) - .5*dt/dx*(xt::pow(phi(level, new_i), 2.) - xt::pow(phi(level, new_i - 1), 2.));
    });

    std::swap(phi.array(), phi_np1.array());
}