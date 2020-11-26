#pragma once

#include <samurai/algorithm.hpp>

template<class Field>
void update_sol(double dt, Field& phi, Field& phi_np1)
{
    auto mesh = phi.mesh();
    using mesh_id_t = typename decltype(mesh)::mesh_id_t;

    samurai::for_each_interval(mesh[mesh_id_t::cells], [&](std::size_t level, const auto& i, auto)
    {
        double dx = 1./(1<<level);

        phi_np1(level, i) = phi(level, i) - .5*dt/dx*(xt::pow(phi(level, i), 2.) - xt::pow(phi(level, i - 1), 2.));
    });

    /////////////////////////
    for (std::size_t level = mesh.min_level(); level < mesh.max_level(); ++level)
    {
        double dx = 1./(1<<level);

        int stencil = 1;
        auto subset_left = samurai::intersection(samurai::translate(mesh[mesh_id_t::cells][level+1], stencil),
                                                 mesh[mesh_id_t::cells][level])
                           .on(level);

        subset_left([&](const auto& i, const auto& )
        {
            phi_np1(level, i) = phi_np1(level, i) - .5*dt/dx*xt::pow(phi(level, i - 1), 2.)
                                                  + .5*dt/dx*xt::pow(phi(level + 1, 2*i - 1), 2.);
        });
    }
    /////////////////////////

    std::swap(phi.array(), phi_np1.array());
}