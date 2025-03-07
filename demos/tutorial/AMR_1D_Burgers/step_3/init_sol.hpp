// Copyright 2018-2025 the samurai's authors
// SPDX-License-Identifier:  BSD-3-Clause

#pragma once

#include <samurai/algorithm.hpp>
#include <samurai/field.hpp>

template <class Mesh>
auto init_sol(Mesh& mesh)
{
    using mesh_id_t = typename Mesh::mesh_id_t; // <-----------------
    auto phi        = samurai::make_field<double, 1>("phi", mesh);
    phi.fill(0.);

    samurai::for_each_cell(mesh[mesh_id_t::cells_and_ghosts],
                           [&](auto& cell) // <-----------------
                           {
                               double x = cell.center(0);

                               // Initial hat solution
                               if (x < -1. || x > 1.)
                               {
                                   phi[cell] = 0.;
                               }
                               else
                               {
                                   phi[cell] = (x < 0.) ? (1 + x) : (1 - x);
                               }
                           });

    return phi;
}
