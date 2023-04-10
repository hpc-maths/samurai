// Copyright 2021 SAMURAI TEAM. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

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
