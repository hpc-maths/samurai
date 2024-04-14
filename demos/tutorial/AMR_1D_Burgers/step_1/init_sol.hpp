// Copyright 2018-2024 the samurai's authors
// SPDX-License-Identifier:  BSD-3-Clause

#pragma once

#include <samurai/field.hpp>

template <class Mesh>
auto init_sol(Mesh& mesh)
{
    // create a field from the mesh
    auto phi = samurai::make_field<double, 1>("phi", mesh);

    samurai::for_each_cell(mesh,
                           [&](auto& cell)
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
