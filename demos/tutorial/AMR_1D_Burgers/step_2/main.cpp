// Copyright 2021 SAMURAI TEAM. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include <samurai/box.hpp>
#include <samurai/cell_array.hpp>
#include <samurai/field.hpp>
#include <samurai/hdf5.hpp>

#include "../step_1/init_sol.hpp"
#include "update_sol.hpp"

/**
 * What will we learn ?
 * ====================
 *
 * - Apply a finite volume scheme
 * - save and plot a field
 *
 */

int main()
{
    constexpr std::size_t dim = 1;
    std::size_t init_level = 6;

    samurai::Box<double, dim> box({-3}, {3});
    samurai::CellArray<dim> mesh;

    mesh[init_level] = {init_level, box};

    auto phi = init_sol(mesh);

    /////////////////////////////////
    double Tf = 1.5;
    double dx = 1./(1 << init_level);
    double dt = 0.99 * dx;

    double t = 0.;
    std::size_t it = 0;

    auto phi_np1 = samurai::make_field<double, 1>("phi", mesh);

    while (t < Tf)
    {
        fmt::print("Iteration = {:4d} Time = {:5.4}\n", it, t);

        update_sol(dt, phi, phi_np1);

        t += dt;

        samurai::save(fmt::format("step_2-{}", it++), mesh, phi);
    }
    /////////////////////////////////

    return 0;
}