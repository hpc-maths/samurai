// Copyright 2021 SAMURAI TEAM. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include <samurai/box.hpp>
#include <samurai/cell_array.hpp>
#include <samurai/hdf5.hpp>

/**
 * What will we learn ?
 * ====================
 *
 * - construct 1D uniform grid from a Box
 * - print mesh information
 * - save and plot a mesh
 *
 */

int main()
{
    constexpr std::size_t dim = 1;
    std::size_t init_level = 4;

    /**
     *
     * level: 1   |--|--|--|--|--|--|--|--|
     *
     * level: 0   |-----|-----|-----|-----|
     *           -2    -1     0     1     2
     */

    samurai::Box<double, dim> box({-2}, {2});
    samurai::CellArray<dim> mesh;

    mesh[init_level] = {init_level, box};

    std::cout << mesh << "\n";

    samurai::save("step_0", mesh);

    return 0;
}