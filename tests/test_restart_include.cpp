// Copyright 2018-2025 the samurai's authors
// SPDX-License-Identifier:  BSD-3-Clause

// samurai/io/restart.hpp must be the first include of this translation unit:
// the test checks that the header compiles on its own, before any other header
// has brought xtensor in. Keep it in its own include block.
#include <samurai/io/restart.hpp>

#include <gtest/gtest.h>

#include <samurai/box.hpp>
#include <samurai/field.hpp>
#include <samurai/level_cell_array.hpp>
#include <samurai/mr/mesh.hpp>

namespace samurai
{
    TEST(restart_include, restart_hpp_first_lca)
    {
        auto mesh = LevelCellArray<2>(4, Box<double, 2>({0, 0}, {1, 1}));
        decltype(mesh) mesh2;
        dump("restart_include_lca", mesh);
        load("restart_include_lca", mesh2);
        EXPECT_TRUE(mesh == mesh2);
    }

    TEST(restart_include, restart_hpp_first_mrmesh_with_field)
    {
        constexpr std::size_t dim = 2;
        Box<double, dim> box({0, 0}, {1, 1});
        auto mesh_cfg = mesh_config<dim>().min_level(2).max_level(5);
        auto mesh     = mra::make_mesh(box, mesh_cfg);
        auto u        = make_scalar_field<double>("u", mesh);
        for_each_cell(mesh,
                      [&](const auto& cell)
                      {
                          u[cell] = cell.center(0) + 2 * cell.center(1);
                      });
        dump("restart_include_mrmesh", mesh, u);

        decltype(mesh) mesh2;
        auto u2 = make_scalar_field<double>("u", mesh2);
        load("restart_include_mrmesh", mesh2, u2);
        EXPECT_TRUE(mesh == mesh2);
        for_each_cell(mesh2,
                      [&](const auto& cell)
                      {
                          EXPECT_EQ(u2[cell], cell.center(0) + 2 * cell.center(1));
                      });
    }
}
