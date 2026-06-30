// Copyright 2018-2025 the samurai's authors
// SPDX-License-Identifier:  BSD-3-Clause

// Weight policies and local load metric of the load balancing module
// (roadmap step 2). These pieces are MPI-free and tested sequentially.

#include <cmath>

#include <gtest/gtest.h>

#include <samurai/box.hpp>
#include <samurai/field.hpp>
#include <samurai/load_balancing/metrics.hpp>
#include <samurai/load_balancing/weight.hpp>
#include <samurai/mr/mesh.hpp>

namespace lb = samurai::load_balancing;

namespace
{
    constexpr std::size_t dim = 2;
    using Mesh                = samurai::MRMesh<samurai::mesh_config<dim>>;
    using mesh_id_t           = Mesh::mesh_id_t;

    constexpr std::size_t level = 4; // uniform 16x16 mesh on [0,1]^2

    Mesh make_uniform_mesh()
    {
        const samurai::Box<double, dim> box({0., 0.}, {1., 1.});
        return samurai::mra::make_mesh(box, samurai::mesh_config<dim>().min_level(level).max_level(level));
    }

    /// Left half at level+1, right half at level (graduated at x = 0.5).
    Mesh make_two_level_mesh()
    {
        const samurai::Box<double, dim> box({0., 0.}, {1., 1.});
        auto fine = samurai::mra::make_mesh(box, samurai::mesh_config<dim>().min_level(level).max_level(level + 1));

        Mesh::cl_type cl;
        samurai::for_each_cell(fine[mesh_id_t::cells],
                               [&](const auto& cell)
                               {
                                   auto yz = xt::view(cell.indices, xt::range(1, cell.indices.size()));
                                   if (cell.center(0) < 0.5)
                                   {
                                       cl[cell.level][yz].add_point(cell.indices[0]);
                                   }
                                   else
                                   {
                                       auto yz_coarse = xt::eval(yz / 2);
                                       cl[cell.level - 1][yz_coarse].add_point(cell.indices[0] >> 1);
                                   }
                               });
        return Mesh(cl, fine);
    }

    TEST(load_balancing_weight, uniform)
    {
        auto mesh = make_uniform_mesh();
        auto w    = lb::weight::uniform();

        samurai::for_each_cell(mesh[mesh_id_t::cells],
                               [&](const auto& cell)
                               {
                                   EXPECT_EQ(w(cell), 1.0);
                               });
        // local load == cell count for the uniform policy
        EXPECT_EQ(lb::local_load(mesh, w), static_cast<double>(mesh.nb_cells(mesh_id_t::cells)));
    }

    TEST(load_balancing_weight, per_level)
    {
        auto mesh = make_two_level_mesh();
        auto w    = lb::weight::per_level(
            [](std::size_t l)
            {
                return std::pow(2.0, static_cast<double>(l - level));
            });

        samurai::for_each_cell(mesh[mesh_id_t::cells],
                               [&](const auto& cell)
                               {
                                   EXPECT_EQ(w(cell), cell.level == level ? 1.0 : 2.0);
                               });

        // analytic load: each half of [0,1]^2 holds half of the 2^(2l) cells
        // of its level; fine cells (level+1) weigh 2.
        const double n_fine   = (1 << (level + 1)) * (1 << (level + 1)) / 2.0;
        const double n_coarse = (1 << level) * (1 << level) / 2.0;
        EXPECT_EQ(lb::local_load(mesh, w), 2.0 * n_fine + n_coarse);
    }

    TEST(load_balancing_weight, from_field)
    {
        auto mesh = make_uniform_mesh();
        auto cost = samurai::make_scalar_field<double>("cost", mesh);
        samurai::for_each_cell(mesh[mesh_id_t::cells],
                               [&](const auto& cell)
                               {
                                   cost[cell] = static_cast<double>(cell.indices[0]);
                               });

        auto w = lb::weight::from_field(cost);
        samurai::for_each_cell(mesh[mesh_id_t::cells],
                               [&](const auto& cell)
                               {
                                   EXPECT_EQ(w(cell), cost[cell]);
                               });

        // sum over a 16x16 grid of the i index: 16 * (0 + 1 + ... + 15)
        const double n = static_cast<double>(1 << level);
        EXPECT_EQ(lb::local_load(mesh, w), n * (n - 1.) * n / 2.);
    }
}
