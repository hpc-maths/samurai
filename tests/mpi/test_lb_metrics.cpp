// Copyright 2018-2025 the samurai's authors
// SPDX-License-Identifier:  BSD-3-Clause

// Metrics and statistics of the load balancing module (roadmap step 2):
// global imbalance, collective require_balance decision, LoadBalanceStats
// consistency, weighted loads through load_balance, partition dump.

#include <cmath>
#include <filesystem>

#include <gtest/gtest.h>

#include <samurai/box.hpp>
#include <samurai/field.hpp>
#include <samurai/load_balancing/dump.hpp>
#include <samurai/load_balancing/load_balancer.hpp>
#include <samurai/load_balancing/weight.hpp>
#include <samurai/mr/mesh.hpp>

#include "mpi_test_utils.hpp"

namespace lb  = samurai::load_balancing;
namespace mpi = boost::mpi;
namespace fs  = std::filesystem;

namespace
{
    using samurai_test::LambdaStrategy;

    class load_balancing_metrics : public samurai_test::MpiTest
    {
    };

    constexpr std::size_t dim   = 2;
    constexpr std::size_t level = 5;
    using Mesh                  = samurai::MRMesh<samurai::mesh_config<dim>>;
    using mesh_id_t             = Mesh::mesh_id_t;

    Mesh make_uniform_mesh()
    {
        const samurai::Box<double, dim> box({0., 0.}, {1., 1.});
        return samurai::mra::make_mesh(box, samurai::mesh_config<dim>().min_level(level).max_level(level));
    }

    auto concentrate_on_rank0()
    {
        return lb::make_load_balancer(lb::LoadBalanceConfig{},
                                      LambdaStrategy{[](const auto&, int, int)
                                                     {
                                                         return 0;
                                                     }});
    }

    // After concentrating everything on rank 0, the imbalance is exactly
    // P - 1 (max = N, avg = N/P).
    TEST_F(load_balancing_metrics, imbalance_exact_value)
    {
        mpi::communicator world;
        auto mesh = make_uniform_mesh();
        auto u    = samurai::make_scalar_field<double>("u", mesh, 0.);

        auto balancer = concentrate_on_rank0();
        balancer.load_balance(lb::weight::uniform(), u);

        const double imb = lb::imbalance(mesh, lb::weight::uniform());
        EXPECT_TRUE_ALL_RANKS(imb == static_cast<double>(world.size() - 1));
    }

    // require_balance is a collective decision: every rank must get the same
    // boolean, before (balanced mesh) and after (fully concentrated mesh).
    TEST_F(load_balancing_metrics, require_balance_collective)
    {
        mpi::communicator world;
        auto mesh = make_uniform_mesh();
        auto u    = samurai::make_scalar_field<double>("u", mesh, 0.);

        auto check_same_everywhere = [&](bool local_decision)
        {
            std::vector<int> all;
            mpi::gather(world, static_cast<int>(local_decision), all, 0);
            bool same = true;
            if (world.rank() == 0)
            {
                for (int v : all)
                {
                    same = same && (v == all[0]);
                }
            }
            mpi::broadcast(world, same, 0);
            return same;
        };

        const bool before = lb::require_balance(mesh, lb::weight::uniform(), 0.05);
        EXPECT_TRUE_ALL_RANKS(check_same_everywhere(before));

        auto balancer = concentrate_on_rank0();
        balancer.load_balance(lb::weight::uniform(), u);

        const bool after = lb::require_balance(mesh, lb::weight::uniform(), 0.05);
        EXPECT_TRUE_ALL_RANKS(check_same_everywhere(after));
        EXPECT_TRUE_ALL_RANKS(after); // imbalance == P-1 >> 0.05

        // and the driver's required() must agree with the free function
        EXPECT_TRUE_ALL_RANKS(balancer.required(mesh, lb::weight::uniform()) == after);
    }

    // A circular rotation preserves the load distribution: stats must show
    // matching global in/out counts and an unchanged imbalance.
    TEST_F(load_balancing_metrics, stats_consistency_on_rotation)
    {
        mpi::communicator world;
        auto mesh = make_uniform_mesh();
        auto u    = samurai::make_scalar_field<double>("u", mesh, 0.);

        auto balancer = lb::make_load_balancer(lb::LoadBalanceConfig{},
                                               LambdaStrategy{[](const auto&, int rank, int size)
                                                              {
                                                                  return (rank + 1) % size;
                                                              }});
        auto stats    = balancer.load_balance_with_stats(lb::weight::uniform(), u);

        const auto total_out = mpi::all_reduce(world, stats.cells_migrated_out, std::plus<std::size_t>());
        const auto total_in  = mpi::all_reduce(world, stats.cells_migrated_in, std::plus<std::size_t>());
        EXPECT_TRUE_ALL_RANKS(total_out == total_in);
        EXPECT_TRUE_ALL_RANKS(total_out > 0);

        EXPECT_TRUE_ALL_RANKS(stats.imbalance_after == stats.imbalance_before);
        EXPECT_TRUE_ALL_RANKS(stats.strategy_name == "test-lambda");
        EXPECT_TRUE_ALL_RANKS(stats.partition_time >= 0. && stats.migration_time >= 0.);
        // the rotation moves everything: local loads swap between ranks
        EXPECT_TRUE_ALL_RANKS(stats.cells_migrated_out == stats.cells_before);
        EXPECT_TRUE_ALL_RANKS(stats.cells_migrated_in == stats.cells_after);
    }

    // Weighted loads flow into the stats: with a from_field weight, the load
    // concentrated on rank 0 equals the global weighted sum.
    TEST_F(load_balancing_metrics, weighted_load_in_stats)
    {
        mpi::communicator world;
        auto mesh = make_uniform_mesh();
        auto u    = samurai::make_scalar_field<double>("u", mesh, 0.);
        auto cost = samurai::make_scalar_field<double>("cost", mesh);
        samurai::for_each_cell(mesh[mesh_id_t::cells],
                               [&](const auto& cell)
                               {
                                   cost[cell] = 1. + static_cast<double>(cell.indices[1]);
                               });

        const double my_load     = lb::local_load(mesh, lb::weight::from_field(cost));
        const double global_load = mpi::all_reduce(world, my_load, std::plus<double>());

        auto balancer = concentrate_on_rank0();
        // note: cost must migrate together with u so that the weight follows the cells
        auto stats = balancer.load_balance_with_stats(lb::weight::from_field(cost), u, cost);

        EXPECT_TRUE_ALL_RANKS(stats.load_before == my_load);
        const bool load_ok = (world.rank() == 0) ? stats.load_after == global_load : stats.load_after == 0.;
        EXPECT_TRUE_ALL_RANKS(load_ok);
        EXPECT_TRUE_ALL_RANKS(stats.imbalance_after == static_cast<double>(world.size() - 1));
    }
}
