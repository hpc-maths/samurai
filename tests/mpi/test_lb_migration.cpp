// Copyright 2018-2025 the samurai's authors
// SPDX-License-Identifier:  BSD-3-Clause

// Tests of the load balancing migration infrastructure (roadmap step 1).
// The strategies here are synthetic flag generators: the point is to validate
// the fused cells+fields migration of the driver, not a real partitioner.

#include <gtest/gtest.h>

#include <samurai/box.hpp>
#include <samurai/field.hpp>
#include <samurai/load_balancing/load_balancer.hpp>
#include <samurai/load_balancing/strategies/void.hpp>
#include <samurai/load_balancing/weight.hpp>
#include <samurai/mr/mesh.hpp>

#include "mpi_test_utils.hpp"

namespace lb  = samurai::load_balancing;
namespace mpi = boost::mpi;

namespace
{
    using samurai_test::LambdaStrategy;

    template <std::size_t d>
    struct DimWrapper
    {
        static constexpr std::size_t dim = d;
    };

    template <class T>
    class LoadBalancingMigration : public samurai_test::MpiTest
    {
      public:

        static constexpr std::size_t dim = T::dim;
        using mesh_config_t              = samurai::mesh_config<dim>;
        using Mesh                       = samurai::MRMesh<mesh_config_t>;
        using mesh_id_t                  = typename Mesh::mesh_id_t;
        using ca_type                    = typename Mesh::ca_type;

        static constexpr std::size_t level = (dim == 2) ? 5 : 4;

        static samurai::Box<double, dim> unit_box()
        {
            xt::xtensor_fixed<double, xt::xshape<dim>> min_corner;
            xt::xtensor_fixed<double, xt::xshape<dim>> max_corner;
            min_corner.fill(0.);
            max_corner.fill(1.);
            return samurai::Box<double, dim>(min_corner, max_corner);
        }

        static Mesh make_uniform_mesh()
        {
            return samurai::mra::make_mesh(unit_box(), samurai::mesh_config<dim>().min_level(level).max_level(level));
        }

        /// Uniform mesh refined one level on the left half of the domain
        /// (graduated: level jump of 1 at x = 0.5).
        static Mesh make_two_level_mesh()
        {
            return samurai_test::make_locally_refined_mesh<Mesh>(unit_box(),
                                                                 level,
                                                                 [](const auto& cell)
                                                                 {
                                                                     return cell.center(0) < 0.5;
                                                                 });
        }

        template <class M>
        static auto make_analytic_field(const std::string& name, M& mesh)
        {
            auto u = samurai::make_scalar_field<double>(name, mesh);
            samurai::for_each_cell(mesh[mesh_id_t::cells],
                                   [&](const auto& cell)
                                   {
                                       u[cell] = samurai_test::analytic(cell);
                                   });
            return u;
        }

        static std::size_t global_count(const Mesh& mesh)
        {
            mpi::communicator world;
            return mpi::all_reduce(world, mesh.nb_cells(mesh_id_t::cells), std::plus<std::size_t>());
        }

        static bool same_local_cells(const ca_type& a, const ca_type& b)
        {
            if (a.nb_cells() != b.nb_cells())
            {
                return false;
            }
            if (a.nb_cells() == 0)
            {
                return true;
            }
            for (std::size_t l = std::min(a.min_level(), b.min_level()); l <= std::max(a.max_level(), b.max_level()); ++l)
            {
                if (!samurai::difference(a[l], b[l]).empty() || !samurai::difference(b[l], a[l]).empty())
                {
                    return false;
                }
            }
            return true;
        }
    };

    using Dims = ::testing::Types<DimWrapper<2>, DimWrapper<3>>;
    TYPED_TEST_SUITE(LoadBalancingMigration, Dims, );

    // Void strategy: nothing moves, mesh and fields are strictly untouched.
    TYPED_TEST(LoadBalancingMigration, identity)
    {
        auto mesh = TestFixture::make_uniform_mesh();
        auto u    = TestFixture::make_analytic_field("u", mesh);

        const auto cells_before = mesh[TestFixture::mesh_id_t::cells];
        const auto count_before = TestFixture::global_count(mesh);

        auto balancer = lb::make_load_balancer<lb::Void>();
        auto stats    = balancer.load_balance_with_stats(lb::weight::uniform(), u);

        EXPECT_TRUE_ALL_RANKS(stats.cells_migrated_out == 0);
        EXPECT_TRUE_ALL_RANKS(stats.cells_migrated_in == 0);
        EXPECT_TRUE_ALL_RANKS(stats.cells_before == stats.cells_after);
        EXPECT_TRUE_ALL_RANKS(TestFixture::same_local_cells(cells_before, mesh[TestFixture::mesh_id_t::cells]));
        samurai_test::check_lb_invariants(mesh,
                                          cells_before,
                                          count_before,
                                          [&](const auto& cell)
                                          {
                                              return u[cell] == samurai_test::analytic(cell);
                                          });
    }

    // Circular rotation: rank r gives everything to r+1. Exercises routing
    // towards a rank that is not necessarily a geometric neighbour.
    TYPED_TEST(LoadBalancingMigration, rotation)
    {
        mpi::communicator world;
        auto mesh = TestFixture::make_uniform_mesh();
        auto u    = TestFixture::make_analytic_field("u", mesh);

        const auto cells_before = mesh[TestFixture::mesh_id_t::cells];
        const auto count_before = TestFixture::global_count(mesh);

        auto balancer = lb::make_load_balancer(lb::LoadBalanceConfig{},
                                               LambdaStrategy{[](const auto&, int rank, int size)
                                                              {
                                                                  return (rank + 1) % size;
                                                              }});
        auto stats    = balancer.load_balance_with_stats(lb::weight::uniform(), u);

        samurai_test::check_lb_invariants(mesh,
                                          cells_before,
                                          count_before,
                                          [&](const auto& cell)
                                          {
                                              return u[cell] == samurai_test::analytic(cell);
                                          });

        // Each rank must now own exactly the cells its predecessor had.
        typename TestFixture::ca_type predecessor_cells;
        const int next = (world.rank() + 1) % world.size();
        const int prev = (world.rank() - 1 + world.size()) % world.size();
        auto req       = world.isend(next, 99, cells_before);
        world.recv(prev, 99, predecessor_cells);
        req.wait();
        EXPECT_TRUE_ALL_RANKS(TestFixture::same_local_cells(predecessor_cells, mesh[TestFixture::mesh_id_t::cells]));
        EXPECT_TRUE_ALL_RANKS(stats.cells_migrated_out == stats.cells_before);
    }

    // Everything to rank 0: validates that an emptied rank is a valid state.
    TYPED_TEST(LoadBalancingMigration, concentration)
    {
        mpi::communicator world;
        auto mesh = TestFixture::make_uniform_mesh();
        auto u    = TestFixture::make_analytic_field("u", mesh);

        const auto cells_before = mesh[TestFixture::mesh_id_t::cells];
        const auto count_before = TestFixture::global_count(mesh);

        auto balancer = lb::make_load_balancer(lb::LoadBalanceConfig{},
                                               LambdaStrategy{[](const auto&, int, int)
                                                              {
                                                                  return 0;
                                                              }});
        balancer.load_balance(lb::weight::uniform(), u);

        samurai_test::check_lb_invariants(mesh,
                                          cells_before,
                                          count_before,
                                          [&](const auto& cell)
                                          {
                                              return u[cell] == samurai_test::analytic(cell);
                                          });
        const std::size_t local = mesh.nb_cells(TestFixture::mesh_id_t::cells);
        EXPECT_TRUE_ALL_RANKS(world.rank() == 0 ? local == count_before : local == 0);
    }

    // Checkerboard on cell parity: a deliberately fragmented, non contiguous
    // partition touching every rank. Robustness test of arbitrary routing.
    TYPED_TEST(LoadBalancingMigration, checkerboard)
    {
        auto mesh = TestFixture::make_uniform_mesh();
        auto u    = TestFixture::make_analytic_field("u", mesh);

        const auto cells_before = mesh[TestFixture::mesh_id_t::cells];
        const auto count_before = TestFixture::global_count(mesh);

        auto balancer = lb::make_load_balancer(lb::LoadBalanceConfig{},
                                               LambdaStrategy{[](const auto& cell, int, int size)
                                                              {
                                                                  long long sum = 0;
                                                                  for (std::size_t d = 0; d < TestFixture::dim; ++d)
                                                                  {
                                                                      sum += cell.indices[d];
                                                                  }
                                                                  return static_cast<int>(sum % size);
                                                              }});
        balancer.load_balance(lb::weight::uniform(), u);

        samurai_test::check_lb_invariants(mesh,
                                          cells_before,
                                          count_before,
                                          [&](const auto& cell)
                                          {
                                              return u[cell] == samurai_test::analytic(cell);
                                          });
    }

    // Simultaneous migration of a scalar double field, a 2-component vector
    // field and an int field: all three must be conserved by the same pass.
    TYPED_TEST(LoadBalancingMigration, multi_fields)
    {
        auto mesh = TestFixture::make_uniform_mesh();

        auto u = TestFixture::make_analytic_field("u", mesh);
        auto v = samurai::make_vector_field<double, 2>("v", mesh);
        auto w = samurai::make_scalar_field<int>("w", mesh);

        // int encoding: injective for level <= 11 and indices < 64
        auto int_analytic = [](const auto& cell)
        {
            int value = static_cast<int>(cell.level);
            for (std::size_t d = 0; d < TestFixture::dim; ++d)
            {
                value = value * 64 + static_cast<int>(cell.indices[d]);
            }
            return value;
        };

        samurai::for_each_cell(mesh[TestFixture::mesh_id_t::cells],
                               [&](const auto& cell)
                               {
                                   v[cell][0] = samurai_test::analytic(cell) * 8.;
                                   v[cell][1] = samurai_test::analytic(cell) * 8. + 1.;
                                   w[cell]    = int_analytic(cell);
                               });

        const auto cells_before = mesh[TestFixture::mesh_id_t::cells];
        const auto count_before = TestFixture::global_count(mesh);

        auto balancer = lb::make_load_balancer(lb::LoadBalanceConfig{},
                                               LambdaStrategy{[](const auto&, int rank, int size)
                                                              {
                                                                  return (rank + 1) % size;
                                                              }});
        balancer.load_balance(lb::weight::uniform(), u, v, w);

        samurai_test::check_lb_invariants(mesh,
                                          cells_before,
                                          count_before,
                                          [&](const auto& cell)
                                          {
                                              return u[cell] == samurai_test::analytic(cell)
                                                  && v[cell][0] == samurai_test::analytic(cell) * 8.
                                                  && v[cell][1] == samurai_test::analytic(cell) * 8. + 1. && w[cell] == int_analytic(cell);
                                          });
    }

    // Cells of different levels migrating in the same pass (left half of the
    // domain refined one level deeper, graduated).
    TYPED_TEST(LoadBalancingMigration, two_levels)
    {
        auto mesh = TestFixture::make_two_level_mesh();
        auto u    = TestFixture::make_analytic_field("u", mesh);

        const auto cells_before = mesh[TestFixture::mesh_id_t::cells];
        const auto count_before = TestFixture::global_count(mesh);

        auto balancer = lb::make_load_balancer(lb::LoadBalanceConfig{},
                                               LambdaStrategy{[](const auto&, int rank, int size)
                                                              {
                                                                  return (rank + 1) % size;
                                                              }});
        balancer.load_balance(lb::weight::uniform(), u);

        samurai_test::check_lb_invariants(mesh,
                                          cells_before,
                                          count_before,
                                          [&](const auto& cell)
                                          {
                                              return u[cell] == samurai_test::analytic(cell);
                                          });
    }
}
