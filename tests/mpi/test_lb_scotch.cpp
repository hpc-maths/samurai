// Copyright 2018-2025 the samurai's authors
// SPDX-License-Identifier:  BSD-3-Clause

// PT-Scotch partitioning strategy tests (roadmap step 4).
// Only compiled when SAMURAI_WITH_PTSCOTCH=ON.

#ifdef SAMURAI_WITH_PTSCOTCH

#include <gtest/gtest.h>

#include <samurai/box.hpp>
#include <samurai/field.hpp>
#include <samurai/load_balancing/load_balancer.hpp>
#include <samurai/load_balancing/metrics.hpp>
#include <samurai/load_balancing/strategies/scotch.hpp>
#include <samurai/load_balancing/weight.hpp>
#include <samurai/mr/mesh.hpp>

#include "mpi_test_utils.hpp"

namespace lb  = samurai::load_balancing;
namespace mpi = boost::mpi;

namespace
{
    template <std::size_t d>
    struct Dim
    {
        static constexpr std::size_t dim = d;
    };

    template <class T>
    class LoadBalancingScotch : public samurai_test::MpiTest
    {
      public:

        static constexpr std::size_t dim = T::dim;
        using mesh_config_t              = samurai::mesh_config<dim>;
        using Mesh                       = samurai::MRMesh<mesh_config_t>;
        using mesh_id_t                  = typename Mesh::mesh_id_t;

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

        static Mesh make_corner_refined_mesh()
        {
            return samurai_test::make_locally_refined_mesh<Mesh>(unit_box(),
                                                                 level,
                                                                 [](const auto& cell)
                                                                 {
                                                                     bool in_corner = true;
                                                                     for (std::size_t d = 0; d < dim; ++d)
                                                                     {
                                                                         in_corner = in_corner && (cell.center(d) < 0.5);
                                                                     }
                                                                     return in_corner;
                                                                 });
        }

        template <class M>
        static auto make_analytic_field(M& mesh)
        {
            auto u = samurai::make_scalar_field<double>("u", mesh);
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
    };

    using Dims = ::testing::Types<Dim<2>, Dim<3>>;
    TYPED_TEST_SUITE(LoadBalancingScotch, Dims, );

    // Uniform mesh + uniform weight: PT-Scotch must achieve near-perfect balance.
    TYPED_TEST(LoadBalancingScotch, uniform_balance)
    {
        auto mesh = TestFixture::make_uniform_mesh();
        auto u    = TestFixture::make_analytic_field(mesh);

        const auto cells_before = mesh[TestFixture::mesh_id_t::cells];
        const auto count_before = TestFixture::global_count(mesh);

        auto balancer = lb::make_load_balancer<lb::Scotch>();
        auto stats    = balancer.load_balance_with_stats(lb::weight::uniform(), u);

        samurai_test::check_lb_invariants(mesh,
                                          cells_before,
                                          count_before,
                                          [&](const auto& cell)
                                          {
                                              return u[cell] == samurai_test::analytic(cell);
                                          });
        EXPECT_TRUE_ALL_RANKS(stats.imbalance_after < 0.10);
    }

    // Refined corner + per-level weight: Scotch balances the weighted load.
    TYPED_TEST(LoadBalancingScotch, weighted_balance)
    {
        auto mesh = TestFixture::make_corner_refined_mesh();
        auto u    = TestFixture::make_analytic_field(mesh);
        auto w    = lb::weight::per_level(
            [](std::size_t l)
            {
                return std::pow(2.0, static_cast<double>(l) - static_cast<double>(TestFixture::level));
            });

        const auto cells_before = mesh[TestFixture::mesh_id_t::cells];
        const auto count_before = TestFixture::global_count(mesh);

        auto balancer = lb::make_load_balancer<lb::Scotch>();
        auto stats    = balancer.load_balance_with_stats(w, u);

        samurai_test::check_lb_invariants(mesh,
                                          cells_before,
                                          count_before,
                                          [&](const auto& cell)
                                          {
                                              return u[cell] == samurai_test::analytic(cell);
                                          });
        EXPECT_TRUE_ALL_RANKS(stats.imbalance_after < 0.25); // looser for 3D and small meshes
    }

    // Non-triviality: partitions should not be simple bands.
    TYPED_TEST(LoadBalancingScotch, non_band_partitions)
    {
        mpi::communicator world;
        if (world.size() < 2)
        {
            GTEST_SKIP() << "Need at least 2 processes";
        }

        auto mesh = TestFixture::make_corner_refined_mesh();
        auto u    = TestFixture::make_analytic_field(mesh);

        auto balancer = lb::make_load_balancer<lb::Scotch>();
        balancer.load_balance(lb::weight::uniform(), u);

        EXPECT_TRUE_ALL_RANKS(mesh.nb_cells(TestFixture::mesh_id_t::cells) > 0);

        std::set<double> x_coords;
        samurai::for_each_cell(mesh[TestFixture::mesh_id_t::cells],
                               [&](const auto& cell)
                               {
                                   x_coords.insert(cell.center(0));
                               });
        EXPECT_TRUE_ALL_RANKS(x_coords.size() > 1u);
    }
}

#endif // SAMURAI_WITH_PTSCOTCH
