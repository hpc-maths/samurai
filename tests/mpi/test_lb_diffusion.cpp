// Copyright 2018-2025 the samurai's authors
// SPDX-License-Identifier:  BSD-3-Clause

// Diffusion partitioning strategy (roadmap step 5), nD by interface layers:
//  - the pure flux solver on analytic process graphs (2 procs, 1D chain);
//  - the layer assignment: balance quality, connectivity (no islands),
//    weighted load, 3D, and the unmet-flux reporting path.

#include <cmath>
#include <vector>

#include <gtest/gtest.h>

#include <samurai/box.hpp>
#include <samurai/field.hpp>
#include <samurai/load_balancing/load_balancer.hpp>
#include <samurai/load_balancing/strategies/diffusion.hpp>
#include <samurai/load_balancing/weight.hpp>
#include <samurai/mr/mesh.hpp>

#include "mpi_test_utils.hpp"

namespace lb  = samurai::load_balancing;
namespace mpi = boost::mpi;

namespace
{
    // ---------------------------------------------------------------------------
    // Phase 1: the pure Cybenko flux solver (detail::diffusion_fluxes).
    // Every rank must call it (lockstep on the collectives) even with no
    // neighbour, so the inactive ranks just pass an empty neighbour list.
    // ---------------------------------------------------------------------------

    // Two adjacent processes with loads 100 and 300: the flux converges to a
    // transfer of exactly 100 (the heavy one gives, the light one receives).
    TEST(LoadBalancingDiffusionFlux, two_procs_exact_transfer)
    {
        mpi::communicator world;
        ASSERT_GE(world.size(), 2);

        auto opts                 = lb::DiffusionOptions{};
        opts.diffusion_iterations = 200;

        std::vector<int> ranks;
        double my_load = 0.;
        if (world.rank() == 0)
        {
            ranks   = {1};
            my_load = 100.;
        }
        else if (world.rank() == 1)
        {
            ranks   = {0};
            my_load = 300.;
        }
        // ranks >= 2: empty neighbour list, zero load (they still join the
        // collectives inside diffusion_fluxes).

        const auto fluxes = lb::detail::diffusion_fluxes(my_load, ranks, opts);

        if (world.rank() == 0)
        {
            ASSERT_EQ(fluxes.size(), 1u);
            EXPECT_NEAR(fluxes[0], 100., 1e-6); // receives +100
        }
        else if (world.rank() == 1)
        {
            ASSERT_EQ(fluxes.size(), 1u);
            EXPECT_NEAR(fluxes[0], -100., 1e-6); // gives -100
        }
    }

    // A 1D chain of all the processes with the whole load (400) on the last
    // rank. Two exact (FP) conservation properties hold whatever the
    // convergence state, plus the signs at the chain ends.
    TEST(LoadBalancingDiffusionFlux, chain_conservation)
    {
        mpi::communicator world;
        const int size = world.size();
        ASSERT_GE(size, 2);

        auto opts                 = lb::DiffusionOptions{};
        opts.diffusion_iterations = 1000;

        std::vector<int> ranks;
        if (world.rank() > 0)
        {
            ranks.push_back(world.rank() - 1);
        }
        if (world.rank() < size - 1)
        {
            ranks.push_back(world.rank() + 1);
        }
        const double my_load = (world.rank() == size - 1) ? 400. : 0.;

        const auto fluxes = lb::detail::diffusion_fluxes(my_load, ranks, opts);

        double net = 0.;
        for (double f : fluxes)
        {
            net += f;
        }

        // 1. global load is conserved: the per-edge fluxes are exactly opposite,
        //    so the sum of every process' net flux is zero.
        const double global_net = mpi::all_reduce(world, net, std::plus<double>());
        EXPECT_NEAR(global_net, 0., 1e-9);

        // 2. the loaded end gives, the empty far end receives.
        if (world.rank() == size - 1)
        {
            EXPECT_LT(net, 0.);
        }
        if (world.rank() == 0)
        {
            EXPECT_GT(net, 0.);
        }
    }

    // ---------------------------------------------------------------------------
    // Phase 2: the layer assignment, through the full driver.
    // ---------------------------------------------------------------------------

    template <std::size_t d>
    struct Dim
    {
        static constexpr std::size_t dim = d;
    };

    template <class T>
    class LoadBalancingDiffusion : public samurai_test::MpiTest
    {
      public:

        static constexpr std::size_t dim = T::dim;
        using Mesh                       = samurai::MRMesh<samurai::mesh_config<dim>>;
        using mesh_id_t                  = typename Mesh::mesh_id_t;

        static constexpr std::size_t level = (dim == 2) ? 5 : 4;

        static samurai::Box<double, dim> box(double lo = 0., double hi = 1.)
        {
            xt::xtensor_fixed<double, xt::xshape<dim>> min_corner;
            xt::xtensor_fixed<double, xt::xshape<dim>> max_corner;
            min_corner.fill(lo);
            max_corner.fill(hi);
            return samurai::Box<double, dim>(min_corner, max_corner);
        }

        static Mesh make_uniform_mesh()
        {
            return samurai::mra::make_mesh(box(), samurai::mesh_config<dim>().min_level(level).max_level(level));
        }

        static Mesh make_corner_refined_mesh()
        {
            return samurai_test::make_locally_refined_mesh<Mesh>(box(),
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

        /// Off-centre refined disk/sphere on top of a uniform mesh: an
        /// irregular refined region like the one the advection demo produces
        /// (this is the geometry that broke the direction-based version).
        static Mesh make_disk_refined_mesh()
        {
            return samurai_test::make_locally_refined_mesh<Mesh>(box(),
                                                                 level,
                                                                 [](const auto& cell)
                                                                 {
                                                                     double d2 = 0.;
                                                                     for (std::size_t d = 0; d < dim; ++d)
                                                                     {
                                                                         const double c = cell.center(d) - 0.35;
                                                                         d2 += c * c;
                                                                     }
                                                                     return d2 < 0.2 * 0.2;
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

        /// Force an imbalanced but adjacent banded partition: rank 0 gets a wide
        /// band [0, 0.7) along x, the other ranks share the rest in equal,
        /// contiguous bands -> a chain of neighbours with a heavy first rank.
        template <class Field, class Weight>
        static void make_banded_imbalance(Mesh& mesh, Field& u, const Weight& weight)
        {
            mpi::communicator world;
            const int size = world.size();
            auto banded    = samurai_test::LambdaStrategy{[size](const auto& cell, int, int)
                                                       {
                                                           const double x  = cell.center(0);
                                                           const double w0 = 0.7;
                                                           if (x < w0 || size == 1)
                                                           {
                                                               return 0;
                                                           }
                                                           const double frac = (x - w0) / (1. - w0);
                                                           int r             = 1 + static_cast<int>(frac * static_cast<double>(size - 1));
                                                           return (r >= size) ? size - 1 : r;
                                                       }};
            lb::make_load_balancer<decltype(banded)>({}, banded).load_balance(weight, u);
        }
    };

    using Cases = ::testing::Types<Dim<2>, Dim<3>>;
    TYPED_TEST_SUITE(LoadBalancingDiffusion, Cases, );

    // From a heavy banded partition, a few diffusion calls bring the imbalance
    // below the threshold while conserving cells and field values.
    TYPED_TEST(LoadBalancingDiffusion, banded_balance)
    {
        using mesh_id_t = typename TestFixture::mesh_id_t;
        mpi::communicator world;

        auto mesh   = TestFixture::make_uniform_mesh();
        auto u      = TestFixture::make_analytic_field(mesh);
        auto weight = lb::weight::uniform();

        const auto cells_before = mesh[mesh_id_t::cells];
        const auto count_before = TestFixture::global_count(mesh);

        TestFixture::make_banded_imbalance(mesh, u, weight);
        EXPECT_TRUE_ALL_RANKS(samurai::load_balancing::imbalance(mesh, weight) > 0.1); // really imbalanced

        auto balancer      = lb::make_load_balancer<lb::Diffusion>();
        const double bound = (TestFixture::dim == 2) ? 0.10 : 0.15;
        for (int call = 0; call < 12 && samurai::load_balancing::imbalance(mesh, weight) > bound; ++call)
        {
            balancer.load_balance(weight, u);
        }

        // conservation + values followed their cells
        samurai_test::check_lb_invariants(mesh,
                                          cells_before,
                                          count_before,
                                          [&](const auto& cell)
                                          {
                                              return u[cell] == samurai_test::analytic(cell);
                                          });
        EXPECT_TRUE_ALL_RANKS(samurai::load_balancing::imbalance(mesh, weight) <= bound);
        EXPECT_TRUE_ALL_RANKS(mesh.nb_cells(mesh_id_t::cells) > 0);
    }

    // Diffusion cedes only cells connected to the interface, so after balancing
    // every rank still owns a single face-connected component (no islands).
    TYPED_TEST(LoadBalancingDiffusion, connectivity_no_islands)
    {
        mpi::communicator world;

        auto mesh   = TestFixture::make_uniform_mesh();
        auto u      = TestFixture::make_analytic_field(mesh);
        auto weight = lb::weight::uniform();

        TestFixture::make_banded_imbalance(mesh, u, weight);

        auto balancer      = lb::make_load_balancer<lb::Diffusion>();
        const double bound = (TestFixture::dim == 2) ? 0.10 : 0.15;
        for (int call = 0; call < 12 && samurai::load_balancing::imbalance(mesh, weight) > bound; ++call)
        {
            balancer.load_balance(weight, u);
        }

        EXPECT_TRUE_ALL_RANKS(samurai_test::local_connected_components(mesh) == 1);
    }

    // The demo scenario that broke the direction-based version: an irregular
    // refined disk on top of the *default* (row-major) decomposition. Starting
    // from connected bands, several diffusion calls must keep every rank a
    // single face-connected island, never scattering cells across the domain.
    TYPED_TEST(LoadBalancingDiffusion, connectivity_from_default_decomposition)
    {
        using mesh_id_t = typename TestFixture::mesh_id_t;
        mpi::communicator world;

        auto mesh   = TestFixture::make_disk_refined_mesh();
        auto u      = TestFixture::make_analytic_field(mesh);
        auto weight = lb::weight::uniform();

        const auto cells_before = mesh[mesh_id_t::cells];
        const auto count_before = TestFixture::global_count(mesh);

        // the default decomposition is already connected; check it stays so
        EXPECT_TRUE_ALL_RANKS(mesh.nb_cells(mesh_id_t::cells) == 0 || samurai_test::local_connected_components(mesh) == 1);

        auto balancer = lb::make_load_balancer<lb::Diffusion>();
        for (int call = 0; call < 8; ++call)
        {
            balancer.load_balance(weight, u);
            EXPECT_TRUE_ALL_RANKS(mesh.nb_cells(mesh_id_t::cells) == 0 || samurai_test::local_connected_components(mesh) == 1);
        }

        samurai_test::check_lb_invariants(mesh,
                                          cells_before,
                                          count_before,
                                          [&](const auto& cell)
                                          {
                                              return u[cell] == samurai_test::analytic(cell);
                                          });
    }

    // Per-level weight on a refined corner: diffusion balances the *work*, not
    // the cell count. Looser bound than SFC (diffusion is incremental).
    TYPED_TEST(LoadBalancingDiffusion, weighted_balance)
    {
        using mesh_id_t = typename TestFixture::mesh_id_t;
        mpi::communicator world;

        auto mesh = TestFixture::make_corner_refined_mesh();
        auto u    = TestFixture::make_analytic_field(mesh);
        auto w    = lb::weight::per_level(
            [](std::size_t l)
            {
                return std::pow(2.0, static_cast<double>(l) - static_cast<double>(TestFixture::level));
            });

        const auto cells_before = mesh[mesh_id_t::cells];
        const auto count_before = TestFixture::global_count(mesh);

        auto balancer = lb::make_load_balancer<lb::Diffusion>();
        for (int call = 0; call < 12 && samurai::load_balancing::imbalance(mesh, w) > 0.15; ++call)
        {
            balancer.load_balance(w, u);
        }

        samurai_test::check_lb_invariants(mesh,
                                          cells_before,
                                          count_before,
                                          [&](const auto& cell)
                                          {
                                              return u[cell] == samurai_test::analytic(cell);
                                          });
        EXPECT_TRUE_ALL_RANKS(samurai::load_balancing::imbalance(mesh, w) <= 0.15);
    }

    // The unmet-flux path is wired end to end (strategy -> stats) and the
    // strategy never deadlocks or corrupts the mesh, even from an extreme
    // banded imbalance. unmet_flux is a finite, non-negative quantity equal to
    // what the strategy reports.
    TYPED_TEST(LoadBalancingDiffusion, unmet_flux_reporting)
    {
        using mesh_id_t = typename TestFixture::mesh_id_t;
        mpi::communicator world;

        auto mesh   = TestFixture::make_uniform_mesh();
        auto u      = TestFixture::make_analytic_field(mesh);
        auto weight = lb::weight::uniform();

        const auto cells_before = mesh[mesh_id_t::cells];
        const auto count_before = TestFixture::global_count(mesh);

        TestFixture::make_banded_imbalance(mesh, u, weight);

        auto balancer = lb::make_load_balancer<lb::Diffusion>();
        auto stats    = balancer.load_balance_with_stats(weight, u);

        EXPECT_TRUE_ALL_RANKS(stats.unmet_flux >= 0.);
        EXPECT_TRUE_ALL_RANKS(stats.unmet_flux == balancer.strategy().last_unmet_flux());
        EXPECT_TRUE_ALL_RANKS(std::isfinite(stats.unmet_flux));

        // the migration stays correct regardless of any unmet flux
        samurai_test::check_lb_invariants(mesh,
                                          cells_before,
                                          count_before,
                                          [&](const auto& cell)
                                          {
                                              return u[cell] == samurai_test::analytic(cell);
                                          });
    }

    // ---------------------------------------------------------------------------
    // Regression: a rank with NO MPI neighbour must not deadlock the others.
    // ---------------------------------------------------------------------------

    using IsoMesh = samurai::MRMesh<samurai::mesh_config<2>>;

    /// Two detached boxes [0,1]x[0,1] and [2,3]x[0,1] separated by an empty gap
    /// [1,2]: a disconnected domain. A rank owning only one box has no neighbour.
    IsoMesh make_two_detached_boxes()
    {
        using mesh_id_t             = IsoMesh::mesh_id_t;
        constexpr std::size_t level = 4;

        xt::xtensor_fixed<double, xt::xshape<2>> lo = {0., 0.};
        xt::xtensor_fixed<double, xt::xshape<2>> hi = {3., 1.};
        const samurai::Box<double, 2> box(lo, hi);
        auto coarse = samurai::mra::make_mesh(box, samurai::mesh_config<2>().min_level(level).max_level(level));

        IsoMesh::cl_type cl;
        samurai::for_each_cell(coarse[mesh_id_t::cells],
                               [&](const auto& cell)
                               {
                                   const double x = cell.center(0);
                                   if (x > 1. && x < 2.)
                                   {
                                       return; // carve the gap -> two detached boxes
                                   }
                                   auto yz = xt::view(cell.indices, xt::range(1, cell.indices.size()));
                                   cl[level][yz].add_point(cell.indices[0]);
                               });
        return samurai::mra::make_mesh(cl, samurai::mesh_config<2>().min_level(level).max_level(level));
    }

    class DiffusionIsolatedRank : public samurai_test::MpiTest
    {
    };

    // The flux solver runs world-collective all_reduce (total load, per-iteration
    // convergence flag); every rank must join them, including one whose subdomain
    // has no MPI neighbour. The strategy used to return early on an empty
    // neighbourhood, skipping those collectives and deadlocking the ranks that
    // did have neighbours -- exactly what isolates a subdomain does (seen with
    // many ranks on a thin domain). Here the small box is owned alone by the last
    // rank (no neighbour) while the big box is shared by the others (adjacent):
    // without the fix this load_balance call never returns.
    TEST_F(DiffusionIsolatedRank, no_neighbour_does_not_deadlock)
    {
        using mesh_id_t = IsoMesh::mesh_id_t;
        mpi::communicator world;
        if (world.size() < 3)
        {
            GTEST_SKIP() << "need >= 3 ranks: with 2, both boxes are isolated (symmetric, no deadlock)";
        }

        auto mesh = make_two_detached_boxes();
        auto u    = samurai::make_scalar_field<double>("u", mesh);
        samurai::for_each_cell(mesh[mesh_id_t::cells],
                               [&](const auto& cell)
                               {
                                   u[cell] = samurai_test::analytic(cell);
                               });

        // big box [0,1] -> ranks 0..size-2 (contiguous x-bands, adjacent);
        // small box [2,3] -> rank size-1, alone and with no neighbour.
        auto decomp = samurai_test::LambdaStrategy{[](const auto& cell, int /*rank*/, int size)
                                                   {
                                                       const double x = cell.center(0);
                                                       if (x > 1.5)
                                                       {
                                                           return size - 1; // isolated rank
                                                       }
                                                       const int r = static_cast<int>(x * static_cast<double>(size - 1));
                                                       return (r > size - 2) ? size - 2 : r;
                                                   }};
        lb::make_load_balancer<decltype(decomp)>({}, decomp).load_balance(lb::weight::uniform(), u);

        const auto cells_before = mesh[mesh_id_t::cells];
        const auto count_before = mpi::all_reduce(world, mesh.nb_cells(mesh_id_t::cells), std::plus<std::size_t>());

        // The call that used to hang when a rank has no neighbour.
        auto balancer = lb::make_load_balancer<lb::Diffusion>();
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
