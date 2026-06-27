// Copyright 2018-2025 the samurai's authors
// SPDX-License-Identifier:  BSD-3-Clause

// SFC partitioning strategy (roadmap step 3), Morton and Hilbert, 2D and 3D:
// weighted balance quality, negative coordinates, idempotence, and the
// migration invariants on curve-shaped (non rectangular) partitions.

#include <array>
#include <cmath>
#include <cstdint>
#include <limits>
#include <utility>
#include <vector>

#include <boost/serialization/utility.hpp> // std::pair serialization for boost::mpi::gather
#include <gtest/gtest.h>

#include <samurai/box.hpp>
#include <samurai/field.hpp>
#include <samurai/load_balancing/load_balancer.hpp>
#include <samurai/load_balancing/strategies/sfc.hpp>
#include <samurai/load_balancing/weight.hpp>
#include <samurai/mr/mesh.hpp>

#include "mpi_test_utils.hpp"

namespace lb  = samurai::load_balancing;
namespace mpi = boost::mpi;

namespace
{
    template <std::size_t d, class C>
    struct Case
    {
        static constexpr std::size_t dim = d;
        using curve_t                    = C;
    };

    /// Two cells (level, indices) are face-adjacent if, normalized to a common
    /// fine level, their boxes overlap in every dimension but one and just
    /// touch in that one. Corner/edge-only contact does not count.
    template <std::size_t dim>
    bool face_adjacent(const std::array<long, dim + 1>& a, const std::array<long, dim + 1>& b, std::size_t max_level)
    {
        int touching = 0;
        for (std::size_t d = 0; d < dim; ++d)
        {
            const long sa = static_cast<long>(max_level) - a[0];
            const long sb = static_cast<long>(max_level) - b[0];
            const long a0 = a[d + 1] << sa, a1 = (a[d + 1] + 1) << sa;
            const long b0 = b[d + 1] << sb, b1 = (b[d + 1] + 1) << sb;
            const bool overlap = a0 < b1 && b0 < a1;
            if (overlap)
            {
                continue;
            }
            if (a1 == b0 || b1 == a0)
            {
                ++touching;
            }
            else
            {
                return false; // separated in this dimension
            }
        }
        return touching == 1;
    }

    /// Number of face-connected components of the cells held locally by this
    /// rank (flood fill on the pairwise adjacency graph).
    template <class Mesh>
    std::size_t local_connected_components(const Mesh& mesh)
    {
        using mesh_id_t           = typename Mesh::mesh_id_t;
        constexpr std::size_t dim = Mesh::dim;

        std::vector<std::array<long, dim + 1>> cells;
        samurai::for_each_cell(mesh[mesh_id_t::cells],
                               [&](const auto& cell)
                               {
                                   std::array<long, dim + 1> c{};
                                   c[0] = static_cast<long>(cell.level);
                                   for (std::size_t d = 0; d < dim; ++d)
                                   {
                                       c[d + 1] = static_cast<long>(cell.indices[d]);
                                   }
                                   cells.push_back(c);
                               });

        const std::size_t n = cells.size();
        std::vector<char> seen(n, 0);
        std::size_t components = 0;
        for (std::size_t s = 0; s < n; ++s)
        {
            if (seen[s])
            {
                continue;
            }
            ++components;
            std::vector<std::size_t> stack{s};
            seen[s] = 1;
            while (!stack.empty())
            {
                const std::size_t u = stack.back();
                stack.pop_back();
                for (std::size_t v = 0; v < n; ++v)
                {
                    if (!seen[v] && face_adjacent<dim>(cells[u], cells[v], mesh.max_level()))
                    {
                        seen[v] = 1;
                        stack.push_back(v);
                    }
                }
            }
        }
        return components;
    }

    /**
     * Universal SFC correctness invariant (valid for *both* curves, unlike
     * connectivity): a correct partition is a contiguous segment of the curve,
     * so when all cells are sorted by their key every rank owns exactly one
     * contiguous block of the global sequence. The pre-fix scan bug produced
     * interleaved blocks (a rank reappearing after another) for Morton and
     * Hilbert alike. Keys are recomputed with the very normalization used by
     * the strategy (same global shift, same `max_level`).
     */
    template <class Mesh, class Curve>
    bool partition_is_curve_contiguous(const Mesh& mesh, const Curve& curve)
    {
        using mesh_id_t = typename Mesh::mesh_id_t;
        using key_t     = samurai::load_balancing::sfc_key_t;
        mpi::communicator world;
        const std::size_t max_level = mesh.max_level();

        // Recompute the keys with the *exact* normalization the strategy used
        // (sfc_normalized_box + sfc_cell_key): same global shift, same
        // per-dimension extent, same curve. Sharing this code is what keeps the
        // validator and the strategy from drifting onto two different gilbert
        // curves (a contiguous arc on one is fractured on the other).
        const auto [global_min, extent] = samurai::load_balancing::sfc_normalized_box(mesh);

        std::vector<std::pair<key_t, int>> local;
        samurai::for_each_cell(mesh[mesh_id_t::cells],
                               [&](const auto& cell)
                               {
                                   local.emplace_back(samurai::load_balancing::sfc_cell_key(curve, cell, max_level, global_min, extent),
                                                      world.rank());
                               });

        std::vector<std::vector<std::pair<key_t, int>>> all;
        mpi::gather(world, local, all, 0);

        bool ok = true;
        if (world.rank() == 0)
        {
            std::vector<std::pair<key_t, int>> flat;
            for (const auto& v : all)
            {
                flat.insert(flat.end(), v.begin(), v.end());
            }
            std::sort(flat.begin(),
                      flat.end(),
                      [](const auto& a, const auto& b)
                      {
                          return a.first < b.first;
                      });
            std::vector<char> closed(static_cast<std::size_t>(world.size()), 0);
            int current = -1;
            for (const auto& [k, r] : flat)
            {
                if (r != current)
                {
                    if (closed[static_cast<std::size_t>(r)]) // rank reappears after another: interleaved
                    {
                        ok = false;
                        break;
                    }
                    if (current >= 0)
                    {
                        closed[static_cast<std::size_t>(current)] = 1;
                    }
                    current = r;
                }
            }
        }
        mpi::broadcast(world, ok, 0);
        return ok;
    }

    template <class T>
    class LoadBalancingSFC : public samurai_test::MpiTest
    {
      public:

        static constexpr std::size_t dim = T::dim;
        using curve_t                    = typename T::curve_t;
        using strategy_t                 = lb::SFC<curve_t>;
        using Mesh                       = samurai::MRMesh<samurai::mesh_config<dim>>;
        using mesh_id_t                  = typename Mesh::mesh_id_t;

        static constexpr std::size_t level = (dim == 2) ? 5 : 4;

        static samurai::Box<double, dim> box(double lo, double hi)
        {
            xt::xtensor_fixed<double, xt::xshape<dim>> min_corner;
            xt::xtensor_fixed<double, xt::xshape<dim>> max_corner;
            min_corner.fill(lo);
            max_corner.fill(hi);
            return samurai::Box<double, dim>(min_corner, max_corner);
        }

        static Mesh make_uniform_mesh(double lo = 0., double hi = 1.)
        {
            return samurai::mra::make_mesh(box(lo, hi), samurai::mesh_config<dim>().min_level(level).max_level(level));
        }

        /// Corner [0, 0.5)^dim refined one level deeper (graduated).
        static Mesh make_corner_refined_mesh()
        {
            return samurai_test::make_locally_refined_mesh<Mesh>(box(0., 1.),
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

        template <class M, class Field, class Weight>
        static void run_and_check(M& mesh, Field& u, const Weight& weight, double imbalance_bound)
        {
            const auto cells_before = mesh[mesh_id_t::cells];
            const auto count_before = global_count(mesh);

            auto balancer = lb::make_load_balancer<strategy_t>();
            auto stats    = balancer.load_balance_with_stats(weight, u);

            samurai_test::check_lb_invariants(mesh,
                                              cells_before,
                                              count_before,
                                              [&](const auto& cell)
                                              {
                                                  return u[cell] == samurai_test::analytic(cell);
                                              });
            EXPECT_TRUE_ALL_RANKS(stats.imbalance_after <= imbalance_bound);
            EXPECT_TRUE_ALL_RANKS(mesh.nb_cells(mesh_id_t::cells) > 0); // every rank keeps work
        }
    };

    using Cases = ::testing::Types<Case<2, lb::Morton>, Case<2, lb::Hilbert>, Case<3, lb::Morton>, Case<3, lb::Hilbert>>;
    TYPED_TEST_SUITE(LoadBalancingSFC, Cases, );

    // Uniform mesh + uniform weight: near-perfect balance in one call.
    TYPED_TEST(LoadBalancingSFC, uniform_balance)
    {
        auto mesh = TestFixture::make_uniform_mesh();
        auto u    = TestFixture::make_analytic_field(mesh);
        TestFixture::run_and_check(mesh, u, lb::weight::uniform(), 0.05);
    }

    // Refined corner + per-level weight: the strategy balances the weighted
    // load, not the cell count.
    TYPED_TEST(LoadBalancingSFC, weighted_balance)
    {
        auto mesh = TestFixture::make_corner_refined_mesh();
        auto u    = TestFixture::make_analytic_field(mesh);
        auto w    = lb::weight::per_level(
            [](std::size_t l)
            {
                return std::pow(2.0, static_cast<double>(l) - static_cast<double>(TestFixture::level));
            });
        TestFixture::run_and_check(mesh, u, w, 0.05);
    }

    // Box [-1,1]^dim: negative indices must be shifted, not asserted away.
    TYPED_TEST(LoadBalancingSFC, negative_indices)
    {
        auto mesh = TestFixture::make_uniform_mesh(-1., 1.);
        auto u    = TestFixture::make_analytic_field(mesh);

        const auto cells_before = mesh[TestFixture::mesh_id_t::cells];
        const auto count_before = TestFixture::global_count(mesh);

        auto balancer = lb::make_load_balancer<typename TestFixture::strategy_t>();
        auto stats    = balancer.load_balance_with_stats(lb::weight::uniform(), u);

        // note: analytic() requires non-negative indices, so check conservation
        // by count and global cell set only
        const auto count_after = TestFixture::global_count(mesh);
        EXPECT_TRUE_ALL_RANKS(count_after == count_before);
        EXPECT_TRUE_ALL_RANKS(samurai_test::same_global_cells(cells_before, mesh[TestFixture::mesh_id_t::cells]));
        EXPECT_TRUE_ALL_RANKS(stats.imbalance_after <= 0.05);
    }

    // The partition is a deterministic function of the global key sequence:
    // re-balancing immediately must migrate (almost) nothing.
    TYPED_TEST(LoadBalancingSFC, idempotence)
    {
        mpi::communicator world;
        auto mesh = TestFixture::make_corner_refined_mesh();
        auto u    = TestFixture::make_analytic_field(mesh);

        auto balancer = lb::make_load_balancer<typename TestFixture::strategy_t>();
        balancer.load_balance(lb::weight::uniform(), u);

        auto stats             = balancer.load_balance_with_stats(lb::weight::uniform(), u);
        const auto total_moved = mpi::all_reduce(world, stats.cells_migrated_out, std::plus<std::size_t>());
        const auto total_cells = TestFixture::global_count(mesh);
        EXPECT_TRUE_ALL_RANKS(total_moved <= total_cells / 100);
    }

    // Regression: the cut must use the *global* cumulative weight, not an MPI
    // scan over the rank index. The default decomposition (partition_mesh)
    // splits the mesh row-major, so the ranks' key chunks interleave on the
    // curve; a scan-based offset mis-assigns cells and fractures each rank into
    // several disconnected islands. For Hilbert -- a continuous curve -- a
    // correct partition gives each rank exactly one face-connected component.
    // (Morton can be legitimately disconnected, so the check is Hilbert-only.)
    TYPED_TEST(LoadBalancingSFC, hilbert_partition_is_connected)
    {
        if constexpr (std::is_same_v<typename TestFixture::curve_t, lb::Hilbert>)
        {
            auto mesh = TestFixture::make_corner_refined_mesh();
            auto u    = TestFixture::make_analytic_field(mesh);
            auto w    = lb::weight::per_level(
                [](std::size_t l)
                {
                    return std::pow(2.0, static_cast<double>(l) - static_cast<double>(TestFixture::level));
                });

            auto balancer = lb::make_load_balancer<typename TestFixture::strategy_t>();
            balancer.load_balance(w, u);

            EXPECT_TRUE_ALL_RANKS(local_connected_components(mesh) == 1);
        }
    }

    // The reported failure mode: a mesh living on only a couple of ranks must be
    // spread over all n ranks. We first collapse the mesh onto ranks {0, 1}
    // (left/right halves) -- a decomposition that occupies two ranks and
    // interleaves with the curve -- then let the SFC strategy redistribute it.
    // The result must be a contiguous segment of the curve on every rank (both
    // curves), connected for Hilbert, with every rank kept busy.
    TYPED_TEST(LoadBalancingSFC, redistribute_from_two_ranks)
    {
        using mesh_id_t = typename TestFixture::mesh_id_t;
        mpi::communicator world;

        auto mesh               = TestFixture::make_corner_refined_mesh();
        auto u                  = TestFixture::make_analytic_field(mesh);
        const auto cells_before = mesh[mesh_id_t::cells];
        const auto count_before = TestFixture::global_count(mesh);

        // 1. collapse onto ranks {0, 1}: left half -> 0, right half -> 1.
        auto squash = samurai_test::LambdaStrategy{[](const auto& cell, int, int)
                                                   {
                                                       return cell.center(0) < 0.5 ? 0 : 1;
                                                   }};
        lb::make_load_balancer<decltype(squash)>({}, squash).load_balance(lb::weight::uniform(), u);
        const auto occupied = mpi::all_reduce(world, (mesh.nb_cells(mesh_id_t::cells) > 0) ? 1 : 0, std::plus<int>());
        EXPECT_TRUE_ALL_RANKS(occupied <= 2); // the starting point really lives on <= 2 ranks

        // 2. SFC redistributes over all ranks.
        auto balancer = lb::make_load_balancer<typename TestFixture::strategy_t>();
        balancer.load_balance(lb::weight::uniform(), u);

        samurai_test::check_lb_invariants(mesh,
                                          cells_before,
                                          count_before,
                                          [&](const auto& cell)
                                          {
                                              return u[cell] == samurai_test::analytic(cell);
                                          });
        EXPECT_TRUE_ALL_RANKS(mesh.nb_cells(mesh_id_t::cells) > 0); // every rank gets work back
        EXPECT_TRUE_ALL_RANKS(partition_is_curve_contiguous(mesh, typename TestFixture::curve_t{}));
        if constexpr (std::is_same_v<typename TestFixture::curve_t, lb::Hilbert>)
        {
            EXPECT_TRUE_ALL_RANKS(local_connected_components(mesh) == 1);
        }
    }
}
