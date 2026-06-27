// Copyright 2018-2025 the samurai's authors
// SPDX-License-Identifier:  BSD-3-Clause

// SFC partitioning on a NON-square domain: a thin tube [0,1] x [0,10] whose
// mesh discretizes several circles placed at different heights. The SFC
// strategy (strategies/sfc.hpp) normalizes the cell coordinates per dimension
// (index << shift - global_min[d]) and feeds them as-is to Morton/Hilbert,
// which interleave the bits assuming an *isotropic* index space. On a 1:10
// aspect ratio the index extent is 2^L x 10.2^L, so the curve locality
// degrades and a Hilbert partition can fracture into several spatial islands.
//
// Cell conservation must hold regardless of the domain shape (true
// correctness); the connectivity / imbalance checks are the discriminants that
// expose the suspected weakness on rectangular domains.

#include <array>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <limits>
#include <string>
#include <utility>
#include <vector>

#include <boost/serialization/utility.hpp> // std::pair serialization for boost::mpi::gather
#include <gtest/gtest.h>

#include <samurai/box.hpp>
#include <samurai/field.hpp>
#include <samurai/io/hdf5.hpp>
#include <samurai/load_balancing/load_balancer.hpp>
#include <samurai/load_balancing/strategies/diffusion.hpp>
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

    /// A circle of the tube: center (x, y) and radius r, in physical coordinates.
    struct Circle
    {
        double x;
        double y;
        double r;
    };

    /// Several circles scattered along the long (y) axis of the [0,1] x [0,10]
    /// tube. Their interfaces are what the mesh resolves; the scattered
    /// refinement is what stresses the SFC on the 1:10 aspect ratio.
    inline const std::vector<Circle>& tube_circles()
    {
        static const std::vector<Circle> circles = {
            {0.5, 1.0, 0.20},
            {0.3, 3.5, 0.25},
            {0.7, 6.0, 0.20},
            {0.5, 8.5, 0.25},
        };
        return circles;
    }

    /// True when (x, y) is within a thin band of any circle's interface: this
    /// marks the cells that resolve (discretize) the circles.
    inline bool near_circle(double x, double y)
    {
        constexpr double band = 0.07;
        for (const auto& circle : tube_circles())
        {
            const double dx   = x - circle.x;
            const double dy   = y - circle.y;
            const double dist = std::sqrt(dx * dx + dy * dy);
            if (std::abs(dist - circle.r) < band)
            {
                return true;
            }
        }
        return false;
    }

    /// Tube meshes shared by every strategy exercised here (SFC and Diffusion):
    /// the geometry is the same, only the partitioner changes.
    namespace tube
    {
        using Mesh                  = samurai::MRMesh<samurai::mesh_config<2>>;
        constexpr std::size_t level = 5;

        /// The non-square tube: 1 unit wide, 10 units long.
        inline samurai::Box<double, 2> box()
        {
            xt::xtensor_fixed<double, xt::xshape<2>> min_corner = {0., 0.};
            xt::xtensor_fixed<double, xt::xshape<2>> max_corner = {1., 10.};
            return samurai::Box<double, 2>(min_corner, max_corner);
        }

        /// Tube refined one level deeper around every circle interface.
        inline Mesh refined()
        {
            return samurai_test::make_locally_refined_mesh<Mesh>(box(),
                                                                 level,
                                                                 [](const auto& cell)
                                                                 {
                                                                     return near_circle(cell.center(0), cell.center(1));
                                                                 });
        }

        /// Same tube refined around the circles, but with a central rectangular
        /// void removed (x in [0.35, 0.65], y in [3, 7]). Matter remains on both
        /// sides, so the *domain* stays connected while the occupied region
        /// becomes non-convex. Built like samurai_test::make_locally_refined_mesh,
        /// plus a `keep` test dropping the hole cells (each rank drops its own; no
        /// duplication).
        inline Mesh holed()
        {
            using mesh_id_t = Mesh::mesh_id_t;
            using value_t   = Mesh::interval_t::value_t;

            auto coarse = samurai::mra::make_mesh(box(), samurai::mesh_config<2>().min_level(level).max_level(level));

            const auto in_hole = [](double x, double y)
            {
                return x > 0.35 && x < 0.65 && y > 3. && y < 7.;
            };

            Mesh::cl_type cl;
            samurai::for_each_cell(coarse[mesh_id_t::cells],
                                   [&](const auto& cell)
                                   {
                                       const double cx = cell.center(0);
                                       const double cy = cell.center(1);
                                       if (in_hole(cx, cy))
                                       {
                                           return; // carve the void
                                       }
                                       auto yz = xt::view(cell.indices, xt::range(1, cell.indices.size()));
                                       if (!near_circle(cx, cy))
                                       {
                                           cl[level][yz].add_point(cell.indices[0]);
                                           return;
                                       }
                                       const auto i = cell.indices[0];
                                       xt::xtensor_fixed<value_t, xt::xshape<1>> yz_child;
                                       for (unsigned m = 0; m < 2U; ++m)
                                       {
                                           yz_child(0) = 2 * yz(0) + static_cast<value_t>(m & 1U);
                                           cl[level + 1][yz_child].add_interval({2 * i, 2 * i + 2});
                                       }
                                   });
            return samurai::mra::make_mesh(cl, samurai::mesh_config<2>().min_level(level).max_level(level + 1));
        }

        /// Field encoding (level, i, j) injectively, to check values follow cells.
        inline auto analytic_field(Mesh& mesh)
        {
            using mesh_id_t = Mesh::mesh_id_t;
            auto u          = samurai::make_scalar_field<double>("u", mesh);
            samurai::for_each_cell(mesh[mesh_id_t::cells],
                                   [&](const auto& cell)
                                   {
                                       u[cell] = samurai_test::analytic(cell);
                                   });
            return u;
        }
    } // namespace tube

    /**
     * Universal SFC correctness invariant (valid for *both* curves, unlike
     * connectivity): a correct partition is a contiguous segment of the curve,
     * so when all cells are sorted by their key every rank owns exactly one
     * contiguous block of the global sequence. Keys are recomputed with the
     * very normalization used by the strategy (same global shift, same
     * `max_level`). Copied from test_lb_sfc.cpp (not exposed in the header).
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
    class LoadBalancingSFCTube : public samurai_test::MpiTest
    {
      public:

        static constexpr std::size_t dim = T::dim; // 2 (a tube is 2D here)
        using curve_t                    = typename T::curve_t;
        using strategy_t                 = lb::SFC<curve_t>;
        using Mesh                       = samurai::MRMesh<samurai::mesh_config<dim>>;
        using mesh_id_t                  = typename Mesh::mesh_id_t;

        static constexpr std::size_t level = tube::level;

        /// Tube refined around the circle interfaces (shared geometry).
        static Mesh make_tube_mesh()
        {
            return tube::refined();
        }

        /// Tube with a central void (non-convex domain): the gilbert curve fills
        /// the bounding box and traverses the void, so a partition segment
        /// straddling the hole can split into spatial islands -- a fundamental
        /// SFC limitation that gilbert does NOT remove (it only fixes the aspect
        /// ratio).
        static Mesh make_holed_tube_mesh()
        {
            return tube::holed();
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

        /// Optional visualization: when SAMURAI_TUBE_DUMP is set, write an
        /// HDF5/XDMF file holding the mesh, the owner rank and the analytic
        /// field, so the partition can be inspected in ParaView. No-op (and no
        /// file) during a normal ctest run.
        template <class M, class Field>
        static void dump(M& mesh, const Field& u, const std::string& tag)
        {
            if (std::getenv("SAMURAI_TUBE_DUMP") == nullptr)
            {
                return;
            }
            mpi::communicator world;
            auto rank = samurai::make_scalar_field<int>("rank", mesh);
            samurai::for_each_cell(mesh[mesh_id_t::cells],
                                   [&](const auto& cell)
                                   {
                                       rank[cell] = world.rank();
                                   });
            const std::string name = "tube_" + strategy_t{}.name() + "_np" + std::to_string(world.size()) + "_" + tag;
            samurai::save(name, MPI_COMM_WORLD, mesh, rank, u);
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
            EXPECT_TRUE_ALL_RANKS(partition_is_curve_contiguous(mesh, curve_t{}));
        }
    };

    using Cases = ::testing::Types<Case<2, lb::Morton>, Case<2, lb::Hilbert>>;
    TYPED_TEST_SUITE(LoadBalancingSFCTube, Cases, );

    // Tube + circles, uniform weight: cells conserved, values follow their
    // cells, the partition is a contiguous segment of the curve and the load is
    // balanced. None of these should depend on the (non-square) domain shape.
    TYPED_TEST(LoadBalancingSFCTube, balance_and_invariants)
    {
        auto mesh = TestFixture::make_tube_mesh();
        auto u    = TestFixture::make_analytic_field(mesh);
        TestFixture::dump(mesh, u, "init"); // initial (row-major) decomposition
        TestFixture::run_and_check(mesh, u, lb::weight::uniform(), 0.05);
        TestFixture::dump(mesh, u, "balanced"); // SFC partition
    }

    // Holed (non-convex) domain: the limit of SFC partitioning. The gilbert
    // curve still fills the bounding box and traverses the central void, so a
    // rank's curve segment can straddle the hole and break into spatial islands.
    // We therefore assert only what SFC *does* guarantee regardless of the
    // domain shape -- cells conserved, values follow their cells, partition
    // contiguous on the curve, load balanced, every rank busy (all checked by
    // run_and_check). Spatial connectivity is intentionally NOT required here:
    // that guarantee needs a graph/diffusion strategy, not an SFC.
    TYPED_TEST(LoadBalancingSFCTube, holed_domain_stays_correct)
    {
        auto mesh = TestFixture::make_holed_tube_mesh();
        auto u    = TestFixture::make_analytic_field(mesh);
        TestFixture::dump(mesh, u, "holed_init");
        TestFixture::run_and_check(mesh, u, lb::weight::uniform(), 0.05);
        TestFixture::dump(mesh, u, "holed_balanced");
        // Observed (measured during development): at np=3 the rank owning the
        // cells around the void splits into 2 face-connected components, for
        // both Morton and Hilbert -- the documented fracture. We do not assert
        // it (it is np-dependent: connected at np=2/4 here) because connectivity
        // is not a property an SFC can promise on a non-convex domain.
    }

    // Same tube with a per-level weight (fine cells around the circles weigh
    // more): the strategy must balance the weighted load, not the cell count.
    TYPED_TEST(LoadBalancingSFCTube, weighted_balance)
    {
        auto mesh = TestFixture::make_tube_mesh();
        auto u    = TestFixture::make_analytic_field(mesh);
        auto w    = lb::weight::per_level(
            [](std::size_t l)
            {
                return std::pow(2.0, static_cast<double>(l) - static_cast<double>(TestFixture::level));
            });
        TestFixture::run_and_check(mesh, u, w, 0.05);
    }

    // The discriminant: on the anisotropic tube, a Hilbert partition whose
    // coordinates are not scaled by the domain extent fractures into several
    // spatial islands. A correct continuous-curve partition keeps each rank's
    // cells in a single face-connected component. (Morton can be legitimately
    // disconnected, so this check is Hilbert-only.)
    TYPED_TEST(LoadBalancingSFCTube, hilbert_partition_is_connected)
    {
        if constexpr (std::is_same_v<typename TestFixture::curve_t, lb::Hilbert>)
        {
            auto mesh = TestFixture::make_tube_mesh();
            auto u    = TestFixture::make_analytic_field(mesh);

            auto balancer = lb::make_load_balancer<typename TestFixture::strategy_t>();
            balancer.load_balance(lb::weight::uniform(), u);

            EXPECT_TRUE_ALL_RANKS(samurai_test::local_connected_components(mesh) == 1);
        }
    }

    // Starting from a mesh collapsed onto ranks {0, 1} (bottom/top halves of the
    // tube), the SFC strategy must spread it over all ranks while preserving the
    // migration invariants, the curve contiguity (both curves) and Hilbert
    // connectivity.
    TYPED_TEST(LoadBalancingSFCTube, redistribute_from_two_ranks)
    {
        using mesh_id_t = typename TestFixture::mesh_id_t;
        mpi::communicator world;

        auto mesh               = TestFixture::make_tube_mesh();
        auto u                  = TestFixture::make_analytic_field(mesh);
        const auto cells_before = mesh[mesh_id_t::cells];
        const auto count_before = TestFixture::global_count(mesh);

        // 1. collapse onto ranks {0, 1}: bottom half (y < 5) -> 0, top -> 1.
        auto squash = samurai_test::LambdaStrategy{[](const auto& cell, int, int)
                                                   {
                                                       return cell.center(1) < 5. ? 0 : 1;
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
            EXPECT_TRUE_ALL_RANKS(samurai_test::local_connected_components(mesh) == 1);
        }
    }

    // ---------------------------------------------------------------------------
    // Diffusion on the same tube geometry. Unlike the SFC (which cuts a global
    // curve), diffusion only moves cells across rank interfaces, so on the convex
    // tube it keeps every rank a single connected component -- the property SFC
    // cannot guarantee on a thin or holed domain. On the non-convex (holed) tube,
    // diffusion stays correct but, like SFC, cannot promise connectivity.
    // ---------------------------------------------------------------------------

    class DiffusionTube : public samurai_test::MpiTest
    {
      public:

        using Mesh      = tube::Mesh;
        using mesh_id_t = Mesh::mesh_id_t;

        static std::size_t global_count(const Mesh& mesh)
        {
            mpi::communicator world;
            return mpi::all_reduce(world, mesh.nb_cells(mesh_id_t::cells), std::plus<std::size_t>());
        }

        /// Heavy bottom slab on rank 0, the rest shared by the other ranks in
        /// contiguous y-slabs: a connected chain of neighbours along the tube.
        template <class Field, class Weight>
        static void make_banded_imbalance(Field& u, const Weight& weight)
        {
            mpi::communicator world;
            const int size = world.size();
            auto banded    = samurai_test::LambdaStrategy{[size](const auto& cell, int, int)
                                                       {
                                                           const double y  = cell.center(1);
                                                           const double y0 = 7.0; // heavy slab [0, 7) -> rank 0
                                                           if (y < y0 || size == 1)
                                                           {
                                                               return 0;
                                                           }
                                                           const double frac = (y - y0) / (10.0 - y0);
                                                           const int r       = 1 + static_cast<int>(frac * static_cast<double>(size - 1));
                                                           return (r >= size) ? size - 1 : r;
                                                       }};
            lb::make_load_balancer<decltype(banded)>({}, banded).load_balance(weight, u);
        }

        /// A few diffusion calls until the load imbalance falls under `bound`
        /// (capped, since diffusion is incremental).
        template <class Field, class Weight>
        static void diffuse(Mesh& mesh, Field& u, const Weight& weight, double bound)
        {
            auto balancer = lb::make_load_balancer<lb::Diffusion>();
            for (int call = 0; call < 16 && samurai::load_balancing::imbalance(mesh, weight) > bound; ++call)
            {
                balancer.load_balance(weight, u);
            }
        }
    };

    // Convex tube: from a heavy banded slab, diffusion balances the load while
    // conserving cells and field values.
    TEST_F(DiffusionTube, balance)
    {
        auto mesh   = tube::refined();
        auto u      = tube::analytic_field(mesh);
        auto weight = lb::weight::uniform();

        const auto cells_before = mesh[mesh_id_t::cells];
        const auto count_before = global_count(mesh);

        make_banded_imbalance(u, weight);
        EXPECT_TRUE_ALL_RANKS(samurai::load_balancing::imbalance(mesh, weight) > 0.1); // really imbalanced

        const double bound = 0.12;
        diffuse(mesh, u, weight, bound);

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

    // Convex tube: diffusion cedes only interface cells, so every rank keeps a
    // single face-connected component -- where the SFC, cutting a space-filling
    // curve through a thin domain, can fracture (cf. hilbert_partition_is_connected).
    TEST_F(DiffusionTube, connectivity)
    {
        auto mesh   = tube::refined();
        auto u      = tube::analytic_field(mesh);
        auto weight = lb::weight::uniform();

        make_banded_imbalance(u, weight);
        diffuse(mesh, u, weight, 0.12);

        EXPECT_TRUE_ALL_RANKS(mesh.nb_cells(mesh_id_t::cells) == 0 || samurai_test::local_connected_components(mesh) == 1);
    }

    // Holed (non-convex) tube: like SFC, diffusion stays *correct* (cells
    // conserved, values follow, load balanced) but cannot guarantee connectivity
    // -- a rank around the void necessarily owns cells on both sides of it. We
    // assert only the shape-independent invariants.
    TEST_F(DiffusionTube, holed_stays_correct)
    {
        auto mesh   = tube::holed();
        auto u      = tube::analytic_field(mesh);
        auto weight = lb::weight::uniform();

        const auto cells_before = mesh[mesh_id_t::cells];
        const auto count_before = global_count(mesh);

        make_banded_imbalance(u, weight);
        diffuse(mesh, u, weight, 0.12);

        samurai_test::check_lb_invariants(mesh,
                                          cells_before,
                                          count_before,
                                          [&](const auto& cell)
                                          {
                                              return u[cell] == samurai_test::analytic(cell);
                                          });
        EXPECT_TRUE_ALL_RANKS(mesh.nb_cells(mesh_id_t::cells) > 0);
    }
}
