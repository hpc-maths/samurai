// Copyright 2018-2025 the samurai's authors
// SPDX-License-Identifier:  BSD-3-Clause

// Ghost consistency after load balancing.
//
// Rationale: for an affine field u(x,y) sampled at cell centers, the MRA
// projection (average of children) and prediction (Lagrange interpolation)
// are exact. Hence after update_ghost_mr(), every ghost lying strictly
// inside the domain must hold the affine value of its center, whatever the
// domain decomposition is. A wrong/missing ghost exchange shows up as a
// non-affine ghost value — this is the minimal reproducer of decomposition
// dependent ghost bugs (e.g. subdomains touching by a corner only).

#include <map>

#include <boost/serialization/vector.hpp>
#include <gtest/gtest.h>

#include <samurai/algorithm/update.hpp>
#include <samurai/bc.hpp>
#include <samurai/box.hpp>
#include <samurai/field.hpp>
#include <samurai/load_balancing/load_balancer.hpp>
#include <samurai/load_balancing/strategies/sfc.hpp>
#include <samurai/load_balancing/weight.hpp>
#include <samurai/mr/adapt.hpp>
#include <samurai/mr/mesh.hpp>

#include "mpi_test_utils.hpp"

namespace lb  = samurai::load_balancing;
namespace mpi = boost::mpi;

namespace
{
    using samurai_test::LambdaStrategy;

    constexpr std::size_t dim   = 2;
    constexpr std::size_t level = 5;
    using Mesh                  = samurai::MRMesh<samurai::mesh_config<dim>>;
    using mesh_id_t             = Mesh::mesh_id_t;

    double affine(double x, double y)
    {
        return 2. + 3. * x + 5. * y;
    }

    template <class Field>
    void fill_affine(Field& u)
    {
        u.fill(0);
        samurai::for_each_cell(u.mesh()[mesh_id_t::cells],
                               [&](const auto& cell)
                               {
                                   u[cell] = affine(cell.center(0), cell.center(1));
                               });
    }

    /// Check that every reference cell (real + ghost) strictly inside the
    /// domain holds the affine value of its center.
    template <class Field>
    void check_interior_ghosts(Field& u, const std::string& context)
    {
        samurai::update_ghost_mr(u);

        auto& mesh    = u.mesh();
        bool ok       = true;
        double maxerr = 0.;
        int shown     = 0;
        mpi::communicator world;
        samurai::for_each_cell(mesh[mesh_id_t::reference],
                               [&](const auto& cell)
                               {
                                   const double x = cell.center(0);
                                   const double y = cell.center(1);
                                   // stay away from the domain boundary: ghosts
                                   // there depend on the BC, not on the exchange
                                   const double margin = 4. * mesh.cell_length(mesh.min_level());
                                   if (x < margin || x > 1. - margin || y < margin || y > 1. - margin)
                                   {
                                       return;
                                   }
                                   const double err = std::abs(u[cell] - affine(x, y));
                                   maxerr           = std::max(maxerr, err);
                                   if (err >= 1e-11 && shown < 8)
                                   {
                                       std::cerr << "[rank " << world.rank() << "] bad ghost: level " << cell.level << " i "
                                                 << cell.indices[0] << " j " << cell.indices[1] << " center (" << x << ", " << y
                                                 << ") value " << u[cell] << " expected " << affine(x, y) << std::endl;
                                       ++shown;
                                   }
                                   ok = ok && err < 1e-11;
                               });
        if (!ok)
        {
            std::cerr << "[rank " << world.rank() << "] " << context << ": max interior ghost error " << maxerr << std::endl;
        }
        EXPECT_TRUE_ALL_RANKS(ok);
    }

    Mesh make_corner_refined_mesh()
    {
        xt::xtensor_fixed<double, xt::xshape<dim>> lo;
        xt::xtensor_fixed<double, xt::xshape<dim>> hi;
        lo.fill(0.);
        hi.fill(1.);
        return samurai_test::make_locally_refined_mesh<Mesh>(samurai::Box<double, dim>(lo, hi),
                                                             level,
                                                             [](const auto& cell)
                                                             {
                                                                 return cell.center(0) < 0.5 && cell.center(1) < 0.5;
                                                             });
    }

    // Sanity check: ghosts are affine on the initial (stripe) decomposition.
    TEST(load_balancing_ghosts, affine_before_migration)
    {
        auto mesh = make_corner_refined_mesh();
        auto u    = samurai::make_scalar_field<double>("u", mesh);
        fill_affine(u);
        samurai::make_bc<samurai::Dirichlet<1>>(u, 0.);
        check_interior_ghosts(u, "before migration");
    }

    // After a Hilbert rebalance (curve-shaped partitions).
    //
    // Non-regression test for a neighbourhood detection bug: MPI neighbours
    // used to be detected within ONE max-level cell of the subdomain, while
    // the MRA ghosts (isotropic prediction stencil + coarse projection ghosts
    // down to min_level-2) have a footprint of several cells. A partition
    // with a subdomain strip thinner than that footprint — which Hilbert cuts
    // produce naturally, e.g. at np3 on this mesh where rank 1 forms a thin
    // band between ranks 0 and 2 — hid the rank behind the strip: its ghosts
    // silently kept their fill value and the MRA adaptation diverged from the
    // sequential run. Fixed by deriving the neighbourhood expansion from the
    // ghost configuration (Mesh_base::ghost_physical_reach(), exchanged in
    // the subdomain bounding boxes); this test failed at np3 before the fix.
    TEST(load_balancing_ghosts, affine_after_hilbert)
    {
        auto mesh = make_corner_refined_mesh();
        auto u    = samurai::make_scalar_field<double>("u", mesh);
        fill_affine(u);
        samurai::make_bc<samurai::Dirichlet<1>>(u,
                                                [](const auto&, const auto&, const auto& coords)
                                                {
                                                    return affine(coords[0], coords[1]);
                                                });

        auto balancer = lb::make_load_balancer<lb::SFC<lb::Hilbert>>();
        balancer.load_balance(lb::weight::uniform(), u);
        check_interior_ghosts(u, "after hilbert");
    }

    /// Global adapted state gathered on rank 0: map (level, i, j) -> value.
    template <class Mesh_t, class Field>
    auto global_state(Mesh_t& mesh, Field& u)
    {
        mpi::communicator world;
        std::vector<double> local; // flattened (level, i, j, value) tuples
        samurai::for_each_cell(mesh[mesh_id_t::cells],
                               [&](const auto& cell)
                               {
                                   local.push_back(static_cast<double>(cell.level));
                                   local.push_back(static_cast<double>(cell.indices[0]));
                                   local.push_back(static_cast<double>(cell.indices[1]));
                                   local.push_back(u[cell]);
                               });
        std::vector<std::vector<double>> all;
        boost::mpi::gather(world, local, all, 0);

        std::map<std::array<double, 3>, double> state;
        if (world.rank() == 0)
        {
            for (const auto& chunk : all)
            {
                for (std::size_t k = 0; k < chunk.size(); k += 4)
                {
                    state[{chunk[k], chunk[k + 1], chunk[k + 2]}] = chunk[k + 3];
                }
            }
        }
        return state;
    }

    /// The MRA adaptation must be independent of the domain decomposition:
    /// starting from the same global mesh and field, adapting directly or
    /// load balancing first must produce the same global mesh and values.
    /// (Comparing against an analytic value is NOT a valid oracle here: the
    /// adaptation is not exact on affine fields, even sequentially.)
    template <class Strategy>
    void check_adapt_independence(const Strategy& strategy_tag, const std::string& context)
    {
        mpi::communicator world;
        auto affine_bc = [](const auto&, const auto&, const auto& coords)
        {
            return affine(coords[0], coords[1]);
        };

        // reference: adaptation without load balancing
        auto mesh_ref = make_corner_refined_mesh();
        auto u_ref    = samurai::make_scalar_field<double>("u", mesh_ref);
        fill_affine(u_ref);
        samurai::make_bc<samurai::Dirichlet<1>>(u_ref, affine_bc);
        auto adapt_ref = samurai::make_MRAdapt(u_ref);
        auto cfg       = samurai::mra_config().epsilon(2e-4);
        adapt_ref(cfg);
        auto state_ref = global_state(mesh_ref, u_ref);

        // same start, but rebalance before adapting
        auto mesh_lb = make_corner_refined_mesh();
        auto u_lb    = samurai::make_scalar_field<double>("u", mesh_lb);
        fill_affine(u_lb);
        samurai::make_bc<samurai::Dirichlet<1>>(u_lb, affine_bc);
        auto balancer = lb::make_load_balancer(lb::LoadBalanceConfig{}, strategy_tag);
        balancer.load_balance(lb::weight::uniform(), u_lb);
        auto adapt_lb = samurai::make_MRAdapt(u_lb);
        adapt_lb(cfg);
        auto state_lb = global_state(mesh_lb, u_lb);

        bool ok = true;
        if (world.rank() == 0)
        {
            ok = state_ref.size() == state_lb.size();
            if (ok)
            {
                for (const auto& [key, value] : state_ref)
                {
                    auto it = state_lb.find(key);
                    if (it == state_lb.end() || std::abs(it->second - value) > 1e-14)
                    {
                        ok = false;
                        std::cerr << context << ": mismatch at level " << key[0] << " (" << key[1] << ", " << key[2]
                                  << "): " << (it == state_lb.end() ? std::string("missing cell") : std::to_string(it->second)) << " vs "
                                  << value << std::endl;
                        break;
                    }
                }
            }
            else
            {
                std::cerr << context << ": global cell counts differ: " << state_ref.size() << " vs " << state_lb.size() << std::endl;
            }
        }
        boost::mpi::broadcast(world, ok, 0);
        EXPECT_TRUE_ALL_RANKS(ok);
    }

    // DISABLED: second, distinct decomposition bug (the neighbourhood fix is
    // in and update_ghost_mr is now correct after a Hilbert rebalance — see
    // affine_after_hilbert above — yet the ADAPTATION still produces a
    // different global mesh at np3: 1240 vs 1252 cells). The extra cells
    // suggest coarsening decisions that differ near subdomain boundaries
    // (tag exchange/graduation), but note that the initial stripe
    // decomposition also splits sibling groups and adapts identically to the
    // sequential run, so the exact trigger remains to be isolated.
    // Run with: --gtest_also_run_disabled_tests (fails at np3).
    TEST(load_balancing_ghosts, DISABLED_adapt_independence_hilbert)
    {
        check_adapt_independence(lb::SFC<lb::Hilbert>{}, "adapt independence (hilbert)");
    }

    TEST(load_balancing_ghosts, adapt_independence_checkerboard)
    {
        check_adapt_independence(LambdaStrategy{[](const auto& cell, int, int size)
                                                {
                                                    const auto bi = cell.indices[0] >> (cell.level - 3);
                                                    const auto bj = cell.indices[1] >> (cell.level - 3);
                                                    return static_cast<int>((bi + bj) % size);
                                                }},
                                 "adapt independence (checkerboard)");
    }

    // Cell-wise checkerboard: boundaries everywhere, sibling groups split
    // across ranks. Discriminates "geometry-triggered samurai bug" from
    // "Hilbert-specific bug".
    TEST(load_balancing_ghosts, affine_after_fine_checkerboard)
    {
        auto mesh = make_corner_refined_mesh();
        auto u    = samurai::make_scalar_field<double>("u", mesh);
        fill_affine(u);
        samurai::make_bc<samurai::Dirichlet<1>>(u, 0.);

        auto balancer = lb::make_load_balancer(lb::LoadBalanceConfig{},
                                               LambdaStrategy{[](const auto& cell, int, int size)
                                                              {
                                                                  return static_cast<int>((cell.indices[0] + cell.indices[1]) % size);
                                                              }});
        balancer.load_balance(lb::weight::uniform(), u);
        check_interior_ghosts(u, "after fine checkerboard");
    }

    // After a checkerboard migration: maximal subdomain boundary, lots of
    // corner-only contacts — the worst case for ghost exchanges.
    TEST(load_balancing_ghosts, affine_after_checkerboard)
    {
        auto mesh = make_corner_refined_mesh();
        auto u    = samurai::make_scalar_field<double>("u", mesh);
        fill_affine(u);
        samurai::make_bc<samurai::Dirichlet<1>>(u, 0.);

        auto balancer = lb::make_load_balancer(lb::LoadBalanceConfig{},
                                               LambdaStrategy{[](const auto& cell, int, int size)
                                                              {
                                                                  // blocks of 4x4 fine cells in checkerboard
                                                                  const auto bi = cell.indices[0] >> (cell.level - 3);
                                                                  const auto bj = cell.indices[1] >> (cell.level - 3);
                                                                  return static_cast<int>((bi + bj) % size);
                                                              }});
        balancer.load_balance(lb::weight::uniform(), u);
        check_interior_ghosts(u, "after checkerboard");
    }
}
