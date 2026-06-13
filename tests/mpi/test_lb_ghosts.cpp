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
#include <samurai/stencil_field.hpp>

#include "mpi_test_utils.hpp"

namespace lb  = samurai::load_balancing;
namespace mpi = boost::mpi;

namespace
{
    using samurai_test::LambdaStrategy;

    class load_balancing_ghosts : public samurai_test::MpiTest
    {
    };

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
    TEST_F(load_balancing_ghosts, affine_before_migration)
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
    TEST_F(load_balancing_ghosts, affine_after_hilbert)
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

    /// Canonicalize the non-cell entries of the field (ghosts) to zero,
    /// keeping the cell values. Rationale: Field::resize() leaves new entries
    /// uninitialized and a few reference entries are never written by
    /// update_ghost_mr (e.g. out-of-domain ghosts whose data owner does not
    /// have them in its own reference). Their garbage content depends on the
    /// heap history, which would make this test order-dependent; scrubbing
    /// restores a deterministic state identical for both variants. The
    /// underlying coverage gap is documented in the roadmap (§ 5bis residue).
    template <class Field>
    void scrub_ghosts(Field& u)
    {
        std::vector<double> kept;
        samurai::for_each_cell(u.mesh()[mesh_id_t::cells],
                               [&](const auto& cell)
                               {
                                   kept.push_back(u[cell]);
                               });
        u.fill(0);
        std::size_t k = 0;
        samurai::for_each_cell(u.mesh()[mesh_id_t::cells],
                               [&](const auto& cell)
                               {
                                   u[cell] = kept[k++];
                               });
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
        scrub_ghosts(u_ref);
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
        scrub_ghosts(u_lb);
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
                std::map<int, int> per_level_ref, per_level_lb;
                for (const auto& [key, value] : state_ref)
                {
                    per_level_ref[static_cast<int>(key[0])]++;
                }
                for (const auto& [key, value] : state_lb)
                {
                    per_level_lb[static_cast<int>(key[0])]++;
                }
                for (const auto& [l, n] : per_level_ref)
                {
                    std::cerr << "  level " << l << ": ref " << n << " vs lb " << per_level_lb[l] << std::endl;
                }
                int shown = 0;
                for (const auto& [key, value] : state_ref)
                {
                    if (!state_lb.contains(key) && shown++ < 8)
                    {
                        std::cerr << "  ref-only cell: level " << key[0] << " (" << key[1] << ", " << key[2] << ") = " << value
                                  << std::endl;
                    }
                }
            }
        }
        boost::mpi::broadcast(world, ok, 0);
        EXPECT_TRUE_ALL_RANKS(ok);
    }

    // Non-regression for the second decomposition bug (failed at np3 before
    // the fix: 1240 vs 1252 cells). Root cause: the out-of-domain ghosts were
    // decomposition dependent — (a) outer_subdomain_corner designated the
    // owner of ALL outer layers with a single translation of ghost_width, so
    // the owner of layer 1 was the rank owning the cell at distance
    // ghost_width instead of the adjacent one, and its unfilled value
    // overwrote the correct one; (b) update_ghost_mr filled the outer ghosts
    // BEFORE any MPI exchange, so the B.C./extrapolation read stale inner
    // ghosts. Fixed by per-layer ownership and by the exchange/fill/exchange
    // order per level in update_ghost_mr.
    TEST_F(load_balancing_ghosts, adapt_independence_hilbert)
    {
        check_adapt_independence(lb::SFC<lb::Hilbert>{}, "adapt independence (hilbert)");
    }

    TEST_F(load_balancing_ghosts, adapt_independence_checkerboard)
    {
        check_adapt_independence(LambdaStrategy{[](const auto& cell, int, int size)
                                                {
                                                    const auto bi = cell.indices[0] >> (cell.level - 3);
                                                    const auto bj = cell.indices[1] >> (cell.level - 3);
                                                    return static_cast<int>((bi + bj) % size);
                                                }},
                                 "adapt independence (checkerboard)");
    }

    // Demo-scale independence: replay of the advected-disk physics (the demo
    // case, min_level 4 / max_level 10) over 30 steps, with Hilbert rebalances
    // at steps 1 and 20, compared step by step against the same run without
    // load balancing. This is the only scenario known to exercise the
    // graduation bound bug (a neighbour owning only fine cells used to bound
    // the coarse loop of the cross-rank graduation check above the local
    // coarse levels): it failed at np4, step 28, before the fix in
    // list_interval_to_refine_for_graduation. The simpler corner-refined
    // cases above do NOT cover it.
    TEST_F(load_balancing_ghosts, adapt_independence_demo_case)
    {
        mpi::communicator world;

        auto run_demo = [&](bool with_lb)
        {
            std::vector<std::map<std::array<double, 3>, double>> states;
            const samurai::Box<double, dim> box({0., 0.}, {1., 1.});
            auto config    = samurai::mesh_config<dim>().min_level(4).max_level(10).max_stencil_size(2).disable_minimal_ghost_width();
            auto demo_mesh = samurai::mra::make_mesh(box, config);
            auto u         = samurai::make_scalar_field<double>("u", demo_mesh);
            u.resize();
            samurai::for_each_cell(demo_mesh,
                                   [&](auto& cell)
                                   {
                                       auto center     = cell.center();
                                       const double d2 = (center[0] - 0.3) * (center[0] - 0.3) + (center[1] - 0.3) * (center[1] - 0.3);
                                       u[cell]         = (d2 <= 0.04) ? 1. : 0.;
                                   });
            samurai::make_bc<samurai::Dirichlet<1>>(u, 0.);

            const std::array<double, 2> a{1., 1.};
            const double dt = 0.5 * demo_mesh.min_cell_length();
            auto unp1       = samurai::make_scalar_field<double>("unp1", demo_mesh);
            auto adaptation = samurai::make_MRAdapt(u);
            auto cfg        = samurai::mra_config().epsilon(2e-4);
            adaptation(cfg);

            auto balancer = lb::make_load_balancer<lb::SFC<lb::Hilbert>>();
            for (std::size_t nt = 0; nt < 30; ++nt)
            {
                if (with_lb && (nt == 1 || nt == 20))
                {
                    balancer.load_balance(lb::weight::uniform(), u);
                }
                scrub_ghosts(u);
                adaptation(cfg);
                samurai::update_ghost_mr(u);
                unp1.resize();
                unp1 = u - dt * samurai::upwind(a, u);
                std::swap(u.array(), unp1.array());

                std::vector<double> local;
                samurai::for_each_cell(demo_mesh[mesh_id_t::cells],
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
                states.push_back(std::move(state));
            }
            return states;
        };

        auto sref = run_demo(false);
        auto slb  = run_demo(true);

        bool ok              = true;
        std::size_t bad_step = 0;
        if (world.rank() == 0)
        {
            for (std::size_t nt = 0; nt < sref.size() && ok; ++nt)
            {
                ok = sref[nt].size() == slb[nt].size();
                for (const auto& [key, value] : sref[nt])
                {
                    if (!ok)
                    {
                        break;
                    }
                    auto it = slb[nt].find(key);
                    ok      = it != slb[nt].end() && std::abs(it->second - value) <= 1e-13;
                }
                if (!ok)
                {
                    bad_step = nt;
                }
            }
            if (!ok)
            {
                std::cerr << "demo-case independence broken at step " << bad_step << " (" << sref[bad_step].size() << " vs "
                          << slb[bad_step].size() << " cells)" << std::endl;
            }
        }
        boost::mpi::broadcast(world, ok, 0);
        EXPECT_TRUE_ALL_RANKS(ok);
    }

    // Cell-wise checkerboard: boundaries everywhere, sibling groups split
    // across ranks. Discriminates "geometry-triggered samurai bug" from
    // "Hilbert-specific bug".
    TEST_F(load_balancing_ghosts, affine_after_fine_checkerboard)
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
    TEST_F(load_balancing_ghosts, affine_after_checkerboard)
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
