// Copyright 2018-2025 the samurai's authors
// SPDX-License-Identifier:  BSD-3-Clause

// Parallel ghost-update robustness suite.
//
// Goal: guarantee that update_ghost_mr() produces the correct ghost values in
// parallel for ANY combination of
//   - dimension (2D and 3D),
//   - stencil size (1 -> 5, i.e. ghost width 1 -> 3),
//   - periodic / non-periodic boundaries, and
//   - "tangled" domain decompositions in which the ghost layer of a subdomain
//     reaches THROUGH the neighbouring subdomain and into a third (or fourth)
//     one (thin strips, checkerboards, diagonal bands, Hilbert curves, and a
//     per-cell RandomHash partition that shreds the domain into islands), and
//   - non-cubic, origin-shifted domains (anisotropic cell counts, non-zero and
//     negative cell indices).
//
// The catalog of meshes and decompositions swept below lives in the gtest-free
// header ghost_cases.hpp, shared with the demos/mpi/ghost_cases visualisation
// tool so that the cases validated here and the cases one can inspect in
// ParaView are, by construction, the same. This file adds the two oracles and
// the GoogleTest wiring on top of that catalog.
//
// Two complementary oracles are used.
//
//  (A) Analytic affine oracle (fixture ghost_update_2d).
//      For an affine field u = a + b.x + c.y + d.z sampled at cell centers the
//      MRA projection (average of children) and prediction (Lagrange
//      interpolation) are BOTH exact, so after update_ghost_mr() every ghost
//      strictly inside the domain must hold the exact affine value of its
//      center, whatever the decomposition is. Any wrong/missing/duplicated
//      exchange shows up as a non-affine interior ghost. This is the strongest
//      per-value check, but it can only inspect INTERIOR ghosts (outer ghosts
//      are set by the boundary condition), and the interior margin scales with
//      the ghost width - which is only affordable in 2D (fine coarsest level).
//
//  (B) Decomposition-independence oracle (fixtures ghost_independence_2d/3d).
//      A ghost value is a function of the field and the boundary condition
//      only, never of the partition. So running update_ghost_mr() on a tangled
//      decomposition must reproduce, cell for cell, the values obtained on the
//      reference (no-load-balancing) decomposition, for ANY field. This scales
//      to 3D and to periodicity (the affine field is not periodic, but its
//      wrapped ghosts are still decomposition independent, so periodic meshes
//      are checked on EVERY ghost).
//
//      For non-periodic meshes a thin boundary band (a few fine cells) is
//      excluded: there the Dirichlet-BC ghosts, and the prediction that reads
//      them, are a decomposition-dependent residue that samurai does not
//      currently guarantee (documented as the "out-of-domain ghosts" note in
//      test_lb_ghosts). NB: this residue reaches one prediction stencil INTO the
//      domain in 3D, whereas in 2D the boundary-adjacent ghosts are already
//      decomposition independent - an asymmetry worth keeping in mind. The
//      genuine cross-rank exchange (interior ghosts and level-jump
//      projection/prediction ghosts, i.e. the ghosts that span several
//      subdomains) is fully checked.
//
// Every combination is a distinct GoogleTest case so a failure pinpoints it,
// and each executable is run at np = 2, 3, 4 by CTest.
//
// This suite surfaced a real bug: the 3D periodic ghost update deadlocked/crashed
// intermittently under MPI because update_ghost_periodic reused its MPI request
// vector across periodic dimensions (double wait_all on completed boost::mpi
// serialized requests). Fixed in update_periodic.hpp; the 3D periodic cases below
// are the regression guard.

#include <array>
#include <cmath>
#include <cstddef>
#include <functional>
#include <map>
#include <set>
#include <string>
#include <vector>

#include <boost/serialization/vector.hpp>
#include <gtest/gtest.h>

#include <samurai/algorithm/update.hpp>

#include "ghost_cases.hpp"
#include "mpi_test_utils.hpp"

namespace mpi = boost::mpi;

namespace
{
    using namespace samurai::ghost_cases;

    // Global max number of MPI neighbours: > 1 means at least one subdomain's
    // ghosts span more than one foreign subdomain.
    template <class Field>
    std::size_t max_mpi_neighbours(Field& u)
    {
        mpi::communicator world;
        return mpi::all_reduce(world, u.mesh().mpi_neighbourhood().size(), mpi::maximum<std::size_t>());
    }

    // ---- (A) analytic affine oracle --------------------------------------

    template <std::size_t Dim, class Field>
    void expect_affine_interior_ghosts(Field& u, int stencil_size, const std::string& ctx)
    {
        using mesh_id_t = typename config<Dim>::mesh_id_t;
        samurai::update_ghost_mr(u);

        auto& mesh      = u.mesh();
        const double dx = mesh.cell_length(mesh.min_level());
        // Stay away from the physical boundary: outer ghosts there are set by the
        // boundary condition (and by coarse projection ghosts reaching down below
        // min_level), not by the inter-rank exchange. The excluded band grows with
        // the ghost width so the test is fair at every stencil size.
        const double margin = (2. * ghost_width_of(stencil_size) + 2.) * dx;

        // Real cells were written by hand and trivially hold the affine value;
        // only the GHOSTS are produced by update_ghost_mr, so the oracle must be
        // applied to them alone.
        std::set<std::array<long, Dim + 1>> real_cells;
        samurai::for_each_cell(mesh[mesh_id_t::cells],
                               [&](const auto& cell)
                               {
                                   std::array<long, Dim + 1> key{};
                                   key[0] = static_cast<long>(cell.level);
                                   for (std::size_t d = 0; d < Dim; ++d)
                                   {
                                       key[d + 1] = static_cast<long>(cell.indices[d]);
                                   }
                                   real_cells.insert(key);
                               });

        mpi::communicator world;
        bool ok                    = true;
        double maxerr              = 0.;
        int shown                  = 0;
        std::size_t checked_ghosts = 0;
        samurai::for_each_cell(mesh[mesh_id_t::reference],
                               [&](const auto& cell)
                               {
                                   std::array<long, Dim + 1> key{};
                                   key[0] = static_cast<long>(cell.level);
                                   for (std::size_t d = 0; d < Dim; ++d)
                                   {
                                       key[d + 1] = static_cast<long>(cell.indices[d]);
                                   }
                                   if (real_cells.count(key))
                                   {
                                       return; // real cell: not produced by the exchange
                                   }

                                   bool interior = true;
                                   for (std::size_t d = 0; d < Dim; ++d)
                                   {
                                       const double xc = cell.center(d);
                                       if (xc < margin || xc > 1. - margin)
                                       {
                                           interior = false;
                                           break;
                                       }
                                   }
                                   if (!interior)
                                   {
                                       return;
                                   }
                                   ++checked_ghosts;
                                   const double err = std::abs(u[cell] - affine_at_center<Dim>(cell));
                                   maxerr           = std::max(maxerr, err);
                                   if (err >= 1e-11 && shown < 8)
                                   {
                                       std::cerr << "[rank " << world.rank() << "] " << ctx << ": bad ghost level " << cell.level
                                                 << " value " << u[cell] << " expected " << affine_at_center<Dim>(cell) << std::endl;
                                       ++shown;
                                   }
                                   ok = ok && err < 1e-11;
                               });
        if (!ok)
        {
            std::cerr << "[rank " << world.rank() << "] " << ctx << ": max interior ghost error " << maxerr << std::endl;
        }
        EXPECT_TRUE_ALL_RANKS(ok);

        // Guard against a vacuous pass: the interior region must actually contain
        // ghosts that were checked, otherwise the oracle proves nothing.
        const std::size_t total_checked = mpi::all_reduce(world, checked_ghosts, std::plus<std::size_t>());
        EXPECT_GT(total_checked, 0u) << ctx << ": no interior ghost was checked (margin too large?)";
    }

    // ---- (B) decomposition-independence oracle ---------------------------

    // Gather, on rank 0, the reference set (real cells + ghosts) of every rank as
    // a map (level, indices...) -> value. `consistent` is set to false if two
    // ranks report the same cell with different values (a ghost that disagrees
    // across ranks - already a bug on its own).
    //
    // `boundary_margin` > 0 (non-periodic case) drops every cell whose center is
    // within that physical distance of the domain boundary. This removes the
    // outer/boundary ghosts (set by the boundary condition) AND the thin in-domain
    // band next to them: the prediction stencil of a near-boundary ghost reaches
    // the coarse boundary ghosts, which are a decomposition-dependent residue that
    // samurai does not currently guarantee (see the "out-of-domain ghosts" note in
    // test_lb_ghosts). A periodic mesh passes margin = 0: every ghost wraps back
    // into the domain and must be decomposition independent, so all are checked.
    template <std::size_t Dim, class Field>
    std::map<std::array<long, Dim + 1>, double>
    gather_reference(Field& u, bool& consistent, double boundary_margin, const DomainCorner<Dim>& lo, const DomainCorner<Dim>& hi)
    {
        using mesh_id_t         = typename config<Dim>::mesh_id_t;
        constexpr std::size_t W = Dim + 2; // level + indices + value

        mpi::communicator world;
        std::vector<double> local;
        samurai::for_each_cell(u.mesh()[mesh_id_t::reference],
                               [&](const auto& cell)
                               {
                                   if (boundary_margin > 0.)
                                   {
                                       bool near_boundary = false;
                                       for (std::size_t d = 0; d < Dim; ++d)
                                       {
                                           const double xc = cell.center(d);
                                           if (xc < lo[d] + boundary_margin || xc > hi[d] - boundary_margin)
                                           {
                                               near_boundary = true;
                                               break;
                                           }
                                       }
                                       if (near_boundary)
                                       {
                                           return;
                                       }
                                   }
                                   local.push_back(static_cast<double>(cell.level));
                                   for (std::size_t d = 0; d < Dim; ++d)
                                   {
                                       local.push_back(static_cast<double>(cell.indices[d]));
                                   }
                                   local.push_back(u[cell]);
                               });

        std::vector<std::vector<double>> all;
        mpi::gather(world, local, all, 0);

        std::map<std::array<long, Dim + 1>, double> state;
        consistent = true;
        if (world.rank() == 0)
        {
            for (const auto& chunk : all)
            {
                for (std::size_t k = 0; k + W <= chunk.size(); k += W)
                {
                    std::array<long, Dim + 1> key{};
                    for (std::size_t d = 0; d <= Dim; ++d)
                    {
                        key[d] = static_cast<long>(std::llround(chunk[k + d]));
                    }
                    const double value = chunk[k + Dim + 1];
                    auto it            = state.find(key);
                    if (it == state.end())
                    {
                        state.emplace(key, value);
                    }
                    else if (std::abs(it->second - value) > 1e-11)
                    {
                        consistent = false;
                    }
                }
            }
        }
        return state;
    }

    // The ghost values on a tangled decomposition must match, cell for cell, the
    // values on the reference (no-LB) decomposition - boundary ghosts included.
    //
    // NB: the mesh must outlive the field it is bound to (the field holds only a
    // reference to it), so both meshes are kept in local variables here.
    template <std::size_t Dim>
    void
    expect_decomposition_independent(Geometry geom, DomainShape shape, int stencil_size, bool periodic, Decomp decomp, const std::string& ctx)
    {
        mpi::communicator world;

        DomainCorner<Dim> lo, hi;
        domain_bounds<Dim>(shape, lo, hi);

        auto mesh_ref = build_mesh_on_domain<Dim>(geom, shape, stencil_size, periodic, lo, hi);
        auto u_ref    = samurai::make_scalar_field<double>("u", mesh_ref);
        fill_affine<Dim>(u_ref, periodic);
        // Non-periodic: exclude a boundary band, where the Dirichlet-BC ghosts
        // (and the prediction that reads them) are a decomposition-dependent
        // residue outside samurai's guarantees. Scaled to the coarsest cells so
        // the band is thick enough whatever the finest level of the geometry.
        const double margin = periodic ? 0. : (ghost_width_of(stencil_size) + 2.) * mesh_ref.cell_length(mesh_ref.min_level());
        apply_decomposition<Dim>(Decomp::None, u_ref);
        samurai::update_ghost_mr(u_ref);
        bool ref_consistent = true;
        auto ref            = gather_reference<Dim>(u_ref, ref_consistent, margin, lo, hi);

        auto mesh_tst = build_mesh_on_domain<Dim>(geom, shape, stencil_size, periodic, lo, hi);
        auto u_tst    = samurai::make_scalar_field<double>("u", mesh_tst);
        fill_affine<Dim>(u_tst, periodic);
        apply_decomposition<Dim>(decomp, u_tst);
        if (world.size() >= 3 && is_tangled(decomp))
        {
            EXPECT_GE(max_mpi_neighbours(u_tst), 2u) << ctx << ": decomposition is not tangled";
        }
        samurai::update_ghost_mr(u_tst);
        bool tst_consistent = true;
        auto tst            = gather_reference<Dim>(u_tst, tst_consistent, margin, lo, hi);

        bool ok = true;
        if (world.rank() == 0)
        {
            ok = ref_consistent && tst_consistent;
            if (!ref_consistent)
            {
                std::cerr << ctx << ": reference decomposition holds inconsistent ghosts across ranks" << std::endl;
            }
            if (!tst_consistent)
            {
                std::cerr << ctx << ": tangled decomposition holds inconsistent ghosts across ranks" << std::endl;
            }

            std::size_t shared = 0;
            int shown          = 0;
            double maxerr      = 0.;
            for (const auto& [key, value] : ref)
            {
                auto it = tst.find(key);
                if (it == tst.end())
                {
                    continue; // the two decompositions need not own the same ghost set
                }
                ++shared;
                const double err = std::abs(it->second - value);
                maxerr           = std::max(maxerr, err);
                if (err > 1e-11)
                {
                    ok = false;
                    if (shown++ < 8)
                    {
                        std::cerr << ctx << ": ghost mismatch at level " << key[0] << ": reference " << value << " vs tangled "
                                  << it->second << std::endl;
                    }
                }
            }
            if (shared == 0)
            {
                ok = false;
                std::cerr << ctx << ": no shared ghost between the two decompositions" << std::endl;
            }
            if (!ok)
            {
                std::cerr << ctx << ": max ghost mismatch " << maxerr << " over " << shared << " shared cells" << std::endl;
            }
        }
        mpi::broadcast(world, ok, 0);
        EXPECT_TRUE_ALL_RANKS(ok);
    }

    // ---- (A) matrix: 2D analytic affine oracle ---------------------------

    std::string case_name(const testing::TestParamInfo<Case>& info)
    {
        return case_label(info.param);
    }

    class ghost_update_2d : public samurai_test::MpiTest,
                            public testing::WithParamInterface<Case>
    {
    };

    TEST_P(ghost_update_2d, affine_interior_ghosts)
    {
        const Case c          = GetParam();
        const std::string ctx = case_label(c);
        mpi::communicator world;

        auto mesh = build_mesh<2>(c.geom, c.stencil_size, /*periodic=*/false);
        auto u    = samurai::make_scalar_field<double>("u", mesh);
        u.fill(0.);
        samurai::for_each_cell(mesh[config<2>::mesh_id_t::cells],
                               [&](const auto& cell)
                               {
                                   u[cell] = affine_at_center<2>(cell);
                               });
        samurai::make_bc<samurai::Dirichlet<1>>(u,
                                                [](const auto&, const auto&, const auto& coords)
                                                {
                                                    return affine_at_coords<2>(coords);
                                                });

        apply_decomposition<2>(c.decomp, u);

        // Property 1: the decomposition is actually tangled - a subdomain has at
        // least two MPI neighbours, i.e. its ghosts span several subdomains.
        if (world.size() >= 3 && is_tangled(c.decomp))
        {
            EXPECT_GE(max_mpi_neighbours(u), 2u) << ctx << ": decomposition is not tangled";
        }

        // Property 2: every interior ghost is exactly affine.
        expect_affine_interior_ghosts<2>(u, c.stencil_size, ctx);
    }

    INSTANTIATE_TEST_SUITE_P(all, ghost_update_2d, testing::ValuesIn(make_cases()), case_name);

    // ---- (B) matrix: decomposition independence (2D + 3D, periodic) -------

    std::string icase_name(const testing::TestParamInfo<ICase>& info)
    {
        return icase_label(info.param);
    }

    class ghost_independence_2d : public samurai_test::MpiTest,
                                  public testing::WithParamInterface<ICase>
    {
    };

    TEST_P(ghost_independence_2d, ghosts_match_reference)
    {
        const ICase c = GetParam();
        expect_decomposition_independent<2>(c.geom, c.domain, c.stencil_size, c.periodic, c.decomp, "2d_" + icase_label(c));
    }

    INSTANTIATE_TEST_SUITE_P(all, ghost_independence_2d, testing::ValuesIn(make_icases()), icase_name);

    class ghost_independence_3d : public samurai_test::MpiTest,
                                  public testing::WithParamInterface<ICase>
    {
    };

    TEST_P(ghost_independence_3d, ghosts_match_reference)
    {
        const ICase c = GetParam();
        // Regression: the 3D periodic ghost update used to deadlock/crash
        // intermittently under MPI because update_ghost_periodic reused its MPI
        // request vector across periodic dimensions, calling wait_all() again on
        // already-completed boost::mpi serialized requests. Fixed by scoping the
        // request vector per dimension in update_periodic.hpp.
        expect_decomposition_independent<3>(c.geom, c.domain, c.stencil_size, c.periodic, c.decomp, "3d_" + icase_label(c));
    }

    INSTANTIATE_TEST_SUITE_P(all, ghost_independence_3d, testing::ValuesIn(make_icases()), icase_name);
}
