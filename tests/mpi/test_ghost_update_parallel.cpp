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
// Two more suites check the exchanges themselves, on the periodic cases:
//
//  (C) Merged fields (fixtures merged_fields_2d/3d). update_ghost_mr(u, v, w)
//      exchanges the ghosts of all its fields in one message per neighbour; the
//      result must be bit-identical to updating each field on its own.
//
//  (D) Tag exchange (fixtures tag_exchange_2d/3d). update_tag_periodic and
//      update_tag_subdomains only combine the tags of copies of the same cell
//      modulo the period, so position-derived tags give an exact oracle (see
//      the comment above expect_tag_exchange).
//
// Every combination is a distinct GoogleTest case so a failure pinpoints it,
// and each executable is run at np = 2, 3, 4 by CTest.
//
// This suite surfaced a real bug: the 3D periodic ghost update deadlocked/crashed
// intermittently under MPI because update_ghost_periodic reused its MPI request
// vector across periodic dimensions (double wait_all on completed boost::mpi
// serialized requests). Fixed in update_periodic.hpp; the 3D periodic cases below
// are the regression guard.

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
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

    // ---- (C) merged periodic exchange of several fields -------------------
    //
    // update_ghost_mr(u, v, w) exchanges the periodic and subdomain ghosts of
    // all its fields in one message per neighbour. The fields are independent,
    // so every value, ghosts included, must be bit-identical to the one obtained
    // by updating each field on its own. Fields with different numbers of
    // components check the packing offsets.

    std::vector<ICase> make_periodic_icases()
    {
        std::vector<ICase> cases;
        for (const auto& c : make_icases())
        {
            if (c.periodic)
            {
                cases.push_back(c);
            }
        }
        return cases;
    }

    template <std::size_t Dim, class Cell>
    std::uint64_t cell_hash(const Cell& cell)
    {
        std::uint64_t h = 1469598103934665603ULL;
        auto mix        = [&](std::uint64_t v)
        {
            h ^= v + 0x9e3779b97f4a7c15ULL + (h << 6) + (h >> 2);
        };
        mix(static_cast<std::uint64_t>(cell.level));
        for (std::size_t d = 0; d < Dim; ++d)
        {
            mix(static_cast<std::uint64_t>(static_cast<long>(cell.indices[d])));
        }
        return h;
    }

    template <std::size_t Dim>
    void expect_merged_fields_match_single_field_updates(const ICase& c, const std::string& ctx)
    {
        using mesh_id_t = typename config<Dim>::mesh_id_t;

        DomainCorner<Dim> lo, hi;
        domain_bounds<Dim>(c.domain, lo, hi);
        auto mesh = build_mesh_on_domain<Dim>(c.geom, c.domain, c.stencil_size, c.periodic, lo, hi);
        auto u    = samurai::make_scalar_field<double>("u", mesh);
        u.fill(0.);
        apply_decomposition<Dim>(c.decomp, u);
        auto& m = u.mesh();

        auto v  = samurai::make_vector_field<double, 2>("v", m);
        auto w  = samurai::make_scalar_field<double>("w", m);
        auto u1 = samurai::make_scalar_field<double>("u1", m);
        auto v1 = samurai::make_vector_field<double, 2>("v1", m);
        auto w1 = samurai::make_scalar_field<double>("w1", m);
        for (auto* f : {&u, &w, &u1, &w1})
        {
            f->fill(0.);
        }
        v.fill(0.);
        v1.fill(0.);
        samurai::for_each_cell(m[mesh_id_t::cells],
                               [&](const auto& cell)
                               {
                                   const auto h = cell_hash<Dim>(cell);
                                   u[cell]      = static_cast<double>(h % 1000003) / 7.;
                                   w[cell]      = static_cast<double>((h >> 20) % 1000003) / 11.;
                                   v[cell][0]   = static_cast<double>((h >> 10) % 1000003) / 13.;
                                   v[cell][1]   = static_cast<double>((h >> 30) % 1000003) / 17.;
                                   u1[cell]     = u[cell];
                                   w1[cell]     = w[cell];
                                   v1[cell][0]  = v[cell][0];
                                   v1[cell][1]  = v[cell][1];
                               });

        samurai::update_ghost_mr(u, v, w);
        samurai::update_ghost_mr(u1);
        samurai::update_ghost_mr(v1);
        samurai::update_ghost_mr(w1);

        std::size_t mismatches = 0;
        samurai::for_each_cell(m[mesh_id_t::reference],
                               [&](const auto& cell)
                               {
                                   // bitwise comparison on purpose: the merged exchange must not
                                   // change a single bit
                                   if (u[cell] != u1[cell] || w[cell] != w1[cell] || v[cell][0] != v1[cell][0] || v[cell][1] != v1[cell][1])
                                   {
                                       ++mismatches;
                                   }
                               });
        if (mismatches != 0)
        {
            mpi::communicator world;
            std::cerr << "[rank " << world.rank() << "] " << ctx << ": " << mismatches
                      << " cells differ between the merged and the single-field updates" << std::endl;
        }
        EXPECT_TRUE_ALL_RANKS(mismatches == 0);
    }

    class merged_fields_2d : public samurai_test::MpiTest,
                             public testing::WithParamInterface<ICase>
    {
    };

    TEST_P(merged_fields_2d, match_single_field_updates)
    {
        expect_merged_fields_match_single_field_updates<2>(GetParam(), "2d_" + icase_label(GetParam()));
    }

    INSTANTIATE_TEST_SUITE_P(all, merged_fields_2d, testing::ValuesIn(make_periodic_icases()), icase_name);

    class merged_fields_3d : public samurai_test::MpiTest,
                             public testing::WithParamInterface<ICase>
    {
    };

    TEST_P(merged_fields_3d, match_single_field_updates)
    {
        expect_merged_fields_match_single_field_updates<3>(GetParam(), "3d_" + icase_label(GetParam()));
    }

    INSTANTIATE_TEST_SUITE_P(all, merged_fields_3d, testing::ValuesIn(make_periodic_icases()), icase_name);

    // ---- (D) periodic and subdomain tag exchange ---------------------------
    //
    // Oracle. Every real cell starts with a tag in [1, 127] derived from its
    // position, every ghost outside the domain with the flag bit 128, every
    // other ghost with 0. update_tag_periodic and update_tag_subdomains only
    // combine (or, or overwrite with) the tags of copies of the SAME cell modulo
    // the period. So after the exchange, at every level:
    //   - a real cell holds its own tag, plus the flag if and only if a copy of
    //     it lies outside the domain along a single dimension within the ghost
    //     width on some rank (the second pass of update_tag_periodic brings the
    //     tag of that copy back to the cell);
    //   - the low bits of any other cell are 0 or the tag of the real cell it
    //     is a copy of (0 if there is none);
    //   - a ghost that some neighbour must fill holds that tag in its low bits:
    //     an in-domain ghost of a real cell (subdomain exchange), and a ghost
    //     outside the domain along a single dimension, within the ghost width,
    //     of a real cell (first pass of the periodic exchange along that
    //     dimension). Corners, filled through a chain of copies, are only
    //     checked against the previous rule.
    // Run once with update_tag_subdomains(..., erase = false) and once with
    // erase = true, the two uses of mr/adapt.hpp.

    template <std::size_t Dim>
    using cell_key = std::array<long, Dim + 1>;

    template <std::size_t Dim>
    std::uint8_t tag_of(const cell_key<Dim>& key)
    {
        std::uint64_t h = 1469598103934665603ULL;
        for (long k : key)
        {
            h ^= static_cast<std::uint64_t>(k) + 0x9e3779b97f4a7c15ULL + (h << 6) + (h >> 2);
        }
        return static_cast<std::uint8_t>(1 + h % 127);
    }

    template <std::size_t Dim>
    void expect_tag_exchange(const ICase& c, bool erase, const std::string& ctx)
    {
        using mesh_id_t = typename config<Dim>::mesh_id_t;
        mpi::communicator world;

        DomainCorner<Dim> lo, hi;
        domain_bounds<Dim>(c.domain, lo, hi);
        auto mesh = build_mesh_on_domain<Dim>(c.geom, c.domain, c.stencil_size, c.periodic, lo, hi);
        auto u    = samurai::make_scalar_field<double>("u", mesh);
        u.fill(0.);
        apply_decomposition<Dim>(c.decomp, u);
        auto& m = u.mesh();

        auto key_of = [](const auto& cell)
        {
            cell_key<Dim> key{};
            key[0] = static_cast<long>(cell.level);
            for (std::size_t d = 0; d < Dim; ++d)
            {
                key[d + 1] = static_cast<long>(cell.indices[d]);
            }
            return key;
        };

        const auto& domain    = m.domain();
        const auto domain_min = domain.min_indices();
        const auto domain_max = domain.max_indices();
        const long gw         = m.ghost_width();

        // Position of a cell with respect to the domain: the cell it is a copy
        // of, the number of dimensions along which it lies outside the domain,
        // and how far outside (along the last of them).
        struct placement
        {
            cell_key<Dim> wrapped;
            std::size_t n_out = 0;
            long out_distance = 0;
        };

        auto place = [&](const cell_key<Dim>& key)
        {
            placement p;
            p.wrapped                 = key;
            const std::size_t delta_l = domain.level() - static_cast<std::size_t>(key[0]);
            for (std::size_t d = 0; d < Dim; ++d)
            {
                const long dmin   = static_cast<long>(domain_min[d] >> delta_l);
                const long dmax   = static_cast<long>(domain_max[d] >> delta_l);
                const long period = dmax - dmin;
                const long x      = key[d + 1];
                if (x < dmin || x >= dmax)
                {
                    ++p.n_out;
                    p.out_distance = (x < dmin) ? dmin - x : x - dmax + 1;
                }
                p.wrapped[d + 1] = dmin + (((x - dmin) % period) + period) % period;
            }
            return p;
        };

        // Gather on every rank the keys selected by `select` on every rank.
        auto gather_keys = [&](const auto& cells, auto&& select)
        {
            std::vector<long> local;
            samurai::for_each_cell(cells,
                                   [&](const auto& cell)
                                   {
                                       const auto key = key_of(cell);
                                       if (select(key))
                                       {
                                           local.insert(local.end(), key.begin(), key.end());
                                       }
                                   });
            std::vector<std::vector<long>> all;
            mpi::all_gather(world, local, all);
            std::set<cell_key<Dim>> keys;
            for (const auto& chunk : all)
            {
                for (std::size_t k = 0; k + Dim + 1 <= chunk.size(); k += Dim + 1)
                {
                    cell_key<Dim> key{};
                    std::copy(chunk.begin() + static_cast<std::ptrdiff_t>(k),
                              chunk.begin() + static_cast<std::ptrdiff_t>(k + Dim + 1),
                              key.begin());
                    keys.insert(key);
                }
            }
            return keys;
        };

        const auto real = gather_keys(m[mesh_id_t::cells],
                                      [](const auto&)
                                      {
                                          return true;
                                      });
        // the real cells with a copy outside the domain along a single
        // dimension within the ghost width, on any rank
        std::set<cell_key<Dim>> flagged;
        for (const auto& key : gather_keys(m[mesh_id_t::reference],
                                           [&](const auto& key)
                                           {
                                               const auto p = place(key);
                                               return p.n_out == 1 && p.out_distance <= gw;
                                           }))
        {
            flagged.insert(place(key).wrapped);
        }

        constexpr std::uint8_t flag = 128;
        auto tag                    = samurai::make_scalar_field<std::uint8_t>("tag", m);
        tag.fill(0);
        samurai::for_each_cell(m[mesh_id_t::reference],
                               [&](const auto& cell)
                               {
                                   if (place(key_of(cell)).n_out > 0)
                                   {
                                       tag[cell] = flag;
                                   }
                               });
        samurai::for_each_cell(m[mesh_id_t::cells],
                               [&](const auto& cell)
                               {
                                   tag[cell] = tag_of<Dim>(key_of(cell));
                               });

        const auto& ref = m[mesh_id_t::reference];
        for (std::size_t level = ref.min_level(); level <= ref.max_level(); ++level)
        {
            samurai::update_tag_periodic(level, tag);
            samurai::update_tag_subdomains(level, tag, erase);
        }

        std::set<cell_key<Dim>> local_cells;
        samurai::for_each_cell(m[mesh_id_t::cells],
                               [&](const auto& cell)
                               {
                                   local_cells.insert(key_of(cell));
                               });

        std::size_t errors           = 0;
        std::size_t checked_interior = 0;
        std::size_t checked_periodic = 0;
        samurai::for_each_cell(ref,
                               [&](const auto& cell)
                               {
                                   const auto key              = key_of(cell);
                                   const auto p                = place(key);
                                   const std::uint8_t expected = real.count(p.wrapped) ? tag_of<Dim>(p.wrapped) : 0;
                                   const std::uint8_t value    = tag[cell];
                                   const std::uint8_t low      = value & static_cast<std::uint8_t>(flag - 1);

                                   bool ok = true;
                                   std::string rule;
                                   if (local_cells.count(key))
                                   {
                                       const auto own = static_cast<std::uint8_t>(tag_of<Dim>(key) | (flagged.count(key) ? flag : 0));
                                       ok             = (value == own);
                                       rule           = "real cell, expected " + std::to_string(own);
                                   }
                                   else if (expected != 0 && p.n_out == 0)
                                   {
                                       ok   = (low == expected);
                                       rule = "in-domain ghost, expected low bits " + std::to_string(expected);
                                       ++checked_interior;
                                   }
                                   else if (expected != 0 && p.n_out == 1 && p.out_distance <= gw)
                                   {
                                       ok   = (low == expected);
                                       rule = "periodic ghost, expected low bits " + std::to_string(expected);
                                       ++checked_periodic;
                                   }
                                   else
                                   {
                                       ok   = (low == 0 || low == expected);
                                       rule = "other cell, expected low bits 0 or " + std::to_string(expected);
                                   }
                                   if (!ok && errors++ < 8)
                                   {
                                       std::cerr << "[rank " << world.rank() << "] " << ctx << ": tag " << int(value) << " at level "
                                                 << key[0] << " (" << rule << ")" << std::endl;
                                   }
                               });
        EXPECT_TRUE_ALL_RANKS(errors == 0);

        // Guard against a vacuous pass: both exchanges must have filled ghosts,
        // and the second periodic pass must have had something to bring back.
        const std::size_t total_interior = mpi::all_reduce(world, checked_interior, std::plus<std::size_t>());
        const std::size_t total_periodic = mpi::all_reduce(world, checked_periodic, std::plus<std::size_t>());
        EXPECT_GT(total_interior, 0u) << ctx << ": no in-domain ghost was checked";
        EXPECT_GT(total_periodic, 0u) << ctx << ": no periodic ghost was checked";
        EXPECT_FALSE(flagged.empty()) << ctx << ": no real cell has a periodic copy";
    }

    class tag_exchange_2d : public samurai_test::MpiTest,
                            public testing::WithParamInterface<ICase>
    {
    };

    TEST_P(tag_exchange_2d, tags_of_copies_agree)
    {
        expect_tag_exchange<2>(GetParam(), false, "2d_or_" + icase_label(GetParam()));
        expect_tag_exchange<2>(GetParam(), true, "2d_erase_" + icase_label(GetParam()));
    }

    INSTANTIATE_TEST_SUITE_P(all, tag_exchange_2d, testing::ValuesIn(make_periodic_icases()), icase_name);

    class tag_exchange_3d : public samurai_test::MpiTest,
                            public testing::WithParamInterface<ICase>
    {
    };

    TEST_P(tag_exchange_3d, tags_of_copies_agree)
    {
        expect_tag_exchange<3>(GetParam(), false, "3d_or_" + icase_label(GetParam()));
        expect_tag_exchange<3>(GetParam(), true, "3d_erase_" + icase_label(GetParam()));
    }

    INSTANTIATE_TEST_SUITE_P(all, tag_exchange_3d, testing::ValuesIn(make_periodic_icases()), icase_name);
}
