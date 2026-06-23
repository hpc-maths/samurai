// Copyright 2018-2025 the samurai's authors
// SPDX-License-Identifier:  BSD-3-Clause

// Flux operator consistency across domain decompositions.
//
// Rationale: a finite-volume flux divergence is a *local* operator. For the
// same global mesh, field and ghosts, the value computed on a cell must not
// depend on which rank owns it nor on where the subdomain boundaries lie.
// In particular it must be identical to the value computed sequentially.
//
// This is a minimal reproducer for a flux-assembly bug at MPI subdomain
// boundaries that coincide with a level jump (the configuration produced by
// non-stripe partitions, e.g. SFC): the coarse/fine interface flux of a
// boundary cell is then mis-assembled, even though the field and all ghosts
// are correct. The ghost-only tests in test_lb_ghosts.cpp do NOT catch it.
//
// Oracle: concentrate every cell on rank 0. That rank then owns the whole
// mesh and computes the flux with no MPI neighbour, i.e. exactly the
// sequential result. Any other decomposition must reproduce it cell by cell.

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
#include <samurai/mr/mesh.hpp>
#include <samurai/schemes/fv.hpp>

#include "mpi_test_utils.hpp"

namespace lb  = samurai::load_balancing;
namespace mpi = boost::mpi;

namespace
{
    using samurai_test::LambdaStrategy;

    class load_balancing_flux : public samurai_test::MpiTest
    {
    };

    constexpr std::size_t dim    = 2;
    constexpr std::size_t base   = 5; // mesh spans levels base .. base+2 (staircase)
    constexpr std::size_t n_comp = 2; // vector Burgers, exactly like the demo
    using Mesh                   = samurai::MRMesh<samurai::mesh_config<dim>>;
    using mesh_id_t              = Mesh::mesh_id_t;

    double curved(double x, double y, std::size_t comp)
    {
        // curved field: the flux of a *linear* field is too special (the
        // mis-assembled boundary contribution cancels); non-zero curvature
        // exposes the level-jump flux bug.
        if (comp == 0)
        {
            return 1. + 0.5 * std::sin(4. * x) + 0.3 * std::cos(3. * y);
        }
        return 0.7 + 0.4 * std::cos(2.5 * x) - 0.2 * std::sin(3.5 * y);
    }

    template <class Field>
    void fill_curved(Field& u)
    {
        u.fill(0);
        samurai::for_each_cell(u.mesh()[mesh_id_t::cells],
                               [&](const auto& cell)
                               {
                                   for (std::size_t c = 0; c < n_comp; ++c)
                                   {
                                       u[cell][c] = curved(cell.center(0), cell.center(1), c);
                                   }
                               });
    }

    /// Uniform mesh at `base`, refine quadrant {x<0.5,y<0.5} to base+1, then
    /// the sub-quadrant {x<0.25,y<0.25} to base+2: a graded 3-level staircase,
    /// reproducing the level-7-coarse-of-7/8-jump configuration of the demo.
    Mesh make_staircase_mesh()
    {
        using value_t = Mesh::interval_t::value_t;
        const samurai::Box<double, dim> box({0., 0.}, {1., 1.});
        auto uniform = samurai::mra::make_mesh(box, samurai::mesh_config<dim>().min_level(base).max_level(base));

        auto add_children = [](auto& cl, std::size_t lvl, value_t i, value_t j)
        {
            for (unsigned m = 0; m < (1U << dim); ++m)
            {
                const value_t jc = 2 * j + static_cast<value_t>((m >> 1) & 1U);
                const value_t ic = 2 * i + static_cast<value_t>(m & 1U);
                cl[lvl + 1][{jc}].add_point(ic);
            }
        };

        Mesh::cl_type cl;
        samurai::for_each_cell(uniform[mesh_id_t::cells],
                               [&](const auto& cell)
                               {
                                   const value_t i = cell.indices[0];
                                   const value_t j = cell.indices[1];
                                   if (!(cell.center(0) < 0.5 && cell.center(1) < 0.5))
                                   {
                                       cl[base][{j}].add_point(i);
                                       return;
                                   }
                                   // refine to base+1
                                   for (unsigned m = 0; m < (1U << dim); ++m)
                                   {
                                       const value_t jc = 2 * j + static_cast<value_t>((m >> 1) & 1U);
                                       const value_t ic = 2 * i + static_cast<value_t>(m & 1U);
                                       // sub-quadrant refined once more to base+2
                                       const double xc = (static_cast<double>(ic) + 0.5) / static_cast<double>(1 << (base + 1));
                                       const double yc = (static_cast<double>(jc) + 0.5) / static_cast<double>(1 << (base + 1));
                                       if (xc < 0.25 && yc < 0.25)
                                       {
                                           add_children(cl, base + 1, ic, jc);
                                       }
                                       else
                                       {
                                           cl[base + 1][{jc}].add_point(ic);
                                       }
                                   }
                               });
        return samurai::mra::make_mesh(cl, samurai::mesh_config<dim>().min_level(base).max_level(base + 2).max_stencil_size(6));
    }

    /// Global flux state gathered on rank 0: map (level, i, j) -> flux value,
    /// restricted to cells strictly inside the domain (the boundary flux
    /// depends on the B.C., not on the decomposition we want to probe).
    template <class Field>
    auto global_flux(Field& flux, std::size_t mid = static_cast<std::size_t>(mesh_id_t::cells))
    {
        auto& mesh = flux.mesh();
        mpi::communicator world;
        const double margin = 4. * mesh.cell_length(mesh.min_level());

        std::vector<double> local;
        samurai::for_each_cell(mesh[static_cast<mesh_id_t>(mid)],
                               [&](const auto& cell)
                               {
                                   const double x = cell.center(0);
                                   const double y = cell.center(1);
                                   if (x < margin || x > 1. - margin || y < margin || y > 1. - margin)
                                   {
                                       return;
                                   }
                                   local.push_back(static_cast<double>(cell.level));
                                   local.push_back(static_cast<double>(cell.indices[0]));
                                   local.push_back(static_cast<double>(cell.indices[1]));
                                   local.push_back(flux[cell][0]);
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

    /// Build the mesh, repartition it with `strategy`, then compute the
    /// upwind convection flux of the affine field and gather it on rank 0.
    // Oracle: everything on rank 0 => sequential flux (no MPI neighbour).
    auto oracle_strategy = LambdaStrategy{[](const auto&, int, int)
                                          {
                                              return 0;
                                          }};

    template <class Strategy>
    auto flux_after(const std::string& name, const Strategy& strategy)
    {
        auto mesh = make_staircase_mesh();
        auto u    = samurai::make_vector_field<n_comp>("u", mesh);
        fill_curved(u);
        samurai::make_bc<samurai::Dirichlet<1>>(
            u,
            [](const auto&, const auto&, const auto& coords)
            {
                return samurai::CollapsArray<double, n_comp, false>({curved(coords[0], coords[1], 0), curved(coords[0], coords[1], 1)});
            });

        auto balancer = lb::make_load_balancer(lb::LoadBalanceConfig{}, strategy);
        balancer.load_balance(lb::weight::uniform(), u);

        // re-impose the exact analytic values on the new partition so the
        // input is identical for every decomposition (isolates the flux
        // assembly from any field-migration question).
        fill_curved(u);
        samurai::update_ghost_mr(u);

        auto conv = samurai::make_convection_weno5<decltype(u)>();
        auto flux = samurai::make_vector_field<n_comp>("flux", mesh);
        flux.fill(0);
        flux = conv(u);
        samurai::save(std::filesystem::current_path(), fmt::format("{}_cells", name), mesh, u, flux);
        samurai::save(std::filesystem::current_path(), name, {true, true}, mesh, u, flux);
        return global_flux(flux);
    }

    // Diagnostic: gather u over ALL reference cells (ghosts included) after the
    // partition + update_ghost_mr, to tell a wrong-ghost bug from a
    // wrong-flux-assembly bug.
    template <class Strategy>
    auto uref_after(const Strategy& strategy)
    {
        auto mesh = make_staircase_mesh();
        auto u    = samurai::make_vector_field<n_comp>("u", mesh);
        fill_curved(u);
        samurai::make_bc<samurai::Dirichlet<1>>(
            u,
            [](const auto&, const auto&, const auto& coords)
            {
                return samurai::CollapsArray<double, n_comp, false>({curved(coords[0], coords[1], 0), curved(coords[0], coords[1], 1)});
            });
        auto balancer = lb::make_load_balancer(lb::LoadBalanceConfig{}, strategy);
        balancer.load_balance(lb::weight::uniform(), u);
        fill_curved(u);
        samurai::update_ghost_mr(u);
        return global_flux(u, static_cast<std::size_t>(mesh_id_t::reference));
    }

    void compare_to_oracle(const std::map<std::array<double, 3>, double>& oracle,
                           const std::map<std::array<double, 3>, double>& test,
                           const std::string& context)
    {
        mpi::communicator world;
        bool ok       = true;
        double maxerr = 0.;
        if (world.rank() == 0)
        {
            ok = oracle.size() == test.size();
            std::vector<std::tuple<double, std::array<double, 3>, double, double>> bad;
            for (const auto& [key, value] : oracle)
            {
                auto it = test.find(key);
                if (it == test.end())
                {
                    ok = false;
                    continue;
                }
                const double err = std::abs(it->second - value);
                maxerr           = std::max(maxerr, err);
                if (err > 1e-12)
                {
                    bad.emplace_back(err, key, value, it->second);
                }
                ok = ok && err <= 1e-12;
            }
            std::cerr << context << ": max flux error vs sequential oracle = " << maxerr << " (" << oracle.size() << " vs " << test.size()
                      << " interior cells), " << bad.size() << " bad" << std::endl;
            std::sort(bad.rbegin(), bad.rend());
            for (std::size_t k = 0; k < std::min<std::size_t>(40, bad.size()); ++k)
            {
                const auto& [err, key, ov, tv] = bad[k];
                std::cerr << "    level " << key[0] << " i " << key[1] << " j " << key[2] << ": oracle " << ov << " test " << tv << " (err "
                          << err << ")" << std::endl;
            }
        }
        boost::mpi::broadcast(world, ok, 0);
        EXPECT_TRUE_ALL_RANKS(ok);
    }

    // The flux must match the sequential oracle whatever the partition is.

    // Stripe-like control (row blocks): expected to already pass.
    TEST_F(load_balancing_flux, flux_row_blocks)
    {
        auto oracle = flux_after("oracle", oracle_strategy);
        auto test   = flux_after("row_blocks",
                               LambdaStrategy{[](const auto& cell, int, int size)
                                              {
                                                  const auto bj = cell.indices[1] >> (cell.level - 3);
                                                  return static_cast<int>(bj % size);
                                              }});
        compare_to_oracle(oracle, test, "row blocks");
    }

    // Block checkerboard: MPI boundaries cut through the refined quadrant and
    // its level jump. This is the configuration that SFC produces and that
    // breaks the flux assembly.
    TEST_F(load_balancing_flux, flux_block_checkerboard)
    {
        auto oracle = flux_after("oracle", oracle_strategy);
        auto test   = flux_after("block_checkerboard",
                               LambdaStrategy{[](const auto& cell, int, int size)
                                              {
                                                  const auto bi = cell.indices[0] >> (cell.level - 3);
                                                  const auto bj = cell.indices[1] >> (cell.level - 3);
                                                  return static_cast<int>((bi + bj) % size);
                                              }});
        compare_to_oracle(oracle, test, "block checkerboard");
    }

    // Cell-wise checkerboard: every sibling group split, maximal boundary.
    TEST_F(load_balancing_flux, flux_fine_checkerboard)
    {
        auto oracle = flux_after("oracle", oracle_strategy);
        auto test   = flux_after("fine_checkerboard",
                               LambdaStrategy{[](const auto& cell, int, int size)
                                              {
                                                  return static_cast<int>((cell.indices[0] + cell.indices[1]) % size);
                                              }});
        compare_to_oracle(oracle, test, "fine checkerboard");
    }

    // The actual Hilbert SFC partition: curve-shaped boundaries cutting the
    // refined staircase. This is the partition that fails in the demo.
    TEST_F(load_balancing_flux, flux_hilbert)
    {
        auto oracle = flux_after("oracle", oracle_strategy);
        auto test   = flux_after("hilbert", lb::SFC<lb::Hilbert>{});
        compare_to_oracle(oracle, test, "hilbert");
    }

    // Diagnostic: is u (ghosts included) identical between the decompositions?
    // If yes, the flux divergence is a flux-assembly bug, not a ghost bug.
    TEST_F(load_balancing_flux, ghosts_hilbert)
    {
        auto oracle = uref_after(oracle_strategy);
        auto test   = uref_after(lb::SFC<lb::Hilbert>{});
        compare_to_oracle(oracle, test, "u over reference (hilbert)");
    }
}
