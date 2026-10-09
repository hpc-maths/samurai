// Copyright 2018-2025 the samurai's authors
// SPDX-License-Identifier:  BSD-3-Clause

// PCFIELDSPLIT on a block operator, in parallel.
//
// set_pc_fieldsplit() hands PETSc one index set per unknown. PETSc reads them as
// global indices of the matrix, so on every rank they must lie in the rows owned
// by that rank. The test checks this for the monolithic assembly, then solves the
// same system with the monolithic and nested assemblies and compares the results.

#include <algorithm>
#include <cmath>
#include <vector>

#include <gtest/gtest.h>

#include <samurai/mr/mesh.hpp>
#include <samurai/schemes/fv.hpp>

#include "mpi_test_utils.hpp"

namespace
{
    constexpr std::size_t dim = 2;

    using mesh_t  = decltype(samurai::mra::make_mesh(std::declval<samurai::Box<double, dim>>(), samurai::mesh_config<dim>()));
    using field_t = decltype(samurai::make_scalar_field<double>("u", std::declval<mesh_t&>()));

    // Coupled heat system, one implicit time step:
    //   (1 + dt k) u + dt diff(u) - dt k v = u_n
    //   (1 + dt k) v + dt diff(v) - dt k u = v_n
    template <samurai::petsc::BlockAssemblyType assembly_type>
    void solve_coupled_heat(mesh_t& mesh, field_t& unp1, field_t& vnp1, bool& fields_IS_owned, int& iterations)
    {
        auto bump = [](const auto& x)
        {
            double r2 = std::pow(x[0] - 0.5, 2) + std::pow(x[1] - 0.5, 2);
            return std::exp(-100 * r2);
        };
        auto u = samurai::make_scalar_field<double>("u", mesh, bump);
        auto v = samurai::make_scalar_field<double>("v", mesh, 0.);
        samurai::make_bc<samurai::Neumann<1>>(unp1, 0.);
        samurai::make_bc<samurai::Neumann<1>>(vnp1, 0.);

        double dt = 1e-3;
        double k  = 10;
        auto diff = samurai::make_diffusion_order2<field_t>();
        auto id   = samurai::make_identity<field_t>();
        auto Aii  = (1 + dt * k) * id + dt * diff;
        auto Aij  = (-dt * k) * id;

        auto op = samurai::make_block_operator<2, 2>(Aii, Aij, Aij, Aii);

        auto solver = samurai::petsc::make_solver<assembly_type>(op);
        solver.set_unknowns(unp1, vnp1);
        solver.configure = [](KSP& ksp, PC&)
        {
            KSPSetTolerances(ksp, 1e-12, 1e-14, PETSC_CURRENT, PETSC_CURRENT);
        };
        solver.after_matrix_assembly = [&](KSP&, PC& pc, Mat& A)
        {
            fields_IS_owned = true;
            if constexpr (assembly_type == samurai::petsc::BlockAssemblyType::Monolithic)
            {
                PetscInt row_start = 0;
                PetscInt row_end   = 0;
                MatGetOwnershipRange(A, &row_start, &row_end);

                std::vector<PetscInt> all_indices;
                auto fields_IS = solver.assembly().create_fields_IS();
                for (auto& is : fields_IS)
                {
                    PetscInt n = 0;
                    ISGetLocalSize(is, &n);
                    const PetscInt* indices = nullptr;
                    ISGetIndices(is, &indices);
                    all_indices.insert(all_indices.end(), indices, indices + n);
                    ISRestoreIndices(is, &indices);
                    ISDestroy(&is);
                }
                // The index sets must partition the rows owned by this rank.
                std::sort(all_indices.begin(), all_indices.end());
                bool local_ok = static_cast<PetscInt>(all_indices.size()) == row_end - row_start;
                for (std::size_t i = 0; local_ok && i < all_indices.size(); ++i)
                {
                    local_ok = all_indices[i] == row_start + static_cast<PetscInt>(i);
                }
                boost::mpi::communicator world;
                fields_IS_owned = boost::mpi::all_reduce(world, local_ok, std::logical_and<bool>());
            }
            // Wrong index sets make PCSetUp() fail on some ranks and hang the others:
            // fall back to the default preconditioner so that the test reports the failure.
            if (fields_IS_owned)
            {
                solver.set_pc_fieldsplit(pc);
            }
        };

        solver.solve(u, v);
        iterations = solver.iterations();
    }

    class PetscBlockFieldsplit : public samurai_test::MpiTest
    {
    };

    TEST_F(PetscBlockFieldsplit, monolithic_matches_nested)
    {
        samurai::Box<double, dim> box({0., 0.}, {1., 1.});
        auto config = samurai::mesh_config<dim>().min_level(5).max_level(5);
        auto mesh   = samurai::mra::make_mesh(box, config);

        using enum samurai::petsc::BlockAssemblyType;

        auto u_nest  = samurai::make_scalar_field<double>("u", mesh, 0.);
        auto v_nest  = samurai::make_scalar_field<double>("v", mesh, 0.);
        bool nest_ok = false;
        int nest_its = 0;
        solve_coupled_heat<NestedMatrices>(mesh, u_nest, v_nest, nest_ok, nest_its);

        auto u_mono  = samurai::make_scalar_field<double>("u", mesh, 0.);
        auto v_mono  = samurai::make_scalar_field<double>("v", mesh, 0.);
        bool mono_ok = false;
        int mono_its = 0;
        solve_coupled_heat<Monolithic>(mesh, u_mono, v_mono, mono_ok, mono_its);

        EXPECT_TRUE(mono_ok) << "the field index sets do not partition the rows owned by each rank";
        EXPECT_GT(nest_its, 0);
        EXPECT_GT(mono_its, 0);

        double max_diff = 0;
        samurai::for_each_cell(mesh[mesh_t::mesh_id_t::cells],
                               [&](const auto& cell)
                               {
                                   max_diff = std::max(max_diff, std::abs(u_mono[cell] - u_nest[cell]));
                                   max_diff = std::max(max_diff, std::abs(v_mono[cell] - v_nest[cell]));
                               });
        EXPECT_TRUE_ALL_RANKS(max_diff < 1e-10);
    }
}
