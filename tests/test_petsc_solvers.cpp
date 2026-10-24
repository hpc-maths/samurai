// Copyright 2018-2025 the samurai's authors
// SPDX-License-Identifier:  BSD-3-Clause

// Ownership of the PETSc objects held by the solvers.
//
// A solver owns its KSP/SNES, matrices and vectors, so it can be moved but not
// copied. Move-assigning a solver must hand over everything the source holds
// (scheme, PETSc objects, user callbacks) and release the PETSc objects the
// target held before, without destroying anything twice.

#include <type_traits>
#include <utility>

#include <gtest/gtest.h>

#include <samurai/mr/mesh.hpp>
#include <samurai/schemes/fv.hpp>

namespace samurai
{
    namespace
    {
        constexpr std::size_t dim = 2;

        auto unit_square_mesh()
        {
            Box<double, dim> box({0., 0.}, {1., 1.});
            auto cfg = mesh_config<dim>().min_level(3).max_level(3);
            return mra::make_mesh(box, cfg);
        }

        template <class Field>
        void expect_fields_near(const Field& actual, const Field& expected)
        {
            std::size_t nb_checked = 0;
            for_each_cell(actual.mesh(),
                          [&](const auto& cell)
                          {
                              EXPECT_NEAR(actual[cell], expected[cell], 1e-8);
                              ++nb_checked;
                          });
            EXPECT_GT(nb_checked, 0u);
        }

        void direct_lu(KSP& ksp, PC& pc)
        {
            KSPSetType(ksp, KSPPREONLY);
            PCSetType(pc, PCLU);
        }

        void newton_direct_lu(SNES& snes, KSP& ksp, PC& pc)
        {
            SNESSetType(snes, SNESNEWTONLS);
            direct_lu(ksp, pc);
        }

        // v -> v^2, applied cell by cell.
        template <class Field>
        auto make_square_reaction()
        {
            using cfg  = LocalCellSchemeConfig<SchemeType::NonLinear, Field, Field>;
            auto react = make_cell_based_scheme<cfg>();
            react.set_name("square");
            react.set_scheme_function(
                [](SchemeValue<cfg>& value, const auto& cell, const auto& field)
                {
                    value = field[cell] * field[cell];
                });
            react.set_jacobian_function(
                [](JacobianMatrix<cfg>& jac, const auto& cell, const auto& field)
                {
                    jac = 2 * field[cell];
                });
            return react;
        }

        // Builds a solver for `op_moved` and one for `op_overwritten`, sets up the
        // second one by solving once (so it holds PETSc objects of its own), then
        // move-assigns the first into the second. The result must be what a fresh
        // solver for `op_moved` computes.
        template <class OpMoved, class OpOverwritten, class Field, class Configure>
        void check_move_assignment(const OpMoved& op_moved, const OpOverwritten& op_overwritten, Field& rhs, Configure configure)
        {
            using solver_t = decltype(petsc::make_solver(op_moved));
            static_assert(!std::is_copy_constructible_v<solver_t>);
            static_assert(!std::is_copy_assignable_v<solver_t>);
            static_assert(std::is_move_assignable_v<solver_t>);

            auto& mesh     = rhs.mesh();
            auto reference = make_scalar_field<double>("reference", mesh, 0.5);
            auto result    = make_scalar_field<double>("result", mesh, 0.5);
            auto scratch   = make_scalar_field<double>("scratch", mesh, 0.5);
            make_bc<Dirichlet<1>>(reference, 0.);
            make_bc<Dirichlet<1>>(result, 0.);
            make_bc<Dirichlet<1>>(scratch, 0.);

            {
                auto fresh      = petsc::make_solver(op_moved);
                fresh.configure = configure;
                fresh.solve(reference, rhs);
            }

            bool configure_called = false;

            auto moved      = petsc::make_solver(op_moved);
            moved.configure = [&](auto&... args)
            {
                configure_called = true;
                configure(args...);
            };

            auto overwritten      = petsc::make_solver(op_overwritten);
            overwritten.configure = configure;
            overwritten.solve(scratch, rhs);

            overwritten = std::move(moved);
            overwritten.solve(result, rhs);

            EXPECT_TRUE(configure_called) << "the user callback was not moved";
            expect_fields_near(result, reference);
        }
    }

    TEST(petsc_solvers, linear_solver_move_assignment)
    {
        auto mesh = unit_square_mesh();
        auto u    = make_scalar_field<double>("u", mesh);
        auto rhs  = make_scalar_field<double>("rhs", mesh, 1.);

        auto diff = make_diffusion_order2<decltype(u)>();
        auto id   = make_identity<decltype(u)>();

        check_move_assignment(id + 0.1 * diff, id + 0.5 * diff, rhs, direct_lu);
    }

    TEST(petsc_solvers, nonlinear_solver_move_assignment)
    {
        auto mesh = unit_square_mesh();
        auto u    = make_scalar_field<double>("u", mesh);
        auto rhs  = make_scalar_field<double>("rhs", mesh, 1.);

        auto diff  = make_diffusion_order2<decltype(u)>();
        auto id    = make_identity<decltype(u)>();
        auto react = make_square_reaction<decltype(u)>();

        check_move_assignment(id + 0.1 * diff + 0.1 * react, id + 0.5 * diff + 0.5 * react, rhs, newton_direct_lu);
    }

    TEST(petsc_solvers, nonlinear_local_solvers_move_assignment)
    {
        auto mesh = unit_square_mesh();
        auto u    = make_scalar_field<double>("u", mesh);
        auto rhs  = make_scalar_field<double>("rhs", mesh, 1.);

        auto id    = make_identity<decltype(u)>();
        auto react = make_square_reaction<decltype(u)>();

        check_move_assignment(id + 0.1 * react, id + 0.5 * react, rhs, newton_direct_lu);
    }
}
