// Copyright 2018-2025 the samurai's authors
// SPDX-License-Identifier:  BSD-3-Clause

// Ownership of the blocks of a nested (MATNEST/VECNEST) block assembly.
//
// MatCreateNest() and VecCreateNest() take their own reference on every block,
// and MatSetLocalToGlobalMapping() on the mappings of each block. Once the nest
// exists, it must be the only owner of all of them, so that destroying the nest
// frees them. Each test keeps an extra reference on these objects, destroys the
// nest (through the solver or directly) and checks that this extra reference is
// the last one left.

#include <array>
#include <cmath>
#include <vector>

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
            return mra::make_mesh(box, mesh_config<dim>().min_level(3).max_level(3));
        }

        PetscObject as_object(auto petsc_handle)
        {
            return reinterpret_cast<PetscObject>(petsc_handle);
        }

        // Takes a reference on every block of the nested matrix A, and on the local-to-global mappings of these blocks
        // (only built with MPI).
        std::vector<PetscObject> reference_blocks(Mat A)
        {
            PetscInt rows = 0;
            PetscInt cols = 0;
            MatNestGetSize(A, &rows, &cols);
            std::vector<PetscObject> objects;
            auto keep = [&](PetscObject object)
            {
                PetscObjectReference(object);
                objects.push_back(object);
            };
            for (PetscInt i = 0; i < rows; ++i)
            {
                for (PetscInt j = 0; j < cols; ++j)
                {
                    Mat block = nullptr;
                    MatNestGetSubMat(A, i, j, &block);
                    keep(as_object(block));

                    ISLocalToGlobalMapping row_mapping = nullptr;
                    ISLocalToGlobalMapping col_mapping = nullptr;
                    MatGetLocalToGlobalMapping(block, &row_mapping, &col_mapping);
                    if (row_mapping)
                    {
                        keep(as_object(row_mapping));
                    }
                    if (col_mapping && col_mapping != row_mapping)
                    {
                        keep(as_object(col_mapping));
                    }
                }
            }
            return objects;
        }

        // Checks that nobody but the caller still holds the objects, then releases them.
        void expect_last_reference_and_destroy(std::vector<PetscObject>& objects)
        {
            ASSERT_FALSE(objects.empty());
            for (auto& object : objects)
            {
                PetscInt count         = 0;
                const char* class_name = nullptr;
                const char* name       = nullptr;
                PetscObjectGetReference(object, &count);
                PetscObjectGetClassName(object, &class_name);
                PetscObjectGetName(object, &name);
                EXPECT_EQ(count, 1) << "leaked " << class_name << " '" << name << "'";
                PetscObjectDestroy(&object);
            }
            objects.clear();
        }

        // Backward Euler step of two coupled heat equations, with a zero block:
        //   [ (1 + dt k) Id + dt Diff    -dt k Id               ] [u]   [u_n]
        //   [ 0                          (1 + dt k) Id + dt Diff] [v] = [v_n]
        template <class Field>
        auto make_coupled_heat_operator()
        {
            double dt = 1e-3;
            double k  = 10;
            auto diff = make_diffusion_order2<Field>();
            auto id   = make_identity<Field>();
            auto zero = make_zero_operator<Field>();
            return make_block_operator<2, 2>((1 + dt * k) * id + dt * diff, (-dt * k) * id, zero, (1 + dt * k) * id + dt * diff);
        }

        class PetscNestedBlockOwnership : public ::testing::Test
        {
          protected:

            decltype(unit_square_mesh()) mesh = unit_square_mesh();

            using field_t = decltype(make_scalar_field<double>("u", mesh));

            field_t u_n = make_scalar_field<double>("u_n",
                                                    mesh,
                                                    [](const auto& x)
                                                    {
                                                        return std::exp(-10 * ((x[0] - 0.5) * (x[0] - 0.5) + (x[1] - 0.5) * (x[1] - 0.5)));
                                                    });
            field_t v_n = make_scalar_field<double>("v_n", mesh, 0.);
            field_t u   = make_scalar_field<double>("u", mesh, 0.);
            field_t v   = make_scalar_field<double>("v", mesh, 0.);

            void SetUp() override
            {
                make_bc<Neumann<1>>(u, 0.);
                make_bc<Neumann<1>>(v, 0.);
            }
        };

        // Solves the system and returns a reference on each block of the matrix it assembled.
        template <class Solver, class Field>
        std::vector<PetscObject> solve_and_reference_blocks(Solver& solver, Field& rhs_u, Field& rhs_v)
        {
            std::vector<PetscObject> blocks;
            solver.after_matrix_assembly = [&](KSP&, PC& pc, Mat& A)
            {
                blocks = reference_blocks(A);
                solver.set_pc_fieldsplit(pc);
            };
            solver.solve(rhs_u, rhs_v);
            return blocks;
        }
    }

    TEST_F(PetscNestedBlockOwnership, solver_destruction_frees_the_matrix_blocks)
    {
        auto op = make_coupled_heat_operator<field_t>();
        std::vector<PetscObject> blocks;
        {
            auto solver = petsc::make_solver<petsc::BlockAssemblyType::NestedMatrices>(op);
            solver.set_unknowns(u, v);
            blocks = solve_and_reference_blocks(solver, u_n, v_n);
        }
        expect_last_reference_and_destroy(blocks);
    }

    TEST_F(PetscNestedBlockOwnership, reset_frees_the_matrix_blocks)
    {
        auto op     = make_coupled_heat_operator<field_t>();
        auto solver = petsc::make_solver<petsc::BlockAssemblyType::NestedMatrices>(op);
        solver.set_unknowns(u, v);

        // Every reset() followed by a new setup must release the previous matrices.
        for (int i = 0; i < 3; ++i)
        {
            auto blocks = solve_and_reference_blocks(solver, u_n, v_n);
            solver.reset();
            expect_last_reference_and_destroy(blocks);
        }
    }

    TEST_F(PetscNestedBlockOwnership, nested_vectors_own_their_blocks)
    {
        auto op     = make_coupled_heat_operator<field_t>();
        auto solver = petsc::make_solver<petsc::BlockAssemblyType::NestedMatrices>(op);
        solver.set_unknowns(u, v);
        auto& assembly = solver.assembly();

        std::array<Vec, 3> nests = {assembly.create_solution_vector_from_unknown_fields(),
                                    assembly.create_rhs_vector(op.tie_rhs(u_n, v_n)),
                                    assembly.create_vector(u, v)};
        for (auto& nest : nests)
        {
            std::vector<PetscObject> blocks;
            for (PetscInt i = 0; i < 2; ++i)
            {
                Vec block = nullptr;
                VecNestGetSubVec(nest, i, &block);
                PetscObjectReference(as_object(block));
                blocks.push_back(as_object(block));
            }
            VecDestroy(&nest);
            expect_last_reference_and_destroy(blocks);
        }
    }
}
