#pragma once
#include "SamuraiDM.hpp"
#include <samurai/print.hpp>

namespace samurai_new
{
    namespace petsc
    {

        enum Smoothers : int
        {
            Petsc,
            GaussSeidel,
            SymGaussSeidel
        };

        template <class Dsctzr>
        class GeometricMultigrid
        {
            using Mesh  = typename Dsctzr::Mesh;
            using Field = typename Dsctzr::field_t;

          private:

            Dsctzr* _discretizer          = nullptr;
            Mesh* _mesh                   = nullptr;
            SamuraiDM<Dsctzr>* _samuraiDM = nullptr;

          public:

            GeometricMultigrid()
            {
            }

            GeometricMultigrid(Dsctzr& assembly, Mesh& mesh)
            {
                _discretizer = &assembly;
                _mesh        = &mesh;
            }

            void destroy_petsc_objects()
            {
                if (_samuraiDM)
                {
                    _samuraiDM->destroy_petsc_objects();
                }
            }

            ~GeometricMultigrid()
            {
                destroy_petsc_objects();
                _samuraiDM = nullptr;
            }

            void apply_as_pc(KSP& ksp)
            {
                PetscInt transfer_ops_arg = samurai_new::TransferOperators::Assembled;
                PetscOptionsGetInt(NULL, NULL, "--samg_transfer_ops", &transfer_ops_arg, NULL);
                samurai_new::TransferOperators transfer_ops = static_cast<samurai_new::TransferOperators>(transfer_ops_arg);

                PetscInt prediction_stencil_radius = 0;
                PetscOptionsGetInt(NULL, NULL, "--samg_pred_order", &prediction_stencil_radius, NULL);

                PetscBool smoother_is_set = PETSC_FALSE;
                Smoothers smoother        = SymGaussSeidel;
                char smoother_char_array[10];
                PetscOptionsGetString(NULL, NULL, "--samg_smooth", smoother_char_array, 10, &smoother_is_set);
                if (smoother_is_set)
                {
                    std::string value = smoother_char_array;
                    if (value == "gs")
                    {
                        smoother = GaussSeidel;
                    }
                    else if (value == "sgs")
                    {
                        smoother = SymGaussSeidel;
                    }
                    else if (value == "petsc")
                    {
                        smoother = Petsc;
                    }
                    else
                    {
                        samurai::io::eprint("ERROR: unknown value for argument --smooth\n\n");
                    }
                }

                samurai::io::print(samurai::io::root, "Samurai multigrid: \n");

                samurai::io::print(samurai::io::root, "    smoothers         : ");
                if (smoother == GaussSeidel)
                {
                    samurai::io::print(samurai::io::root, "Gauss-Seidel (pre: lexico., post: antilexico.)");
                }
                else if (smoother == SymGaussSeidel)
                {
                    samurai::io::print(samurai::io::root, "symmetric Gauss-Seidel");
                }
                else if (smoother == Petsc)
                {
                    samurai::io::print(samurai::io::root, "petsc options");
                }
                samurai::io::print(samurai::io::root, "\n");

                samurai::io::print(samurai::io::root, "    transfer operators: ");
                if (transfer_ops == TransferOperators::Assembled)
                {
                    samurai::io::print(samurai::io::root, "P assembled, R assembled");
                }
                else if (transfer_ops == TransferOperators::Assembled_PTranspose)
                {
                    samurai::io::print(samurai::io::root, "P assembled, R = P^T");
                }
                else if (transfer_ops == TransferOperators::MatrixFree_Arrays)
                {
                    samurai::io::print(samurai::io::root, "P mat-free, R mat-free (via double*)");
                }
                else if (transfer_ops == TransferOperators::MatrixFree_Fields)
                {
                    samurai::io::print(samurai::io::root, "P mat-free, R mat-free (via Fields)");
                }
                samurai::io::print(samurai::io::root, "\n");

                samurai::io::print(samurai::io::root, "    prediction order  : {}\n", prediction_stencil_radius);

                _samuraiDM = new SamuraiDM<Dsctzr>(PETSC_COMM_SELF, *_discretizer, *_mesh, transfer_ops, prediction_stencil_radius);
                KSPSetDM(ksp, _samuraiDM->PetscDM());

                // Default outer solver: CG
                // KSPSetType(_ksp, "cg");

                // Preconditioner: geometric multigrid
                PC mg;
                KSPGetPC(ksp, &mg);
                PCSetType(mg, PCMG);

                KSPSetComputeOperators(ksp, SamuraiDM<Dsctzr>::ComputeMatrix, NULL);

                PetscInt levels = -1;
                PCMGGetLevels(mg, &levels);
                if (_mesh->max_level() == 1)
                {
                    levels = 1;
                }
                else if (levels < 2)
                {
                    levels = std::max(static_cast<int>(_mesh->max_level()) - 3, 2);
                    levels = std::min(levels, 8);
                }
                samurai::io::print(samurai::io::root, "    levels            : {}\n", levels);
                PCMGSetLevels(mg, levels, nullptr);

                // All of the following must be called after PCMGSetLevels()

                if (smoother == GaussSeidel)
                {
                    PCMGSetDistinctSmoothUp(mg);
                }

                if (smoother != Petsc)
                {
                    for (int i = 1; i < levels; i++)
                    {
                        if (smoother == SymGaussSeidel)
                        {
                            KSP smoother_ksp;
                            PCMGGetSmoother(mg, i, &smoother_ksp);
                            KSPSetType(smoother_ksp, "richardson");
                            KSPSetTolerances(smoother_ksp, PETSC_DEFAULT, PETSC_DEFAULT, PETSC_DEFAULT, 1);
                            PC smoother_pc;
                            KSPGetPC(smoother_ksp, &smoother_pc);
                            PCSetType(smoother_pc, PCSOR);
                            PCSORSetSymmetric(smoother_pc, MatSORType::SOR_SYMMETRIC_SWEEP);
                            // PCSetType(smoother_pc, PCJACOBI);
                            PCSORSetIterations(smoother_pc, 1, 1);
                        }
                        else if (smoother == GaussSeidel)
                        {
                            // Pre-smoothing
                            KSP pre_smoother_ksp;
                            PCMGGetSmootherDown(mg, i, &pre_smoother_ksp);
                            KSPSetType(pre_smoother_ksp, "richardson");
                            KSPSetTolerances(pre_smoother_ksp, PETSC_DEFAULT, PETSC_DEFAULT, PETSC_DEFAULT, 1);
                            PC pre_smoother;
                            KSPGetPC(pre_smoother_ksp, &pre_smoother);
                            PCSetType(pre_smoother, PCSOR);
                            PCSORSetSymmetric(pre_smoother, MatSORType::SOR_FORWARD_SWEEP);
                            // PCSetType(pre_smoother, PCJACOBI);
                            PCSORSetIterations(pre_smoother, 1, 1);

                            // Post-smoothing
                            KSP post_smoother_ksp;
                            PCMGGetSmootherUp(mg, i, &post_smoother_ksp);
                            KSPSetType(post_smoother_ksp, "richardson");
                            KSPSetTolerances(post_smoother_ksp, PETSC_DEFAULT, PETSC_DEFAULT, PETSC_DEFAULT, 1);
                            PC post_smoother;
                            KSPGetPC(post_smoother_ksp, &post_smoother);
                            PCSetType(post_smoother, PCSOR);
                            PCSORSetSymmetric(post_smoother, MatSORType::SOR_BACKWARD_SWEEP);
                            // PCSetType(post_smoother, PCJACOBI);
                            PCSORSetIterations(post_smoother, 1, 1);
                        }
                    }
                }
                // Override by command line arguments
                // KSPSetFromOptions(_ksp);
            }
        };

    }
}
