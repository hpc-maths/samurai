#pragma once
#include "PetscDM.hpp"

template<class Dsctzr>
class LaplacianSolver
{
    using Mesh = typename Dsctzr::Mesh;
    using Field = typename Dsctzr::field_t;

private:
    //inline static const std::string _args_prefix = "lap_";
    KSP _ksp;
    DM _dm = nullptr;


public:
    LaplacianSolver(Dsctzr& discretizer, Mesh& mesh)
    {
        create_solver(discretizer, mesh);
    }

private:
    void create_solver(Dsctzr& discretizer, Mesh& mesh)
    {
        KSP user_ksp;
        KSPCreate(PETSC_COMM_SELF, &user_ksp);
        KSPSetFromOptions(user_ksp);
        PC user_pc;
        KSPGetPC(user_ksp, &user_pc);
        PCType user_pc_type;
        PCGetType(user_pc, &user_pc_type);

        if (strcmp(user_pc_type, PCMG) != 0)
        {
            KSPCreate(PETSC_COMM_SELF, &_ksp);
            Mat A;
            discretizer.create_matrix(A);
            discretizer.assemble_matrix(A);
            PetscObjectSetName((PetscObject)A, "A");
            //KSPSetComputeOperators(_ksp, PetscDM<Dsctzr>::ComputeMatrix, NULL);
            KSPSetOperators(_ksp, A, A);
            KSPSetFromOptions(_ksp);
        }
        else
        {

            KSPCreate(PETSC_COMM_SELF, &_ksp);
            KSPSetFromOptions(_ksp);

            _dm = PetscDM<Dsctzr>::Create(PETSC_COMM_SELF, discretizer, mesh);
            KSPSetDM(_ksp, _dm);

            // Default outer solver: CG
            //KSPSetType(_ksp, "cg");

            // Preconditioner: geometric multigrid
            PC mg;
            KSPGetPC(_ksp, &mg);
            PCSetType(mg, PCMG);

            KSPSetComputeOperators(_ksp, PetscDM<Dsctzr>::ComputeMatrix, NULL);

            PetscInt levels = -1;
            PCMGGetLevels(mg, &levels);
            if (levels < 2)
            {
                levels = std::max(static_cast<int>(mesh.max_level()) - 3, 2);
                levels = std::min(levels, 8);
            }
            std::cout << "\tLevels: " << levels << std::endl;
            PCMGSetLevels(mg, levels, nullptr);

            // All of the following must be called after PCMGSetLevels()
            PCMGSetDistinctSmoothUp(mg);

            for (int i=1; i<levels; i++)
            {
                /*KSP smoother_ksp;
                PCMGGetSmoother(mg, i, &smoother_ksp);
                KSPSetType(smoother_ksp, "richardson");
                KSPSetTolerances(smoother_ksp, PETSC_DEFAULT, PETSC_DEFAULT, PETSC_DEFAULT, 1);
                PC smoother;
                KSPGetPC(smoother_ksp, &smoother);
                PCSetType(smoother, PCSOR);
                PCSORSetSymmetric(smoother, MatSORType::SOR_SYMMETRIC_SWEEP);
                //PCSetType(smoother, PCJACOBI);
                PCSORSetIterations(smoother, 1, 1);*/


                // Pre-smoothing
                KSP pre_smoother_ksp;
                PCMGGetSmootherDown(mg, i, &pre_smoother_ksp);
                KSPSetType(pre_smoother_ksp, "richardson");
                KSPSetTolerances(pre_smoother_ksp, PETSC_DEFAULT, PETSC_DEFAULT, PETSC_DEFAULT, 1);
                PC pre_smoother;
                KSPGetPC(pre_smoother_ksp, &pre_smoother);
                PCSetType(pre_smoother, PCSOR);
                PCSORSetSymmetric(pre_smoother, MatSORType::SOR_FORWARD_SWEEP);
                //PCSetType(pre_smoother, PCJACOBI);
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
                //PCSetType(post_smoother, PCJACOBI);
                PCSORSetIterations(post_smoother, 1, 1);
            }
            // Override by command line arguments
            //KSPSetFromOptions(_ksp);
        }
    }

public:
    void setup()
    {
        KSPSetUp(_ksp);
    }

    void solve(const Vec& b, Field& x_field)
    {
        Vec x;
        VecDuplicate(b, &x);

        /*std::cout << "b:" << std::endl;
        VecView(b, PETSC_VIEWER_STDOUT_(PETSC_COMM_SELF));
        std::cout << std::endl;*/
        
        KSPSolve(_ksp, b, x);

        KSPConvergedReason reason_code;
        KSPGetConvergedReason(_ksp, &reason_code);
        if (reason_code < 0)
        {
            using namespace std::string_literals;
            const char* reason_text;
            KSPGetConvergedReasonString(_ksp, &reason_text);
            fatal_error("Divergence of the solver ("s + reason_text + ")");
        }

        PetscInt n_iterations;
        KSPGetIterationNumber(_ksp, &n_iterations);
        std::cout << n_iterations << " iterations" << std::endl;
        std::cout << std::endl;
        //VecView(x, PETSC_VIEWER_STDOUT_(PETSC_COMM_SELF));
        //std::cout << std::endl;

        copy(x, x_field);
        VecDestroy(&x);
    }


    PetscErrorCode destroy()
    {
        KSPDestroy(&_ksp);
        if (_dm)
            DMDestroy(&_dm);
        PetscFunctionReturn(0);
    }
};