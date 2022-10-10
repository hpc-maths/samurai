#pragma once
#include "SamuraiDM.hpp"

namespace samurai_new { namespace petsc {

    template<class Dsctzr>
    class GeometricMultigrid
    {
        using Mesh = typename Dsctzr::Mesh;
        using Field = typename Dsctzr::field_t;

    private:
        Dsctzr* _discretizer = nullptr;
        Mesh* _mesh = nullptr;
        SamuraiDM<Dsctzr>* _samuraiDM = nullptr;
        TransferOperators _transfer_ops;
        int _prediction_order;

    public:
        GeometricMultigrid() {}

        GeometricMultigrid(Dsctzr& discretizer, Mesh& mesh, TransferOperators transfer_ops, int prediction_order)
        {
            _discretizer = &discretizer;
            _mesh = &mesh;
            _transfer_ops = transfer_ops;
            _prediction_order = prediction_order;
        }

        void destroy_petsc_objects()
        {
            if (_samuraiDM)
                _samuraiDM->destroy_petsc_objects();
        }

        ~GeometricMultigrid()
        {
            destroy_petsc_objects();
            _samuraiDM = nullptr;
        }

        void apply_as_pc(KSP& ksp)
        {
            std::cout << "Prediction order: " << _prediction_order << std::endl;

            _samuraiDM = new SamuraiDM<Dsctzr>(PETSC_COMM_SELF, *_discretizer, *_mesh, _transfer_ops, _prediction_order);
            KSPSetDM(ksp, _samuraiDM->PetscDM());

            // Default outer solver: CG
            //KSPSetType(_ksp, "cg");

            // Preconditioner: geometric multigrid
            PC mg;
            KSPGetPC(ksp, &mg);
            PCSetType(mg, PCMG);

            KSPSetComputeOperators(ksp, SamuraiDM<Dsctzr>::ComputeMatrix, NULL);

            PetscInt levels = -1;
            PCMGGetLevels(mg, &levels);
            if (levels < 2)
            {
                levels = std::max(static_cast<int>(_mesh->max_level()) - 3, 2);
                levels = std::min(levels, 8);
            }
            std::cout << "Multigrid levels: " << levels << std::endl;
            PCMGSetLevels(mg, levels, nullptr);

            // All of the following must be called after PCMGSetLevels()
            //PCMGSetDistinctSmoothUp(mg);

            for (int i=1; i<levels; i++)
            {
                KSP smoother_ksp;
                PCMGGetSmoother(mg, i, &smoother_ksp);
                KSPSetType(smoother_ksp, "richardson");
                KSPSetTolerances(smoother_ksp, PETSC_DEFAULT, PETSC_DEFAULT, PETSC_DEFAULT, 1);
                PC smoother;
                KSPGetPC(smoother_ksp, &smoother);
                PCSetType(smoother, PCSOR);
                PCSORSetSymmetric(smoother, MatSORType::SOR_SYMMETRIC_SWEEP);
                //PCSetType(smoother, PCJACOBI);
                PCSORSetIterations(smoother, 1, 1);


                // // Pre-smoothing
                // KSP pre_smoother_ksp;
                // PCMGGetSmootherDown(mg, i, &pre_smoother_ksp);
                // KSPSetType(pre_smoother_ksp, "richardson");
                // KSPSetTolerances(pre_smoother_ksp, PETSC_DEFAULT, PETSC_DEFAULT, PETSC_DEFAULT, 1);
                // PC pre_smoother;
                // KSPGetPC(pre_smoother_ksp, &pre_smoother);
                // PCSetType(pre_smoother, PCSOR);
                // PCSORSetSymmetric(pre_smoother, MatSORType::SOR_FORWARD_SWEEP);
                // //PCSetType(pre_smoother, PCJACOBI);
                // PCSORSetIterations(pre_smoother, 1, 1);

                // // Post-smoothing
                // KSP post_smoother_ksp;
                // PCMGGetSmootherUp(mg, i, &post_smoother_ksp);
                // KSPSetType(post_smoother_ksp, "richardson");
                // KSPSetTolerances(post_smoother_ksp, PETSC_DEFAULT, PETSC_DEFAULT, PETSC_DEFAULT, 1);
                // PC post_smoother;
                // KSPGetPC(post_smoother_ksp, &post_smoother);
                // PCSetType(post_smoother, PCSOR);
                // PCSORSetSymmetric(post_smoother, MatSORType::SOR_BACKWARD_SWEEP);
                // //PCSetType(post_smoother, PCJACOBI);
                // PCSORSetIterations(post_smoother, 1, 1);
            }
            // Override by command line arguments
            //KSPSetFromOptions(_ksp);
        }

    };

}}