#pragma once
#include "intergrid_operators.hpp"
#include <samurai/petsc/utils.hpp>

namespace samurai_new
{
    namespace petsc
    {

        template <class Dsctzr>
        class SamuraiDM
        {
          private:

            DM _dm;
            LevelContext<Dsctzr> _ctx;
            // SamuraiDM* _coarse = nullptr;

          public:

            using Mesh  = typename Dsctzr::Mesh;
            using Field = typename Dsctzr::field_t;

            SamuraiDM(MPI_Comm comm, Dsctzr& assembly, Mesh& mesh, TransferOperators to, int prediction_order)
                : _ctx(assembly, mesh, to, prediction_order)
            {
                DMShellCreate(comm, &_dm);
                DefineShellFunctions(_dm, _ctx);
            }

            /*SamuraiDM(MPI_Comm comm, const LevelCtx<Dsctzr>& fine_ctx) :
                _ctx(fine_ctx)
            {
                DMShellCreate(comm, &_dm);
                DefineShellFunctions(_dm, _ctx);
            }*/

            DM& PetscDM()
            {
                return _dm;
            }

            void destroy_petsc_objects()
            {
                DMDestroy(&_dm);
            }

          private:

            static void DefineShellFunctions(DM& shell, LevelContext<Dsctzr>& ctx)
            {
                DMShellSetContext(shell, &ctx);
                DMShellSetCreateMatrix(shell, CreateMatrix);
                DMShellSetCreateGlobalVector(shell, CreateGlobalVector);
                DMShellSetCreateLocalVector(shell, CreateLocalVector);
                DMShellSetCoarsen(shell, Coarsen);
                if (ctx.transfer_ops == TransferOperators::MatrixFree_Fields || ctx.transfer_ops == TransferOperators::MatrixFree_Arrays)
                {
                    DMShellSetCreateInterpolation(shell, CreateMatFreeProlongation);
                    DMShellSetCreateRestriction(shell, CreateMatFreeRestriction);
                }
                else if (ctx.transfer_ops == TransferOperators::Assembled_PTranspose)
                {
                    DMShellSetCreateInterpolation(shell, CreateProlongationMatrix);
                }
                else if (ctx.transfer_ops == TransferOperators::Assembled)
                {
                    DMShellSetCreateInterpolation(shell, CreateProlongationMatrix);
                    DMShellSetCreateRestriction(shell, CreateRestrictionMatrix);
                }
            }

            static PetscErrorCode Coarsen(DM fine_dm, MPI_Comm /*comm*/, DM* coarse_dm)
            {
                // std::cout << "Coarsen - begin" << std::endl;
                LevelContext<Dsctzr>* fine_ctx;
                DMShellGetContext(fine_dm, &fine_ctx);

                LevelContext<Dsctzr>* coarse_ctx = new LevelContext(*fine_ctx);

                //_coarse = new SamuraiDM(PetscObjectComm((PetscObject)fine_dm),
                //*fine_ctx); *coarse_dm = _coarse->PetscDM();

                DMShellCreate(PetscObjectComm(reinterpret_cast<PetscObject>(fine_dm)), coarse_dm);
                DefineShellFunctions(*coarse_dm, *coarse_ctx);

                // DMShellCreate(PetscObjectComm((PetscObject)fine_dm),
                // coarse_dm); DefineShellFunctions(*coarse_dm, *coarse_ctx);
                // std::cout << "Coarsen (create level " << coarse_ctx->level <<
                // ")" << std::endl;
                return 0;
            }

            static PetscErrorCode CreateMatrix(DM shell, Mat* A)
            {
                LevelContext<Dsctzr>* ctx;
                DMShellGetContext(shell, &ctx);

                ctx->assembly().create_matrix(*A);
                // std::cout << "CreateMatrix - level " << ctx->level <<
                // std::endl;

                MatSetDM(*A, shell); // Why???
                return 0;
            }

          public:

            static PetscErrorCode ComputeMatrix(KSP ksp, Mat /*J*/, Mat jac, void* /*dummy_ctx*/)
            {
                DM shell;
                KSPGetDM(ksp, &shell);
                LevelContext<Dsctzr>* ctx;
                DMShellGetContext(shell, &ctx);

                ctx->assembly().assemble_matrix(jac);

                // MatView(jac, PETSC_VIEWER_STDOUT_(PETSC_COMM_SELF));
                // std::cout << std::endl;
                return 0;
            }

          private:

            static PetscErrorCode CreateMatFreeProlongation(DM coarse_dm, DM fine_dm, Mat* P, Vec* scaling)
            {
                LevelContext<Dsctzr>* coarse_ctx;
                DMShellGetContext(coarse_dm, &coarse_ctx);
                LevelContext<Dsctzr>* fine_ctx;
                DMShellGetContext(fine_dm, &fine_ctx);

                auto nf = static_cast<PetscInt>(fine_ctx->mesh().nb_cells());
                auto nc = static_cast<PetscInt>(coarse_ctx->mesh().nb_cells());

                MatCreateShell(PetscObjectComm(reinterpret_cast<PetscObject>(fine_dm)), nf, nc, nf, nc, coarse_ctx, P);
                MatShellSetOperation(*P, MATOP_MULT, reinterpret_cast<void (*)(void)>(prolongation));

                *scaling = nullptr; // Why???

                return 0;
            }

            static PetscErrorCode prolongation(Mat P, Vec x, Vec y)
            {
                LevelContext<Dsctzr>* coarse_ctx;
                LevelContext<Dsctzr>* fine_ctx;
                MatShellGetContext(P, &coarse_ctx);
                fine_ctx = coarse_ctx->finer;

                // std::cout << "coarse vector x:" << std::endl;
                // VecView(x, PETSC_VIEWER_STDOUT_(PETSC_COMM_SELF)); std::cout
                // << std::endl;

                if (coarse_ctx->transfer_ops == TransferOperators::MatrixFree_Fields)
                {
                    Field coarse_field("coarse_field", coarse_ctx->mesh());
                    samurai::petsc::copy(x, coarse_field);
                    Field fine_field = multigrid::prolong(coarse_field, fine_ctx->mesh(), coarse_ctx->prediction_order);
                    samurai::petsc::copy(fine_field, y);

                    // std::cout << "prolongated vector (marche):" << std::endl;
                    // VecView(y, PETSC_VIEWER_STDOUT_(PETSC_COMM_SELF));
                    // std::cout << std::endl;
                }
                else
                {
                    const double* xarray;
                    double* yarray;
                    VecGetArrayRead(x, &xarray);
                    VecGetArray(y, &yarray);
                    multigrid::prolong(coarse_ctx->mesh(), fine_ctx->mesh(), xarray, yarray, coarse_ctx->prediction_order);
                    VecRestoreArrayRead(x, &xarray);
                    VecRestoreArray(y, &yarray);

                    /*std::cout << "prolongated vector (marche):" << std::endl;
                    VecView(y, PETSC_VIEWER_STDOUT_(PETSC_COMM_SELF)); std::cout
                    << std::endl;

                    // With matrix
                    std::size_t nf = fine_ctx->mesh().nb_cells();
                    std::size_t nc = coarse_ctx->mesh().nb_cells();

                    int stencil_size = 9; // in 2D!!!!!!

                    Mat P2;
                    MatCreate(PETSC_COMM_SELF, &P2);
                    MatSetSizes(P2, nf, nc, nf, nc);
                    MatSetFromOptions(P2);
                    MatSeqAIJSetPreallocation(P2, stencil_size, NULL);
                    multigrid::set_prolong_matrix(coarse_ctx->mesh(),
                    fine_ctx->mesh(), P2); MatAssemblyBegin(P2,
                    MAT_FINAL_ASSEMBLY); MatAssemblyEnd(P2, MAT_FINAL_ASSEMBLY);

                    std::cout << "prolongated vector with matrix (marche pas):"
                    << std::endl; Vec y2; VecCreateSeq(PETSC_COMM_SELF, nf,
                    &y2); MatMult(P2, x, y2); VecView(y2,
                    PETSC_VIEWER_STDOUT_(PETSC_COMM_SELF)); std::cout <<
                    std::endl;*/
                }

                // std::cout << "prolongated vector:" << std::endl;
                // VecView(y, PETSC_VIEWER_STDOUT_(PETSC_COMM_SELF)); std::cout
                // << std::endl; PetscReal norm; VecNorm(y, NORM_2, &norm);
                // std::cout << "prolongated vector norm:" << norm << std::endl;

                assert(samurai::petsc::check_nan_or_inf(y) && "Nan or Inf after prolongation");
                return 0;
            }

            static PetscErrorCode CreateProlongationMatrix(DM coarse_dm, DM fine_dm, Mat* P, Vec* scaling)
            {
                LevelContext<Dsctzr>* coarse_ctx;
                DMShellGetContext(coarse_dm, &coarse_ctx);
                LevelContext<Dsctzr>* fine_ctx;
                DMShellGetContext(fine_dm, &fine_ctx);

                auto nf = static_cast<PetscInt>(fine_ctx->mesh().nb_cells());
                auto nc = static_cast<PetscInt>(coarse_ctx->mesh().nb_cells());

                int stencil_size = 3; // 1D
                if (std::remove_reference_t<decltype(fine_ctx->mesh())>::dim == 2)
                {
                    stencil_size = 9; // 2D
                }

                MatCreate(PETSC_COMM_SELF, P);
                MatSetSizes(*P, nf, nc, nf, nc);
                MatSetFromOptions(*P);
                MatSeqAIJSetPreallocation(*P, stencil_size, NULL);
                multigrid::set_prolong_matrix(coarse_ctx->mesh(), fine_ctx->mesh(), *P, coarse_ctx->prediction_order);
                MatAssemblyBegin(*P, MAT_FINAL_ASSEMBLY);
                MatAssemblyEnd(*P, MAT_FINAL_ASSEMBLY);

                // MatView(*P, PETSC_VIEWER_STDOUT_(PETSC_COMM_SELF)); std::cout
                // << std::endl;

                *scaling = nullptr; // Why???

                return 0;
            }

            static PetscErrorCode CreateMatFreeRestriction(DM coarse_dm, DM fine_dm, Mat* R)
            {
                LevelContext<Dsctzr>* coarse_ctx;
                DMShellGetContext(coarse_dm, &coarse_ctx);
                LevelContext<Dsctzr>* fine_ctx;
                DMShellGetContext(fine_dm, &fine_ctx);

                auto nf = static_cast<PetscInt>(fine_ctx->mesh().nb_cells());
                auto nc = static_cast<PetscInt>(coarse_ctx->mesh().nb_cells());

                MatCreateShell(PetscObjectComm(reinterpret_cast<PetscObject>(fine_dm)), nc, nf, nc, nf, fine_ctx, R);
                MatShellSetOperation(*R, MATOP_MULT, reinterpret_cast<void (*)(void)>(restriction));
                return 0;
            }

            static PetscErrorCode restriction(Mat R, Vec x, Vec y)
            {
                LevelContext<Dsctzr>* fine_ctx;
                MatShellGetContext(R, &fine_ctx);
                LevelContext<Dsctzr>* coarse_ctx = fine_ctx->coarser;

                // std::cout << "restriction - fine vector:" << std::endl;
                // VecView(x, PETSC_VIEWER_STDOUT_(PETSC_COMM_SELF)); std::cout
                // << std::endl;

                if (coarse_ctx->transfer_ops == TransferOperators::MatrixFree_Fields)
                {
                    Field fine_field("fine_field", fine_ctx->mesh());
                    samurai::petsc::copy(x, fine_field);
                    Field coarse_field = multigrid::restrict(fine_field, coarse_ctx->mesh());
                    samurai::petsc::copy(coarse_field, y);
                }
                else
                {
                    const double* xarray;
                    double* yarray;
                    VecGetArrayRead(x, &xarray);
                    VecGetArray(y, &yarray);
                    multigrid::restrict(fine_ctx->mesh(), coarse_ctx->mesh(), xarray, yarray);
                    VecRestoreArrayRead(x, &xarray);
                    VecRestoreArray(y, &yarray);
                }

                // std::cout << "restriction - restricted vector:" << std::endl;
                // VecView(y, PETSC_VIEWER_STDOUT_(PETSC_COMM_SELF)); std::cout
                // << std::endl; PetscReal norm; VecNorm(y, NORM_2, &norm);
                // std::cout << "restricted vector norm:" << norm << std::endl;

                assert(samurai::petsc::check_nan_or_inf(y) && "Nan or Inf after restriction");
                return 0;
            }

            static PetscErrorCode CreateRestrictionMatrix(DM coarse_dm, DM fine_dm, Mat* R)
            {
                LevelContext<Dsctzr>* coarse_ctx;
                DMShellGetContext(coarse_dm, &coarse_ctx);
                LevelContext<Dsctzr>* fine_ctx;
                DMShellGetContext(fine_dm, &fine_ctx);

                auto nf = static_cast<PetscInt>(fine_ctx->mesh().nb_cells());
                auto nc = static_cast<PetscInt>(coarse_ctx->mesh().nb_cells());

                int stencil_size = 2; // 1D
                if (std::remove_reference_t<decltype(fine_ctx->mesh())>::dim == 2)
                {
                    stencil_size = 4; // 2D
                }
                else if (std::remove_reference_t<decltype(fine_ctx->mesh())>::dim == 3)
                {
                    assert(false); // 3D
                }

                MatCreate(PETSC_COMM_SELF, R);
                MatSetSizes(*R, nc, nf, nc, nf);
                MatSetFromOptions(*R);
                MatSeqAIJSetPreallocation(*R, stencil_size, NULL);
                multigrid::set_restrict_matrix(fine_ctx->mesh(), coarse_ctx->mesh(), *R);
                MatAssemblyBegin(*R, MAT_FINAL_ASSEMBLY);
                MatAssemblyEnd(*R, MAT_FINAL_ASSEMBLY);

                // MatView(*R, PETSC_VIEWER_STDOUT_(PETSC_COMM_SELF));
                // std::cout << std::endl;

                return 0;
            }

            static PetscErrorCode CreateGlobalVector(DM shell, Vec* x)
            {
                // std::cout << "CreateGlobalVector - begin " << std::endl;
                LevelContext<Dsctzr>* ctx;
                DMShellGetContext(shell, &ctx);
                VecCreateSeq(PETSC_COMM_SELF, static_cast<PetscInt>(ctx->mesh().nb_cells()), x);
                VecSetDM(*x, shell);
                // std::cout << "CreateGlobalVector - level " << ctx->level <<
                // std::endl;
                return 0;
            }

            static PetscErrorCode CreateLocalVector(DM shell, Vec* x)
            {
                // std::cout << "CreateLocalVector - begin" << std::endl;
                LevelContext<Dsctzr>* ctx;
                DMShellGetContext(shell, &ctx);
                VecCreateSeq(PETSC_COMM_SELF, static_cast<PetscInt>(ctx->mesh().nb_cells()), x);
                VecSetDM(*x, shell);
                // std::cout << "CreateLocalVector - end" << std::endl;
                return 0;
            }
        };

    } // namespace petsc
} // namespace samurai_new
