#pragma once
#include "intergrid_operators.hpp"
#include "utils.hpp"

// 0: P mat-free, R mat-free (via Fields)
// 1: P mat-free, R mat-free (via double*)
// 2: P assembled, R = P^T
// 3: P assembled, R = assembled
#define OPT_PROLONGATION 0

namespace samurai_new
{
    namespace petsc
    {

        template<class Dsctzr>
        class PetscDM
        {
        public:    
            using Mesh = typename Dsctzr::Mesh;
            using Field = typename Dsctzr::field_t;

            static DM Create(MPI_Comm comm, Dsctzr& discretizer, Mesh& mesh)
            {
                auto fine_ctx = new LevelCtx(discretizer, mesh);
                return Create(comm, *fine_ctx);
            }

        private:
            static DM Create(MPI_Comm comm, LevelCtx<Dsctzr>& ctx)
            {
                DM shell;
                DMShellCreate(comm, &shell);
                DMShellSetContext(shell, &ctx);
                DMShellSetCreateMatrix(shell, CreateMatrix);
                DMShellSetCreateGlobalVector(shell, CreateGlobalVector);
                DMShellSetCreateLocalVector(shell, CreateLocalVector);
                DMShellSetCoarsen(shell, Coarsen);
                if (OPT_PROLONGATION == 0 || OPT_PROLONGATION == 1)
                {
                    DMShellSetCreateInterpolation(shell, CreateMatFreeProlongation);
                    DMShellSetCreateRestriction(shell, CreateMatFreeRestriction);
                }
                else if (OPT_PROLONGATION == 2)
                {
                    DMShellSetCreateInterpolation(shell, CreateProlongationMatrix);
                }
                else if (OPT_PROLONGATION == 3)
                {
                    DMShellSetCreateInterpolation(shell, CreateProlongationMatrix);
                    DMShellSetCreateRestriction(shell, CreateRestrictionMatrix);
                }
                //DMShellSetDestroyContext(shell, Destroy);
                return shell;
            }

            static PetscErrorCode Coarsen(DM fine, MPI_Comm comm, DM *coarse) 
            {
                //std::cout << "Coarsen - begin" << std::endl;
                LevelCtx<Dsctzr>* fine_ctx;
                DMShellGetContext(fine, &fine_ctx);

                LevelCtx<Dsctzr>* coarse_ctx = new LevelCtx(*fine_ctx);

                /*std::cout << "fine_mesh:" << std::endl << fine_ctx->mesh() << std::endl << std::endl;
                std::cout << "coarse_mesh:" << std::endl << coarse_ctx->mesh() << std::endl;
                samurai::save("coarse_mesh", coarse_ctx->mesh());*/

                *coarse = Create(PetscObjectComm((PetscObject)fine), *coarse_ctx);
                //std::cout << "Coarsen (create level " << coarse_ctx->level << ")" << std::endl;
                return 0;
            }
            
            /*static PetscErrorCode Destroy(void *ctx) 
            {
                delete ctx;
                return 0;
            }*/

            static PetscErrorCode CreateMatrix(DM shell, Mat *A) 
            {
                LevelCtx<Dsctzr>* ctx;
                DMShellGetContext(shell, &ctx);
                
                ctx->discretizer().create_matrix(*A);
                //std::cout << "CreateMatrix - level " << ctx->level << std::endl;

                MatSetDM(*A, shell); // Why???
                return 0;
            }

        public:
            static PetscErrorCode ComputeMatrix(KSP ksp, Mat J, Mat jac, void *dummy_ctx) 
            {
                DM shell;
                KSPGetDM(ksp, &shell);
                LevelCtx<Dsctzr>* ctx;
                DMShellGetContext(shell, &ctx);

                ctx->discretizer().assemble_matrix(jac);
                MatSetOption(jac, MAT_SPD, PETSC_TRUE);

                // MatView(jac, PETSC_VIEWER_STDOUT_(PETSC_COMM_SELF));
                // std::cout << std::endl;
                // std::cout << "ComputeMatrix - end level " << ctx->level << std::endl;
                return 0;
            }

        private:
            static PetscErrorCode CreateMatFreeProlongation(DM coarse_dm, DM fine_dm, Mat *P, Vec *scaling) 
            {
                LevelCtx<Dsctzr>* coarse_ctx;
                DMShellGetContext(coarse_dm, &coarse_ctx);
                LevelCtx<Dsctzr>* fine_ctx;
                DMShellGetContext(fine_dm, &fine_ctx);

                std::size_t nf = fine_ctx->mesh().nb_cells();
                std::size_t nc = coarse_ctx->mesh().nb_cells();

                MatCreateShell(PetscObjectComm((PetscObject)fine_dm), nf, nc, nf, nc, coarse_ctx, P);
                MatShellSetOperation(*P, MATOP_MULT, (void(*)(void))prolongation);

                *scaling = nullptr; // Why???

                return 0;
            }

            static PetscErrorCode prolongation(Mat P, Vec x, Vec y)
            {
                LevelCtx<Dsctzr>* coarse_ctx;
                LevelCtx<Dsctzr>* fine_ctx;
                MatShellGetContext(P, &coarse_ctx);
                fine_ctx = coarse_ctx->finer;

                // std::cout << "coarse vector x:" << std::endl;
                // VecView(x, PETSC_VIEWER_STDOUT_(PETSC_COMM_SELF));
                // std::cout << std::endl;

                if (OPT_PROLONGATION == 0)
                {
                    Field coarse_field("coarse_field", coarse_ctx->mesh());
                    copy(x, coarse_field);
                    Field fine_field = multigrid::prolong(coarse_field, fine_ctx->mesh());
                    copy(fine_field, y);

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
                    multigrid::prolong(coarse_ctx->mesh(), fine_ctx->mesh(), xarray, yarray);
                    VecRestoreArrayRead(x, &xarray);
                    VecRestoreArray(y, &yarray);

                    // std::cout << "prolongated vector (marche):" << std::endl;
                    // VecView(y, PETSC_VIEWER_STDOUT_(PETSC_COMM_SELF));
                    // std::cout << std::endl;




                    /*// With matrix
                    std::size_t nf = fine_ctx->mesh().nb_cells();
                    std::size_t nc = coarse_ctx->mesh().nb_cells();

                    int stencil_size = 3; // in 1D!!!!!!

                    Mat P2;
                    MatCreate(PETSC_COMM_SELF, &P2);
                    MatSetSizes(P2, nf, nc, nf, nc);
                    MatSetFromOptions(P2);
                    MatSeqAIJSetPreallocation(P2, stencil_size, NULL);
                    PetscMultigrid<Dsctzr>::set_prolong_matrix(*coarse_ctx, *fine_ctx, P2);
                    MatAssemblyBegin(P2, MAT_FINAL_ASSEMBLY);
                    MatAssemblyEnd(P2, MAT_FINAL_ASSEMBLY);

                    
                    std::cout << "prolongated vector with matrix (marche pas):" << std::endl;
                    Vec y2;
                    VecCreateSeq(PETSC_COMM_SELF, nf, &y2);
                    MatMult(P2, x, y2);
                    VecView(y2, PETSC_VIEWER_STDOUT_(PETSC_COMM_SELF));
                    std::cout << std::endl;*/
                }

                // std::cout << "prolongated vector:" << std::endl;
                // VecView(y, PETSC_VIEWER_STDOUT_(PETSC_COMM_SELF));
                // std::cout << std::endl;
                assert(check_nan_or_inf(y) && "Nan or Inf after prolongation");
                return 0;
            }

            static PetscErrorCode CreateProlongationMatrix(DM coarse_dm, DM fine_dm, Mat *P, Vec *scaling) 
            {
                LevelCtx<Dsctzr>* coarse_ctx;
                DMShellGetContext(coarse_dm, &coarse_ctx);
                LevelCtx<Dsctzr>* fine_ctx;
                DMShellGetContext(fine_dm, &fine_ctx);

                std::size_t nf = fine_ctx->mesh().nb_cells();
                std::size_t nc = coarse_ctx->mesh().nb_cells();

                int stencil_size = 3; // in 1D!!!!!!

                MatCreate(PETSC_COMM_SELF, P);
                MatSetSizes(*P, nf, nc, nf, nc);
                MatSetFromOptions(*P);
                MatSeqAIJSetPreallocation(*P, stencil_size, NULL);
                multigrid::set_prolong_matrix(coarse_ctx->mesh(), fine_ctx->mesh(), *P);
                MatAssemblyBegin(*P, MAT_FINAL_ASSEMBLY);
                MatAssemblyEnd(*P, MAT_FINAL_ASSEMBLY);

                //MatView(*P, PETSC_VIEWER_STDOUT_(PETSC_COMM_SELF)); std::cout << std::endl;

                *scaling = nullptr; // Why???

                return 0;
            }

            static PetscErrorCode CreateMatFreeRestriction(DM coarse_dm, DM fine_dm, Mat *R) 
            {
                LevelCtx<Dsctzr>* coarse_ctx;
                DMShellGetContext(coarse_dm, &coarse_ctx);
                LevelCtx<Dsctzr>* fine_ctx;
                DMShellGetContext(fine_dm, &fine_ctx);

                std::size_t nf = fine_ctx->mesh().nb_cells();
                std::size_t nc = coarse_ctx->mesh().nb_cells();

                MatCreateShell(PetscObjectComm((PetscObject)fine_dm), nc, nf, nc, nf, fine_ctx, R);
                MatShellSetOperation(*R, MATOP_MULT, (void(*)(void))restriction);
                return 0;
            }

            static PetscErrorCode restriction(Mat R, Vec x, Vec y)
            {
                LevelCtx<Dsctzr>* fine_ctx;
                MatShellGetContext(R, &fine_ctx);
                LevelCtx<Dsctzr>* coarse_ctx = fine_ctx->coarser;

                //std::cout << "restriction - c=" << coarse_ctx->level << " f=" << fine_ctx->level << std::endl;

                // std::cout << "restriction - fine vector:" << std::endl;
                // VecView(x, PETSC_VIEWER_STDOUT_(PETSC_COMM_SELF));
                // std::cout << std::endl;

                /*std::cout << "fine mesh:" << std::endl;
                print_mesh(fine_ctx->mesh());

                std::cout << std::endl;
                std::cout << "coarse mesh:" << std::endl;
                print_mesh(coarse_ctx->mesh());*/
                if (OPT_PROLONGATION == 0)
                {
                    Field fine_field("fine_field", fine_ctx->mesh());
                    copy(x, fine_field);
                    Field coarse_field = multigrid::restrict(fine_field, coarse_ctx->mesh());
                    copy(coarse_field, y);
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
                // VecView(y, PETSC_VIEWER_STDOUT_(PETSC_COMM_SELF));
                // std::cout << std::endl;
                
                assert(check_nan_or_inf(y) && "Nan or Inf after restriction");
                return 0;
            }

            static PetscErrorCode CreateRestrictionMatrix(DM coarse_dm, DM fine_dm, Mat *R) 
            {
                LevelCtx<Dsctzr>* coarse_ctx;
                DMShellGetContext(coarse_dm, &coarse_ctx);
                LevelCtx<Dsctzr>* fine_ctx;
                DMShellGetContext(fine_dm, &fine_ctx);

                std::size_t nf = fine_ctx->mesh().nb_cells();
                std::size_t nc = coarse_ctx->mesh().nb_cells();

                int stencil_size = 2; // in 1D!!!!!!

                MatCreate(PETSC_COMM_SELF, R);
                MatSetSizes(*R, nc, nf, nc, nf);
                MatSetFromOptions(*R);
                MatSeqAIJSetPreallocation(*R, stencil_size, NULL);
                multigrid::set_restrict_matrix(fine_ctx->mesh(), coarse_ctx->mesh(), *R);
                MatAssemblyBegin(*R, MAT_FINAL_ASSEMBLY);
                MatAssemblyEnd(*R, MAT_FINAL_ASSEMBLY);

                //MatView(*R, PETSC_VIEWER_STDOUT_(PETSC_COMM_SELF));
                //std::cout << std::endl;

                return 0;
            }

            static PetscErrorCode CreateGlobalVector(DM shell, Vec *x) 
            {
                //std::cout << "CreateGlobalVector - begin " << std::endl;
                LevelCtx<Dsctzr>* ctx;
                DMShellGetContext(shell, &ctx);
                VecCreateSeq(PETSC_COMM_SELF, ctx->mesh().nb_cells(), x);
                VecSetDM(*x, shell);
                //std::cout << "CreateGlobalVector - level " << ctx->level << std::endl;
                return 0;
            }
            static PetscErrorCode CreateLocalVector(DM shell, Vec *x) 
            {
                //std::cout << "CreateLocalVector - begin" << std::endl;
                LevelCtx<Dsctzr>* ctx;
                DMShellGetContext(shell, &ctx);
                VecCreateSeq(PETSC_COMM_SELF, ctx->mesh().nb_cells(), x);
                VecSetDM(*x, shell);
                //std::cout << "CreateLocalVector - end" << std::endl;
                return 0;
            }
        };

    } // namespace petsc
} // namespace samurai_new