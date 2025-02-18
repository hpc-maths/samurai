#include <assert.h>
#include <iostream>
#include <omp.h>
#include <petsc.h>

static PetscErrorCode PETSC_nonlinear_function(SNES /*snes*/, Vec x, Vec f, void*)
{
    const PetscScalar* x_data;
    VecGetArrayRead(x, &x_data);
    PetscScalar* f_data;
    VecGetArray(f, &f_data);

    auto v  = x_data[0];
    *f_data = v * v * (1 - v);

    VecRestoreArray(f, &f_data);
    VecRestoreArrayRead(x, &x_data);
    return PETSC_SUCCESS;
}

static PetscErrorCode PETSC_jacobian_function(SNES /*snes*/, Vec x, Mat jac, Mat B, void*)
{
    const PetscScalar* x_data;
    VecGetArrayRead(x, &x_data);

    auto v          = x_data[0];
    auto jac_coeffs = 2 * v * (1 - v) - v * v;
    MatSetValue(B, 0, 0, jac_coeffs, INSERT_VALUES);
    MatAssemblyBegin(B, MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd(B, MAT_FINAL_ASSEMBLY);
    if (jac != B)
    {
        MatAssemblyBegin(jac, MAT_FINAL_ASSEMBLY);
        MatAssemblyEnd(jac, MAT_FINAL_ASSEMBLY);
    }

    VecRestoreArrayRead(x, &x_data);
    return PETSC_SUCCESS;
}

int main(int argc, char* argv[])
{
    PetscInitialize(&argc, &argv, 0, nullptr);

    auto n_threads = omp_get_max_threads();

    std::cout << n_threads << " threads" << std::endl;

    // Loop to try and reproduce the error.
    // The error usually occurs in the first two tries (if it occurs at all).
    for (int k = 0; k < 5; ++k)
    {
        std::cout << "try " << k << std::endl;

        // One SNES object and one Jacobian matrix for each thread
        std::vector<SNES> snes_list(n_threads);
        std::vector<Mat> J_list(n_threads);

#pragma omp parallel for
        for (int thread_num = 0; thread_num < n_threads; ++thread_num)
        {
            assert(thread_num == omp_get_thread_num()); // make sure each thread builds its own objects
            SNESCreate(MPI_COMM_SELF, &snes_list[thread_num]);
            MatCreateSeqDense(MPI_COMM_SELF, 1, 1, NULL, &J_list[thread_num]);
        }

// Solve in parallel independent non-linear systems
#pragma omp parallel for
        for (int i = 0; i < 20; ++i)
        {
            std::size_t thread_num = static_cast<std::size_t>(omp_get_thread_num());

            SNES& snes = snes_list[thread_num];
            Mat& J     = J_list[thread_num];

            SNESSetFunction(snes, nullptr, PETSC_nonlinear_function, nullptr);
            SNESSetJacobian(snes, J, J, PETSC_jacobian_function, nullptr);
            SNESSetFromOptions(snes);

            // Right-hand side
            Vec b;
            VecCreateSeq(PETSC_COMM_SELF, 1, &b);
            PetscScalar* b_data;
            VecGetArray(b, &b_data);
            b_data[0] = 1.;
            VecRestoreArray(b, &b_data);

            // Unknown
            Vec x;
            VecCreateSeq(PETSC_COMM_SELF, 1, &x);
            PetscScalar* x_data;
            VecGetArray(x, &x_data);
            x_data[0] = 2.; // initial guess
            VecRestoreArray(x, &x_data);

            // Solve
            SNESSolve(snes, b, x);

            SNESConvergedReason reason_code;
            SNESGetConvergedReason(snes, &reason_code);
            if (reason_code < 0)
            {
                std::cerr << "Divergence of the non-linear solver" << std::endl;
                exit(EXIT_FAILURE);
            }

            VecDestroy(&b);
            VecDestroy(&x);
        }

#pragma omp parallel for
        for (int thread_num = 0; thread_num < n_threads; ++thread_num)
        {
            assert(thread_num == omp_get_thread_num()); // make sure each thread destroys its own objects
            MatDestroy(&J_list[thread_num]);
            SNESDestroy(&snes_list[thread_num]);
        }
    }

    PetscFinalize();
    return 0;
}
