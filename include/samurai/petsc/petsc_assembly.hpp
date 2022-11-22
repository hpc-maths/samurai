#pragma once
#include <petsc.h>

namespace samurai { namespace petsc
{
    class PetscAssembly
    {
    public:
        /**
         * @brief Performs the memory preallocation of the Petsc matrix.
         * @see assemble_matrix
        */
        void create_matrix(Mat& A)
        {
            auto n = matrix_size();

            MatCreate(PETSC_COMM_SELF, &A);
            MatSetSizes(A, n, n, n, n);
            MatSetFromOptions(A);

            MatSeqAIJSetPreallocation(A, PETSC_DEFAULT, sparsity_pattern().data());
        }

        /**
         * @brief Inserts the coefficent into a preallocated matrix and performs the assembly.
        */
        void assemble_matrix(Mat& A)
        {
            assemble_scheme_on_uniform_grid(A);
            assemble_projection(A);
            assemble_prediction(A);

            PetscBool is_spd = matrix_is_spd() ? PETSC_TRUE : PETSC_FALSE;
            MatSetOption(A, MAT_SPD, is_spd);

            MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY);
            MatAssemblyEnd(A, MAT_FINAL_ASSEMBLY);
        }

        virtual ~PetscAssembly() {}

    private:
        /**
         * @brief Returns the matrix size.
        */
        virtual PetscInt matrix_size() = 0;

        /**
         * @brief Sparsity pattern of the matrix.
         * @return vector that stores, for each row index in the matrix, the number of non-zero coefficients.
        */
        virtual std::vector<PetscInt> sparsity_pattern() = 0;

        /**
         * @brief Is the matrix symmetric positive-definite?
        */
        virtual bool matrix_is_spd() = 0;

        /**
         * @brief Inserts coefficients into the matrix.
         * This function defines the scheme on a uniform, Cartesian grid.
        */
        virtual void assemble_scheme_on_uniform_grid(Mat& A) = 0;

        /**
         * @brief Inserts the coefficients corresponding to the projection operator into the matrix.
        */
        virtual void assemble_projection(Mat& A) = 0;

        /**
         * @brief Inserts the coefficients corresponding the prediction operator into the matrix.
        */
        virtual void assemble_prediction(Mat& A) = 0;
    };


    template <PetscInt scheme_stencil_size_,
              PetscInt proj_stencil_size_,
              PetscInt pred_stencil_size_,
              PetscInt center_index_,
              PetscInt contiguous_indices_start_,
              PetscInt contiguous_indices_size_>
    struct PetscAssemblyConfig
    {
        static constexpr PetscInt scheme_stencil_size = scheme_stencil_size_;
        static constexpr PetscInt proj_stencil_size = proj_stencil_size_;
        static constexpr PetscInt pred_stencil_size = pred_stencil_size_;
        static constexpr PetscInt center_index = center_index_;
        static constexpr PetscInt contiguous_indices_start = contiguous_indices_start_;
        static constexpr PetscInt contiguous_indices_size = contiguous_indices_size_;
    };
}} // end namespace