#pragma once
#include <petsc.h>

namespace samurai
{
    namespace petsc
    {
        class MatrixAssembly
        {
          private:

            bool m_include_bc                       = true;
            bool m_assemble_proj_pred               = true;
            bool m_add_1_on_diag_for_useless_ghosts = true;

          public:

            bool include_bc() const
            {
                return m_include_bc;
            }

            void include_bc_if(bool include)
            {
                m_include_bc = include;
            }

            bool assemble_proj_pred() const
            {
                return m_assemble_proj_pred;
            }

            void assemble_proj_pred_if(bool assemble)
            {
                m_assemble_proj_pred = assemble;
            }

            void add_1_on_diag_for_useless_ghosts_if(bool value)
            {
                m_add_1_on_diag_for_useless_ghosts = value;
            }

            /**
             * @brief Performs the memory preallocation of the Petsc matrix.
             * @see assemble_matrix
             */
            virtual void create_matrix(Mat& A)
            {
                auto m = matrix_rows();
                auto n = matrix_cols();

                MatCreate(PETSC_COMM_SELF, &A);
                MatSetSizes(A, m, n, m, n);
                MatSetFromOptions(A);

                // Number of non-zeros per row. 0 by default.
                std::vector<PetscInt> nnz(static_cast<std::size_t>(m), 0);

                sparsity_pattern_scheme(nnz);
                if (m_include_bc)
                {
                    sparsity_pattern_boundary(nnz);
                }
                if (m_assemble_proj_pred)
                {
                    sparsity_pattern_projection(nnz);
                    sparsity_pattern_prediction(nnz);
                }
                if (m_add_1_on_diag_for_useless_ghosts)
                {
                    sparsity_pattern_useless_ghosts(nnz);
                }


                MatSeqAIJSetPreallocation(A, PETSC_DEFAULT, nnz.data());
                //MatSetOption(A, MAT_NEW_NONZERO_ALLOCATION_ERR, PETSC_FALSE);
            }

            /**
             * @brief Inserts the coefficent into a preallocated matrix and
             * performs the assembly.
             */
            virtual void assemble_matrix(Mat& A)
            {
                assemble_scheme(A);
                if (m_include_bc)
                {
                    assemble_boundary_conditions(A);
                }
                if (m_assemble_proj_pred)
                {
                    assemble_projection(A);
                    assemble_prediction(A);
                }
                if (m_add_1_on_diag_for_useless_ghosts)
                {
                    add_1_on_diag_for_useless_ghosts(A);
                }

                PetscBool is_spd = matrix_is_spd() ? PETSC_TRUE : PETSC_FALSE;
                MatSetOption(A, MAT_SPD, is_spd);

                MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY);
                MatAssemblyEnd(A, MAT_FINAL_ASSEMBLY);
            }

            virtual ~MatrixAssembly()
            {
            }

          protected:

        protected:

            /**
             * @brief Returns the number of matrix rows.
             */
            virtual PetscInt matrix_rows() const = 0;
            /**
             * @brief Returns the number of matrix columns.
             */
            virtual PetscInt matrix_cols() const = 0;


            /**
             * @brief Sets the sparsity pattern of the matrix for the interior of the domain (cells only).
             * @param nnz that stores, for each row index in the matrix, the number of non-zero coefficients.
            */
            virtual void sparsity_pattern_scheme(std::vector<PetscInt>& nnz) const = 0;
            /**
             * @brief Sets the sparsity pattern of the matrix for the boundary conditions.
             * @param nnz that stores, for each row index in the matrix, the number of non-zero coefficients.
            */
            virtual void sparsity_pattern_boundary(std::vector<PetscInt>& nnz) const = 0;
            /**
             * @brief Sets the sparsity pattern of the matrix for the projection ghosts.
             * @param nnz that stores, for each row index in the matrix, the number of non-zero coefficients.
            */
            virtual void sparsity_pattern_projection(std::vector<PetscInt>& nnz) const = 0;
            /**
             * @brief Sets the sparsity pattern of the matrix for the prediction ghosts.
             * @param nnz that stores, for each row index in the matrix, the number of non-zero coefficients.
            */
            virtual void sparsity_pattern_prediction(std::vector<PetscInt>& nnz) const = 0;

            /**
             * @brief Is the matrix symmetric positive-definite?
             */
            virtual bool matrix_is_spd() const
            {
                return false;
            }

            /**
             * @brief Inserts coefficients into the matrix.
             * This function defines the scheme in the inside of the domain.
            */
            virtual void assemble_scheme(Mat& A) = 0;

            /**
             * @brief Inserts the coefficients into the matrix in order to
             * enforce the boundary conditions.
             */
            virtual void assemble_boundary_conditions(Mat& A) = 0;

            /**
             * @brief Inserts the coefficients corresponding to the projection
             * operator into the matrix.
             */
            virtual void assemble_projection(Mat& A) = 0;

            /**
             * @brief Inserts the coefficients corresponding the prediction
             * operator into the matrix.
             */
            virtual void assemble_prediction(Mat& A) = 0;

            virtual void add_1_on_diag_for_useless_ghosts(Mat& A) = 0;

            virtual void sparsity_pattern_useless_ghosts(std::vector<PetscInt>& nnz)
            {
                for (auto& row_nnz : nnz)
                {
                    if (row_nnz == 0)
                    {
                        row_nnz = 1;
                    }
                }
            }
        };

        enum DirichletEnforcement : int
        {
            Equation,
            Elimination
        };

        /**
         * Useful sizes to define the sparsity pattern of the matrix and perform
         * the preallocation.
         */
        template <PetscInt output_field_size_,
                  PetscInt scheme_stencil_size_,
                  PetscInt center_index_,
                  PetscInt contiguous_indices_start_     = 0,
                  PetscInt contiguous_indices_size_      = 0,
                  DirichletEnforcement dirichlet_enfcmt_ = Equation>
        struct PetscAssemblyConfig
        {
            static constexpr PetscInt output_field_size            = output_field_size_;
            static constexpr PetscInt scheme_stencil_size          = scheme_stencil_size_;
            static constexpr PetscInt center_index                 = center_index_;
            static constexpr PetscInt contiguous_indices_start     = contiguous_indices_start_;
            static constexpr PetscInt contiguous_indices_size      = contiguous_indices_size_;
            static constexpr DirichletEnforcement dirichlet_enfcmt = dirichlet_enfcmt_;
        };

        template <std::size_t dim, std::size_t output_field_size, std::size_t neighbourhood_width = 1, DirichletEnforcement dirichlet_enfcmt = Equation>
        using StarStencilFV = PetscAssemblyConfig<output_field_size,
                                                  // ----  Stencil size
                                                  // Cell-centered Finite Volume scheme:
                                                  // center + 'neighbourhood_width' neighbours in each Cartesian
                                                  // direction (2*dim directions) --> 1+2=3 in 1D
                                                  //                                                                                              1+4=5
                                                  //                                                                                              in
                                                  //                                                                                              2D
                                                  1 + 2 * dim * neighbourhood_width,
                                                  // ---- Index of the stencil center
                                                  // (as defined in star_stencil())
                                                  neighbourhood_width,
                                                  // ---- Start index and size of contiguous cell indices
                                                  // (as defined in star_stencil())
                                                  0,
                                                  1 + 2 * neighbourhood_width,
                                                  // ---- Method of Dirichlet condition enforcement
                                                  dirichlet_enfcmt>;

        template <std::size_t output_field_size, DirichletEnforcement dirichlet_enfcmt = Equation>
        using OneCellStencilFV = PetscAssemblyConfig<output_field_size,
                                                     // ----  Stencil size
                                                     // Only one cell:
                                                     1,
                                                     // ---- Index of the stencil center
                                                     // (as defined in center_only_stencil())
                                                     0,
                                                     // ---- Start index and size of contiguous cell indices
                                                     0,
                                                     0,
                                                     // ---- Method of Dirichlet condition enforcement
                                                     dirichlet_enfcmt>;

    } // end namespace petsc
} // end namespace samurai