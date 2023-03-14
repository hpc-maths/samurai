#pragma once
#include <petsc.h>

namespace samurai 
{ 
    namespace petsc
    {
        class MatrixAssembly
        {
        private:
            bool m_include_bc = true;
            bool m_assemble_proj_pred = true;
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

                MatSeqAIJSetPreallocation(A, PETSC_DEFAULT, sparsity_pattern().data());
                //MatSetOption(A, MAT_NEW_NONZERO_ALLOCATION_ERR, PETSC_FALSE);
            }

            /**
             * @brief Inserts the coefficent into a preallocated matrix and performs the assembly.
            */
            virtual void assemble_matrix(Mat& A)
            {
                assemble_scheme_on_uniform_grid(A);
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

            virtual ~MatrixAssembly() {}

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
             * @brief Sparsity pattern of the matrix.
             * @return vector that stores, for each row index in the matrix, the number of non-zero coefficients.
            */
            virtual std::vector<PetscInt> sparsity_pattern() const = 0;

            /**
             * @brief Is the matrix symmetric positive-definite?
            */
            virtual bool matrix_is_spd() const { return false; }

            /**
             * @brief Inserts coefficients into the matrix.
             * This function defines the scheme on a uniform, Cartesian grid.
            */
            virtual void assemble_scheme_on_uniform_grid(Mat& A) = 0;

            /**
             * @brief Inserts the coefficients into the matrix in order to enforce the boundary conditions.
            */
            virtual void assemble_boundary_conditions(Mat& A) = 0;

            /**
             * @brief Inserts the coefficients corresponding to the projection operator into the matrix.
            */
            virtual void assemble_projection(Mat& A) = 0;

            /**
             * @brief Inserts the coefficients corresponding the prediction operator into the matrix.
            */
            virtual void assemble_prediction(Mat& A) = 0;

            virtual void add_1_on_diag_for_useless_ghosts(Mat& A) = 0;
        };


        enum DirichletEnforcement : int
        {
            Equation,
            Elimination
        };

        /**
         * Useful sizes to define the sparsity pattern of the matrix and perform the preallocation.
        */
        template <PetscInt output_field_size_,
                PetscInt scheme_stencil_size_,
                PetscInt center_index_,
                PetscInt contiguous_indices_start_ = 0,
                PetscInt contiguous_indices_size_ = 0,
                DirichletEnforcement dirichlet_enfcmt_ = Equation>
        struct PetscAssemblyConfig
        {
            static constexpr PetscInt output_field_size = output_field_size_;
            static constexpr PetscInt scheme_stencil_size = scheme_stencil_size_;
            static constexpr PetscInt center_index = center_index_;
            static constexpr PetscInt contiguous_indices_start = contiguous_indices_start_;
            static constexpr PetscInt contiguous_indices_size = contiguous_indices_size_;
            static constexpr DirichletEnforcement dirichlet_enfcmt = dirichlet_enfcmt_;
        };
        
        template<std::size_t dim, std::size_t output_field_size, DirichletEnforcement dirichlet_enfcmt = Equation>
        using starStencilFV = PetscAssemblyConfig
        <
            output_field_size,
            // ----  Stencil size 
            // Cell-centered Finite Volume scheme:
            // center + 1 neighbour in each Cartesian direction (2*dim directions) --> 1+2=3 in 1D
            //                                                                         1+4=5 in 2D
            1 + 2*dim,
            // ---- Index of the stencil center
            // (as defined in star_stencil())
            1, 
            // ---- Start index and size of contiguous cell indices
            // (as defined in star_stencil())
            // Here, [left, center, right].
            0, 3,
            // ---- Method of Dirichlet condition enforcement
            dirichlet_enfcmt
        >;

        template<std::size_t output_field_size, DirichletEnforcement dirichlet_enfcmt = Equation>
        using oneCellStencilFV = PetscAssemblyConfig
        <
            output_field_size,
            // ----  Stencil size 
            // Only one cell:
            1,
            // ---- Index of the stencil center
            // (as defined in center_only_stencil())
            0, 
            // ---- Start index and size of contiguous cell indices
            0, 0,
            // ---- Method of Dirichlet condition enforcement
            dirichlet_enfcmt
        >;

    } // end namespace petsc
} // end namespace samurai