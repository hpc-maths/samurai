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

                // for (std::size_t row = 0; row < nnz.size(); ++row)
                // {
                //     std::cout << "nnz[" << row << "] = " << nnz[row] << std::endl;
                // }

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


        
        namespace detail
        {
            /**
             * Local square matrix to store the coefficients of a vectorial field.
            */
            template<class value_type, std::size_t rows, std::size_t cols=rows>
            struct LocalMatrix
            {
                using Type = xt::xtensor_fixed<value_type, xt::xshape<rows, cols>>;
            };

            /**
             * Template specialization: if size=1, then just a scalar coefficient
            */
            template<class value_type>
            struct LocalMatrix<value_type, 1, 1>
            {
                using Type = value_type;
            };
        }

        template<class matrix_type>
        matrix_type eye()
        {
            static constexpr auto s = typename matrix_type::shape_type();
            return xt::eye(s[0]);
        }

        template<>
        double eye<double>()
        {
            return 1;
        }

        template<class matrix_type>
        matrix_type zeros()
        {
            static constexpr auto s = typename matrix_type::shape_type();
            return xt::zeros(s[0], s[1]);
        }

        template<>
        double zeros<double>()
        {
            return 0;
        }

    } // end namespace petsc
} // end namespace samurai