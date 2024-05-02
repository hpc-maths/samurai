#pragma once
#include <petsc.h>

namespace samurai
{
    namespace petsc
    {
        class MatrixAssembly
        {
          private:

            bool m_is_deleted  = false;
            std::string m_name = "(unnamed)";

            bool m_include_bc                       = true;
            bool m_assemble_proj_pred               = true;
            bool m_set_1_on_diag_for_useless_ghosts = true;

            InsertMode m_current_insert_mode = INSERT_VALUES;

          protected:

            bool m_is_block             = false; // is a block in a monolithic block matrix
            bool m_fit_block_dimensions = false;
            PetscInt m_row_shift        = 0;
            PetscInt m_col_shift        = 0;
            PetscInt m_rows             = 0;
            PetscInt m_cols             = 0;

          public:

            std::string name() const
            {
                return m_name;
            }

            void set_name(const std::string& name)
            {
                m_name = name;
            }

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

            bool must_set_1_on_diag_for_useless_ghosts() const
            {
                return m_set_1_on_diag_for_useless_ghosts;
            }

            void set_1_on_diag_for_useless_ghosts_if(bool value)
            {
                m_set_1_on_diag_for_useless_ghosts = value;
            }

            virtual void set_is_block(bool is_block)
            {
                m_is_block = is_block;
            }

            bool is_block() const
            {
                return m_is_block;
            }

            void set_fit_block_dimensions(bool value)
            {
                m_fit_block_dimensions = value;
            }

            bool fit_block_dimensions() const
            {
                return m_fit_block_dimensions;
            }

            template <class int_type>
            void set_row_shift(int_type row_shift)
            {
                m_row_shift = static_cast<PetscInt>(row_shift);
            }

            template <class int_type>
            void set_col_shift(int_type col_shift)
            {
                m_col_shift = static_cast<PetscInt>(col_shift);
            }

            PetscInt row_shift() const
            {
                return m_row_shift;
            }

            PetscInt col_shift() const
            {
                return m_col_shift;
            }

            /**
             * @brief Returns the number of matrix rows.
             */
            virtual PetscInt matrix_rows() const
            {
                return m_rows;
            }

            /**
             * @brief Returns the number of matrix columns.
             */
            virtual PetscInt matrix_cols() const
            {
                return m_cols;
            }

            void set_matrix_rows(PetscInt rows)
            {
                m_rows = rows;
            }

            void set_matrix_cols(PetscInt cols)
            {
                m_cols = cols;
            }

            InsertMode current_insert_mode() const
            {
                return m_current_insert_mode;
            }

            virtual void set_current_insert_mode(InsertMode insert_mode)
            {
                m_current_insert_mode = insert_mode;
            }

            /**
             * @brief Performs the memory preallocation of the Petsc matrix.
             * @see assemble_matrix
             */
            virtual void create_matrix(Mat& A)
            {
                reset();
                auto m = matrix_rows();
                auto n = matrix_cols();

                MatCreate(PETSC_COMM_SELF, &A);
                MatSetSizes(A, m, n, m, n);
                MatSetFromOptions(A);
                PetscObjectSetName(reinterpret_cast<PetscObject>(A), m_name.c_str());

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
                if (m_set_1_on_diag_for_useless_ghosts)
                {
                    sparsity_pattern_useless_ghosts(nnz);
                }

                // for (std::size_t row = 0; row < nnz.size(); ++row)
                // {
                //     std::cout << "nnz[" << row << "] = " << nnz[row] << std::endl;
                // }
                if (!m_is_block)
                {
                    MatSeqAIJSetPreallocation(A, PETSC_DEFAULT, nnz.data());
                }
                // MatSetOption(A, MAT_NEW_NONZERO_ALLOCATION_ERR, PETSC_FALSE);
            }

            /**
             * @brief Inserts the coefficent into a preallocated matrix and
             * performs the assembly.
             */
            virtual void assemble_matrix(Mat& A, bool final_assembly = true)
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
                if (m_set_1_on_diag_for_useless_ghosts)
                {
                    set_1_on_diag_for_useless_ghosts(A);
                }

                if (!m_is_block)
                {
                    PetscBool is_symmetric = matrix_is_symmetric() ? PETSC_TRUE : PETSC_FALSE;
                    MatSetOption(A, MAT_SYMMETRIC, is_symmetric);

                    PetscBool is_spd = matrix_is_spd() ? PETSC_TRUE : PETSC_FALSE;
                    MatSetOption(A, MAT_SPD, is_spd);

                    if (final_assembly)
                    {
                        MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY);
                        MatAssemblyEnd(A, MAT_FINAL_ASSEMBLY);
                    }
                }
            }

            virtual ~MatrixAssembly()
            {
                // std::cout << "Destruction of '" << name() << "'" << std::endl;
                m_is_deleted = true;
            }

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

            virtual void set_1_on_diag_for_useless_ghosts(Mat& A) = 0;

            virtual void sparsity_pattern_useless_ghosts(std::vector<PetscInt>& nnz)
            {
                for (std::size_t row = static_cast<std::size_t>(m_row_shift); row < static_cast<std::size_t>(m_row_shift + matrix_rows());
                     ++row)
                {
                    if (nnz[row] == 0)
                    {
                        nnz[row] = 1;
                    }
                }
            }

            /**
             * @brief Is the matrix symmetric?
             */
            virtual bool matrix_is_symmetric() const
            {
                return false;
            }

            /**
             * @brief Is the matrix symmetric positive-definite?
             */
            virtual bool matrix_is_spd() const
            {
                return false;
            }

            virtual void reset()
            {
            }
        };

        template <class Scheme, class check = void>
        class Assembly : public MatrixAssembly
        {
            template <class>
            static constexpr bool dependent_false = false;

            static_assert(
                dependent_false<typename Scheme::cfg_t>,
                "Either the required file has not been included, or the Assembly class has not been specialized for this type of scheme.");
        };

        // If Scheme already implements MatrixAssembly, then Assembly<Scheme> is defined as Scheme itself.
        // Since it can't be done by a 'using' instruction (-> partial specialization error),
        // we use a trick: Assembly<Scheme> inherits from Scheme.
        template <class Scheme>
        class Assembly<Scheme, std::enable_if_t<std::is_base_of_v<MatrixAssembly, Scheme>>> : public Scheme
        {
        };

        template <class Scheme>
        auto make_assembly(const Scheme& s)
        {
            if constexpr (std::is_base_of_v<MatrixAssembly, Scheme>)
            {
                return *reinterpret_cast<const Assembly<Scheme>*>(&s);
            }
            else
            {
                return Assembly<Scheme>(s);
            }
        }

    } // end namespace petsc
} // end namespace samurai
