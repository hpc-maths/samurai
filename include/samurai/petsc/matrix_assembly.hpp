#pragma once
#include "../timers.hpp"
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

            bool m_include_bc                              = true;
            bool m_assemble_proj_pred                      = true;
            bool m_insert_value_on_diag_for_useless_ghosts = true;
            PetscScalar m_diag_value_for_useless_ghosts    = 1;

            InsertMode m_current_insert_mode = INSERT_VALUES;

          protected:

            bool m_is_block             = false; // is a block in a monolithic block matrix
            bool m_fit_block_dimensions = false; // computes dimensions according to the block's position
            PetscInt m_row_shift        = 0;
            PetscInt m_col_shift        = 0;
            PetscInt m_rows             = 0;
            PetscInt m_cols             = 0;

#ifdef SAMURAI_WITH_MPI
            // Petsc takes a reference to the nnz vectors, so we must keep them alive as long as the matrix is alive
            std::vector<PetscInt> m_d_nnz; // number of non-zeros in the diagonal part of the local submatrix
            std::vector<PetscInt> m_o_nnz; // number of non-zeros in the off-diagonal part of the local submatrix

            ISLocalToGlobalMapping m_local_to_global_rows = nullptr;
            ISLocalToGlobalMapping m_local_to_global_cols = nullptr;
#endif

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

            void include_bc(bool include)
            {
                m_include_bc = include;
            }

            bool assemble_proj_pred() const
            {
                return m_assemble_proj_pred;
            }

            void assemble_proj_pred(bool assemble)
            {
                m_assemble_proj_pred = assemble;
            }

            bool must_insert_value_on_diag_for_useless_ghosts() const
            {
                return m_insert_value_on_diag_for_useless_ghosts;
            }

            void must_insert_value_on_diag_for_useless_ghosts(bool value)
            {
                m_insert_value_on_diag_for_useless_ghosts = value;
            }

            virtual void set_diag_value_for_useless_ghosts(PetscScalar value)
            {
                m_diag_value_for_useless_ghosts = value;
            }

            auto diag_value_for_useless_ghosts() const
            {
                return m_diag_value_for_useless_ghosts;
            }

            virtual void is_block(bool is_block)
            {
                m_is_block = is_block;
            }

            bool is_block() const
            {
                return m_is_block;
            }

            void fit_block_dimensions(bool value)
            {
                m_fit_block_dimensions = value;
            }

            bool fit_block_dimensions() const
            {
                return m_fit_block_dimensions;
            }

            virtual void set_row_shift(PetscInt shift)
            {
                m_row_shift = shift;
            }

            virtual void set_col_shift(PetscInt shift)
            {
                m_col_shift = shift;
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
                times::timers.start("matrix assembly");

                reset();

                //-----------------//
                // Matrix creation //
                //-----------------//

                auto m = matrix_rows();
                auto n = matrix_cols();

                MatCreate(PETSC_COMM_WORLD, &A);
#ifdef SAMURAI_WITH_MPI
                MatSetType(A, MATMPIAIJ);
#else
                MatSetType(A, MATSEQAIJ);
#endif

                PetscInt global_nrows = PETSC_DETERMINE;
                PetscInt global_ncols = PETSC_DETERMINE;
                MatSetSizes(A, m, n, global_nrows, global_ncols);
                MatSetFromOptions(A);
                PetscObjectSetName(reinterpret_cast<PetscObject>(A), m_name.c_str());

#ifdef SAMURAI_WITH_MPI
                // Sets the local to global mapping for the rows and columns, which allows to use the local numbering when inserting
                // values into the matrix.
                MatSetLocalToGlobalMapping(A, m_local_to_global_rows, m_local_to_global_cols);
#endif
                //---------------------------------------//
                // Preallocation of the non-zero entries //
                //---------------------------------------//

#ifdef SAMURAI_WITH_MPI
                mpi::communicator world;

                // Number of non-zeros per row in the diagonal and off-diagonal part of the local submatrix. 0 by default.
                auto& d_nnz = m_d_nnz;
                auto& o_nnz = m_o_nnz;
                d_nnz.resize(static_cast<std::size_t>(m));
                o_nnz.resize(static_cast<std::size_t>(m));
                std::fill(d_nnz.begin(), d_nnz.end(), 0);
                std::fill(o_nnz.begin(), o_nnz.end(), 0);

                if (world.rank() == 1)
                {
                    sleep(1); // to avoid jumbled output
                }

                std::cout << "\n\t> [" << world.rank() << "] sparsity_pattern_scheme" << std::endl;
                sparsity_pattern_scheme(d_nnz, o_nnz);
                // world.barrier(); // TO REMOVE

                if (m_include_bc)
                {
                    std::cout << "\n\t> [" << world.rank() << "] sparsity_pattern_boundary" << std::endl;
                    sparsity_pattern_boundary(d_nnz, o_nnz);
                    // world.barrier(); // TO REMOVE
                }
                if (m_assemble_proj_pred)
                {
                    std::cout << "\n\t> [" << world.rank() << "] sparsity_pattern_projection" << std::endl;
                    sparsity_pattern_projection(d_nnz, o_nnz);
                    world.barrier(); // TO REMOVE
                    std::cout << "\n\t> [" << world.rank() << "] sparsity_pattern_prediction" << std::endl;
                    sparsity_pattern_prediction(d_nnz, o_nnz);
                    // world.barrier(); // TO REMOVE
                }
                if (m_insert_value_on_diag_for_useless_ghosts)
                {
                    std::cout << "\n\t> [" << world.rank() << "] sparsity_pattern_useless_ghosts" << std::endl;
                    sparsity_pattern_useless_ghosts(d_nnz);
                    // world.barrier(); // TO REMOVE
                }

                if (!m_is_block)
                {
                    // if (world.rank() == 1)
                    // {
                    //     sleep(1); // to avoid jumbled output
                    // }

                    // std::cout << "\n\t> [" << world.rank() << "] Preallocation of" << std::endl;
                    // int sum_nnz = 0;
                    // for (std::size_t row = 0; row < static_cast<std::size_t>(m); ++row)
                    // {
                    //     std::cout << "[G"
                    //                  "] d_nnz[L"
                    //               << row << "] = " << d_nnz[row] << ", o_nnz[L" << row << "] = " << o_nnz[row] << std::endl;
                    //     sum_nnz += d_nnz[row] + o_nnz[row];
                    // }
                    // std::cout << "Total number of non-zeros on rank " << world.rank() << ": " << sum_nnz << std::endl;
                    // std::cout << std::endl;

                    // sleep(10); // to avoid jumbled output
                    // world.barrier(); // TO REMOVE

                    std::cout << "\n\t> [" << world.rank() << "] MatMPIAIJSetPreallocation" << std::endl;
                    MatMPIAIJSetPreallocation(A, PETSC_DEFAULT, d_nnz.data(), PETSC_DEFAULT, o_nnz.data());
                    std::cout << "\n\t> [" << world.rank() << "] MatMPIAIJSetPreallocation <done>" << std::endl;

                    std::cout << "\n\t> [" << world.rank() << "] create_matrix done" << std::endl;
                    // world.barrier(); // TO REMOVE
                }
#else
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
                if (m_insert_value_on_diag_for_useless_ghosts)
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
#endif
                // MatSetOption(A, MAT_NEW_NONZERO_ALLOCATION_ERR, PETSC_FALSE);
                times::timers.stop("matrix assembly");
            }

            /**
             * @brief Inserts the coefficent into a preallocated matrix and
             * performs the assembly.
             */
            virtual void assemble_matrix(Mat& A, bool final_assembly = true)
            {
                times::timers.start("matrix assembly");

                mpi::communicator world;

                std::cout << "\n\t> [" << world.rank() << "] start assemble_matrix" << std::endl;

                if (world.rank() == 1)
                {
                    sleep(1);
                }
                std::cout << "\n\t> [" << world.rank() << "] assemble_scheme" << std::endl;
                assemble_scheme(A);

                // world.barrier(); // TO REMOVE

                if (m_include_bc)
                {
                    if (world.rank() == 1)
                    {
                        sleep(1);
                    }
                    std::cout << "\n\t> [" << world.rank() << "] assemble_boundary_conditions" << std::endl;
                    assemble_boundary_conditions(A);

                    // world.barrier(); // TO REMOVE
                }
                if (m_assemble_proj_pred)
                {
                    if (world.rank() == 1)
                    {
                        sleep(1);
                    }
                    std::cout << "\n\t> [" << world.rank() << "] assemble_projection" << std::endl;
                    assemble_projection(A);
                    // world.barrier(); // TO REMOVE

                    if (world.rank() == 1)
                    {
                        sleep(1);
                    }
                    std::cout << "\n\t> [" << world.rank() << "] assemble_prediction" << std::endl;
                    assemble_prediction(A);
                    // world.barrier(); // TO REMOVE
                }
                if (m_insert_value_on_diag_for_useless_ghosts)
                {
                    if (world.rank() == 1)
                    {
                        sleep(1);
                    }
                    std::cout << "\n\t> [" << world.rank() << "] insert_value_on_diag_for_useless_ghosts" << std::endl;
                    insert_value_on_diag_for_useless_ghosts(A);
                    // world.barrier(); // TO REMOVE
                }

                if (!m_is_block)
                {
                    PetscBool is_symmetric = matrix_is_symmetric() ? PETSC_TRUE : PETSC_FALSE;
                    MatSetOption(A, MAT_SYMMETRIC, is_symmetric);

                    PetscBool is_spd = matrix_is_spd() ? PETSC_TRUE : PETSC_FALSE;
                    MatSetOption(A, MAT_SPD, is_spd);

                    if (final_assembly)
                    {
                        std::cout << "\n\t> [" << world.rank() << "] ASSEMBLY" << std::endl;
                        MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY);
                        MatAssemblyEnd(A, MAT_FINAL_ASSEMBLY);
                    }
                }
                times::timers.stop("matrix assembly");
            }

            virtual ~MatrixAssembly()
            {
                // std::cout << "Destruction of '" << name() << "'" << std::endl;
                m_is_deleted = true;
                if (m_local_to_global_rows)
                {
                    ISLocalToGlobalMappingDestroy(&m_local_to_global_rows);
                }
                if (m_local_to_global_cols != m_local_to_global_rows && m_local_to_global_cols)
                {
                    ISLocalToGlobalMappingDestroy(&m_local_to_global_cols);
                }
            }

#ifdef SAMURAI_WITH_MPI
            /**
             * @brief Sets the sparsity pattern of the matrix for the interior of the domain (cells only).
             * @param d_nnz that stores, for each row index, the number of non-zero coefficients in the diagonal (local-local) block.
             * @param o_nnz that stores, for each row index, the number of non-zero coefficients in the off-diagonal (local-neighb.) block.
             */
            virtual void sparsity_pattern_scheme(std::vector<PetscInt>& d_nnz, std::vector<PetscInt>& o_nnz) const = 0;
            /**
             * @brief Sets the sparsity pattern of the matrix for the boundary conditions.
             * @param d_nnz that stores, for each row index, the number of non-zero coefficients in the diagonal (local-local) block.
             * @param o_nnz that stores, for each row index, the number of non-zero coefficients in the off-diagonal (local-neighb.) block.
             */
            virtual void sparsity_pattern_boundary(std::vector<PetscInt>& d_nnz, std::vector<PetscInt>& o_nnz) const = 0;
            /**
             * @brief Sets the sparsity pattern of the matrix for the projection ghosts.
             * @param d_nnz that stores, for each row index, the number of non-zero coefficients in the diagonal (local-local) block.
             * @param o_nnz that stores, for each row index, the number of non-zero coefficients in the off-diagonal (local-neighb.) block.
             */
            virtual void sparsity_pattern_projection(std::vector<PetscInt>& d_nnz, std::vector<PetscInt>& o_nnz) const = 0;
            /**
             * @brief Sets the sparsity pattern of the matrix for the prediction ghosts.
             * @param d_nnz that stores, for each row index, the number of non-zero coefficients in the diagonal (local-local) block.
             * @param o_nnz that stores, for each row index, the number of non-zero coefficients in the off-diagonal (local-neighb.) block.
             */
            virtual void sparsity_pattern_prediction(std::vector<PetscInt>& d_nnz, std::vector<PetscInt>& o_nnz) const = 0;
#else
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

#endif
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

            virtual void insert_value_on_diag_for_useless_ghosts(Mat& A) = 0;

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
        // In order to ensure the conversion also the other way around, we use a 'reinterpret_cast'
        // in the function make_assembly() below.
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
