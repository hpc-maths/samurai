#pragma once
#include "../schemes/block_operator.hpp"
// #include "fv/operator_sum_assembly.hpp"
#include "global_numbering.hpp"
#include "manual_assembly.hpp"
#include "utils.hpp"
#include "zero_block_assembly.hpp"

namespace samurai
{
    namespace petsc
    {
        template <std::size_t rows_, std::size_t cols_, class... Operators>
        class BlockAssemblyBase
        {
          public:

            using block_operator_t            = BlockOperator<rows_, cols_, Operators...>;
            static constexpr std::size_t rows = block_operator_t::rows;
            static constexpr std::size_t cols = block_operator_t::cols;
            using scheme_t                    = block_operator_t;

          private:

            block_operator_t m_block_operator;
            std::tuple<Assembly<Operators>...> m_assembly_ops;

          public:

            explicit BlockAssemblyBase(const block_operator_t& block_op)
                : m_block_operator(block_op)
                , m_assembly_ops(transform(m_block_operator.operators(),
                                           [](auto& op)
                                           {
                                               return make_assembly(op);
                                           }))
            {
                for_each_assembly_op(
                    [&](auto& op, auto row, auto col)
                    {
                        bool diagonal_block = (row == col);
                        op.must_insert_value_on_diag_for_useless_ghosts(diagonal_block);
                        op.include_bc(diagonal_block);
                        op.assemble_proj_pred(diagonal_block);
                    });
            }

            void set_block_operator(const block_operator_t& block_op)
            {
                m_block_operator = block_op;
                for_each_assembly_op(
                    [&](auto& op, auto row, auto col)
                    {
                        op.set_scheme(m_block_operator.template get<row, col>());
                    });
            }

            void set_scheme(const block_operator_t& block_op)
            {
                set_block_operator(block_op);
            }

            // template <class OperatorType>
            // static Assembly<OperatorType> to_assembly(OperatorType& op)
            // {
            //     return Assembly<OperatorType>(op);
            // }

            auto& block_operator()
            {
                return m_block_operator;
            }

            auto& block_operator() const
            {
                return m_block_operator;
            }

            template <class Func>
            void for_each_assembly_op(Func&& f)
            {
                static_for<0, rows>::apply(
                    [&](auto row)
                    {
                        static_for<0, cols>::apply(
                            [&](auto col)
                            {
                                f(get<row, col>(), row, col);
                            });
                    });
            }

            template <class Func>
            void for_each_assembly_op(Func&& f) const
            {
                static_for<0, rows>::apply(
                    [&](auto row)
                    {
                        static_for<0, cols>::apply(
                            [&](auto col)
                            {
                                f(get<row, col>(), row, col);
                            });
                    });
            }

            template <std::size_t row, std::size_t col>
            auto& get()
            {
                return std::get<row * cols + col>(m_assembly_ops);
            }

            template <std::size_t row, std::size_t col>
            const auto& get() const
            {
                return std::get<row * cols + col>(m_assembly_ops);
            }

            const auto& first_block() const
            {
                return std::get<0>(m_assembly_ops);
            }

            const auto& mesh() const
            {
                return first_block().unknown().mesh();
            }

            void check_and_set_sizes()
            {
                // Check compatibility of dimensions and set dimensions for blocks that must fit into the matrix
                // - rows:
                for (std::size_t r = 0; r < rows; r++)
                {
                    PetscInt block_owned_rows = 0;
                    PetscInt block_local_rows = 0;
                    for_each_assembly_op(
                        [&](auto& op, auto row, auto)
                        {
                            using op_t = std::decay_t<decltype(op)>;
                            if (row == r)
                            {
                                if (block_owned_rows == 0 && !std::is_same_v<op_t, ZeroBlockAssembly>)
                                {
                                    block_owned_rows = op.owned_matrix_rows();
                                    block_local_rows = op.local_matrix_rows();
                                }
                            }
                        });
                    for_each_assembly_op(
                        [&](auto& op, auto row, auto col)
                        {
                            using op_t = std::decay_t<decltype(op)>;
                            if (row == r)
                            {
                                if constexpr (std::is_same_v<op_t, ZeroBlockAssembly>)
                                {
                                    op.set_owned_matrix_rows(block_owned_rows);
                                    op.set_local_matrix_rows(block_local_rows);
                                }
                                else if (op.owned_matrix_rows() != block_owned_rows)
                                {
                                    std::cerr << "Assembly failure: incompatible number of rows of block (" << row << ", " << col
                                              << "): " << op.owned_matrix_rows() << " (expected " << block_owned_rows << ")" << std::endl;
                                    exit(EXIT_FAILURE);
                                }
                            }
                        });
                }
                // - cols:
                for (std::size_t c = 0; c < cols; c++)
                {
                    PetscInt block_owned_cols = 0;
                    PetscInt block_local_cols = 0;
                    for_each_assembly_op(
                        [&](auto& op, auto, auto col)
                        {
                            using op_t = std::decay_t<decltype(op)>;
                            if (col == c)
                            {
                                if (block_owned_cols == 0 && !std::is_same_v<op_t, ZeroBlockAssembly>)
                                {
                                    block_owned_cols = op.owned_matrix_cols();
                                    block_local_cols = op.local_matrix_cols();
                                }
                            }
                        });
                    for_each_assembly_op(
                        [&](auto& op, auto row, auto col)
                        {
                            using op_t = std::decay_t<decltype(op)>;
                            if (col == c)
                            {
                                if constexpr (std::is_same_v<op_t, ZeroBlockAssembly>)
                                {
                                    op.set_owned_matrix_cols(block_owned_cols);
                                    op.set_local_matrix_cols(block_local_cols);
                                }
                                else if (op.owned_matrix_cols() != block_owned_cols)
                                {
                                    std::cerr << "Assembly failure: incompatible number of columns of block (" << row << ", " << col
                                              << "): " << op.owned_matrix_cols() << " (expected " << block_owned_cols << ")" << std::endl;
                                    exit(EXIT_FAILURE);
                                }
                            }
                        });
                }
            }

            template <class... Fields>
            void set_unknowns(Fields&... unknowns)
            {
                auto unknown_tuple = block_operator().tie_unknowns(unknowns...);
                set_unknown(unknown_tuple);
            }

            template <class... Fields>
            void set_unknown(std::tuple<Fields...>& unknowns)
            {
                for_each_assembly_op(
                    [&](auto& op, auto row, auto col)
                    {
                        using op_t = std::decay_t<decltype(op)>;

                        std::size_t i = 0;
                        for_each(unknowns,
                                 [&](auto& u)
                                 {
                                     if (col == i)
                                     {
                                         // Verify type compatibility only if scheme_t != void (used for zero block)
                                         if constexpr (!std::is_same_v<op_t, ZeroBlockAssembly>)
                                         {
                                             if constexpr (std::is_same_v<std::decay_t<decltype(u)>, typename op_t::scheme_t::input_field_t>)
                                             {
                                                 op.set_unknown(u);
                                             }
                                             else
                                             {
                                                 std::cerr << "unknown " << i << " is not compatible with the scheme (" << row << ", "
                                                           << col << ") (named '" << op.name() << "')" << std::endl;
                                                 assert(false);
                                                 exit(EXIT_FAILURE);
                                             }
                                         }
                                     }
                                     i++;
                                 });
                    });
            }

            bool undefined_unknown() const
            {
                bool undefined = false;
                for_each_assembly_op(
                    [&](auto& op, auto, auto)
                    {
                        using op_t = std::decay_t<decltype(op)>;
                        if constexpr (!std::is_same_v<op_t, ZeroBlockAssembly>)
                        {
                            undefined = undefined || !op.unknown_ptr();
                        }
                    });
                return undefined;
            }

            std::array<std::string, cols> field_names() const
            {
                std::array<std::string, cols> names;
                for_each_assembly_op(
                    [&](auto& op, auto row, auto col)
                    {
                        if constexpr (row == col)
                        {
                            names[col] = op.unknown().name();
                        }
                    });
                return names;
            }

            void check_create_vector_ambiguity() const
            {
                static_assert(
                    rows == cols,
                    "Function 'create_vector()' is ambiguous in this context, because the block matrix is not square. Use 'create_applicable_vector()' or 'create_rhs_vector()' instead.");

                // All the diagonal blocks must be square.
                for_each_assembly_op(
                    [&](auto& op, auto row, auto col)
                    {
                        if (row == col && op.owned_matrix_rows() != op.owned_matrix_cols())
                        {
                            std::cerr << "Function 'create_vector()' is ambiguous in this context, because the block (" << row << ", " << col
                                      << ") is not square. Use 'create_applicable_vector()' or 'create_rhs_vector()' instead." << std::endl;
                            exit(EXIT_FAILURE);
                        }
                    });
            }

            // Mark the cell ownership as 'not computed' to force recomputation
            void reset_cell_ownership()
            {
                for_each_assembly_op(
                    [&](auto& op, auto, auto)
                    {
                        using op_t             = std::decay_t<decltype(op)>;
                        using op_input_field_t = typename op_t::input_field_t;
                        if constexpr (!std::is_base_of_v<ManualAssembly<op_input_field_t>, op_t> && !std::is_same_v<Assembly<int>, op_t>)
                        {
                            op.mesh().cell_ownership().is_computed = false;
                        }
                    });
            }
        };

        template <bool monolithic, std::size_t rows_, std::size_t cols_, class... Operators>
        class BlockAssembly
        {
        };

        /**
         * Assemble block matrix using PETSc nested matrices.
         */
        template <std::size_t rows_, std::size_t cols_, class... Operators>
        class BlockAssembly<false, rows_, cols_, Operators...> : public BlockAssemblyBase<rows_, cols_, Operators...>
        {
            using base_class = BlockAssemblyBase<rows_, cols_, Operators...>;

          public:

            using base_class::block_operator;
            using block_operator_t = typename base_class::block_operator_t;
            using base_class::cols;
            using base_class::for_each_assembly_op;
            using base_class::rows;

          private:

            std::array<Mat, rows * cols> m_blocks;
            bool m_is_set_up = false;

          public:

            explicit BlockAssembly(const block_operator_t& block_op)
                : base_class(block_op)
            {
                for_each_assembly_op(
                    [&](auto& op, auto row, auto col)
                    {
                        block(row, col) = nullptr;
                        op.is_block_in_nested_matrix(true);
                    });
            }

            void create_matrix(Mat& A)
            {
                if (!m_is_set_up)
                {
                    reset();
                    m_is_set_up = true;
                }

                for_each_assembly_op(
                    [&](auto& op, auto row, auto col)
                    {
                        // std::cout << "create_matrix (" << row << ", " << col << ")" << std::endl;
                        op.create_matrix(block(row, col));
                    });
                MatCreateNest(PETSC_COMM_WORLD, rows, PETSC_IGNORE, cols, PETSC_IGNORE, m_blocks.data(), &A);
            }

            void assemble_matrix(Mat& A, bool final_assembly = true)
            {
                for_each_assembly_op(
                    [&](auto& op, auto row, auto col)
                    {
                        // std::cout << "assemble_matrix (" << row << ", " << col << ") '" << op.name() << "'" << std::endl;
                        op.assemble_matrix(block(row, col));
                    });
                if (final_assembly)
                {
                    MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY);
                    MatAssemblyEnd(A, MAT_FINAL_ASSEMBLY);
                }
            }

            void is_set_up(bool value)
            {
                m_is_set_up = value;
                for_each_assembly_op(
                    [&](auto& op, auto, auto)
                    {
                        op.is_set_up(value);
                    });
            }

            void reset()
            {
                this->reset_cell_ownership();

                for_each_assembly_op(
                    [&](auto& op, auto, auto)
                    {
                        op.reset(*this);
                    });

#ifdef SAMURAI_WITH_MPI
                // Computes numbering for diagonal blocks
                for_each_assembly_op(
                    [&](auto& op, auto row, auto col)
                    {
                        if constexpr (row == col)
                        {
                            op.compute_numbering();
                            op.compute_local_to_global_rows();
                        }
                    });

                // The off-diagonal blocks are reset using the numbering of the diagonal blocks
                for_each_assembly_op(
                    [&](auto& op, auto row, auto col)
                    {
                        if constexpr (row != col)
                        {
                            op.set_row_numbering(this->template get<row, row>().row_numbering());
                            op.set_col_numbering(this->template get<col, col>().col_numbering());
                        }
                    });
#endif

                // Check compatibility of dimensions and set dimensions for blocks that must fit into the matrix
                this->check_and_set_sizes();
            }

            Mat& block(std::size_t row, std::size_t col)
            {
                auto i = row * cols + col;
                return m_blocks[i];
            }

            void destroy_local_to_global_mappings(Mat& A)
            {
                for_each_assembly_op(
                    [&](auto& op, auto row, auto col)
                    {
                        if constexpr (row == col)
                        {
                            Mat block_mat;
                            MatNestGetSubMat(A, static_cast<PetscInt>(row), static_cast<PetscInt>(col), &block_mat);
                            op.destroy_local_to_global_mappings(block_mat);
                        }
                    });
            }

            void enforce_bc(Vec& b) const
            {
                for_each_assembly_op(
                    [&](auto& op, auto row, auto col)
                    {
                        if constexpr (row == col)
                        {
                            if (op.include_bc())
                            {
                                // std::cout << "enforce_bc (" << row << ", " << col << ") on b[" << row << "]" << std::endl;
                                Vec b_block;
                                VecNestGetSubVec(b, static_cast<PetscInt>(row), &b_block);
                                op.enforce_bc(b_block);
                            }
                        }
                    });
            }

            void enforce_projection_prediction(Vec& b) const
            {
                for_each_assembly_op(
                    [&](auto& op, auto row, auto)
                    {
                        if (op.assemble_proj_pred())
                        {
                            Vec b_block;
                            VecNestGetSubVec(b, static_cast<PetscInt>(row), &b_block);
                            op.enforce_projection_prediction(b_block);
                        }
                    });
            }

            void set_diag_value_for_useless_ghosts(PetscScalar value)
            {
                for_each_assembly_op(
                    [&](auto& op, auto row, auto col)
                    {
                        if (row == col)
                        {
                            op.set_diag_value_for_useless_ghosts(value);
                        }
                    });
            }

            void set_0_for_useless_ghosts(Vec& b) const
            {
                for_each_assembly_op(
                    [&](auto& op, auto row, auto)
                    {
                        if (op.must_insert_value_on_diag_for_useless_ghosts())
                        {
                            Vec b_block;
                            VecNestGetSubVec(b, static_cast<PetscInt>(row), &b_block);
                            op.set_0_for_useless_ghosts(b_block);
                        }
                    });
            }

            void set_0_for_all_ghosts(Vec& b) const
            {
                for_each_assembly_op(
                    [&](auto& op, auto row, auto col)
                    {
                        if constexpr (row == col)
                        {
                            Vec b_block;
                            VecNestGetSubVec(b, static_cast<PetscInt>(row), &b_block);
                            op.set_0_for_all_ghosts(b_block);
                        }
                    });
            }

            Vec create_solution_vector_from_unknown_fields() const
            {
                std::array<Vec, cols> x_blocks;
                for_each_assembly_op(
                    [&](auto& op, auto row, auto col)
                    {
                        if constexpr (row == col)
                        {
                            x_blocks[col] = op.create_solution_vector(op.unknown());
                        }
                    });
                Vec x;
                VecCreateNest(PETSC_COMM_WORLD, cols, NULL, x_blocks.data(), &x);
                PetscObjectSetName(reinterpret_cast<PetscObject>(x), "solution");
                return x;
            }

            template <class... Fields>
            Vec create_applicable_vector(const std::tuple<Fields&...>& fields) const
            {
                std::array<Vec, cols> x_blocks;
                this->for_each_assembly_op(
                    [&](auto& op, auto row, auto col)
                    {
                        if constexpr (row == col)
                        {
                            x_blocks[col] = op.create_solution_vector(std::get<col>(fields));
                        }
                    });
                Vec x;
                VecCreateNest(PETSC_COMM_WORLD, cols, NULL, x_blocks.data(), &x);
                return x;
            }

            template <class... Fields>
            Vec create_rhs_vector(const std::tuple<Fields&...>& sources) const
            {
                std::array<Vec, rows> b_blocks;
                this->for_each_assembly_op(
                    [&](auto& op, auto row, auto col)
                    {
                        if constexpr (row == col)
                        {
                            b_blocks[row] = op.create_rhs_vector(std::get<row>(sources));
                        }
                    });
                Vec b;
                VecCreateNest(PETSC_COMM_WORLD, rows, NULL, b_blocks.data(), &b);
                PetscObjectSetName(reinterpret_cast<PetscObject>(b), "right-hand side");
                return b;
            }

            template <class... Fields>
            Vec create_vector(const std::tuple<Fields&...>& fields) const
            {
                this->check_create_vector_ambiguity();
                return create_applicable_vector(fields);
            }

            template <class... Fields>
            Vec create_vector(const Fields&... fields) const
            {
                auto tuple = block_operator().tie_unknowns(fields...);
                return create_vector(tuple);
            }

            void update_unknowns(const Vec& x) const
            {
                for_each_assembly_op(
                    [&](auto& op, auto row, auto)
                    {
                        if constexpr (row == col)
                        {
                            Vec x_block;
                            VecNestGetSubVec(x, static_cast<PetscInt>(row), &x_block);
                            op.copy_unknown(x_block, op.unknown());
                        }
                    });
            }
        };

        /**
         * Assemble block matrix as a monolithic matrix.
         */
        template <std::size_t rows_, std::size_t cols_, class... Operators>
        class BlockAssembly<true, rows_, cols_, Operators...> : public BlockAssemblyBase<rows_, cols_, Operators...>,
                                                                public MatrixAssembly
        {
            using base_class = BlockAssemblyBase<rows_, cols_, Operators...>;

          public:

            using base_class::block_operator;
            using block_operator_t = typename base_class::block_operator_t;
            using base_class::cols;
            using base_class::for_each_assembly_op;
            using base_class::rows;

          private:

            Numbering m_numbering;

          public:

            explicit BlockAssembly(const block_operator_t& block_op)
                : base_class(block_op)
            {
                this->set_name("(unnamed monolithic block operator)");

                for_each_assembly_op(
                    [&](auto& op, auto, auto)
                    {
                        op.is_block_in_monolithic_matrix(true);
                    });
            }

            void is_set_up(bool value) override
            {
                MatrixAssembly::is_set_up(value);
                for_each_assembly_op(
                    [&](auto& op, auto, auto)
                    {
                        op.is_set_up(value);
                    });
            }

            Numbering& numbering()
            {
                return m_numbering;
            }

            void reset() override
            {
                this->reset_cell_ownership();

                for_each_assembly_op(
                    [&](auto& op, auto, auto)
                    {
                        op.reset(*this);
                    });

                // Check compatibility of dimensions and set dimensions for blocks that must fit into the matrix
                this->check_and_set_sizes();

                PetscInt n_owned_rows = this->owned_matrix_rows();
                PetscInt n_owned_cols = this->owned_matrix_cols();

                // Set row_shift and col_shift for each block assembly operator
#ifdef SAMURAI_WITH_MPI
                PetscInt rank_shift_owned_rows = compute_rank_shift(n_owned_rows);
                PetscInt rank_shift_owned_cols = rank_shift_owned_rows;
                if (n_owned_cols != n_owned_rows)
                {
                    rank_shift_owned_cols = compute_rank_shift(n_owned_cols);
                }

                for_each_assembly_op(
                    [&](auto& op, auto, auto)
                    {
                        op.set_rank_row_shift(rank_shift_owned_rows);
                        op.set_rank_col_shift(rank_shift_owned_cols);
                    });
#endif
                PetscInt owned_row_shift = 0;
                PetscInt owned_col_shift = 0;
                for_each_assembly_op(
                    [&](auto& op, auto, auto col)
                    {
                        op.set_row_shift(owned_row_shift);
                        op.set_col_shift(owned_col_shift);
                        owned_col_shift += op.owned_matrix_cols();
                        if (col == cols - 1)
                        {
                            owned_col_shift = 0;
                            owned_row_shift += op.owned_matrix_rows();
                        }
                    });

                PetscInt ghosts_row_shift = n_owned_rows;
                PetscInt ghosts_col_shift = n_owned_cols;
                for_each_assembly_op(
                    [&](auto& op, auto, auto col)
                    {
                        op.set_ghosts_row_shift(ghosts_row_shift);
                        op.set_ghosts_col_shift(ghosts_col_shift);
                        ghosts_col_shift += op.local_matrix_cols() - op.owned_matrix_cols();
                        if (col == cols - 1)
                        {
                            ghosts_col_shift = n_owned_cols;
                            ghosts_row_shift += op.local_matrix_rows() - op.owned_matrix_rows();
                        }
                    });

#ifdef SAMURAI_WITH_MPI
                // if (mpi::communicator().rank() == 1)
                // {
                //     sleep(1); // to have ordered output
                // }

                // for_each_assembly_op(
                //     [&](auto& op, auto row, auto col)
                //     {
                //         if (col == 0)
                //         {
                //             std::cout << "[" << mpi::communicator().rank() << "] (" << row << ", " << col
                //                       << "): rank_col_shift = " << op.rank_col_shift() << std::endl;
                //         }
                //     });
                // for_each_assembly_op(
                //     [&](auto& op, auto row, auto col)
                //     {
                //         if (col == 0)
                //         {
                //             std::cout << "[" << mpi::communicator().rank() << "] (" << row << ", " << col
                //                       << "): owned_col_shift = " << op.col_shift() << std::endl;
                //         }
                //     });
                // for_each_assembly_op(
                //     [&](auto& op, auto row, auto col)
                //     {
                //         if (col == 0)
                //         {
                //             std::cout << "[" << mpi::communicator().rank() << "] (" << row << ", " << col
                //                       << "): ghosts_col_shift = " << op.ghosts_col_shift() << std::endl;
                //         }
                //     });

                m_numbering.resize(this->local_matrix_rows());
                for_each_assembly_op(
                    [&](auto& op, auto row, auto col)
                    {
                        if constexpr (row == col)
                        {
                            op.compute_block_numbering();
                        }
                    });

                for_each_assembly_op(
                    [&](auto& op, auto row, auto col)
                    {
                        if constexpr (col == row)
                        {
                            op.compute_local_to_global_rows(m_numbering.local_to_global_mapping);
                        }
                    });
#endif
            }

            const std::vector<PetscInt>& local_to_global_rows() const override
            {
                return m_numbering.local_to_global_mapping;
            }

            const std::vector<PetscInt>& local_to_global_cols() const override
            {
                assert(false && "Not implemented yet");
                return m_numbering.local_to_global_mapping; // just to return something
            }

          public:

            PetscInt owned_matrix_rows() const override
            {
                PetscInt total_rows = 0;
                for_each_assembly_op(
                    [&](auto& op, auto row, auto col)
                    {
                        if (row == col)
                        {
                            total_rows += op.owned_matrix_rows();
                        }
                    });
                return total_rows;
            }

            PetscInt owned_matrix_cols() const override
            {
                PetscInt total_cols = 0;
                for_each_assembly_op(
                    [&](auto& op, auto row, auto col)
                    {
                        if (row == col)
                        {
                            total_cols += op.owned_matrix_cols();
                        }
                    });
                return total_cols;
            }

            PetscInt local_matrix_rows() const override
            {
                PetscInt total_rows = 0;
                for_each_assembly_op(
                    [&](auto& op, auto row, auto col)
                    {
                        if (row == col)
                        {
                            total_rows += op.local_matrix_rows();
                        }
                    });
                return total_rows;
            }

            PetscInt local_matrix_cols() const override
            {
                PetscInt total_cols = 0;
                for_each_assembly_op(
                    [&](auto& op, auto row, auto col)
                    {
                        if (row == col)
                        {
                            total_cols += op.local_matrix_cols();
                        }
                    });
                return total_cols;
            }

            void sparsity_pattern_scheme(std::vector<PetscInt>& d_nnz, std::vector<PetscInt>& o_nnz) const override
            {
                for_each_assembly_op(
                    [&](auto& op, auto, auto)
                    {
                        op.sparsity_pattern_scheme(d_nnz, o_nnz);
                    });
            }

            void sparsity_pattern_boundary(std::vector<PetscInt>& d_nnz, std::vector<PetscInt>& o_nnz) const override
            {
                for_each_assembly_op(
                    [&](auto& op, auto, auto)
                    {
                        if (op.include_bc())
                        {
                            op.sparsity_pattern_boundary(d_nnz, o_nnz);
                        }
                    });
            }

            void sparsity_pattern_projection(std::vector<PetscInt>& d_nnz, std::vector<PetscInt>& o_nnz) const override
            {
                for_each_assembly_op(
                    [&](auto& op, auto, auto)
                    {
                        if (op.assemble_proj_pred())
                        {
                            op.sparsity_pattern_projection(d_nnz, o_nnz);
                        }
                    });
            }

            void sparsity_pattern_prediction(std::vector<PetscInt>& d_nnz, std::vector<PetscInt>& o_nnz) const override
            {
                for_each_assembly_op(
                    [&](auto& op, auto, auto)
                    {
                        if (op.assemble_proj_pred())
                        {
                            op.sparsity_pattern_prediction(d_nnz, o_nnz);
                        }
                    });
            }

            void sparsity_pattern_useless_ghosts(std::vector<PetscInt>& d_nnz) override
            {
                for_each_assembly_op(
                    [&](auto& op, auto, auto)
                    {
                        if (op.must_insert_value_on_diag_for_useless_ghosts())
                        {
                            op.sparsity_pattern_useless_ghosts(d_nnz);
                        }
                    });
            }

            void assemble_scheme(Mat& A) override
            {
                InsertMode insert_mode;
                for_each_assembly_op(
                    [&](auto& op, auto row, auto col)
                    {
                        if (row > 0 || col > 0)
                        {
                            op.set_current_insert_mode(insert_mode);
                        }
                        // std::cout << "[" << mpi::communicator().rank() << "] assemble_scheme (" << row << ", " << col << ") '" <<
                        // op.name()
                        //           << "'" << std::endl;
                        op.assemble_scheme(A);
                        insert_mode = op.current_insert_mode();
                    });
            }

            void assemble_boundary_conditions(Mat& A) override
            {
                for_each_assembly_op(
                    [&](auto& op, auto, auto)
                    {
                        if (op.include_bc())
                        {
                            op.assemble_boundary_conditions(A);
                        }
                    });
            }

            void assemble_projection(Mat& A) override
            {
                for_each_assembly_op(
                    [&](auto& op, auto, auto)
                    {
                        if (op.assemble_proj_pred())
                        {
                            op.assemble_projection(A);
                        }
                    });
            }

            void assemble_prediction(Mat& A) override
            {
                for_each_assembly_op(
                    [&](auto& op, auto, auto)
                    {
                        if (op.assemble_proj_pred())
                        {
                            op.assemble_prediction(A);
                        }
                    });
            }

            void set_diag_value_for_useless_ghosts(PetscScalar value) override
            {
                for_each_assembly_op(
                    [&](auto& op, auto row, auto col)
                    {
                        if constexpr (row == col)
                        {
                            op.set_diag_value_for_useless_ghosts(value);
                        }
                    });
            }

            void insert_value_on_diag_for_useless_ghosts(Mat& A) override
            {
                for_each_assembly_op(
                    [&](auto& op, auto, auto)
                    {
                        if (op.must_insert_value_on_diag_for_useless_ghosts())
                        {
                            op.insert_value_on_diag_for_useless_ghosts(A);
                        }
                    });
            }

            template <class Func>
            void for_each_useless_ghost_row(Func&& f) const
            {
                for_each_assembly_op(
                    [&](auto& op, auto row, auto col)
                    {
                        if constexpr (row == col)
                        {
                            op.for_each_useless_ghost_row(std::forward<Func>(f));
                        }
                    });
            }

            template <class... Fields>
            Vec create_rhs_vector(const std::tuple<Fields&...>& sources) const
            {
                Vec b = create_petsc_vector(owned_matrix_rows());

                this->for_each_assembly_op(
                    [&](auto& op, auto row, auto col)
                    {
                        if constexpr (col == row)
                        {
                            op.copy_rhs(std::get<col>(sources), b);
                        }
                    });
                PetscObjectSetName(reinterpret_cast<PetscObject>(b), "right-hand side");
                return b;
            }

            template <class... Fields>
            Vec create_applicable_vector(const std::tuple<Fields&...>& fields) const
            {
                Vec x = create_petsc_vector(owned_matrix_cols());

                this->for_each_assembly_op(
                    [&](auto& op, auto row, auto col)
                    {
                        if constexpr (row == col)
                        {
                            op.copy_unknown(std::get<row>(fields), x);
                        }
                    });
                return x;
            }

            Vec create_solution_vector_from_unknown_fields() const
            {
                Vec x = create_petsc_vector(owned_matrix_cols());

                for_each_assembly_op(
                    [&](auto& op, auto row, auto col)
                    {
                        if constexpr (row == col)
                        {
                            op.copy_unknown(op.unknown(), x);
                        }
                    });

                PetscObjectSetName(reinterpret_cast<PetscObject>(x), "solution");
                return x;
            }

            template <class... Fields>
            Vec create_vector(const std::tuple<Fields&...>& fields) const
            {
                this->check_create_vector_ambiguity();
                return create_applicable_vector(fields);
            }

            template <class... Fields>
            Vec create_vector(const Fields&... fields) const
            {
                auto tuple = block_operator().tie_unknowns(fields...);
                return create_vector(tuple);
            }

            void enforce_bc(Vec& b) const
            {
                for_each_assembly_op(
                    [&](auto& op, auto row, auto col)
                    {
                        if constexpr (row == col)
                        {
                            if (op.include_bc())
                            {
                                // std::cout << "enforce_bc (" << row << ", " << col << ") on b[" << row << "]" << std::endl;
                                op.enforce_bc(b);
                            }
                        }
                    });
            }

            void enforce_projection_prediction(Vec& b) const
            {
                for_each_assembly_op(
                    [&](auto& op, auto, auto)
                    {
                        if (op.assemble_proj_pred())
                        {
                            op.enforce_projection_prediction(b);
                        }
                    });
            }

            void set_0_for_useless_ghosts(Vec& b) const
            {
                for_each_assembly_op(
                    [&](auto& op, auto, auto)
                    {
                        if (op.must_insert_value_on_diag_for_useless_ghosts())
                        {
                            op.set_0_for_useless_ghosts(b);
                        }
                    });
            }

            void set_0_for_all_ghosts(Vec& b) const
            {
                for_each_assembly_op(
                    [&](auto& op, auto row, auto col)
                    {
                        if constexpr (row == col)
                        {
                            op.set_0_for_all_ghosts(b);
                        }
                    });
            }

            void update_unknowns(const Vec& x) const
            {
                for_each_assembly_op(
                    [&](auto& op, auto row, auto col)
                    {
                        if constexpr (row == col)
                        {
                            op.copy_unknown(x, op.unknown());
                        }
                    });
            }

            template <class... Fields>
            void update_result_fields(Vec& b, std::tuple<Fields&...>& result_fields) const
            {
                std::size_t i = 0;
                for_each(result_fields,
                         [&](auto& result_field)
                         {
                             for_each_assembly_op(
                                 [&](auto& op, auto row, auto col)
                                 {
                                     if (col == 0 && row == i)
                                     {
                                         // copy(s, b, op.row_shift());
                                         copy(op.row_shift(), b, result_field);
                                     }
                                 });
                             i++;
                         });
            }

            std::array<IS, cols> create_fields_IS()
            {
                std::array<IS, cols> IS_array;
                for_each_assembly_op(
                    [&](auto& op, auto row, auto col)
                    {
                        if (row == 1)
                        {
                            std::vector<PetscInt> idx(static_cast<std::size_t>(op.owned_matrix_cols()));
                            for (std::size_t i = 0; i < idx.size(); ++i)
                            {
                                idx[i] = op.col_shift() + static_cast<PetscInt>(i);
                            }
                            ISCreateGeneral(PETSC_COMM_WORLD, static_cast<PetscInt>(idx.size()), idx.data(), PETSC_COPY_VALUES, &IS_array[col]);
                        }
                    });
                return IS_array;
            }
        };

        // template <std::size_t rows_, std::size_t cols_, class... Operators>
        // using NestedBlockAssembly = BlockAssembly<false, rows_, cols_, Operators...>;

        // template <std::size_t rows_, std::size_t cols_, class... Operators>
        // using MonolithicBlockAssembly = BlockAssembly<true, rows_, cols_, Operators...>;

        template <bool monolithic, std::size_t rows_, std::size_t cols_, class... Operators>
        auto make_assembly(const BlockOperator<rows_, cols_, Operators...>& block_op)
        {
            return BlockAssembly<monolithic, rows_, cols_, Operators...>(block_op);
        }

        template <std::size_t rows_, std::size_t cols_, class... Operators>
        auto make_assembly(const BlockOperator<rows_, cols_, Operators...>& block_op)
        {
            return make_assembly<true>(block_op);
        }

        /**
         * How to apply a block matrix to a vector:
         *
         * // Monolithic assembly
         * auto monolithicAssembly = samurai::petsc::make_assembly<true>(stokes);
         * Mat monolithicA;
         * monolithicAssembly.create_matrix(monolithicA);
         * monolithicAssembly.assemble_matrix(monolithicA);
         * Vec mono_x                = monolithicAssembly.create_applicable_vector(x); // copy
         * auto result_velocity_mono = samurai::make_field<dim, is_soa>("result_velocity", mesh);
         * auto result_pressure_mono = samurai::make_field<1, is_soa>("result_pressure", mesh);
         * auto result_mono          = stokes.tie_rhs(result_velocity_mono, result_pressure_mono);
         * Vec mono_result           = monolithicAssembly.create_rhs_vector(result_mono); // copy
         * MatMult(monolithicA, mono_x, mono_result);
         * monolithicAssembly.update_result_fields(mono_result, result_mono); // copy
         *
         * // Nested assembly
         * auto nestedAssembly = samurai::petsc::make_assembly<false>(stokes);
         * Mat nestedA;
         * nestedAssembly.create_matrix(nestedA);
         * nestedAssembly.assemble_matrix(nestedA);
         * Vec nest_x                = nestedAssembly.create_applicable_vector(x);
         * auto result_velocity_nest = samurai::make_field<dim, is_soa>("result_velocity", mesh);
         * auto result_pressure_nest = samurai::make_field<1, is_soa>("result_pressure", mesh);
         * auto result_nest          = stokes.tie_rhs(result_velocity_nest, result_pressure_nest);
         * Vec nest_result           = nestedAssembly.create_rhs_vector(result_nest);
         * MatMult(nestedA, nest_x, nest_result);
         */

    } // end namespace petsc
} // end namespace samurai
