#pragma once
#include "matrix_assembly.hpp"
#include "utils.hpp"

namespace samurai
{
    namespace petsc
    {
        /**
         * Assemble block matrix using PETSc nested matrices.
         */
        template <int rows, int cols, class... Operators>
        class BlockAssembly
        {
          public:

            static constexpr int n_rows = rows;
            static constexpr int n_cols = cols;

          private:

            std::tuple<Operators...> m_operators;
            std::array<Mat, rows * cols> m_blocks;

          public:

            BlockAssembly(Operators&... operators)
                : m_operators(operators...)
            {
                static constexpr std::size_t n_operators = sizeof...(operators);
                static_assert(n_operators == rows * cols, "The number of operators must correspond to rows*cols.");

                std::size_t i = 0;
                for_each(m_operators,
                         [&](auto& op)
                         {
                             auto row            = i / cols;
                             auto col            = i % cols;
                             m_blocks[i]         = nullptr;
                             bool diagonal_block = (row == col);
                             op.add_1_on_diag_for_useless_ghosts_if(diagonal_block);
                             op.include_bc_if(diagonal_block);
                             op.assemble_proj_pred_if(diagonal_block);
                             i++;
                         });
            }

            std::array<std::string, cols> field_names() const
            {
                std::array<std::string, cols> names;
                std::size_t i = 0;
                for_each(m_operators,
                         [&](auto& op)
                         {
                             auto row = i / cols;
                             auto col = i % cols;
                             if (row == col)
                             {
                                 names[col] = op.unknown().name();
                             }
                             i++;
                         });
                return names;
            }

            void create_matrix(Mat& A)
            {
                std::size_t i = 0;
                for_each(m_operators,
                         [&](auto& op)
                         {
                             /*auto row = i / cols;
                             auto col = i % cols;
                             std::cout << "create_matrix (" << row << ", " << col << ")" << std::endl;*/
                             op.create_matrix(m_blocks[i]);
                             i++;
                         });

                MatCreateNest(PETSC_COMM_SELF, rows, PETSC_NULL, cols, PETSC_NULL, m_blocks.data(), &A);
            }

            void assemble_matrix(Mat& A)
            {
                std::size_t i = 0;
                for_each(m_operators,
                         [&](auto& op)
                         {
                             /*auto row = i / cols;
                             auto col = i % cols;
                             std::cout << "assemble_matrix (" << row << ", " << col << ") '" << op.name() << "'" << std::endl;*/
                             op.assemble_matrix(m_blocks[i]);
                             i++;
                         });
                MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY);
                MatAssemblyEnd(A, MAT_FINAL_ASSEMBLY);
            }

            void reset()
            {
                for_each(m_operators,
                         [&](auto& op)
                         {
                             op.reset();
                         });
            }

            Mat& block(std::size_t row, std::size_t col)
            {
                auto i = row * cols + col;
                return m_blocks[i];
            }

            template <class... Fields>
            Vec create_rhs_vector(const std::tuple<Fields&...>& sources) const
            {
                std::array<Vec, rows> b_blocks;
                std::size_t i = 0;
                for_each(sources,
                         [&](auto& s)
                         {
                             b_blocks[i] = create_petsc_vector_from(s);
                             PetscObjectSetName(reinterpret_cast<PetscObject>(b_blocks[i]), s.name().c_str());
                             i++;
                         });
                Vec b;
                VecCreateNest(PETSC_COMM_SELF, rows, NULL, b_blocks.data(), &b);
                PetscObjectSetName(reinterpret_cast<PetscObject>(b), "right-hand side");
                return b;
            }

            void enforce_bc(Vec& b) const
            {
                std::size_t i = 0;
                for_each(m_operators,
                         [&](const auto& op)
                         {
                             auto row = i / cols;
                             // auto col = i % cols;
                             if (op.include_bc())
                             {
                                 // std::cout << "enforce_bc (" << row << ", " << col << ") on b[" << row << "]" << std::endl;
                                 Vec b_block;
                                 VecNestGetSubVec(b, static_cast<PetscInt>(row), &b_block);
                                 op.enforce_bc(b_block);
                             }
                             i++;
                         });
            }

            void enforce_projection_prediction(Vec& b) const
            {
                std::size_t i = 0;
                for_each(m_operators,
                         [&](const auto& op)
                         {
                             auto row = i / cols;
                             if (op.assemble_proj_pred())
                             {
                                 Vec b_block;
                                 VecNestGetSubVec(b, static_cast<PetscInt>(row), &b_block);
                                 op.enforce_projection_prediction(b_block);
                             }
                             i++;
                         });
            }

            void add_0_for_useless_ghosts(Vec& b) const
            {
                std::size_t i = 0;
                for_each(m_operators,
                         [&](const auto& op)
                         {
                             auto row = i / cols;
                             if (op.must_add_1_on_diag_for_useless_ghosts())
                             {
                                 Vec b_block;
                                 VecNestGetSubVec(b, static_cast<PetscInt>(row), &b_block);
                                 op.add_0_for_useless_ghosts(b_block);
                             }
                             i++;
                         });
            }

            Vec create_solution_vector() const
            {
                std::array<Vec, cols> x_blocks;
                std::size_t i = 0;
                for_each(m_operators,
                         [&](const auto& op)
                         {
                             auto row = i / cols;
                             auto col = i % cols;
                             if (row == 0)
                             {
                                 x_blocks[col] = create_petsc_vector_from(op.unknown());
                                 PetscObjectSetName(reinterpret_cast<PetscObject>(x_blocks[col]), op.unknown().name().c_str());
                             }
                             i++;
                         });
                Vec x;
                VecCreateNest(PETSC_COMM_SELF, cols, NULL, x_blocks.data(), &x);
                PetscObjectSetName(reinterpret_cast<PetscObject>(x), "solution");
                return x;
            }

            template <class... Fields>
            auto tie(Fields&... fields) const
            {
                static constexpr std::size_t n_fields = sizeof...(fields);
                static_assert(n_fields == rows,
                              "The number of fields must correspond to the "
                              "number of rows of the block operator.");

                return std::tuple<Fields&...>(fields...);
            }
        };

        /**
         * Assemble block matrix as a monolithic matrix.
         */
        template <int rows, int cols, class... Operators>
        class MonolithicBlockAssembly : public MatrixAssembly
        {
          public:

            static constexpr int n_rows = rows;
            static constexpr int n_cols = cols;

          private:

            std::tuple<Operators...> m_operators;

          public:

            MonolithicBlockAssembly(Operators&... operators)
                : m_operators(operators...)
            {
                static constexpr std::size_t n_operators = sizeof...(operators);
                static_assert(n_operators == rows * cols, "The number of operators must correspond to rows*cols.");

                this->set_name("(unnamed monolithic block operator)");

                for_each_operator(
                    [&](auto& op, auto row, auto col)
                    {
                        op.set_is_block(true);
                        bool diagonal_block = (row == col);
                        op.add_1_on_diag_for_useless_ghosts_if(diagonal_block);
                        op.include_bc_if(diagonal_block);
                        op.assemble_proj_pred_if(diagonal_block);
                    });
                reset();
            }

            void reset() override
            {
                PetscInt row_shift = 0;
                PetscInt col_shift = 0;
                for_each_operator(
                    [&](auto& op, auto, auto col)
                    {
                        op.reset();
                        op.set_row_shift(row_shift);
                        op.set_col_shift(col_shift);
                        col_shift += op.matrix_cols();
                        if (col == cols - 1)
                        {
                            col_shift = 0;
                            row_shift += op.matrix_rows();
                        }
                    });
            }

          private:

            template <class Func>
            void for_each_operator(Func&& f)
            {
                std::size_t i = 0;
                for_each(m_operators,
                         [&](auto& op)
                         {
                             auto row = i / cols;
                             auto col = i % cols;
                             f(op, row, col);
                             i++;
                         });
            }

            template <class Func>
            void for_each_operator(Func&& f) const
            {
                std::size_t i = 0;
                for_each(m_operators,
                         [&](const auto& op)
                         {
                             auto row = i / cols;
                             auto col = i % cols;
                             f(op, row, col);
                             i++;
                         });
            }

          public:

            PetscInt matrix_rows() const override
            {
                PetscInt total_rows = 0;
                for_each_operator(
                    [&](auto& op, auto row, auto col)
                    {
                        if (row == col)
                        {
                            total_rows += op.matrix_rows();
                        }
                    });
                return total_rows;
            }

            PetscInt matrix_cols() const override
            {
                PetscInt total_cols = 0;
                for_each_operator(
                    [&](auto& op, auto row, auto col)
                    {
                        if (row == col)
                        {
                            total_cols += op.matrix_cols();
                        }
                    });
                return total_cols;
            }

            std::array<std::string, cols> field_names() const
            {
                std::array<std::string, cols> names;
                for_each_operator(
                    [&](auto& op, auto row, auto col)
                    {
                        if (row == col)
                        {
                            names[col] = op.unknown().name();
                        }
                    });
                return names;
            }

            void sparsity_pattern_scheme(std::vector<PetscInt>& nnz) const override
            {
                for_each_operator(
                    [&](auto& op, auto, auto)
                    {
                        op.sparsity_pattern_scheme(nnz);
                    });
            }

            void sparsity_pattern_boundary(std::vector<PetscInt>& nnz) const override
            {
                for_each_operator(
                    [&](auto& op, auto, auto)
                    {
                        if (op.include_bc())
                        {
                            op.sparsity_pattern_boundary(nnz);
                        }
                    });
            }

            void sparsity_pattern_projection(std::vector<PetscInt>& nnz) const override
            {
                for_each_operator(
                    [&](auto& op, auto, auto)
                    {
                        if (op.assemble_proj_pred())
                        {
                            op.sparsity_pattern_projection(nnz);
                        }
                    });
            }

            void sparsity_pattern_prediction(std::vector<PetscInt>& nnz) const override
            {
                for_each_operator(
                    [&](auto& op, auto, auto)
                    {
                        if (op.assemble_proj_pred())
                        {
                            op.sparsity_pattern_prediction(nnz);
                        }
                    });
            }

            void sparsity_pattern_useless_ghosts(std::vector<PetscInt>& nnz) override
            {
                for_each_operator(
                    [&](auto& op, auto, auto)
                    {
                        if (op.must_add_1_on_diag_for_useless_ghosts())
                        {
                            op.sparsity_pattern_useless_ghosts(nnz);
                        }
                    });
            }

            void assemble_scheme(Mat& A) override
            {
                InsertMode insert_mode;
                for_each_operator(
                    [&](auto& op, auto row, auto col)
                    {
                        if (row > 0 || col > 0)
                        {
                            op.set_current_insert_mode(insert_mode);
                        }
                        op.assemble_scheme(A);
                        insert_mode = op.current_insert_mode();
                    });
            }

            void assemble_boundary_conditions(Mat& A) override
            {
                for_each_operator(
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
                for_each_operator(
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
                for_each_operator(
                    [&](auto& op, auto, auto)
                    {
                        if (op.assemble_proj_pred())
                        {
                            op.assemble_prediction(A);
                        }
                    });
            }

            void add_1_on_diag_for_useless_ghosts(Mat& A) override
            {
                for_each_operator(
                    [&](auto& op, auto, auto)
                    {
                        if (op.must_add_1_on_diag_for_useless_ghosts())
                        {
                            op.add_1_on_diag_for_useless_ghosts(A);
                        }
                    });
            }

            template <class... Fields>
            Vec create_rhs_vector(const std::tuple<Fields&...>& sources) const
            {
                Vec b;
                VecCreateSeq(MPI_COMM_SELF, matrix_rows(), &b);
                std::size_t i = 0;
                for_each(sources,
                         [&](auto& s)
                         {
                             for_each_operator(
                                 [&](auto& op, auto row, auto col)
                                 {
                                     if (col == 0 && row == i)
                                     {
                                         copy(s, b, op.row_shift());
                                     }
                                 });
                             i++;
                         });
                PetscObjectSetName(reinterpret_cast<PetscObject>(b), "right-hand side");
                return b;
            }

            void enforce_bc(Vec& b) const
            {
                for_each_operator(
                    [&](auto& op, auto, auto)
                    {
                        if (op.include_bc())
                        {
                            // std::cout << "enforce_bc (" << row << ", "
                            // << col << ") on b[" << row << "]" <<
                            // std::endl;
                            op.enforce_bc(b);
                        }
                    });
            }

            void enforce_projection_prediction(Vec& b) const
            {
                for_each_operator(
                    [&](auto& op, auto, auto)
                    {
                        if (op.assemble_proj_pred())
                        {
                            op.enforce_projection_prediction(b);
                        }
                    });
            }

            void add_0_for_useless_ghosts(Vec& b) const
            {
                for_each_operator(
                    [&](auto& op, auto, auto)
                    {
                        if (op.must_add_1_on_diag_for_useless_ghosts())
                        {
                            // std::cout << "enforce_bc (" << row << ", " << col << ") on b[" << row << "]" << std::endl;
                            op.add_0_for_useless_ghosts(b);
                        }
                    });
            }

            Vec create_solution_vector() const
            {
                Vec x;
                VecCreateSeq(MPI_COMM_SELF, matrix_cols(), &x);
                /*for_each_operator(
                    [&](auto& op, auto row, auto)
                    {
                        if (row == 0)
                        {
                            x_blocks[col] = create_petsc_vector_from(op.unknown());
                        }
                    });*/
                PetscObjectSetName(reinterpret_cast<PetscObject>(x), "solution");
                return x;
            }

            void update_unknowns(Vec& x) const
            {
                for_each_operator(
                    [&](auto& op, auto row, auto)
                    {
                        if (row == 0)
                        {
                            copy(op.col_shift(), x, op.unknown());
                        }
                    });
            }

            template <class... Fields>
            auto tie(Fields&... fields) const
            {
                static constexpr std::size_t n_fields = sizeof...(fields);
                static_assert(n_fields == rows,
                              "The number of fields must correspond to the "
                              "number of rows of the block operator.");

                return std::tuple<Fields&...>(fields...);
            }
        };

        template <int rows, int cols, bool monolithic = false, class... Operators>
        auto make_block_operator(Operators... operators)
        {
            if constexpr (monolithic)
            {
                return MonolithicBlockAssembly<rows, cols, Operators...>(operators...);
            }
            else
            {
                return BlockAssembly<rows, cols, Operators...>(operators...);
            }
        }

    } // end namespace petsc
} // end namespace samurai
