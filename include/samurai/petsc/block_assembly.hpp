#pragma once
#include "../schemes/block_operator.hpp"
#include "utils.hpp"
#include "zero_block_assembly.hpp"

namespace samurai
{
    namespace petsc
    {
        template <std::size_t rows_, std::size_t cols_, class... Operators>
        class BlockAssembly
        {
          public:

            using block_operator_t            = BlockOperator<rows_, cols_, Operators...>;
            static constexpr std::size_t rows = block_operator_t::rows;
            static constexpr std::size_t cols = block_operator_t::cols;
            using scheme_t                    = block_operator_t;

          private:

            const block_operator_t* m_block_operator;
            std::tuple<Assembly<Operators>...> m_assembly_ops;

          public:

            explicit BlockAssembly(const block_operator_t& block_op)
                : m_block_operator(&block_op)
                , m_assembly_ops(transform(block_op.operators(),
                                           [](const auto& op)
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

            // template <class OperatorType>
            // static Assembly<OperatorType> to_assembly(OperatorType& op)
            // {
            //     return Assembly<OperatorType>(op);
            // }

            auto& block_operator()
            {
                return *m_block_operator;
            }

            auto& block_operator() const
            {
                return *m_block_operator;
            }

            template <class Func>
            void for_each_assembly_op(Func&& f)
            {
                std::size_t i = 0;
                for_each(m_assembly_ops,
                         [&](auto& op)
                         {
                             auto row = i / cols;
                             auto col = i % cols;
                             f(op, row, col);
                             i++;
                         });
            }

            template <class Func>
            void for_each_assembly_op(Func&& f) const
            {
                std::size_t i = 0;
                for_each(m_assembly_ops,
                         [&](const auto& op)
                         {
                             auto row = i / cols;
                             auto col = i % cols;
                             f(op, row, col);
                             i++;
                         });
            }

            template <std::size_t row, std::size_t col>
            auto& get()
            {
                return std::get<row * cols + col>(m_assembly_ops);
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
                        std::size_t i = 0;
                        for_each(unknowns,
                                 [&](auto& u)
                                 {
                                     if (col == i)
                                     {
                                         // Verify type compatibility only if scheme_t != void (used for zero block)
                                         if constexpr (!std::is_same_v<typename std::decay_t<decltype(op)>::scheme_t, int>)
                                         {
                                             if constexpr (std::is_same_v<std::decay_t<decltype(u)>,
                                                                          typename std::decay_t<decltype(op)>::scheme_t::field_t>)
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
                        undefined = undefined || !op.unknown_ptr();
                    });
                return undefined;
            }

            std::array<std::string, cols> field_names() const
            {
                std::array<std::string, cols> names;
                for_each_assembly_op(
                    [&](auto& op, auto row, auto col)
                    {
                        if (row == col)
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
                        if (row == col && op.matrix_rows() != op.matrix_cols())
                        {
                            std::cerr << "Function 'create_vector()' is ambiguous in this context, because the block (" << row << ", " << col
                                      << ") is not square. Use 'create_applicable_vector()' or 'create_rhs_vector()' instead." << std::endl;
                            exit(EXIT_FAILURE);
                        }
                    });
            }
        };

        /**
         * Assemble block matrix using PETSc nested matrices.
         */
        template <std::size_t rows_, std::size_t cols_, class... Operators>
        class NestedBlockAssembly : public BlockAssembly<rows_, cols_, Operators...>
        {
            using base_class = BlockAssembly<rows_, cols_, Operators...>;

          public:

            using base_class::block_operator;
            using block_operator_t = typename base_class::block_operator_t;
            using base_class::cols;
            using base_class::for_each_assembly_op;
            using base_class::rows;

          private:

            std::array<Mat, rows * cols> m_blocks;

          public:

            explicit NestedBlockAssembly(const block_operator_t& block_op)
                : base_class(block_op)
            {
                for_each_assembly_op(
                    [&](auto&, auto row, auto col)
                    {
                        block(row, col) = nullptr;
                    });
            }

            void create_matrix(Mat& A)
            {
                for_each_assembly_op(
                    [&](auto& op, auto row, auto col)
                    {
                        // std::cout << "create_matrix (" << row << ", " << col << ")" << std::endl;
                        op.create_matrix(block(row, col));
                    });
                MatCreateNest(PETSC_COMM_SELF, rows, PETSC_IGNORE, cols, PETSC_IGNORE, m_blocks.data(), &A);
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

            void reset()
            {
                for_each_assembly_op(
                    [&](auto& op, auto, auto)
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
                for_each_assembly_op(
                    [&](auto& op, auto row, auto)
                    {
                        if (op.include_bc())
                        {
                            // std::cout << "enforce_bc (" << row << ", " << col << ") on b[" << row << "]" << std::endl;
                            Vec b_block;
                            VecNestGetSubVec(b, static_cast<PetscInt>(row), &b_block);
                            op.enforce_bc(b_block);
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

            Vec create_solution_vector() const
            {
                std::array<Vec, cols> x_blocks;
                for_each_assembly_op(
                    [&](auto& op, auto row, auto col)
                    {
                        if (row == 0)
                        {
                            x_blocks[col] = create_petsc_vector_from(op.unknown());
                            PetscObjectSetName(reinterpret_cast<PetscObject>(x_blocks[col]), op.unknown().name().c_str());
                        }
                    });
                Vec x;
                VecCreateNest(PETSC_COMM_SELF, cols, NULL, x_blocks.data(), &x);
                PetscObjectSetName(reinterpret_cast<PetscObject>(x), "solution");
                return x;
            }

            template <class... Fields>
            Vec create_applicable_vector(const std::tuple<Fields&...>& fields) const
            {
                std::array<Vec, cols> x_blocks;
                std::size_t i = 0;
                for_each(fields,
                         [&](auto& f)
                         {
                             for_each_assembly_op(
                                 [&](auto&, auto row, auto col)
                                 {
                                     if (row == 0 && col == i)
                                     {
                                         x_blocks[col] = create_petsc_vector_from(f);
                                         //  if constexpr (has_name<decltype(f)>())
                                         //  {
                                         //      PetscObjectSetName(reinterpret_cast<PetscObject>(x_blocks[col]), f.name().c_str());
                                         //  }
                                     }
                                 });
                             i++;
                         });
                Vec x;
                VecCreateNest(PETSC_COMM_SELF, cols, NULL, x_blocks.data(), &x);
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
        };

        /**
         * Assemble block matrix as a monolithic matrix.
         */
        template <std::size_t rows_, std::size_t cols_, class... Operators>
        class MonolithicBlockAssembly : public BlockAssembly<rows_, cols_, Operators...>,
                                        public MatrixAssembly
        {
            using base_class = BlockAssembly<rows_, cols_, Operators...>;

          public:

            using base_class::block_operator;
            using block_operator_t = typename base_class::block_operator_t;
            using base_class::cols;
            using base_class::for_each_assembly_op;
            using base_class::rows;

            explicit MonolithicBlockAssembly(const block_operator_t& block_op)
                : base_class(block_op)
            {
                this->set_name("(unnamed monolithic block operator)");

                for_each_assembly_op(
                    [&](auto& op, auto, auto)
                    {
                        op.is_block(true);
                    });
            }

            void reset() override
            {
                for_each_assembly_op(
                    [&](auto& op, auto, auto)
                    {
                        op.reset();
                    });

                // Check compatibility of dimensions and set dimensions for blocks that must fit into the matrix
                // - rows:
                for (std::size_t r = 0; r < rows; r++)
                {
                    PetscInt block_rows = 0;
                    for_each_assembly_op(
                        [&](auto& op, auto row, auto)
                        {
                            if (row == r)
                            {
                                if (block_rows == 0 && !op.fit_block_dimensions())
                                {
                                    block_rows = op.matrix_rows();
                                }
                            }
                        });
                    for_each_assembly_op(
                        [&](auto& op, auto row, auto col)
                        {
                            if (row == r)
                            {
                                if (op.fit_block_dimensions())
                                {
                                    op.set_matrix_rows(block_rows);
                                }
                                else if (op.matrix_rows() != block_rows)
                                {
                                    std::cerr << "Assembly failure: incompatible number of rows of block (" << row << ", " << col
                                              << "): " << op.matrix_rows() << " (expected " << block_rows << ")" << std::endl;
                                    exit(EXIT_FAILURE);
                                }
                            }
                        });
                }
                // - cols:
                for (std::size_t c = 0; c < cols; c++)
                {
                    PetscInt block_cols = 0;
                    for_each_assembly_op(
                        [&](auto& op, auto, auto col)
                        {
                            if (col == c)
                            {
                                if (block_cols == 0 && !op.fit_block_dimensions())
                                {
                                    block_cols = op.matrix_cols();
                                }
                            }
                        });
                    for_each_assembly_op(
                        [&](auto& op, auto row, auto col)
                        {
                            if (col == c)
                            {
                                if (op.fit_block_dimensions())
                                {
                                    op.set_matrix_cols(block_cols);
                                }
                                else if (op.matrix_cols() != block_cols)
                                {
                                    std::cerr << "Assembly failure: incompatible number of columns of block (" << row << ", " << col
                                              << "): " << op.matrix_cols() << " (expected " << block_cols << ")" << std::endl;
                                    exit(EXIT_FAILURE);
                                }
                            }
                        });
                }

                // Set row_shift and col_shift
                PetscInt row_shift = 0;
                PetscInt col_shift = 0;
                for_each_assembly_op(
                    [&](auto& op, auto, auto col)
                    {
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

          public:

            PetscInt matrix_rows() const override
            {
                PetscInt total_rows = 0;
                for_each_assembly_op(
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
                for_each_assembly_op(
                    [&](auto& op, auto row, auto col)
                    {
                        if (row == col)
                        {
                            total_cols += op.matrix_cols();
                        }
                    });
                return total_cols;
            }

            void sparsity_pattern_scheme(std::vector<PetscInt>& nnz) const override
            {
                for_each_assembly_op(
                    [&](auto& op, auto, auto)
                    {
                        op.sparsity_pattern_scheme(nnz);
                    });
            }

            void sparsity_pattern_boundary(std::vector<PetscInt>& nnz) const override
            {
                for_each_assembly_op(
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
                for_each_assembly_op(
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
                for_each_assembly_op(
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
                for_each_assembly_op(
                    [&](auto& op, auto, auto)
                    {
                        if (op.must_insert_value_on_diag_for_useless_ghosts())
                        {
                            op.sparsity_pattern_useless_ghosts(nnz);
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
                        // std::cout << "assemble_scheme (" << row << ", " << col << ") '" << op.name() << "'" << std::endl;
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
                        if (row == col)
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
                        if (row == col)
                        {
                            op.for_each_useless_ghost_row(std::forward<Func>(f));
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
                             this->for_each_assembly_op(
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

            template <class... Fields>
            Vec create_applicable_vector(const std::tuple<Fields&...>& fields) const
            {
                Vec x;
                VecCreateSeq(MPI_COMM_SELF, matrix_cols(), &x);
                std::size_t i = 0;
                for_each(fields,
                         [&](auto& f)
                         {
                             this->for_each_assembly_op(
                                 [&](auto& op, auto row, auto col)
                                 {
                                     if (row == 0 && col == i)
                                     {
                                         copy(f, x, op.col_shift());
                                     }
                                 });
                             i++;
                         });
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
                    [&](auto& op, auto, auto)
                    {
                        if (op.include_bc())
                        {
                            // std::cout << "enforce_bc (" << row << ", " << col << ") on b[" << row << "]" << std::endl;
                            op.enforce_bc(b);
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
                for_each_assembly_op(
                    [&](auto& op, auto row, auto)
                    {
                        if (row == 0)
                        {
                            copy(op.col_shift(), x, op.unknown());
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
        };

        template <bool monolithic, std::size_t rows_, std::size_t cols_, class... Operators>
        auto make_assembly(const BlockOperator<rows_, cols_, Operators...>& block_op)
        {
            if constexpr (monolithic)
            {
                return MonolithicBlockAssembly<rows_, cols_, Operators...>(block_op);
            }
            else
            {
                return NestedBlockAssembly<rows_, cols_, Operators...>(block_op);
            }
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
