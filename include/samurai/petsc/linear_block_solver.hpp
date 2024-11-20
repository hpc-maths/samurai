#pragma once

#include "block_assembly.hpp"
#include "linear_solver.hpp"

namespace samurai
{
    namespace petsc
    {
        /**
         * Block solver
         */
        template <bool monolithic, std::size_t rows_, std::size_t cols_, class... Operators>
        class LinearBlockSolver : public LinearSolverBase<BlockAssembly<monolithic, rows_, cols_, Operators...>>
        {
            using assembly_t = BlockAssembly<monolithic, rows_, cols_, Operators...>;
            using base_class = LinearSolverBase<assembly_t>;
            using base_class::m_A;
            using base_class::m_is_set_up;
            using base_class::m_ksp;

          public:

            using base_class::assembly;

          private:

            using block_operator_t            = typename assembly_t::scheme_t;
            static constexpr std::size_t rows = assembly_t::rows;
            static constexpr std::size_t cols = assembly_t::cols;

          public:

            static constexpr bool is_monolithic = monolithic;

            explicit LinearBlockSolver(const block_operator_t& block_op)
                : base_class(block_op)
            {
                _configure_solver();
            }

          private:

            void _configure_solver()
            {
                KSPCreate(PETSC_COMM_SELF, &m_ksp);
                KSPSetFromOptions(m_ksp);
            }

          public:

            void configure_solver() override
            {
                _configure_solver();
            }

            void set_pc_fieldsplit(PC& pc)
            {
                PCSetType(pc, PCFIELDSPLIT);
                if constexpr (is_monolithic)
                {
                    auto IS_fields   = assembly().create_fields_IS();
                    auto field_names = assembly().field_names();
                    for (std::size_t i = 0; i < cols; ++i)
                    {
                        PCFieldSplitSetIS(pc, field_names[i].c_str(), IS_fields[i]);
                        ISDestroy(&IS_fields[i]);
                    }
                }
                else
                {
                    if (m_A == nullptr)
                    {
                        std::cerr << "The matrix must be assemble before calling set_pc_fieldsplit()." << std::endl;
                        exit(EXIT_FAILURE);
                    }
                    IS IS_fields[cols];
                    MatNestGetISs(m_A, IS_fields, NULL);
                    auto field_names = assembly().field_names();
                    for (std::size_t i = 0; i < cols; ++i)
                    {
                        PCFieldSplitSetIS(pc, field_names[i].c_str(), IS_fields[i]);
                    }
                }
            }

            template <class... Fields>
            void set_unknowns(Fields&... unknowns)
            {
                auto unknown_tuple = assembly().block_operator().tie_unknowns(unknowns...);
                set_unknown(unknown_tuple);
            }

            template <class UnknownTuple>
            void set_unknown(UnknownTuple& unknown_tuple)
            {
                static_assert(std::tuple_size_v<UnknownTuple> == cols,
                              "The number of unknown fields passed to solve() must equal "
                              "the number of columns of the block operator.");
                assembly().set_unknown(unknown_tuple);
                assembly().reset();
            }

            void setup() override
            {
                if constexpr (is_monolithic)
                {
                    base_class::setup();
                    return;
                }

                if (m_is_set_up)
                {
                    return;
                }

                this->assemble_matrix();
                // MatView(m_A, PETSC_VIEWER_STDOUT_(PETSC_COMM_SELF)); std::cout << std::endl;
                KSPSetOperators(m_ksp, m_A, m_A);

                PC pc;
                KSPGetPC(m_ksp, &pc);

                KSPSetFromOptions(m_ksp);
                PCSetUp(pc);
                // KSPSetUp(m_ksp); // PETSc fails at KSPSolve() for some reason.

                m_is_set_up = true;
            }

            template <class... Fields>
            void solve(Fields&... rhs_fields)
            {
                auto rhs_tuple = assembly().block_operator().tie_rhs(rhs_fields...);
                // static_assert(std::tuple_size_v<RHSTuple> == rows,
                //                   "The number of source fields passed to solve() must equal "
                //                   "the number of rows of the block operator.");

                if (!m_is_set_up)
                {
                    setup();
                }
                Vec b = assembly().create_rhs_vector(rhs_tuple);
                Vec x = assembly().create_solution_vector();
                this->prepare_rhs_and_solve(b, x);

                if constexpr (is_monolithic)
                {
                    assembly().update_unknowns(x);
                }

                VecDestroy(&b);
                VecDestroy(&x);
            }
        };

    } // end namespace petsc
} // end namespace samurai
