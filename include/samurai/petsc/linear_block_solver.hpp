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
        class LinearBlockSolver
        {
        };

        /**
         * Nested block solver
         */
        template <std::size_t rows_, std::size_t cols_, class... Operators>
        class LinearBlockSolver<false, rows_, cols_, Operators...> : public LinearSolverBase<NestedBlockAssembly<rows_, cols_, Operators...>>
        {
            using assembly_t = NestedBlockAssembly<rows_, cols_, Operators...>;
            using base_class = LinearSolverBase<assembly_t>;
            using base_class::assembly;
            using base_class::m_A;
            using base_class::m_is_set_up;
            using base_class::m_ksp;

            using block_operator_t            = typename assembly_t::scheme_t;
            static constexpr std::size_t rows = assembly_t::rows;
            static constexpr std::size_t cols = assembly_t::cols;

          public:

            static constexpr bool is_monolithic = false;

            explicit LinearBlockSolver(const block_operator_t& block_op)
                : base_class(block_op)
            {
                configure_solver();
            }

          private:

            void configure_solver() override
            {
                KSPCreate(PETSC_COMM_SELF, &m_ksp);
                // KSPSetFromOptions(m_ksp);
            }

          public:

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
                if (m_is_set_up)
                {
                    return;
                }

                if (assembly().undefined_unknown())
                {
                    std::cerr << "Undefined unknowns for this linear system. Please set the unknowns using the instruction '[solver].set_unknowns(u1, u2...);'."
                              << std::endl;
                    assert(false && "Undefined unknowns");
                    exit(EXIT_FAILURE);
                }

                // assembly().reset();
                assembly().create_matrix(m_A);
                assembly().assemble_matrix(m_A);
                PetscObjectSetName(reinterpret_cast<PetscObject>(m_A), "A");
                // MatView(m_A, PETSC_VIEWER_STDOUT_(PETSC_COMM_SELF)); std::cout << std::endl;
                KSPSetOperators(m_ksp, m_A, m_A);

                // Set names to the petsc fields
                PC pc;
                KSPGetPC(m_ksp, &pc);
                IS is_fields[cols];
                MatNestGetISs(m_A, is_fields, NULL);
                auto field_names = assembly().field_names();
                for (std::size_t i = 0; i < cols; ++i)
                {
                    PCFieldSplitSetIS(pc, field_names[i].c_str(), is_fields[i]);
                }

                KSPSetFromOptions(m_ksp);
                PCSetUp(pc);
                // KSPSetUp(m_ksp); // Here, PETSc fails for some reason.

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

                VecDestroy(&b);
                VecDestroy(&x);
            }
        };

        /**
         * Monolithic block solver
         */
        template <std::size_t rows_, std::size_t cols_, class... Operators>
        class LinearBlockSolver<true, rows_, cols_, Operators...>
            : public LinearSolverBase<MonolithicBlockAssembly<rows_, cols_, Operators...>>
        {
            using assembly_t = MonolithicBlockAssembly<rows_, cols_, Operators...>;
            using base_class = LinearSolverBase<assembly_t>;
            using base_class::assembly;
            using base_class::m_A;
            using base_class::m_is_set_up;
            using base_class::m_ksp;

            using block_operator_t = typename assembly_t::scheme_t;

            static constexpr std::size_t rows = assembly_t::rows;
            static constexpr std::size_t cols = assembly_t::cols;

          public:

            static constexpr bool is_monolithic = true;

            explicit LinearBlockSolver(const block_operator_t& block_op)
                : base_class(block_op)
            {
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

            template <class... Fields>
            void solve(Fields&... rhs_fields)
            {
                auto rhs_tuple = assembly().block_operator().tie_rhs(rhs_fields...);
                // static_assert(std::tuple_size_v<RHSTuple> == rows,
                //                   "The number of source fields passed to solve() must equal "
                //                   "the number of rows of the block operator.");

                if (!m_is_set_up)
                {
                    this->setup();
                }

                Vec b = assembly().create_rhs_vector(rhs_tuple);
                Vec x = assembly().create_solution_vector();
                this->prepare_rhs_and_solve(b, x);

                assembly().update_unknowns(x);

                VecDestroy(&b);
                VecDestroy(&x);
            }
        };

    } // end namespace petsc
} // end namespace samurai
