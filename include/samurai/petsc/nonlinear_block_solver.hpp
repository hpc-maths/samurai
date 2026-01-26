#pragma once

#include "block_assembly.hpp"
#include "nonlinear_solver.hpp"

namespace samurai
{
    namespace petsc
    {
        /**
         * Block solver
         */
        template <BlockAssemblyType assembly_type_, std::size_t rows_, std::size_t cols_, class... Operators>
        class NonLinearBlockSolver : public NonLinearSolverBase<BlockAssembly<assembly_type_, rows_, cols_, Operators...>>
        {
            using assembly_t = BlockAssembly<assembly_type_, rows_, cols_, Operators...>;
            using base_class = NonLinearSolverBase<assembly_t>;
            using base_class::m_is_set_up;
            using base_class::m_J;
            using base_class::m_snes;
            using base_class::m_worker_output_field;

          public:

            using base_class::assembly;
            using base_class::configure;

          private:

            using block_operator_t            = typename assembly_t::scheme_t;
            static constexpr std::size_t rows = assembly_t::rows;
            static constexpr std::size_t cols = assembly_t::cols;

          public:

            static constexpr BlockAssemblyType assembly_type = assembly_type_; // cppcheck-suppress unusedStructMember

            explicit NonLinearBlockSolver(const block_operator_t& block_op)
                : base_class(block_op)
            {
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
            }

            template <class... Fields>
            void solve(Fields&... rhs_fields)
            {
                auto rhs_tuple = assembly().block_operator().tie_rhs(rhs_fields...);

                static_for<0, rows>::apply(
                    [&](auto row)
                    {
                        auto& worker_field = std::get<row>(m_worker_output_field);
                        auto& rhs_field    = std::get<row>(rhs_tuple);

                        using field_t = std::decay_t<decltype(rhs_field)>;

                        worker_field = field_t(fmt::format("worker_output_{}", rhs_field.name()), rhs_field.mesh());
                    });

                if (!m_is_set_up)
                {
                    this->setup();
                }
                Vec b = assembly().create_rhs_vector(rhs_tuple);
                Vec x = assembly().create_solution_vector_from_unknown_fields();

                assembly().set_0_for_all_ghosts(x);
                this->prepare_rhs(x, b);

                this->solve_system(x, b);

#ifdef SAMURAI_WITH_MPI
                assembly().update_unknowns(x);
#else
                if constexpr (assembly_type == BlockAssemblyType::Monolithic)
                {
                    assembly().update_unknowns(x);
                }
#endif
                VecDestroy(&b);
                VecDestroy(&x);
            }

            void set_block_operator(const block_operator_t& block_op)
            {
                this->set_scheme(block_op);
            }
        };

    } // end namespace petsc
} // end namespace samurai
