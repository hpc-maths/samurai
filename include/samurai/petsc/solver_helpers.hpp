#pragma once

#include "linear_block_solver.hpp"
#include "nonlinear_block_solver.hpp"
#include "nonlinear_local_solvers.hpp"
#include "nonlinear_solver.hpp"

namespace samurai
{
    namespace petsc
    {
        /**
         * make_solver
         */

        // Linear solver
        template <class Scheme, std::enable_if_t<Scheme::cfg_t::scheme_type != SchemeType::NonLinear, bool> = true>
        auto make_solver(const Scheme& scheme)
        {
            return LinearSolver<Scheme>(scheme);
        }

        // Linear block solver (choice monolithic or not)
        template <bool monolithic, std::size_t rows, std::size_t cols, class... Operators>
        [[deprecated("Use make_solver<samurai::petsc::BlockAssemblyType::Monolithic/NestedMatrices> instead")]]
        auto make_solver(const BlockOperator<rows, cols, Operators...>& block_operator)
        {
            if constexpr (monolithic)
            {
                return LinearBlockSolver<BlockAssemblyType::Monolithic, rows, cols, Operators...>(block_operator);
            }
            else
            {
                return LinearBlockSolver<BlockAssemblyType::NestedMatrices, rows, cols, Operators...>(block_operator);
            }
        }

        // Linear block solver (choice monolithic or not)
        template <BlockAssemblyType assembly_type, std::size_t rows, std::size_t cols, class... Operators>
            requires(scheme_type_of_block_operator<Operators...>() != SchemeType::NonLinear)
        auto make_solver(const BlockOperator<rows, cols, Operators...>& block_operator)
        {
            return LinearBlockSolver<assembly_type, rows, cols, Operators...>(block_operator);
        }

        // Linear block solver (monolithic)
        template <std::size_t rows, std::size_t cols, class... Operators>
        auto make_solver(const BlockOperator<rows, cols, Operators...>& block_operator)
        {
            return make_solver<BlockAssemblyType::Monolithic, rows, cols, Operators...>(block_operator);
        }

        // Non-linear solver
        template <class Scheme, std::enable_if_t<Scheme::cfg_t::scheme_type == SchemeType::NonLinear, bool> = true>
        auto make_solver(const Scheme& scheme)
        {
            return NonLinearSolver<Scheme>(scheme);
        }

        // Non-linear local solvers
        template <class cfg, class bdry_cfg, std::enable_if_t<cfg::scheme_type == SchemeType::NonLinear && cfg::stencil_size == 1, bool> = true>
        auto make_solver(const CellBasedScheme<cfg, bdry_cfg>& scheme)
        {
            return NonLinearLocalSolvers<CellBasedScheme<cfg, bdry_cfg>>(scheme);
        }

        // Non-linear block solver
        template <BlockAssemblyType assembly_type, std::size_t rows, std::size_t cols, class... Operators>
            requires(scheme_type_of_block_operator<Operators...>() == SchemeType::NonLinear)
        auto make_solver(const BlockOperator<rows, cols, Operators...>& block_operator)
        {
            return NonLinearBlockSolver<assembly_type, rows, cols, Operators...>(block_operator);
        }

        // Non-linear block solver (monolithic)
        template <std::size_t rows, std::size_t cols, class... Operators>
            requires(scheme_type_of_block_operator<Operators...>() == SchemeType::NonLinear)
        auto make_solver(const BlockOperator<rows, cols, Operators...>& block_operator)
        {
            return make_solver<BlockAssemblyType::Monolithic, rows, cols, Operators...>(block_operator);
        }

        /**
         * Solve
         */

        template <class Scheme>
        void solve(const Scheme& scheme, typename Scheme::field_t& unknown, typename Scheme::field_t& rhs)
        {
            auto solver = make_solver(scheme);
            solver.solve(unknown, rhs);
        }

        template <class Scheme, class E>
        void solve(const Scheme& scheme, typename Scheme::field_t& unknown, const field_expression<E>& rhs_expression)
        {
            typename Scheme::field_t rhs = rhs_expression;
            solve(scheme, unknown, rhs);
        }

    } // end namespace petsc
} // end namespace samurai
