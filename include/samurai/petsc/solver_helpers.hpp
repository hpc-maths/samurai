#pragma once

#include "linear_block_solver.hpp"
#include "nonlinear_block_solver.hpp"
#include "nonlinear_local_solvers.hpp"
#include "nonlinear_solver.hpp"

namespace samurai
{
    namespace petsc
    {
        // make_solver ------------------------------------------------------

        /// Linear solver (`LinearSolver`) for a linear scheme.
        template <class Scheme, std::enable_if_t<Scheme::cfg_t::scheme_type != SchemeType::NonLinear, bool> = true>
        auto make_solver(const Scheme& scheme)
        {
            return LinearSolver<Scheme>(scheme);
        }

        /// Linear block solver, monolithic if @p monolithic is true, with nested matrices otherwise.
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

        /// Linear block solver for a linear block operator, assembled as @p assembly_type.
        template <BlockAssemblyType assembly_type, std::size_t rows, std::size_t cols, class... Operators>
            requires(scheme_type_of_block_operator<Operators...>() != SchemeType::NonLinear)
        auto make_solver(const BlockOperator<rows, cols, Operators...>& block_operator)
        {
            return LinearBlockSolver<assembly_type, rows, cols, Operators...>(block_operator);
        }

        /// Linear block solver for a linear block operator, with a monolithic assembly.
        template <std::size_t rows, std::size_t cols, class... Operators>
        auto make_solver(const BlockOperator<rows, cols, Operators...>& block_operator)
        {
            return make_solver<BlockAssemblyType::Monolithic, rows, cols, Operators...>(block_operator);
        }

        /// Non-linear solver (`NonLinearSolver`) for a non-linear scheme.
        template <class Scheme, std::enable_if_t<Scheme::cfg_t::scheme_type == SchemeType::NonLinear, bool> = true>
        auto make_solver(const Scheme& scheme)
        {
            return NonLinearSolver<Scheme>(scheme);
        }

        /// Non-linear local solvers (`NonLinearLocalSolvers`) for a non-linear cell-based scheme
        /// whose stencil is the cell itself (`stencil_size == 1`).
        template <class cfg, class bdry_cfg, std::enable_if_t<cfg::scheme_type == SchemeType::NonLinear && cfg::stencil_size == 1, bool> = true>
        auto make_solver(const CellBasedScheme<cfg, bdry_cfg>& scheme)
        {
            return NonLinearLocalSolvers<CellBasedScheme<cfg, bdry_cfg>>(scheme);
        }

        /// Non-linear block solver for a non-linear block operator, assembled as @p assembly_type.
        template <BlockAssemblyType assembly_type, std::size_t rows, std::size_t cols, class... Operators>
            requires(scheme_type_of_block_operator<Operators...>() == SchemeType::NonLinear)
        auto make_solver(const BlockOperator<rows, cols, Operators...>& block_operator)
        {
            return NonLinearBlockSolver<assembly_type, rows, cols, Operators...>(block_operator);
        }

        /// Non-linear block solver for any block operator, linear ones included, assembled as @p assembly_type.
        template <BlockAssemblyType assembly_type, std::size_t rows, std::size_t cols, class... Operators>
        auto make_nonlinear_solver(const BlockOperator<rows, cols, Operators...>& block_operator)
        {
            return NonLinearBlockSolver<assembly_type, rows, cols, Operators...>(block_operator);
        }

        /// Non-linear block solver for a non-linear block operator, with a monolithic assembly.
        template <std::size_t rows, std::size_t cols, class... Operators>
            requires(scheme_type_of_block_operator<Operators...>() == SchemeType::NonLinear)
        auto make_solver(const BlockOperator<rows, cols, Operators...>& block_operator)
        {
            return make_solver<BlockAssemblyType::Monolithic, rows, cols, Operators...>(block_operator);
        }

        /// Non-linear block solver for any block operator, linear ones included, with a monolithic assembly.
        template <std::size_t rows, std::size_t cols, class... Operators>
        auto make_nonlinear_solver(const BlockOperator<rows, cols, Operators...>& block_operator)
        {
            return make_nonlinear_solver<BlockAssemblyType::Monolithic, rows, cols, Operators...>(block_operator);
        }

        // solve ------------------------------------------------------------

        /// Build the solver of @p scheme with `make_solver` and solve the system for @p unknown
        /// with the right-hand side @p rhs.
        template <class Scheme>
        void solve(const Scheme& scheme, typename Scheme::field_t& unknown, typename Scheme::field_t& rhs)
        {
            auto solver = make_solver(scheme);
            solver.solve(unknown, rhs);
        }

        /// Overload of `solve` taking the right-hand side as a field expression @p rhs_expression.
        template <class Scheme, class E>
        void solve(const Scheme& scheme, typename Scheme::field_t& unknown, const field_expression<E>& rhs_expression)
        {
            typename Scheme::field_t rhs = rhs_expression;
            solve(scheme, unknown, rhs);
        }

    } // end namespace petsc
} // end namespace samurai
