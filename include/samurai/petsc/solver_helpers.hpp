#pragma once

#include "linear_solver.hpp"
#include "nonlinear_local_solvers.hpp"
#include "nonlinear_solver.hpp"

namespace samurai
{
    namespace petsc
    {
        /**
         * make_solver
         */

        // Single field linear solver
        template <class Scheme, std::enable_if_t<Scheme::cfg_t::scheme_type != SchemeType::NonLinear, bool> = true>
        auto make_solver(const Scheme& scheme)
        {
            return SingleFieldSolver<Assembly<Scheme>>(scheme);
        }

        // Linear block solver
        template <bool monolithic, std::size_t rows, std::size_t cols, class... Operators>
        auto make_solver(const BlockOperator<rows, cols, Operators...>& block_operator)
        {
            return BlockSolver<monolithic, rows, cols, Operators...>(block_operator);
        }

        // Linear block solver (monolithic)
        template <std::size_t rows, std::size_t cols, class... Operators>
        auto make_solver(const BlockOperator<rows, cols, Operators...>& block_operator)
        {
            static constexpr bool default_monolithic = true;
            return make_solver<default_monolithic, rows, cols, Operators...>(block_operator);
        }

        // Single field non-linear solver
        template <class Scheme, std::enable_if_t<Scheme::cfg_t::scheme_type == SchemeType::NonLinear, bool> = true>
        auto make_solver(const Scheme& scheme)
        {
            return SingleFieldNonLinearSolver<Assembly<Scheme>>(scheme);
        }

        // Single field non-linear local solvers
        template <class cfg, class bdry_cfg, std::enable_if_t<cfg::scheme_type == SchemeType::NonLinear && cfg::scheme_stencil_size == 1, bool> = true>
        auto make_solver(const CellBasedScheme<cfg, bdry_cfg>& scheme)
        {
            return NonLinearLocalSolvers<CellBasedScheme<cfg, bdry_cfg>>(scheme);
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
