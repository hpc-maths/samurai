// Copyright 2018-2025 the samurai's authors
// SPDX-License-Identifier:  BSD-3-Clause

#pragma once

#include <cassert>
#include <utility>

/**
 * Weight policies.
 *
 * A weight policy is any callable `double(const cell_t&)` that returns the
 * non-negative cost of a cell, so that the strategies balance work instead of
 * cell counts. It is evaluated on local cells only and never communicates.
 */
namespace samurai::load_balancing::weight
{
    /// Uniform weight: every cell costs 1, so balancing work means balancing
    /// cell counts.
    inline auto uniform()
    {
        return [](const auto& /*cell*/)
        {
            return 1.0;
        };
    }

    /**
     * Level-dependent weight: the cost of a cell is `f(cell.level)`.
     *
     * For example, with an explicit time scheme and local time stepping, a
     * cell of level `l` is updated `2^(l - min_level)` times more often than
     * a cell at `min_level`:
     * @code
     * auto w = lb::weight::per_level([&](std::size_t l) {
     *     return std::pow(2.0, static_cast<double>(l - mesh.min_level()));
     * });
     * @endcode
     *
     * @warning Keep the growth of `f` moderate: an over-aggressive law such as
     *          `1 << (l * l)` overflows from level 8 and makes fine cells
     *          completely dominate the balance.
     */
    template <class F>
    auto per_level(F&& f)
    {
        return [f = std::forward<F>(f)](const auto& cell)
        {
            const auto w = static_cast<double>(f(cell.level));
            assert(w >= 0. && "per_level weight must be non-negative");
            return w;
        };
    }

    /**
     * Application-defined weight read from a scalar field, such as the number
     * of particles per cell or the measured cost of a cell.
     *
     * The field is captured by reference: it must outlive every use of the
     * returned policy (typically the call to `load_balance()` or
     * `required()`). Its values must be non-negative, which an assertion
     * checks in Debug builds.
     */
    template <class Field>
    auto from_field(const Field& w)
    {
        return [&w](const auto& cell)
        {
            assert(w[cell] >= 0 && "from_field weight must be non-negative");
            return static_cast<double>(w[cell]);
        };
    }
}
