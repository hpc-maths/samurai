// Copyright 2018-2025 the samurai's authors
// SPDX-License-Identifier:  BSD-3-Clause

#pragma once
#include "bc.hpp"

namespace samurai
{
    template <std::size_t order, class Field>
    struct NeumannImpl : public Bc<Field>
    {
        INIT_BC(NeumannImpl, 2 * order) // stencil_size = 2*order

        stencil_t get_stencil(constant_stencil_size_t) const override
        {
            return line_stencil<dim, 0, 2 * order>();
        }

        apply_function_t get_apply_function(constant_stencil_size_t, const direction_t&) const override
        {
            return [](Field& f, const stencil_cells_t& cells, const value_t& value)
            {
                if constexpr (order == 1)
                {
                    static constexpr std::size_t in  = 0;
                    static constexpr std::size_t out = 1;

                    double dx     = f.mesh().cell_length(cells[out].level);
                    f[cells[out]] = dx * value + f[cells[in]];
                }
                else
                {
                    static_assert(order <= 1, "The Neumann boundary conditions are only implemented at the first order.");
                }
            };
        }
    };

    template <std::size_t order = 1>
    struct Neumann
    {
        template <class Field>
        using impl_t = NeumannImpl<order, Field>;
    };

}
