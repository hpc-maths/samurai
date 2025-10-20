// Copyright 2018-2025 the samurai's authors
// SPDX-License-Identifier:  BSD-3-Clause

#pragma once
#include "../print.hpp"
#include "bc.hpp"

namespace samurai
{
    template <class Field, std::size_t stencil_size_>
    struct PolynomialExtrapolation : public Bc<Field>
    {
        INIT_BC(PolynomialExtrapolation, stencil_size_)

        static constexpr std::size_t max_stencil_size_implemented_PE = 6;

        static_assert(stencil_size_ % 2 == 0, "stencil_size must be even.");
        static_assert(stencil_size_ >= 2 && stencil_size_ <= max_stencil_size_implemented_PE);

        apply_function_t get_apply_function(constant_stencil_size_t, const direction_t&) const override
        {
            return [](Field& u, const stencil_cells_t& cells, const value_t&)
            {
                /*
                                u[0]  u[1]  u[2]   ?
                              |_____|_____│_____|_____|
                    cell index   0     1     2     3     (the ghost to fill is always at the last index in 'cell')
                           x =  -3    -2    -1     0     (we arbitrarily set the coordinate x for the extrapolation
                                                          such that the ghost is at x=0)

                    We search the coefficients c[i] of the polynomial P
                          P(x) = c[0]x^2 + c[1]x + c[2]
                    that passes by all the known u[i]. (Note that deg(P) = stencil_size_ - 2)

                    We inverse the Vandermonde system
                        │ (-3)^2  -3  1 │ │c[0]│   │u[0]│
                        │ (-2)^2  -2  1 │ │c[1]│ = │u[1]│.
                        │ (-1)^2  -1  1 │ │c[2]│   │u[2]│
                    This step is done using a symbolic calculus tool.

                    To get the value at x=0, we actually just need c[2]:
                          P(x=0) = c[2].
                */

                const auto& ghost = cells[stencil_size_ - 1];

#ifdef SAMURAI_CHECK_NAN
                for (std::size_t field_i = 0; field_i < Field::n_comp; field_i++)
                {
                    for (std::size_t c = 0; c < stencil_size_ - 1; ++c)
                    {
                        if (std::isnan(field_value(u, cells[c], field_i)))
                        {
                            samurai::io::eprint("NaN detected in [{}] when applying polynomial extrapolation to fill the outer ghost [{}].\n",
                                                fmt::streamed(cells[c]),
                                                fmt::streamed(ghost));
                            // save(fs::current_path(), "nan_extrapolation", {true, true}, u.mesh(), u);
                            exit(1);
                        }
                    }
                }
#endif

                // Last coefficient of the polynomial
                if constexpr (stencil_size_ == 2)
                {
                    u[ghost] = u[cells[0]];
                }
                else if constexpr (stencil_size_ == 4)
                {
                    u[ghost] = u[cells[0]] - u[cells[1]] * 3.0 + u[cells[2]] * 3.0;
                }
                else if constexpr (stencil_size_ == 6)
                {
                    u[ghost] = u[cells[0]] - u[cells[1]] * 5.0 + u[cells[2]] * 1.0E+1 - u[cells[3]] * 1.0E+1 + u[cells[4]] * 5.0;
                }
            };
        }
    };

}
