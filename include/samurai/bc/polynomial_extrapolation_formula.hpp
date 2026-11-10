// Copyright 2018-2025 the samurai's authors
// SPDX-License-Identifier:  BSD-3-Clause

#pragma once

#include <cstddef>

namespace samurai::detail
{
    // Largest stencil PolynomialExtrapolation implements, and the deepest ghost
    // layer it can fill (see PolynomialExtrapolation).
    inline constexpr std::size_t max_stencil_size_implemented_PE = 6;
    inline constexpr std::size_t max_ghost_layers_implemented_PE = max_stencil_size_implemented_PE - 1;

    /*
                    u(0)  u(1)  u(2)   ?
                  |_____|_____│_____|_____|
        cell index   0     1     2     3     (the ghost to fill is always the last cell of the stencil)
               x =  -3    -2    -1     0     (we arbitrarily set the coordinate x for the extrapolation
                                              such that the ghost is at x=0)

        We search the coefficients c[i] of the polynomial P
              P(x) = c[0]x^2 + c[1]x + c[2]
        that passes by all the known u(i). (Note that deg(P) = stencil_size - 2)

        We inverse the Vandermonde system
            │ (-3)^2  -3  1 │ │c[0]│   │u(0)│
            │ (-2)^2  -2  1 │ │c[1]│ = │u(1)│.
            │ (-1)^2  -1  1 │ │c[2]│   │u(2)│
        This step is done using a symbolic calculus tool.

        To get the value at x=0, we actually just need c[2]:
              P(x=0) = c[2].

       u(k) is the value of one component in the cell k of the stencil. The field
       path and the tables of the mesh both call this function, so that they round
       the same way whatever the compiler contracts into fused multiply-adds.
    */
    template <std::size_t stencil_size, class Get>
    auto extrapolated_value(Get&& u)
    {
        static_assert(stencil_size == 2 || stencil_size == 4 || stencil_size == 6);
        if constexpr (stencil_size == 2)
        {
            return u(0);
        }
        else if constexpr (stencil_size == 4)
        {
            return u(0) - u(1) * 3.0 + u(2) * 3.0;
        }
        else
        {
            return u(0) - u(1) * 5.0 + u(2) * 1.0E+1 - u(3) * 1.0E+1 + u(4) * 5.0;
        }
    }
}
