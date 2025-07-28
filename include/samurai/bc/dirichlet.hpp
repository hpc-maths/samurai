// Copyright 2018-2025 the samurai's authors
// SPDX-License-Identifier:  BSD-3-Clause

#pragma once
#include "bc.hpp"

namespace samurai
{
    template <std::size_t order, class Field>
    struct DirichletImpl : public Bc<Field>
    {
        INIT_BC(DirichletImpl, 2 * order) // stencil_size = 2*order

        stencil_t get_stencil(constant_stencil_size_t) const override
        {
            return line_stencil<dim, 0, 2 * order>();
        }

        apply_function_t get_apply_function(constant_stencil_size_t, const direction_t&) const override
        {
            return [](Field& u, const stencil_cells_t& cells, const value_t& dirichlet_value)
            {
                if constexpr (order == 1)
                {
                    //      [0]   [1]
                    //    |_____|.....|
                    //     cell  ghost

                    u[cells[1]] = 2 * dirichlet_value - u[cells[0]];
                }
                else if constexpr (order == 2)
                {
                    //     [0]   [1]   [2]   [3]
                    //   |_____|_____|.....|.....|
                    //       cells      ghosts

                    // We define a polynomial of degree 2 that passes by 3 points (the 2 cells and the boundary value):
                    //                       p(x) = a*x^2 + b*x + c.
                    // The coefficients a, b, c are found by inverting the Vandermonde matrix obtained by inserting the 3 points into
                    // the polynomial. If we set the abscissa 0 at the center of cells[0], this system reads
                    //                       p( 0 ) = u[cells[0]]
                    //                       p( 1 ) = u[cells[1]]
                    //                       p(3/2) = dirichlet_value.
                    // Then, we want that the ghost values be also located on this polynomial, i.e.
                    //                       u[cells[2]] = p( 2 )
                    //                       u[cells[3]] = p( 3 ).

                    u[cells[2]] = 8. / 3. * dirichlet_value + 1. / 3. * u[cells[0]] - 2. * u[cells[1]];
                    u[cells[3]] = 8. * dirichlet_value + 2. * u[cells[0]] - 9. * u[cells[1]];
                }
                else if constexpr (order == 3)
                {
                    //     [0]   [1]   [2]   [3]   [4]   [5]
                    //   |_____|_____|_____|.....|.....|.....|
                    //          cells             ghosts

                    // We define a polynomial of degree 3 that passes by 4 points (the 3 cells and the boundary value):
                    //                       p(x) = a*x^3 + b*x^2 + c*x + d.
                    // The coefficients a, b, c, d are found by inverting the Vandermonde matrix obtained by inserting the 4 points into
                    // the polynomial. If we set the abscissa 0 at the center of cells[0], this system reads
                    //                       p( 0 ) = u[cells[0]]
                    //                       p( 1 ) = u[cells[1]]
                    //                       p( 2 ) = u[cells[2]]
                    //                       p(5/2) = dirichlet_value.
                    // Then, we want that the ghost values be also located on this polynomial, i.e.
                    //                       u[cells[3]] = p( 3 )
                    //                       u[cells[4]] = p( 4 )
                    //                       u[cells[5]] = p( 5 ).

                    u[cells[3]] = 16. / 5. * dirichlet_value - 1. / 5. * u[cells[0]] + u[cells[1]] - 3. * u[cells[2]];
                    u[cells[4]] = 64. / 5. * dirichlet_value - 9. / 5. * u[cells[0]] + 8. * u[cells[1]] - 18. * u[cells[2]];
                    u[cells[5]] = 32. * dirichlet_value - 6. * u[cells[0]] + 25. * u[cells[1]] - 50. * u[cells[2]];
                }
                else if constexpr (order == 4)
                {
                    u[cells[4]] = 128. / 35 * dirichlet_value + 1. / 7 * u[cells[0]] - 4. / 5 * u[cells[1]] + 2 * u[cells[2]]
                                - 4. * u[cells[3]];
                    u[cells[5]] = 128. / 7. * dirichlet_value + 12. / 7. * u[cells[0]] - 9 * u[cells[1]] + 20 * u[cells[2]]
                                - 30 * u[cells[3]];
                    u[cells[6]] = 384. / 7. * dirichlet_value + 50. / 7. * u[cells[0]] - 36 * u[cells[1]] + 75 * u[cells[2]]
                                - 100 * u[cells[3]];
                    u[cells[7]] = 128 * dirichlet_value + 20 * u[cells[0]] - 98 * u[cells[1]] + 196 * u[cells[2]] - 245 * u[cells[3]];
                }
                else
                {
                    static_assert(order <= 4, "The Dirichlet boundary conditions are only implemented up to order 4.");
                }
            };
        }
    };

    template <std::size_t order = 1>
    struct Dirichlet
    {
        template <class Field>
        using impl_t = DirichletImpl<order, Field>;
    };

}
