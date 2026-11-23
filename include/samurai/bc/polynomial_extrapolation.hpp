// Copyright 2018-2025 the samurai's authors
// SPDX-License-Identifier:  BSD-3-Clause

#pragma once
#include "bc.hpp"
#include "polynomial_extrapolation_formula.hpp"

namespace samurai
{
    template <class Field, std::size_t stencil_size_>
    struct PolynomialExtrapolation : public Bc<Field>
    {
        INIT_BC(PolynomialExtrapolation, stencil_size_)

        static constexpr std::size_t max_stencil_size_implemented_PE = detail::max_stencil_size_implemented_PE;

        /**
         * Deepest outer ghost layer the extrapolation reaches, over all the implemented
         * stencil sizes. The stencil ends on the ghost it fills, so it grows with the layer
         * until it hits @c max_stencil_size_implemented_PE and then slides outward at
         * constant size, resting on the ghosts the shallower layers have filled. Sliding
         * keeps making sense only while the stencil still covers the boundary cell, which
         * holds up to the layer @c max_stencil_size_implemented_PE - 1.
         */
        static constexpr std::size_t max_ghost_layers_implemented_PE = detail::max_ghost_layers_implemented_PE;

        static_assert(stencil_size_ % 2 == 0, "stencil_size must be even.");
        static_assert(stencil_size_ >= 2 && stencil_size_ <= max_stencil_size_implemented_PE);

        apply_function_t get_apply_function(constant_stencil_size_t, const direction_t&) const override
        {
            return [](Field& u, const stencil_cells_t& cells, const value_t&)
            {
                extrapolate(u, cells);
            };
        }

        static void extrapolate(Field& u, const stencil_cells_t& cells)
        {
            // The value of the polynomial through the other cells, at the ghost (see detail::extrapolated_value)
            const auto& ghost = cells[stencil_size_ - 1];

#ifdef SAMURAI_CHECK_NAN
            for (std::size_t field_i = 0; field_i < Field::n_comp; field_i++)
            {
                for (std::size_t c = 0; c < stencil_size_ - 1; ++c)
                {
                    if (std::isnan(field_value(u, cells[c], field_i)))
                    {
                        std::cerr << "NaN detected in [" << cells[c] << "] when applying polynomial extrapolation to fill the outer ghost ["
                                  << ghost << "]." << std::endl;
                        // save(fs::current_path(), "nan_extrapolation", {true, true}, u.mesh(), u);
                        exit(1);
                    }
                }
            }
#endif

            for (std::size_t field_i = 0; field_i < Field::n_comp; ++field_i)
            {
                field_value(u, ghost, field_i) = detail::extrapolated_value<stencil_size_>(
                    [&](std::size_t k)
                    {
                        return field_value(u, cells[k], field_i);
                    });
            }
        }
    };

}
