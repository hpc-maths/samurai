#pragma once
#include "FV_scheme.hpp"

namespace samurai
{
    //-----------------------------------------//
    //          Useful flux functions          //
    //-----------------------------------------//

    /**
     *   |---------|--------|
     *   |         |        |
     *   | cell 0  | cell 1 |
     *   |         |        |
     *   |---------|--------|
     *          ------->
     *        normal flux
     */

    //---------------//
    // Linear fluxes //
    //---------------//

    template <class Field>
    auto get_normal_grad_order1_coeffs(double h)
    {
        static constexpr std::size_t field_size             = Field::size;
        static constexpr std::size_t flux_output_field_size = field_size;
        static constexpr std::size_t stencil_size           = 2;
        using flux_computation_t    = NormalFluxDefinition<FluxType::LinearHomogeneous, Field, flux_output_field_size, stencil_size>;
        using flux_stencil_coeffs_t = typename flux_computation_t::flux_stencil_coeffs_t;

        flux_stencil_coeffs_t coeffs;
        if constexpr (field_size == 1)
        {
            coeffs[0] = -1 / h;
            coeffs[1] = 1 / h;
        }
        else
        {
            coeffs[0].fill(0);
            coeffs[1].fill(0);
            for (std::size_t i = 0; i < field_size; ++i)
            {
                coeffs[0](i, i) = -1 / h;
                coeffs[1](i, i) = 1 / h;
            }
        }
        return coeffs;
    }

    template <class Field>
    auto get_average_coeffs(double)
    {
        static constexpr std::size_t field_size             = Field::size;
        static constexpr std::size_t flux_output_field_size = field_size;
        static constexpr std::size_t stencil_size           = 2;
        using flux_computation_t    = NormalFluxDefinition<FluxType::LinearHomogeneous, Field, flux_output_field_size, stencil_size>;
        using flux_stencil_coeffs_t = typename flux_computation_t::flux_stencil_coeffs_t;

        flux_stencil_coeffs_t coeffs;
        if constexpr (field_size == 1)
        {
            coeffs[0] = 0.5;
            coeffs[1] = 0.5;
        }
        else
        {
            coeffs[0].fill(0);
            coeffs[1].fill(0);
            for (std::size_t i = 0; i < field_size; ++i)
            {
                coeffs[0](i, i) = 0.5;
                coeffs[1](i, i) = 0.5;
            }
        }
        return coeffs;
    }

    //-------------------//
    // Non-linear fluxes //
    //-------------------//

    template <class Field, class Cell = typename Field::cell_t>
    auto get_average_value(const Field& f, std::array<Cell, 2>& cells)
    {
        static constexpr std::size_t field_size = Field::size;
        using flux_computation_t                = NormalFluxDefinition<FluxType::NonLinear, Field, 2>;
        using flux_value_t                      = typename flux_computation_t::flux_value_t;

        flux_value_t flux;
        if constexpr (field_size == 1)
        {
            flux = 0.5 * (f[cells[0]] + f[cells[0]]);
        }
        else
        {
            assert(false && "not implemented");
        }
        return flux;
    }

} // end namespace samurai
