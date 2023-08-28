#pragma once
#include "FV_scheme.hpp"
#include "flux_definition.hpp"

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

    template <class Field>
    auto get_normal_grad_order1_coeffs(double h)
    {
        static constexpr bool is_linear         = true;
        static constexpr bool is_heterogeneous  = false;
        static constexpr std::size_t field_size = Field::size;
        using flux_computation_t                = NormalFluxDefinition<Field, 2, is_linear, is_heterogeneous>;
        using flux_stencil_coeffs_t             = typename flux_computation_t::flux_stencil_coeffs_t;

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

    // template <class Field, class Vector>
    // auto normal_grad_order1(Vector& direction)
    // {
    //     static constexpr std::size_t dim = Field::dim;
    //     using flux_computation_t         = NormalFluxDefinition_LinHom<Field, 2>;

    //     flux_computation_t normal_grad;
    //     normal_grad.direction       = direction;
    //     normal_grad.stencil         = in_out_stencil<dim>(direction);
    //     normal_grad.get_flux_coeffs = get_normal_grad_order1_coeffs<Field>;
    //     return normal_grad;
    // }

    template <class Field>
    auto get_average_coeffs(double)
    {
        static constexpr bool is_linear         = true;
        static constexpr bool is_heterogeneous  = false;
        static constexpr std::size_t field_size = Field::size;
        using flux_computation_t                = NormalFluxDefinition<Field, 2, is_linear, is_heterogeneous>;
        using flux_stencil_coeffs_t             = typename flux_computation_t::flux_stencil_coeffs_t;

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

    // template <class Field, class Vector>
    // auto average_quantity(Vector& direction)
    // {
    //     static constexpr std::size_t dim = Field::dim;
    //     using flux_computation_t         = NormalFluxDefinition_LinHom<Field, 2>;

    //     flux_computation_t flux;
    //     flux.direction       = direction;
    //     flux.stencil         = in_out_stencil<dim>(direction);
    //     flux.get_flux_coeffs = get_average_coeffs<Field>;
    //     return flux;
    // }

} // end namespace samurai
