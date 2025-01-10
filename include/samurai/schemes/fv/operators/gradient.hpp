#pragma once
#include "../flux_based/flux_based_scheme__lin_hom.hpp"

namespace samurai
{
    template <class Field>
    auto make_gradient_order2()
    {
        static_assert(Field::size == 1, "The field type for the gradient operator must be a scalar field.");

        static constexpr std::size_t dim               = Field::dim;
        static constexpr std::size_t output_field_size = dim;
        static constexpr std::size_t stencil_size      = 2;

        using cfg = FluxConfig<SchemeType::LinearHomogeneous, output_field_size, stencil_size, Field>;

        FluxDefinition<cfg> average_coeffs;

        static_for<0, dim>::apply( // for (int d=0; d<dim; d++)
            [&](auto integral_constant_d)
            {
                static constexpr std::size_t d = decltype(integral_constant_d)::value;

                average_coeffs[d].cons_flux_function = [](double)
                {
                    static constexpr std::size_t left  = 0;
                    static constexpr std::size_t right = 1;

                    // Return value: 2 matrices (left, right) of size output_field_size x field_size.
                    // In this case, of size dim x 1, i.e. a column vector of size dim.
                    FluxStencilCoeffs<cfg> coeffs;
                    if constexpr (output_field_size == 1)
                    {
                        coeffs[left]  = 0.5;
                        coeffs[right] = 0.5;
                    }
                    else
                    {
                        coeffs[left].fill(0);
                        coeffs[right].fill(0);
                        coeffs[left](d, 0)  = 0.5;
                        coeffs[right](d, 0) = 0.5;
                        // xt::row(coeffs[left], d)  = 0.5;
                        // xt::row(coeffs[right], d) = 0.5;
                    }
                    return coeffs;
                };
            });

        auto grad = make_flux_based_scheme(average_coeffs);
        grad.set_name("Gradient");
        return grad;
    }

    template <class Field>
    [[deprecated("Use make_gradient_order2() instead.")]] auto make_gradient()
    {
        return make_gradient_order2<Field>();
    }

} // end namespace samurai
