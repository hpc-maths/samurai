#pragma once
#include "../flux_based_scheme__lin_hom.hpp"

namespace samurai
{
    template <class Field>
    auto make_gradient()
    {
        static_assert(Field::size == 1, "The field type for the gradient operator must be a scalar field.");

        static constexpr std::size_t dim               = Field::dim;
        static constexpr std::size_t output_field_size = dim;

        auto average_coeffs = make_flux_definition<FluxType::LinearHomogeneous, Field, output_field_size>();

        static_for<0, dim>::apply( // for (int d=0; d<dim; d++)
            [&](auto integral_constant_d)
            {
                static constexpr int d = decltype(integral_constant_d)::value;

                average_coeffs[d].flux_function = [](double)
                {
                    // Return type: 2 matrices (left, right) of size output_field_size x field_size.
                    // In this case, of size dim x 1, i.e. a column vector of size dim.
                    using flux_stencil_coeffs_t        = typename decltype(average_coeffs)::flux_computation_t::flux_stencil_coeffs_t;
                    static constexpr std::size_t left  = 0;
                    static constexpr std::size_t right = 1;

                    flux_stencil_coeffs_t coeffs;
                    if constexpr (output_field_size == 1)
                    {
                        coeffs[left]  = 0.5;
                        coeffs[right] = 0.5;
                    }
                    else
                    {
                        coeffs[left].fill(0);
                        coeffs[right].fill(0);
                        xt::row(coeffs[left], d)  = 0.5;
                        xt::row(coeffs[right], d) = 0.5;
                    }
                    return coeffs;
                };
            });

        auto grad = make_flux_based_scheme(average_coeffs);
        grad.set_name("Gradient");
        return grad;
    }

    template <class Field>
    [[deprecated("Use make_gradient() instead.")]] auto make_gradient_FV()
    {
        return make_gradient<Field>();
    }

} // end namespace samurai
