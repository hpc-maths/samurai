#pragma once
#include "../flux_based/flux_based_scheme__lin_hom.hpp"

namespace samurai
{
    template <class Field>
    auto make_divergence_order2()
    {
        static constexpr std::size_t field_size        = Field::size;
        static constexpr std::size_t dim               = Field::dim;
        static constexpr std::size_t output_field_size = 1;
        static constexpr std::size_t stencil_size      = 2;

        static_assert(field_size == dim, "The field type for the divergence operator must have a size equal to the space dimension.");

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
                    // In this case, of size 1 x dim, i.e. a row vector of size dim.
                    FluxStencilCoeffs<cfg> coeffs;
                    if constexpr (field_size == 1)
                    {
                        coeffs[left]  = 0.5;
                        coeffs[right] = 0.5;
                    }
                    else
                    {
                        coeffs[left].fill(0);
                        coeffs[right].fill(0);
                        coeffs[left](0, d)  = 0.5;
                        coeffs[right](0, d) = 0.5;
                        // xt::col(coeffs[left], d)  = 0.5;
                        // xt::col(coeffs[right], d) = 0.5;
                    }
                    return coeffs;
                };
            });

        auto div = make_flux_based_scheme(average_coeffs);
        div.set_name("Divergence");
        return div;
    }

    template <class Field>
    [[deprecated("Use make_divergence_order2() instead.")]] auto make_divergence()
    {
        return make_divergence_order2<Field>();
    }

} // end namespace samurai
