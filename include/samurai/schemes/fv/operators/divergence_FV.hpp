#pragma once
#include "../flux_based_scheme__lin_hom.hpp"

namespace samurai
{
    template <class Field, std::size_t output_field_size, std::size_t stencil_size>
    auto make_divergence(const FluxDefinition<FluxType::LinearHomogeneous, Field, output_field_size, stencil_size>& flux_definition)
    {
        return make_flux_based_scheme(flux_definition);
    }

    template <class Field>
    auto make_divergence()
    {
        static constexpr std::size_t field_size = Field::size;
        static constexpr std::size_t dim        = Field::dim;
        static_assert(field_size == dim, "The field type for the divergence operator must have a size equal to the space dimension.");

        static constexpr std::size_t output_field_size = 1;

        auto average_coeffs = make_flux_definition<FluxType::LinearHomogeneous, Field, output_field_size>();

        static_for<0, dim>::apply( // for (int d=0; d<dim; d++)
            [&](auto integral_constant_d)
            {
                static constexpr int d = decltype(integral_constant_d)::value;

                average_coeffs[d].flux_function = [](double)
                {
                    // 2 matrices (left, right) of size output_field_size x field_size.
                    // In this case, of size 1 x dim, i.e. a row vector of size dim.
                    using flux_stencil_coeffs_t        = typename decltype(average_coeffs)::flux_computation_t::flux_stencil_coeffs_t;
                    static constexpr std::size_t left  = 0;
                    static constexpr std::size_t right = 1;

                    flux_stencil_coeffs_t coeffs;
                    if constexpr (field_size == 1)
                    {
                        coeffs[left]  = 0.5;
                        coeffs[right] = 0.5;
                    }
                    else
                    {
                        coeffs[left].fill(0);
                        coeffs[right].fill(0);
                        xt::col(coeffs[left], d)  = 0.5;
                        xt::col(coeffs[right], d) = 0.5;
                    }
                    return coeffs;
                };
            });

        auto div = make_flux_based_scheme(average_coeffs);
        div.set_name("Divergence");
        return div;
    }

    template <class Field>
    [[deprecated("Use make_divergence() instead.")]] auto make_divergence_FV()
    {
        return make_divergence<Field>();
    }

} // end namespace samurai
