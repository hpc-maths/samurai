#pragma once
#include "divergence_FV.hpp"

namespace samurai
{
    template <std::size_t dim>
    using VelocityVector = xt::xtensor_fixed<double, xt::xshape<dim>>;

    template <class Field>
    auto make_convection(const VelocityVector<Field::dim> velocity)
    {
        // static_assert(Field::size == 1, "The field type for the gradient operator must be a scalar field.");

        static constexpr std::size_t dim               = Field::dim;
        static constexpr std::size_t field_size        = Field::size;
        static constexpr std::size_t output_field_size = field_size;

        auto upwind = make_flux_definition<FluxType::LinearHomogeneous, Field, output_field_size>();

        static_for<0, dim>::apply( // for (int d=0; d<dim; d++)
            [&](auto integral_constant_d)
            {
                static constexpr int d = decltype(integral_constant_d)::value;

                // Return type: 2 matrices (left, right) of size output_field_size x field_size.
                // In this case, of size field_size x field_size.
                using flux_stencil_coeffs_t        = typename decltype(upwind)::flux_computation_t::flux_stencil_coeffs_t;
                static constexpr std::size_t left  = 0;
                static constexpr std::size_t right = 1;

                if (velocity(d) >= 0) // use the left values
                {
                    upwind[d].flux_function = [&](double)
                    {
                        flux_stencil_coeffs_t coeffs;
                        if constexpr (output_field_size == 1)
                        {
                            coeffs[left]  = velocity(d);
                            coeffs[right] = 0;
                        }
                        else
                        {
                            coeffs[left].fill(0);
                            coeffs[right].fill(0);
                            xt::col(coeffs[left], d)  = velocity(d);
                            xt::col(coeffs[right], d) = 0;
                        }
                        return coeffs;
                    };
                }
                else // use the right values
                {
                    upwind[d].flux_function = [&](double)
                    {
                        flux_stencil_coeffs_t coeffs;
                        if constexpr (output_field_size == 1)
                        {
                            coeffs[left]  = 0;
                            coeffs[right] = velocity(d);
                        }
                        else
                        {
                            coeffs[left].fill(0);
                            coeffs[right].fill(0);
                            xt::col(coeffs[left], d)  = 0;
                            xt::col(coeffs[right], d) = velocity(d);
                        }
                        return coeffs;
                    };
                }
            });

        return make_flux_based_scheme(upwind);
    }

} // end namespace samurai
