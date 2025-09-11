#pragma once
#include "../flux_based/flux_based_scheme__lin_hom.hpp"

namespace samurai
{
    template <class Field>
    auto make_gradient_order2()
    {
        static_assert(Field::n_comp == 1,
                      "The field type for the gradient operator must be a scalar field or a vector field with 1 component.");

        static constexpr std::size_t dim = Field::dim;

        static constexpr std::size_t stencil_size = 2;
        using input_field_t                       = Field;
        using output_field_t = VectorField<typename Field::mesh_t, typename Field::value_type, dim, detail::is_soa_v<input_field_t>>;

        using cfg = FluxConfig<SchemeType::LinearHomogeneous, stencil_size, output_field_t, input_field_t>;

        FluxDefinition<cfg> average_coeffs;

        static_for<0, dim>::apply( // for (int d=0; d<dim; d++)
            [&](auto _d)
            {
                static constexpr std::size_t d = _d();

                average_coeffs[d].cons_flux_function = [](FluxStencilCoeffs<cfg>& coeffs, double)
                {
                    static constexpr std::size_t left  = 0;
                    static constexpr std::size_t right = 1;

                    // Return value: 2 matrices (left, right) of size output_n_comp x n_comp.
                    // In this case, of size dim x 1, i.e. a column vector of size dim.
                    if constexpr (output_field_t::is_scalar)
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
