#pragma once
#include "divergence_FV__nonlin.hpp"

namespace samurai
{
    /**
     * Convection term where the velocity field is compressible.
     *
     * Let U be the vector (in 2D) [u, v]^T.
     * The convective term is given by
     *          \int_V \div (U \otimes U),
     * which rewrites, by Green's theorem, as
     *          \int_S (U \otimes U).n,
     * Developed in 2D, it means
     *   | u^2  uv || n_x |
     *   | uv  v^2 || n_y |,
     * i.e.
     *   | u^2 | if x-direction and | uv  | if y-direction.
     *   | uv  |                    | v^2 |
     */
    template <class Field>
    auto make_convection(Field& u)
    {
        static constexpr std::size_t dim          = Field::dim;
        static constexpr std::size_t field_size   = Field::size;
        static constexpr std::size_t stencil_size = 2;

        if constexpr (dim == 1 && field_size == 1)
        {
            auto f = [](auto v)
            {
                auto f_v = samurai::make_flux_value<Field, 1>();
                f_v      = v * v; // usually, the 1D Burgers equation requires *1/2, but we don't do it for consitency with 2 and 3D.
                return f_v;
            };

            auto upwind_f = samurai::make_flux_definition<Field, 1>(
                [f](auto& v, auto& cells)
                {
                    auto& left  = cells[0];
                    auto& right = cells[1];
                    return v[left] >= 0 ? f(v[left]) : f(v[right]);
                });

            return samurai::make_divergence_FV(upwind_f, u);
        }
        else if constexpr (field_size == dim)
        {
            static constexpr std::size_t output_field_size = field_size;

            if constexpr (dim == 2)
            {
                auto f_x = [](auto v)
                {
                    auto f_v = samurai::make_flux_value<Field, output_field_size>();
                    f_v[0]   = v[0] * v[0];
                    f_v[1]   = v[0] * v[1];
                    return f_v;
                };

                auto f_y = [](auto v)
                {
                    auto f_v = samurai::make_flux_value<Field, output_field_size>();
                    f_v[0]   = v[0] * v[1];
                    f_v[1]   = v[1] * v[1];
                    return f_v;
                };

                auto upwind_f = samurai::make_flux_definition<Field, output_field_size, stencil_size>();
                // x-direction
                upwind_f[0].flux_function = [f_x](auto& v, auto& cells)
                {
                    static constexpr std::size_t x = 0;
                    auto& left                     = cells[0];
                    auto& right                    = cells[1];
                    return v[left](x) >= 0 ? f_x(v[left]) : f_x(v[right]);
                };
                // y-direction
                upwind_f[1].flux_function = [f_y](auto& v, auto& cells)
                {
                    static constexpr std::size_t y = 1;
                    auto& bottom                   = cells[0];
                    auto& top                      = cells[1];
                    return v[bottom](y) >= 0 ? f_y(v[bottom]) : f_y(v[top]);
                };

                return make_divergence_FV(upwind_f, u);
            }
            else
            {
                static_assert(dim < 3, "make_compressible_convection() is not implemented for dim > 2.");
            }
        }
        else if constexpr (field_size == 1 && dim != field_size)
        {
            static_assert(!(field_size == 1 && dim != field_size),
                          "make_compressible_convection() is not implemented for a scalar field in higher dimensions than 1 (TODO).");
        }
        else
        {
            static_assert(dim == field_size || field_size == 1,
                          "make_compressible_convection() is not implemented for this field size and in this space dimension.");
        }
    }

} // end namespace samurai
