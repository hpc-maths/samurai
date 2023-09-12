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
    auto make_convection()
    {
        static constexpr std::size_t dim               = Field::dim;
        static constexpr std::size_t field_size        = Field::size;
        static constexpr std::size_t output_field_size = field_size;
        static constexpr std::size_t stencil_size      = 2;

        if constexpr (field_size == 1)
        {
            auto f = [](auto v)
            {
                auto f_v = samurai::make_flux_value<Field, output_field_size>();
                f_v      = v * v;
                return f_v;
            };

            auto upwind_f = samurai::make_flux_definition<Field, output_field_size>(
                [f](auto& v, auto& cells)
                {
                    auto& left  = cells[0];
                    auto& right = cells[1];
                    return v[left] >= 0 ? f(v[left]) : f(v[right]);
                });

            return samurai::make_divergence(upwind_f);
        }
        else if constexpr (field_size == dim)
        {
            /**
             * The following commented code is kept as example.
             * The generalized N-dimensional code (below the comments) is used in practice.
             */
            /*
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
                    f_v[0]   = v[1] * v[0];
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

                return make_divergence(upwind_f);
            }
            if constexpr (dim == 3)
            {
                auto f_x = [](auto v)
                {
                    auto f_v = samurai::make_flux_value<Field, output_field_size>();
                    f_v[0]   = v[0] * v[0];
                    f_v[1]   = v[0] * v[1];
                    f_v[2]   = v[0] * v[2];
                    return f_v;
                };

                auto f_y = [](auto v)
                {
                    auto f_v = samurai::make_flux_value<Field, output_field_size>();
                    f_v[0]   = v[1] * v[0];
                    f_v[1]   = v[1] * v[1];
                    f_v[2]   = v[1] * v[2];
                    return f_v;
                };

                auto f_z = [](auto v)
                {
                    auto f_v = samurai::make_flux_value<Field, output_field_size>();
                    f_v[0]   = v[2] * v[0];
                    f_v[1]   = v[2] * v[1];
                    f_v[2]   = v[2] * v[2];
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
                    auto& front                    = cells[0];
                    auto& back                     = cells[1];
                    return v[front](y) >= 0 ? f_y(v[front]) : f_y(v[back]);
                };
                // z-direction
                upwind_f[2].flux_function = [f_z](auto& v, auto& cells)
                {
                    static constexpr std::size_t z = 2;
                    auto& bottom                   = cells[0];
                    auto& top                      = cells[1];
                    return v[bottom](z) >= 0 ? f_z(v[bottom]) : f_z(v[top]);
                };

                return make_divergence(upwind_f);
            }
            else
            {
            */
            auto upwind_f = samurai::make_flux_definition<Field, output_field_size, stencil_size>();

            static_for<0, field_size>::apply( // for (int i=0; i<field_size; i++)
                [&](auto integral_constant_i)
                {
                    static constexpr int i = decltype(integral_constant_i)::value;

                    auto f = [](auto v)
                    {
                        auto f_v = samurai::make_flux_value<Field, output_field_size>();
                        static_for<0, field_size>::apply( // for (int j=0; j<field_size; j++)
                            [&](auto integral_constant_j)
                            {
                                static constexpr int j = decltype(integral_constant_j)::value;

                                f_v[j] = v[i] * v[j];
                            });
                        return f_v;
                    };

                    upwind_f[i].flux_function = [f](auto& v, auto& cells)
                    {
                        auto& left  = cells[0];
                        auto& right = cells[1];
                        return v[left](i) >= 0 ? f(v[left]) : f(v[right]);
                    };
                });

            return make_divergence(upwind_f);
            //}
        }

        else
        {
            static_assert(dim == field_size || field_size == 1,
                          "make_convection() is not implemented for this field size in this space dimension.");
        }
    }

} // end namespace samurai
