#pragma once
#include "flux_divergence.hpp"
#include "weno_impl.hpp"

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
    auto make_convection_upwind()
    {
        using field_value_t = typename Field::value_type;

        static constexpr std::size_t dim               = Field::dim;
        static constexpr std::size_t field_size        = Field::size;
        static constexpr std::size_t output_field_size = field_size;
        static constexpr std::size_t stencil_size      = 2;

        static_assert(dim == field_size || field_size == 1,
                      "make_convection_upwind() is not implemented for this field size in this space dimension.");

        using cfg = FluxConfig<SchemeType::NonLinear, output_field_size, stencil_size, Field>;

        /**
         * The following commented code is kept as example.
         * The generalized N-dimensional code (below the comments) is used in practice.
         */
        /*
        if constexpr (dim == 2)
        {
            auto f_x = [](auto v)
            {
                FluxValue<cfg> f_v;
                f_v(0) = v(0) * v(0);
                f_v(1) = v(0) * v(1);
                return f_v;
            };

            auto f_y = [](auto v)
            {
                FluxValue<cfg> f_v;
                f_v(0) = v(1) * v(0);
                f_v(1) = v(1) * v(1);
                return f_v;
            };

            FluxDefinition<cfg> upwind;
            // x-direction
            upwind[0].cons_flux_function = [f_x](auto& cells, Field& v)
            {
                static constexpr std::size_t x = 0;
                auto& left                     = cells[0];
                auto& right                    = cells[1];
                return v[left](x) >= 0 ? f_x(v[left]) : f_x(v[right]);
            };
            // y-direction
            upwind[1].cons_flux_function = [f_y](auto& cells, Field& v)
            {
                static constexpr std::size_t y = 1;
                auto& bottom                   = cells[0];
                auto& top                      = cells[1];
                return v[bottom](y) >= 0 ? f_y(v[bottom]) : f_y(v[top]);
            };

            return make_flux_based_scheme(upwind);
        }
        if constexpr (dim == 3)
        {
            auto f_x = [](auto v)
            {
                FluxValue<cfg> f_v;
                f_v(0) = v(0) * v(0);
                f_v(1) = v(0) * v(1);
                f_v(2) = v(0) * v(2);
                return f_v;
            };

            auto f_y = [](auto v)
            {
                FluxValue<cfg> f_v;
                f_v(0) = v(1) * v(0);
                f_v(1) = v(1) * v(1);
                f_v(2) = v(1) * v(2);
                return f_v;
            };

            auto f_z = [](auto v)
            {
                FluxValue<cfg> f_v;
                f_v(0) = v(2) * v(0);
                f_v(1) = v(2) * v(1);
                f_v(2) = v(2) * v(2);
                return f_v;
            };

            FluxDefinition<cfg> upwind;
            // x-direction
            upwind[0].cons_flux_function = [f_x](auto& cells, Field& v)
            {
                static constexpr std::size_t x = 0;
                auto& left                     = cells[0];
                auto& right                    = cells[1];
                return v[left](x) >= 0 ? f_x(v[left]) : f_x(v[right]);
            };
            // y-direction
            upwind[1].cons_flux_function = [f_y](auto& cells, Field& v)
            {
                static constexpr std::size_t y = 1;
                auto& front                    = cells[0];
                auto& back                     = cells[1];
                return v[front](y) >= 0 ? f_y(v[front]) : f_y(v[back]);
            };
            // z-direction
            upwind[2].cons_flux_function = [f_z](auto& cells, Field& v)
            {
                static constexpr std::size_t z = 2;
                auto& bottom                   = cells[0];
                auto& top                      = cells[1];
                return v[bottom](z) >= 0 ? f_z(v[bottom]) : f_z(v[top]);
            };

            return make_flux_based_scheme(upwind);
        }
        else
        {*/
        FluxDefinition<cfg> upwind;

        static_for<0, dim>::apply( // for each positive Cartesian direction 'd'
            [&](auto integral_constant_d)
            {
                static constexpr std::size_t d = decltype(integral_constant_d)::value;

                auto f = [](auto u) -> FluxValue<cfg>
                {
                    if constexpr (field_size == 1)
                    {
                        return u * u;
                    }
                    else
                    {
                        return u(d) * u;
                    }
                };

                upwind[d].cons_flux_function = [f](auto& cells, const Field& field)
                {
                    auto& left  = cells[0];
                    auto& right = cells[1];

                    field_value_t v;
                    if constexpr (field_size == 1)
                    {
                        v = field[left];
                    }
                    else
                    {
                        v = field[left](d);
                    }

                    return v >= 0 ? f(field[left]) : f(field[right]);
                };
            });

        return make_flux_based_scheme(upwind);
    }

    template <class Field>
    auto make_convection_weno5()
    {
        using field_value_t = typename Field::value_type;

        static_assert(Field::mesh_t::config::ghost_width >= 3, "WENO5 requires at least 3 ghosts.");

        static constexpr std::size_t dim               = Field::dim;
        static constexpr std::size_t field_size        = Field::size;
        static constexpr std::size_t output_field_size = field_size;
        static constexpr std::size_t stencil_size      = 6;

        static_assert(dim == field_size || field_size == 1,
                      "make_convection_weno5() is not implemented for this field size in this space dimension.");

        using cfg = FluxConfig<SchemeType::NonLinear, output_field_size, stencil_size, Field>;

        FluxDefinition<cfg> weno5;

        static_for<0, dim>::apply( // for each positive Cartesian direction 'd'
            [&](auto integral_constant_d)
            {
                static constexpr std::size_t d = decltype(integral_constant_d)::value;

                auto f = [](auto u) -> FluxValue<cfg>
                {
                    if constexpr (field_size == 1)
                    {
                        return u * u;
                    }
                    else
                    {
                        return u(d) * u;
                    }
                };

                weno5[d].stencil = line_stencil<dim, d>(-2, -1, 0, 1, 2, 3);

                weno5[d].cons_flux_function = [f](auto& cells, const Field& u) -> FluxValue<cfg>
                {
                    static constexpr std::size_t stencil_center = 2;

                    field_value_t v;
                    if constexpr (field_size == 1)
                    {
                        v = u[cells[stencil_center]];
                    }
                    else
                    {
                        v = u[cells[stencil_center]](d);
                    }

                    xt::xtensor_fixed<FluxValue<cfg>, xt::xshape<5>> f_u;
                    if (v >= 0)
                    {
                        f_u = {f(u[cells[0]]), f(u[cells[1]]), f(u[cells[2]]), f(u[cells[3]]), f(u[cells[4]])};
                    }
                    else
                    {
                        f_u = {f(u[cells[5]]), f(u[cells[4]]), f(u[cells[3]]), f(u[cells[2]]), f(u[cells[1]])};
                    }

                    return compute_weno5_flux(f_u);
                };
            });

        return make_flux_based_scheme(weno5);
    }

} // end namespace samurai
