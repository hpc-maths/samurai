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

                upwind[d].cons_flux_function = [f](FluxValue<cfg>& flux, const StencilData<cfg>& /*data*/, const StencilValues<cfg>& field)
                {
                    static constexpr std::size_t left  = 0;
                    static constexpr std::size_t right = 1;

                    field_value_t v;
                    if constexpr (field_size == 1)
                    {
                        v = field[left];
                    }
                    else
                    {
                        v = field[left](d);
                    }

                    flux = v >= 0 ? f(field[left]) : f(field[right]);
                };
            });

        auto scheme = make_flux_based_scheme(upwind);
        scheme.set_name("convection");
        return scheme;
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

                weno5[d].cons_flux_function = [f](FluxValue<cfg>& flux, const StencilData<cfg>& /*data*/, const StencilValues<cfg>& u)
                {
                    static constexpr std::size_t stencil_center = 2;

                    field_value_t v;
                    if constexpr (field_size == 1)
                    {
                        v = u[stencil_center];
                    }
                    else
                    {
                        v = u[stencil_center](d);
                    }

                    if (v >= 0)
                    {
                        std::array<FluxValue<cfg>, 5> f_u = {f(u[0]), f(u[1]), f(u[2]), f(u[3]), f(u[4])};
                        compute_weno5_flux(flux, f_u);
                    }
                    else
                    {
                        std::array<FluxValue<cfg>, 5> f_u = {f(u[5]), f(u[4]), f(u[3]), f(u[2]), f(u[1])};
                        compute_weno5_flux(flux, f_u);
                    }
                };
            });

        auto scheme = make_flux_based_scheme(weno5);
        scheme.set_name("convection");
        return scheme;
    }

} // end namespace samurai
