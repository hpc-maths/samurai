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

        static constexpr std::size_t dim    = Field::dim;
        static constexpr std::size_t n_comp = Field::n_comp;

        static constexpr std::size_t stencil_size = 2;
        using input_field_t                       = Field;
        using output_field_t                      = Field;

        static constexpr std::size_t left  = 0;
        static constexpr std::size_t right = 1;

        static_assert(dim == n_comp || n_comp == 1,
                      "make_convection_upwind() is not implemented for this field size in this space dimension.");

        using cfg = FluxConfig<SchemeType::NonLinear, stencil_size, output_field_t, input_field_t>;

        FluxDefinition<cfg> upwind;

        static_for<0, dim>::apply( // for each positive Cartesian direction 'd'
            [&](auto _d)
            {
                static constexpr std::size_t d = _d();

                auto f = [](auto u) -> FluxValue<cfg>
                {
                    if constexpr (Field::is_scalar)
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
                    field_value_t v;
                    if constexpr (Field::is_scalar)
                    {
                        v = 0.5 * (field[left] + field[right]);
                    }
                    else
                    {
                        v = 0.5 * (field[left](d) + field[right](d));
                    }

                    flux = v >= 0 ? f(field[left]) : f(field[right]);
                };

                upwind[d].cons_jacobian_function = [](samurai::StencilJacobian<cfg>& jac,
                                                      const samurai::StencilData<cfg>& /*data*/,
                                                      const samurai::StencilValues<cfg>& field)
                {
                    field_value_t v;
                    if constexpr (Field::is_scalar)
                    {
                        v = 0.5 * (field[left] + field[right]);
                    }
                    else
                    {
                        v = 0.5 * (field[left](d) + field[right](d));
                    }

                    if constexpr (Field::is_scalar)
                    {
                        if (v >= 0)
                        {
                            jac[right] = 0.;
                            jac[left]  = 2. * field[left];
                        }
                        else
                        {
                            jac[left]  = 0.;
                            jac[right] = 2. * field[right];
                        }
                    }
                    else
                    {
                        if (v >= 0)
                        {
                            jac[left].fill(0.0);
                            jac[right].fill(0.0);

                            if (d == 0)
                            {
                                jac[left](d, d) = 2.0 * field[left](d);
                                jac[left](0, 1) = 0.;
                                jac[left](1, 0) = field[left](1);
                                jac[left](1, 1) = field[left](0);
                            }
                            else if (d == 1)
                            {
                                jac[left](0, 0) = field[left](1);
                                jac[left](0, 1) = field[left](0);
                                jac[left](1, 0) = 0.;
                                jac[left](d, d) = 2.0 * field[left](d);
                            }
                            else
                            {
                                std::cerr << "Not implemented for dim > 2" << std::endl;
                                std::exit(EXIT_FAILURE);
                            }
                        }
                        else
                        {
                            jac[right].fill(0.0);
                            jac[left].fill(0.0);

                            if (d == 0)
                            {
                                jac[right](d, d) = 2.0 * field[right](d);
                                jac[right](0, 1) = 0.;
                                jac[right](1, 0) = field[right](1);
                                jac[right](1, 1) = field[right](0);
                            }
                            else if (d == 1)
                            {
                                jac[right](0, 0) = field[right](1);
                                jac[right](0, 1) = field[right](0);
                                jac[right](1, 0) = 0.;
                                jac[right](d, d) = 2.0 * field[right](d);
                            }
                            else
                            {
                                std::cerr << "Not implemented for dim > 2" << std::endl;
                                std::exit(EXIT_FAILURE);
                            }
                        }
                    }
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

        static constexpr std::size_t dim = Field::dim;

        static constexpr std::size_t stencil_size = 6;
        using input_field_t                       = Field;
        using output_field_t                      = Field;

        static_assert(dim == Field::n_comp || Field::n_comp == 1,
                      "make_convection_weno5() is not implemented for this field size in this space dimension.");

        using cfg = FluxConfig<SchemeType::NonLinear, stencil_size, output_field_t, input_field_t>;

        FluxDefinition<cfg> weno5;

        static_for<0, dim>::apply( // for each positive Cartesian direction 'd'
            [&](auto _d)
            {
                static constexpr std::size_t d = _d();

                auto f = [](auto u) -> FluxValue<cfg>
                {
                    if constexpr (Field::is_scalar)
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
                    static constexpr std::size_t left  = 2;
                    static constexpr std::size_t right = 3;

                    field_value_t v;
                    if constexpr (Field::is_scalar)
                    {
                        v = 0.5 * (u[left] + u[right]);
                    }
                    else
                    {
                        v = 0.5 * (u[left](d) + u[right](d));
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
