#pragma once
#include "../flux_based/flux_based_scheme__lin_het.hpp"
#include "../flux_based/flux_based_scheme__lin_hom.hpp"
#include "weno_impl.hpp"

namespace samurai
{
    template <std::size_t dim>
    using VelocityVector = xt::xtensor_fixed<double, xt::xshape<dim>>;

    /**
     * Linear convection, discretized by a (linear) upwind scheme.
     * @param velocity: constant velocity vector
     */
    template <class Field>
    auto make_convection_upwind(const VelocityVector<Field::dim>& velocity)
    {
        static constexpr std::size_t dim               = Field::dim;
        static constexpr std::size_t field_size        = Field::size;
        static constexpr std::size_t output_field_size = field_size;
        static constexpr std::size_t stencil_size      = 2;

        using cfg = FluxConfig<SchemeType::LinearHomogeneous, output_field_size, stencil_size, Field>;

        FluxDefinition<cfg> upwind;

        static_for<0, dim>::apply( // for (int d=0; d<dim; d++)
            [&](auto integral_constant_d)
            {
                static constexpr std::size_t d = decltype(integral_constant_d)::value;

                static constexpr std::size_t left  = 0;
                static constexpr std::size_t right = 1;

                if (velocity(d) >= 0) // use the left values
                {
                    upwind[d].cons_flux_function = [&](double)
                    {
                        // Return type: 2 matrices (left, right) of size output_field_size x field_size.
                        // In this case, of size field_size x field_size.
                        FluxStencilCoeffs<cfg> coeffs;
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
                    upwind[d].cons_flux_function = [&](double)
                    {
                        FluxStencilCoeffs<cfg> coeffs;
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

        auto scheme = make_flux_based_scheme(upwind);
        scheme.set_name("convection");
        return scheme;
    }

    /**
     * Linear convection, discretized by the WENO5 (Jiang & Shu) scheme.
     * @param velocity: constant velocity vector
     */
    template <class Field>
    auto make_convection_weno5(const VelocityVector<Field::dim>& velocity)
    {
        static_assert(Field::mesh_t::config::ghost_width >= 3, "WENO5 requires at least 3 ghosts.");

        static constexpr std::size_t dim               = Field::dim;
        static constexpr std::size_t field_size        = Field::size;
        static constexpr bool is_soa                   = Field::is_soa;
        static constexpr std::size_t output_field_size = field_size;
        static constexpr std::size_t stencil_size      = 6;

        using cfg = FluxConfig<SchemeType::NonLinear, output_field_size, stencil_size, Field>;

        FluxDefinition<cfg> weno5;

        static_for<0, dim>::apply( // for each positive Cartesian direction 'd'
            [&](auto integral_constant_d)
            {
                static constexpr std::size_t d = decltype(integral_constant_d)::value;

                // Stencil creation:
                //        weno5[0].stencil = {{-2, 0}, {-1, 0}, {0,0}, {1,0}, {2,0}, {3,0}};
                //        weno5[1].stencil = {{ 0,-2}, { 0,-1}, {0,0}, {0,1}, {0,2}, {0,3}};
                weno5[d].stencil = line_stencil<dim, d>(-2, -1, 0, 1, 2, 3);

                if (velocity(d) >= 0)
                {
                    weno5[d].cons_flux_function = [&velocity](auto& cells, const Field& u) -> FluxValue<cfg>
                    {
                        Array<FluxValue<cfg>, 5, is_soa> f({u[cells[0]], u[cells[1]], u[cells[2]], u[cells[3]], u[cells[4]]});
                        f *= velocity(d);
                        return compute_weno5_flux(f);
                    };
                }
                else
                {
                    weno5[d].cons_flux_function = [&velocity](auto& cells, const Field& u) -> FluxValue<cfg>
                    {
                        Array<FluxValue<cfg>, 5, is_soa> f({u[cells[5]], u[cells[4]], u[cells[3]], u[cells[2]], u[cells[1]]});
                        f *= velocity(d);
                        return compute_weno5_flux(f);
                    };
                }
            });

        auto scheme = make_flux_based_scheme(weno5);
        scheme.set_name("convection");
        return scheme;
    }

    /**
     * Linear convection, discretized by a (linear) upwind scheme.
     * @param velocity_field: the velocity field
     */
    template <class Field, class VelocityField>
    auto make_convection_upwind(const VelocityField& velocity_field)
    {
        static_assert(Field::dim == VelocityField::dim && VelocityField::size == VelocityField::dim);

        static constexpr std::size_t dim               = Field::dim;
        static constexpr std::size_t field_size        = Field::size;
        static constexpr std::size_t output_field_size = field_size;
        static constexpr std::size_t stencil_size      = 2;

        using cfg = FluxConfig<SchemeType::LinearHeterogeneous, output_field_size, stencil_size, Field>;

        FluxDefinition<cfg> upwind;

        static_for<0, dim>::apply( // for each positive Cartesian direction 'd'
            [&](auto integral_constant_d)
            {
                static constexpr std::size_t d = decltype(integral_constant_d)::value;

                static constexpr std::size_t left  = 0;
                static constexpr std::size_t right = 1;

                upwind[d].cons_flux_function = [&](const auto& cells)
                {
                    // Return type: 2 matrices (left, right) of size output_field_size x field_size.
                    // In this case, of size field_size x field_size.
                    FluxStencilCoeffs<cfg> coeffs;

                    auto velocity = velocity_field[cells[left]];
                    if (velocity(d) >= 0) // use the left values
                    {
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
                    }
                    else // use the right values
                    {
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
                    }
                    return coeffs;
                };
            });

        auto scheme = make_flux_based_scheme(upwind);
        scheme.set_name("convection");
        return scheme;
    }

    /**
     * Linear convection, discretized by a WENO5 (Jiang & Shu) scheme.
     * @param velocity_field: the velocity field
     */
    template <class Field, class VelocityField>
    auto make_convection_weno5(const VelocityField& velocity_field)
    {
        static_assert(Field::mesh_t::config::ghost_width >= 3, "WENO5 requires at least 3 ghosts.");

        static constexpr std::size_t dim               = Field::dim;
        static constexpr std::size_t field_size        = Field::size;
        static constexpr bool is_soa                   = Field::is_soa;
        static constexpr std::size_t output_field_size = field_size;
        static constexpr std::size_t stencil_size      = 6;

        using cfg = FluxConfig<SchemeType::NonLinear, output_field_size, stencil_size, Field>;

        FluxDefinition<cfg> weno5;

        static_for<0, dim>::apply( // for each positive Cartesian direction 'd'
            [&](auto integral_constant_d)
            {
                static constexpr std::size_t d = decltype(integral_constant_d)::value;

                // Stencil creation:
                //        weno5[0].stencil = {{-2, 0}, {-1, 0}, {0,0}, {1,0}, {2,0}, {3,0}};
                //        weno5[1].stencil = {{ 0,-2}, { 0,-1}, {0,0}, {0,1}, {0,2}, {0,3}};
                weno5[d].stencil = line_stencil<dim, d>(-2, -1, 0, 1, 2, 3);

                weno5[d].cons_flux_function = [&velocity_field](auto& cells, const Field& u) -> FluxValue<cfg>
                {
                    static constexpr std::size_t stencil_center = 2;

                    auto v = velocity_field[cells[stencil_center]](d);

                    if (v >= 0)
                    {
                        Array<FluxValue<cfg>, 5, is_soa> f({u[cells[0]], u[cells[1]], u[cells[2]], u[cells[3]], u[cells[4]]});
                        f *= v;
                        return compute_weno5_flux(f);
                    }
                    else
                    {
                        Array<FluxValue<cfg>, 5, is_soa> f({u[cells[5]], u[cells[4]], u[cells[3]], u[cells[2]], u[cells[1]]});
                        f *= v;
                        return compute_weno5_flux(f);
                    }
                };
            });

        auto scheme = make_flux_based_scheme(weno5);
        scheme.set_name("convection");
        return scheme;
    }

} // end namespace samurai
