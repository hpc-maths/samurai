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

        struct TempVariables
        {
            // StencilValuesBatch<cfg> f;

            // Batch<FluxValue<cfg>> q0;
            // Batch<FluxValue<cfg>> q1;
            // Batch<FluxValue<cfg>> q2;
            // Batch<FluxValue<cfg>> IS0;
            // Batch<FluxValue<cfg>> IS1;
            // Batch<FluxValue<cfg>> IS2;

            // void resize(std::size_t size)
            // {
            // f.resize(size);
            // q0.resize(size);
            // q1.resize(size);
            // q2.resize(size);
            // IS0.resize(size);
            // IS1.resize(size);
            // IS2.resize(size);
            // f.position() = size;
            // q0.position()  = size;
            // q1.position()  = size;
            // q2.position()  = size;
            // IS0.position() = size;
            // IS1.position() = size;
            // IS2.position() = size;
            // }
        };

        FluxDefinition<cfg> weno5;

        static_for<0, dim>::apply( // for each positive Cartesian direction 'd'
            [&](auto integral_constant_d)
            {
                static constexpr std::size_t d = integral_constant_d();

                // Stencil creation:
                //        weno5[0].stencil = {{-2, 0}, {-1, 0}, {0,0}, {1,0}, {2,0}, {3,0}};
                //        weno5[1].stencil = {{ 0,-2}, { 0,-1}, {0,0}, {0,1}, {0,2}, {0,3}};
                weno5[d].stencil = line_stencil<dim, d>(-2, -1, 0, 1, 2, 3);

                weno5[d].create_temp_variables = []()
                {
                    return new TempVariables();
                };

                if (velocity(d) >= 0)
                {
                    weno5[d].cons_flux_function = [&velocity](auto& cells, const Field& u) -> FluxValue<cfg>
                    {
                        Array<FluxValue<cfg>, 5, is_soa> f({u[cells[0]], u[cells[1]], u[cells[2]], u[cells[3]], u[cells[4]]});
                        f *= velocity(d);
                        FluxValue<cfg> flux;
                        compute_weno5_flux(flux, f);
                        return flux;
                    };

                    weno5[d].set_cons_flux_function__batch(
                        [&velocity](const BatchData<cfg>& batch_data, Batch<FluxValue<cfg>>& flux_values, const auto& stencil_values)
                        {
                            // TempVariables* tmp = static_cast<TempVariables*>(ctx);
                            // tmp->resize(stencil_values.size());
                            // auto& f = tmp->f;
                            // assert(stencil_values.size() > 0);
                            // for (std::size_t i = 0; i < stencil_values.size(); ++i)
                            // {
                            //     f[0][i] = velocity(d) * stencil_values[0][i];
                            //     f[1][i] = velocity(d) * stencil_values[1][i];
                            //     f[2][i] = velocity(d) * stencil_values[2][i];
                            //     f[3][i] = velocity(d) * stencil_values[3][i];
                            //     f[4][i] = velocity(d) * stencil_values[4][i];
                            // }
                            // compute_weno5_flux__batch(flux_values, f, *tmp);
                            for (std::size_t i = 0; i < batch_data.batch_size; ++i)
                            {
                                // Array<FluxValue<cfg>, 5> f({velocity(d) * stencil_values[0][i],
                                //                             velocity(d) * stencil_values[1][i],
                                //                             velocity(d) * stencil_values[2][i],
                                //                             velocity(d) * stencil_values[3][i],
                                //                             velocity(d) * stencil_values[4][i]});
                                // compute_weno5_flux(flux_values[i], f);

                                // compute_weno5_flux_vecto(flux_values[i],
                                //                          velocity(d) * stencil_values[0][i],
                                //                          velocity(d) * stencil_values[1][i],
                                //                          velocity(d) * stencil_values[2][i],
                                //                          velocity(d) * stencil_values[3][i],
                                //                          velocity(d) * stencil_values[4][i]);

                                double f_jm2 = velocity(d) * stencil_values[0][i];
                                double f_jm1 = velocity(d) * stencil_values[1][i];
                                double f_j   = velocity(d) * stencil_values[2][i];
                                double f_jp1 = velocity(d) * stencil_values[3][i];
                                double f_jp2 = velocity(d) * stencil_values[4][i];

                                const double eps = 1e-6;

                                // clang-format off

                            // (2.8) and Table I (r=3)
                            auto q0 =  1./3. * f_jm2 - 7./6. * f_jm1 + 11./6. * f_j;
                            auto q1 = -1./6. * f_jm1 + 5./6. * f_j   +  1./3. * f_jp1;
                            auto q2 =  1./3. * f_j   + 5./6. * f_jp1 -  1./6. * f_jp2;

                            // (3.2)-(3.4)
                            auto IS0 = 13./12. * pow(f_jm2 - 2.*f_jm1 + f_j  , 2) + 1./4. * pow(   f_jm2 -4.*f_jm1 + 3.*f_j  , 2);
                            auto IS1 = 13./12. * pow(f_jm1 - 2.*f_j   + f_jp1, 2) + 1./4. * pow(   f_jm1           -    f_jp1, 2);
                            auto IS2 = 13./12. * pow(f_j   - 2.*f_jp1 + f_jp2, 2) + 1./4. * pow(3.*f_j   -4.*f_jp1 +    f_jp2, 2);

                                // clang-format on

                                // (2.16) and Table II (r=3)
                                auto alpha0 = 0.1 / pow((eps + IS0), 2);
                                auto alpha1 = 0.6 / pow((eps + IS1), 2);
                                auto alpha2 = 0.3 / pow((eps + IS2), 2);

                                // (2.15)
                                auto sum_alphas = alpha0 + alpha1 + alpha2;
                                auto omega0     = alpha0 / sum_alphas;
                                auto omega1     = alpha1 / sum_alphas;
                                auto omega2     = alpha2 / sum_alphas;

                                // (2.10)
                                // value_type flux = omega0 * q0 + omega1 * q1 + omega2 * q2;
                                // return flux;
                                flux_values[i] = omega0 * q0 + omega1 * q1 + omega2 * q2;
                            }
                        });
                }
                else
                {
                    weno5[d].cons_flux_function = [&velocity](auto& cells, const Field& u) -> FluxValue<cfg>
                    {
                        Array<FluxValue<cfg>, 5, is_soa> f({u[cells[5]], u[cells[4]], u[cells[3]], u[cells[2]], u[cells[1]]});
                        f *= velocity(d);
                        FluxValue<cfg> flux;
                        compute_weno5_flux(flux, f);
                        return flux;
                    };

                    weno5[d].set_cons_flux_function__batch(
                        [&velocity](const BatchData<cfg>& batch_data, Batch<FluxValue<cfg>>& flux_values, const auto& stencil_values)
                        {
                // TempVariables* tmp = static_cast<TempVariables*>(ctx);
                // tmp->resize(stencil_values.size());
                // auto& f = tmp->f;
                // assert(stencil_values.size() > 0);
                // for (std::size_t i = 0; i < stencil_values.size(); ++i)
                // {
                //     f[0][i] = velocity(d) * stencil_values[5][i];
                //     f[1][i] = velocity(d) * stencil_values[4][i];
                //     f[2][i] = velocity(d) * stencil_values[3][i];
                //     f[3][i] = velocity(d) * stencil_values[2][i];
                //     f[4][i] = velocity(d) * stencil_values[1][i];
                // }
                // compute_weno5_flux__batch(flux_values, f, *tmp);

#pragma omp simd
                            for (std::size_t i = 0; i < batch_data.batch_size; ++i)
                            {
                                // Array<FluxValue<cfg>, 5> f({velocity(d) * stencil_values[5][i],
                                //                             velocity(d) * stencil_values[4][i],
                                //                             velocity(d) * stencil_values[3][i],
                                //                             velocity(d) * stencil_values[2][i],
                                //                             velocity(d) * stencil_values[1][i]});
                                // compute_weno5_flux(flux_values[i], f);

                                // compute_weno5_flux_vecto(flux_values[i],
                                //                          velocity(d) * stencil_values[5][i],
                                //                          velocity(d) * stencil_values[4][i],
                                //                          velocity(d) * stencil_values[3][i],
                                //                          velocity(d) * stencil_values[2][i],
                                //                          velocity(d) * stencil_values[1][i]);

                                double f_jm2 = velocity(d) * stencil_values[5][i];
                                double f_jm1 = velocity(d) * stencil_values[4][i];
                                double f_j   = velocity(d) * stencil_values[3][i];
                                double f_jp1 = velocity(d) * stencil_values[2][i];
                                double f_jp2 = velocity(d) * stencil_values[1][i];

                                const double eps = 1e-6;

                                // clang-format off

                            // (2.8) and Table I (r=3)
                            auto q0 =  1./3. * f_jm2 - 7./6. * f_jm1 + 11./6. * f_j;
                            auto q1 = -1./6. * f_jm1 + 5./6. * f_j   +  1./3. * f_jp1;
                            auto q2 =  1./3. * f_j   + 5./6. * f_jp1 -  1./6. * f_jp2;

                            // (3.2)-(3.4)
                            auto IS0 = 13./12. * pow(f_jm2 - 2.*f_jm1 + f_j  , 2) + 1./4. * pow(   f_jm2 -4.*f_jm1 + 3.*f_j  , 2);
                            auto IS1 = 13./12. * pow(f_jm1 - 2.*f_j   + f_jp1, 2) + 1./4. * pow(   f_jm1           -    f_jp1, 2);
                            auto IS2 = 13./12. * pow(f_j   - 2.*f_jp1 + f_jp2, 2) + 1./4. * pow(3.*f_j   -4.*f_jp1 +    f_jp2, 2);

                                // clang-format on

                                // (2.16) and Table II (r=3)
                                auto alpha0 = 0.1 / pow((eps + IS0), 2);
                                auto alpha1 = 0.6 / pow((eps + IS1), 2);
                                auto alpha2 = 0.3 / pow((eps + IS2), 2);

                                // (2.15)
                                auto sum_alphas = alpha0 + alpha1 + alpha2;
                                auto omega0     = alpha0 / sum_alphas;
                                auto omega1     = alpha1 / sum_alphas;
                                auto omega2     = alpha2 / sum_alphas;

                                // (2.10)
                                // value_type flux = omega0 * q0 + omega1 * q1 + omega2 * q2;
                                // return flux;
                                flux_values[i] = omega0 * q0 + omega1 * q1 + omega2 * q2;
                            }
                        });
                }
            });

        auto scheme = make_flux_based_scheme(weno5);
        scheme.set_name("convection");
        // scheme.enable_batches(false);
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
                        FluxValue<cfg> flux;
                        compute_weno5_flux(flux, f);
                        return flux;
                    }
                    else
                    {
                        Array<FluxValue<cfg>, 5, is_soa> f({u[cells[5]], u[cells[4]], u[cells[3]], u[cells[2]], u[cells[1]]});
                        f *= v;
                        FluxValue<cfg> flux;
                        compute_weno5_flux(flux, f);
                        return flux;
                    }
                };
            });

        auto scheme = make_flux_based_scheme(weno5);
        scheme.set_name("convection");
        return scheme;
    }

} // end namespace samurai
