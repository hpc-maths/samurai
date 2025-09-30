#pragma once
#include <samurai/schemes/fv.hpp>

namespace samurai
{

    template <class xtensor_t, class xtensor_nu, std::size_t order, std::size_t stencil_size>
    auto compute_osmp_flux_limiter(xtensor_t& d_alpha, xtensor_t& lambdalpha, xtensor_nu& nu, std::size_t j)
    {
        // using value_type = typename xtensor_t::value_type;

        static constexpr double zero = 1e-14;

        xt::xtensor_fixed<double, xt::xshape<5, stencil_size - 1>> c_order;

        // value_type flux;
        double phi_o;
        double phi_lim;

        // Lax-Wendroff
        phi_o = d_alpha[j];

        // 3rd order
        if (order >= 2)
        {
            for (std::size_t l = 0; l < stencil_size - 1; l++)
            {
                c_order(0, l) = (1. + nu[l]) / 3.;
            }
            phi_o += -c_order(0, j) * d_alpha[j] + c_order(0, j - 1) * d_alpha[j - 1];
        }

        if (order >= 3)
        {
            for (std::size_t l = 0; l < stencil_size - 1; l++)
            {
                c_order(1, l) = c_order(0, l) * (nu[l] - 2) / 4.;
                c_order(2, l) = c_order(1, l) * (nu[l] + 2) / 5.;
            }
            // 4th order
            phi_o += c_order(1, j + 1) * d_alpha[j + 1] - 2. * c_order(1, j) * d_alpha[j] + c_order(1, j - 1) * d_alpha[j - 1];

            // 5th order
            phi_o += -(c_order(2, j + 1) * d_alpha[j + 1] - 3. * c_order(2, j) * d_alpha[j] + 3. * c_order(2, j - 1) * d_alpha[j - 1]
                       - c_order(2, j - 2) * d_alpha[j - 2]);
        }

        if (order >= 4)
        {
            for (std::size_t l = 0; l < stencil_size - 1; l++)
            {
                c_order(3, l) = c_order(2, l) * (nu[l] - 3) / 6.;
                c_order(4, l) = c_order(3, l) * (nu[l] + 3) / 7.;
            }
            // 6th order
            phi_o += c_order(3, j + 2) * d_alpha[j + 2] - 4. * c_order(3, j + 1) * d_alpha[j + 1] + 6. * c_order(3, j) * d_alpha[j]
                   - 4. * c_order(3, j - 1) * d_alpha[j - 1] + c_order(3, j - 2) * d_alpha[j - 2];
            // 7th order
            phi_o += -(c_order(4, j + 2) * d_alpha[j + 2] - 5. * c_order(4, j + 1) * d_alpha[j + 1] + 10. * c_order(4, j) * d_alpha[j]
                       - 10. * c_order(4, j - 1) * d_alpha[j - 1] + 5. * c_order(4, j - 2) * d_alpha[j - 2]
                       - c_order(4, j - 3) * d_alpha[j - 3]);
        }

        // MP constraint
        xt::xtensor_fixed<double, xt::xshape<stencil_size - 1>> d2;
        xt::xtensor_fixed<double, xt::xshape<stencil_size - 1>> djp12;
        for (std::size_t i = 1; i < stencil_size - 1; i++)
        {
            d2[i] = lambdalpha[i] - lambdalpha[i - 1];
        }

        // minmod limiter between i and i+1
        double dmm   = 0.;
        double d4min = 0.;
        for (std::size_t i = 1; i < stencil_size - 2; i++)
        {
            if (d2[i] * d2[i + 1] < 0.)
            {
                dmm = 0.;
            }
            else
            {
                dmm = std::min(std::abs(d2[i]), std::abs(d2[i + 1]));
                if (d2[i] < 0.)
                {
                    dmm = -dmm;
                }
            }

            if ((4. * d2[i] - d2[i + 1]) * (4. * d2[i + 1] - d2[i]) < 0.)
            {
                d4min = 0.;
            }
            else
            {
                d4min = std::min(std::abs(4. * d2[i] - d2[i + 1]), std::abs(4. * d2[i + 1] - d2[i]));
                if ((4. * d2[i] - d2[i + 1]) < 0.)
                {
                    d4min = -d4min;
                }
            }

            if (d4min * dmm < 0.)
            {
                djp12[i] = 0.;
            }
            else
            {
                djp12[i] = std::min(std::abs(d4min), std::abs(dmm));
                if (d4min < 0)
                {
                    djp12[i] = -djp12[i];
                }
            }
        }
        if (j <= 1)
        {
            djp12[j - 1] = djp12[j];
        }

        // TVD constraints
        double phi_tvdmin = 2. * d_alpha[j - 1] / (nu[j - 1] + zero);
        // double phi_tvdmin = 2. * d_alpha[j-1] * (1.-nu[j-1])/ (nu[j-1]*(1.-nu[j])+zero);
        double phi_tvdmax = 2. * d_alpha[j] / (1 - nu[j] + zero);

        // MP limiter
        double dfo   = 0.5 * phi_o;
        double dfabs = 0.5 * phi_tvdmax;
        double dful  = 0.5 * phi_tvdmin;
        double dfmd  = 0.5 * (dfabs - djp12[j]);
        // double dflc  = 0.5*(dful + (1.-nu[j-1])*djp12[j-1]/(nu[j]+zero));
        double dflc = 0.5 * (dful + (1. - nu[j - 1]) * djp12[j - 1] / (nu[j - 1] + zero));

        double dfmin = std::max(std::min(0., std::min(dfabs, dfmd)), std::min(0., std::min(dful, dflc)));
        double dfmax = std::min(std::max(0., std::max(dfabs, dfmd)), std::max(0., std::max(dful, dflc)));

        double val = (dfmin - dfo) * (dfmax - dfo);
        if (val >= 0.)
        {
            phi_lim = std::max(0., std::min(phi_tvdmin, std::min(phi_o, phi_tvdmax)));
        }
        else
        {
            phi_lim = phi_o;
        }

        return phi_lim;
    }

    template <class Field, std::size_t order>
    auto make_convection_nonlinear_osmp(double& dt)
    {
        // using field_value_t = typename Field::value_type;

        static constexpr std::size_t dim          = Field::dim;
        static constexpr std::size_t field_size   = Field::n_comp;
        static constexpr std::size_t stencil_size = 2 * order;

        using cfg = samurai::FluxConfig<samurai::SchemeType::NonLinear, stencil_size, Field, Field>;

        samurai::FluxDefinition<cfg> osmp;

        samurai::static_for<0, dim>::apply( // for each positive Cartesian direction 'd'
            [&](auto integral_constant_d)
            {
                static constexpr int d = decltype(integral_constant_d)::value;

                auto f = [](auto u) -> samurai::FluxValue<cfg>
                {
                    if constexpr (field_size == 1)
                    {
                        return 0.5 * (u * u);
                        // return u;
                    }
                    else
                    {
                        return u(d) * u;
                    }
                };

                // osmp[d].stencil = line_stencil_from<dim, d, stencil_size>(1-static_cast<int>(order));

                osmp[d].cons_flux_function = [f, &dt](FluxValue<cfg>& flux, const StencilData<cfg>& data, const StencilValues<cfg>& u)
                {
                    static constexpr std::size_t j = order - 1;

                    // for (std::size_t l = 0; l < stencil_size; l++)
                    //     {
                    //         std::cout << " Cell : l = " << l << data.cells[l] << std::endl;
                    //     }
                    // std::cout << " j = " << data.cells[j] << " F = " << f(u[j]) << std::endl;
                    // std::cout << " U = " << u[j] << std::endl;

                    // mean values
                    xt::xtensor_fixed<samurai::FluxValue<cfg>, xt::xshape<stencil_size - 1>> lambda;
                    for (std::size_t l = 0; l < stencil_size - 1; l++)
                    {
                        lambda[l] = 0.5 * (u[l] + u[l + 1]);
                        // lambda[l] = 1.;
                    }

                    auto dx      = data.cell_length;
                    double sigma = dt / dx;

                    // Variation at each interface l+1/2 of the stencil
                    xt::xtensor_fixed<samurai::FluxValue<cfg>, xt::xshape<stencil_size - 1>> dalpha;
                    for (std::size_t l = 0; l < stencil_size - 1; l++)
                    {
                        dalpha[l] = u[l + 1] - u[l];
                    }

                    // Calculation of flux correction for each component
                    xt::xtensor_fixed<double, xt::xshape<stencil_size - 1>> nu;
                    xt::xtensor_fixed<double, xt::xshape<stencil_size - 1>> unmnudalpha;
                    xt::xtensor_fixed<double, xt::xshape<stencil_size - 1>> lambdadalpha;

                    if constexpr (field_size == 1)
                    {
                        if (lambda[j] >= 0)
                        {
                            for (std::size_t l = 0; l < stencil_size - 1; l++)
                            {
                                nu[l] = sigma * std::abs(lambda[l]);
                            }
                            // Factorized term of the Lax-Wendroff scheme
                            for (std::size_t l = 0; l < stencil_size - 1; l++)
                            {
                                unmnudalpha[l]  = std::abs(lambda[l]) * (1. - nu[l]) * dalpha[l];
                                lambdadalpha[l] = lambda[l] * dalpha[l];
                            }
                            flux = f(u[j]);
                        }
                        else
                        {
                            for (std::size_t l = 0; l < stencil_size - 1; l++)
                            {
                                nu[l] = sigma * std::abs(lambda[stencil_size - 2 - l]);
                            }
                            // Factorized term of the Lax-Wendroff scheme
                            for (std::size_t l = 0; l < stencil_size - 1; l++)
                            {
                                unmnudalpha[l]  = std::abs(lambda[stencil_size - 2 - l]) * (1. - nu[l]) * dalpha[stencil_size - 2 - l];
                                lambdadalpha[l] = lambda[stencil_size - 2 - l] * dalpha[stencil_size - 2 - l];
                            }
                            flux = f(u[j + 1]);
                        }

                        // Limiter giving 1st-order Roe scheme
                        double phi_lim = 0.;
                        // Accuracy function for high-order approximations up to 7th-order
                        if (order > 1)
                        {
                            // Flux correction
                            phi_lim = compute_osmp_flux_limiter<decltype(unmnudalpha), decltype(nu), order, stencil_size>(unmnudalpha,
                                                                                                                          lambdadalpha,
                                                                                                                          nu,
                                                                                                                          j);
                        }

                        flux += 0.5 * phi_lim;
                    }
                    else
                    {
                        // For each k-wave
                        samurai::FluxValue<cfg> psi;

                        for (std::size_t k = 0; k < field_size; k++)
                        {
                            if (lambda[j](k) >= 0)
                            {
                                for (std::size_t l = 0; l < stencil_size - 1; l++)
                                {
                                    nu[l] = sigma * std::abs(lambda[l](k));
                                }
                                // Factorized term of the Lax-Wendroff scheme
                                for (std::size_t l = 0; l < stencil_size - 1; l++)
                                {
                                    unmnudalpha[l]  = std::abs(lambda[l](k)) * (1. - nu[l]) * dalpha[l](k);
                                    lambdadalpha[l] = lambda[l](k) * dalpha[l](k);
                                }
                                flux = f(u[j]);
                            }
                            else
                            {
                                for (std::size_t l = 0; l < stencil_size - 1; l++)
                                {
                                    nu[l] = (dt / dx) * std::abs(lambda[stencil_size - 2 - l](k));
                                }
                                // Factorized term of the Lax-Wendroff scheme
                                for (std::size_t l = 0; l < stencil_size - 1; l++)
                                {
                                    unmnudalpha[l] = std::abs(lambda[stencil_size - 2 - l](k)) * (1. - nu[l])
                                                   * dalpha[stencil_size - 2 - l](k);
                                    lambdadalpha[l] = lambda[stencil_size - 2 - l](k) * dalpha[stencil_size - 2 - l](k);
                                }
                                flux = f(u[j + 1]);
                            }

                            // Limiter giving 1st-order Roe scheme
                            double phi_lim = 0.;

                            // Accuracy function for high-order approximations up to 7th-order
                            if (order > 1)
                            {
                                // Flux correction
                                phi_lim = compute_osmp_flux_limiter<decltype(unmnudalpha), decltype(nu), order, stencil_size>(unmnudalpha,
                                                                                                                              lambdadalpha,
                                                                                                                              nu,
                                                                                                                              j);

                                psi(k) = phi_lim;
                            }
                        }

                        // Projection back to the physical space
                        for (std::size_t k = 0; k < field_size; k++)
                        {
                            flux(k) += 0.5 * psi(k);
                        }
                    }
                };
            });

        auto scheme = make_flux_based_scheme(osmp);
        scheme.set_name("convection");
        return scheme;
    }

} // end namespace samurai
