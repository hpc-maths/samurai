#pragma once
#include <samurai/schemes/fv.hpp>

namespace samurai
{

    template <class Field, std::size_t order>
    auto make_convection_os(double dt)
    {
        static constexpr std::size_t dim           = Field::dim;
        static constexpr std::size_t n_comp        = Field::n_comp;
        static constexpr std::size_t output_n_comp = n_comp;
        static constexpr std::size_t stencil_size  = 2 * (order / 2 + order % 2);

        using cfg = FluxConfig<SchemeType::NonLinear, output_n_comp, stencil_size, Field>;

        samurai::FluxDefinition<cfg> ostvd;

        samurai::static_for<0, dim>::apply( // for each positive Cartesian direction 'd'
            [&](auto integral_constant_d)
            {
                static constexpr int d = decltype(integral_constant_d)::value;

                static constexpr std::size_t j = (order / 2 + order % 2) - 1;

                auto f = [](auto u) -> FluxValue<cfg>
                {
                    return 0.5 * u * u;
                };

                ostvd[d].cons_flux_function = [&](FluxValue<cfg>& flux, const StencilData<cfg>& data, const StencilValues<cfg>& u)
                {
                    auto dx = data.cell_length;

                    xt::xtensor_fixed<double, xt::xshape<stencil_size - 1>> nu;
                    for (std::size_t l = 0; l < stencil_size - 1; ++l)
                    {
                        nu[l] = (dt / dx) * 0.5 * std::abs(u[l] + u[l + 1]);
                    }

                    if ((u[j] + u[j + 1]) >= 0)
                    {
                        if (order > 0)
                        {
                            flux = f(u[j]);
                        }
                        if (order > 1)
                        {
                            double c2 = 0.5 * (nu[j] - 1);
                            flux += c2 * (f(u[j]) - f(u[j + 1]));
                        }
                        if (order > 2)
                        {
                            double c3jp12 = ((nu[j] - 1) * (nu[j] + 1) / 6.);
                            double c3jm12 = ((nu[j - 1] - 1) * (nu[j - 1] + 1) / 6.);
                            flux += c3jp12 * (f(u[j + 1]) - f(u[j])) - c3jm12 * (f(u[j]) - f(u[j - 1]));
                        }
                        if (order > 3)
                        {
                            double c4jm12 = ((nu[j - 1] - 1) * (nu[j - 1] + 1) * (nu[j - 1] - 2) / 24.);
                            double c4jp12 = ((nu[j] - 1) * (nu[j] + 1) * (nu[j] - 2) / 24.);
                            double c4jp32 = ((nu[j + 1] - 1) * (nu[j + 1] + 1) * (nu[j + 1] - 2) / 24.);
                            flux += -(c4jp32 * (f(u[j + 2]) - f(u[j + 1])) - 2 * c4jp12 * (f(u[j + 1]) - f(u[j]))
                                      + c4jm12 * (f(u[j]) - f(u[j - 1])));
                        }
                        if (order > 4)
                        {
                            double c5m32 = ((nu[j - 2] - 1) * (nu[j - 2] + 1) * (nu[j - 2] - 2) * (nu[j - 2] + 2) / 120.);
                            double c5m12 = ((nu[j - 1] - 1) * (nu[j - 1] + 1) * (nu[j - 1] - 2) * (nu[j - 1] + 2) / 120.);
                            double c5p12 = ((nu[j] - 1) * (nu[j] + 1) * (nu[j] - 2) * (nu[j] + 2) / 120.);
                            double c5p32 = ((nu[j + 1] - 1) * (nu[j + 1] + 1) * (nu[j + 1] - 2) * (nu[j + 1] + 2) / 120.);
                            flux += c5p32 * (f(u[j + 2]) - f(u[j + 1])) - 3. * c5p12 * (f(u[j + 1]) - f(u[j]))
                                  + 3. * c5m12 * (f(u[j]) - f(u[j - 1])) - c5m32 * (f(u[j - 1]) - f(u[j - 2]));
                        }
                    }
                    else
                    {
                        if (order > 0)
                        {
                            flux = f(u[j + 1]);
                        }
                        if (order > 1)
                        {
                            double c2 = 0.5 * (nu[j] - 1);
                            flux += c2 * (f(u[j + 1]) - f(u[j]));
                        }
                        if (order > 2)
                        {
                            double c3jp12 = ((nu[j] - 1) * (nu[j] + 1) / 6.);
                            double c3jm12 = ((nu[j + 1] - 1) * (nu[j + 1] + 1) / 6.);
                            flux += c3jp12 * (f(u[j]) - f(u[j + 1])) - c3jm12 * (f(u[j + 1]) - f(u[j + 2]));
                        }
                        if (order > 3)
                        {
                            double c4jm12 = ((nu[j + 1] - 1) * (nu[j + 1] + 1) * (nu[j + 1] - 2) / 24.);
                            double c4jp12 = ((nu[j] - 1) * (nu[j] + 1) * (nu[j] - 2) / 24.);
                            double c4jp32 = ((nu[j - 1] - 1) * (nu[j - 1] + 1) * (nu[j - 1] - 2) / 24.);
                            flux += -(c4jp32 * (f(u[j - 1]) - f(u[j])) - 2 * c4jp12 * (f(u[j]) - f(u[j + 1]))
                                      + c4jm12 * (f(u[j + 1]) - f(u[j + 2])));
                        }
                        if (order > 4)
                        {
                            double c5m32 = ((nu[j + 2] - 1) * (nu[j + 2] + 1) * (nu[j + 2] - 2) * (nu[j + 2] + 2) / 120.);
                            double c5m12 = ((nu[j + 1] - 1) * (nu[j + 1] + 1) * (nu[j + 1] - 2) * (nu[j + 1] + 2) / 120.);
                            double c5p12 = ((nu[j] - 1) * (nu[j] + 1) * (nu[j] - 2) * (nu[j] + 2) / 120.);
                            double c5p32 = ((nu[j - 1] - 1) * (nu[j - 1] + 1) * (nu[j - 1] - 2) * (nu[j - 1] + 2) / 120.);
                            flux += c5p32 * (f(u[j - 1]) - f(u[j])) - 3. * c5p12 * (f(u[j]) - f(u[j + 1]))
                                  + 3. * c5m12 * (f(u[j + 1]) - f(u[j + 2])) - c5m32 * (f(u[j + 2]) - f(u[j + 3]));
                        }
                    }
                };
            });

        auto scheme = make_flux_based_scheme(ostvd);
        scheme.set_name("convection");
        return scheme;
    }

} // end namespace samurai
