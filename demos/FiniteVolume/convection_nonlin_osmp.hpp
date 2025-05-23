#pragma once
#include <samurai/schemes/fv.hpp>

namespace samurai
{

    template <class cfg, class xtensor_t, class xtensor_nu, class xtensor_c, std::size_t order, std::size_t field_size>
    auto compute_osmp_flux_limiter(xtensor_t& d_alpha, xtensor_nu& nu, xtensor_c& c_order, std::size_t j)
    {
        // using value_type = typename xtensor_t::value_type;

        static constexpr double zero = 1e-14;

        // std::cout << " compute_OS : d_alpha  = " << d_alpha << std::endl;
        // std::cout << " compute_OS : nu  = " << nu << std::endl;
        // std::cout << " compute_OS : c_order = " << c_order << std::endl;

        // value_type flux;
        samurai::FluxValue<cfg> phi_o;
        samurai::FluxValue<cfg> phi_lim;

        // Lax-Wendroff
        phi_o = d_alpha[j];

        // 3rd order
        if (order >= 2)
        {
            phi_o += -c_order(0, j) * d_alpha[j] + c_order(0, j - 1) * d_alpha[j - 1];
        }

        if (order >= 3)
        {
            // 4th order
            phi_o += c_order(1, j) * d_alpha[j] - 2. * c_order(1, j - 1) * d_alpha[j - 1] + c_order(1, j - 2) * d_alpha[j - 2];
            // 5th order
            phi_o += -(c_order(2, j + 1) * d_alpha[j + 1] - 3. * c_order(2, j) * d_alpha[j] + 3. * c_order(2, j - 1) * d_alpha[j - 1]
                       - c_order(2, j - 2) * d_alpha[j - 2]);
        }

        if (order >= 4)
        {
            // 6th order
            phi_o += c_order(3, j + 2) * d_alpha[j + 2] - 4. * c_order(3, j + 1) * d_alpha[j + 1] + 6. * c_order(3, j) * d_alpha[j]
                   - 4. * c_order(3, j - 1) * d_alpha[j - 1] + c_order(3, j - 2) * d_alpha[j - 2];
            // 7th order
            phi_o += -(c_order(4, j + 2) * d_alpha[j + 2] - 5. * c_order(4, j + 1) * d_alpha[j + 1] + 10. * c_order(4, j) * d_alpha[j]
                       - 10. * c_order(4, j - 1) * d_alpha[j - 1] + 5. * c_order(4, j - 2) * d_alpha[j - 2]
                       - c_order(4, j - 3) * d_alpha[j - 3]);
        }

        phi_o = phi_o / (d_alpha[j] + zero);

        // TVD constraints
        //  samurai::FluxValue<cfg> r;
        //  samurai::FluxValue<cfg> val_min;
        //  samurai::FluxValue<cfg> val_max;
        //  samurai::FluxValue<cfg> val_zero;
        double r       = 0.;
        double val_min = 0.;
        double val_max = 0.;

        if constexpr (field_size == 1)
        {
            r = (d_alpha[j - 1] + zero) / (d_alpha[j] + zero);

            val_min = std::min(phi_o, 2. * r / (nu[j - 1] + zero));
            val_max = 2. / (1 - nu[j] + zero);

            phi_lim = std::max(0., std::min(val_max, val_min));
        }
        else
        {
            for (std::size_t l = 0; l < field_size; ++l)
            {
                r = (d_alpha[j - 1](l) + zero) / (d_alpha[j](l) + zero);

                val_min = std::min(phi_o(l), 2. * r / (nu[j - 1] + zero));
                val_max = 2. / (1 - nu[j] + zero);

                phi_lim(l) = std::max(0., std::min(val_max, val_min));
            }
        }
        // r = (d_alpha[j-1] + zero) / (d_alpha[j] + zero);

        // val_min = xt::amin( phi_o, 2./ (nu[j-1]+zero) * r  );
        // val_max.fill(2./(1-nu[j]+zero));
        // val_zero.fill(0.);

        // phi_lim = xt::amax( val_zero, xt::amax( xt::amin( val_max, val_min ) ) );

        // phi_lim = xt::amax( 0., xt::amax( xt::amin( 2./(1-nu[j]+zero),
        //                                   xt::amin( phi_o, xt::eval( 2./ (nu[j-1]+zero) * r ) ) ) ) );

        // phi_lim = xt::amax( 0., xt::amax( xt::amin( 2./(1-nu[j]+zero),
        //                                   xt::amin( phi_o, 2.*(d_alpha[j-1] + zero) / ((d_alpha[j] + zero)*(nu[j-1]+zero)) ) ) ) );

        return phi_lim;
    }

    template <class Field, std::size_t order>
    auto make_convection_osmp(double& dt)
    {
        // using field_value_t = typename Field::value_type;

        static constexpr std::size_t dim               = Field::dim;
        static constexpr std::size_t field_size        = Field::n_comp;
        static constexpr std::size_t output_field_size = field_size;
        static constexpr std::size_t stencil_size      = 2 * order;

        using cfg = samurai::FluxConfig<samurai::SchemeType::NonLinear, output_field_size, stencil_size, Field>;

        samurai::FluxDefinition<cfg> osmp;

        samurai::static_for<0, dim>::apply( // for each positive Cartesian direction 'd'
            [&](auto integral_constant_d)
            {
                static constexpr int d = decltype(integral_constant_d)::value;

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

                // osmp[d].stencil = line_stencil_from<dim, d, stencil_size>(1 - static_cast<int>(order));

                osmp[d].cons_flux_function = [f, &dt](FluxValue<cfg>& flux, const StencilData<cfg>& data, const StencilValues<cfg>& u)
                {
                    static constexpr std::size_t j = order - 1;

                    // Convective velocity mean values
                    xt::xtensor_fixed<double, xt::xshape<stencil_size - 1>> lambda;
                    for (std::size_t l = 0; l < stencil_size - 1; ++l)
                    {
                        lambda[l] = 0.5 * (u[l](d) + u[l + 1](d));
                    }

                    auto dx = data.cell_length; // cells[j].length;

                    // Calculation of flux correction for each component
                    xt::xtensor_fixed<double, xt::xshape<stencil_size - 1>> nu;
                    xt::xtensor_fixed<samurai::FluxValue<cfg>, xt::xshape<stencil_size - 1>> delta_u;
                    xt::xtensor_fixed<samurai::FluxValue<cfg>, xt::xshape<stencil_size - 1>> d_alpha;

                    for (std::size_t l = 0; l < stencil_size - 1; ++l)
                    {
                        delta_u[l] = u[l + 1] - u[l];
                    }

                    if (lambda[j] >= 0)
                    {
                        for (std::size_t l = 0; l < stencil_size - 1; ++l)
                        {
                            nu[l] = (dt / dx) * std::abs(lambda[l]);
                        }

                        for (std::size_t l = 0; l < stencil_size - 1; ++l)
                        {
                            d_alpha[l] = (1. - nu[l]) * delta_u[l];
                        }

                        // Flux value centered at the interface j+1/2
                        flux = f(u[j]);
                    }
                    else
                    {
                        for (std::size_t l = 0; l < stencil_size - 1; ++l)
                        {
                            nu[l] = (dt / dx) * std::abs(lambda[stencil_size - 2 - l]);
                        }

                        for (std::size_t l = 0; l < stencil_size - 1; ++l)
                        {
                            d_alpha[l] = (1. - nu[l]) * delta_u[stencil_size - 2 - l];
                        }

                        // Flux value centered at the interface j+1/2
                        flux = f(u[j + 1]);
                    }

                    // coefficients for high-order approximations up to 7th-order
                    if (order > 1)
                    {
                        xt::xtensor_fixed<double, xt::xshape<5, stencil_size - 1>> c_order;
                        for (std::size_t l = 0; l < stencil_size - 1; ++l)
                        {
                            c_order(0, l) = (1. + nu[l]) / 3.;
                            c_order(1, l) = c_order(0, l) * (nu[l] - 2) / 4.;
                            c_order(2, l) = c_order(1, l) * (nu[l] - 3) / 5.;
                            c_order(3, l) = c_order(2, l) * (nu[l] + 2) / 6.;
                            c_order(4, l) = c_order(3, l) * (nu[l] + 3) / 7.;
                        }

                        // Flux correction
                        samurai::FluxValue<cfg> phi_lim;
                        // xt::xtensor_fixed<double, xt::xshape<field_size>> phi_lim;

                        phi_lim = compute_osmp_flux_limiter<cfg, decltype(d_alpha), decltype(nu), decltype(c_order), order, field_size>(
                            d_alpha,
                            nu,
                            c_order,
                            j);
                        flux += 0.5 * std::abs(lambda[j]) * phi_lim * (1 - nu[j]) * delta_u[j];
                    }
                };
            });

        auto scheme = make_flux_based_scheme(osmp);
        scheme.set_name("convection");
        return scheme;
    }

} // end namespace samurai
