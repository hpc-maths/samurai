#pragma once
#include<samurai/schemes/fv.hpp>

namespace samurai
{

    template <class xtensor_t, class xtensor_nu, class xtensor_c, std::size_t order, std::size_t field_size>
    auto compute_osmp_flux_limiter(xtensor_t& d_alpha, xtensor_nu& nu, xtensor_c& c_order, std::size_t j)
    {
        //using value_type = typename xtensor_t::value_type;

        static constexpr double zero = 1e-14;

        // std::cout << " compute_OS : d_alpha  = " << d_alpha << std::endl;
        // std::cout << " compute_OS : nu  = " << nu << std::endl;
        // std::cout << " compute_OS : c_order = " << c_order << std::endl;

        //value_type flux;
        double phi_o;
        double phi_lim;

        // Lax-Wendroff
        phi_o = d_alpha[j]; 

        // 3rd order
        if( order >= 2)
        {
            phi_o += - c_order(0,j) * d_alpha[j]
                     + c_order(0,j-1) * d_alpha[j-1];
        }

        if( order >= 3)
        {
            // 4th order       
            phi_o += c_order(1,j) * d_alpha[j] 
                    - 2.*c_order(1,j-1) * d_alpha[j-1]
                    + c_order(1,j-2) * d_alpha[j-2];
            // 5th order
            phi_o += - ( c_order(2,j+1) * d_alpha[j+1]
                        - 3.*c_order(2,j) * d_alpha[j]
                        + 3.*c_order(2,j-1) * d_alpha[j-1]
                        - c_order(2,j-2) * d_alpha[j-2] );
        }

        if( order >= 4)
        {
            // 6th order
            phi_o += c_order(3,j+2) * d_alpha[j+2]
                    - 4.*c_order(3,j+1) * d_alpha[j+1]
                    + 6.*c_order(3,j) * d_alpha[j]
                    - 4.*c_order(3,j-1) * d_alpha[j-1]
                    + c_order(3,j-2) * d_alpha[j-2];
            // 7th order
            phi_o += - ( c_order(4,j+2) * d_alpha[j+2]
                        - 5.*c_order(4,j+1) * d_alpha[j+1]
                        + 10.*c_order(4,j) * d_alpha[j]
                        - 10.*c_order(4,j-1) * d_alpha[j-1]
                        + 5.*c_order(4,j-2) * d_alpha[j-2]
                        - c_order(4,j-3) * d_alpha[j-3] );
        }

        phi_o = phi_o / (d_alpha[j] + zero);

        //TVD constraints
        double r = 0.;
        double tvd_min = 0.;
        double tvd_max = 0.;

        r = (d_alpha[j-1] + zero) / (d_alpha[j] + zero);
        
        // TVD constraint
        tvd_min = std::min( phi_o, 2. * r / (nu[j-1]+zero) );
        tvd_max = 2./(1-nu[j]+zero); 
    
        phi_lim = std::max( 0., std::min( tvd_max, tvd_min ) ) ;
            
        // MP constraint
        // to be written

        return phi_lim;
    }

    template <class xtensor_u, std::size_t dim, std::size_t field_size>
    auto compute_Pressure(xtensor_u& uj, const double& gamma)
    {
        double rho_ec = 0.;
        for (std::size_t l = 1; l < dim+1; ++l)
        {
            rho_ec += uj(l)*uj(l);
        }
        rho_ec = 0.5*rho_ec/uj(0);

        // Pressure / (gamma * Mach**2)
        double PsgM2 = (gamma-1) * ( uj(field_size-1) - rho_ec );

        return PsgM2;
    }

    template <class xtensor_u, std::size_t dim, std::size_t field_size>
    auto compute_SoundSpeed(xtensor_u& uj, const double& gamma)
    {
        double rho_ec = 0.;
        for (std::size_t l = 1; l < dim+1; ++l)
        {
            rho_ec += uj(l)*uj(l);
        }
        rho_ec = 0.5*rho_ec / uj(0);

        // C^2 = sqrt( gamma * Pressure / rho )
        double SoundSpeed = std::sqrt( gamma * (gamma-1)*(uj(field_size-1) - rho_ec)/ uj(0) );

        return SoundSpeed;
    }

    template <class xtensor_u, std::size_t dim, std::size_t field_size>
    auto compute_Enthalpy(xtensor_u& uj, const double& gamma)
    {
        double rho_ec = 0.;
        for (std::size_t l = 1; l < dim+1; ++l)
        {
            rho_ec += uj(l)*uj(l);
        }
        rho_ec = 0.5*rho_ec/uj(0);

        double Hj = ( gamma * uj(field_size-1) - (gamma-1) * rho_ec ) / uj(0);

        return Hj;
    }

    template <class xtensor_u, std::size_t dim, std::size_t field_size>
    auto compute_Roemean(const xtensor_u& uj, const xtensor_u& ujp1, const double& gamma)
    {
        double sqrt_rhoj   = std::sqrt(uj(0));
        double sqrt_rhojp1 = std::sqrt(ujp1(0));

        double Hj = compute_Enthalpy<decltype(uj), dim, field_size>(uj, gamma);
        double Hjp1 = compute_Enthalpy<decltype(ujp1), dim, field_size>(ujp1, gamma);

        //xt::xtensor_fixed<double, xt::xshape<field_size>> mean_Roe;
        xtensor_u mean_Roe;

        // density at j+1/2
        mean_Roe(0) = sqrt_rhoj * sqrt_rhojp1;

        // momentum components at j+1/2
        for (std::size_t l = 1; l < dim+1; ++l)
        {
            mean_Roe(l) =  mean_Roe(0) * (sqrt_rhoj*uj(l)/uj(0) +  sqrt_rhojp1*ujp1(l)/ujp1(0)) / (sqrt_rhoj + sqrt_rhojp1);
        }

        // kinetic energy at j+1/2
        double rho_ec = 0.;
        for (std::size_t l = 1; l < dim+1; ++l)
        {
            rho_ec += mean_Roe(l)*mean_Roe(l);
        }
        rho_ec = 0.5*rho_ec/mean_Roe(0);

        // Total energy at j+1/2
        double rhoH_bar = mean_Roe(0) * (sqrt_rhoj*Hj +  sqrt_rhojp1*Hjp1) / (sqrt_rhoj + sqrt_rhojp1);
        double P_bar = (gamma-1.) * (rhoH_bar - rho_ec) / gamma;
        mean_Roe(field_size-1) =  rhoH_bar - P_bar;

        return mean_Roe;
    }

    template <class xtensor_u, std::size_t dim, std::size_t field_size>
    auto compute_EigenValues(xtensor_u& ujp12, const int& dir, const double& gamma)
    {
        double c_bar = compute_SoundSpeed<decltype(ujp12), dim, field_size>(ujp12, gamma);
        double u_bar = ujp12(dir+1)/ujp12(0);

        xt::xtensor_fixed<double, xt::xshape<field_size>> EV;

        EV(0) = u_bar - c_bar;
        for (std::size_t l = 0; l < dim; ++l)
        {
            EV(l+1) = u_bar;
        }
        EV(field_size-1) = u_bar + c_bar;

        return EV;
    }

    template <class xtensor_u, std::size_t dim, std::size_t field_size>
    auto compute_LeftEigenVectors(xtensor_u& ujp12, const std::size_t& dir, const double& gamma)
    {
        xt::xtensor_fixed<double, xt::xshape<field_size, field_size>> L_jp12;

        //double c_bar = std::sqrt(gamma*compute_Pressure<xtensor_u, dim, field_size>(ujp12, gamma)/ujp12(0));
        double c_bar = compute_SoundSpeed<decltype(ujp12), dim, field_size>(ujp12, gamma);
        double gm1s2u2 = 0.;
        for (std::size_t l = 0; l < dim; ++l)
        {
            gm1s2u2 += (ujp12(l+1)/ujp12(0))*(ujp12(l+1)/ujp12(0));
        }
        gm1s2u2 = 0.5*(gamma-1)*gm1s2u2;

        double oneoverc = 1./c_bar;

        if( dim == 1 )
        {
            double u_bar = ujp12(dir+1)/ujp12(0);

            L_jp12(0, 0) = 0.5 * (gm1s2u2 * oneoverc + u_bar) ;
            L_jp12(0, 1) = - 0.5 * ((gamma-1)*u_bar*oneoverc + 1.);
            L_jp12(0, 2) = (gamma-1) * 0.5 * oneoverc;

            L_jp12(1, 0) = c_bar - gm1s2u2 * oneoverc;
            L_jp12(1, 1) = (gamma-1) * u_bar * oneoverc;
            L_jp12(1, 2) = - (gamma-1) * oneoverc;

            L_jp12(2, 0) = 0.5 * (gm1s2u2 * oneoverc - u_bar) ;
            L_jp12(2, 1) = - 0.5 * ((gamma-1)*u_bar * oneoverc - 1.) ;
            L_jp12(2, 2) = (gamma-1) * 0.5 * oneoverc;
        }
        else if( dim == 2 )
        {
            std::array<double, dim> normal;
            normal.fill(0.);
            normal[dir] = 1.;
  
            double unc = 0.;
            for (std::size_t l = 0; l < dim; ++l)
            {
                unc += normal[l] * (ujp12(l+1)/ujp12(0));
            }
            unc = unc * c_bar;

            L_jp12(0,0) =  .5 * ( gm1s2u2 + unc ) * oneoverc;
            L_jp12(0,1) =  - .5 * ( (gamma-1.) * ujp12(1)/ujp12(0) * oneoverc + normal[0] );
            L_jp12(0,2) =  - .5 * ( (gamma-1.) * ujp12(2)/ujp12(0) * oneoverc + normal[1] ) ;
            L_jp12(0,3) =  (gamma-1.) * .5 * oneoverc;

            L_jp12(1,0) =  c_bar - gm1s2u2 * oneoverc;
            L_jp12(1,1) =  (gamma-1.) * ujp12(1)/ujp12(0) * oneoverc;
            L_jp12(1,2) =  (gamma-1.) * ujp12(2)/ujp12(0) * oneoverc;
            L_jp12(1,3) =  - (gamma-1.) * oneoverc;

            L_jp12(2,0) =  (normal[1] * ujp12(1)/ujp12(0) - normal[0] * ujp12(2)/ujp12(0));
            L_jp12(2,1) =  - normal[1];
            L_jp12(2,2) =    normal[0];
            L_jp12(2,3) =  0.;

            L_jp12(3,0) =  .5 * ( gm1s2u2 - unc ) * oneoverc;
            L_jp12(3,1) =  - .5 * ( (gamma-1.) * ujp12(1)/ujp12(0) * oneoverc - normal[0] ) ;
            L_jp12(3,2) =  - .5 * ( (gamma-1.) * ujp12(2)/ujp12(0) * oneoverc - normal[1] ) ;
            L_jp12(3,3) =  (gamma-1.) * .5 * oneoverc;
        }
        else if( dim == 3 )
        {
            std::cout << "L_jp12 in 3D might be implemented !!" << std::endl;
        }
    
        return L_jp12;
    }

    template <class xtensor_u, std::size_t dim, std::size_t field_size>
    auto compute_RightEigenVectors(xtensor_u& ujp12, const std::size_t& dir, const double& gamma)
    {
        xt::xtensor_fixed<double, xt::xshape<field_size, field_size>> R_jp12;

        double c_bar = compute_SoundSpeed<decltype(ujp12), dim, field_size>(ujp12, gamma);
        double H_bar = compute_Enthalpy<xtensor_u, dim, field_size>(ujp12, gamma);

        double oneoverc = 1./c_bar;

        double ec = 0.;
        for (std::size_t l = 0; l < dim; ++l)
        {
            ec += (ujp12(l+1)/ujp12(0)) * (ujp12(l+1)/ujp12(0));
        }
        ec = 0.5 * ec;

        if( dim == 1 )
        {
            double u_bar = ujp12(dir+1)/ujp12(0);

            R_jp12(0, 0) = oneoverc;
            R_jp12(1, 0) = (u_bar*oneoverc - 1.);
            R_jp12(2, 0) = (H_bar*oneoverc - u_bar);

            R_jp12(0, 1) = oneoverc;
            R_jp12(1, 1) = u_bar * oneoverc;
            R_jp12(2, 1) = 0.5 * u_bar * u_bar * oneoverc;

            R_jp12(0, 2) = oneoverc;
            R_jp12(1, 2) = (u_bar*oneoverc + 1.);
            R_jp12(2, 2) = H_bar*oneoverc + u_bar;
        }
        else if( dim == 2 )
        {
            std::array<double, dim> normal;
            normal.fill(0.);
            normal[dir] = 1.;

            double unc = 0.;
            for (std::size_t l = 0; l < dim; ++l)
            {
                unc += normal[l] * (ujp12(l+1)/ujp12(0));
            }
            unc = unc * c_bar;
    
            R_jp12(0,0) = oneoverc;
            R_jp12(1,0) = ujp12(1)/ujp12(0)*oneoverc - normal[0];
            R_jp12(2,0) = ujp12(2)/ujp12(0)*oneoverc - normal[1];
            R_jp12(3,0) = (H_bar - unc) * oneoverc;
    
            R_jp12(0,1) = oneoverc;
            R_jp12(1,1) = ujp12(1)/ujp12(0) * oneoverc;
            R_jp12(2,1) = ujp12(2)/ujp12(0) * oneoverc;
            R_jp12(3,1) = ec * oneoverc;
    
            R_jp12(0,2) = 0.;
            R_jp12(1,2) = - normal[1];
            R_jp12(2,2) =   normal[0];
            R_jp12(3,2) = R_jp12(1,2) * ujp12(1)/ujp12(0) + R_jp12(2,2) * ujp12(2)/ujp12(0);
    
            R_jp12(0,3) = oneoverc;
            R_jp12(1,3) = ujp12(1)/ujp12(0)*oneoverc + normal[0];
            R_jp12(2,3) = ujp12(2)/ujp12(0)*oneoverc + normal[1];
            R_jp12(3,3) = (H_bar + unc) * oneoverc;
        }
        else if( dim == 3 )
        {
            std::cout << "R_jp12 in 3D might be implemented!!" << std::endl;
        }

        return R_jp12;
    }

    template <class Field, std::size_t order>
    auto make_convection_euler_osmp(double& dt)
    {
        //using field_value_t = typename Field::value_type;

        static constexpr std::size_t dim               = Field::dim;
        static constexpr std::size_t field_size        = Field::size;
        static constexpr std::size_t output_field_size = field_size;
        static constexpr std::size_t stencil_size      = 2*order;

        static constexpr double gamma  = 1.4;

        using cfg = samurai::FluxConfig<samurai::SchemeType::NonLinear, output_field_size, stencil_size, Field>;

        samurai::FluxDefinition<cfg> osmp;

        samurai::static_for<0, dim>::apply( // for each positive Cartesian direction 'd'
            [&](auto integral_constant_d)
        {
            static constexpr int d = decltype(integral_constant_d)::value;
    
            auto f = [](auto u) -> samurai::FluxValue<cfg>
                {
                    double pressure = compute_Pressure<samurai::FluxValue<cfg>, dim, field_size>(u, gamma);
                    double enthalpy = compute_Enthalpy<samurai::FluxValue<cfg>, dim, field_size>(u, gamma);
                    
                    // std::cout << " Pressure = " << pressure << std::endl;
                    std::cout << " U = " << u << std::endl;
                    
                    samurai::FluxValue<cfg> flux;
                    flux[0] = u(d+1);

                    for (std::size_t l = 1; l < dim+1; ++l)
                    {
                        flux[l] = u(d+1)*u(l)/u(0);
                    }
                    flux[d+1] += pressure;
                    flux[field_size-1] = u(d+1)*enthalpy;

                    return  flux;
                };
            
            // std::cout << " Stencil avant = " << osmp[d].stencil.stencil << std::endl;
            
            // osmp[d].stencil = line_stencil_from<dim, d, stencil_size>(1-static_cast<int>(order));

            // std::cout << " Stencil apres = " << osmp[d].stencil.stencil << std::endl;

            osmp[d].cons_flux_function = [f, &dt](FluxValue<cfg>& flux, const StencilData<cfg>& data, const StencilValues<cfg>& u)
            {
            
                static constexpr std::size_t j = order-1;
                
                // Roe mean values
                xt::xtensor_fixed<samurai::FluxValue<cfg>, xt::xshape<stencil_size-1>> ujp12;
                for (std::size_t l = 0; l < stencil_size-1; ++l)
                {
                    ujp12[l] = compute_Roemean<samurai::FluxValue<cfg>, dim, field_size>(u[l], u[l+1], gamma);
                }

                // EigenValues
                xt::xtensor_fixed<samurai::FluxValue<cfg>, xt::xshape<stencil_size-1>> lambda;
                for (std::size_t l = 0; l < stencil_size-1; ++l)
                {
                    lambda[l] = compute_EigenValues<samurai::FluxValue<cfg>, dim, field_size>(ujp12[l], d, gamma);
                }

                // EigenVectors
                xt::xtensor_fixed<xt::xtensor_fixed<double, xt::xshape<field_size, field_size>>, xt::xshape<stencil_size-1>> L_jp12;
                for (std::size_t l = 0; l < stencil_size-1; ++l)
                {
                    L_jp12[l] = compute_LeftEigenVectors<samurai::FluxValue<cfg>, dim, field_size>(ujp12[l], d, gamma);
                }

                xt::xtensor_fixed<xt::xtensor_fixed<double, xt::xshape<field_size, field_size>>, xt::xshape<stencil_size-1>> R_jp12;
                for (std::size_t l = 0; l < stencil_size-1; ++l)
                {
                    R_jp12[l] = compute_RightEigenVectors<samurai::FluxValue<cfg>, dim, field_size>(ujp12[l], d, gamma);
                }

                auto dx = data.cell_length;    //cells[j].length;
                double sigma = dt / dx;

                // Flux value centered at the interface j+1/2
                flux = 0.5*( f(u[j]) + f(u[j+1]) );

                // Variation at each interface l+1/2 of the stencil
                xt::xtensor_fixed<samurai::FluxValue<cfg>, xt::xshape<stencil_size-1>> du;
                for (std::size_t l = 0; l < stencil_size-1; ++l)
                {
                    du[l] = u[l+1] - u[l];
                }

                // Projection onto the Eigenvector basis - Riemann Invariants: dalpha = L * du
                xt::xtensor_fixed<samurai::FluxValue<cfg>, xt::xshape<stencil_size-1>> dalpha;
                
                for (std::size_t l = 0; l < stencil_size-1; ++l)
                {
                    for (std::size_t k = 0; k < field_size; ++k)
                    {
                        dalpha[l](k) = 0.;
                        for (std::size_t m = 0; m < field_size; ++m)
                        {
                            dalpha[l](k) += L_jp12[l](k,m) * du[l](m);    
                        }
                    }
                }

                // Calculation of flux correction for each component
                xt::xtensor_fixed<double, xt::xshape<stencil_size-1>> nu;
                xt::xtensor_fixed<double, xt::xshape<stencil_size-1>> unmnudalpha;
                samurai::FluxValue<cfg> psi;

                // For each k-wave
                for ( std::size_t k = 0; k < field_size; ++k)
                {
                    if (lambda[j](k) >= 0)
                    {
                        for (std::size_t l = 0; l < stencil_size-1; ++l)
                        {   
                            nu[l]  = sigma * std::abs(lambda[l](k));
                        }
                        // Factorized term of the Lax-Wendroff scheme
                        for (std::size_t l = 0; l < stencil_size-1; ++l)
                        {
                            unmnudalpha[l] = (1. - nu[l]) * dalpha[l](k);
                        }
                    }
                    else
                    {
                        for (std::size_t l = 0; l < stencil_size-1; ++l)
                        {   
                            nu[l] = (dt / dx) * std::abs(lambda[stencil_size-2-l](k));
                        }
                        // Factorized term of the Lax-Wendroff scheme
                        for (std::size_t l = 0; l < stencil_size-1; ++l)
                        {
                            unmnudalpha[l] = (1. - nu[l]) * dalpha[stencil_size-2-l](k);
                        }
                    }

                    // Limiter giving 1st-order Roe scheme
                    double phi_lim = 0.;
                    psi(k) = 1.;
                    // Accuracy function for high-order approximations up to 7th-order
                    if( order > 1)
                    {

                        xt::xtensor_fixed<double, xt::xshape<5, stencil_size-1>> c_order;
                        for (std::size_t l = 0; l < stencil_size-1; ++l)
                        {
                            c_order(0,l) = (1.+nu[l])/3.;
                            c_order(1,l) = c_order(0,l) * (nu[l]-2)/4.;
                            c_order(2,l) = c_order(1,l) * (nu[l]-3)/5.;
                            c_order(3,l) = c_order(2,l) * (nu[l]+2)/6.;
                            c_order(4,l) = c_order(3,l) * (nu[l]+3)/7.;
                        }

                        //Flux correction 
                        phi_lim = compute_osmp_flux_limiter<decltype(unmnudalpha), decltype(nu), decltype(c_order), order, field_size>(unmnudalpha, nu, c_order, j);

                        psi(k) += - phi_lim * (1.-nu[j]);
                    }
                
                }

                // Projection back to the physical space
                for (std::size_t k = 0; k < field_size; ++k)
                {
                    for (std::size_t m = 0; m < field_size; ++m)
                    {
                        flux(k) += - 0.5 * R_jp12[j](k,m) * std::abs(lambda[j](m)) * psi(m) * dalpha[j](m);
                    } 
                }
                
            };
    
        });

        auto scheme = make_flux_based_scheme(osmp);
        scheme.set_name("convection");
        return scheme;
    
    }


} // end namespace samurai

 