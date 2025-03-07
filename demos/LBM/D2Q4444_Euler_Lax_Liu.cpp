// Copyright 2018-2025 the samurai's authors
// SPDX-License-Identifier:  BSD-3-Clause

#include <math.h>
#include <vector>

#include <CLI/CLI.hpp>

#include <samurai/bc.hpp>
#include <samurai/field.hpp>
#include <samurai/io/hdf5.hpp>
#include <samurai/mr/adapt.hpp>
#include <samurai/mr/mesh.hpp>
#include <samurai/reconstruction.hpp>
#include <samurai/samurai.hpp>

#include <samurai/load_balancing.hpp>
#include <samurai/load_balancing_sfc.hpp>
#include <samurai/load_balancing_diffusion.hpp>
#include <samurai/load_balancing_force.hpp>
#include <samurai/load_balancing_diffusion_interval.hpp>
#include <samurai/load_balancing_void.hpp>
#include <samurai/load_balancing_life.hpp>
#include <samurai/timers.hpp>

double gm = 1.4; // Gas constant

template <class Config>
auto init_f(samurai::MRMesh<Config>& mesh, int config, double lambda)
{
    constexpr std::size_t nvel = 16;
    using mesh_id_t            = typename samurai::MRMesh<Config>::mesh_id_t;

    auto f = samurai::make_field<double, nvel>("f", mesh);
    f.fill(0);

    samurai::for_each_cell(
        mesh[mesh_id_t::cells],
        [&](auto& cell)
        {
            auto center = cell.center();
            auto x      = center[0];
            auto y      = center[1];

            double rho = 1.0; // Density
            double qx  = 0.0; // x-momentum
            double qy  = 0.0; // y-momentum
            double e   = 0.0;
            double p   = 1.0;

            std::array<double, 4> rho_quad, u_x_quad, u_y_quad, p_quad;

            switch (config)
            {
                case 11:
                {
                    rho_quad = {0.8, 0.5313, 0.5313, 1.};
                    u_x_quad = {0.1, 0.8276, 0.1, 0.1};
                    u_y_quad = {0., 0., 0.7276, 0.0};
                    p_quad   = {0.4, 0.4, 0.4, 1.};
                    break;
                }
                case 12:
                {
                    rho_quad = {0.8, 1., 1., 0.5313};
                    u_x_quad = {0., 0.7276, 0., 0.0};
                    u_y_quad = {0., 0., 0.7276, 0.0};
                    p_quad   = {1., 1., 1., 0.4};
                    break;
                }
                case 17:
                {
                    rho_quad = {1.0625, 2., 0.5197, 1.};
                    u_x_quad = {0., 0., 0., 0.0};
                    u_y_quad = {0.2145, -0.3, -1.1259, 0.0};
                    p_quad   = {0.4, 1., 0.4, 1.5};
                    break;
                }
                case 3:
                {
                    rho_quad = {0.138, 0.5323, 0.5323, 1.5};
                    u_x_quad = {1.206, 1.206, 0., 0.0};
                    u_y_quad = {1.206, 0., 1.206, 0.0};
                    p_quad   = {0.029, 0.3, 0.3, 1.5};
                    break;
                }
                default:
                {
                    rho_quad = {0.8, 1., 1., 0.5313};
                    u_x_quad = {0., 0.7276, 0., 0.0};
                    u_y_quad = {0., 0., 0.7276, 0.0};
                    p_quad   = {1., 1., 1., 0.4};
                }
            }

            if (x < 0.5)
            {
                if (y < 0.5)
                {
                    rho = rho_quad[0];
                    qx  = rho * u_x_quad[0];
                    qy  = rho * u_y_quad[0];
                    p   = p_quad[0];
                }
                else
                {
                    rho = rho_quad[1];
                    qx  = rho * u_x_quad[1];
                    qy  = rho * u_y_quad[1];
                    p   = p_quad[1];
                }
            }
            else
            {
                if (y < 0.5)
                {
                    rho = rho_quad[2];
                    qx  = rho * u_x_quad[2];
                    qy  = rho * u_y_quad[2];
                    p   = p_quad[2];
                }
                else
                {
                    rho = rho_quad[3];
                    qx  = rho * u_x_quad[3];
                    qy  = rho * u_y_quad[3];
                    p   = p_quad[3];
                }
            }

            e = p / (gm - 1.) + 0.5 * (qx * qx + qy * qy) / rho;

            // Conserved momenti
            double m0_0 = rho;
            double m1_0 = qx;
            double m2_0 = qy;
            double m3_0 = e;

            // Non conserved at equilibrium
            double m0_1 = m1_0;
            double m0_2 = m2_0;
            double m0_3 = 0.0;

            double m1_1 = (3. / 2. - gm / 2.) * (m1_0 * m1_0) / (m0_0) + (1. / 2. - gm / 2.) * (m2_0 * m2_0) / (m0_0) + (gm - 1.) * m3_0;
            double m1_2 = m1_0 * m2_0 / m0_0;
            double m1_3 = 0.0;

            double m2_1 = m1_0 * m2_0 / m0_0;

            double m2_2 = (3. / 2. - gm / 2.) * (m2_0 * m2_0) / (m0_0) + (1. / 2. - gm / 2.) * (m1_0 * m1_0) / (m0_0) + (gm - 1.) * m3_0;
            double m2_3 = 0.0;

            double m3_1 = gm * (m1_0 * m3_0) / (m0_0) - (gm / 2. - 1. / 2.) * (m1_0 * m1_0 * m1_0) / (m0_0 * m0_0)
                        - (gm / 2. - 1. / 2.) * (m1_0 * m2_0 * m2_0) / (m0_0 * m0_0);
            double m3_2 = gm * (m2_0 * m3_0) / (m0_0) - (gm / 2. - 1. / 2.) * (m2_0 * m2_0 * m2_0) / (m0_0 * m0_0)
                        - (gm / 2. - 1. / 2.) * (m2_0 * m1_0 * m1_0) / (m0_0 * m0_0);
            double m3_3 = 0.0;

            // We come back to the distributions
            f[cell][0] = .25 * m0_0 + .5 / lambda * (m0_1) + .25 / (lambda * lambda) * m0_3;
            f[cell][1] = .25 * m0_0 + .5 / lambda * (m0_2)-.25 / (lambda * lambda) * m0_3;
            f[cell][2] = .25 * m0_0 - .5 / lambda * (m0_1) + .25 / (lambda * lambda) * m0_3;
            f[cell][3] = .25 * m0_0 - .5 / lambda * (m0_2)-.25 / (lambda * lambda) * m0_3;

            f[cell][4] = .25 * m1_0 + .5 / lambda * (m1_1) + .25 / (lambda * lambda) * m1_3;
            f[cell][5] = .25 * m1_0 + .5 / lambda * (m1_2)-.25 / (lambda * lambda) * m1_3;
            f[cell][6] = .25 * m1_0 - .5 / lambda * (m1_1) + .25 / (lambda * lambda) * m1_3;
            f[cell][7] = .25 * m1_0 - .5 / lambda * (m1_2)-.25 / (lambda * lambda) * m1_3;

            f[cell][8]  = .25 * m2_0 + .5 / lambda * (m2_1) + .25 / (lambda * lambda) * m2_3;
            f[cell][9]  = .25 * m2_0 + .5 / lambda * (m2_2)-.25 / (lambda * lambda) * m2_3;
            f[cell][10] = .25 * m2_0 - .5 / lambda * (m2_1) + .25 / (lambda * lambda) * m2_3;
            f[cell][11] = .25 * m2_0 - .5 / lambda * (m2_2)-.25 / (lambda * lambda) * m2_3;

            f[cell][12] = .25 * m3_0 + .5 / lambda * (m3_1) + .25 / (lambda * lambda) * m3_3;
            f[cell][13] = .25 * m3_0 + .5 / lambda * (m3_2)-.25 / (lambda * lambda) * m3_3;
            f[cell][14] = .25 * m3_0 - .5 / lambda * (m3_1) + .25 / (lambda * lambda) * m3_3;
            f[cell][15] = .25 * m3_0 - .5 / lambda * (m3_2)-.25 / (lambda * lambda) * m3_3;
        });
    return f;
}

template <class Field>
void one_time_step(Field& f,
                   const double lambda,
                   const double sq_rho,
                   const double sxy_rho,
                   const double sq_q,
                   const double sxy_q,
                   const double sq_e,
                   const double sxy_e)
{
    constexpr std::size_t dim = Field::dim;
    auto& mesh                = f.mesh();
    using mesh_id_t           = typename Field::mesh_t::mesh_id_t;

    std::size_t max_level = mesh.max_level();

    samurai::times::timers.start("ugm-step");
    samurai::update_ghost_mr(f);
    samurai::times::timers.stop("ugm-step");

    samurai::times::timers.start("field-step");
    Field new_f{"new_f", mesh};
    new_f.array().fill(0.);
    Field advected{"advected", mesh};
    advected.array().fill(0.);
    samurai::times::timers.stop("field-step");

    samurai::times::timers.start("lbm-step");
    samurai::for_each_interval(
        mesh[mesh_id_t::cells],
        [&](std::size_t level, auto& i, auto& index)
        {
            auto j      = index[0];
            auto jump   = max_level - level;
            double coef = 1. / (1 << (dim * jump));
            for (std::size_t scheme_n = 0; scheme_n < 4; ++scheme_n)
            { // We have 4 schemes
                advected(0 + 4 * scheme_n, level, i, j) = f(0 + 4 * scheme_n, level, i-1, j);
                advected(1 + 4 * scheme_n, level, i, j) = f(1 + 4 * scheme_n, level, i, j-1);
                advected(2 + 4 * scheme_n, level, i, j) = f(2 + 4 * scheme_n, level, i+1, j);
                advected(3 + 4 * scheme_n, level, i, j) = f(3 + 4 * scheme_n, level, i, j+1);
            
            //     advected(
            //         0 + 4 * scheme_n,
            //         level,
            //         i,
            //         j) = f(0 + 4 * scheme_n, level, i, j)
            //            + coef * samurai::portion(f, 0 + 4 * scheme_n, level, i - 1, j, jump, {(1 << jump) - 1, (1 << jump)}, {0, (1 << jump)})
            //            - coef * samurai::portion(f, 0 + 4 * scheme_n, level, i, j, jump, {(1 << jump) - 1, (1 << jump)}, {0, (1 << jump)});

            //     advected(
            //         1 + 4 * scheme_n,
            //         level,
            //         i,
            //         j) = f(1 + 4 * scheme_n, level, i, j)
            //            + coef * samurai::portion(f, 1 + 4 * scheme_n, level, i, j - 1, jump, {0, (1 << jump)}, {(1 << jump) - 1, (1 << jump)})
            //            - coef * samurai::portion(f, 1 + 4 * scheme_n, level, i, j, jump, {0, (1 << jump)}, {(1 << jump) - 1, (1 << jump)});

            //     advected(2 + 4 * scheme_n, level, i, j) = f(
            //         2 + 4 * scheme_n,
            //         level,
            //         i,
            //         j) = f(2 + 4 * scheme_n, level, i, j)
            //            + coef * samurai::portion(f, 2 + 4 * scheme_n, level, i + 1, j, jump, {0, 1}, {0, (1 << jump)})
            //            - coef * samurai::portion(f, 2 + 4 * scheme_n, level, i, j, jump, {0, 1}, {0, (1 << jump)});
            //     advected(3 + 4 * scheme_n,
            //              level,
            //              i,
            //              j) = f(3 + 4 * scheme_n, level, i, j)
            //                 + coef * samurai::portion(f, 3 + 4 * scheme_n, level, i, j + 1, jump, {0, (1 << jump)}, {0, 1})
            //                 - coef * samurai::portion(f, 3 + 4 * scheme_n, level, i, j, jump, {0, (1 << jump)}, {0, 1});
            }

            // We compute the advected momenti
            auto m0_0 = xt::eval(advected(0, level, i, j) + advected(1, level, i, j) + advected(2, level, i, j) + advected(3, level, i, j));
            auto m0_1 = xt::eval(lambda * (advected(0, level, i, j) - advected(2, level, i, j)));
            auto m0_2 = xt::eval(lambda * (advected(1, level, i, j) - advected(3, level, i, j)));
            auto m0_3 = xt::eval(
                lambda * lambda * (advected(0, level, i, j) - advected(1, level, i, j) + advected(2, level, i, j) - advected(3, level, i, j)));

            auto m1_0 = xt::eval(advected(4, level, i, j) + advected(5, level, i, j) + advected(6, level, i, j) + advected(7, level, i, j));
            auto m1_1 = xt::eval(lambda * (advected(4, level, i, j) - advected(6, level, i, j)));
            auto m1_2 = xt::eval(lambda * (advected(5, level, i, j) - advected(7, level, i, j)));
            auto m1_3 = xt::eval(
                lambda * lambda * (advected(4, level, i, j) - advected(5, level, i, j) + advected(6, level, i, j) - advected(7, level, i, j)));

            auto m2_0 = xt::eval(advected(8, level, i, j) + advected(9, level, i, j) + advected(10, level, i, j) + advected(11, level, i, j));
            auto m2_1 = xt::eval(lambda * (advected(8, level, i, j) - advected(10, level, i, j)));
            auto m2_2 = xt::eval(lambda * (advected(9, level, i, j) - advected(11, level, i, j)));
            auto m2_3 = xt::eval(
                lambda * lambda
                * (advected(8, level, i, j) - advected(9, level, i, j) + advected(10, level, i, j) - advected(11, level, i, j)));

            auto m3_0 = xt::eval(advected(12, level, i, j) + advected(13, level, i, j) + advected(14, level, i, j)
                                 + advected(15, level, i, j));
            auto m3_1 = xt::eval(lambda * (advected(12, level, i, j) - advected(14, level, i, j)));
            auto m3_2 = xt::eval(lambda * (advected(13, level, i, j) - advected(15, level, i, j)));
            auto m3_3 = xt::eval(
                lambda * lambda
                * (advected(12, level, i, j) - advected(13, level, i, j) + advected(14, level, i, j) - advected(15, level, i, j)));

            m0_1 = (1 - sq_rho) * m0_1 + sq_rho * (m1_0);
            m0_2 = (1 - sq_rho) * m0_2 + sq_rho * (m2_0);
            m0_3 = (1 - sxy_rho) * m0_3;

            m1_1 = (1 - sq_q) * m1_1
                 + sq_q * ((3. / 2. - gm / 2.) * (m1_0 * m1_0) / (m0_0) + (1. / 2. - gm / 2.) * (m2_0 * m2_0) / (m0_0) + (gm - 1.) * m3_0);
            m1_2 = (1 - sq_q) * m1_2 + sq_q * (m1_0 * m2_0 / m0_0);
            m1_3 = (1 - sxy_q) * m1_3;

            m2_1 = (1 - sq_q) * m2_1 + sq_q * (m1_0 * m2_0 / m0_0);
            m2_2 = (1 - sq_q) * m2_2
                 + sq_q * ((3. / 2. - gm / 2.) * (m2_0 * m2_0) / (m0_0) + (1. / 2. - gm / 2.) * (m1_0 * m1_0) / (m0_0) + (gm - 1.) * m3_0);
            m2_3 = (1 - sxy_q) * m2_3;

            m3_1 = (1 - sq_e) * m3_1
                 + sq_e
                       * (gm * (m1_0 * m3_0) / (m0_0) - (gm / 2. - 1. / 2.) * (m1_0 * m1_0 * m1_0) / (m0_0 * m0_0)
                          - (gm / 2. - 1. / 2.) * (m1_0 * m2_0 * m2_0) / (m0_0 * m0_0));
            m3_2 = (1 - sq_e) * m3_2
                 + sq_e
                       * (gm * (m2_0 * m3_0) / (m0_0) - (gm / 2. - 1. / 2.) * (m2_0 * m2_0 * m2_0) / (m0_0 * m0_0)
                          - (gm / 2. - 1. / 2.) * (m2_0 * m1_0 * m1_0) / (m0_0 * m0_0));
            m3_3 = (1 - sxy_e) * m3_3;

            new_f(0, level, i, j) = .25 * m0_0 + .5 / lambda * (m0_1) + .25 / (lambda * lambda) * m0_3;
            new_f(1, level, i, j) = .25 * m0_0 + .5 / lambda * (m0_2)-.25 / (lambda * lambda) * m0_3;
            new_f(2, level, i, j) = .25 * m0_0 - .5 / lambda * (m0_1) + .25 / (lambda * lambda) * m0_3;
            new_f(3, level, i, j) = .25 * m0_0 - .5 / lambda * (m0_2)-.25 / (lambda * lambda) * m0_3;

            new_f(4, level, i, j) = .25 * m1_0 + .5 / lambda * (m1_1) + .25 / (lambda * lambda) * m1_3;
            new_f(5, level, i, j) = .25 * m1_0 + .5 / lambda * (m1_2)-.25 / (lambda * lambda) * m1_3;
            new_f(6, level, i, j) = .25 * m1_0 - .5 / lambda * (m1_1) + .25 / (lambda * lambda) * m1_3;
            new_f(7, level, i, j) = .25 * m1_0 - .5 / lambda * (m1_2)-.25 / (lambda * lambda) * m1_3;

            new_f(8, level, i, j)  = .25 * m2_0 + .5 / lambda * (m2_1) + .25 / (lambda * lambda) * m2_3;
            new_f(9, level, i, j)  = .25 * m2_0 + .5 / lambda * (m2_2)-.25 / (lambda * lambda) * m2_3;
            new_f(10, level, i, j) = .25 * m2_0 - .5 / lambda * (m2_1) + .25 / (lambda * lambda) * m2_3;
            new_f(11, level, i, j) = .25 * m2_0 - .5 / lambda * (m2_2)-.25 / (lambda * lambda) * m2_3;

            new_f(12, level, i, j) = .25 * m3_0 + .5 / lambda * (m3_1) + .25 / (lambda * lambda) * m3_3;
            new_f(13, level, i, j) = .25 * m3_0 + .5 / lambda * (m3_2)-.25 / (lambda * lambda) * m3_3;
            new_f(14, level, i, j) = .25 * m3_0 - .5 / lambda * (m3_1) + .25 / (lambda * lambda) * m3_3;
            new_f(15, level, i, j) = .25 * m3_0 - .5 / lambda * (m3_2)-.25 / (lambda * lambda) * m3_3;
        });
    
    samurai::times::timers.stop("lbm-step");

    std::swap(f.array(), new_f.array());
}

template <class Field>
void save_solution(Field& f, double eps, std::size_t ite, std::size_t freq_out, std::string ext = "")
{
    using value_t = typename Field::value_type;

    if( ite % freq_out != 0 ) return; 

    auto& mesh = f.mesh();
    auto level = samurai::make_field<std::size_t, 1>("level", mesh);
    auto rho   = samurai::make_field<value_t, 1>("rho", mesh);
    auto qx    = samurai::make_field<value_t, 1>("qx", mesh);
    auto qy    = samurai::make_field<value_t, 1>("qy", mesh);
    auto e     = samurai::make_field<value_t, 1>("e", mesh);
    auto s     = samurai::make_field<value_t, 1>("entropy", mesh);

    samurai::for_each_cell(mesh,
                           [&](auto& cell)
                           {
                               level[cell] = cell.level;
                               rho[cell]   = f[cell][0] + f[cell][1] + f[cell][2] + f[cell][3];
                               qx[cell]    = f[cell][4] + f[cell][5] + f[cell][6] + f[cell][7];
                               qy[cell]    = f[cell][8] + f[cell][9] + f[cell][10] + f[cell][11];
                               e[cell]     = f[cell][12] + f[cell][13] + f[cell][14] + f[cell][15];

                               // Computing the entropy with multiplicative constant 1 and additive
                               // constant 0
                               auto p  = (gm - 1.) * (e[cell] - .5 * (std::pow(qx[cell], 2.) + std::pow(qy[cell], 2.)) / rho[cell]);
                               s[cell] = std::log(p / std::pow(rho[cell], gm));
                           });

    std::size_t min_level = mesh.min_level();
    std::size_t max_level = mesh.max_level();
    std::string filename  = fmt::format("LBM_D2Q4_3_Euler_{}_lmin-{}_lmax-{}_eps-{}_ite-{}", ext, min_level, max_level, eps, ite);
    samurai::save(filename, mesh, rho, qx, qy, e, s, f, level);
}

int main(int argc, char* argv[])
{
    samurai::initialize(argc, argv);

    CLI::App app{"Multi resolution for a D2Q4 LBM scheme for the scalar advection equation"};

    std::size_t freq_out = 1; // frequency for output in iteration
    std::size_t total_nb_ite = 100;
    int configuration        = 12;

    // Multiresolution parameters
    std::size_t min_level = 2;
    std::size_t max_level = 9;
    double mr_epsilon     = 1.e-3; // Threshold used by multiresolution
    double mr_regularity  = 2.;    // Regularity guess for multiresolution

    std::size_t nt_loadbalance = 10;

    app.add_option("--nb-ite", total_nb_ite, "number of iteration")->capture_default_str();
    app.add_option("--config", configuration, "Lax-Liu configuration")->capture_default_str();
    app.add_option("--min-level", min_level, "Minimum level of the multiresolution")->capture_default_str()->group("Multiresolution");
    app.add_option("--max-level", max_level, "Maximum level of the multiresolution")->capture_default_str()->group("Multiresolution");
    app.add_option("--nt-loadbalance", nt_loadbalance, "Maximum level of the multiresolution")->capture_default_str()->group("Multiresolution");
    app.add_option("--mr-eps", mr_epsilon, "The epsilon used by the multiresolution to adapt the mesh")->capture_default_str()
                   ->group("Multiresolution");
    app.add_option("--mr-reg", mr_regularity, "The regularity criteria used by the multiresolution to adapt the mesh")
                   ->capture_default_str()->group("Multiresolution");
    CLI11_PARSE(app, argc, argv);

    constexpr size_t dim = 2;
    using Config         = samurai::MRConfig<dim, 2, 2>;

    double lambda = 1. / 0.2; // This seems to work
    double T      = 0.25;     // 0.3;//1.2;

    double sq_rho  = 1.9;
    double sxy_rho = 1.;

    double sq_q  = 1.75;
    double sxy_q = 1.;

    double sq_e  = 1.75;
    double sxy_e = 1.;

    if (configuration == 12)
    {
        T = .25;
    }
    else
    {
        T = .3;
        // T = 0.1;
    }

    samurai::Box<double, dim> box({0, 0}, {1, 1});
    samurai::MRMesh<Config> mesh(box, min_level, max_level);

    // Initialization
    auto f = init_f(mesh, configuration, lambda); // Adaptive  scheme
    samurai::make_bc<samurai::Neumann<1>>(f, 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.);

    double dx = 1.0 / (1 << max_level);
    double dt = dx / lambda;

    std::size_t N = static_cast<std::size_t>(T / dt);

    // SFC_LoadBalancer_interval<dim, Hilbert> balancer;
    // SFC_LoadBalancer_interval<dim, Morton> balancer;
    // Load_balancing::Life balancer;
    // Void_LoadBalancer<dim> balancer;
    Diffusion_LoadBalancer_cell<dim> balancer;
    // Diffusion_LoadBalancer_interval<dim> balancer;
    // Load_balancing::Diffusion balancer;



    auto MRadaptation = samurai::make_MRAdapt(f);

    double t = 0.;
    samurai::times::timers.start("tloop");
    for (std::size_t nt = 0; nt <= N && nt < total_nb_ite; ++nt)
    {
        std::cout << fmt::format("\n\t> Iteration {}, t: {}, dt: {} ", nt, t, dt ) << std::endl;

        if (nt % nt_loadbalance == 0 && nt > 1)
        {
            samurai::times::timers.start("tloop.lb");
            balancer.load_balance(mesh, f);
            samurai::times::timers.stop("tloop.lb");
        }

        samurai::times::timers.start("tloop.MRAdaptation");
        MRadaptation(mr_epsilon, mr_regularity);
        samurai::times::timers.stop("tloop.MRAdaptation");

        samurai::times::timers.start("tloop.LBM");
        one_time_step(f, lambda, sq_rho, sxy_rho, sq_q, sxy_q, sq_e, sxy_e);
        samurai::times::timers.stop("tloop.LBM");

        samurai::times::timers.start("tloop.io");
        save_solution(f, mr_epsilon, nt, freq_out);
        samurai::times::timers.stop("tloop.io");

        t += dt;
    }
    samurai::times::timers.stop("tloop");

    samurai::finalize();
    return 0;
}