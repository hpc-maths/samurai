// Copyright 2021 SAMURAI TEAM. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include <math.h>
#include <vector>

#include <cxxopts.hpp>

#include <samurai/field.hpp>
#include <samurai/hdf5.hpp>
#include <samurai/statistics.hpp>
#include <samurai/uniform_mesh.hpp>

#include "boundary_conditions.hpp"

/// Timer used in tic & toc
auto tic_timer = std::chrono::high_resolution_clock::now();

/// Launching the timer
void tic()
{
    tic_timer = std::chrono::high_resolution_clock::now();
}

/// Stopping the timer and returning the duration in seconds
double toc()
{
    const auto toc_timer                          = std::chrono::high_resolution_clock::now();
    const std::chrono::duration<double> time_span = toc_timer - tic_timer;
    return time_span.count();
}

double gm = 1.4; // Gas constant

template <class Mesh>
auto init_f(Mesh& mesh, int config, double lambda)
{
    constexpr std::size_t nvel = 16;
    using mesh_id_t            = typename Mesh::mesh_id_t;

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

template <class Field, class Func>
void one_time_step(Field& f,
                   Field& new_f,
                   Func&& update_bc_for_level,
                   const double lambda,
                   const double sq_rho,
                   const double sxy_rho,
                   const double sq_q,
                   const double sxy_q,
                   const double sq_e,
                   const double sxy_e)
{
    constexpr std::size_t nvel = Field::size;
    using coord_index_t        = typename Field::interval_t::coord_index_t;

    auto mesh       = f.mesh();
    using mesh_id_t = typename decltype(mesh)::mesh_id_t;

    update_bc_for_level(f);

    samurai::for_each_interval(mesh[mesh_id_t::cells],
                               [&](std::size_t level, const auto& i, const auto& index)
                               {
                                   auto j = index[0]; // Logical index in y

                                   // We enforce a bounce-back
                                   for (int scheme_n = 0; scheme_n < 4; ++scheme_n)
                                   { // We have 4 schemes
                                       new_f(0 + 4 * scheme_n, level, i, j) = f(0 + 4 * scheme_n, level, i - 1, j);
                                       new_f(1 + 4 * scheme_n, level, i, j) = f(1 + 4 * scheme_n, level, i, j - 1);
                                       new_f(2 + 4 * scheme_n, level, i, j) = f(2 + 4 * scheme_n, level, i + 1, j);
                                       new_f(3 + 4 * scheme_n, level, i, j) = f(3 + 4 * scheme_n, level, i, j + 1);
                                   }
                               });

    samurai::for_each_interval(
        mesh[mesh_id_t::cells],
        [&](std::size_t level, const auto& i, const auto& index)
        {
            auto j = index[0]; // Logical index in y

            // We compute the advected momenti
            auto m0_0 = xt::eval(new_f(0, level, i, j) + new_f(1, level, i, j) + new_f(2, level, i, j) + new_f(3, level, i, j));
            auto m0_1 = xt::eval(lambda * (new_f(0, level, i, j) - new_f(2, level, i, j)));
            auto m0_2 = xt::eval(lambda * (new_f(1, level, i, j) - new_f(3, level, i, j)));
            auto m0_3 = xt::eval(lambda * lambda
                                 * (new_f(0, level, i, j) - new_f(1, level, i, j) + new_f(2, level, i, j) - new_f(3, level, i, j)));

            auto m1_0 = xt::eval(new_f(4, level, i, j) + new_f(5, level, i, j) + new_f(6, level, i, j) + new_f(7, level, i, j));
            auto m1_1 = xt::eval(lambda * (new_f(4, level, i, j) - new_f(6, level, i, j)));
            auto m1_2 = xt::eval(lambda * (new_f(5, level, i, j) - new_f(7, level, i, j)));
            auto m1_3 = xt::eval(lambda * lambda
                                 * (new_f(4, level, i, j) - new_f(5, level, i, j) + new_f(6, level, i, j) - new_f(7, level, i, j)));

            auto m2_0 = xt::eval(new_f(8, level, i, j) + new_f(9, level, i, j) + new_f(10, level, i, j) + new_f(11, level, i, j));
            auto m2_1 = xt::eval(lambda * (new_f(8, level, i, j) - new_f(10, level, i, j)));
            auto m2_2 = xt::eval(lambda * (new_f(9, level, i, j) - new_f(11, level, i, j)));
            auto m2_3 = xt::eval(lambda * lambda
                                 * (new_f(8, level, i, j) - new_f(9, level, i, j) + new_f(10, level, i, j) - new_f(11, level, i, j)));

            auto m3_0 = xt::eval(new_f(12, level, i, j) + new_f(13, level, i, j) + new_f(14, level, i, j) + new_f(15, level, i, j));
            auto m3_1 = xt::eval(lambda * (new_f(12, level, i, j) - new_f(14, level, i, j)));
            auto m3_2 = xt::eval(lambda * (new_f(13, level, i, j) - new_f(15, level, i, j)));
            auto m3_3 = xt::eval(lambda * lambda
                                 * (new_f(12, level, i, j) - new_f(13, level, i, j) + new_f(14, level, i, j) - new_f(15, level, i, j)));

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

            f(0, level, i, j) = .25 * m0_0 + .5 / lambda * (m0_1) + .25 / (lambda * lambda) * m0_3;
            f(1, level, i, j) = .25 * m0_0 + .5 / lambda * (m0_2)-.25 / (lambda * lambda) * m0_3;
            f(2, level, i, j) = .25 * m0_0 - .5 / lambda * (m0_1) + .25 / (lambda * lambda) * m0_3;
            f(3, level, i, j) = .25 * m0_0 - .5 / lambda * (m0_2)-.25 / (lambda * lambda) * m0_3;

            f(4, level, i, j) = .25 * m1_0 + .5 / lambda * (m1_1) + .25 / (lambda * lambda) * m1_3;
            f(5, level, i, j) = .25 * m1_0 + .5 / lambda * (m1_2)-.25 / (lambda * lambda) * m1_3;
            f(6, level, i, j) = .25 * m1_0 - .5 / lambda * (m1_1) + .25 / (lambda * lambda) * m1_3;
            f(7, level, i, j) = .25 * m1_0 - .5 / lambda * (m1_2)-.25 / (lambda * lambda) * m1_3;

            f(8, level, i, j)  = .25 * m2_0 + .5 / lambda * (m2_1) + .25 / (lambda * lambda) * m2_3;
            f(9, level, i, j)  = .25 * m2_0 + .5 / lambda * (m2_2)-.25 / (lambda * lambda) * m2_3;
            f(10, level, i, j) = .25 * m2_0 - .5 / lambda * (m2_1) + .25 / (lambda * lambda) * m2_3;
            f(11, level, i, j) = .25 * m2_0 - .5 / lambda * (m2_2)-.25 / (lambda * lambda) * m2_3;

            f(12, level, i, j) = .25 * m3_0 + .5 / lambda * (m3_1) + .25 / (lambda * lambda) * m3_3;
            f(13, level, i, j) = .25 * m3_0 + .5 / lambda * (m3_2)-.25 / (lambda * lambda) * m3_3;
            f(14, level, i, j) = .25 * m3_0 - .5 / lambda * (m3_1) + .25 / (lambda * lambda) * m3_3;
            f(15, level, i, j) = .25 * m3_0 - .5 / lambda * (m3_2)-.25 / (lambda * lambda) * m3_3;
        });
    // std::swap(f.array(), new_f.array());
}

template <class Field>
void save_solution(Field& f, std::size_t ite, std::string ext = "")
{
    using value_t = typename Field::value_type;

    auto mesh       = f.mesh();
    using mesh_id_t = typename decltype(mesh)::mesh_id_t;

    std::stringstream str;
    str << "LBM_D2Q4_3_Euler_Uniform_ite-" << ite;

    auto rho = samurai::make_field<value_t, 1>("rho", mesh);
    auto qx  = samurai::make_field<value_t, 1>("qx", mesh);
    auto qy  = samurai::make_field<value_t, 1>("qy", mesh);
    auto e   = samurai::make_field<value_t, 1>("e", mesh);
    auto s   = samurai::make_field<value_t, 1>("entropy", mesh);

    samurai::for_each_cell(mesh[mesh_id_t::cells],
                           [&](auto& cell)
                           {
                               rho[cell] = f[cell][0] + f[cell][1] + f[cell][2] + f[cell][3];
                               qx[cell]  = f[cell][4] + f[cell][5] + f[cell][6] + f[cell][7];
                               qy[cell]  = f[cell][8] + f[cell][9] + f[cell][10] + f[cell][11];
                               e[cell]   = f[cell][12] + f[cell][13] + f[cell][14] + f[cell][15];

                               // Computing the entropy with multiplicative constant 1 and additive
                               // constant 0
                               auto p  = (gm - 1.) * (e[cell] - .5 * (std::pow(qx[cell], 2.) + std::pow(qy[cell], 2.)) / rho[cell]);
                               s[cell] = std::log(p / std::pow(rho[cell], gm));
                           });

    samurai::save(str.str().data(), mesh, rho, qx, qy, e, s, f);
}

int main(int argc, char* argv[])
{
    cxxopts::Options options("lbm_d2q4_3_Euler",
                             "Multi resolution for a D2Q4 LBM scheme for the "
                             "scalar advection equation");

    options.add_options()("level", "start level", cxxopts::value<std::size_t>()->default_value("8"))(
        "ite",
        "number of iteration",
        cxxopts::value<std::size_t>()->default_value(
            "100"))("config", "Lax-Liu configuration", cxxopts::value<int>()->default_value("12"))("h, help", "Help");

    try
    {
        auto result = options.parse(argc, argv);

        if (result.count("help"))
        {
            std::cout << options.help() << "\n";
        }
        else
        {
            constexpr size_t dim = 2;
            using Config         = samurai::UniformConfig<dim, 2>;

            std::size_t level        = result["level"].as<std::size_t>();
            std::size_t total_nb_ite = result["ite"].as<std::size_t>();
            int configuration        = result["config"].as<int>();

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
            }

            samurai::Box<double, dim> box({0, 0}, {1, 1});
            samurai::UniformMesh<Config> mesh(box, level);

            // Initialization
            auto f    = init_f(mesh, configuration, lambda); // Adaptive  scheme
            auto fnp1 = init_f(mesh, configuration, lambda); // Adaptive  scheme

            double dx = samurai::cell_length(level);
            double dt = dx / lambda;

            std::size_t N = static_cast<std::size_t>(T / dt);

            std::string dirname("./LaxLiu/");
            std::string suffix("_Config_" + std::to_string(configuration) + "_level_" + std::to_string(level));

            int howoften = 1; // How often is the solution saved ?

            auto update_bc_for_level = [](auto& field)
            {
                update_bc_D2Q4_3_Euler_constant_extension_uniform(field);
            };

            save_solution(f, 0, std::string("init_"));

            tic();
            for (std::size_t nb_ite = 0; nb_ite <= N; ++nb_ite)
            {
                // std::cout<<std::endl<<"   Iteration number =
                // "<<nb_ite<<std::endl;

                if (nb_ite == N)
                {
                    save_solution(f, nb_ite, std::string("final_"));
                }

                one_time_step(f, fnp1, update_bc_for_level, lambda, sq_rho, sxy_rho, sq_q, sxy_q, sq_e, sxy_e);

                // samurai::statistics("D2Q4444_Euler_Lax_Liu", mesh);
            }
            auto duration = toc();
            std::cout << "nb_ite: " << N << " execution time: " << duration
                      << " MLUPS: " << N * (1 << level) * (1 << level) / duration / 1e6 << std::endl;
        }
    }
    catch (const cxxopts::OptionException& e)
    {
        std::cout << options.help() << "\n";
    }
    return 0;
}
