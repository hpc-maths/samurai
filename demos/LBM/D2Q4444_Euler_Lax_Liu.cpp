// Copyright 2021 SAMURAI TEAM. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include <math.h>
#include <vector>

#include <cxxopts.hpp>

#include <samurai/field.hpp>
#include <samurai/hdf5.hpp>
#include <samurai/mr/adapt.hpp>
#include <samurai/mr/mesh_with_overleaves.hpp>
#include <samurai/statistics.hpp>

#include "boundary_conditions.hpp"
#include "prediction_map_2d.hpp"

#include "utils_lbm_mr_2d.hpp"

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

template <class Config>
auto init_f(samurai::MROMesh<Config>& mesh, int config, double lambda)
{
    constexpr std::size_t nvel = 16;
    using mesh_id_t            = typename samurai::MROMesh<Config>::mesh_id_t;

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

template <class coord_index_t>
auto compute_prediction(std::size_t min_level, std::size_t max_level)
{
    coord_index_t i = 0, j = 0;
    std::vector<std::vector<prediction_map<coord_index_t>>> data(max_level - min_level + 1);

    auto rotation_of_pi_over_two = [](int alpha, int k, int h)
    {
        // Returns the rotation of (k, h) of an angle alpha * pi / 2.
        // All the operations are performed on integer, to be exact
        int cosinus = static_cast<int>(std::round(std::cos(alpha * M_PI / 2.)));
        int sinus   = static_cast<int>(std::round(std::sin(alpha * M_PI / 2.)));

        return std::pair<int, int>(cosinus * k - sinus * h, sinus * k + cosinus * h);
    };

    // Transforms the coordinates to apply the rotation
    auto tau = [](int delta, int k)
    {
        // The case in which delta = 0 is rather exceptional
        if (delta == 0)
        {
            return k;
        }
        else
        {
            auto tmp = (1 << (delta - 1));
            return static_cast<int>((k < tmp) ? (k - tmp) : (k - tmp + 1));
        }
    };

    auto tau_inverse = [](int delta, int k)
    {
        if (delta == 0)
        {
            return k;
        }
        else
        {
            auto tmp = (1 << (delta - 1));
            return static_cast<int>((k < 0) ? (k + tmp) : (k + tmp - 1));
        }
    };

    for (std::size_t k = 0; k < max_level - min_level + 1; ++k)
    {
        int size = (1 << k);
        data[k].resize(4);

        for (int alpha = 0; alpha < 4; ++alpha)
        {
            for (int l = 0; l < size; ++l)
            {
                // The reference direction from which the other ones are
                // computed is that of (1, 0)
                auto rotated_in  = rotation_of_pi_over_two(alpha, tau(k, i * size - 1), tau(k, j * size + l));
                auto rotated_out = rotation_of_pi_over_two(alpha, tau(k, (i + 1) * size - 1), tau(k, j * size + l));

                // For the cells inside the domain, we can already combine
                // entering and exiting fluxes and we have a compensation of
                // many cells.
                data[k][alpha] += (prediction(k, tau_inverse(k, rotated_in.first), tau_inverse(k, rotated_in.second))
                                   - prediction(k, tau_inverse(k, rotated_out.first), tau_inverse(k, rotated_out.second)));
            }
        }
    }
    return data;
}

template <class Field, class Func, class pred>
void one_time_step(Field& f,
                   Func&& update_bc_for_level,
                   const pred& pred_coeff,
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

    auto min_level = mesh.min_level();
    auto max_level = mesh.max_level();

    samurai::update_ghost_mr(f, std::forward<Func>(update_bc_for_level));
    samurai::update_overleaves_mr(f, std::forward<Func>(update_bc_for_level));

    Field new_f{"new_f", mesh};
    new_f.array().fill(0.);
    Field fluxes{"fluxes", mesh}; // This stored the fluxes computed at the level of the overleaves
    fluxes.array().fill(0.);
    Field advected{"advected", mesh};
    advected.array().fill(0.);

    for (std::size_t level = min_level; level <= max_level; ++level)
    {
        auto leaves = samurai::intersection(mesh[mesh_id_t::cells][level], mesh[mesh_id_t::cells][level]);

        if (level == max_level)
        { // Advection at the finest level

            leaves(
                [&](auto& interval, auto& index)
                {
                    auto k = interval; // Logical index in x
                    auto h = index[0]; // Logical index in y

                    // We enforce a bounce-back
                    for (int scheme_n = 0; scheme_n < 4; ++scheme_n)
                    { // We have 4 schemes
                        advected(0 + 4 * scheme_n, level, k, h) = f(0 + 4 * scheme_n, level, k - 1, h);
                        advected(1 + 4 * scheme_n, level, k, h) = f(1 + 4 * scheme_n, level, k, h - 1);
                        advected(2 + 4 * scheme_n, level, k, h) = f(2 + 4 * scheme_n, level, k + 1, h);
                        advected(3 + 4 * scheme_n, level, k, h) = f(3 + 4 * scheme_n, level, k, h + 1);
                    }
                });
        }
        else // Advection at the coarse levels using the overleaves
        {
            auto lev_p_1  = level + 1;
            std::size_t j = max_level - (lev_p_1);
            double coeff  = 1. / (1 << (2 * j)); // ATTENTION A LA DIMENSION 2 !!!!

            leaves.on(level + 1)([&](auto& interval, auto& index) { // This are overleaves
                auto k = interval;                                  // Logical index in x
                auto h = index[0];                                  // Logical index in y

                for (int scheme_n = 0; scheme_n < 4; ++scheme_n)
                {
                    auto shift = 4 * scheme_n;

                    for (std::size_t alpha = 0; alpha < 4; ++alpha)
                    {
                        for (auto& c : pred_coeff[j][alpha].coeff)
                        {
                            coord_index_t stencil_x, stencil_y;
                            std::tie(stencil_x, stencil_y) = c.first;

                            fluxes(alpha + shift, lev_p_1, k, h) += c.second * f(alpha + shift, lev_p_1, k + stencil_x, h + stencil_y);
                        }
                    }
                }
            });

            leaves(
                [&](auto& interval, auto& index)
                {
                    auto k = interval; // Logical index in x
                    auto h = index[0]; // Logical index in y

                    for (int alpha = 0; alpha < 16; ++alpha)
                    {
                        advected(alpha, level, k, h) = f(alpha, level, k, h)
                                                     + coeff * 0.25
                                                           * (fluxes(alpha, lev_p_1, 2 * k, 2 * h) + fluxes(alpha, lev_p_1, 2 * k + 1, 2 * h)
                                                              + fluxes(alpha, lev_p_1, 2 * k, 2 * h + 1)
                                                              + fluxes(alpha, lev_p_1, 2 * k + 1, 2 * h + 1));
                    }
                });
        }

        leaves(
            [&](auto& interval, auto& index)
            {
                auto k = interval; // Logical index in x
                auto h = index[0]; // Logical index in y

                // We compute the advected momenti
                auto m0_0 = xt::eval(advected(0, level, k, h) + advected(1, level, k, h) + advected(2, level, k, h)
                                     + advected(3, level, k, h));
                auto m0_1 = xt::eval(lambda * (advected(0, level, k, h) - advected(2, level, k, h)));
                auto m0_2 = xt::eval(lambda * (advected(1, level, k, h) - advected(3, level, k, h)));
                auto m0_3 = xt::eval(
                    lambda * lambda
                    * (advected(0, level, k, h) - advected(1, level, k, h) + advected(2, level, k, h) - advected(3, level, k, h)));

                auto m1_0 = xt::eval(advected(4, level, k, h) + advected(5, level, k, h) + advected(6, level, k, h)
                                     + advected(7, level, k, h));
                auto m1_1 = xt::eval(lambda * (advected(4, level, k, h) - advected(6, level, k, h)));
                auto m1_2 = xt::eval(lambda * (advected(5, level, k, h) - advected(7, level, k, h)));
                auto m1_3 = xt::eval(
                    lambda * lambda
                    * (advected(4, level, k, h) - advected(5, level, k, h) + advected(6, level, k, h) - advected(7, level, k, h)));

                auto m2_0 = xt::eval(advected(8, level, k, h) + advected(9, level, k, h) + advected(10, level, k, h)
                                     + advected(11, level, k, h));
                auto m2_1 = xt::eval(lambda * (advected(8, level, k, h) - advected(10, level, k, h)));
                auto m2_2 = xt::eval(lambda * (advected(9, level, k, h) - advected(11, level, k, h)));
                auto m2_3 = xt::eval(
                    lambda * lambda
                    * (advected(8, level, k, h) - advected(9, level, k, h) + advected(10, level, k, h) - advected(11, level, k, h)));

                auto m3_0 = xt::eval(advected(12, level, k, h) + advected(13, level, k, h) + advected(14, level, k, h)
                                     + advected(15, level, k, h));
                auto m3_1 = xt::eval(lambda * (advected(12, level, k, h) - advected(14, level, k, h)));
                auto m3_2 = xt::eval(lambda * (advected(13, level, k, h) - advected(15, level, k, h)));
                auto m3_3 = xt::eval(
                    lambda * lambda
                    * (advected(12, level, k, h) - advected(13, level, k, h) + advected(14, level, k, h) - advected(15, level, k, h)));

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

                new_f(0, level, k, h) = .25 * m0_0 + .5 / lambda * (m0_1) + .25 / (lambda * lambda) * m0_3;
                new_f(1, level, k, h) = .25 * m0_0 + .5 / lambda * (m0_2)-.25 / (lambda * lambda) * m0_3;
                new_f(2, level, k, h) = .25 * m0_0 - .5 / lambda * (m0_1) + .25 / (lambda * lambda) * m0_3;
                new_f(3, level, k, h) = .25 * m0_0 - .5 / lambda * (m0_2)-.25 / (lambda * lambda) * m0_3;

                new_f(4, level, k, h) = .25 * m1_0 + .5 / lambda * (m1_1) + .25 / (lambda * lambda) * m1_3;
                new_f(5, level, k, h) = .25 * m1_0 + .5 / lambda * (m1_2)-.25 / (lambda * lambda) * m1_3;
                new_f(6, level, k, h) = .25 * m1_0 - .5 / lambda * (m1_1) + .25 / (lambda * lambda) * m1_3;
                new_f(7, level, k, h) = .25 * m1_0 - .5 / lambda * (m1_2)-.25 / (lambda * lambda) * m1_3;

                new_f(8, level, k, h)  = .25 * m2_0 + .5 / lambda * (m2_1) + .25 / (lambda * lambda) * m2_3;
                new_f(9, level, k, h)  = .25 * m2_0 + .5 / lambda * (m2_2)-.25 / (lambda * lambda) * m2_3;
                new_f(10, level, k, h) = .25 * m2_0 - .5 / lambda * (m2_1) + .25 / (lambda * lambda) * m2_3;
                new_f(11, level, k, h) = .25 * m2_0 - .5 / lambda * (m2_2)-.25 / (lambda * lambda) * m2_3;

                new_f(12, level, k, h) = .25 * m3_0 + .5 / lambda * (m3_1) + .25 / (lambda * lambda) * m3_3;
                new_f(13, level, k, h) = .25 * m3_0 + .5 / lambda * (m3_2)-.25 / (lambda * lambda) * m3_3;
                new_f(14, level, k, h) = .25 * m3_0 - .5 / lambda * (m3_1) + .25 / (lambda * lambda) * m3_3;
                new_f(15, level, k, h) = .25 * m3_0 - .5 / lambda * (m3_2)-.25 / (lambda * lambda) * m3_3;
            });
    }
    std::swap(f.array(), new_f.array());
}

template <class Field>
void save_solution(Field& f, double eps, std::size_t ite, std::string ext = "")
{
    using value_t = typename Field::value_type;

    auto mesh       = f.mesh();
    using mesh_id_t = typename decltype(mesh)::mesh_id_t;

    std::size_t min_level = mesh.min_level();
    std::size_t max_level = mesh.max_level();

    std::stringstream str;
    str << "LBM_D2Q4_3_Euler_" << ext << "_lmin_" << min_level << "_lmax-" << max_level << "_eps-" << eps << "_ite-" << ite;

    auto level = samurai::make_field<std::size_t, 1>("level", mesh);
    auto rho   = samurai::make_field<value_t, 1>("rho", mesh);
    auto qx    = samurai::make_field<value_t, 1>("qx", mesh);
    auto qy    = samurai::make_field<value_t, 1>("qy", mesh);
    auto e     = samurai::make_field<value_t, 1>("e", mesh);
    auto s     = samurai::make_field<value_t, 1>("entropy", mesh);

    samurai::for_each_cell(mesh[mesh_id_t::cells],
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

    samurai::save(str.str().data(), mesh, rho, qx, qy, e, s, f, level);
}

// Attention : the number 2 as second template parameter does not mean
// that we are dealing with two fields!!!!
template <class Field, class interval_t, class ordinates_t, class ordinates_t_bis>
xt::xtensor<double, 2>
prediction_all(const Field& f,
               std::size_t level_g,
               std::size_t level,
               const interval_t& k,
               const ordinates_t& h,
               std::map<std::tuple<std::size_t, std::size_t, interval_t, ordinates_t_bis>, xt::xtensor<double, 2>>& mem_map)
{
    // That is used to employ _ with xtensor
    using namespace xt::placeholders;

    // mem_map.clear(); // To be activated if we want to avoid memoization
    auto it = mem_map.find({level_g, level, k, h});

    if (it != mem_map.end() && k.size() == (std::get<2>(it->first)).size())
    {
        return it->second;
    }
    else
    {
        auto mesh       = f.mesh();
        using mesh_id_t = typename decltype(mesh)::mesh_id_t;

        // We put only the size in x (k.size()) because in y we only have slices
        // of size 1. The second term (1) should be adapted according to the
        // number of fields that we have.
        std::vector<std::size_t> shape_x = {k.size(), 16};
        xt::xtensor<double, 2> out       = xt::empty<double>(shape_x);
        auto mask                        = mesh.exists(mesh_id_t::cells_and_ghosts,
                                level_g + level,
                                k,
                                h); // Check if we are on a leaf or a ghost (CHECK IF IT IS OK)
        xt::xtensor<double, 2> mask_all  = xt::empty<double>(shape_x);

        for (int h_field = 0; h_field < 16; ++h_field)
        {
            xt::view(mask_all, xt::all(), h_field) = mask;
        }

        if (xt::all(mask)) // Recursion finished
        {
            return xt::eval(f(0, 16, level_g + level, k, h));
        }

        // If we cannot stop here
        auto kg                    = k >> 1;
        kg.step                    = 1;
        xt::xtensor<double, 2> val = xt::empty<double>(shape_x);
        /*
        --------------------
        NW   |   N   |   NE
        --------------------
         W   | EARTH |   E
        --------------------
        SW   |   S   |   SE
        --------------------
        */

        auto earth = xt::eval(prediction_all(f, level_g, level - 1, kg, (h >> 1), mem_map));
        auto W     = xt::eval(prediction_all(f, level_g, level - 1, kg - 1, (h >> 1), mem_map));
        auto E     = xt::eval(prediction_all(f, level_g, level - 1, kg + 1, (h >> 1), mem_map));
        auto S     = xt::eval(prediction_all(f, level_g, level - 1, kg, (h >> 1) - 1, mem_map));
        auto N     = xt::eval(prediction_all(f, level_g, level - 1, kg, (h >> 1) + 1, mem_map));
        auto SW    = xt::eval(prediction_all(f, level_g, level - 1, kg - 1, (h >> 1) - 1, mem_map));
        auto SE    = xt::eval(prediction_all(f, level_g, level - 1, kg + 1, (h >> 1) - 1, mem_map));
        auto NW    = xt::eval(prediction_all(f, level_g, level - 1, kg - 1, (h >> 1) + 1, mem_map));
        auto NE    = xt::eval(prediction_all(f, level_g, level - 1, kg + 1, (h >> 1) + 1, mem_map));

        // This is to deal with odd/even indices in the x direction
        std::size_t start_even = (k.start & 1) ? 1 : 0;
        std::size_t start_odd  = (k.start & 1) ? 0 : 1;
        std::size_t end_even   = (k.end & 1) ? kg.size() : kg.size() - 1;
        std::size_t end_odd    = (k.end & 1) ? kg.size() - 1 : kg.size();

        int delta_y    = (h & 1) ? 1 : 0;
        int m1_delta_y = (delta_y == 0) ? 1 : -1; // (-1)^(delta_y)

        xt::view(val, xt::range(start_even, _, 2)) = xt::view(
            earth + 1. / 8 * (W - E) + 1. / 8 * m1_delta_y * (S - N) - 1. / 64 * m1_delta_y * (NE - NW - SE + SW),
            xt::range(start_even, _));
        xt::view(val, xt::range(start_odd, _, 2)) = xt::view(
            earth - 1. / 8 * (W - E) + 1. / 8 * m1_delta_y * (S - N) + 1. / 64 * m1_delta_y * (NE - NW - SE + SW),
            xt::range(_, end_odd));

        xt::masked_view(out, !mask_all) = xt::masked_view(val, !mask_all);

        for (int k_mask = 0, k_int = k.start; k_int < k.end; ++k_mask, ++k_int)
        {
            if (mask[k_mask])
            {
                xt::view(out, k_mask) = xt::view(f(0, 16, level_g + level, {k_int, k_int + 1}, h), 0);
            }
        }

        // It is crucial to use insert and not [] in order not to update the
        // value in case of duplicated (same key)
        mem_map.insert(std::make_pair(std::tuple<std::size_t, std::size_t, interval_t, ordinates_t_bis>{level_g, level, k, h}, out));
        return out;
    }
}

template <class Field, class FieldFull, class Func>
double compute_error(Field& f, FieldFull& f_full, Func&& update_bc_for_level)
{
    constexpr std::size_t size = Field::size;
    using value_t              = typename Field::value_type;

    auto mesh       = f.mesh();
    using mesh_id_t = typename decltype(mesh)::mesh_id_t;

    auto min_level = mesh.min_level();
    auto max_level = mesh.max_level();

    auto init_mesh = f_full.mesh();

    samurai::update_ghost_mr(f, std::forward<Func>(update_bc_for_level));

    auto f_reconstructed = samurai::make_field<value_t, size>("f_reconstructed", init_mesh);
    f_reconstructed.fill(0.);

    // For memoization
    using interval_t  = typename Field::interval_t;   // Type in X
    using ordinates_t = typename interval_t::index_t; // Type in Y
    std::map<std::tuple<std::size_t, std::size_t, interval_t, ordinates_t>, xt::xtensor<double, 2>> memoization_map;
    memoization_map.clear();

    double error = 0.;
    double norm  = 0.;
    double dx    = 1. / (1 << max_level);

    for (std::size_t level = 0; level <= max_level; ++level)
    {
        auto leaves_on_finest = samurai::intersection(mesh[mesh_id_t::cells][level], mesh[mesh_id_t::cells][level]);

        leaves_on_finest.on(max_level)(
            [&](auto& interval, auto& index)
            {
                auto k = interval;
                auto h = index[0];

                f_reconstructed(max_level, k, h) = prediction_all(f, level, max_level - level, k, h, memoization_map);

                auto rho_reconstructed = f_reconstructed(0, max_level, k, h) + f_reconstructed(1, max_level, k, h)
                                       + f_reconstructed(2, max_level, k, h) + f_reconstructed(3, max_level, k, h);
                auto rho_full = f_full(0, max_level, k, h) + f_full(1, max_level, k, h) + f_full(2, max_level, k, h)
                              + f_full(3, max_level, k, h);

                error += xt::sum(xt::abs(rho_reconstructed - rho_full))[0];
                norm += xt::sum(xt::abs(rho_full))[0];
            });
    }
    return (error / norm);
}

template <class Field, class FieldFull, class Func>
void save_reconstructed(Field& f, FieldFull& f_full, Func&& update_bc_for_level, double eps, std::size_t ite, std::string ext = "")
{
    constexpr std::size_t size = Field::size;
    using value_t              = typename Field::value_type;

    auto mesh       = f.mesh();
    using mesh_id_t = typename decltype(mesh)::mesh_id_t;

    auto min_level = mesh.min_level();
    auto max_level = mesh.max_level();

    auto init_mesh = f_full.mesh();

    samurai::update_ghost_mr(f, std::forward<Func>(update_bc_for_level));

    auto f_reconstructed = samurai::make_field<value_t, size>("f_reconstructed", init_mesh); // To reconstruct all and
                                                                                             // see entropy
    f_reconstructed.fill(0.);

    auto rho_reconstructed = samurai::make_field<value_t, 1>("rho_reconstructed", init_mesh);
    auto qx_reconstructed  = samurai::make_field<value_t, 1>("qx_reconstructed", init_mesh);
    auto qy_reconstructed  = samurai::make_field<value_t, 1>("qy_reconstructed", init_mesh);
    auto E_reconstructed   = samurai::make_field<value_t, 1>("E_reconstructed", init_mesh);
    auto s_reconstructed   = samurai::make_field<value_t, 1>("s_reconstructed", init_mesh);
    auto level_            = samurai::make_field<std::size_t, 1>("level", init_mesh);

    auto rho = samurai::make_field<value_t, 1>("rho", init_mesh);
    auto qx  = samurai::make_field<value_t, 1>("qx", init_mesh);
    auto qy  = samurai::make_field<value_t, 1>("qy", init_mesh);
    auto E   = samurai::make_field<value_t, 1>("E", init_mesh);
    auto s   = samurai::make_field<value_t, 1>("s", init_mesh);

    // For memoization
    using interval_t  = typename Field::interval_t;   // Type in X
    using ordinates_t = typename interval_t::index_t; // Type in Y
    std::map<std::tuple<std::size_t, std::size_t, interval_t, ordinates_t>, xt::xtensor<double, 2>> memoization_map;

    memoization_map.clear();

    for (std::size_t level = 0; level <= max_level; ++level)
    {
        auto number_leaves = mesh.nb_cells(level, mesh_id_t::cells);

        auto leaves_on_finest = samurai::intersection(mesh[mesh_id_t::cells][level], mesh[mesh_id_t::cells][level]);

        leaves_on_finest.on(max_level)(
            [&](auto& interval, auto& index)
            {
                auto k = interval;
                auto h = index[0];

                f_reconstructed(max_level, k, h) = prediction_all(f, level, max_level - level, k, h, memoization_map);

                level_(max_level, k, h) = level;

                rho_reconstructed(max_level, k, h) = f_reconstructed(0, max_level, k, h) + f_reconstructed(1, max_level, k, h)
                                                   + f_reconstructed(2, max_level, k, h) + f_reconstructed(3, max_level, k, h);

                qx_reconstructed(max_level, k, h) = f_reconstructed(4, max_level, k, h) + f_reconstructed(5, max_level, k, h)
                                                  + f_reconstructed(6, max_level, k, h) + f_reconstructed(7, max_level, k, h);

                qy_reconstructed(max_level, k, h) = f_reconstructed(8, max_level, k, h) + f_reconstructed(9, max_level, k, h)
                                                  + f_reconstructed(10, max_level, k, h) + f_reconstructed(11, max_level, k, h);

                E_reconstructed(max_level, k, h) = f_reconstructed(12, max_level, k, h) + f_reconstructed(13, max_level, k, h)
                                                 + f_reconstructed(14, max_level, k, h) + f_reconstructed(15, max_level, k, h);

                s_reconstructed(max_level, k, h) = xt::log(
                    ((gm - 1.)
                     * (E_reconstructed(max_level, k, h)
                        - .5 * (xt::pow(qx_reconstructed(max_level, k, h), 2.) + xt::pow(qy_reconstructed(max_level, k, h), 2.))
                              / rho_reconstructed(max_level, k, h)))
                    / xt::pow(rho_reconstructed(max_level, k, h), gm));

                rho(max_level, k, h) = f_full(0, max_level, k, h) + f_full(1, max_level, k, h) + f_full(2, max_level, k, h)
                                     + f_full(3, max_level, k, h);

                qx(max_level, k, h) = f_full(4, max_level, k, h) + f_full(5, max_level, k, h) + f_full(6, max_level, k, h)
                                    + f_full(7, max_level, k, h);

                qy(max_level, k, h) = f_full(8, max_level, k, h) + f_full(9, max_level, k, h) + f_full(10, max_level, k, h)
                                    + f_full(11, max_level, k, h);

                E(max_level, k, h) = f_full(12, max_level, k, h) + f_full(13, max_level, k, h) + f_full(14, max_level, k, h)
                                   + f_full(15, max_level, k, h);

                s(max_level, k, h) = xt::log(
                    ((gm - 1.)
                     * (E(max_level, k, h) - .5 * (xt::pow(qx(max_level, k, h), 2.) + xt::pow(qy(max_level, k, h), 2.)) / rho(max_level, k, h)))
                    / xt::pow(rho(max_level, k, h), gm));
            });
    }

    std::stringstream str;
    str << "LBM_D2Q4_3_Euler_Reconstruction_" << ext << "_lmin_" << min_level << "_lmax-" << max_level << "_eps-" << eps << "_ite-" << ite;

    samurai::save(str.str().data(),
                  init_mesh,
                  rho_reconstructed,
                  qx_reconstructed,
                  qy_reconstructed,
                  E_reconstructed,
                  s_reconstructed,
                  rho,
                  qx,
                  qy,
                  E,
                  s,
                  level_);
}

int main(int argc, char* argv[])
{
    cxxopts::Options options("lbm_d2q4_3_Euler",
                             "Multi resolution for a D2Q4 LBM scheme for the "
                             "scalar advection equation");

    options.add_options()("min_level", "minimum level", cxxopts::value<std::size_t>()->default_value("2"))(
        "max_level",
        "maximum level",
        cxxopts::value<std::size_t>()->default_value("7"))("epsilon", "maximum level", cxxopts::value<double>()->default_value("0.0001"))(
        "ite",
        "number of iteration",
        cxxopts::value<std::size_t>()->default_value("100"))("reg", "regularity", cxxopts::value<double>()->default_value("0."))(
        "config",
        "Lax-Liu configuration",
        cxxopts::value<int>()->default_value("12"))("h, help", "Help");

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
            using Config         = samurai::MROConfig<dim, 2>;

            std::size_t min_level    = result["min_level"].as<std::size_t>();
            std::size_t max_level    = result["max_level"].as<std::size_t>();
            std::size_t total_nb_ite = result["ite"].as<std::size_t>();
            double eps               = result["epsilon"].as<double>();
            double regularity        = result["reg"].as<double>();
            int configuration        = result["config"].as<int>();

            // double lambda = 1./0.3; //4.0;
            // double lambda = 1./0.2499; //4.0;
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

            // // This were the old test case (version 3)
            // double sq = 1.75;
            // double sxy = 2.;
            // if (configuration == 12)    {
            //     sxy = 1.5;
            //     T = 0.25;
            // }
            // else    {
            //     sxy = 0.5;
            //     T = 0.3;
            // }

            samurai::Box<double, dim> box({0, 0}, {1, 1});
            samurai::MROMesh<Config> mesh(box, min_level, max_level);
            using mesh_id_t = typename samurai::MROMesh<Config>::mesh_id_t;
            samurai::MROMesh<Config> mesh_ref{box, max_level, max_level};

            using coord_index_t = typename samurai::MROMesh<Config>::coord_index_t;
            auto pred_coeff     = compute_prediction<coord_index_t>(min_level, max_level);

            // Initialization
            auto f     = init_f(mesh, configuration, lambda);     // Adaptive  scheme
            auto f_ref = init_f(mesh_ref, configuration, lambda); // Reference scheme

            double dx = 1.0 / (1 << max_level);
            double dt = dx / lambda;

            std::size_t N = static_cast<std::size_t>(T / dt);

            std::string dirname("./LaxLiu/");
            std::string suffix("_Config_" + std::to_string(configuration) + "_min_" + std::to_string(min_level) + "_max_"
                               + std::to_string(max_level) + "_eps_" + std::to_string(eps));

            // std::ofstream stream_number_leaves;
            // stream_number_leaves.open
            // (dirname+"number_leaves"+suffix+".dat");

            // std::ofstream stream_number_cells;
            // stream_number_cells.open (dirname+"number_cells"+suffix+".dat");

            // std::ofstream stream_time_scheme_ref;
            // stream_time_scheme_ref.open
            // (dirname+"time_scheme_ref"+suffix+".dat");

            // std::ofstream stream_number_leaves_ref;
            // stream_number_leaves_ref.open
            // (dirname+"number_leaves_ref"+suffix+".dat");

            // std::ofstream stream_number_cells_ref;
            // stream_number_cells_ref.open
            // (dirname+"number_cells_ref"+suffix+".dat");

            int howoften = 1; // How often is the solution saved ?

            auto update_bc_for_level = [](auto& field, std::size_t level)
            {
                update_bc_D2Q4_3_Euler_constant_extension(field, level);
            };

            auto MRadaptation = samurai::make_MRAdapt(f, update_bc_for_level);

            for (std::size_t nb_ite = 0; nb_ite <= N; ++nb_ite)
            {
                std::cout << std::endl << "   Iteration number = " << nb_ite << std::endl;

                if (max_level > min_level)
                {
                    MRadaptation(eps, regularity);
                }

                if (nb_ite == N)
                {
                    auto error_density = compute_error(f, f_ref, update_bc_for_level);
                    std::cout << std::endl << "#### Epsilon = " << eps << "   error = " << error_density << std::endl;
                    save_solution(f, eps, nb_ite, std::string("final_"));
                    save_reconstructed(f, f_ref, update_bc_for_level, eps, nb_ite);
                }

                // if (nb_ite % howoften == 0)    {
                //     save_solution(f    , eps, nb_ite/howoften,
                //     std::string("Config_")+std::to_string(configuration)); //
                //     Before applying the scheme
                // }

                one_time_step(f, update_bc_for_level, pred_coeff, lambda, sq_rho, sxy_rho, sq_q, sxy_q, sq_e, sxy_e);
                one_time_step(f_ref, update_bc_for_level, pred_coeff, lambda, sq_rho, sxy_rho, sq_q, sxy_q, sq_e, sxy_e);

                auto number_leaves = mesh.nb_cells(mesh_id_t::cells);
                auto number_cells  = mesh.nb_cells();

                samurai::statistics("D2Q4444_Euler_Lax_Liu", mesh);
                // stream_number_leaves<<number_leaves<<std::endl;
                // stream_number_cells<<number_cells<<std::endl;

                // stream_number_leaves_ref<<mesh_ref.nb_cells(mesh_id_t::cells)<<std::endl;
                // stream_number_cells_ref<<mesh_ref.nb_cells()<<std::endl;
            }
            // stream_number_leaves.close();
            // stream_number_cells.close();

            // stream_number_leaves_ref.close();
            // stream_number_cells_ref.close();
        }
    }
    catch (const cxxopts::OptionException& e)
    {
        std::cout << options.help() << "\n";
    }
    return 0;
}
