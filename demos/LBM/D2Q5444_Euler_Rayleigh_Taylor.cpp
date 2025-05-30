// Copyright 2018-2025 the samurai's authors
// SPDX-License-Identifier:  BSD-3-Clause

#include <math.h>
#include <vector>

#include <cxxopts.hpp>

#include <samurai/field.hpp>
#include <samurai/io/hdf5.hpp>
#include <samurai/mr/adapt.hpp>
#include <samurai/mr/mesh_with_overleaves.hpp>
#include <samurai/samurai.hpp>

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

template <class Config>
auto init_f(samurai::MROMesh<Config>& mesh,
            const double lambda       = 5.,
            const double gas_constant = 1.4,
            const double rho_up       = 2.,
            const double rho_down     = 1.,
            const double p_up         = 1.,
            const double grav         = 2.)
{
    constexpr std::size_t nvel = 17;
    using mesh_id_t            = typename samurai::MROMesh<Config>::mesh_id_t;

    auto f = samurai::make_vector_field<double, nvel>("f", mesh);
    f.fill(0);

    samurai::for_each_cell(
        mesh[mesh_id_t::cells],
        [&](auto& cell)
        {
            auto center = cell.center();
            auto x      = center[0];
            auto y      = center[1];

            double rho = (y < 0.5 + .01 * std::cos(4. * M_PI * x)) ? rho_down : rho_up;
            double qx  = 0.0;
            double qy  = 0.0;

            double p = p_up;
            if (y < 0.5 + .01 * std::cos(8. * M_PI * x))
            {
                p += (1. - (0.5 + .01 * std::cos(4. * M_PI * x))) * grav * rho_up
                   + (0.5 + .01 * std::cos(4. * M_PI * x) - y) * grav * rho_down;
            }
            else
            {
                p += (1. - y) * grav * rho_up;
            }

            double e = p / (gas_constant - 1.) + .5 * (qx * qx + qy * qy) / rho;

            // Conserved momenti
            double m0_0 = rho;
            double m1_0 = qx;
            double m2_0 = qy;
            double m3_0 = e;

            // Non conserved at equilibrium
            double m0_1 = m1_0;
            double m0_2 = m2_0;
            double m0_3 = (m1_0 * m1_0 + m2_0 * m2_0) / m0_0;
            double m0_4 = 0.0;

            double m1_1 = (3. / 2. - gas_constant / 2.) * (m1_0 * m1_0) / (m0_0) + (1. / 2. - gas_constant / 2.) * (m2_0 * m2_0) / (m0_0)
                        + (gas_constant - 1.) * m3_0;
            double m1_2 = m1_0 * m2_0 / m0_0;
            double m1_3 = 0.0;

            double m2_1 = m1_0 * m2_0 / m0_0;

            double m2_2 = (3. / 2. - gas_constant / 2.) * (m2_0 * m2_0) / (m0_0) + (1. / 2. - gas_constant / 2.) * (m1_0 * m1_0) / (m0_0)
                        + (gas_constant - 1.) * m3_0;
            double m2_3 = 0.0;

            double m3_1 = gas_constant * (m1_0 * m3_0) / (m0_0) - (gas_constant / 2. - 1. / 2.) * (m1_0 * m1_0 * m1_0) / (m0_0 * m0_0)
                        - (gas_constant / 2. - 1. / 2.) * (m1_0 * m2_0 * m2_0) / (m0_0 * m0_0);
            double m3_2 = gas_constant * (m2_0 * m3_0) / (m0_0) - (gas_constant / 2. - 1. / 2.) * (m2_0 * m2_0 * m2_0) / (m0_0 * m0_0)
                        - (gas_constant / 2. - 1. / 2.) * (m2_0 * m1_0 * m1_0) / (m0_0 * m0_0);
            double m3_3 = 0.0;

            // We come back to the distributions
            f[cell][0] = 21. / 25 * m0_0 - 1. / (5 * lambda * lambda) * m0_3;
            f[cell][1] = 1. / 25 * m0_0 + .5 / lambda * m0_1 + 1. / (20 * lambda * lambda) * m0_3 + .25 / (lambda * lambda) * m0_4;
            f[cell][2] = 1. / 25 * m0_0 + .5 / lambda * m0_2 + 1. / (20 * lambda * lambda) * m0_3 - .25 / (lambda * lambda) * m0_4;
            f[cell][3] = 1. / 25 * m0_0 - .5 / lambda * m0_1 + 1. / (20 * lambda * lambda) * m0_3 + .25 / (lambda * lambda) * m0_4;
            f[cell][4] = 1. / 25 * m0_0 - .5 / lambda * m0_2 + 1. / (20 * lambda * lambda) * m0_3 - .25 / (lambda * lambda) * m0_4;

            f[cell][5] = .25 * m1_0 + .5 / lambda * (m1_1) + .25 / (lambda * lambda) * m1_3;
            f[cell][6] = .25 * m1_0 + .5 / lambda * (m1_2)-.25 / (lambda * lambda) * m1_3;
            f[cell][7] = .25 * m1_0 - .5 / lambda * (m1_1) + .25 / (lambda * lambda) * m1_3;
            f[cell][8] = .25 * m1_0 - .5 / lambda * (m1_2)-.25 / (lambda * lambda) * m1_3;

            f[cell][9]  = .25 * m2_0 + .5 / lambda * (m2_1) + .25 / (lambda * lambda) * m2_3;
            f[cell][10] = .25 * m2_0 + .5 / lambda * (m2_2)-.25 / (lambda * lambda) * m2_3;
            f[cell][11] = .25 * m2_0 - .5 / lambda * (m2_1) + .25 / (lambda * lambda) * m2_3;
            f[cell][12] = .25 * m2_0 - .5 / lambda * (m2_2)-.25 / (lambda * lambda) * m2_3;

            f[cell][13] = .25 * m3_0 + .5 / lambda * (m3_1) + .25 / (lambda * lambda) * m3_3;
            f[cell][14] = .25 * m3_0 + .5 / lambda * (m3_2)-.25 / (lambda * lambda) * m3_3;
            f[cell][15] = .25 * m3_0 - .5 / lambda * (m3_1) + .25 / (lambda * lambda) * m3_3;
            f[cell][16] = .25 * m3_0 - .5 / lambda * (m3_2)-.25 / (lambda * lambda) * m3_3;
        });
    return f;
}

std::array<double, 5>
vertical_boundary_condition(const bool up = true, const double gas_constant = 1.4, const double grav = 2., double lambda = 5.)
{
    std::array<double, 5> to_return;

    double rho = 0.0;
    double qx  = 0.0;
    double qy  = 0.0;

    if (up)
    {
        rho = 2.;
        // p = 1. - grav * 2. * dx;
    }
    else
    {
        rho = 1.;
        // p = 1. + .5 * grav * 2. + (.5+dx) * grav * 1.;
    }
    // double e = p / (gas_constant - 1.) + .5 * (qx*qx + qy*qy) / rho;

    double m0_0 = rho;
    double m0_1 = qx;
    double m0_2 = qy;
    double m0_3 = (qx * qx + qy * qy) / rho;
    double m0_4 = 0.0;

    to_return[0] = 21. / 25 * m0_0 - 1. / (5 * lambda * lambda) * m0_3;
    to_return[1] = 1. / 25 * m0_0 + .5 / lambda * m0_1 + 1. / (20 * lambda * lambda) * m0_3 + .25 / (lambda * lambda) * m0_4;
    to_return[2] = 1. / 25 * m0_0 + .5 / lambda * m0_2 + 1. / (20 * lambda * lambda) * m0_3 - .25 / (lambda * lambda) * m0_4;
    to_return[3] = 1. / 25 * m0_0 - .5 / lambda * m0_1 + 1. / (20 * lambda * lambda) * m0_3 + .25 / (lambda * lambda) * m0_4;
    to_return[4] = 1. / 25 * m0_0 - .5 / lambda * m0_2 + 1. / (20 * lambda * lambda) * m0_3 - .25 / (lambda * lambda) * m0_4;

    return to_return;
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
        data[k].resize(8);

        for (int alpha = 0; alpha < 4; ++alpha)
        {
            for (int l = 0; l < size; ++l)
            {
                // The reference direction from which the other ones are
                // computed is that of (1, 0)
                auto rotated_in  = rotation_of_pi_over_two(alpha, tau(k, i * size - 1), tau(k, j * size + l));
                auto rotated_out = rotation_of_pi_over_two(alpha, tau(k, (i + 1) * size - 1), tau(k, j * size + l));

                data[k][0 + 2 * alpha] += prediction(k, tau_inverse(k, rotated_in.first), tau_inverse(k, rotated_in.second));
                data[k][1 + 2 * alpha] += prediction(k, tau_inverse(k, rotated_out.first), tau_inverse(k, rotated_out.second));
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
                   const double gas_constant = 1.4,
                   const double s_rho_x      = 1.75,
                   const double s_rho_xy     = 1.,
                   const double s_u_x        = 1.5,
                   const double s_u_xy       = 1.,
                   const double s_p_x        = 1.5,
                   const double s_p_xy       = 1.,
                   const double grav         = 2.)
{
    constexpr std::size_t nvel = Field::n_comp;
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

    auto up_bc = vertical_boundary_condition(true);
    // auto low_bc = vertical_boundary_condition(false);

    for (std::size_t level = min_level; level <= max_level; ++level)
    {
        auto leaves = samurai::intersection(mesh[mesh_id_t::cells][level], mesh[mesh_id_t::cells][level]);

        if (level == max_level)
        { // Advection at the finest level

            auto leaves_east = get_adjacent_boundary_east(mesh, level, mesh_id_t::cells);
            leaves_east(
                [&](auto& interval, auto& index)
                {
                    auto k = interval; // Logical index in x
                    auto h = index[0]; // Logical index in y

                    // advected(0, level, k, h) =  f(0, level, k    , h    );
                    // advected(1, level, k, h) =  f(1, level, k - 1, h    );
                    // advected(2, level, k, h) =  f(2, level, k,     h - 1);
                    // advected(3, level, k, h) =  f(3, level, k + 1, h    );
                    // advected(4, level, k, h) =  f(4, level, k,     h + 1);

                    // D2Q5
                    advected(0, level, k, h) = f(0, level, k, h);
                    advected(1, level, k, h) = f(1, level, k - 1, h);
                    advected(2, level, k, h) = f(2, level, k, h - 1);
                    advected(3, level, k, h) = f(3, level, k, h); // Neumann
                    advected(4, level, k, h) = f(4, level, k, h + 1);

                    // D2Q4
                    for (int scheme_n = 1; scheme_n < 4; ++scheme_n)
                    {
                        int sign                                    = (scheme_n == 1) ? -1 : 1;
                        advected(0 + 4 * scheme_n + 1, level, k, h) = f(0 + 4 * scheme_n + 1, level, k - 1, h);
                        advected(1 + 4 * scheme_n + 1, level, k, h) = f(1 + 4 * scheme_n + 1, level, k, h - 1);
                        advected(2 + 4 * scheme_n + 1, level, k, h) = sign * f(0 + 4 * scheme_n + 1, level, k, h);
                        advected(3 + 4 * scheme_n + 1, level, k, h) = f(3 + 4 * scheme_n + 1, level, k, h + 1);
                    }
                });

            auto leaves_west = get_adjacent_boundary_west(mesh, level, mesh_id_t::cells);
            leaves_west(
                [&](auto& interval, auto& index)
                {
                    auto k = interval; // Logical index in x
                    auto h = index[0]; // Logical index in y

                    advected(0, level, k, h) = f(0, level, k, h);
                    advected(1, level, k, h) = f(1, level, k, h);
                    advected(2, level, k, h) = f(2, level, k, h - 1);
                    advected(3, level, k, h) = f(3, level, k + 1, h);
                    advected(4, level, k, h) = f(4, level, k, h + 1);

                    for (int scheme_n = 1; scheme_n < 4; ++scheme_n)
                    {
                        int sign                                    = (scheme_n == 1) ? -1 : 1;
                        advected(0 + 4 * scheme_n + 1, level, k, h) = sign * f(2 + 4 * scheme_n + 1, level, k, h);
                        advected(1 + 4 * scheme_n + 1, level, k, h) = f(1 + 4 * scheme_n + 1, level, k, h - 1);
                        advected(2 + 4 * scheme_n + 1, level, k, h) = f(2 + 4 * scheme_n + 1, level, k + 1, h);
                        advected(3 + 4 * scheme_n + 1, level, k, h) = f(3 + 4 * scheme_n + 1, level, k, h + 1);
                    }
                });

            auto leaves_south = get_adjacent_boundary_south(mesh, level, mesh_id_t::cells);
            leaves_south(
                [&](auto& interval, auto& index)
                {
                    auto k = interval; // Logical index in x
                    auto h = index[0]; // Logical index in y

                    advected(0, level, k, h) = f(0, level, k, h);
                    advected(1, level, k, h) = f(1, level, k - 1, h);
                    advected(2, level, k, h) = f(2, level, k, h);
                    advected(3, level, k, h) = f(3, level, k + 1, h);
                    advected(4, level, k, h) = f(4, level, k, h + 1);

                    for (int scheme_n = 1; scheme_n < 4; ++scheme_n)
                    {
                        int sign                                    = (scheme_n == 2) ? -1 : 1;
                        advected(0 + 4 * scheme_n + 1, level, k, h) = f(0 + 4 * scheme_n + 1, level, k - 1, h);
                        advected(1 + 4 * scheme_n + 1, level, k, h) = sign * f(3 + 4 * scheme_n + 1, level, k, h);
                        advected(2 + 4 * scheme_n + 1, level, k, h) = f(2 + 4 * scheme_n + 1, level, k + 1, h);
                        advected(3 + 4 * scheme_n + 1, level, k, h) = f(3 + 4 * scheme_n + 1, level, k, h + 1);
                    }
                });

            auto leaves_north = get_adjacent_boundary_north(mesh, level, mesh_id_t::cells);
            leaves_north(
                [&](auto& interval, auto& index)
                {
                    auto k = interval; // Logical index in x
                    auto h = index[0]; // Logical index in y

                    advected(0, level, k, h) = f(0, level, k, h);
                    advected(1, level, k, h) = f(1, level, k - 1, h);
                    advected(2, level, k, h) = f(2, level, k, h - 1);
                    advected(3, level, k, h) = f(3, level, k + 1, h);
                    advected(4, level, k, h) = f(4, level, k, h);
                    // advected(4, level, k, h) =  up_bc[4];

                    for (int scheme_n = 1; scheme_n < 4; ++scheme_n)
                    {
                        int sign                                    = (scheme_n == 2) ? -1 : 1;
                        advected(0 + 4 * scheme_n + 1, level, k, h) = f(0 + 4 * scheme_n + 1, level, k - 1, h);
                        advected(1 + 4 * scheme_n + 1, level, k, h) = f(1 + 4 * scheme_n + 1, level, k, h - 1);
                        advected(2 + 4 * scheme_n + 1, level, k, h) = f(2 + 4 * scheme_n + 1, level, k + 1, h);
                        advected(3 + 4 * scheme_n + 1, level, k, h) = sign * f(1 + 4 * scheme_n + 1, level, k, h);
                    }
                });

            auto leaves_north_east = get_adjacent_boundary_northeast(mesh, level, mesh_id_t::cells);
            leaves_north_east(
                [&](auto& interval, auto& index)
                {
                    auto k = interval; // Logical index in x
                    auto h = index[0]; // Logical index in y

                    advected(0, level, k, h) = f(0, level, k, h);
                    advected(1, level, k, h) = f(1, level, k - 1, h);
                    advected(2, level, k, h) = f(2, level, k, h - 1);
                    advected(3, level, k, h) = f(3, level, k, h);
                    advected(4, level, k, h) = f(4, level, k, h);
                    // advected(4, level, k, h) =  up_bc[4];

                    for (int scheme_n = 1; scheme_n < 4; ++scheme_n)
                    {
                        int sign1                                   = (scheme_n == 1) ? -1 : 1;
                        int sign2                                   = (scheme_n == 2) ? -1 : 1;
                        advected(0 + 4 * scheme_n + 1, level, k, h) = f(0 + 4 * scheme_n + 1, level, k - 1, h);
                        advected(1 + 4 * scheme_n + 1, level, k, h) = f(1 + 4 * scheme_n + 1, level, k, h - 1);
                        advected(2 + 4 * scheme_n + 1, level, k, h) = sign1 * f(0 + 4 * scheme_n + 1, level, k, h);
                        advected(3 + 4 * scheme_n + 1, level, k, h) = sign2 * f(1 + 4 * scheme_n + 1, level, k, h);
                    }
                });

            auto leaves_north_west = get_adjacent_boundary_northwest(mesh, level, mesh_id_t::cells);
            leaves_north_west(
                [&](auto& interval, auto& index)
                {
                    auto k = interval; // Logical index in x
                    auto h = index[0]; // Logical index in y

                    advected(0, level, k, h) = f(0, level, k, h);
                    advected(1, level, k, h) = f(1, level, k, h);
                    advected(2, level, k, h) = f(2, level, k, h - 1);
                    advected(3, level, k, h) = f(3, level, k + 1, h);
                    advected(4, level, k, h) = f(4, level, k, h);
                    // advected(4, level, k, h) =  up_bc[4];

                    for (int scheme_n = 1; scheme_n < 4; ++scheme_n)
                    {
                        int sign1                                   = (scheme_n == 1) ? -1 : 1;
                        int sign2                                   = (scheme_n == 2) ? -1 : 1;
                        advected(0 + 4 * scheme_n + 1, level, k, h) = sign1 * f(2 + 4 * scheme_n + 1, level, k, h);
                        advected(1 + 4 * scheme_n + 1, level, k, h) = f(1 + 4 * scheme_n + 1, level, k, h - 1);
                        advected(2 + 4 * scheme_n + 1, level, k, h) = f(2 + 4 * scheme_n + 1, level, k + 1, h);
                        advected(3 + 4 * scheme_n + 1, level, k, h) = sign2 * f(1 + 4 * scheme_n + 1, level, k, h);
                    }
                });

            auto leaves_south_east = get_adjacent_boundary_southeast(mesh, level, mesh_id_t::cells);
            leaves_south_east(
                [&](auto& interval, auto& index)
                {
                    auto k = interval; // Logical index in x
                    auto h = index[0]; // Logical index in y

                    advected(0, level, k, h) = f(0, level, k, h);
                    advected(1, level, k, h) = f(1, level, k, h);
                    advected(2, level, k, h) = f(2, level, k, h - 1);
                    advected(3, level, k, h) = f(3, level, k, h);
                    advected(4, level, k, h) = f(4, level, k, h + 1);

                    for (int scheme_n = 1; scheme_n < 4; ++scheme_n)
                    {
                        int sign1                                   = (scheme_n == 1) ? -1 : 1;
                        int sign2                                   = (scheme_n == 2) ? -1 : 1;
                        advected(0 + 4 * scheme_n + 1, level, k, h) = f(0 + 4 * scheme_n + 1, level, k - 1, h);
                        advected(1 + 4 * scheme_n + 1, level, k, h) = sign2 * f(3 + 4 * scheme_n + 1, level, k, h);
                        advected(2 + 4 * scheme_n + 1, level, k, h) = sign1 * f(0 + 4 * scheme_n + 1, level, k, h);
                        advected(3 + 4 * scheme_n + 1, level, k, h) = f(3 + 4 * scheme_n + 1, level, k, h + 1);
                    }
                });

            auto leaves_south_west = get_adjacent_boundary_southwest(mesh, level, mesh_id_t::cells);
            leaves_south_west(
                [&](auto& interval, auto& index)
                {
                    auto k = interval; // Logical index in x
                    auto h = index[0]; // Logical index in y

                    advected(0, level, k, h) = f(0, level, k, h);
                    advected(1, level, k, h) = f(1, level, k - 1, h);
                    advected(2, level, k, h) = f(2, level, k, h - 1);
                    advected(3, level, k, h) = f(3, level, k + 1, h);
                    advected(4, level, k, h) = f(4, level, k, h + 1);

                    for (int scheme_n = 1; scheme_n < 4; ++scheme_n)
                    {
                        int sign1                                   = (scheme_n == 1) ? -1 : 1;
                        int sign2                                   = (scheme_n == 2) ? -1 : 1;
                        advected(0 + 4 * scheme_n + 1, level, k, h) = sign1 * f(2 + 4 * scheme_n + 1, level, k, h);
                        advected(1 + 4 * scheme_n + 1, level, k, h) = sign2 * f(3 + 4 * scheme_n + 1, level, k, h);
                        advected(2 + 4 * scheme_n + 1, level, k, h) = f(2 + 4 * scheme_n + 1, level, k + 1, h);
                        advected(3 + 4 * scheme_n + 1, level, k, h) = f(3 + 4 * scheme_n + 1, level, k, h + 1);
                    }
                });

            auto leaves = samurai::intersection(mesh[mesh_id_t::cells][level], mesh[mesh_id_t::cells][level]);

            auto along_x = samurai::union_(leaves_west, leaves_east);
            auto along_y = samurai::union_(leaves_south, leaves_north);

            auto diagonals = samurai::union_(samurai::union_(samurai::union_(leaves_north_east, leaves_north_west), leaves_south_east),
                                             leaves_south_west);

            auto remaining_leaves = samurai::difference(samurai::difference(samurai::difference(leaves, along_x), along_y), diagonals);

            remaining_leaves(
                [&](auto& interval, auto& index)
                {
                    auto k = interval; // Logical index in x
                    auto h = index[0]; // Logical index in y

                    // D2Q5
                    advected(0, level, k, h) = f(0, level, k, h);
                    advected(1, level, k, h) = f(1, level, k - 1, h);
                    advected(2, level, k, h) = f(2, level, k, h - 1);
                    advected(3, level, k, h) = f(3, level, k + 1, h);
                    advected(4, level, k, h) = f(4, level, k, h + 1);

                    // D2Q4
                    for (int scheme_n = 1; scheme_n < 4; ++scheme_n)
                    {
                        advected(0 + 4 * scheme_n + 1, level, k, h) = f(0 + 4 * scheme_n + 1, level, k - 1, h);
                        advected(1 + 4 * scheme_n + 1, level, k, h) = f(1 + 4 * scheme_n + 1, level, k, h - 1);
                        advected(2 + 4 * scheme_n + 1, level, k, h) = f(2 + 4 * scheme_n + 1, level, k + 1, h);
                        advected(3 + 4 * scheme_n + 1, level, k, h) = f(3 + 4 * scheme_n + 1, level, k, h + 1);
                    }
                });
        }
        else // Advection at the coarse levels using the overleaves
        {
            auto lev_p_1  = level + 1;
            std::size_t j = max_level - (lev_p_1);
            double coeff  = 1. / (1 << (2 * j)); // ATTENTION A LA DIMENSION 2 !!!!

            // std::cout<<std::endl<<"Level = "<<level<<std::endl;

            auto leaves = samurai::intersection(mesh[mesh_id_t::cells][level], mesh[mesh_id_t::cells][level]);

            auto ol_east      = samurai::intersection(get_adjacent_boundary_east(mesh, lev_p_1, mesh_id_t::overleaves),
                                                 mesh[mesh_id_t::cells][level]);
            auto ol_northeast = samurai::intersection(get_adjacent_boundary_northeast(mesh, lev_p_1, mesh_id_t::overleaves),
                                                      mesh[mesh_id_t::cells][level]);
            auto ol_southeast = samurai::intersection(get_adjacent_boundary_southeast(mesh, lev_p_1, mesh_id_t::overleaves),
                                                      mesh[mesh_id_t::cells][level]);
            auto ol_west      = samurai::intersection(get_adjacent_boundary_west(mesh, lev_p_1, mesh_id_t::overleaves),
                                                 mesh[mesh_id_t::cells][level]);
            auto ol_northwest = samurai::intersection(get_adjacent_boundary_northwest(mesh, lev_p_1, mesh_id_t::overleaves),
                                                      mesh[mesh_id_t::cells][level]);
            auto ol_southwest = samurai::intersection(get_adjacent_boundary_southwest(mesh, lev_p_1, mesh_id_t::overleaves),
                                                      mesh[mesh_id_t::cells][level]);
            auto ol_north     = samurai::intersection(get_adjacent_boundary_north(mesh, lev_p_1, mesh_id_t::overleaves),
                                                  mesh[mesh_id_t::cells][level]);
            auto ol_south     = samurai::intersection(get_adjacent_boundary_south(mesh, lev_p_1, mesh_id_t::overleaves),
                                                  mesh[mesh_id_t::cells][level]);

            // Exiting fluxes which do not need any correction
            // whatever their position is.
            leaves.on(lev_p_1)(
                [&](auto& interval, auto& index)
                {
                    auto k = interval; // Logical index in x
                    auto h = index[0]; // Logical index in y

                    std::array<unsigned short int, 4> vld_flx{1, 3, 5, 7};

                    for (int scheme_n = 0; scheme_n < 4; ++scheme_n)
                    {
                        for (auto fl_nb : vld_flx)
                        {
                            for (auto& c : pred_coeff[j][fl_nb].coeff)
                            {
                                coord_index_t stencil_x, stencil_y;
                                std::tie(stencil_x, stencil_y) = c.first;

                                fluxes((fl_nb >> 1) + 4 * scheme_n + 1,
                                       lev_p_1,
                                       k,
                                       h) -= c.second * f((fl_nb >> 1) + 4 * scheme_n + 1, lev_p_1, k + stencil_x, h + stencil_y);
                            }
                        }
                    }
                });

            ol_east.on(lev_p_1)(
                [&](auto& interval, auto& index)
                {
                    auto k = interval; // Logical index in x
                    auto h = index[0]; // Logical index in y

                    // std::cout<<std::endl<<"East : "<<k<<"  | 
                    // "<<h<<std::endl;

                    std::array<unsigned short int, 3> vld_flx{0, 2, 6};

                    for (int scheme_n = 0; scheme_n < 4; ++scheme_n)
                    {
                        int sign = (scheme_n == 1) ? -1 : 1;

                        for (auto fl_nb : vld_flx)
                        {
                            for (auto& c : pred_coeff[j][fl_nb].coeff)
                            {
                                coord_index_t stencil_x, stencil_y;
                                std::tie(stencil_x, stencil_y) = c.first;

                                fluxes((fl_nb >> 1) + 4 * scheme_n + 1,
                                       lev_p_1,
                                       k,
                                       h) += c.second * f((fl_nb >> 1) + 4 * scheme_n + 1, lev_p_1, k + stencil_x, h + stencil_y);
                            }
                        }

                        if (scheme_n == 0)
                        {
                            fluxes(2 + 4 * scheme_n + 1, lev_p_1, k, h) += (1 << j) * f(2 + 4 * scheme_n + 1, lev_p_1, k, h);
                        }
                        else
                        {
                            fluxes(2 + 4 * scheme_n + 1, lev_p_1, k, h) += (1 << j) * sign * f(0 + 4 * scheme_n + 1, lev_p_1, k, h);
                        }
                    }
                });
            ol_northeast.on(lev_p_1)(
                [&](auto& interval, auto& index)
                {
                    auto k = interval; // Logical index in x
                    auto h = index[0]; // Logical index in y

                    // std::cout<<std::endl<<"NorthEast : "<<k<<"  | 
                    // "<<h<<std::endl;

                    std::array<unsigned short int, 2> vld_flx{0, 2};

                    for (int scheme_n = 0; scheme_n < 4; ++scheme_n)
                    {
                        int sign1 = (scheme_n == 1) ? -1 : 1;
                        int sign2 = (scheme_n == 2) ? -1 : 1;

                        for (auto fl_nb : vld_flx)
                        {
                            for (auto& c : pred_coeff[j][fl_nb].coeff)
                            {
                                coord_index_t stencil_x, stencil_y;
                                std::tie(stencil_x, stencil_y) = c.first;

                                fluxes((fl_nb >> 1) + 4 * scheme_n + 1,
                                       lev_p_1,
                                       k,
                                       h) += c.second * f((fl_nb >> 1) + 4 * scheme_n + 1, lev_p_1, k + stencil_x, h + stencil_y);
                            }
                        }

                        if (scheme_n == 0)
                        {
                            fluxes(2 + 4 * scheme_n + 1, lev_p_1, k, h) += (1 << j) * f(2 + 4 * scheme_n + 1, lev_p_1, k, h);
                            fluxes(3 + 4 * scheme_n + 1, lev_p_1, k, h) += (1 << j) * f(3 + 4 * scheme_n + 1, lev_p_1, k, h);
                            // fluxes(3 + 4 * scheme_n + 1, lev_p_1, k, h) +=
                            // (1<<j) * up_bc[4];
                        }
                        else
                        {
                            fluxes(2 + 4 * scheme_n + 1, lev_p_1, k, h) += (1 << j) * sign1 * f(0 + 4 * scheme_n + 1, lev_p_1, k, h);
                            fluxes(3 + 4 * scheme_n + 1, lev_p_1, k, h) += (1 << j) * sign2 * f(1 + 4 * scheme_n + 1, lev_p_1, k, h);
                        }
                    }
                });
            ol_southeast.on(lev_p_1)(
                [&](auto& interval, auto& index)
                {
                    auto k = interval; // Logical index in x
                    auto h = index[0]; // Logical index in y

                    // std::cout<<std::endl<<"SouthEast : "<<k<<"  | 
                    // "<<h<<std::endl;

                    std::array<unsigned short int, 2> vld_flx{0, 6};

                    for (int scheme_n = 0; scheme_n < 4; ++scheme_n)
                    {
                        int sign1 = (scheme_n == 1) ? -1 : 1;
                        int sign2 = (scheme_n == 2) ? -1 : 1;

                        for (auto fl_nb : vld_flx)
                        {
                            for (auto& c : pred_coeff[j][fl_nb].coeff)
                            {
                                coord_index_t stencil_x, stencil_y;
                                std::tie(stencil_x, stencil_y) = c.first;

                                fluxes((fl_nb >> 1) + 4 * scheme_n + 1,
                                       lev_p_1,
                                       k,
                                       h) += c.second * f((fl_nb >> 1) + 4 * scheme_n + 1, lev_p_1, k + stencil_x, h + stencil_y);
                            }
                        }

                        if (scheme_n == 0)
                        {
                            fluxes(1 + 4 * scheme_n + 1, lev_p_1, k, h) += (1 << j) * f(1 + 4 * scheme_n + 1, lev_p_1, k, h);
                            fluxes(2 + 4 * scheme_n + 1, lev_p_1, k, h) += (1 << j) * f(2 + 4 * scheme_n + 1, lev_p_1, k, h);
                        }
                        else
                        {
                            fluxes(1 + 4 * scheme_n + 1, lev_p_1, k, h) += (1 << j) * sign2 * f(3 + 4 * scheme_n + 1, lev_p_1, k, h);
                            fluxes(2 + 4 * scheme_n + 1, lev_p_1, k, h) += (1 << j) * sign1 * f(0 + 4 * scheme_n + 1, lev_p_1, k, h);
                        }
                    }
                });
            ol_west.on(lev_p_1)(
                [&](auto& interval, auto& index)
                {
                    auto k = interval; // Logical index in x
                    auto h = index[0]; // Logical index in y
                    // std::cout<<std::endl<<"West : "<<k<<"  | 
                    // "<<h<<std::endl;

                    std::array<unsigned short int, 3> vld_flx{2, 4, 6};

                    for (int scheme_n = 0; scheme_n < 4; ++scheme_n)
                    {
                        int sign = (scheme_n == 1) ? -1 : 1;

                        for (auto fl_nb : vld_flx)
                        {
                            for (auto& c : pred_coeff[j][fl_nb].coeff)
                            {
                                coord_index_t stencil_x, stencil_y;
                                std::tie(stencil_x, stencil_y) = c.first;

                                fluxes((fl_nb >> 1) + 4 * scheme_n + 1,
                                       lev_p_1,
                                       k,
                                       h) += c.second * f((fl_nb >> 1) + 4 * scheme_n + 1, lev_p_1, k + stencil_x, h + stencil_y);
                            }
                        }

                        if (scheme_n == 0)
                        {
                            fluxes(0 + 4 * scheme_n + 1, lev_p_1, k, h) += (1 << j) * f(0 + 4 * scheme_n + 1, lev_p_1, k, h);
                        }
                        else
                        {
                            fluxes(0 + 4 * scheme_n + 1, lev_p_1, k, h) += (1 << j) * sign * f(2 + 4 * scheme_n + 1, lev_p_1, k, h);
                        }
                    }
                });
            ol_northwest.on(lev_p_1)(
                [&](auto& interval, auto& index)
                {
                    auto k = interval; // Logical index in x
                    auto h = index[0]; // Logical index in y
                    // std::cout<<std::endl<<"NorthWest : "<<k<<"  | 
                    // "<<h<<std::endl;

                    std::array<unsigned short int, 2> vld_flx{2, 4};

                    for (int scheme_n = 0; scheme_n < 4; ++scheme_n)
                    {
                        int sign1 = (scheme_n == 1) ? -1 : 1;
                        int sign2 = (scheme_n == 2) ? -1 : 1;

                        for (auto fl_nb : vld_flx)
                        {
                            for (auto& c : pred_coeff[j][fl_nb].coeff)
                            {
                                coord_index_t stencil_x, stencil_y;
                                std::tie(stencil_x, stencil_y) = c.first;

                                fluxes((fl_nb >> 1) + 4 * scheme_n + 1,
                                       lev_p_1,
                                       k,
                                       h) += c.second * f((fl_nb >> 1) + 4 * scheme_n + 1, lev_p_1, k + stencil_x, h + stencil_y);
                            }
                        }
                        if (scheme_n == 0)
                        {
                            fluxes(0 + 4 * scheme_n + 1, lev_p_1, k, h) += (1 << j) * f(0 + 4 * scheme_n + 1, lev_p_1, k, h);
                            fluxes(3 + 4 * scheme_n + 1, lev_p_1, k, h) += (1 << j) * f(3 + 4 * scheme_n + 1, lev_p_1, k, h);
                            // fluxes(3 + 4 * scheme_n + 1, lev_p_1, k, h) +=
                            // (1<<j) * up_bc[4];
                        }
                        else
                        {
                            fluxes(0 + 4 * scheme_n + 1, lev_p_1, k, h) += (1 << j) * sign1 * f(2 + 4 * scheme_n + 1, lev_p_1, k, h);
                            fluxes(3 + 4 * scheme_n + 1, lev_p_1, k, h) += (1 << j) * sign2 * f(1 + 4 * scheme_n + 1, lev_p_1, k, h);
                        }
                    }
                });
            ol_southwest.on(lev_p_1)(
                [&](auto& interval, auto& index)
                {
                    auto k = interval; // Logical index in x
                    auto h = index[0]; // Logical index in y
                    // std::cout<<std::endl<<"SouthWest : "<<k<<"  | 
                    // "<<h<<std::endl;

                    std::array<unsigned short int, 2> vld_flx{4, 6};

                    for (int scheme_n = 0; scheme_n < 4; ++scheme_n)
                    {
                        int sign1 = (scheme_n == 1) ? -1 : 1;
                        int sign2 = (scheme_n == 2) ? -1 : 1;

                        for (auto fl_nb : vld_flx)
                        {
                            for (auto& c : pred_coeff[j][fl_nb].coeff)
                            {
                                coord_index_t stencil_x, stencil_y;
                                std::tie(stencil_x, stencil_y) = c.first;

                                fluxes((fl_nb >> 1) + 4 * scheme_n + 1,
                                       lev_p_1,
                                       k,
                                       h) += c.second * f((fl_nb >> 1) + 4 * scheme_n + 1, lev_p_1, k + stencil_x, h + stencil_y);
                            }
                        }

                        if (scheme_n == 0)
                        {
                            fluxes(0 + 4 * scheme_n + 1, lev_p_1, k, h) += (1 << j) * f(0 + 4 * scheme_n + 1, lev_p_1, k, h);
                            fluxes(1 + 4 * scheme_n + 1, lev_p_1, k, h) += (1 << j) * f(1 + 4 * scheme_n + 1, lev_p_1, k, h);
                        }
                        else
                        {
                            fluxes(0 + 4 * scheme_n + 1, lev_p_1, k, h) += (1 << j) * sign1 * f(2 + 4 * scheme_n + 1, lev_p_1, k, h);
                            fluxes(1 + 4 * scheme_n + 1, lev_p_1, k, h) += (1 << j) * sign2 * f(3 + 4 * scheme_n + 1, lev_p_1, k, h);
                        }
                    }
                });
            ol_north.on(lev_p_1)(
                [&](auto& interval, auto& index)
                {
                    auto k = interval; // Logical index in x
                    auto h = index[0]; // Logical index in y
                    // std::cout<<std::endl<<"North : "<<k<<"  | 
                    // "<<h<<std::endl;

                    std::array<unsigned short int, 3> vld_flx{0, 2, 4};

                    for (int scheme_n = 0; scheme_n < 4; ++scheme_n)
                    {
                        int sign = (scheme_n == 2) ? -1 : 1;

                        for (auto fl_nb : vld_flx)
                        {
                            for (auto& c : pred_coeff[j][fl_nb].coeff)
                            {
                                coord_index_t stencil_x, stencil_y;
                                std::tie(stencil_x, stencil_y) = c.first;

                                fluxes((fl_nb >> 1) + 4 * scheme_n + 1,
                                       lev_p_1,
                                       k,
                                       h) += c.second * f((fl_nb >> 1) + 4 * scheme_n + 1, lev_p_1, k + stencil_x, h + stencil_y);
                            }
                        }
                        if (scheme_n == 0)
                        {
                            fluxes(3 + 4 * scheme_n + 1, lev_p_1, k, h) += (1 << j) * f(3 + 4 * scheme_n + 1, lev_p_1, k, h);
                        }
                        // fluxes(3 + 4 * scheme_n + 1, lev_p_1, k, h) += (1<<j)
                        // * up_bc[4];
                        else
                        {
                            fluxes(3 + 4 * scheme_n + 1, lev_p_1, k, h) += (1 << j) * sign * f(1 + 4 * scheme_n + 1, lev_p_1, k, h);
                        }
                    }
                });
            ol_south.on(lev_p_1)(
                [&](auto& interval, auto& index)
                {
                    auto k = interval; // Logical index in x
                    auto h = index[0]; // Logical index in y
                    // std::cout<<std::endl<<"South : "<<k<<"  | 
                    // "<<h<<std::endl;

                    std::array<unsigned short int, 3> vld_flx{0, 4, 6};

                    for (int scheme_n = 0; scheme_n < 4; ++scheme_n)
                    {
                        int sign = (scheme_n == 2) ? -1 : 1;

                        for (auto fl_nb : vld_flx)
                        {
                            for (auto& c : pred_coeff[j][fl_nb].coeff)
                            {
                                coord_index_t stencil_x, stencil_y;
                                std::tie(stencil_x, stencil_y) = c.first;

                                fluxes((fl_nb >> 1) + 4 * scheme_n + 1,
                                       lev_p_1,
                                       k,
                                       h) += c.second * f((fl_nb >> 1) + 4 * scheme_n + 1, lev_p_1, k + stencil_x, h + stencil_y);
                            }
                        }
                        if (scheme_n == 0)
                        {
                            fluxes(1 + 4 * scheme_n + 1, lev_p_1, k, h) += (1 << j) * f(1 + 4 * scheme_n + 1, lev_p_1, k, h);
                        }
                        else
                        {
                            fluxes(1 + 4 * scheme_n + 1, lev_p_1, k, h) += (1 << j) * sign * f(3 + 4 * scheme_n + 1, lev_p_1, k, h);
                        }
                    }
                });

            // Now we are left for the regular entering fluxes for the cells far
            // from the boundary
            auto ol_touching_east = samurai::union_(samurai::union_(ol_east, ol_northeast), ol_southeast);
            auto ol_touching_west = samurai::union_(samurai::union_(ol_west, ol_northwest), ol_southwest);

            auto ol_boundary = samurai::union_(samurai::union_(samurai::union_(ol_touching_east, ol_touching_west), ol_north), ol_south);

            auto ol_inside = samurai::difference(leaves, ol_boundary).on(lev_p_1);

            // leaves.on(lev_p_1)([&](auto& interval, auto& index) {
            ol_inside(
                [&](auto& interval, auto& index)
                {
                    auto k = interval; // Logical index in x
                    auto h = index[0]; // Logical index in y

                    std::array<unsigned short int, 4> vld_flx{0, 2, 4, 6};

                    for (int scheme_n = 0; scheme_n < 4; ++scheme_n)
                    {
                        for (auto fl_nb : vld_flx)
                        {
                            for (auto& c : pred_coeff[j][fl_nb].coeff)
                            {
                                coord_index_t stencil_x, stencil_y;
                                std::tie(stencil_x, stencil_y) = c.first;

                                fluxes((fl_nb >> 1) + 4 * scheme_n + 1,
                                       lev_p_1,
                                       k,
                                       h) += c.second * f((fl_nb >> 1) + 4 * scheme_n + 1, lev_p_1, k + stencil_x, h + stencil_y);
                            }
                        }
                    }
                });

            leaves(
                [&](auto& interval, auto& index)
                {
                    auto k = interval; // Logical index in x
                    auto h = index[0]; // Logical index in y

                    advected(0, level, k, h) = f(0, level, k, h);

                    for (int alpha = 1; alpha < 17; ++alpha)
                    {
                        advected(alpha, level, k, h) = f(alpha, level, k, h)
                                                     + coeff * 0.25
                                                           * (fluxes(alpha, lev_p_1, 2 * k, 2 * h) + fluxes(alpha, lev_p_1, 2 * k + 1, 2 * h)
                                                              + fluxes(alpha, lev_p_1, 2 * k, 2 * h + 1)
                                                              + fluxes(alpha, lev_p_1, 2 * k + 1, 2 * h + 1));
                    }
                });
        }

        double dx = 1. / (1 << max_level);
        double dt = dx / lambda;

        leaves(
            [&](auto& interval, auto& index)
            {
                auto k = interval; // Logical index in x
                auto h = index[0]; // Logical index in y

                // We compute the advected momenti
                auto m0_0 = xt::eval(advected(0, level, k, h) + advected(1, level, k, h) + advected(2, level, k, h)
                                     + advected(3, level, k, h) + advected(4, level, k, h));
                auto m0_1 = xt::eval(lambda * (advected(1, level, k, h) - advected(3, level, k, h)));
                auto m0_2 = xt::eval(lambda * (advected(2, level, k, h) - advected(4, level, k, h)));
                auto m0_3 = xt::eval(lambda * lambda / 5.
                                     * (-4 * advected(0, level, k, h) + 21 * advected(1, level, k, h) + 21 * advected(2, level, k, h)
                                        + 21 * advected(3, level, k, h) + 21 * advected(4, level, k, h)));
                auto m0_4 = xt::eval(
                    lambda * lambda
                    * (advected(1, level, k, h) - advected(2, level, k, h) + advected(3, level, k, h) - advected(4, level, k, h)));

                auto m1_0 = xt::eval(advected(5, level, k, h) + advected(6, level, k, h) + advected(7, level, k, h)
                                     + advected(8, level, k, h));
                auto m1_1 = xt::eval(lambda * (advected(5, level, k, h) - advected(7, level, k, h)));
                auto m1_2 = xt::eval(lambda * (advected(6, level, k, h) - advected(8, level, k, h)));
                auto m1_3 = xt::eval(
                    lambda * lambda
                    * (advected(5, level, k, h) - advected(6, level, k, h) + advected(7, level, k, h) - advected(8, level, k, h)));

                auto m2_0 = xt::eval(advected(9, level, k, h) + advected(10, level, k, h) + advected(11, level, k, h)
                                     + advected(12, level, k, h));
                auto m2_1 = xt::eval(lambda * (advected(9, level, k, h) - advected(11, level, k, h)));
                auto m2_2 = xt::eval(lambda * (advected(10, level, k, h) - advected(12, level, k, h)));
                auto m2_3 = xt::eval(
                    lambda * lambda
                    * (advected(9, level, k, h) - advected(10, level, k, h) + advected(11, level, k, h) - advected(12, level, k, h)));

                auto m3_0 = xt::eval(advected(13, level, k, h) + advected(14, level, k, h) + advected(15, level, k, h)
                                     + advected(16, level, k, h));
                auto m3_1 = xt::eval(lambda * (advected(13, level, k, h) - advected(15, level, k, h)));
                auto m3_2 = xt::eval(lambda * (advected(14, level, k, h) - advected(16, level, k, h)));
                auto m3_3 = xt::eval(
                    lambda * lambda
                    * (advected(13, level, k, h) - advected(14, level, k, h) + advected(15, level, k, h) - advected(16, level, k, h)));

                m0_1 = (1 - s_rho_x) * m0_1 + s_rho_x * (m1_0);
                m0_2 = (1 - s_rho_x) * m0_2 + s_rho_x * (m2_0);
                m0_3 = (1 - s_rho_xy) * m0_3 + s_rho_xy * (m1_0 * m1_0 + m2_0 * m2_0) / m0_0;
                m0_4 = (1 - s_rho_xy) * m0_4;

                m1_1 = (1 - s_u_x) * m1_1
                     + s_u_x
                           * ((3. / 2. - gas_constant / 2.) * (m1_0 * m1_0) / (m0_0)
                              + (1. / 2. - gas_constant / 2.) * (m2_0 * m2_0) / (m0_0) + (gas_constant - 1.) * m3_0);
                m1_2 = (1 - s_u_x) * m1_2 + s_u_x * (m1_0 * m2_0 / m0_0);
                m1_3 = (1 - s_u_xy) * m1_3;

                m2_1 = (1 - s_u_x) * m2_1 + s_u_x * (m1_0 * m2_0 / m0_0);
                m2_2 = (1 - s_u_x) * m2_2
                     + s_u_x
                           * ((3. / 2. - gas_constant / 2.) * (m2_0 * m2_0) / (m0_0)
                              + (1. / 2. - gas_constant / 2.) * (m1_0 * m1_0) / (m0_0) + (gas_constant - 1.) * m3_0);
                m2_3 = (1 - s_u_xy) * m2_3;

                m3_1 = (1 - s_p_x) * m3_1
                     + s_p_x
                           * (gas_constant * (m1_0 * m3_0) / (m0_0) - (gas_constant / 2. - 1. / 2.) * (m1_0 * m1_0 * m1_0) / (m0_0 * m0_0)
                              - (gas_constant / 2. - 1. / 2.) * (m1_0 * m2_0 * m2_0) / (m0_0 * m0_0));
                m3_2 = (1 - s_p_x) * m3_2
                     + s_p_x
                           * (gas_constant * (m2_0 * m3_0) / (m0_0) - (gas_constant / 2. - 1. / 2.) * (m2_0 * m2_0 * m2_0) / (m0_0 * m0_0)
                              - (gas_constant / 2. - 1. / 2.) * (m2_0 * m1_0 * m1_0) / (m0_0 * m0_0));
                m3_3 = (1 - s_p_xy) * m3_3;

                // Source terms
                m2_0 += -m0_0 * grav * dt; // Vertical gravity
                m3_0 += -m2_0 / m0_0 * grav * dt;

                new_f(0, level, k, h) = 21. / 25 * m0_0 - 1. / (5 * lambda * lambda) * m0_3;
                new_f(1, level, k, h) = 1. / 25 * m0_0 + .5 / lambda * m0_1 + 1. / (20 * lambda * lambda) * m0_3
                                      + .25 / (lambda * lambda) * m0_4;
                new_f(2, level, k, h) = 1. / 25 * m0_0 + +.5 / lambda * m0_2 + 1. / (20 * lambda * lambda) * m0_3
                                      - .25 / (lambda * lambda) * m0_4;
                new_f(3, level, k, h) = 1. / 25 * m0_0 - .5 / lambda * m0_1 + 1. / (20 * lambda * lambda) * m0_3
                                      + .25 / (lambda * lambda) * m0_4;
                new_f(4, level, k, h) = 1. / 25 * m0_0 + -.5 / lambda * m0_2 + 1. / (20 * lambda * lambda) * m0_3
                                      - .25 / (lambda * lambda) * m0_4;

                new_f(5, level, k, h) = .25 * m1_0 + .5 / lambda * (m1_1) + .25 / (lambda * lambda) * m1_3;
                new_f(6, level, k, h) = .25 * m1_0 + .5 / lambda * (m1_2)-.25 / (lambda * lambda) * m1_3;
                new_f(7, level, k, h) = .25 * m1_0 - .5 / lambda * (m1_1) + .25 / (lambda * lambda) * m1_3;
                new_f(8, level, k, h) = .25 * m1_0 - .5 / lambda * (m1_2)-.25 / (lambda * lambda) * m1_3;

                new_f(9, level, k, h)  = .25 * m2_0 + .5 / lambda * (m2_1) + .25 / (lambda * lambda) * m2_3;
                new_f(10, level, k, h) = .25 * m2_0 + .5 / lambda * (m2_2)-.25 / (lambda * lambda) * m2_3;
                new_f(11, level, k, h) = .25 * m2_0 - .5 / lambda * (m2_1) + .25 / (lambda * lambda) * m2_3;
                new_f(12, level, k, h) = .25 * m2_0 - .5 / lambda * (m2_2)-.25 / (lambda * lambda) * m2_3;

                new_f(13, level, k, h) = .25 * m3_0 + .5 / lambda * (m3_1) + .25 / (lambda * lambda) * m3_3;
                new_f(14, level, k, h) = .25 * m3_0 + .5 / lambda * (m3_2)-.25 / (lambda * lambda) * m3_3;
                new_f(15, level, k, h) = .25 * m3_0 - .5 / lambda * (m3_1) + .25 / (lambda * lambda) * m3_3;
                new_f(16, level, k, h) = .25 * m3_0 - .5 / lambda * (m3_2)-.25 / (lambda * lambda) * m3_3;
            });
    }
    std::swap(f.array(), new_f.array());
}

template <class Field>
void save_solution(Field& f, double eps, std::size_t ite, const double gas_constant, std::string ext = "")
{
    using value_t = typename Field::value_type;

    auto mesh       = f.mesh();
    using mesh_id_t = typename decltype(mesh)::mesh_id_t;

    std::size_t min_level = mesh.min_level();
    std::size_t max_level = mesh.max_level();

    std::stringstream str;
    str << "LBM_D2Q5444_RayleighTaylor_" << ext << "_lmin_" << min_level << "_lmax-" << max_level << "_eps-" << eps << "_ite-" << ite;

    auto level = samurai::make_scalar_field<std::size_t>("level", mesh);
    auto rho   = samurai::make_scalar_field<value_t>("rho", mesh);
    auto qx    = samurai::make_scalar_field<value_t>("qx", mesh);
    auto qy    = samurai::make_scalar_field<value_t>("qy", mesh);
    auto e     = samurai::make_scalar_field<value_t>("e", mesh);
    auto s     = samurai::make_scalar_field<value_t>("entropy", mesh);

    samurai::for_each_cell(mesh[mesh_id_t::cells],
                           [&](auto& cell)
                           {
                               level[cell] = cell.level;
                               rho[cell]   = f[cell][0] + f[cell][1] + f[cell][2] + f[cell][3] + f[cell][4];
                               qx[cell]    = f[cell][5] + f[cell][6] + f[cell][7] + f[cell][8];
                               qy[cell]    = f[cell][9] + f[cell][10] + f[cell][11] + f[cell][12];
                               e[cell]     = f[cell][13] + f[cell][14] + f[cell][15] + f[cell][16];

                               // Computing the entropy with multiplicative constant 1 and additive
                               // constant 0
                               auto p = (gas_constant - 1.) * (e[cell] - .5 * (std::pow(qx[cell], 2.) + std::pow(qy[cell], 2.)) / rho[cell]);
                               s[cell] = std::log(p / std::pow(rho[cell], gas_constant));
                           });

    samurai::save(str.str().data(), mesh, rho, qx, qy, e, s, f, level);
}

int main(int argc, char* argv[])
{
    samurai::initialize(argc, argv);

    cxxopts::Options options("lbm_d2q5444_RayleighTaylor", "");

    options.add_options()("min_level", "minimum level", cxxopts::value<std::size_t>()->default_value("2"))(
        "max_level",
        "maximum level",
        cxxopts::value<std::size_t>()->default_value("7"))("epsilon", "maximum level", cxxopts::value<double>()->default_value("0.0001"))(
        "ite",
        "number of iteration",
        cxxopts::value<std::size_t>()->default_value("100"))("reg", "regularity", cxxopts::value<double>()->default_value("0."))("h, help",
                                                                                                                                 "Help");

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

            const double gas_constant = 1.4;

            double lambda = 5.;

            // We are obliged to multiply the final time by 1/0.3
            // because the domain is of size 1 instead of 0.3
            const double T = 2.7; // 30.;

            samurai::Box<double, dim> box({0, 0}, {1, 1});
            samurai::MROMesh<Config> mesh(box, min_level, max_level);
            using mesh_id_t     = typename samurai::MROMesh<Config>::mesh_id_t;
            using coord_index_t = typename samurai::MROMesh<Config>::coord_index_t;
            auto pred_coeff     = compute_prediction<coord_index_t>(min_level, max_level);

            // Initialization
            auto f    = init_f(mesh);
            double dx = 1.0 / (1 << max_level);
            double dt = dx / lambda;

            std::size_t N = static_cast<std::size_t>(T / dt);

            int howoften = 32; // How often is the solution saved ?

            auto update_bc_for_level = [](auto& field, std::size_t level)
            {
                // update_bc_D2Q4_3_Euler_constant_extension(field, level);
                update_bc_D2Q4_3_Euler_linear_extension(field, level);
            };

            auto MRadaptation = samurai::make_MRAdapt(f, update_bc_for_level);

            double t = 0.;

            for (std::size_t nb_ite = 0; nb_ite <= N; ++nb_ite)
            {
                std::cout << std::endl << "Time = " << t << "   Iteration number = " << nb_ite << std::endl;

                if (max_level > min_level)
                {
                    MRadaptation(eps, regularity);
                }

                if (nb_ite % howoften == 0)
                {
                    save_solution(f, eps, nb_ite / howoften,
                                  gas_constant); // Before applying the scheme
                }

                one_time_step(f, update_bc_for_level, pred_coeff, lambda);
                // save_solution(f, eps, nb_ite, gas_constant, "post");
                t += dt;
            }
        }
    }
    catch (const cxxopts::OptionException& e)
    {
        std::cout << options.help() << "\n";
    }
    samurai::finalize();
    return 0;
}
