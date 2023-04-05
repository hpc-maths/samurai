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

#include "boundary_conditions.hpp"
#include "prediction_map_2d.hpp"

#include "utils_lbm_mr_2d.hpp"

bool inside_obstacle(double x, double y, const double radius)
{
    double x_center = 5. / 16., y_center = 0.5;
    if ((std::sqrt(std::pow(x - x_center, 2.) + std::pow(y - y_center, 2.))) > radius)
    {
        return false;
    }
    else
    {
        return true;
    }
}

double level_set_obstacle(double x, double y, const double radius)
{
    double x_center = 5. / 16., y_center = 0.5;

    return ((std::sqrt(std::pow(x - x_center, 2.) + std::pow(y - y_center, 2.))) - radius);
}

double volume_inside_obstacle_estimation(double xLL, double yLL, double dx, const double radius)
{
    // Super stupid way of
    // computing the volume of the cell inside the obstacle

    const int n_point_per_direction = 30;
    double step                     = dx / n_point_per_direction;

    int volume    = 0;
    int volume_in = 0;

    for (int i = 0; i < n_point_per_direction; ++i)
    {
        for (int j = 0; j < n_point_per_direction; ++j)
        {
            volume++;

            if (inside_obstacle(xLL + i * step, yLL + j * step, radius))
            {
                volume_in++;
            }
        }
    }

    // That is the volumic fraction
    return static_cast<double>(volume_in) / static_cast<double>(volume);
}

double length_obstacle_inside_cell(double xLL, double yLL, double dx, const double radius)
{
    double delta = 0.05 * dx;

    const int n_point_per_direction = 30;
    double step                     = dx / n_point_per_direction;

    int volume    = 0;
    int volume_in = 0;

    for (int i = 0; i < n_point_per_direction; ++i)
    {
        for (int j = 0; j < n_point_per_direction; ++j)
        {
            volume++;

            if (std::abs(level_set_obstacle(xLL + i * step, yLL + j * step, radius)) < 0.5 * delta)
            {
                volume_in++;
            }
        }
    }

    return dx * dx * (static_cast<double>(volume_in) / static_cast<double>(volume)) / delta;
}

std::array<double, 9> inlet_bc(double rho0, double u0, double lambda, std::string& momenti)
{
    std::array<double, 9> to_return;

    double rho = rho0;
    double qx  = rho0 * u0;
    double qy  = 0.;

    double r1 = 1.0 / lambda;
    double r2 = 1.0 / (lambda * lambda);
    double r3 = 1.0 / (lambda * lambda * lambda);
    double r4 = 1.0 / (lambda * lambda * lambda * lambda);

    double cs2 = (lambda * lambda) / 3.0; // Sound velocity of the lattice squared

    if (!momenti.compare(std::string("Geier")))
    {
        // This is the Geier choice of moment

        double m0 = rho;
        double m1 = qx;
        double m2 = qy;
        double m3 = (qx * qx + qy * qy) / rho + 2. * rho * cs2;
        double m4 = qx * (cs2 + (qy / rho) * (qy / rho));
        double m5 = qy * (cs2 + (qx / rho) * (qx / rho));
        double m6 = rho * (cs2 + (qx / rho) * (qx / rho)) * (cs2 + (qy / rho) * (qy / rho));
        double m7 = (qx * qx - qy * qy) / rho;
        double m8 = qx * qy / rho;

        // We come back to the distributions

        to_return[0] = m0 - r2 * m3 + r4 * m6;
        to_return[1] = .5 * r1 * m1 + .25 * r2 * m3 - .5 * r3 * m4 - .5 * r4 * m6 + .25 * r2 * m7;
        to_return[2] = .5 * r1 * m2 + .25 * r2 * m3 - .5 * r3 * m5 - .5 * r4 * m6 - .25 * r2 * m7;
        to_return[3] = -.5 * r1 * m1 + .25 * r2 * m3 + .5 * r3 * m4 - .5 * r4 * m6 + .25 * r2 * m7;
        to_return[4] = -.5 * r1 * m2 + .25 * r2 * m3 + .5 * r3 * m5 - .5 * r4 * m6 - .25 * r2 * m7;
        to_return[5] = .25 * r3 * m4 + .25 * r3 * m5 + .25 * r4 * m6 + .25 * r2 * m8;
        to_return[6] = -.25 * r3 * m4 + .25 * r3 * m5 + .25 * r4 * m6 - .25 * r2 * m8;
        to_return[7] = -.25 * r3 * m4 - .25 * r3 * m5 + .25 * r4 * m6 + .25 * r2 * m8;
        to_return[8] = .25 * r3 * m4 - .25 * r3 * m5 + .25 * r4 * m6 - .25 * r2 * m8;
    }

    if (!momenti.compare(std::string("Lallemand")))
    {
        double m0 = rho;
        double m1 = qx;
        double m2 = qy;
        double m3 = -2 * lambda * lambda * rho + 3. / rho * (qx * qx + qy * qy);
        double m4 = -lambda * lambda * qx;
        double m5 = -lambda * lambda * qy;
        double m6 = lambda * lambda * lambda * lambda * rho - 3. * lambda * lambda / rho * (qx * qx + qy * qy);
        double m7 = (qx * qx - qy * qy) / rho;
        double m8 = qx * qy / rho;

        to_return[0] = (1. / 9) * m0 - (1. / 9) * r2 * m3 + (1. / 9) * r4 * m6;
        to_return[1] = (1. / 9) * m0 + (1. / 6) * r1 * m1 - (1. / 36) * r2 * m3 - (1. / 6) * r3 * m4 - (1. / 18) * r4 * m6 + .25 * r2 * m7;
        to_return[2] = (1. / 9) * m0 + (1. / 6) * r1 * m2 - (1. / 36) * r2 * m3 - (1. / 6) * r3 * m5 - (1. / 18) * r4 * m6 - .25 * r2 * m7;
        to_return[3] = (1. / 9) * m0 - (1. / 6) * r1 * m1 - (1. / 36) * r2 * m3 + (1. / 6) * r3 * m4 - (1. / 18) * r4 * m6 + .25 * r2 * m7;
        to_return[4] = (1. / 9) * m0 - (1. / 6) * r1 * m2 - (1. / 36) * r2 * m3 + (1. / 6) * r3 * m5 - (1. / 18) * r4 * m6 - .25 * r2 * m7;
        to_return[5] = (1. / 9) * m0 + (1. / 6) * r1 * m1 + (1. / 6) * r1 * m2 + (1. / 18) * r2 * m3 + (1. / 12) * r3 * m4
                     + (1. / 12) * r3 * m5 + (1. / 36) * r4 * m6 + .25 * r2 * m8;
        to_return[6] = (1. / 9) * m0 - (1. / 6) * r1 * m1 + (1. / 6) * r1 * m2 + (1. / 18) * r2 * m3 - (1. / 12) * r3 * m4
                     + (1. / 12) * r3 * m5 + (1. / 36) * r4 * m6 - .25 * r2 * m8;
        to_return[7] = (1. / 9) * m0 - (1. / 6) * r1 * m1 - (1. / 6) * r1 * m2 + (1. / 18) * r2 * m3 - (1. / 12) * r3 * m4
                     - (1. / 12) * r3 * m5 + (1. / 36) * r4 * m6 + .25 * r2 * m8;
        to_return[8] = (1. / 9) * m0 + (1. / 6) * r1 * m1 - (1. / 6) * r1 * m2 + (1. / 18) * r2 * m3 + (1. / 12) * r3 * m4
                     - (1. / 12) * r3 * m5 + (1. / 36) * r4 * m6 - .25 * r2 * m8;
    }

    return to_return;
}

template <class Config>
auto init_f(samurai::MROMesh<Config>& mesh, const double radius, double rho0, double u0, double lambda, std::string& momenti)
{
    using mesh_id_t            = typename samurai::MROMesh<Config>::mesh_id_t;
    constexpr std::size_t nvel = 9;

    auto f = samurai::make_field<double, nvel>("f", mesh);
    f.fill(0);

    samurai::for_each_cell(mesh[mesh_id_t::cells],
                           [&](auto& cell)
                           {
                               auto center = cell.center();
                               auto x      = center[0];
                               auto y      = center[1];

                               double rho = inside_obstacle(x, y, radius) ? rho0 : rho0;
                               double qx  = inside_obstacle(x, y, radius) ? 0. : rho0 * u0;
                               double qy  = inside_obstacle(x, y, radius) ? 0. : 0.01 * rho0 * u0; // 0.; // TO start instability earlier

                               double r1 = 1.0 / lambda;
                               double r2 = 1.0 / (lambda * lambda);
                               double r3 = 1.0 / (lambda * lambda * lambda);
                               double r4 = 1.0 / (lambda * lambda * lambda * lambda);

                               double cs2 = (lambda * lambda) / 3.0; // Sound velocity of the lattice squared

                               if (!momenti.compare(std::string("Geier")))
                               {
                                   // This is the Geier choice of momenti
                                   double m0 = rho;
                                   double m1 = qx;
                                   double m2 = qy;
                                   double m3 = (qx * qx + qy * qy) / rho + 2. * rho * cs2;
                                   double m4 = qx * (cs2 + (qy / rho) * (qy / rho));
                                   double m5 = qy * (cs2 + (qx / rho) * (qx / rho));
                                   double m6 = rho * (cs2 + (qx / rho) * (qx / rho)) * (cs2 + (qy / rho) * (qy / rho));
                                   double m7 = (qx * qx - qy * qy) / rho;
                                   double m8 = qx * qy / rho;

                                   // We come back to the distributions
                                   f[cell][0] = m0 - r2 * m3 + r4 * m6;
                                   f[cell][1] = .5 * r1 * m1 + .25 * r2 * m3 - .5 * r3 * m4 - .5 * r4 * m6 + .25 * r2 * m7;
                                   f[cell][2] = .5 * r1 * m2 + .25 * r2 * m3 - .5 * r3 * m5 - .5 * r4 * m6 - .25 * r2 * m7;
                                   f[cell][3] = -.5 * r1 * m1 + .25 * r2 * m3 + .5 * r3 * m4 - .5 * r4 * m6 + .25 * r2 * m7;
                                   f[cell][4] = -.5 * r1 * m2 + .25 * r2 * m3 + .5 * r3 * m5 - .5 * r4 * m6 - .25 * r2 * m7;
                                   f[cell][5] = .25 * r3 * m4 + .25 * r3 * m5 + .25 * r4 * m6 + .25 * r2 * m8;
                                   f[cell][6] = -.25 * r3 * m4 + .25 * r3 * m5 + .25 * r4 * m6 - .25 * r2 * m8;
                                   f[cell][7] = -.25 * r3 * m4 - .25 * r3 * m5 + .25 * r4 * m6 + .25 * r2 * m8;
                                   f[cell][8] = .25 * r3 * m4 - .25 * r3 * m5 + .25 * r4 * m6 - .25 * r2 * m8;
                               }

                               if (!momenti.compare(std::string("Lallemand")))
                               {
                                   double m0 = rho;
                                   double m1 = qx;
                                   double m2 = qy;
                                   double m3 = -2 * lambda * lambda * rho + 3. / rho * (qx * qx + qy * qy);
                                   double m4 = -lambda * lambda * qx;
                                   double m5 = -lambda * lambda * qy;
                                   double m6 = lambda * lambda * lambda * lambda * rho - 3. * lambda * lambda / rho * (qx * qx + qy * qy);
                                   double m7 = (qx * qx - qy * qy) / rho;
                                   double m8 = qx * qy / rho;

                                   f[cell][0] = (1. / 9) * m0 - (1. / 9) * r2 * m3 + (1. / 9) * r4 * m6;
                                   f[cell][1] = (1. / 9) * m0 + (1. / 6) * r1 * m1 - (1. / 36) * r2 * m3 - (1. / 6) * r3 * m4
                                              - (1. / 18) * r4 * m6 + .25 * r2 * m7;
                                   f[cell][2] = (1. / 9) * m0 + (1. / 6) * r1 * m2 - (1. / 36) * r2 * m3 - (1. / 6) * r3 * m5
                                              - (1. / 18) * r4 * m6 - .25 * r2 * m7;
                                   f[cell][3] = (1. / 9) * m0 - (1. / 6) * r1 * m1 - (1. / 36) * r2 * m3 + (1. / 6) * r3 * m4
                                              - (1. / 18) * r4 * m6 + .25 * r2 * m7;
                                   f[cell][4] = (1. / 9) * m0 - (1. / 6) * r1 * m2 - (1. / 36) * r2 * m3 + (1. / 6) * r3 * m5
                                              - (1. / 18) * r4 * m6 - .25 * r2 * m7;
                                   f[cell][5] = (1. / 9) * m0 + (1. / 6) * r1 * m1 + (1. / 6) * r1 * m2 + (1. / 18) * r2 * m3
                                              + (1. / 12) * r3 * m4 + (1. / 12) * r3 * m5 + (1. / 36) * r4 * m6 + .25 * r2 * m8;
                                   f[cell][6] = (1. / 9) * m0 - (1. / 6) * r1 * m1 + (1. / 6) * r1 * m2 + (1. / 18) * r2 * m3
                                              - (1. / 12) * r3 * m4 + (1. / 12) * r3 * m5 + (1. / 36) * r4 * m6 - .25 * r2 * m8;
                                   f[cell][7] = (1. / 9) * m0 - (1. / 6) * r1 * m1 - (1. / 6) * r1 * m2 + (1. / 18) * r2 * m3
                                              - (1. / 12) * r3 * m4 - (1. / 12) * r3 * m5 + (1. / 36) * r4 * m6 + .25 * r2 * m8;
                                   f[cell][8] = (1. / 9) * m0 + (1. / 6) * r1 * m1 - (1. / 6) * r1 * m2 + (1. / 18) * r2 * m3
                                              + (1. / 12) * r3 * m4 - (1. / 12) * r3 * m5 + (1. / 36) * r4 * m6 - .25 * r2 * m8;
                               }
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

        // We have 9 velocity out of which 8 are moving
        // 4 are moving along the axis, thus needing only 2 fluxes each
        // (entering-exiting) and 4 along the diagonals, thus needing  6 fluxes

        // 4 * 2 + 4 * 6 = 8 + 24 = 32
        data[k].resize(32);

        // Parallel velocities
        for (int alpha = 0; alpha <= 3; ++alpha)
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

        // Diagonal velocities

        // Translation of the indices from which we start saving the new
        // computations
        int offset = 4 * 2;
        for (int alpha = 0; alpha <= 3; ++alpha)
        {
            // First side
            for (int l = 0; l < size - 1; ++l)
            {
                auto rotated_in  = rotation_of_pi_over_two(alpha, tau(k, i * size - 1), tau(k, j * size + l));
                auto rotated_out = rotation_of_pi_over_two(alpha, tau(k, (i + 1) * size - 1), tau(k, j * size + l));

                data[k][offset + 6 * alpha + 0] += prediction(k, tau_inverse(k, rotated_in.first), tau_inverse(k, rotated_in.second));
                data[k][offset + 6 * alpha + 3] += prediction(k, tau_inverse(k, rotated_out.first), tau_inverse(k, rotated_out.second));
            }
            // Cell on the diagonal
            {
                auto rotated_in  = rotation_of_pi_over_two(alpha, tau(k, i * size - 1), tau(k, j * size - 1));
                auto rotated_out = rotation_of_pi_over_two(alpha, tau(k, (i + 1) * size - 1), tau(k, (j + 1) * size - 1));

                data[k][offset + 6 * alpha + 1] += prediction(k, tau_inverse(k, rotated_in.first), tau_inverse(k, rotated_in.second));
                data[k][offset + 6 * alpha + 4] += prediction(k, tau_inverse(k, rotated_out.first), tau_inverse(k, rotated_out.second));
            }
            // Second side
            for (int l = 0; l < size - 1; ++l)
            {
                auto rotated_in  = rotation_of_pi_over_two(alpha, tau(k, i * size + l), tau(k, j * size - 1));
                auto rotated_out = rotation_of_pi_over_two(alpha, tau(k, i * size + l), tau(k, (j + 1) * size - 1));

                data[k][offset + 6 * alpha + 2] += prediction(k, tau_inverse(k, rotated_in.first), tau_inverse(k, rotated_in.second));
                data[k][offset + 6 * alpha + 5] += prediction(k, tau_inverse(k, rotated_out.first), tau_inverse(k, rotated_out.second));
            }
        }
    }
    return data;
}

// We have to average only the fluxes
template <class Field, class Func, class pred>
std::pair<double, double> one_time_step(Field& f,
                                        Func&& update_bc_for_level,
                                        const pred& pred_coeff,
                                        double rho0,
                                        double u0,
                                        double lambda,
                                        double mu,
                                        double zeta,
                                        double radius,
                                        std::string& momenti)
{
    constexpr std::size_t nvel = Field::size;
    using coord_index_t        = typename Field::interval_t::coord_index_t;

    double Fx = 0.; // Force on the obstacle along x
    double Fy = 0.; // Force on the obstacle along y

    auto mesh       = f.mesh();
    using mesh_id_t = typename decltype(mesh)::mesh_id_t;
    auto max_level  = mesh.max_level();
    auto min_level  = mesh.min_level();

    samurai::update_ghost_mr(f, std::forward<Func>(update_bc_for_level));
    samurai::update_overleaves_mr(f, std::forward<Func>(update_bc_for_level));

    Field new_f{"new_f", mesh};
    new_f.array().fill(0.);
    Field fluxes{"fluxes", mesh}; // This stored the fluxes computed at the level of the overleaves
    fluxes.array().fill(0.);
    Field advected{"advected", mesh};
    advected.array().fill(0.);

    auto inlet_condition = inlet_bc(rho0, u0, lambda, momenti);

    for (std::size_t level = min_level; level <= max_level; ++level)
    {
        if (level == max_level)
        {
            auto leaves_east = get_adjacent_boundary_east(mesh, max_level, mesh_id_t::cells);
            leaves_east.on(max_level)(
                [&](auto& interval, auto& index)
                {
                    auto k = interval; // Logical index in x
                    auto h = index[0]; // Logical index in y

                    advected(3, level, k, h) = f(3, level, k, h);
                    advected(6, level, k, h) = f(6, level, k, h - 1);
                    advected(7, level, k, h) = f(7, level, k, h + 1);
                    advected(0, level, k, h) = f(0, level, k, h);
                    advected(1, level, k, h) = f(1, level, k - 1, h);
                    advected(2, level, k, h) = f(2, level, k, h - 1);
                    advected(4, level, k, h) = f(4, level, k, h + 1);
                    advected(5, level, k, h) = f(5, level, k - 1, h - 1);
                    advected(8, level, k, h) = f(8, level, k - 1, h + 1);
                });

            auto leaves_north = get_adjacent_boundary_north(mesh, max_level, mesh_id_t::cells);
            leaves_north.on(max_level)(
                [&](auto& interval, auto& index)
                {
                    auto k = interval; // Logical index in x
                    auto h = index[0]; // Logical index in y

                    // // Working
                    advected(0, level, k, h) = f(0, level, k, h);
                    advected(1, level, k, h) = f(1, level, k - 1, h);
                    advected(2, level, k, h) = f(2, level, k, h - 1);
                    advected(3, level, k, h) = f(3, level, k + 1, h);
                    advected(4, level, k, h) = inlet_condition[4];
                    advected(5, level, k, h) = f(5, level, k - 1, h - 1);
                    advected(6, level, k, h) = f(6, level, k + 1, h - 1);
                    advected(7, level, k, h) = inlet_condition[7];
                    advected(8, level, k, h) = inlet_condition[8];
                });

            auto leaves_northeast = get_adjacent_boundary_northeast(mesh, max_level, mesh_id_t::cells);
            leaves_northeast.on(max_level)(
                [&](auto& interval, auto& index)
                {
                    auto k = interval; // Logical index in x
                    auto h = index[0]; // Logical index in y

                    // // Working
                    advected(0, level, k, h) = f(0, level, k, h);
                    advected(1, level, k, h) = f(1, level, k - 1, h);
                    advected(2, level, k, h) = f(2, level, k, h - 1);
                    advected(3, level, k, h) = f(3, level, k, h);
                    advected(4, level, k, h) = inlet_condition[4];
                    advected(5, level, k, h) = f(5, level, k - 1, h - 1);
                    advected(6, level, k, h) = f(6, level, k, h);
                    advected(7, level, k, h) = inlet_condition[7];
                    advected(8, level, k, h) = inlet_condition[8];
                });

            auto leaves_west = get_adjacent_boundary_west(mesh, max_level, mesh_id_t::cells);
            leaves_west.on(max_level)(
                [&](auto& interval, auto& index)
                {
                    auto k = interval; // Logical index in x
                    auto h = index[0]; // Logical index in y

                    advected(0, level, k, h) = f(0, level, k, h);
                    advected(1, level, k, h) = inlet_condition[1];
                    advected(2, level, k, h) = f(2, level, k, h - 1);
                    advected(3, level, k, h) = f(3, level, k + 1, h);
                    advected(4, level, k, h) = f(4, level, k, h + 1);
                    advected(5, level, k, h) = inlet_condition[5];
                    advected(6, level, k, h) = f(6, level, k + 1, h - 1);
                    advected(7, level, k, h) = f(7, level, k + 1, h + 1);
                    advected(8, level, k, h) = inlet_condition[8];
                });

            auto leaves_northwest = get_adjacent_boundary_northwest(mesh, max_level, mesh_id_t::cells);
            leaves_northwest.on(max_level)(
                [&](auto& interval, auto& index)
                {
                    auto k = interval; // Logical index in x
                    auto h = index[0]; // Logical index in y

                    // // Working
                    advected(0, level, k, h) = f(0, level, k, h);
                    advected(1, level, k, h) = inlet_condition[1];
                    advected(2, level, k, h) = f(2, level, k, h - 1);
                    advected(3, level, k, h) = f(3, level, k + 1, h);
                    advected(4, level, k, h) = inlet_condition[4];
                    advected(5, level, k, h) = inlet_condition[5];
                    advected(6, level, k, h) = f(6, level, k + 1, h - 1);
                    advected(7, level, k, h) = inlet_condition[7];
                    advected(8, level, k, h) = inlet_condition[8];
                });

            auto leaves_south = get_adjacent_boundary_south(mesh, max_level, mesh_id_t::cells);
            leaves_south.on(max_level)(
                [&](auto& interval, auto& index)
                {
                    auto k = interval; // Logical index in x
                    auto h = index[0]; // Logical index in y

                    // // Working
                    advected(0, level, k, h) = f(0, level, k, h);
                    advected(1, level, k, h) = f(1, level, k - 1, h);
                    advected(2, level, k, h) = inlet_condition[2];
                    advected(3, level, k, h) = f(3, level, k + 1, h);
                    advected(4, level, k, h) = f(4, level, k, h + 1);
                    advected(5, level, k, h) = inlet_condition[5];
                    advected(6, level, k, h) = inlet_condition[6];
                    advected(7, level, k, h) = f(7, level, k + 1, h + 1);
                    advected(8, level, k, h) = f(8, level, k - 1, h + 1);
                });

            auto leaves_southwest = get_adjacent_boundary_southwest(mesh, max_level, mesh_id_t::cells);
            leaves_southwest.on(max_level)(
                [&](auto& interval, auto& index)
                {
                    auto k = interval; // Logical index in x
                    auto h = index[0]; // Logical index in y

                    // // Working
                    advected(0, level, k, h) = f(0, level, k, h);
                    advected(1, level, k, h) = inlet_condition[1];
                    advected(2, level, k, h) = inlet_condition[2];
                    advected(3, level, k, h) = f(3, level, k + 1, h);
                    advected(4, level, k, h) = f(4, level, k, h + 1);
                    advected(5, level, k, h) = inlet_condition[5];
                    advected(6, level, k, h) = inlet_condition[6];
                    advected(7, level, k, h) = f(7, level, k + 1, h + 1);
                    advected(8, level, k, h) = inlet_condition[8];
                });

            auto leaves_southeast = get_adjacent_boundary_southeast(mesh, max_level, mesh_id_t::cells);
            leaves_southeast.on(max_level)(
                [&](auto& interval, auto& index)
                {
                    auto k = interval; // Logical index in x
                    auto h = index[0]; // Logical index in y

                    // // Working
                    advected(0, level, k, h) = f(0, level, k, h);
                    advected(1, level, k, h) = f(1, level, k - 1, h);
                    advected(2, level, k, h) = inlet_condition[2];
                    advected(3, level, k, h) = f(3, level, k, h);
                    advected(4, level, k, h) = f(4, level, k, h + 1);
                    advected(5, level, k, h) = inlet_condition[5];
                    advected(6, level, k, h) = inlet_condition[6];
                    advected(7, level, k, h) = f(7, level, k, h);
                    advected(8, level, k, h) = f(8, level, k - 1, h + 1);
                });

            // Advection far from the boundary
            auto tmp1                = union_(union_(union_(leaves_east, leaves_north), leaves_west), leaves_south);
            auto tmp2                = union_(union_(union_(leaves_northeast, leaves_northwest), leaves_southwest), leaves_southeast);
            auto all_leaves_boundary = union_(tmp1, tmp2);
            auto internal_leaves     = samurai::difference(mesh[mesh_id_t::cells][max_level],
                                                       all_leaves_boundary)
                                       .on(max_level); // It is very important to project
                                                       // at this point

            internal_leaves(
                [&](auto& interval, auto& index)
                {
                    auto k = interval; // Logical index in x
                    auto h = index[0]; // Logical index in y

                    advected(0, level, k, h) = f(0, level, k, h);
                    advected(1, level, k, h) = f(1, level, k - 1, h);
                    advected(2, level, k, h) = f(2, level, k, h - 1);
                    advected(3, level, k, h) = f(3, level, k + 1, h);
                    advected(4, level, k, h) = f(4, level, k, h + 1);
                    advected(5, level, k, h) = f(5, level, k - 1, h - 1);
                    advected(6, level, k, h) = f(6, level, k + 1, h - 1);
                    advected(7, level, k, h) = f(7, level, k + 1, h + 1);
                    advected(8, level, k, h) = f(8, level, k - 1, h + 1);
                });
        }
        else
        {
            std::size_t j = max_level - (level + 1);
            double coeff  = 1. / (1 << (2 * j)); // ATTENTION A LA DIMENSION 2 !!!!

            // Touching west
            auto overleaves_west = intersection(get_adjacent_boundary_west(mesh, level + 1, mesh_id_t::overleaves),
                                                mesh[mesh_id_t::cells][level]);

            auto overleaves_northwest = intersection(get_adjacent_boundary_northwest(mesh, level + 1, mesh_id_t::overleaves),
                                                     mesh[mesh_id_t::cells][level]);

            auto overleaves_southwest = intersection(get_adjacent_boundary_southwest(mesh, level + 1, mesh_id_t::overleaves),
                                                     mesh[mesh_id_t::cells][level]);

            auto touching_west = union_(union_(overleaves_west, overleaves_northwest), overleaves_southwest);

            touching_west.on(level + 1)(
                [&](auto& interval, auto& index)
                {
                    auto k = interval; // Logical index in x
                    auto h = index[0]; // Logical index in y

                    std::array<int, 3> flx_num{4, 16, 20};
                    std::array<int, 3> flx_vel{3, 6, 7};

                    for (int idx = 0; idx < flx_num.size(); ++idx)
                    {
                        for (auto& c : pred_coeff[j][flx_num[idx]].coeff)
                        {
                            coord_index_t stencil_x, stencil_y;
                            std::tie(stencil_x, stencil_y) = c.first;

                            fluxes(flx_vel[idx], level + 1, k, h) += coeff * c.second
                                                                   * f(flx_vel[idx], level + 1, k + stencil_x, h + stencil_y);
                        }
                    }
                });

            overleaves_west.on(level + 1)(
                [&](auto& interval, auto& index)
                {
                    auto k = interval; // Logical index in x
                    auto h = index[0]; // Logical index in y

                    // Regular
                    std::array<int, 8> flx_num{2, 6, 10, 14, 15, 21, 22, 26};
                    std::array<int, 8> flx_vel{2, 4, 5, 6, 6, 7, 7, 8};

                    for (int idx = 0; idx < flx_num.size(); ++idx)
                    {
                        for (auto& c : pred_coeff[j][flx_num[idx]].coeff)
                        {
                            coord_index_t stencil_x, stencil_y;
                            std::tie(stencil_x, stencil_y) = c.first;

                            fluxes(flx_vel[idx], level + 1, k, h) += coeff * c.second
                                                                   * f(flx_vel[idx], level + 1, k + stencil_x, h + stencil_y);
                        }
                    }
                    // h = 1
                    fluxes(1, level + 1, k, h) += (1 << j) * coeff * inlet_condition[1];
                    // h = 5
                    fluxes(5, level + 1, k, h) += (1 << j) * coeff * inlet_condition[5];
                    // h = 8
                    fluxes(8, level + 1, k, h) += (1 << j) * coeff * inlet_condition[8];
                });

            overleaves_northwest.on(level + 1)(
                [&](auto& interval, auto& index)
                {
                    auto k = interval; // Logical index in x
                    auto h = index[0]; // Logical index in y

                    // Regular
                    std::array<int, 4> flx_num{2, 10, 14, 15};
                    std::array<int, 4> flx_vel{2, 5, 6, 6};

                    for (int idx = 0; idx < flx_num.size(); ++idx)
                    {
                        for (auto& c : pred_coeff[j][flx_num[idx]].coeff)
                        {
                            coord_index_t stencil_x, stencil_y;
                            std::tie(stencil_x, stencil_y) = c.first;

                            fluxes(flx_vel[idx], level + 1, k, h) += coeff * c.second
                                                                   * f(flx_vel[idx], level + 1, k + stencil_x, h + stencil_y);
                        }
                    }
                    // // Worked
                    // h = 1
                    fluxes(1, level + 1, k, h) += (1 << j) * coeff * (inlet_condition[1]);
                    // h = 4
                    fluxes(4, level + 1, k, h) += (1 << j) * coeff * (inlet_condition[4]);
                    // h = 5
                    fluxes(5, level + 1, k, h) += (1 << j) * coeff * (inlet_condition[5]);
                    // h = 7
                    fluxes(7, level + 1, k, h) += (1 << j) * coeff * (inlet_condition[7]);
                    // h = 8
                    fluxes(8, level + 1, k, h) += (2 * (1 << j) - 1) * coeff * (inlet_condition[8]);
                });

            overleaves_southwest.on(level + 1)(
                [&](auto& interval, auto& index)
                {
                    auto k = interval; // Logical index in x
                    auto h = index[0]; // Logical index in y

                    // Regular
                    std::array<int, 4> flx_num{6, 21, 22, 26};
                    std::array<int, 4> flx_vel{4, 7, 7, 8};

                    for (int idx = 0; idx < flx_num.size(); ++idx)
                    {
                        for (auto& c : pred_coeff[j][flx_num[idx]].coeff)
                        {
                            coord_index_t stencil_x, stencil_y;
                            std::tie(stencil_x, stencil_y) = c.first;

                            fluxes(flx_vel[idx], level + 1, k, h) += coeff * c.second
                                                                   * f(flx_vel[idx], level + 1, k + stencil_x, h + stencil_y);
                        }
                    }
                    // Worked
                    // h = 1
                    fluxes(1, level + 1, k, h) += (1 << j) * coeff * (inlet_condition[1]);
                    // h = 2
                    fluxes(2, level + 1, k, h) += (1 << j) * coeff * (inlet_condition[2]);
                    // h = 5
                    fluxes(5, level + 1, k, h) += (2 * (1 << j) - 1) * coeff * (inlet_condition[5]);
                    // h = 6
                    fluxes(6, level + 1, k, h) += (1 << j) * coeff * (inlet_condition[6]);
                    // h = 8
                    fluxes(8, level + 1, k, h) += (1 << j) * coeff * (inlet_condition[8]);
                });

            // This is necessary because the only overleaves we have to advect
            // on are the ones superposed with the leaves to which we come back
            // eventually in the process
            auto overleaves_east = intersection(get_adjacent_boundary_east(mesh, level + 1, mesh_id_t::overleaves),
                                                mesh[mesh_id_t::cells][level]);

            auto overleaves_northeast = intersection(get_adjacent_boundary_northeast(mesh, level + 1, mesh_id_t::overleaves),
                                                     mesh[mesh_id_t::cells][level]);

            auto overleaves_southeast = intersection(get_adjacent_boundary_southeast(mesh, level + 1, mesh_id_t::overleaves),
                                                     mesh[mesh_id_t::cells][level]);

            auto touching_east = union_(union_(overleaves_east, overleaves_northeast), overleaves_southeast);

            touching_east.on(level + 1)(
                [&](auto& interval, auto& index)
                {
                    auto k = interval; // Logical index in x
                    auto h = index[0]; // Logical index in y

                    std::array<int, 3> flx_num{0, 8, 28};
                    std::array<int, 3> flx_vel{1, 5, 8};

                    for (int idx = 0; idx < flx_num.size(); ++idx)
                    {
                        for (auto& c : pred_coeff[j][flx_num[idx]].coeff)
                        {
                            coord_index_t stencil_x, stencil_y;
                            std::tie(stencil_x, stencil_y) = c.first;

                            fluxes(flx_vel[idx], level + 1, k, h) += coeff * c.second
                                                                   * f(flx_vel[idx], level + 1, k + stencil_x, h + stencil_y);
                        }
                    }
                });

            overleaves_east.on(level + 1)(
                [&](auto& interval, auto& index)
                {
                    auto k = interval; // Logical index in x
                    auto h = index[0]; // Logical index in y

                    // Regular
                    std::array<int, 8> flx_num{2, 6, 9, 10, 14, 22, 26, 27};
                    std::array<int, 8> flx_vel{2, 4, 5, 5, 6, 7, 8, 8};

                    for (int idx = 0; idx < flx_num.size(); ++idx)
                    {
                        for (auto& c : pred_coeff[j][flx_num[idx]].coeff)
                        {
                            coord_index_t stencil_x, stencil_y;
                            std::tie(stencil_x, stencil_y) = c.first;

                            fluxes(flx_vel[idx], level + 1, k, h) += coeff * c.second
                                                                   * f(flx_vel[idx], level + 1, k + stencil_x, h + stencil_y);
                        }
                    }
                    // h = 3
                    fluxes(3, level + 1, k, h) += (1 << j) * coeff * f(3, level + 1, k, h);
                    // h = 6
                    fluxes(6, level + 1, k, h) += (1 << j) * coeff * f(6, level + 1, k, h);
                    // h = 7
                    fluxes(7, level + 1, k, h) += (1 << j) * coeff * f(7, level + 1, k, h);
                });

            overleaves_northeast.on(level + 1)(
                [&](auto& interval, auto& index)
                {
                    auto k = interval; // Logical index in x
                    auto h = index[0]; // Logical index in y

                    // Regular
                    std::array<int, 4> flx_num{2, 9, 10, 14};
                    std::array<int, 4> flx_vel{2, 5, 5, 6};

                    for (int idx = 0; idx < flx_num.size(); ++idx)
                    {
                        for (auto& c : pred_coeff[j][flx_num[idx]].coeff)
                        {
                            coord_index_t stencil_x, stencil_y;
                            std::tie(stencil_x, stencil_y) = c.first;

                            fluxes(flx_vel[idx], level + 1, k, h) += coeff * c.second
                                                                   * f(flx_vel[idx], level + 1, k + stencil_x, h + stencil_y);
                        }
                    }
                    // // Worked
                    // h = 3
                    fluxes(3, level + 1, k, h) += (1 << j) * coeff * f(3, level + 1, k, h);
                    // h = 4
                    fluxes(4, level + 1, k, h) += (1 << j) * coeff * (inlet_condition[4]);
                    // h = 6
                    fluxes(6, level + 1, k, h) += (1 << j) * coeff * f(6, level + 1, k, h);
                    // h = 7
                    fluxes(7, level + 1, k, h) += (1 << j) * coeff * (inlet_condition[7]);
                    fluxes(7, level + 1, k, h) += ((1 << j) - 1) * coeff * f(7, level + 1, k, h);
                    // h = 8
                    fluxes(8, level + 1, k, h) += (1 << j) * coeff * (inlet_condition[8]);
                });

            overleaves_southeast.on(level + 1)(
                [&](auto& interval, auto& index)
                {
                    auto k = interval; // Logical index in x
                    auto h = index[0]; // Logical index in y

                    // Regular
                    std::array<int, 4> flx_num{6, 22, 26, 27};
                    std::array<int, 4> flx_vel{4, 7, 8, 8};

                    for (int idx = 0; idx < flx_num.size(); ++idx)
                    {
                        for (auto& c : pred_coeff[j][flx_num[idx]].coeff)
                        {
                            coord_index_t stencil_x, stencil_y;
                            std::tie(stencil_x, stencil_y) = c.first;

                            fluxes(flx_vel[idx], level + 1, k, h) += coeff * c.second
                                                                   * f(flx_vel[idx], level + 1, k + stencil_x, h + stencil_y);
                        }
                    }
                    // Worked
                    // h = 2
                    fluxes(2, level + 1, k, h) += (1 << j) * coeff * (inlet_condition[2]);
                    // h = 3
                    fluxes(3, level + 1, k, h) += (1 << j) * coeff * f(3, level + 1, k, h);
                    // h = 5
                    fluxes(5, level + 1, k, h) += (1 << j) * coeff * (inlet_condition[5]);
                    // h = 6
                    fluxes(6, level + 1, k, h) += ((1 << j)) * coeff * (inlet_condition[6]);
                    fluxes(6, level + 1, k, h) += ((1 << j) - 1) * coeff * f(6, level + 1, k, h);
                    // h = 7
                    fluxes(7, level + 1, k, h) += (1 << j) * coeff * f(7, level + 1, k, h);
                });

            auto overleaves_south = intersection(get_adjacent_boundary_south(mesh, level + 1, mesh_id_t::overleaves),
                                                 mesh[mesh_id_t::cells][level]);

            auto overleaves_north = intersection(get_adjacent_boundary_north(mesh, level + 1, mesh_id_t::overleaves),
                                                 mesh[mesh_id_t::cells][level]);

            auto north_and_south = union_(overleaves_south, overleaves_north);

            overleaves_north.on(level + 1)(
                [&](auto& interval, auto& index)
                {
                    auto k = interval; // Logical index in x
                    auto h = index[0]; // Logical index in y

                    // Regular
                    std::array<int, 11> flx_num{0, 2, 4, 8, 9, 10, 14, 15, 16, 20, 28};
                    std::array<int, 11> flx_vel{1, 2, 3, 5, 5, 5, 6, 6, 6, 7, 8};

                    for (int idx = 0; idx < flx_num.size(); ++idx)
                    {
                        for (auto& c : pred_coeff[j][flx_num[idx]].coeff)
                        {
                            coord_index_t stencil_x, stencil_y;
                            std::tie(stencil_x, stencil_y) = c.first;

                            fluxes(flx_vel[idx], level + 1, k, h) += coeff * c.second
                                                                   * f(flx_vel[idx], level + 1, k + stencil_x, h + stencil_y);
                        }
                    }
                    // Worked
                    // h = 4
                    fluxes(4, level + 1, k, h) += (1 << j) * coeff * (inlet_condition[4]);
                    // h = 7
                    fluxes(7, level + 1, k, h) += (1 << j) * coeff * (inlet_condition[7]);
                    // h = 8
                    fluxes(8, level + 1, k, h) += (1 << j) * coeff * (inlet_condition[8]);
                });

            overleaves_south.on(level + 1)(
                [&](auto& interval, auto& index)
                {
                    auto k = interval; // Logical index in x
                    auto h = index[0]; // Logical index in y

                    // Regular
                    std::array<int, 11> flx_num{0, 4, 6, 8, 16, 20, 21, 22, 26, 27, 28};
                    std::array<int, 11> flx_vel{1, 3, 4, 5, 6, 7, 7, 7, 8, 8, 8};

                    for (int idx = 0; idx < flx_num.size(); ++idx)
                    {
                        for (auto& c : pred_coeff[j][flx_num[idx]].coeff)
                        {
                            coord_index_t stencil_x, stencil_y;
                            std::tie(stencil_x, stencil_y) = c.first;

                            fluxes(flx_vel[idx], level + 1, k, h) += coeff * c.second
                                                                   * f(flx_vel[idx], level + 1, k + stencil_x, h + stencil_y);
                        }
                    }
                    // // Worked
                    // h = 2
                    fluxes(2, level + 1, k, h) += (1 << j) * coeff * (inlet_condition[2]);
                    // h = 5
                    fluxes(5, level + 1, k, h) += (1 << j) * coeff * (inlet_condition[5]);
                    // h = 6
                    fluxes(6, level + 1, k, h) += (1 << j) * coeff * (inlet_condition[6]);
                });

            // All the exiting fluxes are valid, thus we perform them
            // once for all
            auto all_overleaves = intersection(mesh[mesh_id_t::cells][level], mesh[mesh_id_t::cells][level]).on(level + 1);

            all_overleaves(
                [&](auto& interval, auto& index)
                {
                    auto k = interval; // Logical index in x
                    auto h = index[0]; // Logical index in y

                    std::array<int, 16> flx_num{1, 3, 5, 7, 11, 12, 13, 17, 18, 19, 23, 24, 25, 29, 30, 31};
                    std::array<int, 16> flx_vel{1, 2, 3, 4, 5, 5, 5, 6, 6, 6, 7, 7, 7, 8, 8, 8};

                    for (int idx = 0; idx < flx_num.size(); ++idx)
                    {
                        for (auto& c : pred_coeff[j][flx_num[idx]].coeff)
                        {
                            coord_index_t stencil_x, stencil_y;
                            std::tie(stencil_x, stencil_y) = c.first;

                            // Be careful about the - sign because we are
                            // dealing with exiting fluxes
                            fluxes(flx_vel[idx], level + 1, k, h) -= coeff * c.second
                                                                   * f(flx_vel[idx], level + 1, k + stencil_x, h + stencil_y);
                        }
                    }
                });

            auto overleaves_far_boundary = difference(mesh[mesh_id_t::cells][level],
                                                      union_(union_(touching_east, touching_west),
                                                             north_and_south))
                                               .on(level + 1); // Again, it is very important to project
                                                               // before using

            // We are just left to add the incoming fluxes to to the internal
            // overleaves
            overleaves_far_boundary(
                [&](auto& interval, auto& index)
                {
                    auto k = interval; // Logical index in x
                    auto h = index[0]; // Logical index in y

                    std::array<int, 16> flx_num{0, 2, 4, 6, 8, 9, 10, 14, 15, 16, 20, 21, 22, 26, 27, 28};
                    std::array<int, 16> flx_vel{1, 2, 3, 4, 5, 5, 5, 6, 6, 6, 7, 7, 7, 8, 8, 8};

                    for (int idx = 0; idx < flx_num.size(); ++idx)
                    {
                        for (auto& c : pred_coeff[j][flx_num[idx]].coeff)
                        {
                            coord_index_t stencil_x, stencil_y;
                            std::tie(stencil_x, stencil_y) = c.first;

                            fluxes(flx_vel[idx], level + 1, k, h) += coeff * c.second
                                                                   * f(flx_vel[idx], level + 1, k + stencil_x, h + stencil_y);
                        }
                    }
                });

            auto leaves = samurai::intersection(mesh[mesh_id_t::cells][level], mesh[mesh_id_t::cells][level]);

            leaves(
                [&](auto& interval, auto& index)
                {
                    auto k = interval; // Logical index in x
                    auto h = index[0]; // Logical index in y

                    advected(0, level, k, h) = f(0, level, k, h); // Not moving so no flux
                    for (int pop = 1; pop < 9; ++pop)
                    {
                        advected(pop, level, k, h) = f(pop, level, k, h)
                                                   + 0.25
                                                         * (fluxes(pop, level + 1, 2 * k, 2 * h) + fluxes(pop, level + 1, 2 * k + 1, 2 * h)
                                                            + fluxes(pop, level + 1, 2 * k, 2 * h + 1)
                                                            + fluxes(pop, level + 1, 2 * k + 1, 2 * h + 1));
                    }
                });
        }

        // Collision
        double l1 = lambda;
        double l2 = l1 * lambda;
        double l3 = l2 * lambda;
        double l4 = l3 * lambda;

        double r1 = 1.0 / lambda;
        double r2 = 1.0 / (lambda * lambda);
        double r3 = 1.0 / (lambda * lambda * lambda);
        double r4 = 1.0 / (lambda * lambda * lambda * lambda);

        auto leaves = samurai::intersection(mesh[mesh_id_t::cells][level], mesh[mesh_id_t::cells][level]);

        leaves(
            [&](auto& interval, auto& index)
            {
                auto k = interval; // Logical index in x
                auto h = index[0]; // Logical index in y

                if (!momenti.compare(std::string("Geier")))
                {
                    // // Choice of momenti by Geier

                    auto m0 = xt::eval(advected(0, level, k, h) + advected(1, level, k, h) + advected(2, level, k, h)
                                       + advected(3, level, k, h) + advected(4, level, k, h) + advected(5, level, k, h)
                                       + advected(6, level, k, h) + advected(7, level, k, h) + advected(8, level, k, h));
                    auto m1 = xt::eval(l1
                                       * (advected(1, level, k, h) - advected(3, level, k, h) + advected(5, level, k, h)
                                          - advected(6, level, k, h) - advected(7, level, k, h) + advected(8, level, k, h)));
                    auto m2 = xt::eval(l1
                                       * (advected(2, level, k, h) - advected(4, level, k, h) + advected(5, level, k, h)
                                          + advected(6, level, k, h) - advected(7, level, k, h) - advected(8, level, k, h)));
                    auto m3 = xt::eval(l2
                                       * (advected(1, level, k, h) + advected(2, level, k, h) + advected(3, level, k, h)
                                          + advected(4, level, k, h) + 2 * advected(5, level, k, h) + 2 * advected(6, level, k, h)
                                          + 2 * advected(7, level, k, h) + 2 * advected(8, level, k, h)));
                    auto m4 = xt::eval(
                        l3 * (advected(5, level, k, h) - advected(6, level, k, h) - advected(7, level, k, h) + advected(8, level, k, h)));
                    auto m5 = xt::eval(
                        l3 * (advected(5, level, k, h) + advected(6, level, k, h) - advected(7, level, k, h) - advected(8, level, k, h)));
                    auto m6 = xt::eval(
                        l4 * (advected(5, level, k, h) + advected(6, level, k, h) + advected(7, level, k, h) + advected(8, level, k, h)));
                    auto m7 = xt::eval(
                        l2 * (advected(1, level, k, h) - advected(2, level, k, h) + advected(3, level, k, h) - advected(4, level, k, h)));
                    auto m8 = xt::eval(
                        l2 * (advected(5, level, k, h) - advected(6, level, k, h) + advected(7, level, k, h) - advected(8, level, k, h)));

                    double space_step = 1.0 / (1 << max_level);
                    double dummy      = 3.0 / (lambda * rho0 * space_step);

                    double cs2     = (lambda * lambda) / 3.0; // sound velocity squared
                    double sigma_1 = dummy * (zeta - 2. * mu / 3.);
                    double sigma_2 = dummy * mu;
                    double s_1     = 1 / (.5 + sigma_1);
                    double s_2     = 1 / (.5 + sigma_2);

                    m3 = (1. - s_1) * m3 + s_1 * ((m1 * m1 + m2 * m2) / m0 + 2. * m0 * cs2);
                    m4 = (1. - s_1) * m4 + s_1 * (m1 * (cs2 + (m2 / m0) * (m2 / m0)));
                    m5 = (1. - s_1) * m5 + s_1 * (m2 * (cs2 + (m1 / m0) * (m1 / m0)));
                    m6 = (1. - s_1) * m6 + s_1 * (m0 * (cs2 + (m1 / m0) * (m1 / m0)) * (cs2 + (m2 / m0) * (m2 / m0)));
                    m7 = (1. - s_2) * m7 + s_2 * ((m1 * m1 - m2 * m2) / m0);
                    m8 = (1. - s_2) * m8 + s_2 * (m1 * m2 / m0);

                    new_f(0, level, k, h) = m0 - r2 * m3 + r4 * m6;
                    new_f(1, level, k, h) = .5 * r1 * m1 + .25 * r2 * m3 - .5 * r3 * m4 - .5 * r4 * m6 + .25 * r2 * m7;
                    new_f(2, level, k, h) = .5 * r1 * m2 + .25 * r2 * m3 - .5 * r3 * m5 - .5 * r4 * m6 - .25 * r2 * m7;
                    new_f(3, level, k, h) = -.5 * r1 * m1 + .25 * r2 * m3 + .5 * r3 * m4 - .5 * r4 * m6 + .25 * r2 * m7;
                    new_f(4, level, k, h) = -.5 * r1 * m2 + .25 * r2 * m3 + .5 * r3 * m5 - .5 * r4 * m6 - .25 * r2 * m7;
                    new_f(5, level, k, h) = .25 * r3 * m4 + .25 * r3 * m5 + .25 * r4 * m6 + .25 * r2 * m8;
                    new_f(6, level, k, h) = -.25 * r3 * m4 + .25 * r3 * m5 + .25 * r4 * m6 - .25 * r2 * m8;
                    new_f(7, level, k, h) = -.25 * r3 * m4 - .25 * r3 * m5 + .25 * r4 * m6 + .25 * r2 * m8;
                    new_f(8, level, k, h) = .25 * r3 * m4 - .25 * r3 * m5 + .25 * r4 * m6 - .25 * r2 * m8;
                }
                if (!momenti.compare(std::string("Lallemand")))
                {
                    auto m0 = xt::eval(advected(0, level, k, h) + advected(1, level, k, h) + advected(2, level, k, h)
                                       + advected(3, level, k, h) + advected(4, level, k, h) + advected(5, level, k, h)
                                       + advected(6, level, k, h) + advected(7, level, k, h) + advected(8, level, k, h));
                    auto m1 = xt::eval(l1
                                       * (advected(1, level, k, h) - advected(3, level, k, h) + advected(5, level, k, h)
                                          - advected(6, level, k, h) - advected(7, level, k, h) + advected(8, level, k, h)));
                    auto m2 = xt::eval(l1
                                       * (advected(2, level, k, h) - advected(4, level, k, h) + advected(5, level, k, h)
                                          + advected(6, level, k, h) - advected(7, level, k, h) - advected(8, level, k, h)));
                    auto m3 = xt::eval(l2
                                       * (-4 * advected(0, level, k, h) - advected(1, level, k, h) - advected(2, level, k, h)
                                          - advected(3, level, k, h) - advected(4, level, k, h) + 2 * advected(5, level, k, h)
                                          + 2 * advected(6, level, k, h) + 2 * advected(7, level, k, h) + 2 * advected(8, level, k, h)));
                    auto m4 = xt::eval(l3
                                       * (-2 * advected(1, level, k, h) + 2 * advected(3, level, k, h) + advected(5, level, k, h)
                                          - advected(6, level, k, h) - advected(7, level, k, h) + advected(8, level, k, h)));
                    auto m5 = xt::eval(l3
                                       * (-2 * advected(2, level, k, h) + 2 * advected(4, level, k, h) + advected(5, level, k, h)
                                          + advected(6, level, k, h) - advected(7, level, k, h) - advected(8, level, k, h)));
                    auto m6 = xt::eval(l4
                                       * (4 * advected(0, level, k, h) - 2 * advected(1, level, k, h) - 2 * advected(2, level, k, h)
                                          - 2 * advected(3, level, k, h) - 2 * advected(4, level, k, h) + advected(5, level, k, h)
                                          + advected(6, level, k, h) + advected(7, level, k, h) + advected(8, level, k, h)));
                    auto m7 = xt::eval(
                        l2 * (advected(1, level, k, h) - advected(2, level, k, h) + advected(3, level, k, h) - advected(4, level, k, h)));
                    auto m8 = xt::eval(
                        l2 * (advected(5, level, k, h) - advected(6, level, k, h) + advected(7, level, k, h) - advected(8, level, k, h)));

                    double space_step = 1.0 / (1 << max_level);
                    double dummy      = 3.0 / (lambda * rho0 * space_step);

                    double cs2     = (lambda * lambda) / 3.0; // sound velocity squared
                    double sigma_1 = dummy * zeta;
                    double sigma_2 = dummy * mu;
                    double s_1     = 1 / (.5 + sigma_1);
                    double s_2     = 1 / (.5 + sigma_2);

                    m3 = (1. - s_1) * m3 + s_1 * (-2 * lambda * lambda * m0 + 3. / m0 * (m1 * m1 + m2 * m2));
                    m4 = (1. - s_1) * m4 + s_1 * (-lambda * lambda * m1);
                    m5 = (1. - s_1) * m5 + s_1 * (-lambda * lambda * m2);
                    m6 = (1. - s_1) * m6 + s_1 * (lambda * lambda * lambda * lambda * m0 - 3. * lambda * lambda / m0 * (m1 * m1 + m2 * m2));
                    m7 = (1. - s_2) * m7 + s_2 * ((m1 * m1 - m2 * m2) / m0);
                    m8 = (1. - s_2) * m8 + s_2 * (m1 * m2 / m0);

                    new_f(0, level, k, h) = (1. / 9) * m0 - (1. / 9) * r2 * m3 + (1. / 9) * r4 * m6;
                    new_f(1, level, k, h) = (1. / 9) * m0 + (1. / 6) * r1 * m1 - (1. / 36) * r2 * m3 - (1. / 6) * r3 * m4
                                          - (1. / 18) * r4 * m6 + .25 * r2 * m7;
                    new_f(2, level, k, h) = (1. / 9) * m0 + (1. / 6) * r1 * m2 - (1. / 36) * r2 * m3 - (1. / 6) * r3 * m5
                                          - (1. / 18) * r4 * m6 - .25 * r2 * m7;
                    new_f(3, level, k, h) = (1. / 9) * m0 - (1. / 6) * r1 * m1 - (1. / 36) * r2 * m3 + (1. / 6) * r3 * m4
                                          - (1. / 18) * r4 * m6 + .25 * r2 * m7;
                    new_f(4, level, k, h) = (1. / 9) * m0 - (1. / 6) * r1 * m2 - (1. / 36) * r2 * m3 + (1. / 6) * r3 * m5
                                          - (1. / 18) * r4 * m6 - .25 * r2 * m7;
                    new_f(5, level, k, h) = (1. / 9) * m0 + (1. / 6) * r1 * m1 + (1. / 6) * r1 * m2 + (1. / 18) * r2 * m3
                                          + (1. / 12) * r3 * m4 + (1. / 12) * r3 * m5 + (1. / 36) * r4 * m6 + .25 * r2 * m8;
                    new_f(6, level, k, h) = (1. / 9) * m0 - (1. / 6) * r1 * m1 + (1. / 6) * r1 * m2 + (1. / 18) * r2 * m3
                                          - (1. / 12) * r3 * m4 + (1. / 12) * r3 * m5 + (1. / 36) * r4 * m6 - .25 * r2 * m8;
                    new_f(7, level, k, h) = (1. / 9) * m0 - (1. / 6) * r1 * m1 - (1. / 6) * r1 * m2 + (1. / 18) * r2 * m3
                                          - (1. / 12) * r3 * m4 - (1. / 12) * r3 * m5 + (1. / 36) * r4 * m6 + .25 * r2 * m8;
                    new_f(8, level, k, h) = (1. / 9) * m0 + (1. / 6) * r1 * m1 - (1. / 6) * r1 * m2 + (1. / 18) * r2 * m3
                                          + (1. / 12) * r3 * m4 - (1. / 12) * r3 * m5 + (1. / 36) * r4 * m6 - .25 * r2 * m8;
                }
            });

        // Enforcing the obstacle condition
        samurai::for_each_cell(
            mesh[mesh_id_t::cells],
            [&](auto& cell)
            {
                auto center = cell.center();
                auto x      = center[0];
                auto y      = center[1];

                double dx = cell.length;

                double rho = rho0;
                double qx  = 0.;
                double qy  = 0.;

                // This is the Geier choice of momenti
                double cs2 = (lambda * lambda) / 3.0; // Sound velocity of the lattice squared

                if (!momenti.compare(std::string("Geier")))
                {
                    double m0 = rho;
                    double m1 = qx;
                    double m2 = qy;
                    double m3 = (qx * qx + qy * qy) / rho + 2. * rho * cs2;
                    double m4 = qx * (cs2 + (qy / rho) * (qy / rho));
                    double m5 = qy * (cs2 + (qx / rho) * (qx / rho));
                    double m6 = rho * (cs2 + (qx / rho) * (qx / rho)) * (cs2 + (qy / rho) * (qy / rho));
                    double m7 = (qx * qx - qy * qy) / rho;
                    double m8 = qx * qy / rho;

                    // The cell is fully inside the obstacle
                    if (inside_obstacle(x - .5 * dx, y - .5 * dx, radius) && inside_obstacle(x + .5 * dx, y - .5 * dx, radius)
                        && inside_obstacle(x + .5 * dx, y + .5 * dx, radius) && inside_obstacle(x - .5 * dx, y + .5 * dx, radius))
                    {
                        new_f[cell][0] = m0 - r2 * m3 + r4 * m6;
                        new_f[cell][1] = .5 * r1 * m1 + .25 * r2 * m3 - .5 * r3 * m4 - .5 * r4 * m6 + .25 * r2 * m7;
                        new_f[cell][2] = .5 * r1 * m2 + .25 * r2 * m3 - .5 * r3 * m5 - .5 * r4 * m6 - .25 * r2 * m7;
                        new_f[cell][3] = -.5 * r1 * m1 + .25 * r2 * m3 + .5 * r3 * m4 - .5 * r4 * m6 + .25 * r2 * m7;
                        new_f[cell][4] = -.5 * r1 * m2 + .25 * r2 * m3 + .5 * r3 * m5 - .5 * r4 * m6 - .25 * r2 * m7;
                        new_f[cell][5] = .25 * r3 * m4 + .25 * r3 * m5 + .25 * r4 * m6 + .25 * r2 * m8;
                        new_f[cell][6] = -.25 * r3 * m4 + .25 * r3 * m5 + .25 * r4 * m6 - .25 * r2 * m8;
                        new_f[cell][7] = -.25 * r3 * m4 - .25 * r3 * m5 + .25 * r4 * m6 + .25 * r2 * m8;
                        new_f[cell][8] = .25 * r3 * m4 - .25 * r3 * m5 + .25 * r4 * m6 - .25 * r2 * m8;
                    }
                    else
                    {
                        // The cell has the interface cutting through it
                        if (inside_obstacle(x - .5 * dx, y - .5 * dx, radius) || inside_obstacle(x + .5 * dx, y - .5 * dx, radius)
                            || inside_obstacle(x + .5 * dx, y + .5 * dx, radius) || inside_obstacle(x - .5 * dx, y + .5 * dx, radius))
                        {
                            // We compute the volume fraction
                            double vol_fraction = volume_inside_obstacle_estimation(x - .5 * dx, y - .5 * dx, dx, radius);
                            double len_boundary = length_obstacle_inside_cell(x - .5 * dx, y - .5 * dx, dx, radius);

                            double dt = 1. / (1 << max_level) / lambda;
                            Fx += dx / dt * (1. / (1 << max_level)) * vol_fraction * lambda
                                * (new_f[cell][1] - new_f[cell][3] + new_f[cell][5] - new_f[cell][6] - new_f[cell][7] + new_f[cell][8]);
                            Fy += dx / dt * (1. / (1 << max_level)) * vol_fraction * lambda
                                * (new_f[cell][2] - new_f[cell][4] + new_f[cell][5] + new_f[cell][6] - new_f[cell][7] - new_f[cell][8]);

                            new_f[cell][0] = (1. - vol_fraction) * new_f[cell][0] + vol_fraction * (m0 - r2 * m3 + r4 * m6);
                            new_f[cell][1] = (1. - vol_fraction) * new_f[cell][1]
                                           + vol_fraction * (.5 * r1 * m1 + .25 * r2 * m3 - .5 * r3 * m4 - .5 * r4 * m6 + .25 * r2 * m7);
                            new_f[cell][2] = (1. - vol_fraction) * new_f[cell][2]
                                           + vol_fraction * (.5 * r1 * m2 + .25 * r2 * m3 - .5 * r3 * m5 - .5 * r4 * m6 - .25 * r2 * m7);
                            new_f[cell][3] = (1. - vol_fraction) * new_f[cell][3]
                                           + vol_fraction * (-.5 * r1 * m1 + .25 * r2 * m3 + .5 * r3 * m4 - .5 * r4 * m6 + .25 * r2 * m7);
                            new_f[cell][4] = (1. - vol_fraction) * new_f[cell][4]
                                           + vol_fraction * (-.5 * r1 * m2 + .25 * r2 * m3 + .5 * r3 * m5 - .5 * r4 * m6 - .25 * r2 * m7);
                            new_f[cell][5] = (1. - vol_fraction) * new_f[cell][5]
                                           + vol_fraction * (.25 * r3 * m4 + .25 * r3 * m5 + .25 * r4 * m6 + .25 * r2 * m8);
                            new_f[cell][6] = (1. - vol_fraction) * new_f[cell][6]
                                           + vol_fraction * (-.25 * r3 * m4 + .25 * r3 * m5 + .25 * r4 * m6 - .25 * r2 * m8);
                            new_f[cell][7] = (1. - vol_fraction) * new_f[cell][7]
                                           + vol_fraction * (-.25 * r3 * m4 - .25 * r3 * m5 + .25 * r4 * m6 + .25 * r2 * m8);
                            new_f[cell][8] = (1. - vol_fraction) * new_f[cell][8]
                                           + vol_fraction * (.25 * r3 * m4 - .25 * r3 * m5 + .25 * r4 * m6 - .25 * r2 * m8);
                        }
                    }
                }
                if (!momenti.compare(std::string("Lallemand")))
                {
                    double m0 = rho;
                    double m1 = qx;
                    double m2 = qy;
                    double m3 = -2 * lambda * lambda * rho + 3. / rho * (qx * qx + qy * qy);
                    double m4 = -lambda * lambda * qx;
                    double m5 = -lambda * lambda * qy;
                    double m6 = lambda * lambda * lambda * lambda * rho - 3. * lambda * lambda / rho * (qx * qx + qy * qy);
                    double m7 = (qx * qx - qy * qy) / rho;
                    double m8 = qx * qy / rho;

                    // The cell is fully inside the obstacle
                    if (inside_obstacle(x - .5 * dx, y - .5 * dx, radius) && inside_obstacle(x + .5 * dx, y - .5 * dx, radius)
                        && inside_obstacle(x + .5 * dx, y + .5 * dx, radius) && inside_obstacle(x - .5 * dx, y + .5 * dx, radius))
                    {
                        new_f[cell][0] = (1. / 9) * m0 - (1. / 9) * r2 * m3 + (1. / 9) * r4 * m6;
                        new_f[cell][1] = (1. / 9) * m0 + (1. / 6) * r1 * m1 - (1. / 36) * r2 * m3 - (1. / 6) * r3 * m4 - (1. / 18) * r4 * m6
                                       + .25 * r2 * m7;
                        new_f[cell][2] = (1. / 9) * m0 + (1. / 6) * r1 * m2 - (1. / 36) * r2 * m3 - (1. / 6) * r3 * m5 - (1. / 18) * r4 * m6
                                       - .25 * r2 * m7;
                        new_f[cell][3] = (1. / 9) * m0 - (1. / 6) * r1 * m1 - (1. / 36) * r2 * m3 + (1. / 6) * r3 * m4 - (1. / 18) * r4 * m6
                                       + .25 * r2 * m7;
                        new_f[cell][4] = (1. / 9) * m0 - (1. / 6) * r1 * m2 - (1. / 36) * r2 * m3 + (1. / 6) * r3 * m5 - (1. / 18) * r4 * m6
                                       - .25 * r2 * m7;
                        new_f[cell][5] = (1. / 9) * m0 + (1. / 6) * r1 * m1 + (1. / 6) * r1 * m2 + (1. / 18) * r2 * m3 + (1. / 12) * r3 * m4
                                       + (1. / 12) * r3 * m5 + (1. / 36) * r4 * m6 + .25 * r2 * m8;
                        new_f[cell][6] = (1. / 9) * m0 - (1. / 6) * r1 * m1 + (1. / 6) * r1 * m2 + (1. / 18) * r2 * m3 - (1. / 12) * r3 * m4
                                       + (1. / 12) * r3 * m5 + (1. / 36) * r4 * m6 - .25 * r2 * m8;
                        new_f[cell][7] = (1. / 9) * m0 - (1. / 6) * r1 * m1 - (1. / 6) * r1 * m2 + (1. / 18) * r2 * m3 - (1. / 12) * r3 * m4
                                       - (1. / 12) * r3 * m5 + (1. / 36) * r4 * m6 + .25 * r2 * m8;
                        new_f[cell][8] = (1. / 9) * m0 + (1. / 6) * r1 * m1 - (1. / 6) * r1 * m2 + (1. / 18) * r2 * m3 + (1. / 12) * r3 * m4
                                       - (1. / 12) * r3 * m5 + (1. / 36) * r4 * m6 - .25 * r2 * m8;
                    }
                    else
                    {
                        // The cell has the interface cutting through it
                        if (inside_obstacle(x - .5 * dx, y - .5 * dx, radius) || inside_obstacle(x + .5 * dx, y - .5 * dx, radius)
                            || inside_obstacle(x + .5 * dx, y + .5 * dx, radius) || inside_obstacle(x - .5 * dx, y + .5 * dx, radius))
                        {
                            // We compute the volume fraction
                            double vol_fraction = volume_inside_obstacle_estimation(x - .5 * dx, y - .5 * dx, dx, radius);
                            double len_boundary = length_obstacle_inside_cell(x - .5 * dx, y - .5 * dx, dx, radius);

                            double dt = 1. / (1 << max_level) / lambda;
                            Fx += dx / dt * (1. / (1 << max_level)) * vol_fraction * lambda
                                * (new_f[cell][1] - new_f[cell][3] + new_f[cell][5] - new_f[cell][6] - new_f[cell][7] + new_f[cell][8]);
                            Fy += dx / dt * (1. / (1 << max_level)) * vol_fraction * lambda
                                * (new_f[cell][2] - new_f[cell][4] + new_f[cell][5] + new_f[cell][6] - new_f[cell][7] - new_f[cell][8]);

                            new_f[cell][0] = (1. - vol_fraction) * new_f[cell][0]
                                           + vol_fraction * ((1. / 9) * m0 - (1. / 9) * r2 * m3 + (1. / 9) * r4 * m6);
                            new_f[cell][1] = (1. - vol_fraction) * new_f[cell][1]
                                           + vol_fraction
                                                 * ((1. / 9) * m0 + (1. / 6) * r1 * m1 - (1. / 36) * r2 * m3 - (1. / 6) * r3 * m4
                                                    - (1. / 18) * r4 * m6 + .25 * r2 * m7);
                            new_f[cell][2] = (1. - vol_fraction) * new_f[cell][2]
                                           + vol_fraction
                                                 * ((1. / 9) * m0 + (1. / 6) * r1 * m2 - (1. / 36) * r2 * m3 - (1. / 6) * r3 * m5
                                                    - (1. / 18) * r4 * m6 - .25 * r2 * m7);
                            new_f[cell][3] = (1. - vol_fraction) * new_f[cell][3]
                                           + vol_fraction
                                                 * ((1. / 9) * m0 - (1. / 6) * r1 * m1 - (1. / 36) * r2 * m3 + (1. / 6) * r3 * m4
                                                    - (1. / 18) * r4 * m6 + .25 * r2 * m7);
                            new_f[cell][4] = (1. - vol_fraction) * new_f[cell][4]
                                           + vol_fraction
                                                 * ((1. / 9) * m0 - (1. / 6) * r1 * m2 - (1. / 36) * r2 * m3 + (1. / 6) * r3 * m5
                                                    - (1. / 18) * r4 * m6 - .25 * r2 * m7);
                            new_f[cell][5] = (1. - vol_fraction) * new_f[cell][5]
                                           + vol_fraction
                                                 * ((1. / 9) * m0 + (1. / 6) * r1 * m1 + (1. / 6) * r1 * m2 + (1. / 18) * r2 * m3
                                                    + (1. / 12) * r3 * m4 + (1. / 12) * r3 * m5 + (1. / 36) * r4 * m6 + .25 * r2 * m8);
                            new_f[cell][6] = (1. - vol_fraction) * new_f[cell][6]
                                           + vol_fraction
                                                 * ((1. / 9) * m0 - (1. / 6) * r1 * m1 + (1. / 6) * r1 * m2 + (1. / 18) * r2 * m3
                                                    - (1. / 12) * r3 * m4 + (1. / 12) * r3 * m5 + (1. / 36) * r4 * m6 - .25 * r2 * m8);
                            new_f[cell][7] = (1. - vol_fraction) * new_f[cell][7]
                                           + vol_fraction
                                                 * ((1. / 9) * m0 - (1. / 6) * r1 * m1 - (1. / 6) * r1 * m2 + (1. / 18) * r2 * m3
                                                    - (1. / 12) * r3 * m4 - (1. / 12) * r3 * m5 + (1. / 36) * r4 * m6 + .25 * r2 * m8);
                            new_f[cell][8] = (1. - vol_fraction) * new_f[cell][8]
                                           + vol_fraction
                                                 * ((1. / 9) * m0 + (1. / 6) * r1 * m1 - (1. / 6) * r1 * m2 + (1. / 18) * r2 * m3
                                                    + (1. / 12) * r3 * m4 - (1. / 12) * r3 * m5 + (1. / 36) * r4 * m6 - .25 * r2 * m8);
                        }
                    }
                }
            });
    }

    std::swap(f.array(), new_f.array());
    return std::make_pair(Fx / (rho0 * u0 * u0 * radius), Fy / (rho0 * u0 * u0 * radius));
}

template <class Field>
void save_solution(Field& f, double eps, std::size_t ite, double lambda, std::string ext = "")
{
    using value_t   = typename Field::value_type;
    auto mesh       = f.mesh();
    using mesh_id_t = typename decltype(mesh)::mesh_id_t;

    std::size_t min_level = mesh.min_level();
    std::size_t max_level = mesh.max_level();

    std::stringstream str;
    str << "LBM_D2Q9_von_Karman_street_" << ext << "_lmin_" << min_level << "_lmax-" << max_level << "_eps-" << eps << "_ite-" << ite;

    auto level_  = samurai::make_field<std::size_t, 1>("level", mesh);
    auto rho     = samurai::make_field<value_t, 1>("rho", mesh);
    auto qx      = samurai::make_field<value_t, 1>("qx", mesh);
    auto qy      = samurai::make_field<value_t, 1>("qy", mesh);
    auto vel_mod = samurai::make_field<value_t, 1>("vel_modulus", mesh);

    samurai::for_each_cell(mesh[mesh_id_t::cells],
                           [&](auto& cell)
                           {
                               level_[cell] = cell.level;
                               rho[cell]    = f[cell][0] + f[cell][1] + f[cell][2] + f[cell][3] + f[cell][4] + f[cell][5] + f[cell][6]
                                         + f[cell][7] + f[cell][8];

                               qx[cell] = lambda * (f[cell][1] - f[cell][3] + f[cell][5] - f[cell][6] - f[cell][7] + f[cell][8]);
                               qy[cell] = lambda * (f[cell][2] - f[cell][4] + f[cell][5] + f[cell][6] - f[cell][7] - f[cell][8]);

                               vel_mod[cell] = std::sqrt(qx[cell] * qx[cell] + qy[cell] * qy[cell]) / rho[cell];
                           });

    samurai::save(str.str().data(), mesh, rho, qx, qy, vel_mod, f, level_);
}

int main(int argc, char* argv[])
{
    cxxopts::Options options("D2Q9 scheme for the simulation of the Von Karman vortex street", "...");

    options.add_options()("min_level", "minimum level", cxxopts::value<std::size_t>()->default_value("8"))(
        "max_level",
        "maximum level",
        cxxopts::value<std::size_t>()->default_value("8"))("epsilon", "maximum level", cxxopts::value<double>()->default_value("0.01"))(
        "reg",
        "regularity",
        cxxopts::value<double>()->default_value("0."))("h, help", "Help");

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

            std::size_t min_level = result["min_level"].as<std::size_t>();
            std::size_t max_level = result["max_level"].as<std::size_t>();
            double eps            = result["epsilon"].as<double>();
            double regularity     = result["reg"].as<double>();

            samurai::Box<double, dim> box({0, 0}, {2, 1});
            samurai::MROMesh<Config> mesh{box, min_level, max_level};
            using mesh_id_t     = typename samurai::MROMesh<Config>::mesh_id_t;
            using coord_index_t = typename samurai::MROMesh<Config>::coord_index_t;
            auto pred_coeff     = compute_prediction<coord_index_t>(min_level, max_level);

            const double radius = 1. / 32.;                       // Radius of the obstacle
            const double Re     = 1200;                           // Reynolds number
            const double rho0   = 1.;                             // Reference density
            const double mu     = 5.e-6;                          // Bulk viscosity
            const double zeta   = 10. * mu;                       // Shear viscosity
            const double u0     = mu * Re / (2. * rho0 * radius); // Reference velocity

            // std::string momenti("Geier"); // Momenta by Geier
            std::string momenti("Lallemand"); // Momenta by Lallemand

            const double lambda = 1.; // Lattice velocity

            auto f = init_f(mesh, radius, rho0, u0, lambda, momenti);

            double T  = 1000.;
            double dx = 1.0 / (1 << max_level);
            double dt = dx / lambda;

            std::size_t N = static_cast<std::size_t>(T / dt);

            std::string suffix("_" + std::to_string(min_level) + "_" + std::to_string(max_level) + "_" + std::to_string(eps));

            std::ofstream time_frames;
            time_frames.open("./drag/time_frames" + suffix + ".dat");
            std::ofstream time_frames_saved;
            time_frames_saved.open("./drag/time_frames_saved" + suffix + ".dat");
            std::ofstream CD;
            CD.open("./drag/CD" + suffix + ".dat");
            std::ofstream CL;
            CL.open("./drag/CL" + suffix + ".dat");
            std::ofstream num_leaves;
            num_leaves.open("./drag/leaves" + suffix + ".dat");
            std::ofstream num_cells;
            num_cells.open("./drag/cells" + suffix + ".dat");

            auto update_bc_for_level = [](auto& field, std::size_t level)
            {
                update_bc_D2Q4_3_Euler_constant_extension(field, level);
            };
            auto MRadaptation = samurai::make_MRAdapt(f, update_bc_for_level);

            // std::size_t howoften = 128 * static_cast<double>(max_level)
            // / 10.;
            std::size_t howoften = 128 * std::pow(2., static_cast<double>(max_level) - 10.);

            for (std::size_t nb_ite = 0; nb_ite < N; ++nb_ite)
            {
                std::cout << std::endl << "Iteration number = " << nb_ite << std::endl;
                time_frames << nb_ite * dt << std::endl;

                if (max_level > min_level)
                {
                    MRadaptation(eps, regularity);
                }

                if (nb_ite % howoften == 0)
                {
                    save_solution(f, eps, nb_ite / howoften, lambda,
                                  "Re_std_" + std::to_string(Re)); // Before applying the scheme
                    time_frames_saved << (nb_ite * dt) << std::endl;
                }

                auto CDCL = one_time_step(f, update_bc_for_level, pred_coeff, rho0, u0, lambda, mu, zeta, radius, momenti);
                std::cout << std::endl << "CD = " << CDCL.first << "   CL = " << CDCL.second << std::endl;
                CD << CDCL.first << std::endl;
                CL << CDCL.second << std::endl;

                num_leaves << mesh.nb_cells(mesh_id_t::cells) << std::endl;
                num_cells << mesh.nb_cells() << std::endl;
            }
            CD.close();
            CL.close();
            time_frames.close();
            num_leaves.close();
            num_cells.close();
            time_frames_saved.close();
        }
    }
    catch (const cxxopts::OptionException& e)
    {
        std::cout << options.help() << "\n";
    }
    return 0;
}
