// Copyright 2021 SAMURAI TEAM. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include <fstream>
#include <math.h>
#include <vector>

#include <cxxopts.hpp>

#include <xtensor/xio.hpp>

#include <samurai/field.hpp>
#include <samurai/hdf5.hpp>
#include <samurai/mr/adapt.hpp>
#include <samurai/mr/mesh_with_overleaves.hpp>

#include "boundary_conditions.hpp"
#include "prediction_map_1d.hpp"

#include "utils_lbm_mr_1d.hpp"

#include <chrono>

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

template <class coord_index_t>
auto compute_prediction_separate_inout(std::size_t min_level, std::size_t max_level)
{
    coord_index_t i = 0;
    std::vector<std::vector<prediction_map<coord_index_t>>> data(max_level - min_level + 1);

    for (std::size_t k = 0; k < max_level - min_level + 1; ++k)
    {
        int size = (1 << k);
        data[k].resize(4);

        data[k][0] = prediction(k, i * size - 1);
        data[k][1] = prediction(k, (i + 1) * size - 1);
        data[k][2] = prediction(k, (i + 1) * size);
        data[k][3] = prediction(k, i * size);
    }
    return data;
}

std::array<double, 3> exact_solution(double x, double t)
{
    double density  = 0.0;
    double velocity = 0.0;
    double pressure = 0.0;

    if (t <= 0.0)
    {
        density  = (x <= 0.0) ? 1.0 : 0.125;
        velocity = 0.0;
        pressure = (x <= 0.0) ? 1.0 : 0.1;
    }
    else
    {
        double gm = 1.4;

        double rhoL = 1.0;
        double rhoR = 0.125;
        double uL   = 0.0;
        double uR   = 0.0;
        double pL   = 1.0;
        double pR   = 0.1;

        double cL = sqrt(gm * rhoL / pL);
        double cR = sqrt(gm * rhoR / pR);

        double pStar = 0.30313;
        double uStar = 0.92745;

        double cLStar = cL * pow(pStar / pL, (gm - 1) / (2 * gm));

        double rhoLStar = rhoL * pow(pStar / pL, 1. / gm);
        double rhoRStar = rhoR * ((pStar / pR + (gm - 1) / (gm + 1)) / ((gm - 1) / (gm + 1) * pStar / pR + 1));

        double xFL      = (uL - cL) * t;
        double xFR      = (uStar - cLStar) * t;
        double xContact = (uL + 2 * cL / (gm - 1) * (1 - pow(pStar / pL, (gm - 1) / (2 * gm)))) * t;
        double xShock   = (uR + cR * sqrt((gm + 1) / (2 * gm) * pStar / pR + (gm - 1) / (2 * gm))) * t;

        if (x <= xFL)
        {
            density  = rhoL;
            velocity = uL;
            pressure = pL;
        }
        else
        {
            if (x <= xFR)
            {
                density  = rhoL * pow(2 / (gm + 1) + (gm - 1) / (cL * (gm + 1)) * (uL - x / t), 2 / (gm - 1));
                velocity = 2. / (gm + 1) * (cL + (gm - 1) / 2 * uL + x / t);
                pressure = pL * pow(2 / (gm + 1) + (gm - 1) / ((gm + 1) * cL) * (uL - x / t), 2 * gm / (gm - 1));
            }
            else
            {
                if (x <= xContact)
                {
                    density  = rhoLStar;
                    velocity = uStar;
                    pressure = pStar;
                }
                else
                {
                    if (x <= xShock)
                    {
                        density  = rhoRStar;
                        velocity = uStar;
                        pressure = pStar;
                    }
                    else
                    {
                        density  = rhoR;
                        velocity = uR;
                        pressure = pR;
                    }
                }
            }
        }
    }
    return {density, velocity, pressure};
}

template <class Config>
auto init_f(samurai::MROMesh<Config>& mesh, const double lambda)
{
    using mesh_id_t            = typename samurai::MROMesh<Config>::mesh_id_t;
    constexpr std::size_t nvel = 6;

    auto f = samurai::make_field<double, nvel>("f", mesh);
    f.fill(0);

    double gamma = 1.4;

    samurai::for_each_cell(mesh[mesh_id_t::cells],
                           [&](auto& cell)
                           {
                               auto center = cell.center();
                               auto x      = center[0];

                               auto initial_data = exact_solution(x, 0.0);

                               double density  = initial_data[0];
                               double velocity = initial_data[1];
                               double pressure = initial_data[2];

                               double u10 = density;
                               double u20 = density * velocity;
                               double u30 = 0.5 * density * velocity * velocity + pressure / (gamma - 1.0);
                               double u11 = u20;
                               double u21 = (gamma - 1.0) * u30 + (3.0 - gamma) / (2.0) * (u20 * u20) / u10;
                               double u31 = gamma * (u20 * u30) / (u10) + (1.0 - gamma) / 2.0 * (u20 * u20 * u20) / (u10 * u10);

                               f[cell][0] = .5 * (u10 + u11 / lambda);
                               f[cell][1] = .5 * (u10 - u11 / lambda);
                               f[cell][2] = .5 * (u20 + u21 / lambda);
                               f[cell][3] = .5 * (u20 - u21 / lambda);
                               f[cell][4] = .5 * (u30 + u31 / lambda);
                               f[cell][5] = .5 * (u30 - u31 / lambda);
                           });
    return f;
}

template <class Field, class Pred, class Func>
void one_time_step(Field& f, const Pred& pred_coeff, Func&& update_bc_for_level, double s_rel, const double lambda)
{
    constexpr std::size_t nvel = Field::size;
    double gamma               = 1.4;

    auto mesh           = f.mesh();
    using mesh_t        = typename Field::mesh_t;
    using mesh_id_t     = typename mesh_t::mesh_id_t;
    using coord_index_t = typename mesh_t::interval_t::coord_index_t;
    using interval_t    = typename mesh_t::interval_t;

    auto min_level = mesh.max_level();
    auto max_level = mesh.max_level();

    samurai::update_ghost_mr(f, std::forward<Func>(update_bc_for_level));
    samurai::update_overleaves_mr(f, std::forward<Func>(update_bc_for_level));

    auto new_f = samurai::make_field<double, nvel>("new_f", mesh);
    new_f.fill(0.);
    auto advected_f = samurai::make_field<double, nvel>("advected_f", mesh);
    advected_f.fill(0.);
    auto help_f = samurai::make_field<double, nvel>("help_f", mesh);
    help_f.fill(0.);

    for (std::size_t level = 0; level <= max_level; ++level)
    {
        // If we are at the finest level, we no not need to correct
        if (level == max_level)
        {
            auto leaves = samurai::intersection(mesh[mesh_id_t::cells][max_level], mesh[mesh_id_t::cells][max_level]);

            leaves.on(max_level)(
                [&](auto& interval, auto)
                {
                    auto i = interval;

                    for (int n_scheme = 0; n_scheme < 3; ++n_scheme)
                    {
                        advected_f(0 + 2 * n_scheme, max_level, i) = xt::eval(f(0 + 2 * n_scheme, max_level, i - 1));
                        advected_f(1 + 2 * n_scheme, max_level, i) = xt::eval(f(1 + 2 * n_scheme, max_level, i + 1));
                    }
                });
        }

        // Otherwise, correction is needed
        else
        {
            // We do the advection on the overleaves
            std::size_t j = max_level - (level + 1);
            double coeff  = 1. / (1 << j);

            auto ol = samurai::intersection(mesh[mesh_id_t::cells][level], mesh[mesh_id_t::cells][level]).on(level + 1);

            ol(
                [&](auto& interval, auto)
                {
                    auto k = interval; // Logical index in x

                    auto fp1 = xt::eval(f(0, level + 1, k));
                    auto fm1 = xt::eval(f(1, level + 1, k));
                    auto fp2 = xt::eval(f(2, level + 1, k));
                    auto fm2 = xt::eval(f(3, level + 1, k));
                    auto fp3 = xt::eval(f(4, level + 1, k));
                    auto fm3 = xt::eval(f(5, level + 1, k));

                    for (auto& c : pred_coeff[j][0].coeff)
                    {
                        coord_index_t stencil = c.first;
                        double weight         = c.second;

                        fp1 += coeff * weight * f(0, level + 1, k + stencil);
                        fp2 += coeff * weight * f(2, level + 1, k + stencil);
                        fp3 += coeff * weight * f(4, level + 1, k + stencil);
                    }
                    for (auto& c : pred_coeff[j][1].coeff)
                    {
                        coord_index_t stencil = c.first;
                        double weight         = c.second;

                        fp1 -= coeff * weight * f(0, level + 1, k + stencil);
                        fp2 -= coeff * weight * f(2, level + 1, k + stencil);
                        fp3 -= coeff * weight * f(4, level + 1, k + stencil);
                    }

                    for (auto& c : pred_coeff[j][2].coeff)
                    {
                        coord_index_t stencil = c.first;
                        double weight         = c.second;

                        fm1 += coeff * weight * f(1, level + 1, k + stencil);
                        fm2 += coeff * weight * f(3, level + 1, k + stencil);
                        fm3 += coeff * weight * f(5, level + 1, k + stencil);
                    }
                    for (auto& c : pred_coeff[j][3].coeff)
                    {
                        coord_index_t stencil = c.first;
                        double weight         = c.second;

                        fm1 -= coeff * weight * f(1, level + 1, k + stencil);
                        fm2 -= coeff * weight * f(3, level + 1, k + stencil);
                        fm3 -= coeff * weight * f(5, level + 1, k + stencil);
                    }

                    // Save it
                    help_f(0, level + 1, k) = fp1;
                    help_f(1, level + 1, k) = fm1;
                    help_f(2, level + 1, k) = fp2;
                    help_f(3, level + 1, k) = fm2;
                    help_f(4, level + 1, k) = fp3;
                    help_f(5, level + 1, k) = fm3;
                });

            // Now that projection has been done, we have to come back on the
            // leaves below the overleaves
            auto leaves = samurai::intersection(mesh[mesh_id_t::cells][level], mesh[mesh_id_t::cells][level]);

            leaves(
                [&](auto& interval, auto)
                {
                    auto i = interval;
                    // Projection
                    for (int n_pop = 0; n_pop < 6; ++n_pop)
                    {
                        advected_f(n_pop, level, i) = 0.5 * (help_f(n_pop, level + 1, 2 * i) + help_f(n_pop, level + 1, 2 * i + 1));
                    }
                });
        }
    }

    for (std::size_t level = 0; level <= max_level; ++level)
    {
        auto leaves = samurai::intersection(mesh[mesh_id_t::cells][level], mesh[mesh_id_t::cells][level]);

        leaves(
            [&](auto& interval, auto)
            {
                auto i = interval;

                double gamma = 1.4;

                auto u10 = xt::eval(advected_f(0, level, i) + advected_f(1, level, i));
                auto u11 = xt::eval(lambda * (advected_f(0, level, i) - advected_f(1, level, i)));
                auto u20 = xt::eval(advected_f(2, level, i) + advected_f(3, level, i));
                auto u21 = xt::eval(lambda * (advected_f(2, level, i) - advected_f(3, level, i)));
                auto u30 = xt::eval(advected_f(4, level, i) + advected_f(5, level, i));
                auto u31 = xt::eval(lambda * (advected_f(4, level, i) - advected_f(5, level, i)));

                auto u11_coll = (1 - s_rel) * u11 + s_rel * (u20);
                auto u21_coll = (1 - s_rel) * u21 + s_rel * ((gamma - 1.0) * u30 + (3.0 - gamma) / (2.0) * (u20 * u20) / u10);
                auto u31_coll = (1 - s_rel) * u31
                              + s_rel * (gamma * (u20 * u30) / (u10) + (1.0 - gamma) / 2.0 * (u20 * u20 * u20) / (u10 * u10));

                new_f(0, level, i) = .5 * (u10 + 1. / lambda * u11_coll);
                new_f(1, level, i) = .5 * (u10 - 1. / lambda * u11_coll);
                new_f(2, level, i) = .5 * (u20 + 1. / lambda * u21_coll);
                new_f(3, level, i) = .5 * (u20 - 1. / lambda * u21_coll);
                new_f(4, level, i) = .5 * (u30 + 1. / lambda * u31_coll);
                new_f(5, level, i) = .5 * (u30 - 1. / lambda * u31_coll);
            });
    }
    std::swap(f.array(), new_f.array());
}

template <class Config, class FieldR, class Func>
std::array<double, 6> compute_error(samurai::Field<Config, double, 6>& f, FieldR& fR, Func&& update_bc_for_level, double t)
{
    auto mesh       = f.mesh();
    using mesh_id_t = typename decltype(mesh)::mesh_id_t;

    auto meshR     = fR.mesh();
    auto max_level = meshR.max_level();

    samurai::update_ghost_mr(f, std::forward<Func>(update_bc_for_level));

    // Getting ready for memoization
    // using interval_t = typename Field::Config::interval_t;
    using interval_t = typename Config::interval_t;
    std::map<std::tuple<std::size_t, std::size_t, interval_t>, xt::xtensor<double, 2>> error_memoization_map;
    error_memoization_map.clear();

    double error_rho = 0.0; // First momentum
    double error_q   = 0.0; // Second momentum
    double error_E   = 0.0; // Third momentum

    double diff_rho = 0.0;
    double diff_q   = 0.0;
    double diff_E   = 0.0;

    double dx = 1.0 / (1 << max_level);

    for (std::size_t level = 0; level <= max_level; ++level)
    {
        auto exp = samurai::intersection(mesh[mesh_id_t::cells][level], mesh[mesh_id_t::cells][level]).on(max_level);

        exp(
            [&](auto& interval, auto)
            {
                auto i = interval;
                auto j = max_level - level;

                auto sol  = prediction_all(f, level, j, i, error_memoization_map);
                auto solR = xt::view(fR(max_level, i), xt::all(), xt::range(0, 3));

                xt::xtensor<double, 1> x = dx * xt::linspace<int>(i.start, i.end - 1, i.size()) + 0.5 * dx;

                xt::xtensor<double, 1> rhoexact = xt::zeros<double>(x.shape());
                xt::xtensor<double, 1> qexact   = xt::zeros<double>(x.shape());
                xt::xtensor<double, 1> Eexact   = xt::zeros<double>(x.shape());

                double gm = 1.4;

                for (std::size_t idx = 0; idx < x.shape()[0]; ++idx)
                {
                    auto ex_sol = exact_solution(x[idx], t);

                    rhoexact[idx] = ex_sol[0];
                    qexact[idx]   = ex_sol[0] * ex_sol[1];
                    Eexact[idx]   = 0.5 * ex_sol[0] * pow(ex_sol[1], 2.0) + ex_sol[2] / (gm - 1.);
                }

                auto rho = xt::eval(xt::view(sol, xt::all(), 0) + xt::view(sol, xt::all(), 1));
                auto q   = xt::eval(xt::view(sol, xt::all(), 2) + xt::view(sol, xt::all(), 3));
                auto E   = xt::eval(xt::view(sol, xt::all(), 4) + xt::view(sol, xt::all(), 5));

                auto rho_ref = xt::eval(fR(0, max_level, i) + fR(1, max_level, i));
                auto q_ref   = xt::eval(fR(2, max_level, i) + fR(3, max_level, i));
                auto E_ref   = xt::eval(fR(4, max_level, i) + fR(5, max_level, i));

                error_rho += xt::sum(xt::abs(rho_ref - rhoexact))[0];
                error_q += xt::sum(xt::abs(q_ref - qexact))[0];
                error_E += xt::sum(xt::abs(E_ref - Eexact))[0];
                diff_rho += xt::sum(xt::abs(rho_ref - rho))[0];
                diff_q += xt::sum(xt::abs(q_ref - q))[0];
                diff_E += xt::sum(xt::abs(E_ref - E))[0];
            });
    }
    return {dx * error_rho, dx * diff_rho, dx * error_q, dx * diff_q, dx * error_E, dx * diff_E};
}

int main(int argc, char* argv[])
{
    cxxopts::Options options("lbm_d1q2_3_Euler", "");

    options.add_options()("min_level", "minimum level", cxxopts::value<std::size_t>()->default_value("2"))(
        "max_level",
        "maximum level",
        cxxopts::value<std::size_t>()->default_value("10"))("epsilon", "maximum level", cxxopts::value<double>()->default_value("0.01"))(
        "s",
        "relaxation parameter",
        cxxopts::value<double>()->default_value("1.0"))("h, help", "Help");

    try
    {
        auto result = options.parse(argc, argv);

        if (result.count("help"))
        {
            std::cout << options.help() << "\n";
        }
        else
        {
            constexpr size_t dim = 1;
            using Config         = samurai::MROConfig<dim, 2>;
            using mesh_t         = samurai::MROMesh<Config>;
            using mesh_id_t      = typename mesh_t::mesh_id_t;
            using coord_index_t  = typename mesh_t::interval_t::coord_index_t;

            std::size_t min_level = 2; // result["min_level"].as<std::size_t>();
            std::size_t max_level = 9; // result["max_level"].as<std::size_t>();

            auto pred_coeff_separate = compute_prediction_separate_inout<coord_index_t>(min_level, max_level);

            // We set some parameters according
            // to the problem.
            double sol_reg = 0.0;
            double T       = 0.4;
            double lambda  = 3.;

            std::string case_name("s_d");
            ;

            samurai::Box<double, dim> box({-1}, {1});
            std::vector<double> s_vect{0.75, 1.0, 1.25, 1.5, 1.75};

            auto update_bc_for_level = [](auto& field, std::size_t level)
            {
                update_bc_1D_constant_extension(field, level);
            };

            for (auto s : s_vect)
            {
                std::cout << std::endl << "Relaxation parameter s = " << s;

                std::string prefix(case_name + "_s_" + std::to_string(s) + "_");

                std::cout << std::endl << "Testing time behavior" << std::endl;
                {
                    double eps = 1.0e-4; // This remains fixed

                    samurai::MROMesh<Config> mesh{box, min_level, max_level};
                    samurai::MROMesh<Config> meshR{box, max_level, max_level}; // This is the reference scheme

                    // Initialization
                    auto f  = init_f(mesh, lambda);
                    auto fR = init_f(meshR, lambda);

                    double dx = 1.0 / (1 << max_level);
                    double dt = dx / lambda;

                    std::size_t N = static_cast<std::size_t>(T / dt);

                    double t = 0.0;

                    std::ofstream out_time_frames;

                    std::ofstream out_error_rho_exact_ref; // On the density
                    std::ofstream out_diff_rho_ref_adap;
                    std::ofstream out_error_q_exact_ref; // On the momentum
                    std::ofstream out_diff_q_ref_adap;
                    std::ofstream out_error_E_exact_ref; // On the energy
                    std::ofstream out_diff_E_ref_adap;
                    std::ofstream out_compression;

                    out_time_frames.open("./d1q2_3/time/" + prefix + "time.dat");
                    out_error_rho_exact_ref.open("./d1q2_3/time/" + prefix + "error_rho.dat");
                    out_diff_rho_ref_adap.open("./d1q2_3/time/" + prefix + "diff_rho.dat");
                    out_error_q_exact_ref.open("./d1q2_3/time/" + prefix + "error_q.dat");
                    out_diff_q_ref_adap.open("./d1q2_3/time/" + prefix + "diff_q.dat");
                    out_error_E_exact_ref.open("./d1q2_3/time/" + prefix + "error_E.dat");
                    out_diff_E_ref_adap.open("./d1q2_3/time/" + prefix + "diff_E.dat");
                    out_compression.open("./d1q2_3/time/" + prefix + "comp.dat");

                    auto MRadaptation = samurai::make_MRAdapt(f, update_bc_for_level);

                    for (std::size_t nb_ite = 0; nb_ite < N; ++nb_ite)
                    {
                        MRadaptation(eps, sol_reg);

                        auto error = compute_error(f, fR, update_bc_for_level, t);

                        out_time_frames << t << std::endl;
                        out_error_rho_exact_ref << error[0] << std::endl;
                        out_diff_rho_ref_adap << error[1] << std::endl;
                        out_error_q_exact_ref << error[2] << std::endl;
                        out_diff_q_ref_adap << error[3] << std::endl;
                        out_error_E_exact_ref << error[4] << std::endl;
                        out_diff_E_ref_adap << error[5] << std::endl;

                        out_compression << static_cast<double>(mesh.nb_cells(mesh_id_t::cells))
                                               / static_cast<double>(meshR.nb_cells(mesh_id_t::cells))
                                        << std::endl;

                        std::cout << std::endl
                                  << "Time = " << t << " Diff_h = " << error[1] << std::endl
                                  << "Diff q = " << error[3] << std::endl
                                  << "Diff E = " << error[5];

                        one_time_step(f, pred_coeff_separate, update_bc_for_level, s, lambda);
                        one_time_step(fR, pred_coeff_separate, update_bc_for_level, s, lambda);
                        t += dt;
                    }

                    std::cout << std::endl;

                    out_time_frames.close();
                    out_error_rho_exact_ref.close();
                    out_diff_rho_ref_adap.close();
                    out_error_q_exact_ref.close();
                    out_diff_q_ref_adap.close();
                    out_error_E_exact_ref.close();
                    out_diff_E_ref_adap.close();
                    out_compression.close();
                }

                std::cout << std::endl << "Testing eps behavior" << std::endl;
                {
                    double eps         = 1.0e-1; // 0.1;
                    std::size_t N_test = 50;     // 50;
                    double factor      = 0.60;
                    std::ofstream out_eps;

                    std::ofstream out_diff_rho_ref_adap;
                    std::ofstream out_diff_q_ref_adap;
                    std::ofstream out_diff_E_ref_adap;
                    std::ofstream out_compression;

                    out_eps.open("./d1q2_3/eps/" + prefix + "eps.dat");
                    out_diff_rho_ref_adap.open("./d1q2_3/eps/" + prefix + "diff_rho.dat");
                    out_diff_q_ref_adap.open("./d1q2_3/eps/" + prefix + "diff_q.dat");
                    out_diff_E_ref_adap.open("./d1q2_3/eps/" + prefix + "diff_E.dat");
                    out_compression.open("./d1q2_3/eps/" + prefix + "comp.dat");

                    for (std::size_t n_test = 0; n_test < N_test; ++n_test)
                    {
                        std::cout << std::endl << "Test " << n_test << " eps = " << eps;

                        samurai::MROMesh<Config> mesh{box, min_level, max_level};
                        samurai::MROMesh<Config> meshR{box, max_level, max_level}; // This is the reference scheme

                        // Initialization
                        auto f  = init_f(mesh, lambda);
                        auto fR = init_f(meshR, lambda);

                        double dx = 1.0 / (1 << max_level);
                        double dt = dx / lambda;

                        std::size_t N = static_cast<std::size_t>(T / dt);

                        double t          = 0.0;
                        auto MRadaptation = samurai::make_MRAdapt(f, update_bc_for_level);

                        for (std::size_t nb_ite = 0; nb_ite < N; ++nb_ite)
                        {
                            MRadaptation(eps, sol_reg);

                            one_time_step(f, pred_coeff_separate, update_bc_for_level, s, lambda);
                            one_time_step(fR, pred_coeff_separate, update_bc_for_level, s, lambda);
                            t += dt;
                        }

                        auto error = compute_error(f, fR, update_bc_for_level, t);
                        std::cout << "Diff  h= " << error[1] << std::endl
                                  << "Diff q = " << error[3] << std::endl
                                  << "Diff E = " << error[5] << std::endl;

                        out_eps << eps << std::endl;
                        out_diff_rho_ref_adap << error[1] << std::endl;
                        out_diff_q_ref_adap << error[3] << std::endl;
                        out_diff_E_ref_adap << error[5] << std::endl;

                        out_compression << static_cast<double>(mesh.nb_cells(mesh_id_t::cells))
                                               / static_cast<double>(meshR.nb_cells(mesh_id_t::cells))
                                        << std::endl;

                        eps *= factor;
                    }

                    out_eps.close();
                    out_diff_rho_ref_adap.close();
                    out_diff_q_ref_adap.close();
                    out_diff_E_ref_adap.close();
                    out_compression.close();
                }
            }
        }
    }
    catch (const cxxopts::OptionException& e)
    {
        std::cout << options.help() << "\n";
    }
    return 0;
}
