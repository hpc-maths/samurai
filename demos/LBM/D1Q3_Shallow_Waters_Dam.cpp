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

std::array<double, 2> exact_solution(double x, double t, const double g)
{
    // Warning : this computation works for the parameters we have used
    // it should be redone otherwise

    double x0 = 0.0;

    double hL = 2.0;
    double hR = 1.0;
    double uL = 0.0;
    double uR = 0.0;

    double cL    = std::sqrt(g * hL);
    double cR    = std::sqrt(g * hR);
    double cStar = 1.20575324689; // To be computed
    double hStar = cStar * cStar / g;

    double xFanL  = x0 - cL * t;
    double xFanR  = x0 + (2 * cL - 3 * cStar) * t;
    double xShock = x0 + (2 * cStar * cStar * (cL - cStar)) / (cStar * cStar - cR * cR) * t;

    double h = (x <= xFanL) ? hL : ((x <= xFanR) ? 4. / (9. * g) * pow(cL - (x - x0) / (2. * t), 2.0) : ((x < xShock) ? hStar : hR));
    double u = (x <= xFanL) ? uL : ((x <= xFanR) ? 2. / 3. * (cL + (x - x0) / t) : ((x < xShock) ? 2. * (cL - cStar) : uR));

    return {h, u};
}

template <class Config>
auto init_f(samurai::MROMesh<Config>& mesh, double t, const double lambda, const double g)
{
    using mesh_id_t            = typename samurai::MROMesh<Config>::mesh_id_t;
    constexpr std::size_t nvel = 3;
    auto f                     = samurai::make_field<double, nvel>("f", mesh);
    f.fill(0);

    samurai::for_each_cell(mesh[mesh_id_t::cells],
                           [&](auto& cell)
                           {
                               auto center = cell.center();
                               auto x      = center[0];

                               auto u = exact_solution(x, 0.0, g);

                               double h = u[0];
                               double q = h * u[1]; // Linear momentum
                               double k = q * q / h + 0.5 * g * h * h;

                               f[cell][0] = h - k / (lambda * lambda);
                               f[cell][1] = 0.5 * (q + k / lambda) / lambda;
                               f[cell][2] = 0.5 * (-q + k / lambda) / lambda;
                           });
    return f;
}

template <class Field, class Pred, class Func>
void one_time_step(Field& f, const Pred& pred_coeff, Func&& update_bc_for_level, double s_rel, const double lambda, const double g)
{
    constexpr std::size_t nvel = Field::size;
    auto mesh                  = f.mesh();
    using mesh_t               = typename Field::mesh_t;
    using mesh_id_t            = typename mesh_t::mesh_id_t;
    using coord_index_t        = typename mesh_t::interval_t::coord_index_t;
    using interval_t           = typename mesh_t::interval_t;

    auto min_level = mesh.min_level();
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
            leaves(
                [&](auto& interval, auto)
                {
                    auto k                      = interval;
                    advected_f(0, max_level, k) = xt::eval(f(0, max_level, k));
                    advected_f(1, max_level, k) = xt::eval(f(1, max_level, k - 1));
                    advected_f(2, max_level, k) = xt::eval(f(2, max_level, k + 1));
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

                    // auto f0 = xt::eval(f(0, level + 1, k));
                    auto fp = xt::eval(f(1, level + 1, k));
                    auto fm = xt::eval(f(2, level + 1, k));

                    for (auto& c : pred_coeff[j][0].coeff)
                    {
                        coord_index_t stencil = c.first;
                        double weight         = c.second;

                        fp += coeff * weight * f(1, level + 1, k + stencil);
                    }
                    for (auto& c : pred_coeff[j][1].coeff)
                    {
                        coord_index_t stencil = c.first;
                        double weight         = c.second;

                        fp -= coeff * weight * f(1, level + 1, k + stencil);
                    }
                    for (auto& c : pred_coeff[j][2].coeff)
                    {
                        coord_index_t stencil = c.first;
                        double weight         = c.second;

                        fm += coeff * weight * f(2, level + 1, k + stencil);
                    }
                    for (auto& c : pred_coeff[j][3].coeff)
                    {
                        coord_index_t stencil = c.first;
                        double weight         = c.second;

                        fm -= coeff * weight * f(2, level + 1, k + stencil);
                    }
                    // Save it
                    help_f(1, level + 1, k) = fp;
                    help_f(2, level + 1, k) = fm;
                });
            // Now that projection has been done, we have to come back on the
            // leaves below the overleaves
            auto leaves = samurai::intersection(mesh[mesh_id_t::cells][level], mesh[mesh_id_t::cells][level]);

            leaves(
                [&](auto& interval, auto)
                {
                    auto k                  = interval;
                    advected_f(0, level, k) = f(0, level, k); // Does not move so no flux
                    advected_f(1, level, k) = xt::eval(0.5 * (help_f(1, level + 1, 2 * k) + help_f(1, level + 1, 2 * k + 1)));
                    advected_f(2, level, k) = xt::eval(0.5 * (help_f(2, level + 1, 2 * k) + help_f(2, level + 1, 2 * k + 1)));
                });
        }
    }

    for (std::size_t level = 0; level <= max_level; ++level)
    {
        auto leaves = samurai::intersection(mesh[mesh_id_t::cells][level], mesh[mesh_id_t::cells][level]);

        leaves(
            [&](auto& interval, auto)
            {
                auto k = interval;

                auto h   = xt::eval(advected_f(0, level, k) + advected_f(1, level, k) + advected_f(2, level, k));
                auto q   = xt::eval(lambda * (advected_f(1, level, k) - advected_f(2, level, k)));
                auto kin = xt::eval(lambda * lambda * (advected_f(1, level, k) + advected_f(2, level, k)));

                auto k_coll = (1 - s_rel) * kin + s_rel * q * q / h + 0.5 * g * h * h;

                new_f(0, level, k) = h - k_coll / (lambda * lambda);
                new_f(1, level, k) = 0.5 * (q + k_coll / lambda) / lambda;
                new_f(2, level, k) = 0.5 * (-q + k_coll / lambda) / lambda;
            });
    }
    std::swap(f.array(), new_f.array());
}

template <class Field>
void save_solution(Field& f, double eps, std::size_t ite, const double lambda, std::string ext = "")
{
    auto mesh       = f.mesh();
    using value_t   = typename Field::value_type;
    using mesh_id_t = typename decltype(mesh)::mesh_id_t;

    std::size_t min_level = mesh.min_level();
    std::size_t max_level = mesh.max_level();

    std::stringstream str;
    str << "LBM_D1Q3_ShallowWaters_" << ext << "_lmin_" << min_level << "_lmax-" << max_level << "_eps-" << eps << "_ite-" << ite;

    auto level_ = samurai::make_field<std::size_t, 1>("level", mesh);
    auto h      = samurai::make_field<value_t, 1>("h", mesh);
    auto q      = samurai::make_field<value_t, 1>("q", mesh);
    auto u      = samurai::make_field<value_t, 1>("u", mesh);

    samurai::for_each_cell(mesh[mesh_id_t::cells],
                           [&](auto& cell)
                           {
                               level_[cell] = static_cast<double>(cell.level);
                               h[cell]      = f[cell][0] + f[cell][1] + f[cell][2];
                               q[cell]      = lambda * (f[cell][1] - f[cell][2]);
                               u[cell]      = q[cell] / h[cell];
                           });
    samurai::save(str.str().data(), mesh, h, q, u, f, level_);
}

template <class Config, class FieldR, class Func>
std::array<double, 4>
compute_error(samurai::Field<Config, double, 3>& f, FieldR& fR, Func&& update_bc_for_level, double t, const double lambda, const double g)
{
    auto mesh       = f.mesh();
    using mesh_id_t = typename decltype(mesh)::mesh_id_t;
    auto meshR      = fR.mesh();
    auto max_level  = meshR.max_level();

    samurai::update_ghost_mr(f, std::forward<Func>(update_bc_for_level));

    // Getting ready for memoization
    // using interval_t = typename Field::Config::interval_t;
    using interval_t = typename Config::interval_t;
    std::map<std::tuple<std::size_t, std::size_t, interval_t>, xt::xtensor<double, 2>> error_memoization_map;
    error_memoization_map.clear();

    double error_h = 0.0; // First momentum
    double error_q = 0.0; // Second momentum
    double diff_h  = 0.0;
    double diff_q  = 0.0;

    double dx = 1.0 / (1 << max_level);

    for (std::size_t level = 0; level <= max_level; ++level)
    {
        auto exp = samurai::intersection(mesh[mesh_id_t::cells][level], mesh[mesh_id_t::cells][level]).on(max_level);

        exp(
            [&](auto& interval, auto)
            {
                auto i = interval;
                auto j = max_level - level;

                auto sol                      = prediction_all(f, level, j, i, error_memoization_map);
                auto solR                     = xt::view(fR(max_level, i), xt::all(), xt::range(0, 3));
                xt::xtensor<double, 1> x      = dx * xt::linspace<int>(i.start, i.end - 1, i.size()) + 0.5 * dx;
                xt::xtensor<double, 1> hexact = xt::zeros<double>(x.shape());
                xt::xtensor<double, 1> qexact = xt::zeros<double>(x.shape());

                for (std::size_t idx = 0; idx < x.shape()[0]; ++idx)
                {
                    auto ex_sol = exact_solution(x[idx], t, g);
                    hexact[idx] = ex_sol[0];
                    qexact[idx] = ex_sol[0] * ex_sol[1];
                }

                auto h     = xt::eval(xt::view(sol, xt::all(), 0) + xt::view(sol, xt::all(), 1) + xt::view(sol, xt::all(), 2));
                auto q     = lambda * xt::eval(xt::view(sol, xt::all(), 1) - xt::view(sol, xt::all(), 2));
                auto h_ref = xt::eval(fR(0, max_level, i) + fR(1, max_level, i) + fR(2, max_level, i));
                auto q_ref = lambda * xt::eval(fR(1, max_level, i) - fR(2, max_level, i));

                error_h += xt::sum(xt::abs(h_ref - hexact))[0];
                error_q += xt::sum(xt::abs(q_ref - qexact))[0];
                diff_h += xt::sum(xt::abs(h_ref - h))[0];
                diff_q += xt::sum(xt::abs(q_ref - q))[0];
            });
    }
    return {dx * error_h, dx * diff_h, dx * error_q, dx * diff_q};
}

int main(int argc, char* argv[])
{
    cxxopts::Options options("lbm_d1q3_shallow waters", "...");

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

            std::size_t min_level = result["min_level"].as<std::size_t>();
            std::size_t max_level = result["max_level"].as<std::size_t>();
            double eps            = result["epsilon"].as<double>();
            double s              = result["s"].as<double>();

            samurai::Box<double, dim> box({-1}, {1});
            samurai::MROMesh<Config> mesh{box, min_level, max_level};
            samurai::MROMesh<Config> meshR{box, max_level, max_level}; // This is the reference scheme

            auto pred_coeff_separate = compute_prediction_separate_inout<coord_index_t>(min_level, max_level);

            const double lambda = 2.0;
            const double g      = 1.0; // Gravity

            // Initialization
            auto f  = init_f(mesh, 0.0, lambda, g);
            auto fR = init_f(meshR, 0.0, lambda, g);

            // double T = 0.2; // On the paper
            double T = 0.35;

            double dx = 1.0 / (1 << max_level);
            double dt = dx / lambda;

            std::size_t N = static_cast<std::size_t>(T / dt);

            double t = 0.0;

            auto update_bc_for_level = [](auto& field, std::size_t level)
            {
                update_bc_1D_constant_extension(field, level);
            };

            auto MRadaptation = samurai::make_MRAdapt(f, update_bc_for_level);

            for (std::size_t nb_ite = 0; nb_ite < N; ++nb_ite)
            {
                std::cout << std::endl << "Iteration " << nb_ite << " Time = " << t;

                MRadaptation(eps, 0.); // Regularity 0

                save_solution(f, eps, nb_ite, lambda);

                auto error = compute_error(f, fR, update_bc_for_level, t, lambda, g);
                std::cout << std::endl
                          << "Error h = " << error[0] << std::endl
                          << "Diff h = " << error[1] << std::endl
                          << "Error q = " << error[2] << std::endl
                          << "Diff q = " << error[3];

                tic();
                one_time_step(f, pred_coeff_separate, update_bc_for_level, s, lambda, g);
                auto duration_scheme = toc();
                one_time_step(fR, pred_coeff_separate, update_bc_for_level, s, lambda, g);

                t += dt;
            }
        }
    }
    catch (const cxxopts::OptionException& e)
    {
        std::cout << options.help() << "\n";
    }
    std::cout << std::endl;
    return 0;
}
