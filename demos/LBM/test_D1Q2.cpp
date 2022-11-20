// Copyright 2021 SAMURAI TEAM. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include <vector>
#include <string>
#include <fstream>

#include <filesystem>
namespace fs = std::filesystem;

#include <CLI/CLI.hpp>

#include <fmt/format.h>

#include <samurai/mr/adapt.hpp>
#include <samurai/field.hpp>
#include <samurai/mr/mesh_with_overleaves.hpp>
#include <samurai/hdf5.hpp>

#include "prediction_map_1d.hpp"
#include "boundary_conditions.hpp"

#include "utils_lbm_mr_1d.hpp"

/*
TEST CASES
1 : transport - gaussienne
2 : transport - probleme de Riemann
3 : Burgers - tangente hyperbolique reguliere
4 : Burgers - fonction chapeau avec changement de regularite
5 : Burgers - probleme de Riemann
*/

enum class TestCase
{
    adv_gaussian,
    adv_riemann,
    burgers_tanh,
    burgers_hat,
    burgers_riemann
};

template <>
struct fmt::formatter<TestCase>: formatter<string_view>
{
    template <typename FormatContext>
    auto format(TestCase c, FormatContext& ctx)
    {
        string_view name = "unknown";
        switch (c) {
        case TestCase::adv_gaussian:    name = "adv_gaussian"; break;
        case TestCase::adv_riemann:     name = "adv_riemann"; break;
        case TestCase::burgers_tanh:    name = "burgers_tanh"; break;
        case TestCase::burgers_hat:     name = "burgers_hat"; break;
        case TestCase::burgers_riemann: name = "burgers_riemann"; break;
        }
        return formatter<string_view>::format(name, ctx);
    }
};


template<class coord_index_t>
auto compute_prediction_separate_inout(std::size_t min_level, std::size_t max_level)
{
    coord_index_t i = 0;
    std::vector<std::vector<prediction_map<coord_index_t>>> data(max_level-min_level+1);

    for(std::size_t k=0; k<max_level-min_level+1; ++k)
    {
        int size = (1<<k);
        data[k].resize(4);

        data[k][0] = prediction(k, i*size - 1);
        data[k][1] = prediction(k, (i+1)*size - 1);
        data[k][2] = prediction(k, (i+1)*size);
        data[k][3] = prediction(k, i*size);
    }
    return data;
}

double exact_solution(double x, double t, double ad_vel, TestCase test)
{
    double u = 0;

    switch(test)
    {
        case TestCase::adv_gaussian :
        {
            u = exp(-20.0 * (x-ad_vel*t) * (x-ad_vel*t)); // Used in the first draft
            // u = exp(-60.0 * (x-ad_vel*t) * (x-ad_vel*t));
            break;
        }

        case TestCase::adv_riemann :
        {
            double sigma = 0.5;
            double rhoL = 0.0;
            double rhoC = 1.0;
            double rhoR = 0.0;

            double xtr = x - ad_vel*t;
            u =  (xtr <= -sigma) ? (rhoL) : ((xtr <= sigma) ? (rhoC) : rhoR );
            break;
        }
        case TestCase::burgers_tanh :
        {
            double sigma = 100.0;
            if (t <= 0.0)
                u = 0.5 * (1.0 + tanh(sigma * x));
            else
            {   // We proceed by dichotomy
                double a = -3.2;
                double b =  3.2;

                double tol = 1.0e-8;

                auto F = [sigma, x, t] (double y)   {
                    return y + 0.5 * (1.0 + tanh(sigma * y))*t - x;
                };
                double res = 0.0;

                while (b-a > tol)   {
                    double mean = 0.5 * (b + a);
                    double eval = F(mean);
                    if (eval <= 0.0)
                        a = mean;
                    else
                        b = mean;
                    res = mean;
                }

                u =  0.5 * (1.0 + tanh(sigma * res));
            }
            break;
        }

        case TestCase::burgers_hat :
        {
            if (x >= -1 and x < t)
            {
                u = (1 + x) / (1 + t);
            }

            if (x >= t and x < 1)
            {
                u = (1 - x) / (1 - t);
            }
            break;
        }

        case TestCase::burgers_riemann :
        {
            double sigma = 0.5;
            double rhoL = 0.0;
            double rhoC = 1.0;
            double rhoR = 0.0;

            u =  (x + sigma <= rhoL * t) ? rhoL : ((x + sigma <= rhoC*t) ? (x+sigma)/t : ((x-sigma <= t/2*(rhoC + rhoR)) ? rhoC : rhoR ));
            break;
        }
    }

    return u;
}

double flux(double u, double ad_vel, TestCase test)
{
    // Advection
    if (test == TestCase::adv_gaussian || test == TestCase::adv_riemann)
    {
        return ad_vel * u;
    }
    // Burgers
    else
    {
        return 0.5 * u * u;
    }
}

template<class Config>
auto init_f(samurai::MROMesh<Config> &mesh, double ad_vel, double lambda, TestCase test)
{
    constexpr std::size_t nvel = 2;
    using mesh_id_t = typename samurai::MROMesh<Config>::mesh_id_t;

    auto f = samurai::make_field<double, nvel>("f", mesh);
    f.fill(0);

    samurai::for_each_cell(mesh[mesh_id_t::cells], [&](auto &cell)
    {
        auto center = cell.center();
        auto x = center[0];

        double u = 0;

        u = exact_solution(x, 0.0, ad_vel, test);
        double v = flux(u, ad_vel, test);

        f[cell][0] = .5 * (u + v/lambda);
        f[cell][1] = .5 * (u - v/lambda);
    });

    return f;
}


template<class Field, class Func, class Pred>
void one_time_step(Field &f, Func&& update_bc_for_level,
                            const Pred& pred_coeff, double s_rel, double lambda, double ad_vel, TestCase test,
                            bool finest_collision = false)
{

    constexpr std::size_t nvel = Field::size;

    auto mesh = f.mesh();
    using mesh_t = typename Field::mesh_t;
    using mesh_id_t = typename mesh_t::mesh_id_t;
    using coord_index_t = typename mesh_t::interval_t::coord_index_t;
    using interval_t = typename mesh_t::interval_t;

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
        if (level == max_level)
        {
            auto leaves = samurai::intersection(mesh[mesh_id_t::cells][max_level],
                                                mesh[mesh_id_t::cells][max_level]);
            leaves([&](auto &interval, auto) {
                auto k = interval;
                advected_f(0, max_level, k) = xt::eval(f(0, max_level, k - 1));
                advected_f(1, max_level, k) = xt::eval(f(1, max_level, k + 1));
            });
        }
        else
        {
            // We do the advection on the overleaves
            std::size_t j = max_level - (level + 1);
            double coeff = 1. / (1 << j);

            auto ol = samurai::intersection(mesh[mesh_id_t::cells][level],
                                            mesh[mesh_id_t::cells][level]).on(level + 1);

            ol([&](auto& interval, auto)
            {
                auto k = interval; // Logical index in x

                auto fp = xt::eval(f(0, level + 1, k));
                auto fm = xt::eval(f(1, level + 1, k));

                for(auto &c: pred_coeff[j][0].coeff)
                {
                    coord_index_t stencil = c.first;
                    double weight = c.second;

                    fp += coeff * weight * f(0, level + 1, k + stencil);
                }

                for(auto &c: pred_coeff[j][1].coeff)
                {
                    coord_index_t stencil = c.first;
                    double weight = c.second;

                    fp -= coeff * weight * f(0, level + 1, k + stencil);
                }

                for(auto &c: pred_coeff[j][2].coeff)
                {
                    coord_index_t stencil = c.first;
                    double weight = c.second;

                    fm += coeff * weight * f(1, level + 1, k + stencil);
                }

                for(auto &c: pred_coeff[j][3].coeff)
                {
                    coord_index_t stencil = c.first;
                    double weight = c.second;

                    fm -= coeff * weight * f(1, level + 1, k + stencil);
                }

                // Save it
                help_f(0, level + 1, k) = fp;
                help_f(1, level + 1, k) = fm;
            });

            // Now that projection has been done, we have to come back on the leaves below the overleaves
            auto leaves = samurai::intersection(mesh[mesh_id_t::cells][level],
                                                mesh[mesh_id_t::cells][level]);

            leaves([&](auto &interval, auto)
            {
                auto k = interval;
                // Projection
                advected_f(0, level, k) = xt::eval(0.5 * (help_f(0, level + 1, 2*k) + help_f(0, level + 1, 2*k + 1)));
                advected_f(1, level, k) = xt::eval(0.5 * (help_f(1, level + 1, 2*k) + help_f(1, level + 1, 2*k + 1)));
            });
        }
    }

    // Collision
    if (!finest_collision)
    {
        for (std::size_t level = 0; level <= max_level; ++level)
        {
            auto leaves = samurai::intersection(mesh[mesh_id_t::cells][level],
                                                mesh[mesh_id_t::cells][level]);

            leaves([&](auto &interval, auto)
            {
                auto k = interval;
                auto uu = xt::eval(          advected_f(0, level, k) + advected_f(1, level, k));
                auto vv = xt::eval(lambda * (advected_f(0, level, k) - advected_f(1, level, k)));

                if (test == TestCase::adv_gaussian || test == TestCase::adv_riemann)
                {
                    vv = (1 - s_rel) * vv + s_rel * ad_vel * uu;
                }
                else
                {
                    vv = (1 - s_rel) * vv + s_rel * .5 * uu * uu;
                }

                new_f(0, level, k) = .5 * (uu + 1. / lambda * vv);
                new_f(1, level, k) = .5 * (uu - 1. / lambda * vv);
            });
        }
    }

    else
    {
        samurai::update_ghost_mr(advected_f, std::forward<Func>(update_bc_for_level));

        std::map<std::tuple<std::size_t, std::size_t, interval_t>, xt::xtensor<double, 2>> memoization_map;
        memoization_map.clear();

        for (std::size_t level = 0; level <= max_level; ++level)
        {
            auto leaves_on_finest = samurai::intersection(mesh[mesh_id_t::cells][level],
                                                          mesh[mesh_id_t::cells][level]).on(max_level);

            leaves_on_finest([&](auto &interval, auto)
            {
                auto i = interval;
                auto j = max_level - level;

                auto f_on_finest  = prediction_all(advected_f, level, j, i, memoization_map);

                auto uu = xt::eval(xt::view(f_on_finest, xt::all(), 0)
                                 + xt::view(f_on_finest, xt::all(), 1));

                auto vv = xt::eval(lambda*(xt::view(f_on_finest, xt::all(), 0)
                                         - xt::view(f_on_finest, xt::all(), 1)));

                if (test == TestCase::adv_gaussian || test == TestCase::adv_riemann)
                {
                    vv = (1 - s_rel) * vv + s_rel * ad_vel * uu;
                }
                else
                {
                    vv = (1 - s_rel) * vv + s_rel * .5 * uu * uu;
                }

                auto f_0_post_coll = .5 * (uu + 1. / lambda * vv);
                auto f_1_post_coll = .5 * (uu - 1. / lambda * vv);

                int step = 1 << j;

                for (auto i_start = 0; i_start < (i.end - i.start); i_start = i_start + step)    {
                    new_f(0, level, {(i.start + i_start)/step, (i.start + i_start)/step + 1}) = xt::mean(xt::view(f_0_post_coll, xt::range(i_start, i_start + step)));
                    new_f(1, level, {(i.start + i_start)/step, (i.start + i_start)/step + 1}) = xt::mean(xt::view(f_1_post_coll, xt::range(i_start, i_start + step)));
                }
            });
        }
    }
    std::swap(f.array(), new_f.array());
}

template<class Config, class FieldR, class Func>
std::array<double, 2> compute_error(samurai::Field<Config, double, 2> &f, FieldR & fR, Func&& update_bc_for_level, double t, double ad_vel, TestCase test)
{

    auto mesh = f.mesh();
    using mesh_id_t = typename decltype(mesh)::mesh_id_t;

    auto meshR = fR.mesh();
    auto max_level = meshR.max_level();

    update_bc_for_level(fR, max_level); // It is important to do so

    samurai::update_ghost_mr(f, std::forward<Func>(update_bc_for_level));

    // Getting ready for memoization
    // using interval_t = typename Field::Config::interval_t;
    using interval_t = typename Config::interval_t;
    std::map<std::tuple<std::size_t, std::size_t, interval_t>, xt::xtensor<double, 2>> error_memoization_map;

    error_memoization_map.clear();

    double error = 0; // To return
    double diff = 0.0;

    double dx = 1.0 / (1 << max_level);

    for (std::size_t level = 0; level <= max_level; ++level)
    {
        auto exp = samurai::intersection(mesh[mesh_id_t::cells][level],
                                         mesh[mesh_id_t::cells][level]).on(max_level);

        exp([&](auto &interval, auto)
        {
            auto i = interval;
            auto j = max_level - level;

            auto sol  = prediction_all(f, level, j, i, error_memoization_map);
            auto solR = xt::view(fR(max_level, i), xt::all(), xt::range(0, 2));

            xt::xtensor<double, 1> x = dx*xt::linspace<int>(i.start, i.end - 1, i.size()) + 0.5*dx;
            xt::xtensor<double, 1> uexact = xt::zeros<double>(x.shape());

            for (std::size_t idx = 0; idx < x.shape()[0]; ++idx)    {
                uexact[idx] = exact_solution(x[idx], t, ad_vel, test); // We can probably do better
            }

            auto rho_ref = xt::eval(fR(0, max_level, i) + fR(1, max_level, i));
            auto rho = xt::eval(xt::view(sol, xt::all(), 0) +  xt::view(sol, xt::all(), 1));

            error += xt::sum(xt::abs(rho_ref - uexact))[0];
            diff  += xt::sum(xt::abs(rho_ref - rho))[0];
        });
    }
    return {dx * error, dx * diff}; // Normalization by dx before returning
}

int main(int argc, char *argv[])
{
    TestCase test{TestCase::adv_gaussian};
    std::vector<double> s_vect {0.75, 1.0, 1.25, 1.5, 1.75};

    double eps = 0.1;
    std::size_t N_test = 50;
    double factor = 0.60;

    bool finest_collision = false; // Do you want to reconstruct also for the collision ?
    std::size_t min_level = 2;
    std::size_t max_level = 9;

    std::map<std::string, TestCase> tc_map{
        {"adv_gaussian", TestCase::adv_gaussian},
        {"adv_riemann", TestCase::adv_riemann},
        {"burgers_tanh", TestCase::burgers_tanh},
        {"burgers_hat", TestCase::burgers_hat},
        {"burgers_riemann", TestCase::burgers_riemann}
    };

    fs::path path = fs::current_path() / "d1q2";

    constexpr size_t dim = 1;
    using Config = samurai::MROConfig<dim, 2>;
    using mesh_t = samurai::MROMesh<Config>;
    using mesh_id_t = typename mesh_t::mesh_id_t;
    using coord_index_t = typename mesh_t::interval_t::coord_index_t;

    // We set some parameters according
    // to the problem.
    double sol_reg = 0.0;
    double T = 0.0;

    const double lambda = 1.; // Lattice velocity
    const double ad_vel = 0.75; // Should be < lambda

    switch(test)
    {
        case TestCase::adv_gaussian :
        {
            sol_reg = 600.0; // The solution is very smooth
            T = 0.4;
            break;
        }
        case TestCase::adv_riemann :
        {
            sol_reg = 0.0;
            T = 0.4;
            break;
        }
        case TestCase::burgers_tanh:
        {
            sol_reg = 600.0;
            // sol_reg = 1.0;
            T = 0.4;
            break;
        }
        case TestCase::burgers_hat :
        {
            sol_reg = 0.0;
            T = 1.3; // Let it develop the discontinuity
            break;
        }
        case TestCase::burgers_riemann :
        {
            sol_reg = 0.0;
            T = 0.7;
            break;
        }
    }

    samurai::Box<double, dim> box({-3}, {3});

    CLI::App app{"Multi resolution for a D1Q2 LBM scheme for Burgers equation"};
    app.add_option("--test", test, "Test case")->transform(CLI::CheckedTransformer(tc_map, CLI::ignore_case))->capture_default_str()->group("Simulation");
    app.add_option("--Tf", T, "final time")->capture_default_str()->group("Time behavior");
    app.add_option("--relax-sample", s_vect, "Relaxation sample used to study the time behavior")->capture_default_str()->group("Time behavior");
    app.add_option("--eps", eps, "First epsilon used by the multiresolution")->capture_default_str()->group("Epsilon behavior");
    app.add_option("--Neps", N_test, "Number of epsilon tests")->capture_default_str()->group("Epsilon behavior");
    app.add_option("--factor", factor, "Factor used to update epsilon at each iteration")->capture_default_str()->group("Epsilon behavior");
    app.add_flag("--with_fc", finest_collision, "Apply the reconstruction of the solution at the finest level during the collision")->capture_default_str()->group("Simulation");
    app.add_option("--min-level", min_level, "Minimum level of the multiresolution")->capture_default_str()->group("Multiresolution");
    app.add_option("--max-level", max_level, "Maximum level of the multiresolution")->capture_default_str()->group("Multiresolution");
    app.add_option("--path", path, "Output path")->capture_default_str()->group("Ouput");
    CLI11_PARSE(app, argc, argv);

    auto pred_coeff_separate = compute_prediction_separate_inout<coord_index_t>(min_level, max_level);

    auto update_bc_for_level = [](auto& field, std::size_t level)
    {
        update_bc_1D_constant_extension(field, level);
    };

    for (auto s : s_vect)
    {
        std::cout << "Relaxation parameter s = " << s << std::endl;

        std::string prefix = fmt::format("tc_{}_s_{}", test, s);

        auto time_path = path / "time";
        if (!fs::exists(time_path))
        {
            fs::create_directories(time_path);
        }

        std::cout << "Testing time behavior" << std::endl;
        {
            double mr_eps = 1.0e-4; // This remains fixed

            samurai::MROMesh<Config> mesh{box, min_level, max_level};
            samurai::MROMesh<Config> meshR{box, max_level, max_level}; // This is the reference scheme

            // Initialization
            auto f      = init_f(mesh , ad_vel, lambda, test);
            auto fR     = init_f(meshR, ad_vel, lambda, test);

            double dx = 1.0 / (1 << max_level);
            double dt = dx/lambda;

            std::size_t N = static_cast<std::size_t>(T / dt);

            double t = 0.0;

            std::ofstream out_time_frames(time_path / fmt::format("{}_time.dat", prefix));
            std::ofstream out_error_exact_ref(time_path / fmt::format("{}_error.dat", prefix));
            std::ofstream out_diff_ref_adap(time_path / fmt::format("{}_diff.dat", prefix));
            std::ofstream out_compression(time_path / fmt::format("{}_comp.dat", prefix));

            auto MRadaptation = samurai::make_MRAdapt(f, update_bc_for_level);

            for (std::size_t nb_ite = 0; nb_ite < N; ++nb_ite)
            {
                MRadaptation(mr_eps, sol_reg);

                auto error = compute_error(f, fR, update_bc_for_level, t, ad_vel, test);

                out_time_frames << t <<std::endl;
                out_error_exact_ref << error[0] << std::endl;
                out_diff_ref_adap << error[1] << std::endl;
                out_compression << static_cast<double>(mesh.nb_cells(mesh_id_t::cells))
                                 / static_cast<double>(meshR.nb_cells(mesh_id_t::cells)) << std::endl;

                std::cout << fmt::format("n = {}, Time = {}, Diff = {}", nb_ite, t, error[1]) << std::endl;

                one_time_step(f, update_bc_for_level, pred_coeff_separate, s, lambda, ad_vel, test, finest_collision);
                one_time_step(fR, update_bc_for_level, pred_coeff_separate, s, lambda, ad_vel, test);
                t += dt;
            }

            std::cout << std::endl;

            out_time_frames.close();
            out_error_exact_ref.close();
            out_diff_ref_adap.close();
            out_compression.close();
        }

        std::cout << std::endl << "Testing eps behavior" << std::endl;
        {
            auto eps_path = path / "eps";
            if (!fs::exists(eps_path))
            {
                fs::create_directories(eps_path);
            }

            std::ofstream out_eps(eps_path / fmt::format("{}_eps.dat", prefix));
            std::ofstream out_diff_ref_adap(eps_path / fmt::format("{}_diff.dat", prefix));
            std::ofstream out_compression(eps_path / fmt::format("{}_comp.dat", prefix));
            std::ofstream out_max_level(eps_path / fmt::format("{}_maxlevel.dat", prefix));

            for (std::size_t n_test = 0; n_test < N_test; ++ n_test)
            {
                std::cout << fmt::format("Test {} eps = {}", test, eps) << std::endl;

                mesh_t mesh{box, min_level, max_level};
                mesh_t meshR{box, max_level, max_level}; // This is the reference scheme

                // Initialization
                auto f  = init_f(mesh , ad_vel, lambda, test);
                auto fR = init_f(meshR, ad_vel, lambda, test);

                double dx = 1.0 / (1 << max_level);
                double dt = dx/lambda;

                std::size_t N = static_cast<std::size_t>(T / dt);

                double t = 0.0;

                auto MRadaptation = samurai::make_MRAdapt(f, update_bc_for_level);

                for (std::size_t nb_ite = 0; nb_ite < N; ++nb_ite)
                {
                    MRadaptation(eps, sol_reg);

                    one_time_step(f , update_bc_for_level, pred_coeff_separate, s, lambda, ad_vel, test, finest_collision);
                    one_time_step(fR, update_bc_for_level, pred_coeff_separate, s, lambda, ad_vel, test);
                    t += dt;
                }

                auto error = compute_error(f, fR, update_bc_for_level, t, ad_vel, test);
                std::cout << "Diff = " << error[1] << std::endl;

                std::size_t max_level_effective = mesh.min_level();

                for (std::size_t level = mesh.min_level() + 1; level <= mesh.max_level(); ++level)
                {
                    if (!mesh[mesh_id_t::cells][level].empty())
                    {
                        max_level_effective = level;
                    }
                }

                out_max_level << max_level_effective << std::endl;
                out_eps << eps << std::endl;
                out_diff_ref_adap << error[1] << std::endl;
                out_compression << static_cast<double>(mesh.nb_cells(mesh_id_t::cells))
                                 / static_cast<double>(meshR.nb_cells(mesh_id_t::cells)) << std::endl;

                eps *= factor;
            }

            out_eps.close();
            out_diff_ref_adap.close();
            out_compression.close();
            out_max_level.close();
        }
    }
    return 0;
}
