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

template <class coord_index_t>
auto compute_prediction(std::size_t min_level, std::size_t max_level)
{
    coord_index_t i = 0;
    std::vector<std::vector<prediction_map<coord_index_t>>> data(max_level - min_level + 1);

    for (std::size_t k = 0; k < max_level - min_level + 1; ++k)
    {
        int size = (1 << k);
        data[k].resize(2);

        data[k][0] = prediction(k, i * size - 1) - prediction(k, (i + 1) * size - 1);
        data[k][1] = prediction(k, (i + 1) * size) - prediction(k, i * size);
    }
    return data;
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

double exact_solution(double x, double t)
{
    double u = 0;

    if (x >= -1 and x < t)
    {
        u = (1 + x) / (1 + t);
    }

    if (x >= t and x < 1)
    {
        u = (1 - x) / (1 - t);
    }
    return u;
}

double flux(double u)
{
    return 0.5 * u * u; // Burgers
    // return 0.75 * u;
}

template <class Config>
auto init_f(samurai::MROMesh<Config>& mesh, const double lambda)
{
    using mesh_id_t            = typename samurai::MROMesh<Config>::mesh_id_t;
    constexpr std::size_t nvel = 2;

    auto f = samurai::make_field<double, nvel>("f", mesh);
    f.fill(0);

    samurai::for_each_cell(mesh[mesh_id_t::cells],
                           [&](auto& cell)
                           {
                               auto center = cell.center();
                               auto x      = center[0];

                               double u = exact_solution(x, 0.0);
                               double v = flux(u);

                               f[cell][0] = .5 * (u + v / lambda);
                               f[cell][1] = .5 * (u - v / lambda);
                           });
    return f;
}

template <class Field, class interval_t>
xt::xtensor<double, 1> prediction(const Field& f,
                                  std::size_t level_g,
                                  std::size_t level,
                                  const interval_t& i,
                                  const std::size_t item,
                                  std::map<std::tuple<std::size_t, std::size_t, std::size_t, interval_t>, xt::xtensor<double, 1>>& mem_map,
                                  bool cheap = false)
{
    // We check if the element is already in the map
    auto it = mem_map.find({item, level_g, level, i});
    if (it != mem_map.end())
    {
        // std::cout<<std::endl<<"Found by memoization";
        return it->second;
    }
    else
    {
        auto mesh                  = f.mesh();
        using mesh_id_t            = typename Field::mesh_t::mesh_id_t;
        xt::xtensor<double, 1> out = xt::empty<double>({i.size() / i.step}); // xt::eval(f(item, level_g, i));
        auto mask                  = mesh.exists(mesh_id_t::cells_and_ghosts, level_g + level, i);

        // std::cout << level_g + level << " " << i << " " << mask << "\n";
        if (xt::all(mask))
        {
            return xt::eval(f(item, level_g + level, i));
        }

        auto step                = i.step;
        auto ig                  = i / 2;
        ig.step                  = step >> 1;
        xt::xtensor<double, 1> d = xt::empty<double>({i.size() / i.step});

        for (int ii = i.start, iii = 0; ii < i.end; ii += i.step, ++iii)
        {
            d[iii] = (ii & 1) ? -1. : 1.;
        }

        xt::xtensor<double, 1> val;

        if (cheap)
        { // This is the cheap prediction
            val = xt::eval(prediction(f, level_g, level - 1, ig, item, mem_map, cheap));
        }
        else
        {
            val = xt::eval(prediction(f, level_g, level - 1, ig, item, mem_map, cheap)
                           - 1. / 8 * d
                                 * (prediction(f, level_g, level - 1, ig + 1, item, mem_map, cheap)
                                    - prediction(f, level_g, level - 1, ig - 1, item, mem_map, cheap)));
        }

        xt::masked_view(out, !mask) = xt::masked_view(val, !mask);
        for (int i_mask = 0, i_int = i.start; i_int < i.end; ++i_mask, i_int += i.step)
        {
            if (mask[i_mask])
            {
                out[i_mask] = f(item, level_g + level, {i_int, i_int + 1})[0];
            }
        }

        // The value should be added to the memoization map before returning
        return mem_map[{item, level_g, level, i}] = out;
    }
}

template <class Field, class Pred, class Func>
void one_time_step_overleaves(Field& f, const Pred& pred_coeff, Func&& update_bc_for_level, double s_rel, double lambda)
{
    constexpr std::size_t nvel = Field::size;

    auto mesh           = f.mesh();
    using mesh_t        = typename Field::mesh_t;
    using mesh_id_t     = typename mesh_t::mesh_id_t;
    using coord_index_t = typename mesh_t::interval_t::coord_index_t;
    using interval_t    = typename mesh_t::interval_t;

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
        if (level == max_level)
        {
            auto leaves = samurai::intersection(mesh[mesh_id_t::cells][max_level], mesh[mesh_id_t::cells][max_level]);
            leaves(
                [&](auto& interval, auto)
                {
                    auto k                      = interval;
                    advected_f(0, max_level, k) = xt::eval(f(0, max_level, k - 1));
                    advected_f(1, max_level, k) = xt::eval(f(1, max_level, k + 1));
                });
        }
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

                    auto fp = xt::eval(f(0, level + 1, k));
                    auto fm = xt::eval(f(1, level + 1, k));

                    for (auto& c : pred_coeff[j][0].coeff)
                    {
                        coord_index_t stencil = c.first;
                        double weight         = c.second;

                        fp += coeff * weight * f(0, level + 1, k + stencil);
                    }

                    for (auto& c : pred_coeff[j][1].coeff)
                    {
                        coord_index_t stencil = c.first;
                        double weight         = c.second;

                        fp -= coeff * weight * f(0, level + 1, k + stencil);
                    }

                    for (auto& c : pred_coeff[j][2].coeff)
                    {
                        coord_index_t stencil = c.first;
                        double weight         = c.second;

                        fm += coeff * weight * f(1, level + 1, k + stencil);
                    }

                    for (auto& c : pred_coeff[j][3].coeff)
                    {
                        coord_index_t stencil = c.first;
                        double weight         = c.second;

                        fm -= coeff * weight * f(1, level + 1, k + stencil);
                    }

                    // Save it
                    help_f(0, level + 1, k) = fp;
                    help_f(1, level + 1, k) = fm;
                });

            // Now that projection has been done, we have to come back on the
            // leaves below the overleaves
            auto leaves = samurai::intersection(mesh[mesh_id_t::cells][level], mesh[mesh_id_t::cells][level]);

            leaves(
                [&](auto& interval, auto)
                {
                    auto k = interval;
                    // Projection
                    advected_f(0, level, k) = xt::eval(0.5 * (help_f(0, level + 1, 2 * k) + help_f(0, level + 1, 2 * k + 1)));
                    advected_f(1, level, k) = xt::eval(0.5 * (help_f(1, level + 1, 2 * k) + help_f(1, level + 1, 2 * k + 1)));
                });
        }
    }

    for (std::size_t level = 0; level <= max_level; ++level)
    {
        auto leaves = samurai::intersection(mesh[mesh_id_t::cells][level], mesh[mesh_id_t::cells][level]);

        leaves(
            [&](auto& interval, auto)
            {
                auto k  = interval;
                auto uu = xt::eval(advected_f(0, level, k) + advected_f(1, level, k));
                auto vv = xt::eval(lambda * (advected_f(0, level, k) - advected_f(1, level, k)));

                vv = (1 - s_rel) * vv + s_rel * .5 * uu * uu;

                new_f(0, level, k) = .5 * (uu + 1. / lambda * vv);
                new_f(1, level, k) = .5 * (uu - 1. / lambda * vv);
            });
    }

    std::swap(f.array(), new_f.array());
}

template <class Field, class Func>
void one_time_step(Field& f, Func&& update_bc_for_level, double s)
{
    constexpr std::size_t nvel = Field::size;
    using mesh_id_t            = typename Field::mesh_t::mesh_id_t;

    double lambda  = 1.; //, s = 1.0;
    auto mesh      = f.mesh();
    auto max_level = mesh.max_level();

    samurai::update_ghost_mr(f, std::forward<Func>(update_bc_for_level));

    // MEMOIZATION
    // All is ready to do a little bit  of mem...
    using interval_t = typename Field::Config::interval_t;
    std::map<std::tuple<std::size_t, std::size_t, std::size_t, interval_t>, xt::xtensor<double, 1>> memoization_map;
    memoization_map.clear(); // Just to be sure...

    Field new_f{"new_f", mesh};
    new_f.array().fill(0.);

    for (std::size_t level = 0; level <= max_level; ++level)
    {
        auto exp = samurai::intersection(mesh[mesh_id_t::cells][level], mesh[mesh_id_t::cells][level]);
        exp(
            [&](auto, auto& interval, auto)
            {
                auto i = interval[0];

                // STREAM

                std::size_t j = max_level - level;

                double coeff = 1. / (1 << j);

                // This is the STANDARD FLUX EVALUATION

                bool cheap = false;

                auto fp = f(0, level, i)
                        + coeff
                              * (prediction(f, level, j, i * (1 << j) - 1, 0, memoization_map, cheap)
                                 - prediction(f, level, j, (i + 1) * (1 << j) - 1, 0, memoization_map, cheap));

                auto fm = f(1, level, i)
                        - coeff
                              * (prediction(f, level, j, i * (1 << j), 1, memoization_map, cheap)
                                 - prediction(f, level, j, (i + 1) * (1 << j), 1, memoization_map, cheap));

                // COLLISION

                auto uu = xt::eval(fp + fm);
                auto vv = xt::eval(lambda * (fp - fm));

                // vv = (1 - s) * vv + s * 0.75 * uu;

                vv = (1 - s) * vv + s * .5 * uu * uu;

                new_f(0, level, i) = .5 * (uu + 1. / lambda * vv);
                new_f(1, level, i) = .5 * (uu - 1. / lambda * vv);
            });
    }

    std::swap(f.array(), new_f.array());
}

template <class Field>
void save_solution(Field& f, double eps, std::size_t ite, std::string ext = "")
{
    auto mesh       = f.mesh();
    using value_t   = typename Field::value_type;
    using mesh_id_t = typename decltype(mesh)::mesh_id_t;

    std::size_t min_level = mesh.min_level();
    std::size_t max_level = mesh.max_level();

    std::stringstream str;
    str << "LBM_D1Q2_Burgers_" << ext << "_lmin_" << min_level << "_lmax-" << max_level << "_eps-" << eps << "_ite-" << ite;

    auto level_ = samurai::make_field<std::size_t, 1>("level", mesh);
    auto u      = samurai::make_field<value_t, 1>("u", mesh);

    samurai::for_each_cell(mesh[mesh_id_t::cells],
                           [&](auto& cell)
                           {
                               level_[cell] = static_cast<double>(cell.level);
                               u[cell]      = f[cell][0] + f[cell][1];
                           });

    samurai::save(str.str().data(), mesh, u, f, level_);
}

template <class Field, class FullField, class Func>
void save_reconstructed(Field& f, FullField& f_full, Func&& update_bc_for_level, double eps, std::size_t ite, std::string ext = "")
{
    constexpr std::size_t size = Field::size;
    using value_t              = typename Field::value_type;
    auto mesh                  = f.mesh();
    using mesh_id_t            = typename decltype(mesh)::mesh_id_t;

    auto min_level = mesh.min_level();
    auto max_level = mesh.max_level();

    auto init_mesh = f_full.mesh();

    samurai::update_ghost_mr(f, std::forward<Func>(update_bc_for_level));

    auto frec = samurai::make_field<value_t, size>("f_reconstructed", init_mesh);
    frec.fill(0.);

    using interval_t = typename Field::interval_t; // Type in X
    std::map<std::tuple<std::size_t, std::size_t, interval_t>, xt::xtensor<double, 2>> memoization_map;
    memoization_map.clear();

    for (std::size_t level = 0; level <= max_level; ++level)
    {
        auto exp = samurai::intersection(mesh[mesh_id_t::cells][level], mesh[mesh_id_t::cells][level]).on(max_level);

        exp(
            [&](auto& interval, auto)
            {
                auto i = interval;
                auto j = max_level - level;

                frec(max_level, i) = prediction_all(f, level, j, i, memoization_map);
            });
    }

    std::stringstream str;
    str << "LBM_D1Q2_Burgers_reconstructed_" << ext << "_lmin_" << min_level << "_lmax-" << max_level << "_eps-" << eps << "_ite-" << ite;

    auto u_rec  = samurai::make_field<value_t, 1>("u_reconstructed", init_mesh);
    auto u_full = samurai::make_field<value_t, 1>("u_full", init_mesh);

    samurai::for_each_cell(init_mesh[mesh_id_t::cells],
                           [&](auto& cell)
                           {
                               u_rec[cell]  = frec[cell][0] + frec[cell][1];
                               u_full[cell] = f_full[cell][0] + f_full[cell][1];
                           });

    samurai::save(str.str().data(), init_mesh, u_rec, u_full);
}

template <class Config, class FieldR, class Func>
std::array<double, 2> compute_error(samurai::Field<Config, double, 2>& f, FieldR& fR, Func&& update_bc_for_level, double t)
{
    auto mesh       = f.mesh();
    using mesh_id_t = typename decltype(mesh)::mesh_id_t;

    auto meshR     = fR.mesh();
    auto max_level = meshR.max_level();

    update_bc_for_level(fR, max_level); // It is important to do so

    samurai::update_ghost_mr(f, std::forward<Func>(update_bc_for_level));

    // Getting ready for memoization
    // using interval_t = typename Field::Config::interval_t;
    using interval_t = typename Config::interval_t;
    std::map<std::tuple<std::size_t, std::size_t, interval_t>, xt::xtensor<double, 2>> error_memoization_map;

    error_memoization_map.clear();

    double error = 0; // To return
    double diff  = 0.0;

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
                auto solR = xt::view(fR(max_level, i), xt::all(), xt::range(0, 2));

                xt::xtensor<double, 1> x      = dx * xt::linspace<int>(i.start, i.end - 1, i.size()) + 0.5 * dx;
                xt::xtensor<double, 1> uexact = xt::zeros<double>(x.shape());

                for (std::size_t idx = 0; idx < x.shape()[0]; ++idx)
                {
                    uexact[idx] = exact_solution(x[idx], t); // We can probably do better
                }

                auto rho_ref = xt::eval(fR(0, max_level, i) + fR(1, max_level, i));
                auto rho     = xt::eval(xt::view(sol, xt::all(), 0) + xt::view(sol, xt::all(), 1));

                error += xt::sum(xt::abs(rho_ref - uexact))[0];
                diff += xt::sum(xt::abs(rho_ref - rho))[0];
            });
    }
    return {dx * error, dx * diff}; // Normalization by dx before returning
}

int main(int argc, char* argv[])
{
    cxxopts::Options options("lbm_d1q2_burgers", "Multi resolution for a D1Q2 LBM scheme for Burgers equation");

    options.add_options()("min_level", "minimum level", cxxopts::value<std::size_t>()->default_value("2"))(
        "max_level",
        "maximum level",
        cxxopts::value<std::size_t>()->default_value("9"))("epsilon", "maximum level", cxxopts::value<double>()->default_value("0.0001"))(
        "s",
        "relaxation parameter",
        cxxopts::value<double>()->default_value("0.75"))("h, help", "Help");

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

            samurai::Box<double, dim> box({-3}, {3});
            samurai::MROMesh<Config> mesh{box, min_level, max_level};  // This is for the adaptive scheme
            samurai::MROMesh<Config> meshR{box, max_level, max_level}; // This is for the reference scheme

            auto pred_coeff_separate = compute_prediction_separate_inout<coord_index_t>(min_level, max_level);

            const double lambda     = 1.;
            const double regularity = 0.;

            double T  = 1.3;
            double dx = 1.0 / (1 << max_level);
            double dt = dx / lambda;

            // Initialization
            auto f  = init_f(mesh, lambda);
            auto fR = init_f(meshR, lambda);

            std::size_t N = static_cast<std::size_t>(T / dt);
            double t      = 0.0;

            std::ofstream out_time_frames;
            std::ofstream out_error_exact_ref;
            std::ofstream out_diff_ref_adap;
            std::ofstream out_compression;

            out_time_frames.open("./d1q2/time_frame_s_" + std::to_string(s) + "_eps_" + std::to_string(eps) + ".dat");
            out_error_exact_ref.open("./d1q2/error_exact_ref_" + std::to_string(s) + "_eps_" + std::to_string(eps) + ".dat");
            out_diff_ref_adap.open("./d1q2/diff_ref_adap_s_" + std::to_string(s) + "_eps_" + std::to_string(eps) + ".dat");
            out_compression.open("./d1q2/compression_s_" + std::to_string(s) + "_eps_" + std::to_string(eps) + ".dat");

            auto update_bc_for_level = [](auto& field, std::size_t level)
            {
                update_bc_1D_constant_extension(field, level);
            };

            auto MRadaptation = samurai::make_MRAdapt(f, update_bc_for_level);

            for (std::size_t nb_ite = 0; nb_ite < N; ++nb_ite)
            {
                MRadaptation(eps, regularity);

                save_solution(f, eps, nb_ite);
                save_reconstructed(f, fR, update_bc_for_level, eps, nb_ite);

                auto error = compute_error(f, fR, update_bc_for_level, t);

                std::cout << std::endl << "Diff = " << error[1] << std::flush;

                out_time_frames << t << std::endl;
                out_error_exact_ref << error[0] << std::endl;
                out_diff_ref_adap << error[1] << std::endl;
                out_compression << static_cast<double>(mesh.nb_cells(mesh_id_t::cells))
                                       / static_cast<double>(meshR.nb_cells(mesh_id_t::cells))
                                << std::endl;

                one_time_step_overleaves(f, pred_coeff_separate, update_bc_for_level, s, lambda);
                one_time_step_overleaves(fR, pred_coeff_separate, update_bc_for_level, s, lambda);

                t += dt;
            }

            out_time_frames.close();
            out_error_exact_ref.close();
            out_diff_ref_adap.close();
            out_compression.close();
        }
    }
    catch (const cxxopts::OptionException& e)
    {
        std::cout << options.help() << "\n";
    }
    return 0;
}
