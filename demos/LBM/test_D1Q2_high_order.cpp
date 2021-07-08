// Copyright 2021 SAMURAI TEAM. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include <math.h>
#include <vector>
#include <fstream>

#include <cxxopts.hpp>
#include <spdlog/spdlog.h>

#include <xtensor/xio.hpp>

#include <samurai/mr/adapt.hpp>
#include <samurai/mr/coarsening.hpp>
#include <samurai/mr/criteria.hpp>
#include <samurai/mr/harten.hpp>
#include <samurai/mr/mesh.hpp>
#include <samurai/mr/refinement.hpp>
#include <samurai/hdf5.hpp>

#include "prediction_map_1d.hpp"
#include "boundary_conditions.hpp"

#include "utils_lbm_mr_1d.hpp"

#include <chrono>

/*
TEST CASES
1 : transport - gaussienne
2 : transport - probleme de Riemann
3 : Burgers - tangente hyperbolique reguliere
4 : Burgers - fonction chapeau avec changement de regularite
5 : Burgers - probleme de Riemann
*/

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
    const auto toc_timer = std::chrono::high_resolution_clock::now();
    const std::chrono::duration<double> time_span = toc_timer - tic_timer;
    return time_span.count();
}

double exact_solution(double x, double t, double ad_vel, int test_number)
{
    double u = 0;

    switch(test_number) {
        case 1 : {
            u = exp(-20.0 * (x-ad_vel*t) * (x-ad_vel*t)); // Used in the first draft
            // u = exp(-60.0 * (x-ad_vel*t) * (x-ad_vel*t));
            break;
        }

        case 2 : {
            double sigma = 0.5;
            double rhoL = 0.0;
            double rhoC = 1.0;
            double rhoR = 0.0;

            double xtr = x - ad_vel*t;
            u =  (xtr <= -sigma) ? (rhoL) : ((xtr <= sigma) ? (rhoC) : rhoR );
            break;
        }
        case 3 : {
            double sigma = 100.0;
            if (t <= 0.0)
                u = 0.5 * (1.0 + tanh(sigma * x));
            else
            {   // We proceed by dicothomy
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

        case 4 : {
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

        case 5 : {
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

double flux(double u, double ad_vel, int test_number)   {

    if (test_number == 1 or test_number == 2)   { // Advection
        return ad_vel * u;
    }
    else // Burgers
    {
        return 0.5 * u *u;
    }
}

template<class Config>
auto init_f(samurai::MRMesh<Config> &mesh, double t, double ad_vel, double lambda, int test_number)
{
    constexpr std::size_t nvel = 2;
    using mesh_id_t = typename samurai::MRMesh<Config>::mesh_id_t;

    auto f = samurai::make_field<double, nvel>("f", mesh);
    f.fill(0);

    samurai::for_each_cell(mesh[mesh_id_t::cells], [&](auto &cell)
    {
        auto center = cell.center();
        auto x = center[0];

        double u = 0;

        u = exact_solution(x, 0.0, ad_vel, test_number);
        double v = flux(u, ad_vel, test_number);

        f[cell][0] = .5 * (u + v/lambda);
        f[cell][1] = .5 * (u - v/lambda);
    });

    return f;
}

template<class Field, class interval_t>
xt::xtensor<double, 1> prediction(const Field& f, std::size_t level_g, std::size_t level, const interval_t &i, const std::size_t item,
                                  std::map<std::tuple<std::size_t, std::size_t, std::size_t, interval_t>,
                                  xt::xtensor<double, 1>> & mem_map, bool cheap = false)
{
    // mem_map.clear();

    // We check if the element is already in the map
    auto it = mem_map.find({item, level_g, level, i});
    if (it != mem_map.end())   {
        // std::cout<<std::endl<<"Found by memoization = "<<level_g<<"\t"<<level<<"\t"<<i<<std::endl;
        // std::cout<<"What has been found "<<std::get<0>(it->first)<<"\t"<<std::get<1>(it->first)<<"\t"<<std::get<2>(it->first)<<"\t"<<std::get<3>(it->first)<<std::endl;
        return it->second;
    }
    else {

        auto mesh = f.mesh();
        using mesh_id_t = typename Field::mesh_t::mesh_id_t;
        xt::xtensor<double, 1> out = xt::empty<double>({i.size()/i.step});//xt::eval(f(item, level_g, i));
        auto mask = mesh.exists(mesh_id_t::cells_and_ghosts, level_g + level, i);

        // std::cout << level_g + level << " " << i << " " << mask << "\n";
        if (xt::all(mask))
        {
            // std::cout << "all true " <<  level_g + level << " " << i << std::endl;
            return xt::eval(f(item, level_g + level, i));
        }

        auto step = i.step;
        auto ig = i / 2;
        ig.step = step >> 1;
        xt::xtensor<double, 1> d = xt::empty<double>({i.size()/i.step});

        // std::cout<<"Info "<<i<<"\t"<<d.size()<<std::endl;

        for (int ii=i.start, iii=0; ii<i.end; ii+=i.step, ++iii)
        {
            d[iii] = (ii & 1)? -1.: 1.;
        }



        // xt::xtensor<double, 1> val;

        // if (cheap)  {  // This is the cheap prediction
        //     val = xt::eval(prediction(f, level_g, level-1, ig, item, mem_map, cheap));
        // }
        // else {
            // val = xt::eval(prediction(f, level_g, level-1, ig, item, mem_map, cheap) - 1./8 * d * (prediction(f, level_g, level-1, ig+1, item, mem_map, cheap)
            //                                                                            - prediction(f, level_g, level-1, ig-1, item, mem_map, cheap)));

            // std::cout<<"Hello "<<level_g<<"\t"<<level<<"\t"<<ig<<"\t"<<i<<std::endl;

            auto c = prediction(f, level_g, level-1, ig, item, mem_map, cheap);
            auto cp1 = prediction(f, level_g, level-1, ig+1, item, mem_map, cheap);
            auto cm1 = prediction(f, level_g, level-1, ig-1, item, mem_map, cheap);
            auto cp2 = prediction(f, level_g, level-1, ig+2, item, mem_map, cheap);
            auto cm2 = prediction(f, level_g, level-1, ig-2, item, mem_map, cheap);

            // std::cout << fmt::format("c = {}, cp1 = {}, cm1 = {}, cp2 = {}, cm2 = {}", c.size(), cp1.size(), cm1.size(), cp2.size(), cm2.size()) << std::endl;


            auto val = xt::eval(c- 22./128 * d * (cp1 - cm1)  +  3./128 * d * (cp2 - cm2));

            // val = xt::eval(prediction(f, level_g, level-1, ig, item, mem_map, cheap) - 22./128 * d * (prediction(f, level_g, level-1, ig+1, item, mem_map, cheap)
            //                                                                                         - prediction(f, level_g, level-1, ig-1, item, mem_map, cheap))
            //                                                                          +  3./128 * d * (prediction(f, level_g, level-1, ig+2, item, mem_map, cheap)
            //                                                                                         - prediction(f, level_g, level-1, ig-2, item, mem_map, cheap)));
        // }


        xt::masked_view(out, !mask) = xt::masked_view(val, !mask);
        for(int i_mask=0, i_int=i.start; i_int<i.end; ++i_mask, i_int+=i.step)
        {
            if (mask[i_mask])
            {
                out[i_mask] = f(item, level_g + level, {i_int, i_int + 1})[0];
            }
        }

        // The value should be added to the memoization map before returning
        mem_map.insert(std::make_pair(std::tuple<std::size_t, std::size_t, std::size_t, interval_t>{item, level_g, level, i} ,out));
        return out;
    }

}



// Attention : the number 2 as second template parameter does not mean
// that we are dealing with two fields!!!!
template<class Field, class interval_t>
xt::xtensor<double, 2> prediction_all_high_order(const Field & f, std::size_t level_g, std::size_t level,
                                                const interval_t & k,
                                                std::map<std::tuple<std::size_t, std::size_t, interval_t>, xt::xtensor<double, 2>> & mem_map)
{
    constexpr std::size_t nvel = Field::size; // Number of velocities
    // That is used to employ _ with xtensor
    using namespace xt::placeholders;

    auto it = mem_map.find({level_g, level, k});

    if (it != mem_map.end() && k.size() == (std::get<2>(it->first)).size())
        return it->second;
    else
    {
        auto mesh = f.mesh();
        using mesh_id_t = typename decltype(mesh)::mesh_id_t;

        std::vector<std::size_t> shape_x = {k.size(), nvel};
        xt::xtensor<double, 2> out = xt::empty<double>(shape_x);

        auto mask = mesh.exists(mesh_id_t::cells_and_ghosts, level_g + level, k); // Check if we are on a leaf or a ghost (CHECK IF IT IS OK)

        xt::xtensor<double, 2> mask_all = xt::empty<double>(shape_x);

        for (int h_field = 0; h_field < nvel; ++h_field)  {
            xt::view(mask_all, xt::all(), h_field) = mask;
        }

        // Recursion finished
        if (xt::all(mask))
        {
            return xt::eval(f(0, nvel, level_g + level, k));
        }

        // If we cannot stop here
        auto kg = k >> 1;
        kg.step = 1;

        xt::xtensor<double, 2> val = xt::empty<double>(shape_x);

        auto earth  = xt::eval(prediction_all_high_order(f, level_g, level - 1, kg     , mem_map));
        auto W      = xt::eval(prediction_all_high_order(f, level_g, level - 1, kg - 1 , mem_map));
        auto E      = xt::eval(prediction_all_high_order(f, level_g, level - 1, kg + 1 , mem_map));

        auto WW      = xt::eval(prediction_all_high_order(f, level_g, level - 1, kg - 2 , mem_map));
        auto EE      = xt::eval(prediction_all_high_order(f, level_g, level - 1, kg + 2 , mem_map));

        // This is to deal with odd/even indices in the x direction
        std::size_t start_even = (k.start & 1) ?     1         :     0        ;
        std::size_t start_odd  = (k.start & 1) ?     0         :     1        ;
        std::size_t end_even   = (k.end & 1)   ? kg.size()     : kg.size() - 1;
        std::size_t end_odd    = (k.end & 1)   ? kg.size() - 1 : kg.size()    ;

        xt::view(val, xt::range(start_even, _, 2)) = xt::view(earth + 22./128 * (W - E) - 3./128 * (WW - EE), xt::range(start_even, _      ));
        xt::view(val, xt::range(start_odd, _, 2))  = xt::view(earth - 22./128 * (W - E) + 3./128 * (WW - EE), xt::range(_         , end_odd));

        xt::masked_view(out, !mask_all) = xt::masked_view(val, !mask_all);

        for(int k_mask = 0, k_int = k.start; k_int < k.end; ++k_mask, ++k_int)
        {
            if (mask[k_mask])
                xt::view(out, k_mask) = xt::view(f(0, nvel, level_g + level, {k_int, k_int + 1}), 0);
        }

        // It is crucial to use insert and not []
        // in order not to update the value in case of duplicated (same key)
        mem_map.insert(std::make_pair(std::tuple<std::size_t, std::size_t, interval_t>{level_g, level, k} ,out));
        return out;
    }
}



template<class Field, class Func>
void one_time_step(Field &f, Func&& update_bc_for_level,
                            double s_rel, double lambda, double ad_vel, int test_number, bool finest_collision = false)
{
    constexpr std::size_t nvel = Field::size;

    auto mesh = f.mesh();
    using mesh_t = typename Field::mesh_t;
    using mesh_id_t = typename mesh_t::mesh_id_t;
    using coord_index_t = typename mesh_t::interval_t::coord_index_t;
    using interval_t = typename mesh_t::interval_t;

    auto min_level = mesh.min_level();
    auto max_level = mesh.max_level();

    samurai::update_ghost_mr(f, std::forward<Func>(update_bc_for_level));

    // MEMOIZATION
    // All is ready to do a little bit  of mem...
    std::map<std::tuple<std::size_t, std::size_t, std::size_t, interval_t>, xt::xtensor<double, 1>> memoization_map;
    memoization_map.clear(); // Just to be sure...

    auto advected_f = samurai::make_field<double, nvel>("advected_f", mesh);
    advected_f.fill(0.);

    for (std::size_t level = 0; level <= max_level; ++level)
    {
        std::size_t j = max_level - level;
        double coeff = 1. / (1 << j);

        auto leaves = samurai::intersection(mesh[mesh_id_t::cells][level],
                                            mesh[mesh_id_t::cells][level]);
        leaves([&](auto &interval, auto) {
            auto i = interval;

            advected_f(0, level, i) = f(0, level, i) + coeff * (prediction(f, level, j, i*(1<<j)-1, 0, memoization_map, false)
                                                             -  prediction(f, level, j, (i+1)*(1<<j)-1, 0, memoization_map, false));

            advected_f(1, level, i) = f(1, level, i) - coeff * (prediction(f, level, j, i*(1<<j), 1, memoization_map, false)
                                                             -  prediction(f, level, j, (i+1)*(1<<j), 1, memoization_map, false));
        });
    }

    if (!finest_collision)  {
    for (std::size_t level = 0; level <= max_level; ++level)    {

        auto leaves = samurai::intersection(mesh[mesh_id_t::cells][level],
                                            mesh[mesh_id_t::cells][level]);

        leaves([&](auto &interval, auto) {
            auto k = interval;
            auto uu = xt::eval(          advected_f(0, level, k) + advected_f(1, level, k));
            auto vv = xt::eval(lambda * (advected_f(0, level, k) - advected_f(1, level, k)));

            if (test_number == 1 or test_number == 2)
            {
                vv = (1 - s_rel) * vv + s_rel * ad_vel * uu;
            }
            else
            {
                vv = (1 - s_rel) * vv + s_rel * .5 * uu * uu;
            }

            f(0, level, k) = .5 * (uu + 1. / lambda * vv);
            f(1, level, k) = .5 * (uu - 1. / lambda * vv);
        });
    }
    }
    else {
 samurai::update_ghost_mr(advected_f, std::forward<Func>(update_bc_for_level));

        std::map<std::tuple<std::size_t, std::size_t, interval_t>, xt::xtensor<double, 2>> memoization_map;
        memoization_map.clear();

        for (std::size_t level = 0; level <= max_level; ++level)    {

            auto leaves_on_finest = samurai::intersection(mesh[mesh_id_t::cells][level],
                                                          mesh[mesh_id_t::cells][level]).on(max_level);

            leaves_on_finest([&](auto &interval, auto) {
                auto i = interval;
                auto j = max_level - level;

                auto f_on_finest  = prediction_all_high_order(advected_f, level, j, i, memoization_map);

                auto uu = xt::eval(xt::view(f_on_finest, xt::all(), 0)
                                 + xt::view(f_on_finest, xt::all(), 1));

                auto vv = xt::eval(lambda*(xt::view(f_on_finest, xt::all(), 0)
                                         - xt::view(f_on_finest, xt::all(), 1)));

                if (test_number == 1 or test_number == 2)   {

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
                    f(0, level, {(i.start + i_start)/step, (i.start + i_start)/step + 1}) = xt::mean(xt::view(f_0_post_coll, xt::range(i_start, i_start + step)));
                    f(1, level, {(i.start + i_start)/step, (i.start + i_start)/step + 1}) = xt::mean(xt::view(f_1_post_coll, xt::range(i_start, i_start + step)));
                }
            });
        }
    }
}

template<class Config, class FieldR, class Func>
std::array<double, 3> compute_error(samurai::Field<Config, double, 2> &f, FieldR & fR, Func&& update_bc_for_level, double t, double ad_vel, int test_number)
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

    double error_ref = 0; // To return
    double error_ad = 0; // To return
    double diff = 0.0;

    double dx = 1.0 / (1 << max_level);

    for (std::size_t level = 0; level <= max_level; ++level)
    {
        auto exp = samurai::intersection(mesh[mesh_id_t::cells][level],
                                         mesh[mesh_id_t::cells][level]).on(max_level);

        exp([&](auto &interval, auto) {
            auto i = interval;
            auto j = max_level - level;

            auto sol  = prediction_all_high_order(f, level, j, i, error_memoization_map);
            auto solR = xt::view(fR(max_level, i), xt::all(), xt::range(0, 2));

            xt::xtensor<double, 1> x = dx*xt::linspace<int>(i.start, i.end - 1, i.size()) + 0.5*dx;
            xt::xtensor<double, 1> uexact = xt::zeros<double>(x.shape());

            for (std::size_t idx = 0; idx < x.shape()[0]; ++idx)    {
                uexact[idx] = exact_solution(x[idx], t, ad_vel, test_number); // We can probably do better
            }

            auto rho_ref = xt::eval(fR(0, max_level, i) + fR(1, max_level, i));
            auto rho = xt::eval(xt::view(sol, xt::all(), 0) +  xt::view(sol, xt::all(), 1));

            error_ref += xt::sum(xt::abs(rho_ref - uexact))[0];
            error_ad += xt::sum(xt::abs(rho - uexact))[0];
            diff  += xt::sum(xt::abs(rho_ref - rho))[0];
        });
    }
    return {dx * error_ref, dx * error_ad, dx * diff}; // Normalization by dx before returning
}

int main(int argc, char *argv[])
{
    cxxopts::Options options("lbm_d1q2_burgers",
                             "Multi resolution for a D1Q2 LBM scheme for Burgers equation");

    options.add_options()
                       ("min_level", "minimum level", cxxopts::value<std::size_t>()->default_value("2"))
                       ("max_level", "maximum level", cxxopts::value<std::size_t>()->default_value("10"))
                       ("epsilon", "maximum level", cxxopts::value<double>()->default_value("0.01"))
                       ("s", "relaxation parameter", cxxopts::value<double>()->default_value("1.0"))
                       ("log", "log level", cxxopts::value<std::string>()->default_value("warning"))
                       ("test", "test number", cxxopts::value<int>()->default_value("1"))
                       ("h, help", "Help");

    try
    {
        auto result = options.parse(argc, argv);

        if (result.count("help"))
            std::cout << options.help() << "\n";
        else
        {
            std::map<std::string, spdlog::level::level_enum> log_level{{"debug", spdlog::level::debug},
                                                               {"warning", spdlog::level::warn}};
            constexpr size_t dim = 1;
            using Config = samurai::MRConfig<dim, 4, 2>;
            using mesh_t = samurai::MRMesh<Config>;
            using mesh_id_t = typename mesh_t::mesh_id_t;
            using coord_index_t = typename mesh_t::interval_t::coord_index_t;

            spdlog::set_level(log_level[result["log"].as<std::string>()]);
            // std::size_t min_level = 4;//result["min_level"].as<std::size_t>();

            std::size_t min_level = 2;//result["min_level"].as<std::size_t>();
            std::size_t max_level = 9;//result["max_level"].as<std::size_t>();
            int test_number = result["test"].as<int>();
            const bool finest_collision = true; // Do you want to reconstruct also for the collision ?

            // We set some parameters according
            // to the problem.
            double sol_reg = 0.0;
            double T = 0.0;
            std::string case_name;

            const double lambda = 1.; // Lattice velocity
            const double ad_vel = 0.75; // Should be < lambda

            switch(test_number){
                case 1 : {
                    sol_reg = 600.0; // The solution is very smooth
                    T = 0.4;
                    case_name = std::string("t_r"); break;
                }
                case 2 : {
                    sol_reg = 0.0;
                    T = 0.4;
                    case_name = std::string("t_d"); break;
                }
                case 3 : {
                    sol_reg = 600.0;
                    // sol_reg = 1.0;
                    T = 0.4;
                    case_name = std::string("b_r"); break;
                }
                case 4 : {
                    sol_reg = 0.0;
                    T = 1.3; // Let it develop the discontinuity
                    case_name = std::string("b_c"); break;
                }
                case 5 : {
                    sol_reg = 0.0;
                    T = 0.7;
                    case_name = std::string("b_d"); break;
                }
            }

            samurai::Box<double, dim> box({-3}, {3});

            std::vector<double> s_vect {0.75, 1.0, 1.25, 1.5, 1.75};

            auto update_bc_for_level = [](auto& field, std::size_t level)
            {
                // update_bc_1D_constant_extension(field, level);
                update_bc_1D_constant_extension_four_ghosts(field, level);
            };

            for (auto s : s_vect)   {
                std::cout<<std::endl<<"Relaxation parameter s = "<<s;

                std::string prefix (case_name + "_s_"+std::to_string(s)+"_");

                std::cout<<std::endl<<"Testing time behavior"<<std::endl;
                // {
                //     double eps = 1.0e-4; // This remains fixed

                //     samurai::MRMesh<Config> mesh{box, min_level, max_level};
                //     samurai::MRMesh<Config> meshR{box, max_level, max_level}; // This is the reference scheme

                //     // Initialization
                //     auto f      = init_f(mesh , 0.0, ad_vel, lambda, test_number);
                //     auto fR     = init_f(meshR, 0.0, ad_vel, lambda, test_number);

                //     double dx = 1.0 / (1 << max_level);
                //     double dt = dx/lambda;

                //     std::size_t N = static_cast<std::size_t>(T / dt);

                //     double t = 0.0;

                //     std::ofstream out_time_frames;
                //     std::ofstream out_error_exact_ref;
                //     std::ofstream out_error_exact_ad;
                //     std::ofstream out_diff_ref_adap;
                //     std::ofstream out_compression;

                //     out_time_frames.open     ("./d1q2/time/"+prefix+"time.dat");
                //     out_error_exact_ref.open ("./d1q2/time/"+prefix+"E.dat");
                //     out_error_exact_ad.open ("./d1q2/time/"+prefix+"e.dat");
                //     out_diff_ref_adap.open   ("./d1q2/time/"+prefix+"delta.dat");
                //     out_compression.open     ("./d1q2/time/"+prefix+"comp.dat");

                //     auto MRadaptation = samurai::make_MRAdapt(f, update_bc_for_level);

                //     for (std::size_t nb_ite = 0; nb_ite < N; ++nb_ite)
                //     {
                //         MRadaptation(eps, sol_reg);

                //         auto error = compute_error(f, fR, update_bc_for_level, t, ad_vel, test_number);

                //         out_time_frames    <<t       <<std::endl;
                //         out_error_exact_ref<<error[0]<<std::endl;
                //         out_error_exact_ad<<error[1]<<std::endl;
                //         out_diff_ref_adap  <<error[2]<<std::endl;
                //         out_compression    <<static_cast<double>(mesh.nb_cells(mesh_id_t::cells))
                //                            / static_cast<double>(meshR.nb_cells(mesh_id_t::cells))<<std::endl;

                //         std::cout<<std::endl<<"n = "<<nb_ite<<"   Time = "<<t<<" Diff = "<<error[1];

                //         one_time_step(f, update_bc_for_level, s, lambda, ad_vel, test_number, finest_collision);
                //         one_time_step(fR, update_bc_for_level, s, lambda, ad_vel, test_number);
                //         t += dt;
                //     }

                //     std::cout<<std::endl;

                //     out_time_frames.close();
                //     out_error_exact_ref.close();
                //     out_error_exact_ad.close();
                //     out_diff_ref_adap.close();
                //     out_compression.close();
                // }

                std::cout<<std::endl<<"Testing eps behavior"<<std::endl;
                {
                    double eps = 0.1;
                    std::size_t N_test = 50;
                    double factor = 0.60;
                    std::ofstream out_eps;
                    std::ofstream out_error_adap;
                    std::ofstream out_diff_ref_adap;
                    std::ofstream out_compression;
                    std::ofstream out_max_level;
                    std::ofstream out_compression_avg;


                    out_eps.open             ("./d1q2/eps/"+prefix+"eps.dat");
                    out_error_adap.open   ("./d1q2/eps/"+prefix+"e.dat");
                    out_diff_ref_adap.open   ("./d1q2/eps/"+prefix+"diff.dat");
                    out_compression_avg.open     ("./d1q2/eps/"+prefix+"comp_avf.dat");
                    out_compression.open     ("./d1q2/eps/"+prefix+"comp.dat");
                    out_max_level.open       ("./d1q2/eps/"+prefix+"maxlevel.dat");

                    for (std::size_t n_test = 0; n_test < N_test; ++ n_test)    {
                        std::cout<<std::endl<<"Test "<<n_test<<" eps = "<<eps;

                        mesh_t mesh{box, min_level, max_level};
                        mesh_t meshR{box, max_level, max_level}; // This is the reference scheme

                        // Initialization
                        auto f  = init_f(mesh , 0.0, ad_vel, lambda, test_number);
                        auto fR = init_f(meshR, 0.0, ad_vel, lambda, test_number);

                        double dx = 1.0 / (1 << max_level);
                        double dt = dx/lambda;

                        std::size_t N = static_cast<std::size_t>(T / dt);

                        double t = 0.0;

                        auto MRadaptation = samurai::make_MRAdapt(f, update_bc_for_level);
                        double comp_avg = 0.;

                        // for (std::size_t nb_ite = 0; nb_ite < N; ++nb_ite)
                        // for (std::size_t nb_ite = 0; nb_ite < N; ++nb_ite)
                        for (std::size_t nb_ite = 0; nb_ite < 1; ++nb_ite)

                        {
                            MRadaptation(eps, sol_reg);
                            comp_avg += (static_cast<double>(mesh.nb_cells(mesh_id_t::cells))
                                           / static_cast<double>(meshR.nb_cells(mesh_id_t::cells)))/N;

                            // one_time_step(f , update_bc_for_level, s, lambda, ad_vel, test_number, finest_collision);
                            // one_time_step(fR, update_bc_for_level, s, lambda, ad_vel, test_number);
                            t += dt;
                        }

                        auto error = compute_error(f, fR, update_bc_for_level, t, ad_vel, test_number);
                        std::cout<<"Diff = "<<error[2]<<std::endl;

                        std::size_t max_level_effective = mesh.min_level();

                        for (std::size_t level = mesh.min_level() + 1; level <= mesh.max_level(); ++level)  {
                            if (!mesh[mesh_id_t::cells][level].empty())
                                max_level_effective = level;
                        }

                        out_max_level<<max_level_effective<<std::endl;

                        out_eps<<eps<<std::endl;
                        out_error_adap<<error[1]<<std::endl;
                        out_diff_ref_adap<<error[2]<<std::endl;
                        out_compression<<static_cast<double>(mesh.nb_cells(mesh_id_t::cells))
                                           / static_cast<double>(meshR.nb_cells(mesh_id_t::cells))<<std::endl;
                        out_compression_avg<<comp_avg<<std::endl;

                        eps *= factor;
                    }

                    out_eps.close();
                    out_error_adap.close();
                    out_compression_avg.close();
                    out_diff_ref_adap.close();
                    out_compression.close();
                    out_max_level.close();
                }
            }
        }
    }

    catch (const cxxopts::OptionException &e)
    {
        std::cout << options.help() << "\n";
    }



    return 0;
}
