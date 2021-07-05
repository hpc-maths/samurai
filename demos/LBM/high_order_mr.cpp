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

template<class Field>
void init_f(Field & field, const double lambda, const double ad_vel)
{
    using mesh_t = typename Field::mesh_t;
    using mesh_id_t = typename mesh_t::mesh_id_t;

    auto mesh = field.mesh();
    field.fill(0);

    samurai::for_each_cell(mesh[mesh_id_t::cells], [&](auto &cell)
    {
        auto center = cell.center();
        auto x = center[0];

        double u = std::exp(-20*x*x);
        double v = ad_vel * u;

        field[cell][0] = .5 * (u + v/lambda);
        field[cell][1] = .5 * (u - v/lambda);
    });
}



// template<class Field, class interval_t>
// xt::xtensor<double, 1> prediction(const Field& f, std::size_t level_g, std::size_t level, const interval_t &i, const std::size_t item,
//                                   std::map<std::tuple<std::size_t, std::size_t, std::size_t, interval_t>,
//                                   xt::xtensor<double, 1>> & mem_map, bool cheap = false)
// {
//     mem_map.clear();

//     // We check if the element is already in the map
//     auto it = mem_map.find({item, level_g, level, i});
//     if (it != mem_map.end())   {
//         //std::cout<<std::endl<<"Found by memoization";
//         return it->second;
//     }
//     else {

//         auto mesh = f.mesh();
//         using mesh_id_t = typename Field::mesh_t::mesh_id_t;
//         xt::xtensor<double, 1> out = xt::empty<double>({i.size()/i.step});//xt::eval(f(item, level_g, i));
//         auto mask = mesh.exists(mesh_id_t::cells_and_ghosts, level_g + level, i);

//         // std::cout << level_g + level << " " << i << " " << mask << "\n";
//         if (xt::all(mask))
//         {
//             return xt::eval(f(item, level_g + level, i));
//         }

//         auto step = i.step;
//         auto ig = i / 2;
//         ig.step = step >> 1;
//         xt::xtensor<double, 1> d = xt::empty<double>({i.size()/i.step});

//         for (int ii=i.start, iii=0; ii<i.end; ii+=i.step, ++iii)
//         {
//             d[iii] = (ii & 1)? -1.: 1.;
//         }


//         xt::xtensor<double, 1> val;

//         if (cheap)  {  // This is the cheap prediction
//             val = xt::eval(prediction(f, level_g, level-1, ig, item, mem_map, cheap));
//         }
//         else {
//             // val = xt::eval(prediction(f, level_g, level-1, ig, item, mem_map, cheap) - 1./8 * d * (prediction(f, level_g, level-1, ig+1, item, mem_map, cheap)
//             //                                                                            - prediction(f, level_g, level-1, ig-1, item, mem_map, cheap)));

//             val = xt::eval(prediction(f, level_g, level-1, ig, item, mem_map, cheap) - 1./8 * d * (prediction(f, level_g, level-1, ig+1, item, mem_map, cheap)
//                                                                                                  - prediction(f, level_g, level-1, ig-1, item, mem_map, cheap)));
//         }


//         xt::masked_view(out, !mask) = xt::masked_view(val, !mask);
//         for(int i_mask=0, i_int=i.start; i_int<i.end; ++i_mask, i_int+=i.step)
//         {
//             if (mask[i_mask])
//             {
//                 out[i_mask] = f(item, level_g + level, {i_int, i_int + 1})[0];
//             }
//         }

//         // The value should be added to the memoization map before returning
//         mem_map.insert(std::make_pair(std::tuple<std::size_t, std::size_t, std::size_t, interval_t>{item, level_g, level, i} ,out));
//         return out;
//     }

// }


// template<class Field, class Func>
// void one_time_step(Field &f, Func&& update_bc_for_level, double s)
// {
//     auto mesh = f.mesh();
//     using mesh_t = typename Field::mesh_t;
//     using mesh_id_t = typename mesh_t::mesh_id_t;
//     using coord_index_t = typename mesh_t::interval_t::coord_index_t;
//     using interval_t = typename mesh_t::interval_t;

//     constexpr std::size_t nvel = Field::size;

//     double lambda = 1.;//, s = 1.0;
//     auto max_level = mesh.max_level();

//     samurai::update_ghost_mr(f, std::forward<Func>(update_bc_for_level));

//     // MEMOIZATION
//     // All is ready to do a little bit  of mem...
//     std::map<std::tuple<std::size_t, std::size_t, std::size_t, interval_t>, xt::xtensor<double, 1>> memoization_map;
//     memoization_map.clear(); // Just to be sure...

//     Field new_f{"new_f", mesh};
//     new_f.array().fill(0.);

//     for (std::size_t level = 0; level <= max_level; ++level)
//     {
//         auto exp = samurai::intersection(mesh[mesh_id_t::cells][level],
//                                          mesh[mesh_id_t::cells][level]);
//         exp([&](auto &interval, auto) {
//             auto i = interval;


//             // STREAM

//             std::size_t j = max_level - level;

//             double coeff = 1. / (1 << j);

//             // This is the STANDARD FLUX EVALUATION

//             bool cheap = false;

//             auto fp = f(0, level, i) + coeff * (prediction(f, level, j, i*(1<<j)-1, 0, memoization_map, cheap)
//                                              -  prediction(f, level, j, (i+1)*(1<<j)-1, 0, memoization_map, cheap));

//             auto fm = f(1, level, i) - coeff * (prediction(f, level, j, i*(1<<j), 1, memoization_map, cheap)
//                                              -  prediction(f, level, j, (i+1)*(1<<j), 1, memoization_map, cheap));


//             // COLLISION

//             auto uu = xt::eval(fp + fm);
//             auto vv = xt::eval(lambda * (fp - fm));


//             vv = (1 - s) * vv + s * 0.75 * uu;

//             // vv = (1 - s) * vv + s * .5 * uu * uu;

//             new_f(0, level, i) = .5 * (uu + 1. / lambda * vv);
//             new_f(1, level, i) = .5 * (uu - 1. / lambda * vv);
//         });
//     }

//     std::swap(f.array(), new_f.array());
// }

// template<class Field, class Func, class Pred>
// void one_time_step(Field &f, Func&& update_bc_for_level,
//                             const Pred& pred_coeff, double s_rel, double lambda, double ad_vel)
// {

//     constexpr std::size_t nvel = Field::size;

//     auto mesh = f.mesh();
//     using mesh_t = typename Field::mesh_t;
//     using mesh_id_t = typename mesh_t::mesh_id_t;
//     using coord_index_t = typename mesh_t::interval_t::coord_index_t;
//     using interval_t = typename mesh_t::interval_t;

//     auto min_level = mesh.min_level();
//     auto max_level = mesh.max_level();

//     samurai::update_ghost_mr(f, std::forward<Func>(update_bc_for_level));
//     samurai::update_overleaves_mr(f, std::forward<Func>(update_bc_for_level));

//     auto new_f = samurai::make_field<double, nvel>("new_f", mesh);
//     new_f.fill(0.);
//     auto advected_f = samurai::make_field<double, nvel>("advected_f", mesh);
//     advected_f.fill(0.);
//     auto help_f = samurai::make_field<double, nvel>("help_f", mesh);
//     help_f.fill(0.);

//     for (std::size_t level = 0; level <= max_level; ++level)
//     {
//         if (level == max_level) {
//             auto leaves = samurai::intersection(mesh[mesh_id_t::cells][max_level],
//                                                 mesh[mesh_id_t::cells][max_level]);
//             leaves([&](auto &interval, auto) {
//                 auto k = interval;
//                 advected_f(0, max_level, k) = xt::eval(f(0, max_level, k - 1));
//                 advected_f(1, max_level, k) = xt::eval(f(1, max_level, k + 1));
//             });
//         }
//         else
//         {
//             // We do the advection on the overleaves
//             std::size_t j = max_level - (level + 1);
//             double coeff = 1. / (1 << j);

//             auto ol = samurai::intersection(mesh[mesh_id_t::cells][level],
//                                             mesh[mesh_id_t::cells][level]).on(level + 1);

//             ol([&](auto& interval, auto) {
//                 auto k = interval; // Logical index in x

//                 auto fp = xt::eval(f(0, level + 1, k));
//                 auto fm = xt::eval(f(1, level + 1, k));

//                 for(auto &c: pred_coeff[j][0].coeff)
//                 {
//                     coord_index_t stencil = c.first;
//                     double weight = c.second;

//                     fp += coeff * weight * f(0, level + 1, k + stencil);
//                 }

//                 for(auto &c: pred_coeff[j][1].coeff)
//                 {
//                     coord_index_t stencil = c.first;
//                     double weight = c.second;

//                     fp -= coeff * weight * f(0, level + 1, k + stencil);
//                 }

//                 for(auto &c: pred_coeff[j][2].coeff)
//                 {
//                     coord_index_t stencil = c.first;
//                     double weight = c.second;

//                     fm += coeff * weight * f(1, level + 1, k + stencil);
//                 }

//                 for(auto &c: pred_coeff[j][3].coeff)
//                 {
//                     coord_index_t stencil = c.first;
//                     double weight = c.second;

//                     fm -= coeff * weight * f(1, level + 1, k + stencil);
//                 }

//                 // Save it
//                 help_f(0, level + 1, k) = fp;
//                 help_f(1, level + 1, k) = fm;
//             });

//             // Now that projection has been done, we have to come back on the leaves below the overleaves
//             auto leaves = samurai::intersection(mesh[mesh_id_t::cells][level],
//                                                 mesh[mesh_id_t::cells][level]);

//             leaves([&](auto &interval, auto) {
//                 auto k = interval;
//                 // Projection
//                 advected_f(0, level, k) = xt::eval(0.5 * (help_f(0, level + 1, 2*k) + help_f(0, level + 1, 2*k + 1)));
//                 advected_f(1, level, k) = xt::eval(0.5 * (help_f(1, level + 1, 2*k) + help_f(1, level + 1, 2*k + 1)));
//             });
//         }
//     }

//     for (std::size_t level = 0; level <= max_level; ++level)    {

//         double dx = 1./(1 << level);

//         auto leaves = samurai::intersection(mesh[mesh_id_t::cells][level],
//                                             mesh[mesh_id_t::cells][level]);

//         leaves([&](auto &interval, auto) {
//             auto k = interval;
//             auto uu = xt::eval(          advected_f(0, level, k) + advected_f(1, level, k));
//             auto vv = xt::eval(lambda * (advected_f(0, level, k) - advected_f(1, level, k)));

//             vv = (1 - s_rel) * vv + s_rel * ad_vel * uu;
              
//             new_f(0, level, k) = .5 * (uu + 1. / lambda * vv);
//             new_f(1, level, k) = .5 * (uu - 1. / lambda * vv);
//         });
//     }
//     std::swap(f.array(), new_f.array());
// }



template<class Field, class Func, class Pred>
void one_time_step(Field &f, Func&& update_bc_for_level,
                            const Pred& pred_coeff, double s_rel, double lambda, double ad_vel)
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
    // samurai::update_overleaves_mr(f, std::forward<Func>(update_bc_for_level));

    auto new_f = samurai::make_field<double, nvel>("new_f", mesh);
    new_f.fill(0.);
    auto advected_f = samurai::make_field<double, nvel>("advected_f", mesh);
    advected_f.fill(0.);
    auto help_f = samurai::make_field<double, nvel>("help_f", mesh);
    help_f.fill(0.);

    for (std::size_t level = 0; level <= max_level; ++level)
    {
        if (level == max_level) {
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
            std::size_t j = max_level - level;
            double coeff = 1. / (1 << j);

            auto lv = samurai::intersection(mesh[mesh_id_t::cells][level],
                                            mesh[mesh_id_t::cells][level]);

            lv([&](auto& interval, auto) {
                auto k = interval; // Logical index in x

                // auto fp = xt::eval(f(0, level, k));
                // auto fm = xt::eval(f(1, level, k));
                auto fp = xt::eval((1.-coeff)*f(0, level, k) + coeff*f(0, level, k-1));
                auto fm = xt::eval((1.-coeff)*f(1, level, k) + coeff*f(1, level, k+1));
                // for(auto &c: pred_coeff[j][0].coeff)
                // {
                //     coord_index_t stencil = c.first;
                //     double weight = c.second;

                //     fp += coeff * weight * f(0, level, k + stencil);
                // }

                // for(auto &c: pred_coeff[j][1].coeff)
                // {
                //     coord_index_t stencil = c.first;
                //     double weight = c.second;

                //     fp -= coeff * weight * f(0, level, k + stencil);
                // }

                // for(auto &c: pred_coeff[j][2].coeff)
                // {
                //     coord_index_t stencil = c.first;
                //     double weight = c.second;

                //     fm += coeff * weight * f(1, level, k + stencil);
                // }

                // for(auto &c: pred_coeff[j][3].coeff)
                // {
                //     coord_index_t stencil = c.first;
                //     double weight = c.second;

                //     fm -= coeff * weight * f(1, level, k + stencil);
                // }

                // Save it
                advected_f(0, level, k) = fp;
                advected_f(1, level, k) = fm;
            });
        }
    }

    for (std::size_t level = 0; level <= max_level; ++level)    {

        double dx = 1./(1 << level);

        auto leaves = samurai::intersection(mesh[mesh_id_t::cells][level],
                                            mesh[mesh_id_t::cells][level]);

        leaves([&](auto &interval, auto) {
            auto k = interval;
            auto uu = xt::eval(          advected_f(0, level, k) + advected_f(1, level, k));
            auto vv = xt::eval(lambda * (advected_f(0, level, k) - advected_f(1, level, k)));

            vv = (1 - s_rel) * vv + s_rel * ad_vel * uu;
              
            new_f(0, level, k) = .5 * (uu + 1. / lambda * vv);
            new_f(1, level, k) = .5 * (uu - 1. / lambda * vv);
        });
    }
    std::swap(f.array(), new_f.array());
}

template<class Field>
void save_solution(const Field & field, const std::size_t it, const std::string ext = "")
{
    auto mesh = field.mesh();
    using mesh_id_t = typename decltype(mesh)::mesh_id_t;


    std::stringstream str;
    str << "high_order_mr_"<<ext<<"-"<< it;

    auto u = samurai::make_field<double, 1>("u", mesh);
    auto level_ = samurai::make_field<std::size_t, 1>("level", mesh);

    samurai::for_each_cell(mesh[mesh_id_t::cells], [&](auto &cell) {
        // auto center = cell.center();
        // auto x = center[0];

        u[cell] = field[cell][0] + field[cell][1];
        level_[cell] = cell.level;
    });
    samurai::save(str.str().data(), mesh, u, level_);
}

int main()
{
    constexpr size_t dim = 1;
    using Config = samurai::MRConfig<dim, 2>;
    using mesh_t = samurai::MRMesh<Config>;
    using mesh_id_t = typename mesh_t::mesh_id_t;
    using coord_index_t = typename mesh_t::interval_t::coord_index_t;

    std::size_t min_level = 5;
    std::size_t max_level = 9;
    double epsilon = 1.e-5;
    double regularity = 1.;

    double ad_vel = 0.75;
    double lambda = 1.;
    double s_rel = 1.25;

    samurai::Box<double, dim> box({-3}, {3});
    mesh_t mesh{box, min_level, max_level};

    auto f_field = samurai::make_field<double, 2>("f", mesh);
    init_f(f_field, lambda, ad_vel);

    auto update_bc_for_level = [](auto& field, std::size_t level)
    {
        update_bc_1D_constant_extension(field, level);
    };

    auto pred_coeff_separate = compute_prediction_separate_inout<coord_index_t>(min_level, max_level);
    auto MRadaptation = samurai::make_MRAdapt(f_field, update_bc_for_level);

    for (std::size_t ite = 0; ite < 100; ++ite) {

        MRadaptation(epsilon, regularity, false);
        save_solution(f_field, ite);

        std::cout<<mesh<<std::endl;
        
        one_time_step(f_field, update_bc_for_level, pred_coeff_separate, s_rel, lambda, ad_vel);

        // one_time_step(f_field, update_bc_for_level, s_rel);
        save_solution(f_field, ite, "post");

        // one_time_step(f_field, update_bc_for_level, pred_coeff_separate, s_rel, lambda, ad_vel);

    }


    return 0;
}