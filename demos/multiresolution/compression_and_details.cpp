// Copyright 2021 SAMURAI TEAM. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include <math.h>
#include <vector>

#include <cxxopts.hpp>
#include <spdlog/spdlog.h>
#include <spdlog/stopwatch.h>

#include <samurai/mr/coarsening.hpp>
#include <samurai/mr/refinement.hpp>
#include <samurai/mr/criteria.hpp>
#include <samurai/mr/harten.hpp>
#include <samurai/mr/adapt.hpp>
#include "../FiniteVolume-MR/boundary_conditions.hpp"


template<class Config>
auto init_f(samurai::MRMesh<Config> & mesh)
{
    constexpr std::size_t nvel = 1;
    using mesh_id_t = typename samurai::MRMesh<Config>::mesh_id_t;

    auto f = samurai::make_field<double, nvel>("f", mesh);
    f.fill(0);

    samurai::for_each_cell(mesh[mesh_id_t::cells], [&](auto &cell) {
        auto center = cell.center();
        auto x = center[0];
        auto y = center[1];

        f[cell] = ((std::sqrt(       std::pow(x + .5, 2.) + std::pow(y + .5, 2.)) < .25) ? 1. : 0.)
                    + std::exp(-50.*(std::pow(x - .5, 2.) + std::pow(y - .5, 2.)));
    });

    return f;
}

template<class Field, class Func>
void save_solution(Field &f, Func && update_bc_for_level, double eps)
{
    auto mesh = f.mesh();
    using value_t = typename Field::value_type;
    using mesh_id_t = typename decltype(mesh)::mesh_id_t;

    std::size_t min_level = mesh.min_level();
    std::size_t max_level = mesh.max_level();

    std::stringstream str;
    str<<"CompressionAndDetails_lmin_"<<min_level<<"_lmax_"<<max_level<<"_eps_"<<eps;

    auto level     = samurai::make_field<std::size_t, 1>("level"             , mesh);
    auto details   = samurai::make_field<double     , 1>("details"           , mesh);
    auto details_n = samurai::make_field<double     , 1>("details_normalized", mesh);

    details.fill(2.);

    // auto details_tmp   = samurai::make_field<double     , 1>("details"           , mesh);

    mr_projection(f);
    for (std::size_t level = min_level - 2; level <= max_level; ++level)
    {
        update_bc_for_level(f, level);
    }
    mr_prediction(f, update_bc_for_level);

    for (std::size_t level = min_level - 1; level < max_level; ++level)
    {
        auto subset = intersection(mesh[mesh_id_t::all_cells][level],
                                   mesh[mesh_id_t::cells][level + 1]).on(level);
        subset.apply_op(compute_detail(details, f));
    }

    // for (std::size_t i = 0; i < max_level - min_level; ++i) {

    //     mr_projection(f);
    //     for (std::size_t level = min_level - 1; level <= max_level; ++level)
    //     {
    //         update_bc_for_level(f, level);
    //     }
    //     mr_prediction(f, update_bc_for_level);

    //     for (std::size_t level = min_level - 1; level < max_level - i; ++level)
    //     {
    //         auto subset = intersection(mesh[mesh_id_t::all_cells][level],
    //                                    mesh[mesh_id_t::cells][level + 1]).on(level);
    //         subset.apply_op(compute_detail(details_tmp, f));
    //     }

    //     auto leaves = samurai::intersection(mesh[mesh_id_t::cells][max_level - i],
    //                                         mesh[mesh_id_t::cells][max_level - i]);

    //     leaves([&](auto& interval, auto& index) {
    //         auto k = interval;
    //         auto h = index[0];

    //         details(max_level - i, k, h) = details_tmp(max_level - i, k, h);
    //     });
    // }

    samurai::for_each_cell(mesh[mesh_id_t::cells], [&](auto &cell) {
        double eps_l = std::pow(2., -2 * static_cast<double>(max_level - cell.level)) * eps;
        level[cell] = cell.level;
        details_n[cell] = details[cell] / eps_l;
    });

    samurai::save(str.str().data(), mesh, f, details, details_n, level);
}

int main()
{
    constexpr size_t dim = 2;
    using Config = samurai::MRConfig<dim, 2>;

    const std::size_t min_level = 2;
    const std::size_t max_level = 8;
    const double eps = 1.e-4;
    const double regularity = 8000.; // Just not to do Harten ....

    samurai::Box<double, dim> box({-1, -1}, {1, 1});
    samurai::MRMesh<Config> mesh(box, min_level, max_level);
    auto f = init_f(mesh);

    auto update_bc_for_level = [](auto& field, std::size_t level)
    {
        update_bc_D2Q4_3_Euler_constant_extension(field, level);
    };
    auto MRadaptation = samurai::make_MRAdapt(f, update_bc_for_level);

    MRadaptation(eps, regularity);

    save_solution(f, update_bc_for_level, eps);

    return 0;
}

