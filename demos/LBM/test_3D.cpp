// Copyright 2021 SAMURAI TEAM. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include <math.h>
#include <vector>

#include <cxxopts.hpp>
#include <spdlog/spdlog.h>
#include <spdlog/stopwatch.h>

#include <samurai/mr/adapt.hpp>
#include <samurai/mr/coarsening.hpp>
#include <samurai/mr/criteria.hpp>
#include <samurai/mr/harten.hpp>
#include <samurai/mr/mesh.hpp>
#include <samurai/mr/refinement.hpp>
#include <samurai/hdf5.hpp>
#include <samurai/statistics.hpp>

#include "prediction_map_3d.hpp"

template<class coord_index_t>
auto compute_prediction(std::size_t min_level, std::size_t max_level)
{
    coord_index_t i = 0, j = 0, k = 0;
    std::vector<std::vector<prediction_map<coord_index_t>>> data(max_level-min_level+1);

    for(std::size_t dl = 0; dl < max_level - min_level + 1; ++dl)
    {
        int size = (1<<dl);
        data[dl].resize(6);

        for (int l1 = 0; l1 < size; ++l1)   {
            for (int l2 = 0; l2 < size; ++l2)   {

                data[dl][0] = prediction(dl,  i   *size-1, j*size+l1, k*size+l2) - prediction(dl, (i+1)*size-1, j*size+l1, k*size+l2); // ( 1, 0, 0)
                data[dl][1] = prediction(dl, (i+1)*size  , j*size+l1, k*size+l2) - prediction(dl,  i   *size  , j*size+l1, k*size+l2); // (-1, 0, 0)

                data[dl][2] = prediction(dl, i*size+l1,  j   *size-1, k*size+l2) - prediction(dl, i*size+l1, (i+1)*size-1, k*size+l2); // (0,- 1, 0)
                data[dl][3] = prediction(dl, i*size+l1, (j+1)*size  , k*size+l2) - prediction(dl, i*size+l1,  i   *size  , k*size+l2); // (0, -1, 0)

                data[dl][4] = prediction(dl, i*size+l1, j*size+l2,  k   *size-1) - prediction(dl, i*size+l1, j*size+l2, (k+1)*size-1); // (0, 0,  1)
                data[dl][5] = prediction(dl, i*size+l1, j*size+l2, (k+1)*size  ) - prediction(dl, i*size+l1, j*size+l2,  k   *size  ); // (0, 0, -1)
            }
        }
    }
    return data;
}

template<class Field>
void init_f(Field & field, const double lambda, const double Vx, const double Vy, const double Vz)
{
    auto mesh = field.mesh();
    using mesh_id_t = typename decltype(mesh)::mesh_id_t;

    const double x_ct = .5;
    const double y_ct = .5;
    const double z_ct = .5;
    const double radius = 0.15;

    samurai::for_each_cell(mesh[mesh_id_t::cells], [&](auto &cell) {
        auto center = cell.center();
        auto x = center[0];
        auto y = center[1];
        auto z = center[2];

        const double dist_ct = std::sqrt(std::pow(x-x_ct, 2.)+std::pow(y-y_ct, 2.)+std::pow(z-z_ct, 2.));

        const double u = (dist_ct < radius) ? 1. : 0.;
        // const double q_x = Vx*u;
        // const double q_y = Vy*u;
        // const double q_z = Vz*u;
        // // Burgers like
        const double q_x = .5*Vx*u*u;
        const double q_y = .5*Vy*u*u;
        const double q_z = .5*Vz*u*u;

        const double w_1 = 0.;
        const double w_2 = 0.;

        field[cell][0] = 1./6*u + (.5/lambda)*q_x                                 + 1./(6*lambda*lambda)*w_1 + 1./(6*lambda*lambda)*w_2;
        field[cell][1] = 1./6*u - (.5/lambda)*q_x                                 + 1./(6*lambda*lambda)*w_1 + 1./(6*lambda*lambda)*w_2;
        field[cell][2] = 1./6*u                 + (.5/lambda)*q_y                 - 1./(3*lambda*lambda)*w_1 + 1./(6*lambda*lambda)*w_2;
        field[cell][3] = 1./6*u                 - (.5/lambda)*q_y                 - 1./(3*lambda*lambda)*w_1 + 1./(6*lambda*lambda)*w_2;
        field[cell][4] = 1./6*u                                 + (.5/lambda)*q_z + 1./(6*lambda*lambda)*w_1 - 1./(3*lambda*lambda)*w_2;
        field[cell][5] = 1./6*u                                 - (.5/lambda)*q_z + 1./(6*lambda*lambda)*w_1 - 1./(3*lambda*lambda)*w_2;
    });
}

template<class Field, class Func>

void one_step_debug(Field & f, Func && update_bc_for_level)
{
    using mesh_id_t = typename Field::mesh_t::mesh_id_t;
    using coord_index_t = typename Field::interval_t::coord_index_t;

    auto mesh = f.mesh();
    Field f_ad{"advected", mesh};
    f_ad.array().fill(0.);

    auto min_level = mesh[mesh_id_t::cells].min_level();
    auto max_level = mesh[mesh_id_t::cells].max_level();
    
    // samurai::update_ghost_mr(f, std::forward<Func>(update_bc_for_level));

    for (std::size_t level = min_level; level <= max_level; ++level)
    {
        auto set = samurai::difference(mesh[mesh_id_t::reference][level], mesh[mesh_id_t::cells][level]);
        set([&](auto& interval, auto& index) {
            auto i = interval; // Logical index in x
            auto j = index[0]; // Logical index in y
            auto k = index[1]; 
            f(level, i, j, k) = 0.;
        });
    }

    for (std::size_t level = min_level; level <= max_level; ++level)
    {
        auto leaves = samurai::intersection(mesh[mesh_id_t::cells][level],
                                            mesh[mesh_id_t::cells][level]);


        leaves([&](auto& interval, auto& index) {
            auto i = interval; // Logical index in x
            auto j = index[0]; // Logical index in y
            auto k = index[1]; 

            for (std::size_t nf = 0; nf<6; ++nf)    
            {
                f_ad(nf, level, i, j, k) = (1.-0.001/(1<<(max_level-level)))*f(nf, level, i, j, k) + 0.001/(1<<(max_level-level))*f(nf, level, i, j, k -1);
            }

            // f_ad(0, level, i, j, k) = f(0, level, i, j  , k -1);
            // f_ad(1, level, i, j, k) = f(1, level, i, j  , k );
            // f_ad(2, level, i, j, k) = f(2, level, i  , j, k );
            // f_ad(3, level, i, j, k) = f(3, level, i  , j, k );
            // f_ad(4, level, i, j, k) = f(4, level, i  , j  , k);
            // f_ad(5, level, i, j, k) = f(5, level, i  , j  , k);
        });
    }

    std::swap(f_ad.array(), f.array());
}

template<class Field, class Func, class Pred>
void one_step(Field & f, Func && update_bc_for_level, const Pred & pred_coeff,
              const double lambda, const double s1, const double s2, const double Vx, const double Vy, const double Vz)
{
    using mesh_id_t = typename Field::mesh_t::mesh_id_t;
    using coord_index_t = typename Field::interval_t::coord_index_t;

    auto mesh = f.mesh();
    Field f_ad{"advected", mesh};
    f_ad.array().fill(0.);

    auto min_level = mesh[mesh_id_t::cells].min_level();
    auto max_level = mesh[mesh_id_t::cells].max_level();
    
    samurai::update_ghost_mr(f, std::forward<Func>(update_bc_for_level));

    for (std::size_t level = min_level; level <= max_level; ++level)
    {
        auto leaves = samurai::intersection(mesh[mesh_id_t::cells][level],
                                            mesh[mesh_id_t::cells][level]);

        if (level == max_level) { // Advection at the finest level

            leaves([&](auto& interval, auto& index) {
                auto i = interval; // Logical index in x
                auto j = index[0]; // Logical index in y
                auto k = index[1]; 

                f_ad(0, level, i, j, k) = f(0, level, i-1, j  , k  );
                f_ad(1, level, i, j, k) = f(1, level, i+1, j  , k  );
                f_ad(2, level, i, j, k) = f(2, level, i  , j-1, k  );
                f_ad(3, level, i, j, k) = f(3, level, i  , j+1, k  );
                f_ad(4, level, i, j, k) = f(4, level, i  , j  , k-1);
                f_ad(5, level, i, j, k) = f(5, level, i  , j  , k+1);
            });
        }
        else
        {
            std::size_t dl = max_level - level;
            double coeff = 1. / (1 << (3*dl)); // ATTENTION A LA DIMENSION 3 !!!!



            leaves([&](auto& interval, auto& index) {
                auto i = interval; // Logical index in x
                auto j = index[0]; // Logical index in y
                auto k = index[1]; 

                for (std::size_t nf = 0; nf < 6; ++nf)  {
                    f_ad(nf, level, i, j, k) = f(nf, level, i, j, k);

                    for(auto &c: pred_coeff[dl][nf].coeff)
                        {
                            coord_index_t stencil_x, stencil_y, stencil_z;
                            std::tie(stencil_x, stencil_y, stencil_z) = c.first;

                            f_ad(nf, level, i, j, k) += coeff * c.second * f(nf, level, i + stencil_x, j + stencil_y, k + stencil_z);
                        }
                }
            });
        }
        
    }

    for (std::size_t level = min_level; level <= max_level; ++level)
    {
        auto leaves = samurai::intersection(mesh[mesh_id_t::cells][level],
                                            mesh[mesh_id_t::cells][level]);

        leaves([&](auto& interval, auto& index) {
            auto i = interval; // Logical index in x
            auto j = index[0]; // Logical index in y
            auto k = index[1]; 
            
            auto u    = xt::eval(               f_ad(0, level, i, j, k)+f_ad(1, level, i, j, k)+f_ad(2, level, i, j, k)+f_ad(3, level, i, j, k)+f_ad(4, level, i, j, k)+f_ad(5, level, i, j, k));
            auto q_x  = xt::eval(lambda*       (f_ad(0, level, i, j, k)-f_ad(1, level, i, j, k)));
            auto q_y  = xt::eval(lambda*       (                                                f_ad(2, level, i, j, k)-f_ad(3, level, i, j, k)));
            auto q_z  = xt::eval(lambda*       (                                                                                                f_ad(4, level, i, j, k)-f_ad(5, level, i, j, k)));
            auto w_1  = xt::eval(lambda*lambda*(f_ad(0, level, i, j, k)+f_ad(1, level, i, j, k)-f_ad(2, level, i, j, k)-f_ad(3, level, i, j, k)));
            auto w_2  = xt::eval(lambda*lambda*(f_ad(0, level, i, j, k)+f_ad(1, level, i, j, k)                                                -f_ad(4, level, i, j, k)-f_ad(5, level, i, j, k)));

            // q_x = (1.-s1)*q_x + s1*Vx*u;
            // q_y = (1.-s1)*q_y + s1*Vy*u;
            // q_z = (1.-s1)*q_z + s1*Vz*u;

            // // Burgers-like
            q_x = (1.-s1)*q_x + s1*.5*Vx*xt::pow(u, 2.);
            q_y = (1.-s1)*q_y + s1*.5*Vy*xt::pow(u, 2.);
            q_z = (1.-s1)*q_z + s1*.5*Vz*xt::pow(u, 2.);

            w_1 = (1-s2)*w_1;
            w_2 = (1-s2)*w_2;

            f(0, level, i, j, k) = 1./6*u + (.5/lambda)*q_x                                 + 1./(6*lambda*lambda)*w_1 + 1./(6*lambda*lambda)*w_2;
            f(1, level, i, j, k) = 1./6*u - (.5/lambda)*q_x                                 + 1./(6*lambda*lambda)*w_1 + 1./(6*lambda*lambda)*w_2;
            f(2, level, i, j, k) = 1./6*u                 + (.5/lambda)*q_y                 - 1./(3*lambda*lambda)*w_1 + 1./(6*lambda*lambda)*w_2;
            f(3, level, i, j, k) = 1./6*u                 - (.5/lambda)*q_y                 - 1./(3*lambda*lambda)*w_1 + 1./(6*lambda*lambda)*w_2;
            f(4, level, i, j, k) = 1./6*u                                 + (.5/lambda)*q_z + 1./(6*lambda*lambda)*w_1 - 1./(3*lambda*lambda)*w_2;
            f(5, level, i, j, k) = 1./6*u                                 - (.5/lambda)*q_z + 1./(6*lambda*lambda)*w_1 - 1./(3*lambda*lambda)*w_2;

        });
    }
}

template<class Field>
void save_solution(const Field &f, std::size_t ite, std::string ext = "")
{
    using value_t = typename Field::value_type;

    auto mesh = f.mesh();
    using mesh_id_t = typename decltype(mesh)::mesh_id_t;

    std::size_t min_level = mesh.min_level();
    std::size_t max_level = mesh.max_level();

    std::stringstream str;
    str << "test_3D"<<ext<<"-"<< ite;

    auto u = samurai::make_field<value_t, 1>("u", mesh);
    auto level_ = samurai::make_field<std::size_t, 1>("level", mesh);


    samurai::for_each_cell(mesh[mesh_id_t::cells], [&](auto &cell) {
        u[cell] = f[cell][0] + f[cell][1] + f[cell][2] + f[cell][3] + f[cell][4] + f[cell][5];
        level_[cell] = cell.level;
    });

    samurai::save(str.str().data(), mesh, u, level_);
}


int main()
{
    constexpr size_t dim = 3;
    using Config = samurai::MRConfig<dim, 2>;

    std::size_t min_level = 2;
    std::size_t max_level = 6;

    const double eps = 1.e-3;
    const double reg = 2.;

    // const double Vx = 0.4 / 3.;
    // const double Vy = 0.4 / 3.;
    // const double Vz = 0.4 / 3.;

    const double Vx = 0.;
    const double Vy = 0.;
    const double Vz = 0.4;

    const double lambda = 1.;
    const double s1 = 1.4;
    const double s2 = 1.;

    samurai::Box<double, dim> box({0, 0, 0}, {1, 1, 1});
    samurai::MRMesh<Config> mesh(box, min_level, max_level);

    using mesh_id_t = typename samurai::MRMesh<Config>::mesh_id_t;
    using coord_index_t = typename samurai::MRMesh<Config>::coord_index_t;
    // auto pred_coeff = compute_prediction<coord_index_t>(min_level, max_level);

    auto f_field = samurai::make_field<double, 6>("f", mesh);

    init_f(f_field, lambda, Vx, Vy, Vz);

    // samurai::save(std::string("test_3D"), mesh, f_field);

    auto update_bc_for_level = [](auto& field, std::size_t level)
    {
        auto mesh = field.mesh();
        using mesh_id_t = typename decltype(mesh)::mesh_id_t;

        auto set = samurai::difference(mesh[mesh_id_t::cells_and_ghosts][level], mesh.domain()).on(level);
        set([&](const auto& i, const auto& index)
        {
            auto j = index[0];
            auto k = index[1];
            
            field(level, i, j, k) = 0.;
        });
    };

    auto MRadaptation = samurai::make_MRAdapt(f_field, update_bc_for_level);

    // if (max_level > min_level)  {
    //     MRadaptation(1.e-3, 0.);
    // }

    // save_solution(f_field, 0);
    // one_step(f_field, update_bc_for_level, pred_coeff, lambda, s1, s2, Vx, Vy, Vz);
    // save_solution(f_field, 1);
    
    // samurai::save(std::string("test_3D_after"), mesh, f_field);

    // using mesh_id_t = typename samurai::MRMesh<Config>::mesh_id_t;


    for (std::size_t it = 0; it < 17; it++) {

        std::cout<<"Iteration = "<<it<<std::endl;

        std::cout<<"Doing mesh adaptation"<<std::endl;
        if (max_level > min_level)  {
            MRadaptation(eps, reg, it == 0);
        }
        std::cout<<"Mesh adaptation done, saving solution"<<std::endl;

        save_solution(f_field, it);
        
        std::cout<<"Saving done, time stepping"<<std::endl;

        // one_step(f_field, update_bc_for_level, pred_coeff, lambda, s1, s2, Vx, Vy, Vz);

        one_step_debug(f_field, update_bc_for_level);
        // save_solution(f_field, it, "post");

    }

    return 0;
}

