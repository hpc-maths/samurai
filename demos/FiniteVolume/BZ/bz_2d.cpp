// Copyright 2021 SAMURAI TEAM. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include <chrono>
#include <cmath>
#include <math.h>
#include <samurai/algorithm.hpp>
#include <samurai/field.hpp>
#include <samurai/hdf5.hpp>
#include <samurai/mr/adapt.hpp>
#include <samurai/mr/mesh.hpp>
#include <samurai/stencil_field.hpp>

#include "../../LBM/boundary_conditions.hpp"
#include "RockAndRadau/integration_stiff.h"

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

/*
    ======= 2D BZ Test ======

    Equations :
     --
    |  db/dt - D_b * d^2 b/dxx = 1/eps * (b(1-b) + f(q-b)c/(q+b))
    |  dc/dt - D_c * d^2 c/dxx = b-c
     --
*/

template <class Mesh>
auto init_field(Mesh& mesh, const double f = 1.6, const double q = 2.e-3)
{
    using mesh_id_t = typename Mesh::mesh_id_t;
    /*
    field[0] : 'b' in the model
    field[1] : 'c' in the model
    */
    auto field = samurai::make_field<double, 2>("solution", mesh);

    samurai::for_each_cell(mesh[mesh_id_t::reference],
                           [&](auto cell)
                           {
                               double x     = cell.corner(0);
                               double y     = cell.corner(1);
                               double coeff = q * (f + 1) / (f - 1);

                               if (0.3 * x >= y && y >= 0 && x >= 0)
                               {
                                   field[cell][0] = 0.8;
                               }
                               else
                               {
                                   field[cell][0] = coeff;
                               }

                               if (x > 0)
                               {
                                   if (y >= 0)
                                   {
                                       field[cell][1] = coeff + (std::atan(y / x)) / (8 * M_PI * f);
                                   }
                                   else
                                   {
                                       field[cell][1] = coeff + (std::atan(y / x) + 2 * M_PI) / (8 * M_PI * f);
                                   }
                               }
                               else
                               {
                                   if (x < 0)
                                   {
                                       field[cell][1] = coeff + (std::atan(y / x) + M_PI) / (8 * M_PI * f);
                                   }
                                   else
                                   {
                                       if (y >= 0)
                                       {
                                           field[cell][1] = coeff + 1. / 16 / f;
                                       }
                                       else
                                       {
                                           field[cell][1] = coeff + 3. / 16 / f;
                                       }
                                   }
                               }
                           });

    return field;
}

void f_radau(const int* dimension_of_the_system, const double* time, const double* input, double* der_output, const double* rpar, const int* ipar)
{
    const double f       = 1.6;
    const double q       = 2.e-3;
    const double epsilon = 1.e-2;

    der_output[0] = 1.0 / epsilon * (input[0] - input[0] * input[0] + f * (q - input[0]) * input[1] / (q + input[0]));
    der_output[1] = input[0] - input[1];
}

void s_radau(int*, double*, double*, double*, double*, int*, int*, double*, int*, int*)
{
}

/*
template <class Field>
void diffusion_rock4(Field & field, const double t,
                             const double dt,
                             const double f       = 1.6,
                             const double q       = 2.e-3,
                             const double epsilon = 1.e-2)
{
    const double t0 = t;
    const double t1 = t + dt;
    const double tol = 1e-6;

    using mesh_id_t = typename Field::mesh_t::mesh_id_t;
    auto mesh = field.mesh();


    int info[8]; // TODO: vérifier la taille
    const int neq = field.size() ?


    rock4_integration(t1, t2, neq, field.data()???,
                diffusion_fun_rock4, tol, info)
}


void diffusion_fun_rock4(Field & field) // pas sûr pour la nature de l'argument
{
    bc(field); // mise à jour des ghosts
    TODO;
}
*/

template <class Field>
void reaction(Field& field, const double t0, const double t1, const double f = 1.6, const double q = 2.e-3, const double epsilon = 1.e-2)
{
    const double tol = 1e-6;

    using mesh_id_t = typename Field::mesh_t::mesh_id_t;
    auto mesh       = field.mesh();

    samurai::for_each_cell(mesh[mesh_id_t::cells],
                           [&](auto cell)
                           {
                               // TODO: Be sure that sure that we can call with field
                               std::array<double, 2> values{
                                   {field[cell][0], field[cell][1]}
                               };
                               int info[8];

                               radau5_integration(t0, t1, 2, values.data(), f_radau, s_radau, tol, 2, info);

                               field[cell][0] = values[0];
                               field[cell][1] = values[1];
                           });
}

template <class TInterval>
class diffusion_op : public samurai::field_operator_base<TInterval>,
                     public samurai::finite_volume<diffusion_op<TInterval>>
{
  public:

    INIT_OPERATOR(diffusion_op)

    template <class T1, class T2>
    inline auto flux(T1&& ul, T2&& ur) const
    {
        return xt::eval((std::forward<T1>(ur) - std::forward<T2>(ul)) / dx());
    }

    template <class T1>
    inline auto left_flux(const T1& u) const
    {
        return flux(u(level, i - 1, j), u(level, i, j));
    }

    template <class T1>
    inline auto right_flux(const T1& u) const
    {
        return flux(u(level, i, j), u(level, i + 1, j));
    }

    template <class T1>
    inline auto down_flux(const T1& u) const
    {
        return flux(u(level, i, j - 1), u(level, i, j));
    }

    template <class T1>
    inline auto up_flux(const T1& u) const
    {
        return flux(u(level, i, j), u(level, i, j + 1));
    }
};

template <class... CT>
inline auto diffusion(CT&&... e)
{
    return samurai::make_field_operator_function<diffusion_op>(std::forward<CT>(e)...);
}

template <class Field, class Func>
void RK4(Field& field, const double dt, std::size_t nbstep, Func&& bc, const double D_b = 2.5e-3, const double D_c = 1.5e-3)
{
    using mesh_id_t = typename Field::mesh_t::mesh_id_t;

    auto mesh             = field.mesh();
    std::size_t min_level = mesh.min_level();
    std::size_t max_level = mesh.max_level();

    double dt_rk = dt / nbstep; // Sub time steps
    xt::xtensor_fixed<double, xt::xshape<2>> nu{D_b, D_c};

    auto apply_nu = [&](auto& field)
    {
        samurai::for_each_interval(mesh[mesh_id_t::cells],
                                   [&](std::size_t level, const auto& i, const auto& index)
                                   {
                                       auto j = index[0];
                                       field(level, i, j) *= nu;
                                   });
    };

    Field k1("k1", mesh);
    Field k2("k2", mesh);
    Field k3("k3", mesh);
    Field k4("k4", mesh);

    for (std::size_t nite = 0; nite < nbstep; ++nite)
    {
        samurai::update_ghost_mr(field, bc);
        k1 = diffusion(field);
        apply_nu(k1);
        samurai::update_ghost_mr(k1, bc);

        k2 = diffusion(field + dt_rk / 2. * k1);
        apply_nu(k2);
        samurai::update_ghost_mr(k2, bc);

        k3 = diffusion(field + dt_rk / 2. * k2);
        apply_nu(k3);
        samurai::update_ghost_mr(k3, bc);

        k4 = diffusion(field + dt_rk * k3);
        apply_nu(k4);

        field = field + dt_rk / 6. * (k1 + 2. * k2 + 2. * k3 + k4);
    }
}

int main()
{
    constexpr std::size_t dim = 2; // Spatial dimension
    std::size_t max_level     = 8; // Maximum level of resolution
    std::size_t min_level     = 1; // Minimum level of resolution
    samurai::Box<double, dim> box{
        {-1, -1},
        {1,  1 }
    }; // Domain [-1, 1]^2

    const double D_b     = 2.5e-3; // Diffusion coefficient 'b'
    const double D_c     = 1.5e-3; // Diffusion coefficient 'c'
    const double epsilon = 1.e-2;  // Stiffness parameter

    const double dx = 1. / (1 << max_level); // Space step

    const double Tf = 1.;

    const double regularity = 1.;    // Regularity guess for multiresolution
    const double epsilon_MR = 2.e-4; // Threshold used by multiresolution

    using Config = samurai::MRConfig<dim, 2>;
    samurai::MRMesh<Config> mesh(box, min_level, max_level); // Constructing mesh from the box

    using mesh_id_t = typename decltype(mesh)::mesh_id_t;

    auto field = init_field(mesh);                      // Initializing solution field
    samurai::save(std::string("bz_init"), mesh, field); // Saving

    auto update_bc = [](auto& field, std::size_t level)
    {
        update_bc_D2Q4_3_Euler_constant_extension(field, level);
    };

    double t           = 0.;
    std::size_t nb_ite = 0;
    std::size_t nsave  = 0;

    double dt           = 1.e-3;                                      // Time step (splitting time)
    double dt_diffusion = 0.25 * dx * dx / (2. * std::max(D_b, D_c)); // Diffusion time step

    auto MRadaptation = samurai::make_MRAdapt(field, update_bc);

    while (t < Tf)
    {
        fmt::print(fmt::format("Iteration = {:4d}, t: {}\n", nb_ite, t));

        if (max_level > min_level)
        {
            MRadaptation(epsilon_MR, regularity);
        }

        tic();
        reaction(field, t, t + .5 * dt);
        auto duration = toc();
        fmt::print(fmt::format("first reaction: {}\n", duration));

        tic();
        RK4(field, dt, std::ceil(dt / dt_diffusion), update_bc, D_b, D_c);
        duration = toc();
        fmt::print(fmt::format("diffusion: {}\n", duration));
        /*
        rock4_integration(double tini, double tend, int neq, double *u,
                    func_rock fcn, double tol, int *info)
        */
        tic();
        reaction(field, t + .5 * dt, t + dt);
        duration = toc();
        fmt::print(fmt::format("second reaction: {}\n", duration));

        if (nsave == 20)
        {
            samurai::save(fmt::format("bz_ite-{}", nb_ite), mesh, field); // Saving
            nsave = 0;
        }
        nsave++;
        t += dt;
        nb_ite++;
    }

    return 0;
}
