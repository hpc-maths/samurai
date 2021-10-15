// Copyright 2021 SAMURAI TEAM. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.
#include <iostream>
#include <fmt/format.h>

#include <samurai/box.hpp>
#include <samurai/field.hpp>
#include <samurai/hdf5.hpp>
#include <samurai/amr/mesh.hpp>
#include <samurai/algorithm/update.hpp>
#include <samurai/algorithm/graduation.hpp>

#include "stencil_field.hpp"

#include "../LBM/boundary_conditions.hpp"

template <class Mesh>
auto init_solution(Mesh & mesh)
{
    using mesh_id_t = typename Mesh::mesh_id_t;

    auto phi = samurai::make_field<double, 1>("phi", mesh);
    phi.fill(0);

    samurai::for_each_cell(mesh[mesh_id_t::cells], [&](auto &cell)
    {
        auto center = cell.center();
        double x = center[0];

        double u = 0.;

        // Initial hat solution
        if (x < -1. or x > 1.)  {
            u = 0.;
        }
        else
        {
            u = (x < 0.) ? (1 + x) : (1 - x);
        }

        // phi[cell] = u;
        phi[cell] = std::exp(-20.*x*x);
    });

    return phi;
}

template<class Field, class Tag>
void AMR_criteria(Field& f, Tag& tag)
{
    auto mesh = f.mesh();
    using mesh_id_t = typename Field::mesh_t::mesh_id_t;
    std::size_t min_level = mesh.min_level();
    std::size_t max_level = mesh.max_level();

    tag.fill(static_cast<int>(samurai::CellFlag::keep)); // Important

    for (std::size_t level = min_level; level <= max_level; ++level)    {
        double dx = 1./(1 << level);

        auto leaves = samurai::intersection(mesh[mesh_id_t::cells][level],
                                            mesh[mesh_id_t::cells][level]);

        leaves([&](auto& interval, auto& ) {
            auto k = interval;

            auto der_approx = xt::eval(xt::abs((f(level, k + 1) - f(level, k - 1)) / (2.*dx)));
            auto der_der_approx = xt::eval(xt::abs((f(level, k + 1) - 2.*f(level, k) + f(level, k - 1)) / (dx*dx)));

            auto der_plus  = xt::eval(xt::abs((f(level, k + 1) - f(level, k)) / (dx)));
            auto der_minus = xt::eval(xt::abs((f(level, k) - f(level, k - 1)) / (dx)));

            // auto mask = xt::abs(f(level, k)) > 0.001;
            auto mask = der_approx > 0.01;
            // auto mask = der_der_approx > 0.01;

            // auto mask = (xt::abs(der_plus) - xt::abs(der_minus)) > 0.001;

            if (level == max_level) {
                xt::masked_view(tag(level, k),   mask) = static_cast<int>(samurai::CellFlag::keep);
                xt::masked_view(tag(level, k),  !mask) = static_cast<int>(samurai::CellFlag::coarsen);

            }
            else
            {
                if (level == min_level) {
                    tag(level, k) = static_cast<int>(samurai::CellFlag::keep);
                }
                else
                {
                    xt::masked_view(tag(level, k),   mask) = static_cast<int>(samurai::CellFlag::refine);
                    xt::masked_view(tag(level, k),  !mask) = static_cast<int>(samurai::CellFlag::coarsen);
                }
            }
        });
    }
}

template<class Field>
void save_solution(Field &f, std::size_t ite)
{
    // using Config = typename Field::Config;
    auto mesh = f.mesh();
    using mesh_id_t = typename Field::mesh_t::mesh_id_t;

    std::size_t min_level = mesh.min_level();
    std::size_t max_level = mesh.max_level();

    auto level_ = samurai::make_field<double, 1>("level", mesh);

    samurai::for_each_cell(mesh[mesh_id_t::cells], [&](auto &cell)
    {
        level_[cell] = static_cast<double>(cell.level);
    });

    samurai::save(fmt::format("Burgers_AMR_lmin-{}_lmax-{}_ite-{}", min_level, max_level, ite), mesh, f, level_);

}

template <class Field>
void flux_correction(Field& phi_np1, const Field& phi_n, double dt)
{
    using mesh_t = typename Field::mesh_t;
    using mesh_id_t = typename mesh_t::mesh_id_t;
    using interval_t = typename mesh_t::interval_t;

    auto mesh = phi_np1.mesh();
    std::size_t min_level = mesh[mesh_id_t::cells].min_level();
    std::size_t max_level = mesh[mesh_id_t::cells].max_level();

    double dx = 1./(1 << max_level);

    for (std::size_t level = min_level; level < max_level; ++level)
    {
        double dx_loc = 1./(1<<level);
        xt::xtensor_fixed<int, xt::xshape<1>> stencil;

        stencil = {{-1}};
        auto subset_right = samurai::intersection(samurai::translate(mesh[mesh_id_t::cells][level+1], stencil),
                                               mesh[mesh_id_t::cells][level])
                           .on(level);

        subset_right([&](const auto& i, const auto& )
        {
            phi_np1(level, i) = phi_np1(level, i) + dt/dx_loc * (samurai::upwind_Burgers_op<interval_t>(level, i).right_flux(phi_n, dx/dt)
                                                                -samurai::upwind_Burgers_op<interval_t>(level+1, 2*i+1).right_flux(phi_n, dx/dt));
        });

        stencil = {{1}};
        auto subset_left = samurai::intersection(samurai::translate(mesh[mesh_id_t::cells][level+1], stencil),
                                        mesh[mesh_id_t::cells][level])
                            .on(level);

        subset_left([&](const auto& i, const auto& )
        {
            phi_np1(level, i) = phi_np1(level, i) - dt/dx_loc * (samurai::upwind_Burgers_op<interval_t>(level, i).left_flux(phi_n, dx/dt)
                                                                -samurai::upwind_Burgers_op<interval_t>(level+1, 2*i).left_flux(phi_n, dx/dt));
        });
    }
}

int main(int argc, char *argv[])
{
    constexpr std::size_t dim = 1;
    std::size_t start_level = 6;
    std::size_t min_level = 1;
    std::size_t max_level = 6;

    samurai::Box<double, dim> box({-3}, {3});
    using Config = samurai::amr::Config<dim>;
    samurai::amr::Mesh<Config> mesh(box, start_level, min_level, max_level);
    using mesh_id_t = typename Config::mesh_id_t;

    auto phi = init_solution(mesh);

    auto update_bc = [](std::size_t level, auto& field)
    {
        update_bc_D2Q4_3_Euler_constant_extension(field, level);
    };

    xt::xtensor_fixed<int, xt::xshape<2, 1>> stencil_grad{{ 1 }, { -1 }};

    double Tf = 1.5; // We have blowup at t = 1
    double dx = 1./(1 << max_level);
    double dt = 0.99 * dx;
    double t = 0.;
    std::size_t it = 0;

    while (t < Tf)
    {
        std::cout << "Iteration = " << it << " Time = " << t <<std::endl;

        std::size_t ite_adapt = 0;
        while(1)
        {
            std::cout << "\tmesh adaptation: " << ite_adapt++ << std::endl;
            samurai::update_ghost(update_bc, phi);
            auto tag = samurai::make_field<int, 1>("tag", mesh);
            AMR_criteria(phi, tag);
            samurai::graduation(tag, stencil_grad);
            if (samurai::update_field(tag, phi))
            {
                break;
            }
        }

        // Numerical scheme
         samurai::update_ghost(update_bc, phi);
        auto phinp1 = samurai::make_field<double, 1>("phi", mesh);
        phinp1 = phi - dt * samurai::upwind_Burgers(phi, dx/dt);
        flux_correction(phinp1, phi, dt);
        std::swap(phi.array(), phinp1.array());

        save_solution(phi, it);

        t  += dt;
        it += 1;
    }

    return 0;

}
