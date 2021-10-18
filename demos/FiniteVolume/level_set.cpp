// Copyright 2021 SAMURAI TEAM. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include <samurai/box.hpp>
#include <samurai/field.hpp>
#include <samurai/hdf5.hpp>
#include <samurai/amr/mesh.hpp>
#include <samurai/algorithm/update.hpp>
#include <samurai/algorithm/graduation.hpp>

#include "stencil_field.hpp"

#include "../LBM/boundary_conditions.hpp"


template <class Mesh>
auto init_level_set(Mesh & mesh)
{
    using mesh_id_t = typename Mesh::mesh_id_t;

    auto phi = samurai::make_field<double, 1>("phi", mesh);
    phi.fill(0);

    samurai::for_each_cell(mesh[mesh_id_t::cells], [&](auto &cell)
    {
        auto center = cell.center();


        double x = center[0];
        double y = center[1];
        double radius = .15;
        double x_center = 0.5, y_center = 0.75;

        phi[cell] = std::sqrt(std::pow(x - x_center, 2.) +
                              std::pow(y - y_center, 2.)) - radius;
    });

    return phi;
}

template <class Mesh>
auto init_velocity(Mesh &mesh)
{

    using mesh_id_t = typename Mesh::mesh_id_t;

    auto u = samurai::make_field<double, 2>("u", mesh);
    u.fill(0);


    samurai::for_each_cell(mesh[mesh_id_t::cells_and_ghosts], [&](auto &cell)
    {
        auto center = cell.center();

        double x = center[0];
        double y = center[1];

        u[cell][0] = -std::pow(std::sin(M_PI*x), 2.) * std::sin(2.*M_PI*y);
        u[cell][1] =  std::pow(std::sin(M_PI*y), 2.) * std::sin(2.*M_PI*x);
    });

    return u;
}

template<class Field, class Tag>
void AMR_criteria(const Field& f, Tag& tag)
{
    auto mesh = f.mesh();
    using mesh_id_t = typename Field::mesh_t::mesh_id_t;

    std::size_t min_level = mesh.min_level();
    std::size_t max_level = mesh.max_level();

    samurai::for_each_cell(mesh[mesh_id_t::cells], [&](auto cell)
    {

        double dx = 1./(1 << (max_level));

        if (std::abs(f[cell]) < 1.2 * 5 * std::sqrt(2.) * dx)
        {
            if (cell.level == max_level)
            {
                tag[cell] = static_cast<int>(samurai::CellFlag::keep);
            }
            else
            {
                tag[cell] = static_cast<int>(samurai::CellFlag::refine);
            }
        }
        else
        {
            if (cell.level == min_level)
            {
                tag[cell] = static_cast<int>(samurai::CellFlag::keep);
            }
            else
            {
                tag[cell] = static_cast<int>(samurai::CellFlag::coarsen);
            }
        }
    });
}

template <class Field, class Field_u>
void flux_correction(Field& phi_np1, const Field& phi_n, const Field_u& u, double dt)
{
    using mesh_t = typename Field::mesh_t;
    using mesh_id_t = typename mesh_t::mesh_id_t;
    using interval_t = typename mesh_t::interval_t;

    auto mesh = phi_np1.mesh();
    std::size_t min_level = mesh[mesh_id_t::cells].min_level();
    std::size_t max_level = mesh[mesh_id_t::cells].max_level();
    for (std::size_t level = min_level; level < max_level; ++level)
    {
        xt::xtensor_fixed<int, xt::xshape<2>> stencil;

        stencil = {{-1, 0}};

        auto subset_right = samurai::intersection(samurai::translate(mesh[mesh_id_t::cells][level+1], stencil),
                                               mesh[mesh_id_t::cells][level])
                           .on(level);

        subset_right([&](const auto& i, const auto& index)
        {
            auto j = index[0];
            double dx = 1./(1<<level);

            phi_np1(level, i, j) = phi_np1(level, i, j) + dt/dx * (samurai::upwind_variable_op<interval_t>(level, i, j).right_flux(u, phi_n, dt)
                                                              - .5*samurai::upwind_variable_op<interval_t>(level+1, 2*i+1, 2*j).right_flux(u, phi_n, dt)
                                                              - .5*samurai::upwind_variable_op<interval_t>(level+1, 2*i+1, 2*j+1).right_flux(u, phi_n, dt));
        });

        stencil = {{1, 0}};

        auto subset_left = samurai::intersection(samurai::translate(mesh[mesh_id_t::cells][level+1], stencil),
                                        mesh[mesh_id_t::cells][level])
                            .on(level);

        subset_left([&](const auto& i, const auto& index)
        {
            auto j = index[0];
            double dx = 1./(1<<level);

            phi_np1(level, i, j) = phi_np1(level, i, j) - dt/dx * (samurai::upwind_variable_op<interval_t>(level, i, j).left_flux(u, phi_n, dt)
                                                            - .5 * samurai::upwind_variable_op<interval_t>(level+1, 2*i, 2*j).left_flux(u, phi_n, dt)
                                                            - .5 * samurai::upwind_variable_op<interval_t>(level+1, 2*i, 2*j+1).left_flux(u, phi_n, dt));
        });

        stencil = {{0, -1}};

        auto subset_up = samurai::intersection(samurai::translate(mesh[mesh_id_t::cells][level+1], stencil),
                                    mesh[mesh_id_t::cells][level])
                            .on(level);

        subset_up([&](const auto& i, const auto& index)
        {
            auto j = index[0];
            double dx = 1./(1<<level);

            phi_np1(level, i, j) = phi_np1(level, i, j) + dt/dx * (samurai::upwind_variable_op<interval_t>(level, i, j).up_flux(u, phi_n, dt)
                                                            - .5 * samurai::upwind_variable_op<interval_t>(level+1, 2*i, 2*j+1).up_flux(u, phi_n, dt)
                                                            - .5 * samurai::upwind_variable_op<interval_t>(level+1, 2*i+1, 2*j+1).up_flux(u, phi_n, dt));
        });

        stencil = {{0, 1}};

        auto subset_down = samurai::intersection(samurai::translate(mesh[mesh_id_t::cells][level+1], stencil),
                                    mesh[mesh_id_t::cells][level])
                            .on(level);

        subset_down([&](const auto& i, const auto& index)
        {
            auto j = index[0];
            double dx = 1./(1<<level);

            phi_np1(level, i, j) = phi_np1(level, i, j) - dt/dx * (samurai::upwind_variable_op<interval_t>(level, i, j).down_flux(u, phi_n, dt)
                                                            - .5 * samurai::upwind_variable_op<interval_t>(level+1, 2*i, 2*j).down_flux(u, phi_n, dt)
                                                            - .5 * samurai::upwind_variable_op<interval_t>(level+1, 2*i+1, 2*j).down_flux(u, phi_n, dt));
        });
    }
}

template<class Phi, class U>
void save_solution(const Phi& phi, const U& u, std::size_t ite)
{
    // using Config = typename Field::Config;
    auto mesh = phi.mesh();
    using mesh_id_t = typename Phi::mesh_t::mesh_id_t;

    std::size_t min_level = mesh.min_level();
    std::size_t max_level = mesh.max_level();

    auto level_field = samurai::make_field<double, 1>("level", mesh);

    samurai::for_each_cell(mesh[mesh_id_t::cells], [&](auto &cell)
    {
        level_field[cell] = static_cast<double>(cell.level);
    });

    samurai::save(fmt::format("LevelSet_Advection_ite_{}", ite), mesh, phi, u, level_field);

}

int main(int argc, char *argv[])
{
    constexpr std::size_t dim = 2;
    std::size_t start_level = 8;
    std::size_t min_level = 4;
    std::size_t max_level = 8;

    samurai::Box<double, dim> box({0, 0}, {1, 1});
    using Config = samurai::amr::Config<dim>;
    samurai::amr::Mesh<Config> mesh(box, start_level, min_level, max_level);

    double Tf = 3.14; // Final time
    double dt = 5./8/(1<<max_level);

    auto phi = init_level_set(mesh);

    auto u   = init_velocity(mesh);

    auto phinp1 = samurai::make_field<double, 1>("phi", mesh);
    auto phihat = samurai::make_field<double, 1>("phi", mesh);
    auto tag = samurai::make_field<int, 1>("tag", mesh);

    auto update_bc = [](std::size_t level, auto& phi, auto& u)
    {
        update_bc_D2Q4_3_Euler_constant_extension(phi, level);
        update_bc_D2Q4_3_Euler_constant_extension(u, level);
    };

    auto update_bc_phi = [](std::size_t level, auto& phi)
    {
        update_bc_D2Q4_3_Euler_constant_extension(phi, level);
    };

    xt::xtensor_fixed<int, xt::xshape<4, 2>> stencil_grad{{ 1, 0 }, { -1,  0 },
                                                          { 0, 1 }, {  0, -1 }};

    std::size_t Ntot = Tf/dt;
    for (std::size_t nt = 0; nt <= Ntot; ++nt)

    {
        std::cout<< "Iteration " << nt << std::endl;

        std::size_t ite = 0;
        while(true)
        {
            std::cout << "Mesh adaptation iteration " << ite++ << std::endl;
            tag.resize();
            AMR_criteria(phi, tag);
            samurai::graduation(tag, stencil_grad);
            samurai::update_ghost(update_bc, phi, u);
            if( (samurai::update_field(tag, phi, u)))
            {
                break;
            }
        }

        save_solution(phi, u, nt);

        samurai::update_ghost(update_bc, phi, u);
        phinp1.resize();
        phinp1 = phi - dt * samurai::upwind_variable(u, phi, dt);
        flux_correction(phinp1, phi, u, dt);

        std::swap(phi.array(), phinp1.array());

        // Reinitialization of the level set
        std::size_t fict_iteration = 2; // Number of fictitious iterations
        double dt_fict = 0.01 * dt;     // Fictitious Time step

        auto phi_0 = phi;
        for (std::size_t k = 0; k < fict_iteration; ++k)
        {
            // Forward Euler
            // update_ghosts(phi, u, update_bc_for_level);
            // phinp1 = phi - dt_fict * H_wrap(phi, phi_0, max_level);

            // TVD-RK2
            samurai::update_ghost(update_bc_phi, phi);
            phihat.resize();
            phihat = phi - dt_fict * H_wrap(phi, phi_0, max_level);
            samurai::update_ghost(update_bc_phi, phihat);
            phinp1 = .5 * phi_0 + .5 * (phihat - dt_fict * H_wrap(phihat, phi_0, max_level));

            std::swap(phi.array(), phinp1.array());
        }
    }

    return 0;
}