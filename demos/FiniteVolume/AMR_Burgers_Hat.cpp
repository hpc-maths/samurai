// Copyright 2021 SAMURAI TEAM. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.
#include "CLI/CLI.hpp"
#include <fmt/format.h>
#include <iostream>

#include <samurai/algorithm/graduation.hpp>
#include <samurai/algorithm/update.hpp>
#include <samurai/amr/mesh.hpp>
#include <samurai/bc.hpp>
#include <samurai/box.hpp>
#include <samurai/field.hpp>
#include <samurai/hdf5.hpp>

#include "stencil_field.hpp"

#include "../LBM/boundary_conditions.hpp"

#include <filesystem>
namespace fs = std::filesystem;

template <class Mesh>
auto init_solution(Mesh& mesh)
{
    using mesh_id_t = typename Mesh::mesh_id_t;

    auto phi = samurai::make_field<double, 1>("phi", mesh);
    phi.fill(0);

    samurai::for_each_cell(mesh[mesh_id_t::cells],
                           [&](auto& cell)
                           {
                               const double x = cell.center(0);
                               // double u = 0.;

                               // // Initial hat solution
                               // if (x < -1. || x > 1.)
                               // {
                               //     u = 0.;
                               // }
                               // else
                               // {
                               //     u = (x < 0.) ? (1 + x) : (1 - x);
                               // }

                               // phi[cell] = u;
                               phi[cell] = std::exp(-20. * x * x);
                           });

    return phi;
}

template <class Field, class Tag>
void AMR_criteria(const Field& f, Tag& tag)
{
    auto mesh             = f.mesh();
    using mesh_id_t       = typename Field::mesh_t::mesh_id_t;
    std::size_t min_level = mesh.min_level();
    std::size_t max_level = mesh.max_level();

    tag.fill(static_cast<int>(samurai::CellFlag::keep));

    for (std::size_t level = min_level; level <= max_level; ++level)
    {
        const double dx = samurai::cell_length(level);

        samurai::for_each_interval(
            mesh[mesh_id_t::cells][level],
            [&](std::size_t, auto& i, auto)
            {
                auto der_approx     = xt::eval(xt::abs((f(level, i + 1) - f(level, i - 1)) / (2. * dx)));
                auto der_der_approx = xt::eval(xt::abs((f(level, i + 1) - 2. * f(level, i) + f(level, i - 1)) / (dx * dx)));

                auto der_plus  = xt::eval(xt::abs((f(level, i + 1) - f(level, i)) / (dx)));
                auto der_minus = xt::eval(xt::abs((f(level, i) - f(level, i - 1)) / (dx)));

                // auto mask = xt::abs(f(level, i)) > 0.001;
                auto mask = der_approx > 0.01;
                // auto mask = der_der_approx > 0.01;
                // auto mask = (xt::abs(der_plus) - xt::abs(der_minus)) > 0.001;

                if (level == max_level)
                {
                    xt::masked_view(tag(level, i), mask)  = static_cast<int>(samurai::CellFlag::keep);
                    xt::masked_view(tag(level, i), !mask) = static_cast<int>(samurai::CellFlag::coarsen);
                }
                else
                {
                    if (level == min_level)
                    {
                        tag(level, i) = static_cast<int>(samurai::CellFlag::keep);
                    }
                    else
                    {
                        xt::masked_view(tag(level, i), mask)  = static_cast<int>(samurai::CellFlag::refine);
                        xt::masked_view(tag(level, i), !mask) = static_cast<int>(samurai::CellFlag::coarsen);
                    }
                }
            });
    }
}

template <class Field>
void save(const fs::path& path, const std::string& filename, const Field& u, const std::string& suffix = "")
{
    auto mesh   = u.mesh();
    auto level_ = samurai::make_field<std::size_t, 1>("level", mesh);

    if (!fs::exists(path))
    {
        fs::create_directory(path);
    }

    samurai::for_each_cell(mesh,
                           [&](const auto& cell)
                           {
                               level_[cell] = cell.level;
                           });

    samurai::save(path, fmt::format("{}{}", filename, suffix), mesh, u, level_);
}

template <class Field>
void flux_correction(Field& phi_np1, const Field& phi_n, double dt)
{
    using mesh_t     = typename Field::mesh_t;
    using mesh_id_t  = typename mesh_t::mesh_id_t;
    using interval_t = typename mesh_t::interval_t;

    auto mesh                   = phi_np1.mesh();
    const std::size_t min_level = mesh[mesh_id_t::cells].min_level();
    const std::size_t max_level = mesh[mesh_id_t::cells].max_level();

    const double dx = 1. / (1 << max_level);

    for (std::size_t level = min_level; level < max_level; ++level)
    {
        const double dx_loc = samurai::cell_length(level);
        xt::xtensor_fixed<int, xt::xshape<1>> stencil;

        stencil           = {-1};
        auto subset_right = samurai::intersection(samurai::translate(mesh[mesh_id_t::cells][level + 1], stencil),
                                                  mesh[mesh_id_t::cells][level])
                                .on(level);

        subset_right(
            [&](const auto& i, const auto&)
            {
                phi_np1(level, i) = phi_np1(level, i)
                                  + dt / dx_loc
                                        * (samurai::upwind_Burgers_op<interval_t>(level, i).right_flux(phi_n, dx / dt)
                                           - samurai::upwind_Burgers_op<interval_t>(level + 1, 2 * i + 1).right_flux(phi_n, dx / dt));
            });

        stencil          = {1};
        auto subset_left = samurai::intersection(samurai::translate(mesh[mesh_id_t::cells][level + 1], stencil),
                                                 mesh[mesh_id_t::cells][level])
                               .on(level);

        subset_left(
            [&](const auto& i, const auto&)
            {
                phi_np1(level, i) = phi_np1(level, i)
                                  - dt / dx_loc
                                        * (samurai::upwind_Burgers_op<interval_t>(level, i).left_flux(phi_n, dx / dt)
                                           - samurai::upwind_Burgers_op<interval_t>(level + 1, 2 * i).left_flux(phi_n, dx / dt));
            });
    }
}

int main(int argc, char* argv[])
{
    constexpr std::size_t dim = 1;
    using Config              = samurai::amr::Config<dim>;

    // Simulation parameters
    double left_box  = -3;
    double right_box = 3;
    double Tf        = 1.5; // We have blowup at t = 1
    double cfl       = 0.99;

    // AMR parameters
    std::size_t start_level = 6;
    std::size_t min_level   = 1;
    std::size_t max_level   = 6;
    bool correction         = false;

    // Output parameters
    fs::path path        = fs::current_path();
    std::string filename = "FV_AMR_Burgers_Hat_1d";
    std::size_t nfiles   = 1;

    CLI::App app{"Finite volume example for the Burgers equation in 2d using AMR"};
    app.add_option("--left", left_box, "The left border of the box")->capture_default_str()->group("Simulation parameters");
    app.add_option("--right", right_box, "The right border of the box")->capture_default_str()->group("Simulation parameters");
    app.add_option("--cfl", cfl, "The CFL")->capture_default_str()->group("Simulation parameters");
    app.add_option("--Tf", Tf, "Final time")->capture_default_str()->group("Simulation parameters");
    app.add_option("--start-level", start_level, "Start level of AMR")->capture_default_str()->group("AMR parameters");
    app.add_option("--min-level", min_level, "Minimum level of AMR")->capture_default_str()->group("AMR parameters");
    app.add_option("--max-level", max_level, "Maximum level of AMR")->capture_default_str()->group("AMR parameters");
    app.add_option("--with-correction", correction, "Apply flux correction at the interface of two refinement levels")
        ->capture_default_str()
        ->group("AMR parameters");
    app.add_option("--path", path, "Output path")->capture_default_str()->group("Ouput");
    app.add_option("--filename", filename, "File name prefix")->capture_default_str()->group("Ouput");
    app.add_option("--nfiles", nfiles, "Number of output files")->capture_default_str()->group("Ouput");
    CLI11_PARSE(app, argc, argv);

    const samurai::Box<double, dim> box({left_box}, {right_box});
    samurai::amr::Mesh<Config> mesh(box, start_level, min_level, max_level);

    auto phi = init_solution(mesh);
    samurai::make_bc<samurai::Neumann>(phi, 0.);

    auto phinp1 = samurai::make_field<double, 1>("phi", mesh);

    auto tag = samurai::make_field<int, 1>("tag", mesh);
    const xt::xtensor_fixed<int, xt::xshape<2, 1>> stencil_grad{{1}, {-1}};

    const double dx      = 1. / (1 << max_level);
    double dt            = 0.99 * dx;
    const double dt_save = Tf / static_cast<double>(nfiles);
    double t             = 0.;

    std::size_t nsave = 1;
    std::size_t nt    = 0;

    while (t != Tf)
    {
        // AMR adaptation
        std::size_t ite_adapt = 0;
        while (true)
        {
            std::cout << "\tmesh adaptation: " << ite_adapt++ << std::endl;
            samurai::update_ghost(phi);
            tag.resize();
            AMR_criteria(phi, tag);
            samurai::graduation(tag, stencil_grad);
            if (samurai::update_field(tag, phi))
            {
                break;
            }
        }

        t += dt;
        if (t > Tf)
        {
            dt += Tf - t;
            t = Tf;
        }

        std::cout << fmt::format("iteration {}: t = {}, dt = {}", nt++, t, dt) << std::endl;

        // Numerical scheme
        samurai::update_ghost(phi);
        phinp1.resize();
        phinp1 = phi - dt * samurai::upwind_Burgers(phi, dx / dt);
        if (correction)
        {
            flux_correction(phinp1, phi, dt);
        }

        std::swap(phi.array(), phinp1.array());

        if (t >= static_cast<double>(nsave + 1) * dt_save || t == Tf)
        {
            const std::string suffix = (nfiles != 1) ? fmt::format("_ite_{}", nsave++) : "";
            save(path, filename, phi, suffix);
        }
    }
    return 0;
}
