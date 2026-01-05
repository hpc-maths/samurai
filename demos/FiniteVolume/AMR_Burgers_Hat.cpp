// Copyright 2018-2025 the samurai's authors
// SPDX-License-Identifier:  BSD-3-Clause

#include <fmt/format.h>
#include <iostream>

#include <samurai/algorithm/graduation.hpp>
#include <samurai/algorithm/update.hpp>
#include <samurai/amr/mesh.hpp>
#include <samurai/bc.hpp>
#include <samurai/box.hpp>
#include <samurai/field.hpp>
#include <samurai/io/hdf5.hpp>
#include <samurai/io/restart.hpp>
#include <samurai/print.hpp>
#include <samurai/samurai.hpp>

#include "stencil_field.hpp"

#include "../LBM/boundary_conditions.hpp"

#include <filesystem>
namespace fs = std::filesystem;

template <class Field>
void init_solution(Field& phi)
{
    using mesh_id_t = typename Field::mesh_t::mesh_id_t;

    auto& mesh = phi.mesh();
    phi.resize();
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
}

template <class Field, class Tag>
void AMR_criteria(const Field& f, Tag& tag)
{
    using namespace samurai::math;
    auto mesh             = f.mesh();
    using mesh_id_t       = typename Field::mesh_t::mesh_id_t;
    std::size_t min_level = mesh.min_level();
    std::size_t max_level = mesh.max_level();

    tag.fill(static_cast<int>(samurai::CellFlag::keep));

    for (std::size_t level = min_level; level <= max_level; ++level)
    {
        const double dx = mesh.cell_length(level);

        samurai::for_each_interval(
            mesh[mesh_id_t::cells][level],
            [&](std::size_t, auto& i, auto)
            {
                auto der_approx     = samurai::eval(abs((f(level, i + 1) - f(level, i - 1)) / (2. * dx)));
                auto der_der_approx = samurai::eval(abs((f(level, i + 1) - 2. * f(level, i) + f(level, i - 1)) / (dx * dx)));

                auto der_plus  = samurai::eval(abs((f(level, i + 1) - f(level, i)) / (dx)));
                auto der_minus = samurai::eval(abs((f(level, i) - f(level, i - 1)) / (dx)));

                // auto mask = xt::abs(f(level, i)) > 0.001;
                auto mask = der_approx > 0.01;
                // auto mask = der_der_approx > 0.01;
                // auto mask = (xt::abs(der_plus) - xt::abs(der_minus)) > 0.001;

                if (level == max_level)
                {
                    samurai::apply_on_masked(tag(level, i),
                                             mask,
                                             [](auto& e)
                                             {
                                                 e = static_cast<int>(samurai::CellFlag::keep);
                                             });
                    samurai::apply_on_masked(tag(level, i),
                                             !mask,
                                             [](auto& e)
                                             {
                                                 e = static_cast<int>(samurai::CellFlag::coarsen);
                                             });
                }
                else
                {
                    if (level == min_level)
                    {
                        tag(level, i) = static_cast<int>(samurai::CellFlag::keep);
                    }
                    else
                    {
                        samurai::apply_on_masked(tag(level, i),
                                                 mask,
                                                 [](auto& e)
                                                 {
                                                     e = static_cast<int>(samurai::CellFlag::refine);
                                                 });
                        samurai::apply_on_masked(tag(level, i),
                                                 !mask,
                                                 [](auto& e)
                                                 {
                                                     e = static_cast<int>(samurai::CellFlag::coarsen);
                                                 });
                    }
                }
            });
    }
}

template <class Field>
void save(const fs::path& path, const std::string& filename, const Field& u, const std::string& suffix = "")
{
    auto mesh   = u.mesh();
    auto level_ = samurai::make_scalar_field<std::size_t>("level", mesh);

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
    samurai::dump(path, fmt::format("{}_restart{}", filename, suffix), mesh, u);
}

template <class Field>
void flux_correction(Field& phi_np1, const Field& phi_n, double dt)
{
    using mesh_t                     = typename Field::mesh_t;
    static constexpr std::size_t dim = Field::dim;
    using mesh_id_t                  = typename mesh_t::mesh_id_t;
    using interval_t                 = typename mesh_t::interval_t;

    auto mesh                   = phi_np1.mesh();
    const std::size_t min_level = mesh[mesh_id_t::cells].min_level();
    const std::size_t max_level = mesh[mesh_id_t::cells].max_level();

    const double dx = mesh.cell_length(max_level);

    for (std::size_t level = min_level; level < max_level; ++level)
    {
        const double dx_loc = mesh.cell_length(level);
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
                                        * (samurai::upwind_Burgers_op<dim, interval_t>(level, i).right_flux(phi_n, dx / dt)
                                           - samurai::upwind_Burgers_op<dim, interval_t>(level + 1, 2 * i + 1).right_flux(phi_n, dx / dt));
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
                                        * (samurai::upwind_Burgers_op<dim, interval_t>(level, i).left_flux(phi_n, dx / dt)
                                           - samurai::upwind_Burgers_op<dim, interval_t>(level + 1, 2 * i).left_flux(phi_n, dx / dt));
            });
    }
}

int main(int argc, char* argv[])
{
    auto& app = samurai::initialize("Finite volume example for the Burgers equation in 2d using AMR", argc, argv);

    constexpr std::size_t dim = 1; // cppcheck-suppress unreadVariable

    // Simulation parameters
    double left_box  = -3;
    double right_box = 3;
    double Tf        = 1.5; // We have blowup at t = 1
    double cfl       = 0.99;
    double t         = 0.;
    std::string restart_file;

    // AMR parameters
    bool correction = false;

    // Output parameters
    fs::path path        = fs::current_path();
    std::string filename = "FV_AMR_Burgers_Hat_1d";
    std::size_t nfiles   = 1;

    app.add_option("--left", left_box, "The left border of the box")->capture_default_str()->group("Simulation parameters");
    app.add_option("--right", right_box, "The right border of the box")->capture_default_str()->group("Simulation parameters");
    app.add_option("--cfl", cfl, "The CFL")->capture_default_str()->group("Simulation parameters");
    app.add_option("--Ti", t, "Initial time")->capture_default_str()->group("Simulation parameters");
    app.add_option("--Tf", Tf, "Final time")->capture_default_str()->group("Simulation parameters");
    app.add_option("--restart-file", restart_file, "Restart file")->capture_default_str()->group("Simulation parameters");
    app.add_option("--with-correction", correction, "Apply flux correction at the interface of two refinement levels")
        ->capture_default_str()
        ->group("AMR parameters");
    app.add_option("--path", path, "Output path")->capture_default_str()->group("Output");
    app.add_option("--filename", filename, "File name prefix")->capture_default_str()->group("Output");
    app.add_option("--nfiles", nfiles, "Number of output files")->capture_default_str()->group("Output");
    SAMURAI_PARSE(argc, argv);

    const samurai::Box<double, dim> box({left_box}, {right_box});
    auto config = samurai::mesh_config<dim>().min_level(2).max_level(7).max_stencil_size(2).start_level(7).disable_minimal_ghost_width();
    auto mesh   = samurai::amr::make_empty_mesh(config);
    auto phi    = samurai::make_scalar_field<double>("phi", mesh);

    if (restart_file.empty())
    {
        mesh = samurai::amr::make_mesh(box, config);
        init_solution(phi);
    }
    else
    {
        samurai::load(restart_file, mesh, phi);
    }

    samurai::make_bc<samurai::Neumann<1>>(phi, 0.);

    auto phinp1 = samurai::make_scalar_field<double>("phi", mesh);

    auto tag = samurai::make_scalar_field<int>("tag", mesh);
    const xt::xtensor_fixed<int, xt::xshape<2, 1>> stencil_grad{{1}, {-1}};

    const double dx      = mesh.min_cell_length();
    double dt            = 0.99 * dx;
    const double dt_save = Tf / static_cast<double>(nfiles);

    std::size_t nsave = 1;
    std::size_t nt    = 0;

    while (t != Tf)
    {
        // AMR adaptation
        std::size_t ite_adapt = 0;
        while (true)
        {
            samurai::io::print("mesh adaptation: {}\n", ite_adapt++);
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

        samurai::io::print(samurai::io::root, "iteration {}: t = {}, dt = {}\n", nt++, t, dt);

        // Numerical scheme
        samurai::update_ghost(phi);
        phinp1.resize();
        phinp1 = phi - dt * samurai::upwind_Burgers(phi, dx / dt);
        if (correction)
        {
            flux_correction(phinp1, phi, dt);
        }

        std::swap(phi.array(), phinp1.array());

        if (t >= static_cast<double>(nsave) * dt_save || t == Tf)
        {
            const std::string suffix = (nfiles != 1) ? fmt::format("_ite_{}", nsave++) : "";
            save(path, filename, phi, suffix);
        }
    }
    samurai::finalize();
    return 0;
}
