// Copyright 2021 SAMURAI TEAM. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.
#include "CLI/CLI.hpp"

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
auto init_level_set(Mesh& mesh)
{
    using mesh_id_t = typename Mesh::mesh_id_t;

    auto phi = samurai::make_field<double, 1>("phi", mesh);

    samurai::for_each_cell(mesh[mesh_id_t::cells],
                           [&](auto& cell)
                           {
                               auto center    = cell.center();
                               const double x = center[0];
                               const double y = center[1];

                               constexpr double radius   = .15;
                               constexpr double x_center = 0.5;
                               constexpr double y_center = 0.75;

                               phi[cell] = std::sqrt(std::pow(x - x_center, 2.) + std::pow(y - y_center, 2.)) - radius;
                           });

    samurai::make_bc<samurai::Neumann>(phi, 0.);

    return phi;
}

template <class Mesh>
auto init_velocity(Mesh& mesh)
{
    using mesh_id_t = typename Mesh::mesh_id_t;
    const double PI = xt::numeric_constants<double>::PI;

    auto u = samurai::make_field<double, 2>("u", mesh);
    u.fill(0);

    samurai::for_each_cell(mesh[mesh_id_t::cells_and_ghosts],
                           [&](auto& cell)
                           {
                               auto center    = cell.center();
                               const double x = center[0];
                               const double y = center[1];

                               u[cell][0] = -std::pow(std::sin(PI * x), 2.) * std::sin(2. * PI * y);
                               u[cell][1] = std::pow(std::sin(PI * y), 2.) * std::sin(2. * PI * x);
                           });

    samurai::make_bc<samurai::Neumann>(u, 0., 0.);

    // samurai::make_bc<samurai::Dirichlet>(u, [PI](auto& coords)
    // {
    //     return xt::xtensor_fixed<double, xt::xshape<2>>{
    //         -std::pow(std::sin(PI*coords[0]), 2.) *
    //         std::sin(2.*PI*coords[1]),
    //          std::pow(std::sin(PI*coords[1]), 2.) * std::sin(2.*PI*coords[0])
    //     };
    // });

    return u;
}

template <class Field, class Tag>
void AMR_criteria(const Field& f, Tag& tag)
{
    auto mesh       = f.mesh();
    using mesh_id_t = typename Field::mesh_t::mesh_id_t;

    const std::size_t min_level = mesh.min_level();
    const std::size_t max_level = mesh.max_level();

    samurai::for_each_cell(mesh[mesh_id_t::cells],
                           [&](auto cell)
                           {
                               const double dx = 1. / (1 << (max_level));

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
    using mesh_t     = typename Field::mesh_t;
    using mesh_id_t  = typename mesh_t::mesh_id_t;
    using interval_t = typename mesh_t::interval_t;

    auto mesh                   = phi_np1.mesh();
    const std::size_t min_level = mesh[mesh_id_t::cells].min_level();
    const std::size_t max_level = mesh[mesh_id_t::cells].max_level();
    for (std::size_t level = min_level; level < max_level; ++level)
    {
        xt::xtensor_fixed<int, xt::xshape<2>> stencil;

        stencil = {
            {-1, 0}
        };

        auto subset_right = samurai::intersection(samurai::translate(mesh[mesh_id_t::cells][level + 1], stencil),
                                                  mesh[mesh_id_t::cells][level])
                                .on(level);

        subset_right(
            [&](const auto& i, const auto& index)
            {
                auto j          = index[0];
                const double dx = samurai::cell_length(level);

                phi_np1(level,
                        i,
                        j) = phi_np1(level, i, j)
                           + dt / dx
                                 * (samurai::upwind_variable_op<interval_t>(level, i, j).right_flux(u, phi_n, dt)
                                    - .5 * samurai::upwind_variable_op<interval_t>(level + 1, 2 * i + 1, 2 * j).right_flux(u, phi_n, dt)
                                    - .5 * samurai::upwind_variable_op<interval_t>(level + 1, 2 * i + 1, 2 * j + 1).right_flux(u, phi_n, dt));
            });

        stencil = {
            {1, 0}
        };

        auto subset_left = samurai::intersection(samurai::translate(mesh[mesh_id_t::cells][level + 1], stencil),
                                                 mesh[mesh_id_t::cells][level])
                               .on(level);

        subset_left(
            [&](const auto& i, const auto& index)
            {
                auto j          = index[0];
                const double dx = samurai::cell_length(level);

                phi_np1(level,
                        i,
                        j) = phi_np1(level, i, j)
                           - dt / dx
                                 * (samurai::upwind_variable_op<interval_t>(level, i, j).left_flux(u, phi_n, dt)
                                    - .5 * samurai::upwind_variable_op<interval_t>(level + 1, 2 * i, 2 * j).left_flux(u, phi_n, dt)
                                    - .5 * samurai::upwind_variable_op<interval_t>(level + 1, 2 * i, 2 * j + 1).left_flux(u, phi_n, dt));
            });

        stencil = {
            {0, -1}
        };

        auto subset_up = samurai::intersection(samurai::translate(mesh[mesh_id_t::cells][level + 1], stencil), mesh[mesh_id_t::cells][level])
                             .on(level);

        subset_up(
            [&](const auto& i, const auto& index)
            {
                auto j          = index[0];
                const double dx = samurai::cell_length(level);

                phi_np1(level,
                        i,
                        j) = phi_np1(level, i, j)
                           + dt / dx
                                 * (samurai::upwind_variable_op<interval_t>(level, i, j).up_flux(u, phi_n, dt)
                                    - .5 * samurai::upwind_variable_op<interval_t>(level + 1, 2 * i, 2 * j + 1).up_flux(u, phi_n, dt)
                                    - .5 * samurai::upwind_variable_op<interval_t>(level + 1, 2 * i + 1, 2 * j + 1).up_flux(u, phi_n, dt));
            });

        stencil = {
            {0, 1}
        };

        auto subset_down = samurai::intersection(samurai::translate(mesh[mesh_id_t::cells][level + 1], stencil),
                                                 mesh[mesh_id_t::cells][level])
                               .on(level);

        subset_down(
            [&](const auto& i, const auto& index)
            {
                auto j          = index[0];
                const double dx = samurai::cell_length(level);

                phi_np1(level,
                        i,
                        j) = phi_np1(level, i, j)
                           - dt / dx
                                 * (samurai::upwind_variable_op<interval_t>(level, i, j).down_flux(u, phi_n, dt)
                                    - .5 * samurai::upwind_variable_op<interval_t>(level + 1, 2 * i, 2 * j).down_flux(u, phi_n, dt)
                                    - .5 * samurai::upwind_variable_op<interval_t>(level + 1, 2 * i + 1, 2 * j).down_flux(u, phi_n, dt));
            });
    }
}

template <class Field, class Phi>
void save(const fs::path& path, const std::string& filename, const Field& u, const Phi& phi, const std::string& suffix = "")
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

    samurai::save(path, fmt::format("{}{}", filename, suffix), mesh, phi, u, level_);
}

int main(int argc, char* argv[])
{
    constexpr std::size_t dim = 2;
    using Config              = samurai::amr::Config<dim, 2>;

    // Simulation parameters
    xt::xtensor_fixed<double, xt::xshape<dim>> min_corner = {0., 0.};
    xt::xtensor_fixed<double, xt::xshape<dim>> max_corner = {1., 1.};
    double Tf                                             = 3.14;
    double cfl                                            = 5. / 8;

    // AMR parameters
    std::size_t start_level = 8;
    std::size_t min_level   = 4;
    std::size_t max_level   = 8;
    bool correction         = false;

    // Output parameters
    fs::path path        = fs::current_path();
    std::string filename = "FV_level_set_2d";
    std::size_t nfiles   = 1;

    CLI::App app{"Finite volume example with a level set in 2d using AMR"};
    app.add_option("--min-corner", min_corner, "The min corner of the box")->capture_default_str()->group("Simulation parameters");
    app.add_option("--max-corner", max_corner, "The max corner of the box")->capture_default_str()->group("Simulation parameters");
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

    const samurai::Box<double, dim> box(min_corner, max_corner);
    samurai::amr::Mesh<Config> mesh(box, start_level, min_level, max_level);

    double dt            = cfl / (1 << max_level);
    const double dt_save = Tf / static_cast<double>(nfiles);
    double t             = 0.;

    auto phi = init_level_set(mesh);
    auto u   = init_velocity(mesh);

    auto phinp1 = samurai::make_field<double, 1>("phi", mesh);
    auto phihat = samurai::make_field<double, 1>("phi", mesh);
    samurai::make_bc<samurai::Neumann>(phihat, 0.);
    auto tag = samurai::make_field<int, 1>("tag", mesh);

    const xt::xtensor_fixed<int, xt::xshape<4, 2>> stencil_grad{
        {1,  0 },
        {-1, 0 },
        {0,  1 },
        {0,  -1}
    };

    std::size_t nsave = 1;
    std::size_t nt    = 0;

    while (t != Tf)
    {
        // AMR adaptation
        std::size_t ite = 0;
        while (true)
        {
            std::cout << "Mesh adaptation iteration " << ite++ << std::endl;
            tag.resize();
            AMR_criteria(phi, tag);
            samurai::graduation(tag, stencil_grad);
            samurai::update_ghost(phi, u);
            if ((samurai::update_field(tag, phi, u)))
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
        samurai::update_ghost(phi, u);
        phinp1.resize();
        phinp1 = phi - dt * samurai::upwind_variable(u, phi, dt);
        flux_correction(phinp1, phi, u, dt);

        std::swap(phi.array(), phinp1.array());

        // Reinitialization of the level set
        const std::size_t fict_iteration = 2;         // Number of fictitious iterations
        const double dt_fict             = 0.01 * dt; // Fictitious Time step

        auto phi_0 = phi;
        for (std::size_t k = 0; k < fict_iteration; ++k)
        {
            // Forward Euler
            // update_ghosts(phi, u, update_bc_for_level);
            // phinp1 = phi - dt_fict * H_wrap(phi, phi_0, max_level);

            // TVD-RK2
            samurai::update_ghost(phi);
            phihat.resize();
            phihat = phi - dt_fict * H_wrap(phi, phi_0, max_level);
            samurai::update_ghost(phihat);
            phinp1 = .5 * phi_0 + .5 * (phihat - dt_fict * H_wrap(phihat, phi_0, max_level));

            std::swap(phi.array(), phinp1.array());
        }

        if (t >= static_cast<double>(nsave + 1) * dt_save || t == Tf)
        {
            const std::string suffix = (nfiles != 1) ? fmt::format("_ite_{}", nsave++) : "";
            save(path, filename, u, phi, suffix);
        }
    }

    return 0;
}
