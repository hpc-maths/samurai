// Copyright 2021 SAMURAI TEAM. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.
#include "CLI/CLI.hpp"

#include <xtensor/xfixed.hpp>

#include <samurai/algorithm.hpp>
#include <samurai/bc.hpp>
#include <samurai/field.hpp>
#include <samurai/hdf5.hpp>
#include <samurai/mr/adapt.hpp>
#include <samurai/mr/mesh.hpp>
#include <samurai/stencil_field.hpp>
#include <samurai/subset/subset_op.hpp>

#include <filesystem>
namespace fs = std::filesystem;

template <class Mesh>
auto init(Mesh& mesh)
{
    auto u = samurai::make_field<double, 1>("u", mesh);
    u.fill(0.);

    samurai::for_each_cell(mesh,
                           [&](auto& cell)
                           {
                               auto center         = cell.center();
                               const double radius = .2;

                               const double x_center = 0;
                               if (std::abs(center[0] - x_center) <= radius)
                               {
                                   u[cell] = 1;
                               }
                           });

    return u;
}

template <class Field>
void flux_correction(double dt, double a, const Field& u, Field& unp1)
{
    using mesh_t              = typename Field::mesh_t;
    using mesh_id_t           = typename mesh_t::mesh_id_t;
    using interval_t          = typename mesh_t::interval_t;
    constexpr std::size_t dim = Field::dim;

    auto mesh = u.mesh();

    for (std::size_t level = mesh.min_level(); level < mesh.max_level(); ++level)
    {
        xt::xtensor_fixed<int, xt::xshape<dim>> stencil;

        stencil = {{-1}};

        auto subset_right = samurai::intersection(samurai::translate(mesh[mesh_id_t::cells][level + 1], stencil),
                                                  mesh[mesh_id_t::cells][level])
                                .on(level);

        subset_right(
            [&](const auto& i, auto)
            {
                const double dx = samurai::cell_length(level);

                unp1(level, i) = unp1(level, i)
                               - dt / dx
                                     * (-samurai::upwind_op<interval_t>(level, i).right_flux(a, u)
                                        + samurai::upwind_op<interval_t>(level + 1, 2 * i + 1).right_flux(a, u));
            });

        stencil = {{1}};

        auto subset_left = samurai::intersection(samurai::translate(mesh[mesh_id_t::cells][level + 1], stencil),
                                                 mesh[mesh_id_t::cells][level])
                               .on(level);

        subset_left(
            [&](const auto& i, auto)
            {
                const double dx = samurai::cell_length(level);

                unp1(level, i) = unp1(level, i)
                               - dt / dx
                                     * (samurai::upwind_op<interval_t>(level, i).left_flux(a, u)
                                        - samurai::upwind_op<interval_t>(level + 1, 2 * i).left_flux(a, u));
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

int main(int argc, char* argv[])
{
    constexpr std::size_t dim = 1;
    using Config              = samurai::MRConfig<dim>;

    // Simulation parameters
    double left_box  = -2;
    double right_box = 2;
    bool is_periodic = false;
    double a         = 1.;
    double Tf        = 1.;
    double cfl       = 0.95;

    // Multiresolution parameters
    std::size_t min_level = 4;
    std::size_t max_level = 10;
    double mr_epsilon     = 2.e-4; // Threshold used by multiresolution
    double mr_regularity  = 1.;    // Regularity guess for multiresolution
    bool correction       = false;

    // Output parameters
    fs::path path        = fs::current_path();
    std::string filename = "FV_advection_1d";
    std::size_t nfiles   = 1;

    CLI::App app{"Finite volume example for the advection equation in 1d "
                 "using multiresolution"};
    app.add_option("--left", left_box, "The left border of the box")->capture_default_str()->group("Simulation parameters");
    app.add_option("--right", right_box, "The right border of the box")->capture_default_str()->group("Simulation parameters");
    app.add_flag("--periodic", is_periodic, "Set the domain periodic")->capture_default_str()->group("Simulation parameters");
    app.add_option("--velocity", a, "The velocity of the advection equation")->capture_default_str()->group("Simulation parameters");
    app.add_option("--cfl", cfl, "The CFL")->capture_default_str()->group("Simulation parameters");
    app.add_option("--Tf", Tf, "Final time")->capture_default_str()->group("Simulation parameters");
    app.add_option("--min-level", min_level, "Minimum level of the multiresolution")->capture_default_str()->group("Multiresolution");
    app.add_option("--max-level", max_level, "Maximum level of the multiresolution")->capture_default_str()->group("Multiresolution");
    app.add_option("--mr-eps", mr_epsilon, "The epsilon used by the multiresolution to adapt the mesh")
        ->capture_default_str()
        ->group("Multiresolution");
    app.add_option("--mr-reg",
                   mr_regularity,
                   "The regularity criteria used by the multiresolution to "
                   "adapt the mesh")
        ->capture_default_str()
        ->group("Multiresolution");
    app.add_option("--with-correction", correction, "Apply flux correction at the interface of two refinement levels")
        ->capture_default_str()
        ->group("Multiresolution");
    app.add_option("--path", path, "Output path")->capture_default_str()->group("Ouput");
    app.add_option("--filename", filename, "File name prefix")->capture_default_str()->group("Ouput");
    app.add_option("--nfiles", nfiles, "Number of output files")->capture_default_str()->group("Ouput");
    CLI11_PARSE(app, argc, argv);

    const samurai::Box<double, dim> box({left_box}, {right_box});
    samurai::MRMesh<Config> mesh(box, min_level, max_level, {is_periodic});

    double dt            = cfl / (1 << max_level);
    const double dt_save = Tf / static_cast<double>(nfiles);
    double t             = 0.;

    auto u = init(mesh);
    if (!is_periodic)
    {
        const xt::xtensor_fixed<int, xt::xshape<1>> left{-1};
        const xt::xtensor_fixed<int, xt::xshape<1>> right{1};
        samurai::make_bc<samurai::Dirichlet>(u, 0.)->on(left, right);
        // same as (just to test OnDirection instead of Everywhere)
        // samurai::make_bc<samurai::Dirichlet>(u, 0.);
    }
    auto unp1 = samurai::make_field<double, 1>("unp1", mesh);

    auto MRadaptation = samurai::make_MRAdapt(u);
    MRadaptation(mr_epsilon, mr_regularity);
    save(path, filename, u, "_init");

    std::size_t nsave = 1;
    std::size_t nt    = 0;

    while (t != Tf)
    {
        MRadaptation(mr_epsilon, mr_regularity);

        t += dt;
        if (t > Tf)
        {
            dt += Tf - t;
            t = Tf;
        }

        std::cout << fmt::format("iteration {}: t = {}, dt = {}", nt++, t, dt) << std::endl;

        samurai::update_ghost_mr(u);
        unp1.resize();
        unp1.fill(0);
        unp1 = u - dt * samurai::upwind(a, u);
        if (correction)
        {
            flux_correction(dt, a, u, unp1);
        }

        std::swap(u.array(), unp1.array());

        if (t >= static_cast<double>(nsave + 1) * dt_save || t == Tf)
        {
            const std::string suffix = (nfiles != 1) ? fmt::format("_ite_{}", nsave++) : "";
            save(path, filename, u, suffix);
        }
    }
    return 0;
}
