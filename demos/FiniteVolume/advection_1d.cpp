// Copyright 2018-2025 the samurai's authors
// SPDX-License-Identifier:  BSD-3-Clause
#include <xtensor/containers/xfixed.hpp>

#include <samurai/algorithm.hpp>
#include <samurai/bc.hpp>
#include <samurai/field.hpp>
#include <samurai/io/hdf5.hpp>
#include <samurai/io/restart.hpp>
#include <samurai/mr/adapt.hpp>
#include <samurai/mr/mesh.hpp>
#include <samurai/samurai.hpp>
#include <samurai/stencil_field.hpp>
#include <samurai/subset/node.hpp>

#include <filesystem>
namespace fs = std::filesystem;

template <class Field>
void init(Field& u)
{
    auto& mesh = u.mesh();
    u.resize();

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

int main(int argc, char* argv[])
{
    auto& app = samurai::initialize("Finite volume example for the advection equation in 1d using multiresolution", argc, argv);

    constexpr std::size_t dim = 1; // cppcheck-suppress unreadVariable

    // Simulation parameters
    double left_box  = -2;
    double right_box = 2;
    bool is_periodic = false;
    double a         = 1.;
    double Tf        = 1.;
    double cfl       = 0.95;
    double t         = 0.;
    std::string restart_file;

    // Output parameters
    fs::path path        = fs::current_path();
    std::string filename = "FV_advection_1d";
    std::size_t nfiles   = 1;

    app.add_option("--left", left_box, "The left border of the box")->capture_default_str()->group("Simulation parameters");
    app.add_option("--right", right_box, "The right border of the box")->capture_default_str()->group("Simulation parameters");
    app.add_flag("--periodic", is_periodic, "Set the domain periodic")->capture_default_str()->group("Simulation parameters");
    app.add_option("--velocity", a, "The velocity of the advection equation")->capture_default_str()->group("Simulation parameters");
    app.add_option("--cfl", cfl, "The CFL")->capture_default_str()->group("Simulation parameters");
    app.add_option("--Ti", t, "Initial time")->capture_default_str()->group("Simulation parameters");
    app.add_option("--Tf", Tf, "Final time")->capture_default_str()->group("Simulation parameters");
    app.add_option("--restart-file", restart_file, "Restart file")->capture_default_str()->group("Simulation parameters");
    app.add_option("--path", path, "Output path")->capture_default_str()->group("Output");
    app.add_option("--filename", filename, "File name prefix")->capture_default_str()->group("Output");
    app.add_option("--nfiles", nfiles, "Number of output files")->capture_default_str()->group("Output");

    SAMURAI_PARSE(argc, argv);

    const samurai::Box<double, dim> box({left_box}, {right_box});

    auto config = samurai::mesh_config<dim>().min_level(6).max_level(12).periodic(is_periodic).max_stencil_size(2).disable_minimal_ghost_width();
    auto mesh = samurai::make_MRMesh(config, box);
    // samurai::MRMesh<Config> mesh;
    auto u = samurai::make_scalar_field<double>("u", mesh);

    if (restart_file.empty())
    {
        mesh = {config, box};
        init(u);
    }
    else
    {
        samurai::load(restart_file, mesh, u);
    }

    double dt            = cfl * mesh.min_cell_length();
    const double dt_save = Tf / static_cast<double>(nfiles);

    if (!is_periodic)
    {
        const xt::xtensor_fixed<int, xt::xshape<1>> left{-1};
        const xt::xtensor_fixed<int, xt::xshape<1>> right{1};
        samurai::make_bc<samurai::Dirichlet<1>>(u, 0.)->on(left, right);
        // same as (just to test OnDirection instead of Everywhere)
        // samurai::make_bc<samurai::Dirichlet<1>>(u, 0.);
    }
    auto unp1 = samurai::make_scalar_field<double>("unp1", mesh);

    auto MRadaptation = samurai::make_MRAdapt(u);
    auto mra_config   = samurai::mra_config().epsilon(2e-4);
    MRadaptation(mra_config);
    save(path, filename, u, "_init");

    std::size_t nsave = 1;
    std::size_t nt    = 0;

    while (t != Tf)
    {
        MRadaptation(mra_config);

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

        std::swap(u.array(), unp1.array());

        if (t >= static_cast<double>(nsave) * dt_save || t == Tf)
        {
            const std::string suffix = (nfiles != 1) ? fmt::format("_ite_{}", nsave++) : "";
            save(path, filename, u, suffix);
        }
    }
    samurai::finalize();
    return 0;
}
