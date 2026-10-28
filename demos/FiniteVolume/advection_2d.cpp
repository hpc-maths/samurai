// Copyright 2018-2025 the samurai's authors
// SPDX-License-Identifier:  BSD-3-Clause

#include <array>

#include <xtensor/containers/xfixed.hpp>

#include <samurai/algorithm.hpp>
#include <samurai/bc.hpp>
#include <samurai/field.hpp>
#include <samurai/io/hdf5.hpp>
#include <samurai/io/restart.hpp>
#include <samurai/mr/adapt.hpp>
#include <samurai/mr/mesh.hpp>
#include <samurai/samurai.hpp>
#include <samurai/schemes/fv.hpp>
#include <samurai/subset/node.hpp>

#include <filesystem>
namespace fs = std::filesystem;

template <class Field>
void init(Field& u)
{
    auto& mesh = u.mesh();
    u.resize();

    samurai::for_each_cell(
        mesh,
        [&](auto& cell)
        {
            auto center           = cell.center();
            const double radius   = .2;
            const double x_center = 0.3;
            const double y_center = 0.3;
            if (((center[0] - x_center) * (center[0] - x_center) + (center[1] - y_center) * (center[1] - y_center)) <= radius * radius)
            {
                u[cell] = 1;
            }
            else
            {
                u[cell] = 0;
            }
        });
}

template <class Field>
void save(const fs::path& path, const std::string& filename, const Field& u, const std::string& suffix = "")
{
    auto& mesh = u.mesh();

#ifdef SAMURAI_WITH_MPI
    mpi::communicator world;
    samurai::save(path, fmt::format("{}_size_{}{}", filename, world.size(), suffix), mesh, u);
#else
    samurai::save(path, fmt::format("{}{}", filename, suffix), mesh, u);
    samurai::dump(path, fmt::format("{}_restart{}", filename, suffix), mesh, u);
#endif
}

struct Parameters
{
    static constexpr std::size_t dim = 2;

    // Simulation parameters
    xt::xtensor_fixed<double, xt::xshape<dim>> min_corner = {0., 0.};
    xt::xtensor_fixed<double, xt::xshape<dim>> max_corner = {1., 1.};
    std::array<double, dim> a{
        {1, 1}
    };
    double Tf  = .1;
    double cfl = 0.5;
    double Ti  = 0.;
    std::string restart_file;

    // Output parameters
    fs::path path        = fs::current_path();
    std::string filename = "FV_advection_2d";
    std::size_t nfiles   = 1;
};

template <std::size_t pred_stencil_size>
void run(const Parameters& params)
{
    constexpr std::size_t dim = Parameters::dim;

    double t                   = params.Ti;
    const std::string filename = fmt::format("{}_pred_{}", params.filename, pred_stencil_size);

    const samurai::Box<double, dim> box(params.min_corner, params.max_corner);
    auto config = samurai::mesh_config<dim, pred_stencil_size>().min_level(4).max_level(10).max_stencil_size(2).disable_minimal_ghost_width();
    auto mesh = samurai::mra::make_empty_mesh(config);
    auto u    = samurai::make_scalar_field<double>("u", mesh);

    if (params.restart_file.empty())
    {
        mesh = samurai::mra::make_mesh(box, config);
        init(u);
    }
    else
    {
        samurai::load(params.restart_file, mesh, u);
    }
    samurai::make_bc<samurai::Dirichlet<1>>(u, 0.);

    double dt            = params.cfl * mesh.min_cell_length();
    const double dt_save = params.Tf / static_cast<double>(params.nfiles);

    auto unp1 = samurai::make_scalar_field<double>("unp1", mesh);

    samurai::VelocityVector<dim> velocity;
    for (std::size_t d = 0; d < dim; ++d)
    {
        velocity(d) = params.a[d];
    }
    auto conv = samurai::make_convection_upwind<decltype(u)>(velocity);

    auto MRadaptation = samurai::make_MRAdapt(u);
    auto mra_config   = samurai::mra_config().epsilon(2e-4);
    MRadaptation(mra_config);
    save(params.path, filename, u, "_init");

    std::size_t nsave = 1;
    std::size_t nt    = 0;

    while (t != params.Tf)
    {
        MRadaptation(mra_config);

        t += dt;
        if (t > params.Tf)
        {
            dt += params.Tf - t;
            t = params.Tf;
        }

        std::cout << fmt::format("iteration {}: t = {}, dt = {}", nt++, t, dt) << std::endl;

        unp1.resize();
        unp1 = u - dt * conv(u);

        samurai::swap(u, unp1);

        if (t >= static_cast<double>(nsave) * dt_save || t == params.Tf)
        {
            const std::string suffix = (params.nfiles != 1) ? fmt::format("_ite_{}", nsave++) : "";
            save(params.path, filename, u, suffix);
        }
    }
}

int main(int argc, char* argv[])
{
    auto& app = samurai::initialize("Finite volume example for the advection equation in 2d using multiresolution", argc, argv);

    // The options are bound to the members of params, which lives until the end of main.
    Parameters params;
    app.add_option("--min-corner", params.min_corner, "The min corner of the box")->capture_default_str()->group("Simulation parameters");
    app.add_option("--max-corner", params.max_corner, "The max corner of the box")->capture_default_str()->group("Simulation parameters");
    app.add_option("--velocity", params.a, "The velocity of the advection equation")->capture_default_str()->group("Simulation parameters");
    app.add_option("--cfl", params.cfl, "The CFL")->capture_default_str()->group("Simulation parameters");
    app.add_option("--Ti", params.Ti, "Initial time")->capture_default_str()->group("Simulation parameters");
    app.add_option("--Tf", params.Tf, "Final time")->capture_default_str()->group("Simulation parameters");
    app.add_option("--restart-file", params.restart_file, "Restart file")->capture_default_str()->group("Simulation parameters");
    app.add_option("--path", params.path, "Output path")->capture_default_str()->group("Output");
    app.add_option("--filename", params.filename, "File name prefix")->capture_default_str()->group("Output");
    app.add_option("--nfiles", params.nfiles, "Number of output files")->capture_default_str()->group("Output");
    SAMURAI_PARSE(argc, argv);

    run<0>(params);
    run<1>(params);
    samurai::finalize();
    return 0;
}
