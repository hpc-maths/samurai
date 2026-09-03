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
#include <samurai/stencil_field.hpp>
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

/**
 * The command-line parameters, parsed once in main() and copied into each run. The options
 * used to be registered on the locals of the first main_fct() call and parsed again in the
 * second: CLI11 then wrote through pointers into a frame that no longer existed, which
 * corrupted whatever the second call kept there and ended, on some platforms, in a free() of
 * an invalid pointer at the end of the second run.
 */
struct Parameters
{
    static constexpr std::size_t dim = 2;

    xt::xtensor_fixed<double, xt::xshape<dim>> min_corner = {0., 0.};
    xt::xtensor_fixed<double, xt::xshape<dim>> max_corner = {1., 1.};
    std::array<double, dim> a{
        {1, 1}
    };
    double Tf  = .1;
    double cfl = 0.5;
    double t   = 0.;
    std::string restart_file;

    fs::path path        = fs::current_path();
    std::string filename = "FV_advection_2d";
    std::size_t nfiles   = 1;

    void register_options(CLI::App& app)
    {
        app.add_option("--min-corner", min_corner, "The min corner of the box")->capture_default_str()->group("Simulation parameters");
        app.add_option("--max-corner", max_corner, "The max corner of the box")->capture_default_str()->group("Simulation parameters");
        app.add_option("--velocity", a, "The velocity of the advection equation")->capture_default_str()->group("Simulation parameters");
        app.add_option("--cfl", cfl, "The CFL")->capture_default_str()->group("Simulation parameters");
        app.add_option("--Ti", t, "Initial time")->capture_default_str()->group("Simulation parameters");
        app.add_option("--Tf", Tf, "Final time")->capture_default_str()->group("Simulation parameters");
        app.add_option("--restart-file", restart_file, "Restart file")->capture_default_str()->group("Simulation parameters");
        app.add_option("--path", path, "Output path")->capture_default_str()->group("Output");
        app.add_option("--filename", filename, "File name prefix")->capture_default_str()->group("Output");
        app.add_option("--nfiles", nfiles, "Number of output files")->capture_default_str()->group("Output");
    }
};

template <std::size_t pred_stencil_size>
int main_fct(const Parameters& params)
{
    constexpr std::size_t dim = Parameters::dim;

    const auto& min_corner   = params.min_corner;
    const auto& max_corner   = params.max_corner;
    const auto& a            = params.a;
    const double Tf          = params.Tf;
    const double cfl         = params.cfl;
    double t                 = params.t;
    const auto& restart_file = params.restart_file;
    const auto& path         = params.path;
    std::string filename     = params.filename;
    const std::size_t nfiles = params.nfiles;

    filename = fmt::format("{}_pred_{}", filename, pred_stencil_size);

    const samurai::Box<double, dim> box(min_corner, max_corner);
    auto config = samurai::mesh_config<dim, pred_stencil_size>().min_level(4).max_level(10).max_stencil_size(2).disable_minimal_ghost_width();
    auto mesh = samurai::mra::make_empty_mesh(config);
    auto u    = samurai::make_scalar_field<double>("u", mesh);

    if (restart_file.empty())
    {
        mesh = samurai::mra::make_mesh(box, config);
        init(u);
    }
    else
    {
        samurai::load(restart_file, mesh, u);
    }
    samurai::make_bc<samurai::Dirichlet<1>>(u, 0.);

    double dt            = cfl * mesh.min_cell_length();
    const double dt_save = Tf / static_cast<double>(nfiles);

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
        unp1 = u - dt * samurai::upwind(a, u);

        std::swap(u.array(), unp1.array());

        if (t >= static_cast<double>(nsave) * dt_save || t == Tf)
        {
            const std::string suffix = (nfiles != 1) ? fmt::format("_ite_{}", nsave++) : "";
            save(path, filename, u, suffix);
        }
    }
    return 0;
}

int main(int argc, char* argv[])
{
    samurai::initialize("Finite volume example for the advection equation in 2d using multiresolution", argc, argv);
    Parameters params;
    params.register_options(samurai::app);
    SAMURAI_PARSE(argc, argv);
    main_fct<0>(params);
    main_fct<1>(params);
    samurai::finalize();
    return 0;
}
