// Copyright 2018-2025 the samurai's authors
// SPDX-License-Identifier:  BSD-3-Clause

#include <samurai/io/hdf5.hpp>
#include <samurai/io/restart.hpp>
#include <samurai/mr/adapt.hpp>
#include <samurai/mr/mesh.hpp>
#include <samurai/samurai.hpp>
#include <samurai/schemes/fv.hpp>

#include <filesystem>
namespace fs = std::filesystem;

#include <chrono>
#include <numbers>
#include <thread>
#include <unistd.h>

#include "convection_nonlinear_osmp.hpp"

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
    samurai::save(path, fmt::format("{}_full_{}", filename, suffix), {true, true}, mesh, u, level_);
}

void check_diff(auto& mesh)
{
    using mesh_id_t = typename std::decay_t<decltype(mesh)>::mesh_id_t;

    mpi::communicator world;

    auto my_min_indices = mesh.subdomain().min_indices();

    bool different = false;
    for (const auto& neighbour : mesh.mpi_neighbourhood())
    {
        auto neighbour_min_indices = neighbour.mesh.subdomain().min_indices();

        xt::xtensor_fixed<int, xt::xshape<2>> translation{my_min_indices[0] - neighbour_min_indices[0],
                                                          my_min_indices[1] - neighbour_min_indices[1]};

        samurai::for_each_level(mesh,
                                [&](auto level)
                                {
                                    auto set  = samurai::difference(mesh[mesh_id_t::cells][level],
                                                                   samurai::translate(neighbour.mesh[mesh_id_t::cells][level],
                                                                                      translation >> (mesh.subdomain().level() - level)));
                                    different = !set.empty();
                                });
        if (different)
        {
            break;
        }
    }

    if (mpi::all_reduce(world, different, std::logical_or<>()))
    {
        for (const auto& neighbour : mesh.mpi_neighbourhood())
        {
            auto neighbour_min_indices = neighbour.mesh.subdomain().min_indices();

            xt::xtensor_fixed<int, xt::xshape<2>> translation{my_min_indices[0] - neighbour_min_indices[0],
                                                              my_min_indices[1] - neighbour_min_indices[1]};

            samurai::for_each_level(mesh,
                                    [&](auto level)
                                    {
                                        auto set = samurai::difference(mesh[mesh_id_t::cells][level],
                                                                       samurai::translate(neighbour.mesh[mesh_id_t::cells][level],
                                                                                          translation >> (mesh.subdomain().level() - level)));
                                        set(
                                            [&](const auto& i, const auto& index)
                                            {
                                                std::cout << "Difference found !! " << level << " " << i << " " << index << " for domain "
                                                          << world.rank() << " with subdomain " << neighbour.rank << "\n";
                                                different = true;
                                            });
                                    });
        }
        samurai::save("diff_mesh", mesh);
        throw std::runtime_error("Difference found between subdomains");
    }
}

auto get_box(const xt::xtensor_fixed<double, xt::xshape<2>>& min_corner, const xt::xtensor_fixed<double, xt::xshape<2>>& max_corner, int npx)
{
    mpi::communicator world;

    const xt::xtensor_fixed<double, xt::xshape<2>> pcoords{static_cast<double>(world.rank() % npx), static_cast<double>(world.rank() / npx)};

    auto length = max_corner - min_corner;

    return samurai::Box<double, 2>(min_corner + pcoords * length, max_corner + pcoords * length);
}

int main(int argc, char* argv[])
{
    static constexpr std::size_t dim = 2;

    auto& app = samurai::initialize("Finite volume example for the linear convection equation", argc, argv);

    mpi::communicator world;

    std::cout << world.rank() << " / " << world.size() << std::endl;

    std::cout << "------------------------- Burgers 2D with OSMP scheme -------------------------" << std::endl;

    //--------------------//
    // Program parameters //
    //--------------------//

    // Simulation parameters
    xt::xtensor_fixed<double, xt::xshape<dim>> min_corner = {-1., -1.};
    xt::xtensor_fixed<double, xt::xshape<dim>> max_corner = {1., 1.};

    // Time integration
    double Tf  = 0.5;
    double dt  = 0;
    double cfl = 0.95;
    double t   = 0.;

    // MPI parameters
    int npx = 1;
    int npy = 1;

    // Multiresolution parameters
    std::size_t min_level = 8;
    std::size_t max_level = 8;

    // Output parameters
    fs::path path        = fs::current_path();
    std::string filename = "burgers";
    std::size_t nfiles   = 0;

    bool pause = false;

    app.add_option("--min-corner", min_corner, "The min corner of the first box")->capture_default_str()->group("Simulation parameters");
    app.add_option("--max-corner", max_corner, "The max corner of the first box")->capture_default_str()->group("Simulation parameters");
    app.add_option("--Ti", t, "Initial time")->capture_default_str()->group("Simulation parameters");
    app.add_option("--Tf", Tf, "Final time")->capture_default_str()->group("Simulation parameters");
    app.add_option("--dt", dt, "Time step")->capture_default_str()->group("Simulation parameters");
    app.add_option("--cfl", cfl, "The CFL")->capture_default_str()->group("Simulation parameters");
    app.add_option("--npx", npx, "Number of processes in x direction")->capture_default_str()->group("MPI parameters");
    app.add_option("--npy", npy, "Number of processes in y direction")->capture_default_str()->group("MPI parameters");
    app.add_option("--path", path, "Output path")->capture_default_str()->group("Output");
    app.add_option("--filename", filename, "File name prefix")->capture_default_str()->group("Output");
    app.add_option("--nfiles", nfiles, "Number of output files")->capture_default_str()->group("Output");
    app.add_flag("--pause", pause, "Pause before starting the simulation")->group("Debugging");
    app.allow_extras();
    SAMURAI_PARSE(argc, argv);

    if (world.size() != npx * npy)
    {
        throw std::runtime_error("Number of MPI processes must be equal to npx * npy");
    }

    if (pause)
    {
        // Print the process ID (PID) for debugging or profiling purposes
        std::cout << "PID: " << ::getpid() << std::endl;
        // Pause execution for 10 seconds to allow for debugging or profiling attachment
        std::cout << "Pausing for 10 seconds..." << std::endl;
        std::this_thread::sleep_for(std::chrono::seconds(10));
    }

    double approx_box_tol = 0.05;
    double scaling_factor = 1;

    //--------------------//
    // Problem definition //
    //--------------------//

    auto config = samurai::mesh_config<dim>().min_level(4).max_level(10).max_stencil_size(4).graduation_width(2).disable_minimal_ghost_width();
    auto box = get_box(min_corner, max_corner, npx);
    samurai::CellArray<2> cells;
    for (std::size_t level = 0; level < cells.max_size; ++level)
    {
        cells[level].set_origin_point(min_corner);
        cells[level].set_scaling_factor(scaling_factor);
    }
    cells[max_level] = {max_level, box, min_corner, approx_box_tol, scaling_factor};
    auto mesh        = samurai::mra::make_mesh(cells, config);
    mesh.cfg().periodic(true);
    mesh.box_like();
    mesh = {cells, mesh};

    auto u    = samurai::make_scalar_field<double>("u", mesh);
    auto unp1 = samurai::make_scalar_field<double>("unp1", mesh);
    auto u1   = samurai::make_scalar_field<double>("u1", mesh);
    auto u2   = samurai::make_scalar_field<double>("u2", mesh);

    auto middle = xt::eval(0.5 * (box.min_corner() + box.max_corner()));
    middle += 0.25;

    // Initial solution
    samurai::for_each_cell(mesh,
                           [&](auto& cell)
                           {
                               double coef      = 0.5;
                               double stiffness = 50.;
                               u[cell]          = coef
                                       * std::exp(-stiffness
                                                  * ((cell.center(0) - middle(0)) * (cell.center(0) - middle(0))
                                                     + (cell.center(1) - middle(1)) * (cell.center(1) - middle(1))));
                           });

    auto MRadaptation = samurai::make_MRAdapt(u);
    auto mra_config   = samurai::mra_config().epsilon(1e-3);
    MRadaptation(mra_config);

    double dt_save    = nfiles == 0 ? dt : Tf / static_cast<double>(nfiles);
    std::size_t nsave = 0, nt = 0;
    if (nfiles != 1)
    {
        std::string suffix = (nfiles != 1) ? fmt::format("_level_{}_{}_np_{}_{}_ite_{}", min_level, max_level, npx, npy, nsave) : "";
        save(path, filename, u, suffix);
    }

    // Convection operator
    xt::xtensor_fixed<double, xt::xshape<dim>> velocity = {0.5, 0.5};
    static constexpr std::size_t norder                 = 2;
    auto conv                                           = samurai::make_convection_nonlinear_osmp<decltype(u), norder>(dt);

    //--------------------//
    //   Time iteration   //
    //--------------------//

    double dx = mesh.cell_length(max_level);
    dt        = cfl * dx / velocity(0);

    while (t != Tf)
    {
        // Move to next timestep
        t += dt;
        if (t > Tf)
        {
            dt += Tf - t;
            t = Tf;
        }
        std::cout << fmt::format("iteration {}: t = {:.12f}, dt = {}", nt++, t, dt) << std::flush;

        // Mesh adaptation
        MRadaptation(mra_config);
        check_diff(mesh);
        u1.resize();
        u2.resize();
        unp1.resize();

        u1   = u - 0.5 * dt * conv(0, u);
        u2   = u1 - dt * conv(1, u1);
        unp1 = u2 - 0.5 * dt * conv(0, u2);

        // u <-- unp1
        std::swap(u.array(), unp1.array());

        // Save the result
        if (nfiles == 0 || t >= static_cast<double>(nsave + 1) * dt_save || t == Tf)
        {
            std::cout << "  (saving results)" << std::flush;
            std::string suffix = (nfiles != 1) ? fmt::format("_level_{}_{}_np_{}_{}_ite_{}", min_level, max_level, npx, npy, nsave) : "";
            save(path, filename, u, suffix);
            nsave++;
        }

        std::cout << std::endl;
    }

    samurai::finalize();
    return 0;
}
