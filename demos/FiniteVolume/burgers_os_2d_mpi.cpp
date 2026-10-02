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
#include <cstdint>
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
    samurai::save(path, fmt::format("{}_full{}", filename, suffix), {true, true}, mesh, u, level_);
}

// Every rank solves the same problem on its own box: in the coordinates of its
// subdomain, every rank must hold the same cells. Each rank hashes its cells in
// those coordinates and two all_reduce compare the hashes of all the ranks. On a
// mismatch, rank 0 broadcasts its cells and every rank that differs lists the
// cells it has and rank 0 has not, and conversely, before the run stops.
void check_diff(auto& mesh)
{
    using mesh_id_t           = typename std::decay_t<decltype(mesh)>::mesh_id_t;
    static constexpr auto dim = std::decay_t<decltype(mesh)>::dim;

    mpi::communicator world;

    const auto origin    = mesh.subdomain().min_indices();
    const auto top_level = mesh.subdomain().level();

    std::uint64_t hash = 14695981039346656037ULL; // FNV-1a
    auto mix           = [&](std::int64_t value)
    {
        hash ^= static_cast<std::uint64_t>(value);
        hash *= 1099511628211ULL;
    };

    samurai::CellList<dim> local_cl;
    samurai::for_each_level(mesh,
                            [&](auto level)
                            {
                                xt::xtensor_fixed<int, xt::xshape<dim>> shift;
                                for (std::size_t d = 0; d < dim; ++d)
                                {
                                    shift[d] = -(origin[d] >> (top_level - level));
                                }
                                samurai::translate(mesh[mesh_id_t::cells][level], shift)(
                                    [&](const auto& i, const auto& index)
                                    {
                                        mix(static_cast<std::int64_t>(level));
                                        mix(i.start);
                                        mix(i.end);
                                        for (std::size_t d = 0; d < dim - 1; ++d)
                                        {
                                            mix(index[d]);
                                        }
                                        local_cl[level][index].add_interval(i);
                                    });
                            });

    const auto hash_min = mpi::all_reduce(world, hash, mpi::minimum<std::uint64_t>());
    const auto hash_max = mpi::all_reduce(world, hash, mpi::maximum<std::uint64_t>());
    if (hash_min == hash_max)
    {
        return;
    }

    samurai::CellArray<dim> local{local_cl};
    samurai::CellArray<dim> reference = local;
    mpi::broadcast(world, reference, 0);
    for (std::size_t level = 0; level < local.max_size; ++level)
    {
        // std::cerr: std::cout is redirected to /dev/null on every rank but 0
        samurai::difference(local[level], reference[level])(
            [&](const auto& i, const auto& index)
            {
                std::cerr << "Difference found !! level " << level << " " << i << " " << index << " on rank " << world.rank()
                          << " but not on rank 0\n";
            });
        samurai::difference(reference[level], local[level])(
            [&](const auto& i, const auto& index)
            {
                std::cerr << "Difference found !! level " << level << " " << i << " " << index << " on rank 0 but not on rank "
                          << world.rank() << "\n";
            });
    }
    samurai::save("diff_mesh", mesh);
    throw std::runtime_error("Difference found between subdomains");
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

    // Output parameters
    fs::path path        = fs::current_path();
    std::string filename = "burgers";
    std::size_t nfiles   = 1;
    bool no_output       = false;

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
    app.add_option("--nfiles", nfiles, "Number of output files (0 saves every time step)")->capture_default_str()->group("Output");
    app.add_flag("--no-output", no_output, "Do not write any output file")->group("Output");
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

    // The levels are those of the mesh configuration, overridable by --min-level and --max-level.
    // The multiresolution analysis requires the initial mesh to be uniform at the max level;
    // it is adapted before the first time step.
    auto config = samurai::mesh_config<dim>().min_level(4).max_level(10).max_stencil_size(4).graduation_width(2).disable_minimal_ghost_width();
    config.parse_args();
    const std::size_t max_level = config.max_level();

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

    auto save_solution = [&](std::size_t isave)
    {
        std::string suffix = (nfiles != 1) ? fmt::format("_level_{}_{}_np_{}_{}_ite_{}", mesh.min_level(), mesh.max_level(), npx, npy, isave)
                                           : "";
        save(path, filename, u, suffix);
    };

    // Save k is written at t >= k * dt_save. Save 0 is the initial solution, skipped when only the final one is wanted.
    if (!no_output && nfiles != 1)
    {
        save_solution(0);
    }
    std::size_t nsave = 1, nt = 0;

    // Convection operator
    xt::xtensor_fixed<double, xt::xshape<dim>> velocity = {0.5, 0.5};
    static constexpr std::size_t norder                 = 2;
    auto conv                                           = samurai::make_convection_nonlinear_osmp<decltype(u), norder>(dt);

    //--------------------//
    //   Time iteration   //
    //--------------------//

    // The time step must satisfy the CFL condition on the finest level the mesh may reach,
    // not on the level of the initial mesh.
    if (dt == 0.)
    {
        dt = cfl * mesh.min_cell_length() / velocity(0);
    }
    const double dt_save = (nfiles == 0) ? dt : Tf / static_cast<double>(nfiles);

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
        if (!no_output && (nfiles == 0 || t >= static_cast<double>(nsave) * dt_save || t == Tf))
        {
            std::cout << "  (saving results)" << std::flush;
            save_solution(nsave++);
        }

        std::cout << std::endl;
    }

    samurai::finalize();
    return 0;
}
