// Copyright 2021 SAMURAI TEAM. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.
#include "CLI/CLI.hpp"

#include <samurai/mr/mesh.hpp>
#include <samurai/mr/adapt.hpp>
#include <samurai/hdf5.hpp>
#include <samurai/petsc.hpp>

#include <filesystem>
namespace fs = std::filesystem;

auto exact_solution(double x, double t)
{
    assert(t > 0 && "t must be > 0");
    return 1/(2*sqrt(M_PI*t)) * exp(-x*x/(4*t));
}

template <class Field>
void save(const fs::path& path, const std::string& filename, const Field& u, const std::string& suffix="")
{
    auto mesh = u.mesh();
    auto level_ = samurai::make_field<std::size_t, 1>("level", mesh);

    if (!fs::exists(path))
    {
        fs::create_directory(path);
    }

    samurai::for_each_cell(mesh, [&](auto &cell)
    {
        level_[cell] = cell.level;
    });

    samurai::save(path, fmt::format("{}{}", filename, suffix), mesh, u, level_);
}

int main(int argc, char *argv[])
{
    constexpr size_t dim = 1;
    using Config = samurai::MRConfig<dim>;

    //--------------------//
    // Program parameters //
    //--------------------//

    // Simulation parameters
    double left_box = -20, right_box = 20;
    double Tf = 1.;
    double dt = Tf / 100;

    // Multiresolution parameters
    std::size_t min_level = 0, max_level = 5;
    double mr_epsilon = 1e-7; // Threshold used by multiresolution
    double mr_regularity = 1.; // Regularity guess for multiresolution

    // Output parameters
    fs::path path = fs::current_path();
    std::string filename = "FV_heat_1d";
    std::size_t nfiles = 50;

    CLI::App app{"Finite volume example for the heat equation in 1d using backward Euler multiresolution"};
    app.add_option("--left", left_box, "The left border of the box")->capture_default_str()->group("Simulation parameters");
    app.add_option("--right", right_box, "The right border of the box")->capture_default_str()->group("Simulation parameters");
    app.add_option("--Tf", Tf, "Final time")->capture_default_str()->group("Simulation parameters");
    app.add_option("--dt", dt, "Time step")->capture_default_str()->group("Simulation parameters");
    app.add_option("--min-level", min_level, "Minimum level of the multiresolution")->capture_default_str()->group("Multiresolution");
    app.add_option("--max-level", max_level, "Maximum level of the multiresolution")->capture_default_str()->group("Multiresolution");
    app.add_option("--mr-eps", mr_epsilon, "The epsilon used by the multiresolution to adapt the mesh")->capture_default_str()->group("Multiresolution");
    app.add_option("--mr-reg", mr_regularity, "The regularity criteria used by the multiresolution to adapt the mesh")->capture_default_str()->group("Multiresolution");
    app.add_option("--path", path, "Output path")->capture_default_str()->group("Ouput");
    app.add_option("--filename", filename, "File name prefix")->capture_default_str()->group("Ouput");
    app.add_option("--nfiles", nfiles,  "Number of output files")->capture_default_str()->group("Ouput");
    app.allow_extras();
    CLI11_PARSE(app, argc, argv);

    //------------------//
    // Petsc initialize //
    //------------------//

    PetscInitialize(&argc, &argv, 0, nullptr);

    PetscMPIInt size;
    PetscCallMPI(MPI_Comm_size(PETSC_COMM_WORLD, &size));
    PetscCheck(size == 1, PETSC_COMM_WORLD, PETSC_ERR_WRONG_MPI_SIZE, "This is a uniprocessor example only!");
    PetscOptionsSetValue(NULL, "-options_left", "off"); // If on, Petsc will issue warnings saying that the options managed by CLI are unused

    //--------------------//
    // Problem definition //
    //--------------------//

    samurai::Box<double, dim> box({left_box}, {right_box});
    samurai::MRMesh<Config> mesh{box, min_level, max_level};

    auto u = samurai::make_field<double, 1>("u", mesh);

    double t0 = 1e-2; // in this particular case, the exact solution is not defined for t=0
    samurai::for_each_cell(mesh, [&](auto &cell) {
        u[cell] = exact_solution(cell.center(0), t0);
    });

    auto unp1 = samurai::make_field<double, 1>("unp1", mesh);

       u.set_neumann([](auto&){ return 0.; }).everywhere();
    unp1.set_neumann([](auto&){ return 0.; }).everywhere();

    //--------------------//
    //   Time iteration   //
    //--------------------//

    auto MRadaptation = samurai::make_MRAdapt(u);
    MRadaptation(mr_epsilon, mr_regularity);

    save(path, filename, u, "_init");
    double dt_save = Tf/static_cast<double>(nfiles);
    std::size_t nsave = 1, nt = 0;

    double t = t0;
    while (t != Tf)
    {
        // Move to next timestep
        t += dt;
        if (t > Tf)
        {
            dt += Tf - t;
            t = Tf;
        }
        std::cout << fmt::format("iteration {}: t = {}, dt = {}", nt++, t, dt) << std::endl;

        // Mesh adaptation
        MRadaptation(mr_epsilon, mr_regularity);
        samurai::update_ghost_mr(u);
        unp1.resize();

        // Solve the linear equation:
        //            unp1 - dt*Lap(unp1) = u
        auto diff_unp1  = samurai::petsc::make_diffusion_FV(unp1);            // diff_unp1  = -Lap(unp1)
        auto back_euler = samurai::petsc::make_backward_euler(diff_unp1, dt); // back_euler = [Id - dt*Lap](unp1)
        samurai::petsc::solve(back_euler, u); // solves the linear equation   [Id - dt*Lap](unp1) = u

        // u <-- unp1
        std::swap(u.array(), unp1.array());

        // Save the result
        if (t >= static_cast<double>(nsave+1)*dt_save || t == Tf)
        {
            std::string suffix = (nfiles!=1)? fmt::format("_ite_{}", nsave++): "";
            save(path, filename, u, suffix);
        }

        // Compute the error at instant t with respect to the exact solution
        double error = decltype(diff_unp1)::L2Error(u, [&](auto& coord)
        {
            double x = coord[0];
            return exact_solution(x, t);
        });
        std::cout.precision(2);
        std::cout << "L2-error: " << std::scientific << error << std::endl;
    }

    std::cout << std::endl;
    std::cout << "Run the following command to view the results:" << std::endl;
    std::cout << "<<path to samurai>>/python/read_mesh.py FV_heat_1d_ite_ --field u level --start 1 --end " << nsave << std::endl;
    PetscFinalize();

    return 0;
}