// Copyright 2021 SAMURAI TEAM. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.
#include "CLI/CLI.hpp"

#include <samurai/hdf5.hpp>
#include <samurai/mr/adapt.hpp>
#include <samurai/mr/mesh.hpp>
#include <samurai/petsc.hpp>

#include <filesystem>
namespace fs = std::filesystem;

template <std::size_t dim>
double exact_solution(xt::xtensor_fixed<double, xt::xshape<dim>> coords, double t, double diff_coeff)
{
    assert(t > 0 && "t must be > 0");
    double result = 1;
    for (std::size_t d = 0; d < dim; ++d)
    {
        result *= 1 / (2 * sqrt(M_PI * diff_coeff * t)) * exp(-coords(d) * coords(d) / (4 * diff_coeff * t));
    }
    return result;
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
                           [&](auto& cell)
                           {
                               level_[cell] = cell.level;
                           });

    samurai::save(path, fmt::format("{}{}", filename, suffix), mesh, u, level_);
}

int main(int argc, char* argv[])
{
    static constexpr std::size_t dim = 2;
    using Config                     = samurai::MRConfig<dim>;
    using Box                        = samurai::Box<double, dim>;
    using point_t                    = typename Box::point_t;

    std::cout << "------------------------- Heat -------------------------" << std::endl;

    //--------------------//
    // Program parameters //
    //--------------------//

    // Simulation parameters
    double left_box  = -1;
    double right_box = 1;
    if constexpr (dim == 1)
    {
        left_box  = -20;
        right_box = 20;
    }
    else if constexpr (dim == 2)
    {
        left_box  = -4;
        right_box = 4;
    }
    std::string init_sol = "dirac";
    double diff_coeff    = 1;

    // Time integration
    double Tf     = 1.;
    double dt     = Tf / 100;
    bool implicit = false;
    double cfl    = 0.95;

    // Multiresolution parameters
    std::size_t min_level = 0;
    std::size_t max_level = dim == 1 ? 5 : 3;
    double mr_epsilon     = 1e-4; // Threshold used by multiresolution
    double mr_regularity  = 1.;   // Regularity guess for multiresolution

    // Output parameters
    fs::path path        = fs::current_path();
    std::string filename = "heat_" + std::to_string(dim) + "D";
    std::size_t nfiles   = 50;

    CLI::App app{"Finite volume example for the heat equation in 1d"};
    app.add_option("--left", left_box, "The left border of the box")->capture_default_str()->group("Simulation parameters");
    app.add_option("--right", right_box, "The right border of the box")->capture_default_str()->group("Simulation parameters");
    app.add_option("--init-sol", init_sol, "Initial solution: dirac/crenel")->capture_default_str()->group("Simulation parameters");
    app.add_option("--diff-coeff", diff_coeff, "Diffusion coefficient")->capture_default_str()->group("Simulation parameters");
    app.add_flag("--implicit", implicit, "Implicit scheme instead of explicit")->group("Simulation parameters");
    app.add_option("--Tf", Tf, "Final time")->capture_default_str()->group("Simulation parameters");
    app.add_option("--dt", dt, "Time step")->capture_default_str()->group("Simulation parameters");
    app.add_option("--cfl", cfl, "The CFL")->capture_default_str()->group("Simulation parameters");
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
    app.add_option("--path", path, "Output path")->capture_default_str()->group("Ouput");
    app.add_option("--filename", filename, "File name prefix")->capture_default_str()->group("Ouput");
    app.add_option("--nfiles", nfiles, "Number of output files")->capture_default_str()->group("Ouput");
    app.allow_extras();
    CLI11_PARSE(app, argc, argv);

    //------------------//
    // Petsc initialize //
    //------------------//

    PetscInitialize(&argc, &argv, 0, nullptr);

    PetscMPIInt size;
    PetscCallMPI(MPI_Comm_size(PETSC_COMM_WORLD, &size));
    PetscCheck(size == 1, PETSC_COMM_WORLD, PETSC_ERR_WRONG_MPI_SIZE, "This is a uniprocessor example only!");
    PetscOptionsSetValue(NULL, "-options_left", "off"); // If on, Petsc will issue warnings saying that
                                                        // the options managed by CLI are unused

    //--------------------//
    // Problem definition //
    //--------------------//

    point_t box_corner1, box_corner2;
    box_corner1.fill(left_box);
    box_corner2.fill(right_box);
    Box box(box_corner1, box_corner2);
    samurai::MRMesh<Config> mesh{box, min_level, max_level};

    auto u = samurai::make_field<1>("u", mesh);

    // Initial solution
    double t0 = 0;
    if (init_sol == "dirac")
    {
        t0 = 1e-2; // in this particular case, the exact solution is not defined for t=0
        samurai::for_each_cell(mesh,
                               [&](auto& cell)
                               {
                                   u[cell] = exact_solution(cell.center(), t0, diff_coeff);
                               });
    }
    else // crenel
    {
        samurai::for_each_cell(mesh,
                               [&](auto& cell)
                               {
                                   bool is_in_crenel = true;
                                   for (std::size_t d = 0; d < dim; ++d)
                                   {
                                       is_in_crenel = is_in_crenel && (abs(cell.center(d)) < right_box / 3);
                                   }
                                   u[cell] = is_in_crenel ? 1 : 0;
                               });
    }

    auto unp1 = samurai::make_field<1>("unp1", mesh);

    samurai::make_bc<samurai::Neumann>(u, 0.);
    samurai::make_bc<samurai::Neumann>(unp1, 0.);

    auto diff = diff_coeff * samurai::make_diffusion<decltype(u)>(); // diffusion = -Laplacian
    auto id   = samurai::make_identity<decltype(u)>();

    //--------------------//
    //   Time iteration   //
    //--------------------//

    if (!implicit)
    {
        double dx = samurai::cell_length(max_level);
        dt        = cfl * (dx * dx) / (pow(2, dim) * diff_coeff);
    }

    auto MRadaptation = samurai::make_MRAdapt(u);
    MRadaptation(mr_epsilon, mr_regularity);

    save(path, filename, u, "_init");
    double dt_save    = Tf / static_cast<double>(nfiles);
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
        std::cout << fmt::format("iteration {}: t = {:.2f}, dt = {}", nt++, t, dt) << std::flush;

        // Mesh adaptation
        MRadaptation(mr_epsilon, mr_regularity);
        samurai::update_ghost_mr(u);
        unp1.resize();

        if (implicit)
        {
            auto back_euler = id + dt * diff;
            samurai::petsc::solve(back_euler, unp1, u); // solves the linear equation   [Id - dt*Lap](unp1) = u
        }
        else
        {
            auto diff_u = diff(u);
            unp1        = u - dt * diff_u;
        }

        // u <-- unp1
        std::swap(u.array(), unp1.array());

        // Save the result
        if (t >= static_cast<double>(nsave + 1) * dt_save || t == Tf)
        {
            std::string suffix = (nfiles != 1) ? fmt::format("_ite_{}", nsave++) : "";
            save(path, filename, u, suffix);
        }

        // Compute the error at instant t with respect to the exact solution
        if (init_sol == "dirac")
        {
            double error = samurai::L2_error(u,
                                             [&](auto& coord)
                                             {
                                                 return exact_solution(coord, t, diff_coeff);
                                             });
            std::cout.precision(2);
            std::cout << ", L2-error: " << std::scientific << error;
        }
        std::cout << std::endl;
    }

    if constexpr (dim == 1)
    {
        std::cout << std::endl;
        std::cout << "Run the following command to view the results:" << std::endl;
        std::cout << "python <<path to samurai>>/python/read_mesh.py " << filename << "_ite_ --field u level --start 1 --end " << nsave
                  << std::endl;
    }
    PetscFinalize();

    return 0;
}
