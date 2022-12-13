// Copyright 2021 SAMURAI TEAM. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.
#include "CLI/CLI.hpp"

#include <samurai/mr/mesh.hpp>
#include <samurai/mr/adapt.hpp>
#include <samurai/hdf5.hpp>
#include <samurai/petsc/petsc_diffusion_FV_star_stencil.hpp>
#include <samurai/petsc/petsc_backward_euler.hpp>
#include <samurai/petsc/petsc_solver.hpp>

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
    using Mesh = samurai::MRMesh<Config>;
    using Field = samurai::Field<Mesh, double, 1>;
    using DiscreteDiffusion = samurai::petsc::PetscDiffusionFV_StarStencil<Field>;
    using BackwardEuler = samurai::petsc::PetscBackwardEuler<DiscreteDiffusion>;

    //--------------------//
    // Program parameters //
    //--------------------//

    // Simulation parameters
    double left_box = -2, right_box = 2;
    double Tf = 1.;
    double dt = Tf / 10;

    // Multiresolution parameters
    std::size_t min_level = 4, max_level = 7;
    double mr_epsilon = 2.e-4; // Threshold used by multiresolution
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

    double dt_save = Tf/static_cast<double>(nfiles);
    double t = 1e-2;

    auto u = samurai::make_field<double, 1>("u", mesh);
    samurai::for_each_cell(mesh, [&](auto &cell) {
        u[cell] = exact_solution(cell.center(0), t);
    });
    
    auto unp1 = samurai::make_field<double, 1>("unp1", mesh);

    bool is_dirichlet = false;
    if (is_dirichlet)
    {
           u.set_dirichlet([](auto&){ return 0.; }).everywhere();
        unp1.set_dirichlet([](auto&){ return 0.; }).everywhere();
    }
    else
    {
           u.set_neumann([](auto&){ return 0.; }).everywhere();
        unp1.set_neumann([](auto&){ return 0.; }).everywhere();
    }

    auto update_bc = [&](auto& field, std::size_t)
    {
        samurai::for_each_stencil_center_and_outside_ghost(field.mesh(), DiscreteDiffusion::stencil(),
        [&] (const auto& cells, const auto& towards_ghost)
        {
            const auto& cell  = cells[0];
            const auto& ghost = cells[1];
            const double& h = cell.length;
            auto boundary_point = cell.face_center(towards_ghost);
            auto bc = find(field.boundary_conditions(), boundary_point);

            if (bc.is_dirichlet())
            {
                auto dirichlet_value = bc.get_value(boundary_point);
                field[ghost] = 2 * dirichlet_value - field[cell];
            }
            else
            {
                auto neumann_value = bc.get_value(boundary_point);
                field[ghost] = h * neumann_value + field[cell];
            }
        });
    };

    //--------------------//
    //   Time iteration   //
    //--------------------//

    auto MRadaptation = samurai::make_MRAdapt(u, update_bc);
    MRadaptation(mr_epsilon, mr_regularity);
    save(path, filename, u, "_init");

    std::size_t nsave = 1, nt = 0;

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

        samurai::update_ghost_mr(u, update_bc);
        unp1.resize();
        DiscreteDiffusion diff(unp1.mesh(), unp1.boundary_conditions());
        samurai::petsc::solve(BackwardEuler(diff, dt), u, unp1);

        std::swap(u.array(), unp1.array());

        if ( t >= static_cast<double>(nsave+1)*dt_save || t == Tf)
        {
            std::string suffix = (nfiles!=1)? fmt::format("_ite_{}", nsave++): "";
            save(path, filename, u, suffix);
        }

        if (!is_dirichlet)
        {
            double error = DiscreteDiffusion::L2Error(u, 
                            [&](auto& coord) 
                            {
                                double x = coord[0];
                                return exact_solution(x, t);
                            });
            std::cout.precision(2);
            std::cout << "L2-error: " << std::scientific << error << std::endl;
        }
    }


    PetscFinalize();

    return 0;
}