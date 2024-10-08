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
    static constexpr std::size_t dim = 1;
    using Config                     = samurai::MRConfig<dim>;
    using Box                        = samurai::Box<double, dim>;
    using point_t                    = typename Box::point_t;

    std::cout << "------------------------- Nagumo -------------------------" << std::endl;

    /**
     * Nagumo, or Fisher-KPP equation:
     *
     * du/dt -D*Lap(u) = k u^2 (1-u)
     */

    //--------------------//
    // Program parameters //
    //--------------------//

    // Simulation parameters
    double left_box  = -10;
    double right_box = 10;

    double D = 1;
    double k = 10;

    bool explicit_diffusion = false;
    bool explicit_reaction  = false;

    // Time integration
    double Tf  = 1.;
    double dt  = Tf / 100;
    double cfl = 0.95;

    // Multiresolution parameters
    std::size_t min_level = 0;
    std::size_t max_level = 4;
    double mr_epsilon     = 1e-5; // Threshold used by multiresolution
    double mr_regularity  = 1.;   // Regularity guess for multiresolution

    // Output parameters
    fs::path path              = fs::current_path();
    std::string filename       = "nagumo";
    bool save_final_state_only = false;

    CLI::App app{"Finite volume example for the Nagumo equation"};
    app.add_option("--left", left_box, "The left border of the box")->capture_default_str()->group("Simulation parameters");
    app.add_option("--right", right_box, "The right border of the box")->capture_default_str()->group("Simulation parameters");
    app.add_option("--D", D, "Diffusion coefficient")->capture_default_str()->group("Simulation parameters");
    app.add_option("--k", k, "Parameter of the reaction operator")->capture_default_str()->group("Simulation parameters");
    app.add_option("--Tf", Tf, "Final time")->capture_default_str()->group("Simulation parameters");
    app.add_option("--dt", dt, "Time step")->capture_default_str()->group("Simulation parameters");
    app.add_option("--cfl", cfl, "The CFL")->capture_default_str()->group("Simulation parameters");
    app.add_flag("--explicit-reaction", explicit_reaction, "Explicit the reaction term")->capture_default_str()->group("Simulation parameters");
    app.add_flag("--explicit-diffusion", explicit_diffusion, "Explicit the diffusion term")->capture_default_str()->group("Simulation parameters");
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
    app.add_flag("--save-final-state-only", save_final_state_only, "Save final state only")->group("Output");
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

    double z0 = left_box / 5;    // wave initial position
    double c  = sqrt(k * D / 2); // wave velocity

    auto beta = [&](double z)
    {
        double e = exp(-sqrt(k / (2 * D)) * (z - z0));
        return e / (1 + e);
    };

    auto exact_solution = [&](double x, double t)
    {
        return beta(x - c * t);
    };

    // Initial solution
    samurai::for_each_cell(mesh,
                           [&](auto& cell)
                           {
                               u[cell] = exact_solution(cell.center(0), 0);
                           });

    auto unp1 = samurai::make_field<1>("unp1", mesh);

    samurai::make_bc<samurai::Neumann<1>>(u, 0.);
    samurai::make_bc<samurai::Neumann<1>>(unp1, 0.);

    auto diff = samurai::make_diffusion<decltype(u)>(D);
    auto id   = samurai::make_identity<decltype(u)>();

    // Reaction operator
    using cfg  = samurai::LocalCellSchemeConfig<samurai::SchemeType::NonLinear, 1, decltype(u)>;
    auto react = samurai::make_cell_based_scheme<cfg>();
    react.set_name("Reaction");
    react.set_scheme_function(
        [&](auto& cell, auto& field) //-> samurai::SchemeValue<cfg>
        {
            auto v = field[cell];
            return k * v * v * (1 - v);
        });
    react.set_jacobian_function(
        [&](auto& cell, auto& field)
        {
            auto v = field[cell];
            return k * (2 * v * (1 - v) - v * v);
        });

    //--------------------//
    //   Time iteration   //
    //--------------------//

    auto MRadaptation = samurai::make_MRAdapt(u);
    MRadaptation(mr_epsilon, mr_regularity);

    std::size_t nsave = 0, nt = 0;
    if (!save_final_state_only)
    {
        save(path, filename, u, fmt::format("_ite_{}", nsave++));
    }

    // auto rhs = samurai::make_field<1>("rhs", mesh);

    ///////////////////////////////////////////
    double gamma = 1.0 - 0.5 * std::sqrt(2.0);

    auto f_t = [&](double)
    {
        return -diff + react;
    };

    auto f = [&](double t, auto& u)
    {
        samurai::update_ghost_mr(u);
        return f_t(t)(u);
    };

    std::array<decltype(u), 2> k_stages;
    std::array<decltype(u), 2> u_stages;

    // auto& u1 = u_stages[0];
    // auto& u2 = u_stages[1];

    auto u1 = samurai::make_field<1>("u1", mesh);
    auto u2 = samurai::make_field<1>("u2", mesh);
    samurai::make_bc<samurai::Neumann>(u1, 0.);
    samurai::make_bc<samurai::Neumann>(u2, 0.);

    // auto& k1 = k_stages[0];
    // auto& k2 = k_stages[1];

    auto k1 = samurai::make_field<1>("k1", mesh);
    auto k2 = samurai::make_field<1>("k2", mesh);
    samurai::make_bc<samurai::Neumann>(k1, 0.);
    samurai::make_bc<samurai::Neumann>(k2, 0.);

    double t = 0;
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
        // rhs.resize();
        //  for (auto& ui : u_stages)
        //  {
        //      ui.resize();
        //  }
        //  for (auto& ki : k_stages)
        //  {
        //      ki.resize();
        //  }
        k1.resize();
        k2.resize();

        // u1 - dt*gamma*f(t+gamma*dt, u1) = un
        u1                = u;
        auto implicit_op1 = id - dt * gamma * f_t(t + gamma * dt);
        samurai::petsc::solve(implicit_op1, u1, u);

        k1 = f(t + gamma * dt, u1);

        // u2 - dt*gamma*f(t+gamma*dt, u1) = un + dt*(1-2*gamma)*k1
        u2                = u;
        auto implicit_op2 = id - dt * gamma * f_t(t + (1 - gamma) * dt);
        // rhs               = u + dt * (1 - 2 * gamma) * k1;
        samurai::petsc::solve(implicit_op2, u2, u + dt * (1 - 2 * gamma) * k1);

        k2 = f(t + (1 - gamma) * dt, u2);

        // unp1 = un + dt/2*k1 + dt/2*k2
        unp1 = u + 0.5 * dt * k1 + 0.5 * dt * k2;

        // u <-- unp1
        std::swap(u.array(), unp1.array());

        // Compute error
        double error = samurai::L2_error(u,
                                         [&](auto& coord)
                                         {
                                             return exact_solution(coord(0), t);
                                         });
        std::cout.precision(2);
        std::cout << ", L2-error: " << std::scientific << error;

        // Save the result
        if (!save_final_state_only)
        {
            save(path, filename, u, fmt::format("_ite_{}", nsave++));
        }

        std::cout << std::endl;
    }

    if (!save_final_state_only && dim == 1)
    {
        std::cout << std::endl;
        std::cout << "Run the following command to view the results:" << std::endl;
        std::cout << "python <<path to samurai>>/python/read_mesh.py " << filename << "_ite_ --field u level --start 1 --end " << nsave
                  << std::endl;
    }

    if (save_final_state_only)
    {
        save(path, filename, u);
    }

    PetscFinalize();

    return 0;
}
