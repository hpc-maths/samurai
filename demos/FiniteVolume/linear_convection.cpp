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
    auto& app = samurai::initialize("Finite volume example for the linear convection equation", argc, argv);

    static constexpr std::size_t dim = 2;
    using Config                     = samurai::MRConfig<dim, 3>;
    using Box                        = samurai::Box<double, dim>;
    using point_t                    = typename Box::point_t;

    std::cout << "------------------------- Linear convection -------------------------" << std::endl;

    //--------------------//
    // Program parameters //
    //--------------------//

    // Simulation parameters
    double left_box      = -1;
    double right_box     = 1;
    bool implicit_scheme = false;

    // Time integration
    double Tf  = 3;
    double dt  = 0;
    double cfl = 0.95;
    double t   = 0.;
    std::string restart_file;

    // Multiresolution parameters
    std::size_t min_level = 1;
    std::size_t max_level = dim == 1 ? 6 : 4;

    // Output parameters
    fs::path path        = fs::current_path();
    std::string filename = "linear_convection_" + std::to_string(dim) + "D";
    std::size_t nfiles   = 0;

    app.add_option("--left", left_box, "The left border of the box")->capture_default_str()->group("Simulation parameters");
    app.add_option("--right", right_box, "The right border of the box")->capture_default_str()->group("Simulation parameters");
    app.add_option("--Ti", t, "Initial time")->capture_default_str()->group("Simulation parameters");
    app.add_option("--Tf", Tf, "Final time")->capture_default_str()->group("Simulation parameters");
    app.add_option("--restart-file", restart_file, "Restart file")->capture_default_str()->group("Simulation parameters");
    app.add_flag("--implicit", implicit_scheme, "Implicit scheme instead of explicit")->group("Simulation parameters");
    app.add_option("--dt", dt, "Time step")->capture_default_str()->group("Simulation parameters");
    app.add_option("--cfl", cfl, "The CFL")->capture_default_str()->group("Simulation parameters");
    app.add_option("--path", path, "Output path")->capture_default_str()->group("Output");
    app.add_option("--filename", filename, "File name prefix")->capture_default_str()->group("Output");
    app.add_option("--nfiles", nfiles, "Number of output files")->capture_default_str()->group("Output");

    app.allow_extras();
    SAMURAI_PARSE(argc, argv);

    //--------------------//
    // Problem definition //
    //--------------------//

    point_t box_corner1, box_corner2;
    box_corner1.fill(left_box);
    box_corner2.fill(right_box);
    Box box(box_corner1, box_corner2);
    std::array<bool, dim> periodic;
    periodic.fill(true);
    samurai::MRMesh<Config> mesh;
    auto u = samurai::make_scalar_field<double>("u", mesh);

    if (restart_file.empty())
    {
        auto config = samurai::mesh_config<dim>().min_level(min_level).max_level(max_level).periodic(periodic);
        mesh        = {config, box};
        // Initial solution
        u = samurai::make_scalar_field<double>("u",
                                               mesh,
                                               [](const auto& coords)
                                               {
                                                   if constexpr (dim == 1)
                                                   {
                                                       const auto& x = coords(0);
                                                       return (x >= -0.8 && x <= -0.3) ? 1. : 0.;
                                                   }
                                                   else
                                                   {
                                                       const auto& x = coords(0);
                                                       const auto& y = coords(1);
                                                       return (x >= -0.8 && x <= -0.3 && y >= 0.3 && y <= 0.8) ? 1. : 0.;
                                                   }
                                               });
    }
    else
    {
        samurai::load(restart_file, mesh, u);
    }

    auto unp1 = samurai::make_scalar_field<double>("unp1", mesh);
    // Intermediary fields for the RK3 scheme
    auto u1 = samurai::make_scalar_field<double>("u1", mesh);
    auto u2 = samurai::make_scalar_field<double>("u2", mesh);

    // Convection operator
    samurai::VelocityVector<dim> velocity;
    velocity.fill(1);
    if constexpr (dim == 2)
    {
        velocity(1) = -1;
    }
    auto conv = samurai::make_convection_weno5<decltype(u)>(velocity);
    auto id   = samurai::make_identity<decltype(u)>();

    //--------------------//
    //   Time iteration   //
    //--------------------//

    if (dt == 0)
    {
        double dx             = mesh.cell_length(mesh.max_level());
        auto a                = xt::abs(velocity);
        double sum_velocities = xt::sum(xt::abs(velocity))();
        dt                    = cfl * dx / sum_velocities;
    }

    auto MRadaptation = samurai::make_MRAdapt(u);
    auto mra_config   = samurai::mra_config();
    MRadaptation(mra_config);

    double dt_save    = nfiles == 0 ? dt : Tf / static_cast<double>(nfiles);
    std::size_t nsave = 0, nt = 0;
    if (nfiles != 1)
    {
        std::string suffix = (nfiles != 1) ? fmt::format("_ite_{}", nsave++) : "";
        save(path, filename, u, suffix);
    }
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
        MRadaptation(mra_config);
        unp1.resize();
        u1.resize();
        u2.resize();

        if (implicit_scheme) // Backward Euler
        {
            // Newton solver for non-linear equation  [Id + dt*Conv](unp1) = u
            auto solver = samurai::petsc::make_solver(id + dt * conv);

            // Configure the PETSc solver
            SNESSetTolerances(solver.Snes(), PETSC_CURRENT /* abstol */, 1e-5 /* rtol */, PETSC_CURRENT /* stol */, 500 /* maxit */, PETSC_CURRENT);
            KSP ksp;
            PC pc;
            SNESGetKSP(solver.Snes(), &ksp);
            KSPSetType(ksp, KSPPREONLY);
            KSPGetPC(ksp, &pc);
            PCSetType(pc, PCLU); // Set the PC type to LU

            // Solve the non-linear equation   [Id + dt*Conv](unp1) = u
            solver.set_unknown(unp1);
            unp1 = u; // set initial guess for the Newton solver
            solver.solve(u);
        }
        else
        {
            // Forward Euler
            // unp1 = u - dt * conv(u);

            // TVD-RK3 (SSPRK3)
            u1   = u - dt * conv(u);
            u2   = 3. / 4 * u + 1. / 4 * (u1 - dt * conv(u1));
            unp1 = 1. / 3 * u + 2. / 3 * (u2 - dt * conv(u2));
        }

        // u <-- unp1
        samurai::swap(u, unp1);

        // Save the result
        if (nfiles == 0 || t >= static_cast<double>(nsave) * dt_save || t == Tf)
        {
            if (nfiles != 1)
            {
                std::string suffix = (nfiles != 1) ? fmt::format("_ite_{}", nsave++) : "";
                save(path, filename, u, suffix);
            }
            else
            {
                save(path, filename, u);
            }
        }

        std::cout << std::endl;
    }

    if constexpr (dim == 1)
    {
        std::cout << std::endl;
        std::cout << "Run the following command to view the results:" << std::endl;
        std::cout << "python <<path to samurai>>/python/read_mesh.py " << filename << "_ite_ --field u level --start 0 --end " << nsave
                  << std::endl;
    }

    samurai::finalize();
    return 0;
}
