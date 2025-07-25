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
    auto& app = samurai::initialize("Finite volume example for the heat equation", argc, argv);

    static constexpr std::size_t dim = 2;
    using Config                     = samurai::MRConfig<dim>;
    using Box                        = samurai::Box<double, dim>;
    using point_t                    = typename Box::point_t;

    std::cout << "------------------------- Heat -------------------------" << std::endl;

    //--------------------//
    // Program parameters //
    //--------------------//

    // Simulation parameters
    double left_box  = -4;
    double right_box = 4;

    // Time integration
    double Tf            = 1.;
    double dt            = Tf / 100;
    bool explicit_scheme = false;
    double cfl           = 0.95;
    double t             = 0.;
    std::string restart_file;

    // Multiresolution parameters
    std::size_t min_level = 3;
    std::size_t max_level = 6;

    // Output parameters
    fs::path path              = fs::current_path();
    std::string filename       = "heat_heterog_" + std::to_string(dim) + "D";
    bool save_final_state_only = false;

    app.add_flag("--explicit", explicit_scheme, "Explicit scheme instead of implicit")->group("Simulation parameters");
    app.add_option("--Ti", t, "Initial time")->capture_default_str()->group("Simulation parameters");
    app.add_option("--Tf", Tf, "Final time")->capture_default_str()->group("Simulation parameters");
    app.add_option("--restart-file", restart_file, "Restart file")->capture_default_str()->group("Simulation parameters");
    app.add_option("--dt", dt, "Time step")->capture_default_str()->group("Simulation parameters");
    app.add_option("--cfl", cfl, "The CFL")->capture_default_str()->group("Simulation parameters");
    app.add_option("--min-level", min_level, "Minimum level of the multiresolution")->capture_default_str()->group("Multiresolution");
    app.add_option("--max-level", max_level, "Maximum level of the multiresolution")->capture_default_str()->group("Multiresolution");
    app.add_option("--path", path, "Output path")->capture_default_str()->group("Output");
    app.add_option("--filename", filename, "File name prefix")->capture_default_str()->group("Output");
    app.add_flag("--save-final-state-only", save_final_state_only, "Save final state only")->group("Output");
    SAMURAI_PARSE(argc, argv);

    PetscMPIInt size;
    PetscCallMPI(MPI_Comm_size(PETSC_COMM_WORLD, &size));
    PetscCheck(size == 1, PETSC_COMM_WORLD, PETSC_ERR_WRONG_MPI_SIZE, "This is a uniprocessor example only!");

    //--------------------//
    // Problem definition //
    //--------------------//

    point_t box_corner1, box_corner2;
    box_corner1.fill(left_box);
    box_corner2.fill(right_box);
    Box box(box_corner1, box_corner2);
    samurai::MRMesh<Config> mesh;

    auto u    = samurai::make_scalar_field<double>("u", mesh);
    auto unp1 = samurai::make_scalar_field<double>("unp1", mesh);

    if (restart_file.empty())
    {
        mesh = {box, min_level, max_level};
        u.resize();
        // Initial solution: crenel
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
    else
    {
        samurai::load(restart_file, mesh, u);
    }

    samurai::make_bc<samurai::Neumann<1>>(u, 0.);
    samurai::make_bc<samurai::Neumann<1>>(unp1, 0.);

    auto K = samurai::make_scalar_field<samurai::DiffCoeff<dim>>("K", mesh);

    auto set_K_values = [&]()
    {
        samurai::for_each_cell(mesh[decltype(mesh)::mesh_id_t::reference],
                               [&](auto& cell)
                               {
                                   double x = cell.center(0);
                                   double y = cell.center(1);
                                   // Domain divided into 4 quadrants
                                   if ((x < 0 && y > 0) || (x > 0 && y < 0))
                                   {
                                       K[cell] = {0.1, 4};
                                   }
                                   else
                                   {
                                       K[cell] = {4, 0.1};
                                   }
                               });
    };

    auto diff = samurai::make_diffusion_order2<decltype(u)>(K);
    auto id   = samurai::make_identity<decltype(u)>();

    //--------------------//
    //   Time iteration   //
    //--------------------//

    if (explicit_scheme)
    {
        double diff_coeff = 4;
        double dx         = mesh.cell_length(max_level);
        dt                = cfl * (dx * dx) / (pow(2, dim) * diff_coeff);
    }

    auto MRadaptation = samurai::make_MRAdapt(u);
    auto mra_config   = samurai::mra_config();
    MRadaptation(mra_config);

    std::size_t nsave = 0, nt = 0;
    if (!save_final_state_only)
    {
        save(path, filename, u, fmt::format("_ite_{}", nsave++));
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
        samurai::update_ghost_mr(u);
        unp1.resize();
        K.resize();
        set_K_values();

        if (explicit_scheme)
        {
            unp1 = u - dt * diff(u);
        }
        else
        {
            auto back_euler = id + dt * diff;
            samurai::petsc::solve(back_euler, unp1, u); // solves the linear equation   [Id + dt*Diff](unp1) = u
        }

        // u <-- unp1
        std::swap(u.array(), unp1.array());

        // Save the result
        if (!save_final_state_only)
        {
            save(path, filename, u, fmt::format("_ite_{}", nsave++));
        }

        std::cout << std::endl;
    }

    if (save_final_state_only)
    {
        save(path, filename, u);
    }

    samurai::finalize();
    return 0;
}
