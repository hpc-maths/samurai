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
    auto& app = samurai::initialize("Finite volume example for the Nagumo equation", argc, argv);

    static constexpr std::size_t dim    = 1;
    static constexpr std::size_t n_comp = 1;
    using Config                        = samurai::MRConfig<dim>;
    using Box                           = samurai::Box<double, dim>;
    using point_t                       = typename Box::point_t;

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
    double dt  = 0;
    double cfl = 0.95;
    double t   = 0.;
    std::string restart_file;

    // Multiresolution parameters
    std::size_t min_level = 4;
    std::size_t max_level = 8;

    // Output parameters
    fs::path path              = fs::current_path();
    std::string filename       = "nagumo";
    bool save_final_state_only = false;

    app.add_option("--left", left_box, "The left border of the box")->capture_default_str()->group("Simulation parameters");
    app.add_option("--right", right_box, "The right border of the box")->capture_default_str()->group("Simulation parameters");
    app.add_option("--D", D, "Diffusion coefficient")->capture_default_str()->group("Simulation parameters");
    app.add_option("--k", k, "Parameter of the reaction operator")->capture_default_str()->group("Simulation parameters");
    app.add_option("--Ti", t, "Initial time")->capture_default_str()->group("Simulation parameters");
    app.add_option("--Tf", Tf, "Final time")->capture_default_str()->group("Simulation parameters");
    app.add_option("--restart-file", restart_file, "Restart file")->capture_default_str()->group("Simulation parameters");
    app.add_option("--dt", dt, "Time step")->capture_default_str()->group("Simulation parameters");
    app.add_option("--cfl", cfl, "The CFL")->capture_default_str()->group("Simulation parameters");
    app.add_flag("--explicit-reaction", explicit_reaction, "Explicit the reaction term")->capture_default_str()->group("Simulation parameters");
    app.add_flag("--explicit-diffusion", explicit_diffusion, "Explicit the diffusion term")->capture_default_str()->group("Simulation parameters");
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
    auto config = samurai::mesh_config<dim>().min_level(min_level).max_level(max_level);
    samurai::MRMesh<Config> mesh{config, box};

    auto u = samurai::make_vector_field<double, n_comp>("u", mesh);

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

    if (restart_file.empty())
    {
        mesh = {box, min_level, max_level};
        u.resize();
        // Initial solution
        samurai::for_each_cell(mesh,
                               [&](auto& cell)
                               {
                                   u[cell] = exact_solution(cell.center(0), 0);
                               });
    }
    else
    {
        samurai::load(restart_file, mesh, u);
    }

    auto unp1 = samurai::make_vector_field<double, n_comp>("unp1", mesh);

    samurai::make_bc<samurai::Neumann<1>>(u);
    samurai::make_bc<samurai::Neumann<1>>(unp1);

    auto diff = samurai::make_diffusion_order2<decltype(u)>(D);
    auto id   = samurai::make_identity<decltype(u)>();

    // Reaction operator
    using cfg  = samurai::LocalCellSchemeConfig<samurai::SchemeType::NonLinear, decltype(u), decltype(u)>;
    auto react = samurai::make_cell_based_scheme<cfg>();
    react.set_name("Reaction");
    react.set_scheme_function(
        [&](samurai::SchemeValue<cfg>& value, const auto& cell, const auto& field)
        {
            auto v = field[cell];
            value  = k * v * v * (1 - v);
        });
    react.set_jacobian_function(
        [&](samurai::JacobianMatrix<cfg>& jac, const auto& cell, const auto& field)
        {
            auto v = field[cell];
            jac    = k * (2 * v * (1 - v) - v * v);
        });

    //--------------------//
    //   Time iteration   //
    //--------------------//

    if (dt == 0)
    {
        dt = Tf / 100;
        if (explicit_diffusion)
        {
            double dx = mesh.cell_length(mesh.max_level());
            dt        = cfl * (dx * dx) / (pow(2, dim) * D);
        }
    }

    auto MRadaptation = samurai::make_MRAdapt(u);
    auto mra_config   = samurai::mra_config().epsilon(1e-5);
    MRadaptation(mra_config);

    std::size_t nsave = 0, nt = 0;
    if (!save_final_state_only)
    {
        save(path, filename, u, fmt::format("_ite_{}", nsave++));
    }

    auto rhs = samurai::make_vector_field<double, n_comp>("rhs", mesh);

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
        rhs.resize();

        if (explicit_diffusion && explicit_reaction)
        {
            unp1 = u + dt * react(u) - dt * diff(u);
        }
        else if (!explicit_diffusion && explicit_reaction)
        {
            // u_np1 + dt*diff(u_np1) = u + dt*react(u)
            auto implicit_operator = id + dt * diff;
            rhs                    = u + dt * react(u);
            // Solve the linear equation   [Id + dt*Diff](unp1) = rhs
            samurai::petsc::solve(implicit_operator, unp1, rhs);
        }
        else if (explicit_diffusion && !explicit_reaction)
        {
            // u_np1 - dt*react(u_np1) = u - dt*diff(u)
            auto implicit_operator = id - dt * react;
            rhs                    = u - dt * diff(u);
            // Set initial guess for the Newton algorithm
            unp1 = u;
            // Solve the non-linear equation   [Id - dt*React](unp1) = u - dt*Diff(u)
            // Here, small independent local Newton methods are used.
            samurai::petsc::solve(implicit_operator, unp1, rhs);
        }
        else
        {
            // u_np1 + dt*diff(u_np1) - dt*react(u_np1) = u
            auto implicit_operator = id + dt * diff - dt * react;
            // Set initial guess for the Newton algorithm
            unp1 = u;
            // Solve the non-linear equation   [Id + dt*Diff - dt*React](unp1) = u
            samurai::petsc::solve(implicit_operator, unp1, u);
        }

        // u <-- unp1
        samurai::swap(u, unp1);

        // Compute error
        double error = samurai::L2_error(u,
                                         [&](const auto& coord)
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

    samurai::finalize();
    return 0;
}
