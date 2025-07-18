// Copyright 2021 SAMURAI TEAM. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.
#include <CLI/CLI.hpp>

#include <samurai/io/hdf5.hpp>
#include <samurai/mr/adapt.hpp>
#include <samurai/mr/mesh.hpp>
#include <samurai/samurai.hpp>
#include <samurai/schemes/fv.hpp>

#include <algorithm>

#include <filesystem>
namespace fs = std::filesystem;

double hat_exact_solution(double x, double t)
{
    const double x0 = -1;
    double x_top;
    double x1;
    double top;

    if (t < 1)
    {
        x_top = t;
        x1    = 1;
        top   = 1;
    }
    else
    {
        x_top = std::sqrt(2 * (1 + t)) - 1;
        x1    = x_top;
        top   = std::sqrt(2 / (1 + t));
    }
    if (x > x0 && x < x_top)
    {
        double a = top / (x_top - x0);
        return a * (x - x0);
    }
    else if (x >= x_top && x < x1)
    {
        double a = -top / (x1 - x_top);
        return a * (x - x_top) + top;
    }
    return 0.;
}

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
}

template <class Field, class Scheme>
void run_simulation(Field& u,
                    Field& unp1,
                    Field& u_max,
                    Field& unp1_max,
                    Scheme& scheme,
                    double cfl,
                    double mr_epsilon,
                    double mr_regularity,
                    const std::string& init_sol,
                    std::size_t nfiles,
                    const fs::path& path,
                    std::string filename,
                    std::size_t& nsave)
{
    auto& mesh = u.mesh();

    std::cout << std::endl << "finer-level-flux enabled: " << scheme.enable_finer_level_flux() << std::endl;

    if (scheme.enable_finer_level_flux())
    {
        if (samurai::args::finer_level_flux == -1)
        {
            filename += "_mlf";
        }
        else
        {
            filename += fmt::format("_lf_{}", samurai::args::finer_level_flux);
        }
    }

    // Initial solution
    if (init_sol == "hat")
    {
        samurai::for_each_cell(mesh,
                               [&](auto& cell)
                               {
                                   u[cell] = hat_exact_solution(cell.center(0), 0);
                               });
        samurai::for_each_cell(u_max.mesh(),
                               [&](auto& cell)
                               {
                                   u_max[cell] = hat_exact_solution(cell.center(0), 0);
                               });
    }
    else // gaussian
    {
        samurai::for_each_cell(mesh,
                               [&](auto& cell)
                               {
                                   double x = cell.center(0);
                                   u[cell]  = 0.1 * exp(-2.0 * x * x);
                               });
        samurai::for_each_cell(u_max.mesh(),
                               [&](auto& cell)
                               {
                                   double x    = cell.center(0);
                                   u_max[cell] = 0.1 * exp(-2.0 * x * x);
                               });
    }

    nsave          = 0;
    std::size_t nt = 0;

    if (nfiles > 1)
    {
        std::string suffix = fmt::format("_ite_{}", nsave);
        save(path, filename, u, suffix);

        samurai::update_ghost_mr(u);
        auto u_recons    = samurai::reconstruction(u);
        std::string file = fmt::format("{}_recons_ite_{}", filename, nsave);
        samurai::save(path, file, u_recons.mesh(), u_recons);
        nsave++;
    }

    //--------------------//
    //   Time iteration   //
    //--------------------//

    double Tf = init_sol == "hat" ? 4. : 16.;
    if (nfiles == 1)
    {
        Tf = 0.1; // for automatic test
    }

    double dx = mesh.cell_length(mesh.max_level());
    double dt = cfl * dx;

    auto MRadaptation = samurai::make_MRAdapt(u);
    MRadaptation(mr_epsilon, mr_regularity);

    double dt_save = Tf / static_cast<double>(nfiles);

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

        unp1     = u - dt * scheme(u);
        unp1_max = u_max - dt * scheme(u_max);

        // u <-- unp1
        std::swap(u.array(), unp1.array());
        std::swap(u_max.array(), unp1_max.array());

        // Reconstruction
        samurai::update_ghost_mr(u);
        auto u_recons = samurai::reconstruction(u);

        // Error
        std::cout << ", L2-error: " << std::scientific;
        std::cout.precision(2);
        if (init_sol == "hat")
        {
            double error = samurai::L2_error(u,
                                             [&](const auto& coord)
                                             {
                                                 return hat_exact_solution(coord[0], t);
                                             });
            std::cout << "[w.r.t. exact, no recons] " << error;

            error = samurai::L2_error(u_recons,
                                      [&](const auto& coord)
                                      {
                                          return hat_exact_solution(coord[0], t);
                                      });
            std::cout << ", [w.r.t. exact, recons] " << error;
        }

        double error = 0;
        samurai::for_each_cell(u_max.mesh(),
                               [&](auto& cell)
                               {
                                   auto cell2 = samurai::find_cell(u_recons.mesh(), cell.center());
                                   error += std::pow(u_max[cell] - u_recons[cell2], 2) * std::pow(cell.length, 1);
                               });
        error = std::sqrt(error);
        std::cout << ", [w.r.t. max level, recons] " << error;

        // Save the result
        if (t >= static_cast<double>(nsave + 1) * dt_save || t == Tf)
        {
            std::string suffix = (nfiles != 1) ? fmt::format("_ite_{}", nsave) : "";
            save(path, filename, u, suffix);

            std::string file = (nfiles != 1) ? fmt::format("{}_recons_ite_{}", filename, nsave) : fmt::format("{}_recons", filename);
            samurai::save(path, file, u_recons.mesh(), u_recons);
            nsave++;
        }

        std::cout << std::endl;
    }
}

int main(int argc, char* argv[])
{
    constexpr std::size_t dim = 1;
    using Config              = samurai::MRConfig<dim, 2, 2>;
    using Box                 = samurai::Box<double, dim>;

    auto& app = samurai::initialize("Finite volume example for the Burgers equation in 1d", argc, argv);

    std::cout << "------------------------- Burgers -------------------------" << std::endl;

    //--------------------//
    // Program parameters //
    //--------------------//

    // Simulation parameters
    double left_box      = -2;
    double right_box     = 3;
    std::string init_sol = "hat";

    // Time integration
    double cfl = 0.95;

    // Multiresolution parameters
    std::size_t min_level = 1;
    std::size_t max_level = 7;
    double mr_epsilon     = 1e-5; // Threshold used by multiresolution
    double mr_regularity  = 0.;   // Regularity guess for multiresolution

    // Output parameters
    fs::path path        = fs::current_path();
    std::string filename = "burgers_mra";
    std::size_t nfiles   = 50;

    app.add_option("--init-sol", init_sol, "Initial solution: hat/gaussian")->capture_default_str()->group("Simulation parameters");
    app.add_option("--cfl", cfl, "The CFL")->capture_default_str()->group("Simulation parameters");
    app.add_option("--min-level", min_level, "Minimum level of the multiresolution")->capture_default_str()->group("Multiresolution");
    app.add_option("--max-level", max_level, "Maximum level of the multiresolution")->capture_default_str()->group("Multiresolution");
    app.add_option("--mr-eps", mr_epsilon, "The epsilon used by the multiresolution to adapt the mesh")
        ->capture_default_str()
        ->group("Multiresolution");
    app.add_option("--mr-reg", mr_regularity, "The regularity criteria used by the multiresolution to adapt the mesh")
        ->capture_default_str()
        ->group("Multiresolution");
    app.add_option("--path", path, "Output path")->capture_default_str()->group("Ouput");
    app.add_option("--filename", filename, "File name prefix")->capture_default_str()->group("Ouput");
    app.add_option("--nfiles", nfiles, "Number of output files")->capture_default_str()->group("Ouput");
    app.allow_extras();
    SAMURAI_PARSE(argc, argv);

    //--------------------//
    // Problem definition //
    //--------------------//

    Box box({left_box}, {right_box});
    samurai::MRMesh<Config> mesh{box, min_level, max_level};
    samurai::MRMesh<Config> max_level_mesh{box, max_level, max_level};

    auto u    = samurai::make_scalar_field<double>("u", mesh);
    auto unp1 = samurai::make_scalar_field<double>("unp1", mesh);

    auto u_max    = samurai::make_scalar_field<double>("u", max_level_mesh);
    auto unp1_max = samurai::make_scalar_field<double>("unp1", max_level_mesh);

    filename += "_" + init_sol;

    // Boundary conditions
    samurai::make_bc<samurai::Dirichlet<1>>(u, 0.0);
    samurai::make_bc<samurai::Dirichlet<1>>(u_max, 0.0);

    auto scheme = 0.5 * samurai::make_convection_upwind<decltype(u)>();

    //-----------------//
    // Run simulations //
    //-----------------//

    std::size_t nsave;

    scheme.enable_finer_level_flux(false);
    run_simulation(u, unp1, u_max, unp1_max, scheme, cfl, mr_epsilon, mr_regularity, init_sol, nfiles, path, filename, nsave);

    scheme.enable_finer_level_flux(true);
    samurai::args::finer_level_flux = -1;
    run_simulation(u, unp1, u_max, unp1_max, scheme, cfl, mr_epsilon, mr_regularity, init_sol, nfiles, path, filename, nsave);

    std::cout << std::endl;
    std::cout << "Run the following commands to view the results:" << std::endl;
    std::cout << "max-level-flux disabled:" << std::endl;
    std::cout << "     python ../python/read_mesh.py " << filename << "_recons_ite_ --field u --start 0 --end " << nsave << std::endl;
    std::cout << "max-level-flux enabled:" << std::endl;
    std::cout << "     python ../python/read_mesh.py " << filename << "_mlf_recons_ite_ --field u --start 0 --end " << nsave << std::endl;

    samurai::finalize();
    return 0;
}
