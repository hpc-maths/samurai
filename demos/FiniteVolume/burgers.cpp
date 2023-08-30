// Copyright 2021 SAMURAI TEAM. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.
#include "CLI/CLI.hpp"

#include <samurai/hdf5.hpp>
#include <samurai/mr/adapt.hpp>
#include <samurai/mr/mesh.hpp>
#include <samurai/reconstruction.hpp>
#include <samurai/schemes/fv.hpp>

#include <filesystem>
namespace fs = std::filesystem;

template <std::size_t dim>
double exact_solution(xt::xtensor_fixed<double, xt::xshape<dim>> coords, double t)
{
    const double a = 1;
    const double b = 0;
    double x       = coords(0);
    return (a * x + b) / (a * t + 1);
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
    static constexpr std::size_t dim = 1;
    using Config                     = samurai::MRConfig<dim>;
    using Box                        = samurai::Box<double, dim>;
    using point_t                    = typename Box::point_t;

    std::cout << "------------------------- Burgers -------------------------" << std::endl;

    //--------------------//
    // Program parameters //
    //--------------------//

    // Simulation parameters
    double left_box      = -1;
    double right_box     = 1;
    std::string init_sol = "hat";

    // Time integration
    double Tf  = 1.;
    double dt  = Tf / 100;
    double cfl = 0.95;

    // Multiresolution parameters
    std::size_t min_level = 0;
    std::size_t max_level = dim == 1 ? 5 : 3;
    double mr_epsilon     = 1e-4; // Threshold used by multiresolution
    double mr_regularity  = 1.;   // Regularity guess for multiresolution

    // Output parameters
    fs::path path        = fs::current_path();
    std::string filename = "burgers";
    std::size_t nfiles   = 50;

    CLI::App app{"Finite volume example for the heat equation in 1d"};
    app.add_option("--left", left_box, "The left border of the box")->capture_default_str()->group("Simulation parameters");
    app.add_option("--right", right_box, "The right border of the box")->capture_default_str()->group("Simulation parameters");
    app.add_option("--init-sol", init_sol, "Initial solution: hat/linear")->capture_default_str()->group("Simulation parameters");
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

    //--------------------//
    // Problem definition //
    //--------------------//

    point_t box_corner1, box_corner2;
    box_corner1.fill(left_box);
    box_corner2.fill(right_box);
    Box box(box_corner1, box_corner2);
    samurai::MRMesh<Config> mesh{box, min_level, max_level};

    auto u    = samurai::make_field<1>("u", mesh);
    auto unp1 = samurai::make_field<1>("unp1", mesh);

    // Initial solution
    if (init_sol == "linear")
    {
        samurai::for_each_cell(mesh,
                               [&](auto& cell)
                               {
                                   u[cell] = exact_solution(cell.center(), 0);
                               });
    }
    else
    {
        samurai::for_each_cell(mesh,
                               [&](auto& cell)
                               {
                                   double x   = cell.center(0);
                                   double max = 2;
                                   if (x >= -0.5 && x <= 0)
                                   {
                                       u[cell] = 2 * max * x + max;
                                   }
                                   else if (x >= 0 && x <= 0.5)
                                   {
                                       u[cell] = -2 * max * x + max;
                                   }
                                   else
                                   {
                                       u[cell] = 0;
                                   }
                               });
    }

    // Boundary conditions
    if (init_sol == "linear")
    {
        samurai::make_bc<samurai::Dirichlet>(u,
                                             [&](const auto& coord)
                                             {
                                                 auto corrected = coord;
                                                 if (corrected(0) < -1)
                                                 {
                                                     corrected(0) = -1;
                                                 }
                                                 else if (corrected(0) > 1)
                                                 {
                                                     corrected(0) = 1;
                                                 }
                                                 return exact_solution(corrected, 0);
                                                 // return exact_solution(coord, 0);
                                             });
    }
    else
    {
        samurai::make_bc<samurai::Dirichlet>(u, 0.0);
    }

    auto f = [](auto x)
    {
        return pow(x, 2) / 2;
    };

    auto upwind_f = samurai::make_flux_definition<decltype(u)>(
        [&](auto& v, auto& cells)
        {
            using flux_computation_t = samurai::NormalFluxDefinition<std::decay_t<decltype(v)>>;
            using flux_value_t       = typename flux_computation_t::flux_value_t;
            // static_assert(std::is_same_v<flux_value_t, double>);

            // flux_value_t flux = (f(v[cells[0]]) + f(v[cells[1]])) / 2;
            flux_value_t flux = v[cells[0]] >= 0 ? f(v[cells[0]]) : f(v[cells[1]]);
            return flux;
        });

    auto div_f = samurai::make_divergence_FV(upwind_f, u);

    //--------------------//
    //   Time iteration   //
    //--------------------//

    double dx = samurai::cell_length(max_level);
    dt        = cfl * dx / pow(2, dim);

    auto MRadaptation = samurai::make_MRAdapt(u);
    MRadaptation(mr_epsilon, mr_regularity);

    save(path, filename, u, "_init");
    // double dt_save    = Tf / static_cast<double>(nfiles);
    std::size_t nsave = 0, nt = 0;

    {
        std::string suffix = (nfiles != 1) ? fmt::format("_ite_{}", nsave++) : "";
        save(path, filename, u, suffix);
    }

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

        // Boundary conditions
        if (init_sol == "linear")
        {
            u.get_bc().clear();
            samurai::make_bc<samurai::Dirichlet>(u,
                                                 [&](const auto& coord)
                                                 {
                                                     auto corrected = coord;
                                                     if (corrected(0) < -1)
                                                     {
                                                         corrected(0) = -1;
                                                     }
                                                     else if (corrected(0) > 1)
                                                     {
                                                         corrected(0) = 1;
                                                     }
                                                     return exact_solution(corrected, t - dt);
                                                     // return exact_solution(coord, t - dt);
                                                 });
        }

        auto div_f_u = div_f(u);
        unp1         = u - dt * div_f_u;

        // u <-- unp1
        std::swap(u.array(), unp1.array());

        // Save the result
        // if (t >= static_cast<double>(nsave + 1) * dt_save || t == Tf)
        {
            std::string suffix = (nfiles != 1) ? fmt::format("_ite_{}", nsave++) : "";
            save(path, filename, u, suffix);
        }

        // Compute the error at instant t with respect to the exact solution
        if (init_sol == "linear")
        {
            double error = samurai::L2_error(u,
                                             [&](auto& coord)
                                             {
                                                 return exact_solution(coord, t);
                                             });
            std::cout << ", L2-error: " << std::scientific << std::setprecision(2) << error;

            if (mesh.min_level() != mesh.max_level())
            {
                // Reconstruction on the finest level
                samurai::update_ghost_mr(u);
                auto u_recons = samurai::reconstruction(u);
                error         = samurai::L2_error(u_recons,
                                          [&](auto& coord)
                                          {
                                              return exact_solution(coord, t);
                                          });
                std::cout << ", L2-error (recons): " << std::scientific << std::setprecision(2) << error;
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

    return 0;
}
