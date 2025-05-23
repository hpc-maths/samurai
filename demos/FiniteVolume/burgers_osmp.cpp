// Copyright 2018-2024 the samurai's authors
// SPDX-License-Identifier:  BSD-3-Clause

#include <samurai/io/hdf5.hpp>
#include <samurai/mr/adapt.hpp>
#include <samurai/mr/mesh.hpp>
#include <samurai/samurai.hpp>
#include <samurai/schemes/fv.hpp>

#include "convection_nonlin_osmp.hpp"
#include <algorithm>

#include <filesystem>
namespace fs = std::filesystem;

template <std::size_t field_size, std::size_t dim>
auto exact_solution(xt::xtensor_fixed<double, xt::xshape<dim>> coords, double t)
{
    const double a  = 1;
    const double b  = 0;
    const double& x = coords(0);
    auto value      = (a * x + b) / (a * t + 1);
    return samurai::CollapsArray<double, field_size, false>(value);
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
                           [&](const auto& cell)
                           {
                               level_[cell] = cell.level;
                           });

#ifdef SAMURAI_WITH_MPI
    mpi::communicator world;
    samurai::save(path, fmt::format("{}_size_{}{}", filename, world.size(), suffix), mesh, u, level_);
#else
    samurai::save(path, fmt::format("{}{}", filename, suffix), mesh, u, level_);
#endif
}

template <std::size_t dim, std::size_t field_size>
int main_dim(int argc, char* argv[])
{
    auto& app = samurai::initialize("Finite volume example for the Burgers equation", argc, argv);

    static constexpr std::size_t norder = 3;

    using Config  = samurai::MRConfig<dim, norder>;
    using Box     = samurai::Box<double, dim>;
    using point_t = typename Box::point_t;

    std::cout << "------------------------- Burgers -------------------------" << std::endl;

    //--------------------//
    // Program parameters //
    //--------------------//

    // Simulation parameters
    double left_box      = -2;
    double right_box     = 3;
    std::string init_sol = "gaussian";

    // Time integration
    double Tf     = 24.;
    double dt_CFL = Tf / 100;
    double cfl    = 0.5;

    // Multiresolution parameters
    std::size_t min_level = 3;
    std::size_t max_level = 9;
    double mr_epsilon     = 1e-3; // Threshold used by multiresolution
    double mr_regularity  = 1.;   // Regularity guess for multiresolution

    // Output parameters
    fs::path path        = fs::current_path();
    std::string filename = "burgers_" + std::to_string(dim) + "D";
    std::size_t nfiles   = 50;

    app.add_option("--left", left_box, "The left border of the box")->capture_default_str()->group("Simulation parameters");
    app.add_option("--right", right_box, "The right border of the box")->capture_default_str()->group("Simulation parameters");
    app.add_option("--init-sol", init_sol, "Initial solution: hat/linear/bands/gaussian")->capture_default_str()->group("Simulation parameters");
    // app.add_option("--norder", norder, "norder of OSMP scheme")->capture_default_str()->group("Simulation parameters");
    app.add_option("--Tf", Tf, "Final time")->capture_default_str()->group("Simulation parameters");
    app.add_option("--dt", dt_CFL, "Time step")->capture_default_str()->group("Simulation parameters");
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
    samurai::MRMesh<Config> mesh{box, min_level, max_level};

    auto u    = samurai::make_field<field_size>("u", mesh);
    auto u1   = samurai::make_field<field_size>("u1", mesh);
    auto u2   = samurai::make_field<field_size>("u2", mesh);
    auto unp1 = samurai::make_field<field_size>("unp1", mesh);

    double dt = dt_CFL;

    // Initial solution
    if (dim == 1 && init_sol == "linear")
    {
        samurai::for_each_cell(mesh,
                               [&](auto& cell)
                               {
                                   u[cell] = exact_solution<field_size>(cell.center(), 0);
                               });
    }
    else if (init_sol == "hat")
    {
        samurai::for_each_cell(mesh,
                               [&](auto& cell)
                               {
                                   const double max = 1;
                                   const double r   = 0.5;

                                   double dist = 0;
                                   for (std::size_t d = 0; d < dim; ++d)
                                   {
                                       dist += std::pow(cell.center(d), 2);
                                   }
                                   dist = std::sqrt(dist);

                                   double value = (dist <= r) ? (-max / r * dist + max) : 0;
                                   u[cell]      = value;
                               });
    }
    else if (dim > 1 && field_size > 1 && init_sol == "bands")
    {
        if constexpr (dim > 1 && field_size > 1)
        {
            samurai::for_each_cell(mesh,
                                   [&](auto& cell)
                                   {
                                       const double max = 2;
                                       using size_type  = typename decltype(u)::size_type;
                                       for (std::size_t d = 0; d < dim; ++d)
                                       {
                                           if (cell.center(d) >= -0.5 && cell.center(d) <= 0)
                                           {
                                               u[cell][static_cast<size_type>(d)] = 2 * max * cell.center(d) + max;
                                           }
                                           else if (cell.center(d) >= 0 && cell.center(d) <= 0.5)
                                           {
                                               u[cell][static_cast<size_type>(d)] = -2 * max * cell.center(d) + max;
                                           }
                                           else
                                           {
                                               u[cell][static_cast<size_type>(d)] = 0;
                                           }
                                       }
                                   });
        }
    }
    else if (dim == 2 && init_sol == "gaussian")
    {
        samurai::for_each_cell(mesh,
                               [&](auto& cell)
                               {
                                   double dist2 = 0;
                                   for (std::size_t d = 0; d < dim; ++d)
                                   {
                                       dist2 += std::pow(cell.center(d), 2);
                                   }
                                   u[cell] = 0.1 * exp(-2.0 * dist2);
                               });
    }
    else
    {
        std::cerr << "Unmanaged initial solution '" << init_sol << "'.";
        return EXIT_FAILURE;
    }

    // Boundary conditions
    if (dim == 1 && init_sol == "linear")
    {
        samurai::make_bc<samurai::Dirichlet<norder>>(u,
                                                     [&](const auto&, const auto&, const auto& coord)
                                                     {
                                                         return exact_solution<field_size>(coord, 0);
                                                     });
    }
    else
    {
        if constexpr (field_size == 1)
        {
            samurai::make_bc<samurai::Dirichlet<norder>>(u, 0.0);
        }
        else if constexpr (field_size == 2)
        {
            samurai::make_bc<samurai::Dirichlet<norder>>(u, 0.0, 0.0);
        }
        else if constexpr (field_size == 3)
        {
            samurai::make_bc<samurai::Dirichlet<norder>>(u, 0.0, 0.0, 0.0);
        }
    }

    u1.copy_bc_from(u);
    u2.copy_bc_from(u);

    double cst = dim == 1 ? 0.5 : 1; // if dim == 1, we want f(u) = (1/2)*u^2
    auto conv  = cst * samurai::make_convection_osmp<decltype(u), norder>(dt);
    // auto conv  = cst * samurai::make_convection_upwind<decltype(u)>();

    //--------------------//
    //   Time iteration   //
    //--------------------//
    double dx = mesh.cell_length(max_level);

    double velocity = 0.;
    double vel2;
    samurai::for_each_cell(mesh,
                           [&](auto& cell)
                           {
                               vel2 = 0.;
                               for (std::size_t l = 0; l < dim; ++l)
                               {
                                   vel2 += u[cell][l] * u[cell][l];
                               }
                               velocity = std::max(velocity, std::sqrt(vel2));
                           });
    std::cout << "velocity :" << velocity << std::endl;
    if (velocity != 0.)
    {
        dt_CFL = cfl * dx / (pow(2, dim - 1) * velocity);
    }
    else
    {
        dt_CFL = cfl * dx;
    }
    // dt_CFL = cfl * dx / 0.1;
    std::cout << "dt_CFL :" << dt_CFL << std::endl;

    auto MRadaptation = samurai::make_MRAdapt(u);
    MRadaptation(mr_epsilon, mr_regularity);

    double dt_save    = Tf / static_cast<double>(nfiles);
    std::size_t nsave = 0, nt = 0;

    {
        std::string suffix = (nfiles != 1) ? fmt::format("_ite_{}", nsave++) : "";
        save(path, filename, u, suffix);
    }

    double t = 0;
    while (t != Tf)
    {
        // Move to next timestep
        t += dt_CFL;
        if (t > Tf)
        {
            dt_CFL += Tf - t;
            t = Tf;
        }
        std::cout << fmt::format("iteration {}: t = {:.2f}, dt = {}", nt++, t, dt_CFL) << std::flush;

        // Mesh adaptation
        MRadaptation(mr_epsilon, mr_regularity);
        u1.resize();
        u2.resize();
        unp1.resize();

        // Boundary conditions
        if (dim == 1 && init_sol == "linear")
        {
            u.get_bc().clear();
            samurai::make_bc<samurai::Dirichlet<norder>>(u,
                                                         [&](const auto&, const auto&, const auto& coord)
                                                         {
                                                             return exact_solution<field_size>(coord, t - dt_CFL);
                                                         });
        }

        // osmp scheme
        samurai::update_ghost_mr(u);
        dt = 0.5 * dt_CFL;
        u1 = u - dt * conv(0, u);
        samurai::update_ghost_mr(u1);
        dt = dt_CFL;
        u2 = u1 - dt * conv(1, u1);
        samurai::update_ghost_mr(u2);
        dt   = 0.5 * dt_CFL;
        unp1 = u2 - dt * conv(0, u2);
        // samurai::update_ghost_mr(u);
        // unp1 =  u  - dt_CFL * conv(u);

        // samurai::update_ghost_mr(u);
        // unp1 =  u  - dt_CFL * conv(0, u);

        // u <-- unp1
        std::swap(u.array(), unp1.array());

        // Save the result
        if (t >= static_cast<double>(nsave + 1) * dt_save || t == Tf)
        {
            std::string suffix = (nfiles != 1) ? fmt::format("_ite_{}", nsave++) : "";
            save(path, filename, u, suffix);
        }

        // Compute the error at instant t with respect to the exact solution
        if (init_sol == "linear")
        {
            double error = samurai::L2_error(u,
                                             [&](const auto& coord)
                                             {
                                                 return exact_solution<field_size>(coord, t);
                                             });
            std::cout << ", L2-error: " << std::scientific << std::setprecision(2) << error;

            if (mesh.min_level() != mesh.max_level())
            {
                // Reconstruction on the finest level
                samurai::update_ghost_mr(u);
                auto u_recons = samurai::reconstruction(u);
                error         = samurai::L2_error(u_recons,
                                          [&](const auto& coord)
                                          {
                                              return exact_solution<field_size>(coord, t);
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

    samurai::finalize();
    return 0;
}

int main(int argc, char* argv[])
{
    static constexpr std::size_t dim = 2;

    // 1  : scalar equation
    // dim: vector equation
    static constexpr std::size_t field_size = 2;
    return main_dim<dim, field_size>(argc, argv);
}
