// Copyright 2018-2024 the samurai's authors
// SPDX-License-Identifier:  BSD-3-Clause

#include <samurai/hdf5.hpp>
#include <samurai/mr/adapt.hpp>
#include <samurai/mr/mesh.hpp>
#include <samurai/samurai.hpp>
#include <samurai/schemes/fv.hpp>

#include <algorithm>
#include "convection_euler_osmp.hpp"

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
void save(const fs::path& path, const std::string& filename, const Field& u, const double& gamma, const std::string& suffix = "")
{
    static constexpr std::size_t dim        = Field::dim;
    static constexpr std::size_t field_size = Field::size;
    
    auto mesh   = u.mesh();
    auto level_ = samurai::make_field<std::size_t, 1>("level", mesh);
    auto density = samurai::make_field<double, 1>("rho", mesh);
    auto vel = samurai::make_field<double, 1>("v", mesh);
    auto pressure = samurai::make_field<double, 1>("pressure", mesh);
    
    // xt::xtensor_fixed<double, xt::xshape<dim>> velocity;

    if (!fs::exists(path))
    {
        fs::create_directory(path);
    }

    samurai::for_each_cell(mesh,
                           [&](const auto& cell)
                           {
                                level_[cell] = cell.level;
                                density[cell] = u[cell](0);

                                double rho_ec = 0.;
                                for (std::size_t d = 0; d < dim; ++d)
                                {
                                    // vel[cell](d) = u[cell](d+1) / u[cell](0);
                                    vel[cell] = u[cell](d+1) / u[cell](0);
                                    rho_ec += u[cell](d+1) * u[cell](d+1);
                                }
                                rho_ec = 0.5 * rho_ec / u[cell](0);
                                pressure[cell] = (gamma-1) * ( u[cell](field_size-1) - rho_ec );
                            //    pressure[cell] = compute_Pressure<const Field, dim, field_size>(vel_, gamma);
                           });

#ifdef SAMURAI_WITH_MPI
    mpi::communicator world;
    samurai::save(path, fmt::format("{}_size_{}{}", filename, world.size(), suffix), mesh, density, vel, pressure, level_);
#else
    samurai::save(path, fmt::format("{}{}", filename, suffix), mesh, density, vel, pressure, level_);
#endif
}

template <std::size_t dim, std::size_t field_size>
int main_dim(int argc, char* argv[])
{
    auto& app = samurai::initialize("Finite volume example for the Burgers equation", argc, argv);

    static constexpr std::size_t norder = 1;
    
    static constexpr double gamma  = 1.4;

    using Config  = samurai::MRConfig<dim, norder>;
    using Box     = samurai::Box<double, dim>;
    using point_t = typename Box::point_t;

    std::cout << "------------------------- EULER -------------------------" << std::endl;

    //--------------------//
    // Program parameters //
    //--------------------//

    // Simulation parameters
    double left_box      = -1.;
    double right_box     = 1.;
    std::string init_sol = "ShockWave";

    // Time integration
    double Tf  = 0.5;
    double dt_CFL  = Tf / 100;
    double cfl = 0.5;

    // Multiresolution parameters
    std::size_t min_level = 6;
    std::size_t max_level = 6;
    double mr_epsilon     = 1e-3; // Threshold used by multiresolution
    double mr_regularity  = 1.;   // Regularity guess for multiresolution

    // Output parameters
    fs::path path        = fs::current_path();
    std::string filename = "Euler_" + std::to_string(dim) + "D";
    std::size_t nfiles   = 50;

    app.add_option("--left", left_box, "The left border of the box")->capture_default_str()->group("Simulation parameters");
    app.add_option("--right", right_box, "The right border of the box")->capture_default_str()->group("Simulation parameters");
    app.add_option("--init-sol", init_sol, "Initial solution: hat/linear/bands/gaussian")->capture_default_str()->group("Simulation parameters");
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

    double rho_left = 1.;
    double vel_left = 0.;
    double energ_left = 2.5; 
    double rho_right = 0.125;
    double vel_right = 0.;
    double energ_right = 0.25; 

    // Initial solution
    if (dim == 1 && init_sol == "ShockWave")
    {
        double x_0 = 0.;
        samurai::for_each_cell(mesh,
            [&](auto& cell)
            {
                if ( cell.center(0) < x_0 )
               {
                    u[cell].fill(0.);
                    u[cell](0) = rho_left;
                    u[cell](1) = rho_left * vel_left;
                    u[cell](2) = rho_left * energ_left; 
               }
               else
               {
                    u[cell].fill(0.);
                    u[cell](0) = rho_right;
                    u[cell](1) = rho_right * vel_right;
                    u[cell](2) = rho_right * energ_right; 
               }
           });
    }
    else if (dim == 2 && init_sol == "ShockWave")
    {
        double x_0 = 0.;
        samurai::for_each_cell(mesh,
            [&](auto& cell)
            {
                if ( cell.center(0) < x_0 )
                {
                    u[cell].fill(0.);
                    u[cell](0) = rho_left;
                    u[cell](1) = rho_left * vel_left;
                    u[cell](2) = 0.;
                    u[cell](3) = rho_left * energ_left; 
                }
                else
                {
                    u[cell].fill(0.);
                    u[cell](0) = rho_right;
                    u[cell](1) = rho_right * vel_right;
                    u[cell](2) = 0.;
                    u[cell](3) = rho_right * energ_right; 
                }
            });
    }
    else
    {
        std::cerr << "Unmanaged initial solution '" << init_sol << "'.";
        return EXIT_FAILURE;
    }

    // Boundary conditions
    if (dim == 1 && init_sol == "ShockWave")
    {
        samurai::DirectionVector<dim> left   = {-1};
        samurai::DirectionVector<dim> right  = {1};
        std::cout << " B.C. ShockWave : field_size = " << field_size << std::endl;
        samurai::make_bc<samurai::Dirichlet<norder>>(u, rho_left, rho_left*vel_left, rho_left*energ_left)->on(left);
        samurai::make_bc<samurai::Dirichlet<norder>>(u, rho_right, rho_right*vel_right, rho_right*energ_right)->on(right);
    }
    else
    {
        std::cout << " Autre B.C. : field_size = " << field_size << std::endl;

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

    auto conv  = samurai::make_convection_euler_osmp<decltype(u), norder>(dt);

    //--------------------//
    //   Time iteration   //
    //--------------------//
    
    // double dx = mesh.cell_length(max_level);

    auto MRadaptation = samurai::make_MRAdapt(u);
    MRadaptation(mr_epsilon, mr_regularity);

    double dt_save    = Tf / static_cast<double>(nfiles);
    std::size_t nsave = 0, nt = 0;

    {
        std::string suffix = (nfiles != 1) ? fmt::format("_ite_{}", nsave++) : "";
        save(path, filename, u, gamma, suffix);
    }

    double t = 0;
    while (t != Tf)
    {
        // Time step calculus satisfying CFL constraint
        dt_CFL = 1.e+10;
        samurai::for_each_cell(mesh, [&](auto & cell)
        {
            // EigenValues
 
            double rho_ec = 0.;
            for (std::size_t l = 1; l < dim+1; ++l)
            {
                rho_ec += u[cell](l)*u[cell](l);
            }
            rho_ec = 0.5*rho_ec / u[cell](0);
            double c_son = std::sqrt( gamma * (gamma-1)*(u[cell](field_size-1) - rho_ec)/ u[cell](0) );

            double velocity = 0.;
            for (std::size_t d = 0; d < dim; ++d)
            {
                velocity = std::max(velocity, std::abs(u[cell](d+1)/u[cell](0))+c_son);
            }

            for (std::size_t l = 0; l < field_size; ++l)
            {
                dt_CFL =  std::min( cfl * mesh.cell_length(cell.level) / velocity, dt_CFL );
            }
        });
        
        // Move to next timestep
        t += dt_CFL;
        if (t > Tf)
        {
            dt_CFL += Tf - t;
            t = Tf;
        }
        std::cout << " Time = " << t << " dt_CFL :" << dt_CFL << std::endl;

        std::cout << fmt::format("iteration {}: t = {:.2f}, dt = {}", nt++, t, dt_CFL) << std::flush;

        // Mesh adaptation
        MRadaptation(mr_epsilon, mr_regularity);
        u1.resize();
        u2.resize();
        unp1.resize();

        // osmp scheme
        if constexpr (dim == 1)
        {
            samurai::update_ghost_mr(u);
            unp1 =  u  - dt_CFL * conv(0, u);
        }
        else
        {
            samurai::update_ghost_mr(u);
            dt = 0.5 * dt_CFL;
            u1   =  u   - dt * conv(0, u);
            samurai::update_ghost_mr(u1);
            dt = dt_CFL;
            u2 =  u1  - dt * conv(1, u1);
            samurai::update_ghost_mr(u2);
            dt = 0.5 * dt_CFL;
            unp1 =  u2  - dt * conv(0, u2);
        }

        // u <-- unp1
        std::swap(u.array(), unp1.array());

        // Save the result
        if (t >= static_cast<double>(nsave + 1) * dt_save || t == Tf)
        {
            std::string suffix = (nfiles != 1) ? fmt::format("_ite_{}", nsave++) : "";
            save(path, filename, u, gamma, suffix);
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
    static constexpr std::size_t dim = 1;

    // 1  : scalar equation
    // dim: vector equation
    static constexpr std::size_t field_size = dim+2;
    return main_dim<dim, field_size>(argc, argv);
}
