// Copyright 2018-2024 the samurai's authors
// SPDX-License-Identifier:  BSD-3-Clause
#include <CLI/CLI.hpp>

#include <samurai/hdf5.hpp>
#include <samurai/mr/adapt.hpp>
#include <samurai/mr/mesh.hpp>
#include <samurai/schemes/fv.hpp>
#include <samurai/samurai.hpp>

#include <samurai/load_balancing.hpp>
#include <samurai/load_balancing_sfc.hpp>
#include <samurai/load_balancing_diffusion.hpp>
#include <samurai/load_balancing_diffusion_cell.hpp>
#include <samurai/load_balancing_diffusion_interval.hpp>
#include <samurai/load_balancing_void.hpp>

#include <samurai/timers.hpp>

#include <filesystem>
namespace fs = std::filesystem;

template <std::size_t dim>
double exact_solution(xt::xtensor_fixed<double, xt::xshape<dim>> coords, double t)
{
    const double a  = 1;
    const double b  = 0;
    const double& x = coords(0);
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
                           [&](const auto& cell)
                           {
                               level_[cell] = cell.level;
                           });

    samurai::save(path, fmt::format("{}{}", filename, suffix), mesh, u, level_);
}

int main(int argc, char* argv[])
{
    samurai::initialize(argc, argv);

    static constexpr std::size_t dim = 2;
    using Config                     = samurai::MRConfig<dim, 3>;
    using Box                        = samurai::Box<double, dim>;
    using point_t                    = typename Box::point_t;

    std::cout << "------------------------- Linear convection -------------------------" << std::endl;

    //--------------------//
    // Program parameters //
    //--------------------//

    // Simulation parameters
    double left_box      = 0;
    double right_box     = 4;
    std::string init_sol = "hat";

    // Time integration
    double Tf  = 3;
    double dt  = 0;
    double cfl = 0.95;

    // Multiresolution parameters
    std::size_t min_level = 0;
    std::size_t max_level = dim == 1 ? 5 : 3;
    double mr_epsilon     = 1e-4; // Threshold used by multiresolution
    double mr_regularity  = 1.;   // Regularity guess for multiresolution

    // Output parameters
    fs::path path        = fs::current_path();
    std::string filename = "linear_convection_" + std::to_string(dim) + "D";
    std::size_t nfiles   = 0;
    std::size_t nt_loadbalance = 10;

    CLI::App app{"Finite volume example for the heat equation in 1d"};
    app.add_option("--left", left_box, "The left border of the box")->capture_default_str()->group("Simulation parameters");
    app.add_option("--right", right_box, "The right border of the box")->capture_default_str()->group("Simulation parameters");
    app.add_option("--init-sol", init_sol, "Initial solution: hat/linear/bands")->capture_default_str()->group("Simulation parameters");
    app.add_option("--Tf", Tf, "Final time")->capture_default_str()->group("Simulation parameters");
    app.add_option("--dt", dt, "Time step")->capture_default_str()->group("Simulation parameters");
    app.add_option("--cfl", cfl, "The CFL")->capture_default_str()->group("Simulation parameters");
    app.add_option("--min-level", min_level, "Minimum level of the multiresolution")->capture_default_str()->group("Multiresolution");
    app.add_option("--max-level", max_level, "Maximum level of the multiresolution")->capture_default_str()->group("Multiresolution");
    app.add_option("--nt-loadbalance", nt_loadbalance, "Maximum level of the multiresolution")->capture_default_str()->group("Multiresolution");
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
    std::array<bool, dim> periodic;
    periodic.fill(false);
    
    samurai::times::timers.start("init");
    samurai::MRMesh<Config> mesh{box, min_level, max_level, periodic};

    // Initial solution
    auto u = samurai::make_field<double, 1>("u",
                                    mesh,
                                    [&](const auto& coords)
                                    {
                                        if constexpr (dim == 1)
                                        {
                                            auto& x = coords(0);
                                            return (x >= -0.8 && x <= -0.3) ? 1. : 0.;
                                        }
                                        else if ( dim == 2 )
                                        {
                                            auto& x = coords(0);
                                            auto& y = coords(1);
                                            return (x <= 0.8 && x >= 0.3 && y >= 0.3 && y <= 0.8) ? 1. : 0.;
                                        }else {
                                            auto & x = coords( 0 );
                                            auto & y = coords( 1 );
                                            auto & z = coords( 2 );
                                            return (x <= 0.8 && x >= 0.3 && y >= 0.3 && y <= 0.8 && z >= 0.3 && z <= 0.8 ) ? 1. : 0.;
                                        }

                                    });
    samurai::times::timers.stop("init");

    auto unp1 = samurai::make_field<double, 1>("unp1", mesh);
    // Intermediary fields for the RK3 scheme
    auto u1 = samurai::make_field<double, 1>("u1", mesh);
    auto u2 = samurai::make_field<double, 1>("u2", mesh);

    samurai::make_bc<samurai::Dirichlet<1>>(u, 0.);
    samurai::make_bc<samurai::Dirichlet<1>>(unp1, 0.);
    samurai::make_bc<samurai::Dirichlet<1>>(u1, 0.);
    samurai::make_bc<samurai::Dirichlet<1>>(u2, 0.); 

    unp1.fill(0);
    u1.fill(0);
    u2.fill(0);

    // Convection operator
    samurai::VelocityVector<dim> velocity;
    velocity.fill(1);

    // origin weno5
    auto conv = samurai::make_convection_weno5<decltype(u)>(velocity);

    // SFC_LoadBalancer_interval<dim, Morton> balancer;
    // Void_LoadBalancer<dim> balancer;
    // Diffusion_LoadBalancer_cell<dim> balancer;
    // Diffusion_LoadBalancer_interval<dim> balancer;
    Load_balancing::Diffusion balancer;

    //--------------------//
    //   Time iteration   //
    //--------------------//

    if (dt == 0)
    {
        double dx             = samurai::cell_length(max_level);
        auto a                = xt::abs(velocity);
        double sum_velocities = xt::sum(xt::abs(velocity))();
        dt                    = cfl * dx / sum_velocities;
    }

    auto MRadaptation = samurai::make_MRAdapt(u);
    
    samurai::times::timers.start("MRadaptation");
    MRadaptation(mr_epsilon, mr_regularity);
    samurai::times::timers.stop("MRadaptation");

    double dt_save    = nfiles == 0 ? dt : Tf / static_cast<double>(nfiles);
    std::size_t nsave = 0, nt = 0;
    if (nfiles != 1)
    {
        std::string suffix = (nfiles != 1) ? fmt::format("_ite_{}", nsave++) : "";
        save(path, filename, u, suffix);
    }

    double t = 0;
    while (t != Tf)
    {

        if (nt % nt_loadbalance == 0 && nt > 1 )
        {
            samurai::times::timers.start("load-balancing");
            balancer.load_balance(mesh, u);
            samurai::times::timers.stop("load-balancing");
        }
        
        // Move to next timestep
        t += dt;
        if (t > Tf)
        {
            dt += Tf - t;
            t = Tf;
        }
        std::cout << fmt::format("iteration {}: t = {:.2f}, dt = {}", nt++, t, dt) << std::flush << std::endl;

        // Mesh adaptation
        samurai::times::timers.start("MRadaptation");
        MRadaptation(mr_epsilon, mr_regularity);
        samurai::times::timers.stop("MRadaptation");

        samurai::times::timers.start("update_ghost_mr");
        samurai::update_ghost_mr(u);
        samurai::times::timers.stop("update_ghost_mr");

        unp1.resize();
        unp1.fill(0);

        u1.resize();
        u2.resize();
        u1.fill(0);
        u2.fill(0);

        // unp1 = u - dt * conv(u);

        // TVD-RK3 (SSPRK3)
        samurai::times::timers.start("RK3");
        u1 = u - dt * conv(u);
        samurai::update_ghost_mr(u1);
        u2 = 3. / 4 * u + 1. / 4 * (u1 - dt * conv(u1));
        samurai::update_ghost_mr(u2);
        unp1 = 1. / 3 * u + 2. / 3 * (u2 - dt * conv(u2));
        samurai::times::timers.stop("RK3");
        // u <-- unp1
        std::swap(u.array(), unp1.array());

        // Save the result
        if (nfiles == 0 || t >= static_cast<double>(nsave + 1) * dt_save || t == Tf)
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
