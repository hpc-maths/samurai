// Copyright 2018-2024 the samurai's authors
// SPDX-License-Identifier:  BSD-3-Clause

#include <samurai/domain_builder.hpp>
#include <samurai/hdf5.hpp>
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
    auto& app = samurai::initialize("Finite volume example for the linear convection equation", argc, argv);

    static constexpr std::size_t dim = 2;
    using Config                     = samurai::MRConfig<dim, 3>;
    using Mesh                       = samurai::MRMesh<Config>;

    std::cout << "------------------------- Linear convection -------------------------" << std::endl;

    //--------------------//
    // Program parameters //
    //--------------------//

    // Time integration
    double Tf  = 3;
    double dt  = 0;
    double cfl = 0.95;

    // Multiresolution parameters
    std::size_t min_level = 1;
    std::size_t max_level = dim == 1 ? 6 : 4;
    double mr_epsilon     = 1e-3; // Threshold used by multiresolution
    double mr_regularity  = 1.;   // Regularity guess for multiresolution

    // Output parameters
    fs::path path        = fs::current_path();
    std::string filename = "linear_convection_obstacle_" + std::to_string(dim) + "D";
    std::size_t nfiles   = 0;

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
    app.add_option("--path", path, "Output path")->capture_default_str()->group("Output");
    app.add_option("--filename", filename, "File name prefix")->capture_default_str()->group("Output");
    app.add_option("--nfiles", nfiles, "Number of output files")->capture_default_str()->group("Output");
    app.allow_extras();
    SAMURAI_PARSE(argc, argv);

    //--------------------//
    // Problem definition //
    //--------------------//

    samurai::DomainBuilder<dim> domain({-1., -1.}, {1., 1.});
    domain.remove({0.0, 0.0}, {0.4, 0.4});

    Mesh mesh(domain, min_level, max_level);

    // Initial solution
    auto u = samurai::make_field<1>("u",
                                    mesh,
                                    [](const auto& coords)
                                    {
                                        const auto& x = coords(0);
                                        const auto& y = coords(1);
                                        return (x >= -0.8 && x <= -0.3 && y >= 0.3 && y <= 0.8) ? 1. : 0.;
                                    });

    auto unp1 = samurai::make_field<1>("unp1", mesh);
    // Intermediary fields for the RK3 scheme
    auto u1 = samurai::make_field<1>("u1", mesh);
    auto u2 = samurai::make_field<1>("u2", mesh);

    // Convection operator
    samurai::VelocityVector<dim> constant_velocity = {1, -1};

    auto velocity = samurai::make_field<dim>("velocity",
                                             mesh,
                                             [&](const auto&)
                                             {
                                                 return constant_velocity;
                                             });

    samurai::make_bc<samurai::Dirichlet<1>>(velocity, 0., 0.); // Wall boundary condition
    samurai::make_bc<samurai::Dirichlet<3>>(u, 0.);
    u1.copy_bc_from(u);
    u2.copy_bc_from(u);

    auto conv = samurai::make_convection_weno5<decltype(u)>(velocity);

    //--------------------//
    //   Time iteration   //
    //--------------------//

    if (dt == 0)
    {
        double dx             = mesh.cell_length(max_level);
        auto a                = xt::abs(constant_velocity);
        double sum_velocities = xt::sum(xt::abs(constant_velocity))();
        dt                    = cfl * dx / sum_velocities;
    }

    auto MRadaptation = samurai::make_MRAdapt(u);
    MRadaptation(mr_epsilon, mr_regularity, velocity);

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
        // Move to next timestep
        t += dt;
        if (t > Tf)
        {
            dt += Tf - t;
            t = Tf;
        }
        std::cout << fmt::format("iteration {}: t = {:.2f}, dt = {}", nt++, t, dt) << std::flush;

        // Mesh adaptation

        MRadaptation(mr_epsilon, mr_regularity, velocity);
        // samurai::save(path, fmt::format("{}_mesh", filename), {true, true}, mesh, u);
        samurai::update_ghost_mr(u);
        samurai::for_each_cell(mesh,
                               [&](auto& cell)
                               {
                                   velocity[cell] = constant_velocity;
                               });
        samurai::update_ghost_mr(velocity);
        unp1.resize();
        u1.resize();
        u2.resize();

        // unp1 = u - dt * conv(u);

        // TVD-RK3 (SSPRK3)
        u1 = u - dt * conv(u);
        samurai::update_ghost_mr(u1);
        u2 = 3. / 4 * u + 1. / 4 * (u1 - dt * conv(u1));
        samurai::update_ghost_mr(u2);
        unp1 = 1. / 3 * u + 2. / 3 * (u2 - dt * conv(u2));

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

        std::cout << std::endl;
    }

    samurai::finalize();
    return 0;
}
