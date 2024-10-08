// Copyright 2018-2024 the samurai's authors
// SPDX-License-Identifier:  BSD-3-Clause
#include <CLI/CLI.hpp>

#include <samurai/hdf5.hpp>
#include <samurai/mr/adapt.hpp>
#include <samurai/mr/mesh.hpp>
#include <samurai/petsc.hpp>
#include <samurai/samurai.hpp>

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

    auto cell_index = samurai::make_field<long long, 1>("cell_index", mesh);
    samurai::for_each_cell(mesh,
                           [&](const auto& cell)
                           {
                               cell_index[cell] = cell.index;
                           });

    auto rank = samurai::make_field<int, 1>("rank", mesh);

    mpi::communicator world;
    auto r = world.rank();
    samurai::for_each_cell(mesh,
                           [&](const auto& cell)
                           {
                               // std::cout << r << ": " << cell.index << std::endl;
                               rank[cell] = r;
                           });

    // std::cout << rank << std::endl;

    samurai::save(path, fmt::format("{}{}", filename, suffix), mesh, u, level_, rank, cell_index);
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
    double left_box  = -1;
    double right_box = 1;

    // Time integration
    double Tf  = 3;
    double dt  = 0;
    double cfl = 0.95;

    // Multiresolution parameters
    std::size_t min_level = 1;
    std::size_t max_level = dim == 1 ? 6 : 4;
    double mr_epsilon     = 1e-4; // Threshold used by multiresolution
    double mr_regularity  = 1.;   // Regularity guess for multiresolution

    // Output parameters
    fs::path path        = fs::current_path();
    std::string filename = "linear_convection_" + std::to_string(dim) + "D";
    std::size_t nfiles   = 0;

    CLI::App app{"Finite volume example for the linear convection equation"};
    app.add_option("--left", left_box, "The left border of the box")->capture_default_str()->group("Simulation parameters");
    app.add_option("--right", right_box, "The right border of the box")->capture_default_str()->group("Simulation parameters");
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

    static constexpr std::size_t dim_ = 1;
    // static constexpr int test         = 0;
    if constexpr (dim_ == 1)
    {
        samurai::Box<double, dim_> box1({-1}, {6});
        samurai::Box<double, dim_> box2({1}, {4});
        samurai::LevelCellArray<dim_> lca1(2, box1);
        samurai::LevelCellArray<dim_> lca2(1, box2, 0, lca1.scaling_factor());

        std::cout << "lca1:" << std::endl;
        samurai::for_each_cell(lca1,
                               [](auto& cell)
                               {
                                   std::cout << cell << std::endl;
                               });
        std::cout << "lca2:" << std::endl;
        samurai::for_each_cell(lca2,
                               [](auto& cell)
                               {
                                   std::cout << cell << std::endl;
                               });

        samurai::LevelCellArray<dim_> inters = samurai::intersection(lca1, lca2).on(0);
        std::cout << "inters:" << std::endl;
        samurai::for_each_cell(inters,
                               [](auto& cell)
                               {
                                   std::cout << cell << std::endl;
                               });
    }
    if constexpr (dim_ == 2)
    {
        samurai::Box<double, dim_> box10({-1, -1}, {6, 6});
        samurai::Box<double, dim_> box11({3, 3}, {5, 5});
        samurai::Box<double, dim_> box2({1, 1}, {4, 4});
        // samurai::LevelCellArray<dim_> lca10(0, box10, 0, 1);
        // samurai::LevelCellArray<dim_> lca11(0, box11, 0, 1);
        // samurai::LevelCellArray<dim_> lca2(0, box2, 0, 1);
        samurai::LevelCellArray<dim_> lca10(1, box10);
        samurai::LevelCellArray<dim_> lca11(0, box11, 0, lca10.scaling_factor());
        samurai::LevelCellArray<dim_> lca2(0, box2, 0, lca10.scaling_factor());

        // std::cout << "lca1:" << std::endl;
        // samurai::for_each_cell(lca10,
        //                        [](auto& cell)
        //                        {
        //                            std::cout << cell << std::endl;
        //                        });
        // std::cout << "lca2:" << std::endl;
        // samurai::for_each_cell(lca2,
        //                        [](auto& cell)
        //                        {
        //                            std::cout << cell << std::endl;
        //                        });
        std::cout << "sub:" << std::endl;
        auto sub = samurai::difference(lca10, lca11).on(0);
        // samurai::for_each_cell(sub,
        //                        [](auto& cell)
        //                        {
        //                            std::cout << cell << std::endl;
        //                        });

        samurai::LevelCellArray<dim_> inters = samurai::intersection(sub, lca2).on(0);
        std::cout << "inters:" << std::endl;
        samurai::for_each_cell(inters,
                               [](auto& cell)
                               {
                                   std::cout << cell << std::endl;
                               });
    }
    else
    {
        // samurai::Box<double, dim_> box({-1, -1}, {2, 2});
        // samurai::LevelCellArray<dim_> lca(0, box, 0, 1);
        // samurai::LevelCellArray<dim_> trans = samurai::translate(lca, samurai::DirectionVector<dim_>{1, 1});
        // samurai::for_each_cell(trans,
        //                        [](auto& cell)
        //                        {
        //                            std::cout << cell << std::endl;
        //                        });
    }
    std::cout << "------------------------------------------------" << std::endl;
    std::cout << "------------------------------------------------" << std::endl;

    point_t box_corner1, box_corner2;
    box_corner1.fill(left_box);
    box_corner2.fill(right_box);
    Box box(box_corner1, box_corner2);
    std::array<bool, dim> periodic;
    periodic.fill(false);
    samurai::MRMesh<Config> mesh{box, min_level, max_level, periodic};

    // Initial solution
    auto u = samurai::make_field<1>("u",
                                    mesh,
                                    [](const auto& coords)
                                    {
                                        if constexpr (dim == 1)
                                        {
                                            auto& x = coords(0);
                                            return (x >= -0.8 && x <= -0.3) ? 1. : 0.;
                                        }
                                        else
                                        {
                                            auto& x = coords(0);
                                            auto& y = coords(1);
                                            return (x >= -0.8 && x <= -0.3 && y >= 0.3 && y <= 0.8) ? 1. : 0.;
                                        }
                                    });

    auto unp1 = samurai::make_field<1>("unp1", mesh);
    // Intermediary fields for the RK3 scheme
    auto u1 = samurai::make_field<1>("u1", mesh);
    auto u2 = samurai::make_field<1>("u2", mesh);

    samurai::make_bc<samurai::Neumann<1>>(u, 0.);
    samurai::make_bc<samurai::Neumann<1>>(unp1, 0.);
    samurai::make_bc<samurai::Neumann<1>>(u1, 0.);
    samurai::make_bc<samurai::Neumann<1>>(u2, 0.);

    unp1.fill(0);
    u1.fill(0);
    u2.fill(0);

    // Convection operator
    samurai::VelocityVector<dim> velocity;
    velocity.fill(1);
    if constexpr (dim == 2)
    {
        velocity(1) = -1;
    }
    auto conv = samurai::make_convection_weno5<decltype(u)>(velocity);

    static constexpr std::size_t field_size = 2;

    auto u_ = samurai::make_field<field_size>("u_", mesh);
    std::array<samurai::VelocityVector<dim>, field_size> velocities;
    velocities[0] = {1, 1};
    velocities[1] = {-1, 1};

    auto multi_conv = samurai::make_multi_convection_weno5<decltype(u_)>(velocities);

    //--------------------//
    //   Time iteration   //
    //--------------------//

    if (dt == 0)
    {
        double dx             = mesh.cell_length(max_level);
        auto a                = xt::abs(velocity);
        double sum_velocities = xt::sum(xt::abs(velocity))();
        dt                    = cfl * dx / sum_velocities;
    }

    save(path, filename, u, "_init");
    // samurai::finalize();
    // return 0;

    auto MRadaptation = samurai::make_MRAdapt(u);
    MRadaptation(mr_epsilon, mr_regularity);

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
        MRadaptation(mr_epsilon, mr_regularity);
        samurai::update_ghost_mr(u);
        unp1.resize();
        u1.resize();
        u2.resize();
        u1.fill(0);
        u2.fill(0);

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
