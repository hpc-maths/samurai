// Copyright 2018-2025 the samurai's authors
// SPDX-License-Identifier:  BSD-3-Clause

#include <samurai/io/hdf5.hpp>
#include <samurai/io/restart.hpp>
#include <samurai/mr/adapt.hpp>
#include <samurai/mr/mesh.hpp>
#include <samurai/schemes/fv.hpp>

namespace fs = std::filesystem;

#include <numbers>

#include "convection_nonlin_os.hpp"

// #include "convection_nonlinear_osmp.hpp"

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

void check_diff(auto& field, auto& lca_left, auto& lca_right)
{
    auto& mesh      = field.mesh();
    using mesh_id_t = typename std::decay_t<decltype(mesh)>::mesh_id_t;
    samurai::for_each_level(
        mesh,
        [&](auto level)
        {
            auto set = samurai::difference(samurai::intersection(mesh[mesh_id_t::cells][level], self(lca_left).on(level)),
                                           samurai::translate(samurai::intersection(mesh[mesh_id_t::cells][level], self(lca_right).on(level)),
                                                              xt::xtensor_fixed<int, xt::xshape<1>>{-(1 << level)}));
            set(
                [&](auto& i, auto)
                {
                    std::cout << "Difference found !! " << level << " " << i << "\n";
                    auto level_ = samurai::make_scalar_field<std::size_t>("level", mesh);
                    samurai::for_each_cell(mesh,
                                           [&](const auto& cell)
                                           {
                                               level_[cell] = cell.level;
                                           });
                    samurai::save("mesh_throw", mesh, field, level_);
                    throw std::runtime_error("Difference found in check_diff function for the mesh");
                });
            auto set_field = samurai::intersection(mesh[mesh_id_t::cells][level], self(lca_left).on(level));
            set_field(
                [&](auto i, auto)
                {
                    if (xt::any(xt::abs(field(level, i) - field(level, i + (1 << level))) > 1e-13))
                    {
                        std::cout << fmt::format("\nDifference found at level {} on interval {}:\n", level, i);
                        std::cout << fmt::format("\tleft = {}\n", field(level, i));
                        std::cout << fmt::format("\tright = {}\n", field(level, i + (1 << level)));
                        std::cout << fmt::format("\terror = {}\n", xt::abs(field(level, i) - field(level, i + (1 << level))));
                        std::cout << mesh << std::endl;
                        auto level_ = samurai::make_scalar_field<std::size_t>("level", mesh);
                        samurai::for_each_cell(mesh,
                                               [&](const auto& cell)
                                               {
                                                   level_[cell] = cell.level;
                                               });
                        samurai::save("mesh_throw", mesh, field, level_);
                        throw std::runtime_error("Difference found in check_diff function for the field values");
                    }
                });
        });
}

int main(int argc, char* argv[])
{
    auto& app = samurai::initialize("Finite volume example for the linear convection equation", argc, argv);

    static constexpr std::size_t dim = 1;
    using Config                     = samurai::MRConfig<dim, 2, 2>;
    using Box                        = samurai::Box<double, dim>;
    using point_t                    = typename Box::point_t;

    std::cout << "------------------------- Burgers 1D -------------------------" << std::endl;

    //--------------------//
    // Program parameters //
    //--------------------//

    // Simulation parameters
    double left_box  = -1.;
    double right_box = 3.;

    // Time integration
    double Tf  = 0.5;
    double dt  = 0;
    double cfl = 0.5;
    double t   = 0.;

    // Multiresolution parameters
    std::size_t min_level = 10;
    std::size_t max_level = 10;

    // Output parameters
    fs::path path        = fs::current_path();
    std::string filename = "burgers";
    std::size_t nfiles   = 0;

    app.add_option("--left", left_box, "The left border of the box")->capture_default_str()->group("Simulation parameters");
    app.add_option("--right", right_box, "The right border of the box")->capture_default_str()->group("Simulation parameters");
    app.add_option("--Ti", t, "Initial time")->capture_default_str()->group("Simulation parameters");
    app.add_option("--Tf", Tf, "Final time")->capture_default_str()->group("Simulation parameters");
    app.add_option("--dt", dt, "Time step")->capture_default_str()->group("Simulation parameters");
    app.add_option("--cfl", cfl, "The CFL")->capture_default_str()->group("Simulation parameters");
    app.add_option("--min-level", min_level, "Minimum level of the multiresolution")->capture_default_str()->group("Multiresolution");
    app.add_option("--max-level", max_level, "Maximum level of the multiresolution")->capture_default_str()->group("Multiresolution");
    app.add_option("--path", path, "Output path")->capture_default_str()->group("Output");
    app.add_option("--filename", filename, "File name prefix")->capture_default_str()->group("Output");
    app.add_option("--nfiles", nfiles, "Number of output files")->capture_default_str()->group("Output");
    app.allow_extras();

    SAMURAI_PARSE(argc, argv);

    std::cout << "  max_level = " << max_level << "   min_level = " << min_level << std::endl;

    //--------------------//
    // Problem definition //
    //--------------------//

    point_t box_corner1, box_corner2;
    box_corner1.fill(left_box);
    box_corner2.fill(right_box);
    Box box(box_corner1, box_corner2);

    Box box_left(box_corner1, 0.5 * (box_corner1 + box_corner2));
    Box box_right(0.5 * (box_corner1 + box_corner2), box_corner2);

    samurai::LevelCellArray<1> lca_left(max_level, box_left, {-1}, 0.05, 2);
    samurai::LevelCellArray<1> lca_right(max_level, box_right, {-1}, 0.05, 2);

    std::array<bool, dim> periodic;
    periodic.fill(true);
    samurai::MRMesh<Config> mesh{box, min_level, max_level, periodic, 0.05, 2};

    auto u    = samurai::make_scalar_field<double>("u", mesh);
    auto unp1 = samurai::make_scalar_field<double>("unp1", mesh);

    // Initial solution
    samurai::for_each_cell(mesh,
                           [&](auto& cell)
                           {
                               u[cell] = 0.5 * (1. + std::sin(std::numbers::pi * (cell.center(0) - 1.)));
                           });

    auto MRadaptation = samurai::make_MRAdapt(u);
    auto mra_config   = samurai::mra_config().epsilon(1e-3);
    MRadaptation(mra_config);

    double dt_save    = nfiles == 0 ? dt : Tf / static_cast<double>(nfiles);
    std::size_t nsave = 0, nt = 0;
    if (nfiles != 1)
    {
        std::string suffix = (nfiles != 1) ? fmt::format("_level_{}_{}_ite_{}", min_level, max_level, nsave) : "";
        save(path, filename, u, suffix);
        nsave++;
    }

    // Convection operator
    xt::xtensor_fixed<double, xt::xshape<dim>> velocity = {1.};
    static constexpr std::size_t order                  = 2;
    auto conv                                           = samurai::make_convection_os<decltype(u), order>(dt);

    //--------------------//
    //   Time iteration   //
    //--------------------//

    double dx = mesh.cell_length(max_level);
    dt        = cfl * dx / velocity(0);

    while (t != Tf)
    {
        // Move to next timestep
        t += dt;
        if (t > Tf)
        {
            dt += Tf - t;
            t = Tf;
        }
        std::cout << fmt::format("iteration {}: t = {:.12f}, dt = {}", nt++, t, dt) << std::flush;

        // Mesh adaptation
        MRadaptation(mra_config);

        try
        {
            check_diff(u, lca_left, lca_right);
        }
        catch (...)
        {
            std::cout << "Exception caught in check_diff after adaptation" << std::endl;
            samurai::finalize();
            return 1;
        }

        samurai::update_ghost_mr(u);

        unp1.resize();

        unp1 = u - dt * conv(u);

        // u <-- unp1
        std::swap(u.array(), unp1.array());

        try
        {
            check_diff(u, lca_left, lca_right);
        }
        catch (...)
        {
            std::cout << "Exception caught in check_diff after integration" << std::endl;
            samurai::finalize();
            return 1;
        }

        // Save the result
        if (nfiles == 0 || t >= static_cast<double>(nsave + 1) * dt_save || t == Tf)
        {
            std::cout << "  (saving results)" << std::flush;
            std::string suffix = (nfiles != 1) ? fmt::format("_level_{}_{}_ite_{}", min_level, max_level, nsave) : "";
            save(path, filename, u, suffix);
            nsave++;
        }

        std::cout << std::endl;
    }

    samurai::finalize();
    return 0;
}
