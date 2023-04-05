// Copyright 2021 SAMURAI TEAM. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.
#include "CLI/CLI.hpp"

#include <xtensor/xfixed.hpp>

#include <samurai/algorithm.hpp>
#include <samurai/bc.hpp>
#include <samurai/field.hpp>
#include <samurai/hdf5.hpp>
#include <samurai/mr/adapt.hpp>
#include <samurai/mr/mesh.hpp>
#include <samurai/reconstruction.hpp>
#include <samurai/stencil_field.hpp>
#include <samurai/subset/subset_op.hpp>

#include <filesystem>
namespace fs = std::filesystem;

template <class Mesh>
auto init(Mesh& mesh, double t = 0)
{
    double dx = 1. / (1 << mesh.max_level());
    auto u    = samurai::make_field<double, 1>("u", mesh);
    u.fill(0.);

    samurai::for_each_cell(mesh,
                           [&](auto& cell)
                           {
                               auto center   = cell.center();
                               double radius = std::floor(.2 / dx) * dx;

                               // if (xt::sum(center*center)[0] <=
                               // radius*radius)
                               // {
                               //     u[cell] = 1;
                               // }
                               if (xt::all(xt::abs(center - t * dx) <= radius))
                               {
                                   u[cell] = 1;
                               }
                           });

    return u;
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

    // samurai::save(path, fmt::format("{}{}", filename, suffix), {true, true},
    // mesh, u, level_);
    samurai::save(path, fmt::format("{}{}", filename, suffix), mesh, u, level_);
}

int main(int argc, char* argv[])
{
    constexpr size_t dim = 1;
    using Config         = samurai::MRConfig<dim, 2>;

    // Simulation parameters
    xt::xtensor_fixed<double, xt::xshape<dim>> min_corner, max_corner;
    min_corner.fill(-1);
    max_corner.fill(1);

    bool is_periodic = false;
    double a         = 1.;
    double Tf        = 1.;
    double cfl       = 0.95;

    // Multiresolution parameters
    std::size_t min_level = 4, max_level = 10;
    double mr_epsilon    = 2.e-4; // Threshold used by multiresolution
    double mr_regularity = 1.;    // Regularity guess for multiresolution
    bool correction      = false;

    // Output parameters
    fs::path path        = fs::current_path();
    std::string filename = "FV_advection_1d";
    std::size_t nfiles   = 1;

    CLI::App app{"Finite volume example for the advection equation in 1d "
                 "using multiresolution"};
    app.add_option("--min-corner", min_corner, "The min corner of the box")->capture_default_str()->group("Simulation parameters");
    app.add_option("--max-corner", min_corner, "The max corner of the box")->capture_default_str()->group("Simulation parameters");
    app.add_flag("--periodic", is_periodic, "Set the domain periodic")->capture_default_str()->group("Simulation parameters");
    app.add_option("--velocity", a, "The velocity of the advection equation")->capture_default_str()->group("Simulation parameters");
    app.add_option("--cfl", cfl, "The CFL")->capture_default_str()->group("Simulation parameters");
    app.add_option("--Tf", Tf, "Final time")->capture_default_str()->group("Simulation parameters");
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
    app.add_option("--with-correction", correction, "Apply flux correction at the interface of two refinement levels")
        ->capture_default_str()
        ->group("Multiresolution");
    app.add_option("--path", path, "Output path")->capture_default_str()->group("Ouput");
    app.add_option("--filename", filename, "File name prefix")->capture_default_str()->group("Ouput");
    app.add_option("--nfiles", nfiles, "Number of output files")->capture_default_str()->group("Ouput");
    CLI11_PARSE(app, argc, argv);

    samurai::Box<double, dim> box(min_corner, max_corner);
    // samurai::MRMesh<Config> mesh{box, min_level, max_level, {true, false}};
    samurai::MRMesh<Config> mesh{box, min_level, max_level, {true}};
    using mesh_id_t = typename samurai::MRMesh<Config>::mesh_id_t;

    double dt      = 1;
    Tf             = 4 * (1 << max_level);
    double dt_save = Tf / static_cast<double>(nfiles);
    double t       = 0.;

    auto u    = init(mesh);
    auto unp1 = samurai::make_field<double, 1>("unp1", mesh);
    unp1.fill(0);
    auto MRadaptation = samurai::make_MRAdapt(u);
    MRadaptation(mr_epsilon, mr_regularity);
    save(path, filename, u, "_init");

    std::size_t nsave = 1, nt = 0;

    while (t != Tf)
    {
        MRadaptation(mr_epsilon, mr_regularity);

        t += dt;
        if (t > Tf)
        {
            dt += Tf - t;
            t = Tf;
        }

        std::cout << fmt::format("iteration {}: t = {}, dt = {}", nt++, t, dt) << std::endl;

        samurai::update_ghost_mr(u);
        std::string suffix = fmt::format("_before_ite_{}", t);
        save(path, filename, u, suffix);
        unp1.resize();
        samurai::for_each_interval(mesh[mesh_id_t::cells],
                                   [&](std::size_t level, auto& i, auto& index)
                                   {
                                       std::size_t shift = max_level - level;
                                       if constexpr (dim == 1)
                                       {
                                           unp1(level, i) = u(level, i) + samurai::portion(u, level, i - 1, shift, (1 << shift) - 1)
                                                          - samurai::portion(u, level, i, shift, (1 << shift) - 1);
                                       }
                                       else if constexpr (dim == 2)
                                       {
                                           auto j                           = index[0];
                                           xt::xtensor<double, 1> to_add    = xt::zeros<double>(std::array<std::size_t, 1>{i.size()});
                                           xt::xtensor<double, 1> to_remove = xt::zeros<double>(std::array<std::size_t, 1>{i.size()});
                                           for (int jj = 0; jj < (1 << shift); ++jj)
                                           {
                                               to_add += samurai::portion(u, level, i - 1, j, shift, (1 << shift) - 1, jj);
                                               to_remove += samurai::portion(u, level, i, j, shift, (1 << shift) - 1, jj);
                                           }
                                           unp1(level, i, j) = u(level, i, j) + to_add - to_remove;
                                       }
                                   });

        std::swap(u.array(), unp1.array());

        if (t >= static_cast<double>(nsave + 1) * dt_save || t == Tf)
        {
            std::string suffix = (nfiles != 1) ? fmt::format("_ite_{}", nsave++) : "";
            save(path, filename, u, suffix);
        }
    }
    MRadaptation(mr_epsilon, mr_regularity);

    samurai::MRMesh<Config> mesh_init{box, min_level, max_level, {true}};

    auto u_init            = init(mesh_init);
    auto MRadaptation_init = samurai::make_MRAdapt(u_init);
    MRadaptation_init(mr_epsilon, mr_regularity);

    save(path, filename, u_init, "_init_adapt");

    auto error = samurai::make_field<double, 1>("error", mesh);
    if (u != u_init)
    {
        samurai::for_each_cell(mesh,
                               [&](auto& cell)
                               {
                                   error[cell] = std::abs(u[cell] - u_init[cell]);
                               });
        std::cout << error << std::endl;
    }

    return 0;
}