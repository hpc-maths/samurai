// Copyright 2021 SAMURAI TEAM. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.
#include "CLI/CLI.hpp"
#include <array>

#include <xtensor/xfixed.hpp>

// Lines needed to create new BC
// -----------------------------
struct new_bc_types;
#define BC_TYPES new_bc_types

#include <samurai/samurai_config.hpp>

ADD_BC(Mybc)

struct new_bc_types
{
    template <class Field>
    using types = xtl::mpl::push_back_t<samurai::bc_types::types<Field>, Mybc<Field>>;
};

// -----------------------------

#include <samurai/algorithm.hpp>
#include <samurai/bc.hpp>
#include <samurai/field.hpp>
#include <samurai/hdf5.hpp>
#include <samurai/mr/adapt.hpp>
#include <samurai/mr/mesh.hpp>
#include <samurai/stencil_field.hpp>
#include <samurai/subset/subset_op.hpp>

#include <filesystem>
namespace fs = std::filesystem;

template <class Field>
void apply_bc_impl(Mybc<Field>& bc, std::size_t level, Field& field)
{
    constexpr int ghost_width = std::max(static_cast<int>(Field::mesh_t::config::max_stencil_width),
                                         static_cast<int>(Field::mesh_t::config::prediction_order));

    using mesh_id_t = typename Field::mesh_t::mesh_id_t;
    auto& mesh      = field.mesh()[mesh_id_t::reference];

    auto region     = bc.get_region();
    auto& direction = region.first;
    auto& lca       = region.second;
    for (std::size_t d = 0; d < direction.size(); ++d)
    {
        std::size_t delta_l = lca[d].level() - level;
        for (int ig = 0; ig < ghost_width; ++ig)
        {
            auto first_layer_ghosts = intersection(mesh[level], translate(lca[d], (ig + 1) * (direction[d] << delta_l))).on(level);
            first_layer_ghosts(
                [&](const auto& i, const auto& index)
                {
                    auto j             = index[0];
                    field(level, i, j) = 2 * bc.constant_value()
                                       - field(level, i - (2 * ig + 1) * direction[d][0], j - (2 * ig + 1) * direction[d][1]);
                });
        }
    }
}

template <class Mesh>
auto init(Mesh& mesh)
{
    auto u = samurai::make_field<double, 1>("u", mesh);

    samurai::for_each_cell(
        mesh,
        [&](auto& cell)
        {
            auto center           = cell.center();
            const double radius   = .2;
            const double x_center = 0.3;
            const double y_center = 0.3;
            if (((center[0] - x_center) * (center[0] - x_center) + (center[1] - y_center) * (center[1] - y_center)) <= radius * radius)
            {
                u[cell] = 1;
            }
            else
            {
                u[cell] = 0;
            }
        });

    return u;
}

template <class Field>
void flux_correction(double dt, const std::array<double, 2>& a, const Field& u, Field& unp1)
{
    using mesh_t              = typename Field::mesh_t;
    using mesh_id_t           = typename mesh_t::mesh_id_t;
    using interval_t          = typename mesh_t::interval_t;
    constexpr std::size_t dim = Field::dim;

    auto mesh = u.mesh();

    for (std::size_t level = mesh.min_level(); level < mesh.max_level(); ++level)
    {
        xt::xtensor_fixed<int, xt::xshape<dim>> stencil;

        stencil = {
            {-1, 0}
        };

        auto subset_right = samurai::intersection(samurai::translate(mesh[mesh_id_t::cells][level + 1], stencil),
                                                  mesh[mesh_id_t::cells][level])
                                .on(level);

        subset_right(
            [&](const auto& i, const auto& index)
            {
                auto j          = index[0];
                const double dx = samurai::cell_length(level);

                unp1(level, i, j) = unp1(level, i, j)
                                  + dt / dx
                                        * (samurai::upwind_op<interval_t>(level, i, j).right_flux(a, u)
                                           - .5 * samurai::upwind_op<interval_t>(level + 1, 2 * i + 1, 2 * j).right_flux(a, u)
                                           - .5 * samurai::upwind_op<interval_t>(level + 1, 2 * i + 1, 2 * j + 1).right_flux(a, u));
            });

        stencil = {
            {1, 0}
        };

        auto subset_left = samurai::intersection(samurai::translate(mesh[mesh_id_t::cells][level + 1], stencil),
                                                 mesh[mesh_id_t::cells][level])
                               .on(level);

        subset_left(
            [&](const auto& i, const auto& index)
            {
                auto j          = index[0];
                const double dx = samurai::cell_length(level);

                unp1(level, i, j) = unp1(level, i, j)
                                  - dt / dx
                                        * (samurai::upwind_op<interval_t>(level, i, j).left_flux(a, u)
                                           - .5 * samurai::upwind_op<interval_t>(level + 1, 2 * i, 2 * j).left_flux(a, u)
                                           - .5 * samurai::upwind_op<interval_t>(level + 1, 2 * i, 2 * j + 1).left_flux(a, u));
            });

        stencil = {
            {0, -1}
        };

        auto subset_up = samurai::intersection(samurai::translate(mesh[mesh_id_t::cells][level + 1], stencil), mesh[mesh_id_t::cells][level])
                             .on(level);

        subset_up(
            [&](const auto& i, const auto& index)
            {
                auto j          = index[0];
                const double dx = samurai::cell_length(level);

                unp1(level, i, j) = unp1(level, i, j)
                                  + dt / dx
                                        * (samurai::upwind_op<interval_t>(level, i, j).up_flux(a, u)
                                           - .5 * samurai::upwind_op<interval_t>(level + 1, 2 * i, 2 * j + 1).up_flux(a, u)
                                           - .5 * samurai::upwind_op<interval_t>(level + 1, 2 * i + 1, 2 * j + 1).up_flux(a, u));
            });

        stencil = {
            {0, 1}
        };

        auto subset_down = samurai::intersection(samurai::translate(mesh[mesh_id_t::cells][level + 1], stencil),
                                                 mesh[mesh_id_t::cells][level])
                               .on(level);

        subset_down(
            [&](const auto& i, const auto& index)
            {
                auto j          = index[0];
                const double dx = samurai::cell_length(level);

                unp1(level, i, j) = unp1(level, i, j)
                                  - dt / dx
                                        * (samurai::upwind_op<interval_t>(level, i, j).down_flux(a, u)
                                           - .5 * samurai::upwind_op<interval_t>(level + 1, 2 * i, 2 * j).down_flux(a, u)
                                           - .5 * samurai::upwind_op<interval_t>(level + 1, 2 * i + 1, 2 * j).down_flux(a, u));
            });
    }
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
    constexpr size_t dim = 2;
    using Config         = samurai::MRConfig<dim>;

    // Simulation parameters
    xt::xtensor_fixed<double, xt::xshape<dim>> min_corner = {0., 0.};
    xt::xtensor_fixed<double, xt::xshape<dim>> max_corner = {1., 1.};
    std::array<double, dim> a{
        {1, 1}
    };
    double Tf  = .1;
    double cfl = 0.5;

    // Multiresolution parameters
    std::size_t min_level = 4;
    std::size_t max_level = 10;
    double mr_epsilon     = 2.e-4; // Threshold used by multiresolution
    double mr_regularity  = 1.;    // Regularity guess for multiresolution
    bool correction       = false;

    // Output parameters
    fs::path path        = fs::current_path();
    std::string filename = "FV_advection_2d";
    std::size_t nfiles   = 1;

    CLI::App app{"Finite volume example for the advection equation in 2d "
                 "using multiresolution"};
    app.add_option("--min-corner", min_corner, "The min corner of the box")->capture_default_str()->group("Simulation parameters");
    app.add_option("--max-corner", max_corner, "The max corner of the box")->capture_default_str()->group("Simulation parameters");
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

    const samurai::Box<double, dim> box(min_corner, max_corner);
    samurai::MRMesh<Config> mesh{box, min_level, max_level};

    double dt            = cfl / (1 << max_level);
    const double dt_save = Tf / static_cast<double>(nfiles);
    double t             = 0.;

    auto u = init(mesh);
    samurai::make_bc<Mybc>(u, 0.);
    auto unp1 = samurai::make_field<double, 1>("unp1", mesh);

    auto MRadaptation = samurai::make_MRAdapt(u);
    MRadaptation(mr_epsilon, mr_regularity);
    save(path, filename, u, "_init");

    std::size_t nsave = 1;
    std::size_t nt    = 0;

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
        unp1.resize();
        unp1 = u - dt * samurai::upwind(a, u);
        if (correction)
        {
            flux_correction(dt, a, u, unp1);
        }

        std::swap(u.array(), unp1.array());

        if (t >= static_cast<double>(nsave + 1) * dt_save || t == Tf)
        {
            const std::string suffix = (nfiles != 1) ? fmt::format("_ite_{}", nsave++) : "";
            save(path, filename, u, suffix);
        }
    }
    return 0;
}
