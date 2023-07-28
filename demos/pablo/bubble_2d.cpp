// Copyright 2021 SAMURAI TEAM. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.
#include "CLI/CLI.hpp"
#include <iostream>

#include <filesystem>

#include <xtensor/xadapt.hpp>
#include <xtensor/xrandom.hpp>
#include <xtensor/xtensor.hpp>

#include <samurai/box.hpp>
#include <samurai/cell_array.hpp>
#include <samurai/cell_flag.hpp>
#include <samurai/field.hpp>
#include <samurai/hdf5.hpp>
#include <samurai/subset/subset_op.hpp>

namespace fs = std::filesystem;

template <class Mesh, class Container>
void update_mesh(Mesh& mesh,
                 std::size_t min_level,
                 std::size_t max_level,
                 const Container& bb_xcenter,
                 const Container& bb_ycenter,
                 const Container& bb_radius)
{
    constexpr std::size_t dim = Mesh::dim;
    std::size_t nb_bubbles    = bb_xcenter.shape(0);

    auto tag = samurai::make_field<int, 1>("tag", mesh);
    tag.fill(static_cast<int>(samurai::CellFlag::keep));

    samurai::for_each_cell(
        mesh,
        [&](auto cell)
        {
            bool inside    = false;
            std::size_t ib = 0;

            while (!inside && ib < nb_bubbles)
            {
                const double xc     = bb_xcenter[ib];
                const double yc     = bb_ycenter[ib];
                const double radius = bb_radius[ib];

                auto corner     = cell.corner();
                auto center     = cell.center();
                const double dx = cell.length;

                for (std::size_t i = 0; i < 2; ++i)
                {
                    const double x = corner[0] + static_cast<double>(i) * dx;
                    for (std::size_t j = 0; j < 2; ++j)
                    {
                        const double y = corner[1] + static_cast<double>(j) * dx;
                        if (((!inside)
                             && (std::pow((x - xc), 2.0) + pow((y - yc), 2.0) <= 1.25 * std::pow(radius, 2.0)
                                 && std::pow((x - xc), 2.0) + pow((y - yc), 2.0) >= 0.75 * std::pow(radius, 2.0)))
                            || ((!inside)
                                && (std::pow((center[0] - xc), 2.0) + std::pow((center[1] - yc), 2.0) <= 1.25 * std::pow(radius, 2.0)
                                    && std::pow((center[0] - xc), 2.0) + std::pow((center[1] - yc), 2.0) >= 0.75 * std::pow(radius, 2.0))))
                        {
                            if (cell.level < max_level)
                            {
                                tag[cell] = static_cast<int>(samurai::CellFlag::refine);
                            }
                            inside = true;
                        }
                    }
                }
                ib++;
            }

            if (cell.level > min_level && !inside)
            {
                tag[cell] = static_cast<int>(samurai::CellFlag::coarsen);
            }
        });

    samurai::CellList<dim> cell_list;

    samurai::for_each_interval(mesh,
                               [&](std::size_t level, const auto& interval, const auto& index_yz)
                               {
                                   auto itag = interval.start + interval.index;
                                   for (int i = interval.start; i < interval.end; ++i)
                                   {
                                       if (tag[itag] & static_cast<int>(samurai::CellFlag::refine))
                                       {
                                           samurai::static_nested_loop<dim - 1, 0, 2>(
                                               [&](auto stencil)
                                               {
                                                   auto index = 2 * index_yz + stencil;
                                                   cell_list[level + 1][index].add_interval({2 * i, 2 * i + 2});
                                               });
                                       }
                                       else if (tag[itag] & static_cast<int>(samurai::CellFlag::keep))
                                       {
                                           cell_list[level][index_yz].add_point(i);
                                       }
                                       else
                                       {
                                           cell_list[level - 1][index_yz >> 1].add_point(i >> 1);
                                       }
                                       itag++;
                                   }
                               });

    mesh = {cell_list, true};
}

template <std::size_t dim>
void remove_intersection(samurai::CellArray<dim>& ca)
{
    auto min_level = ca.min_level();
    auto max_level = ca.max_level();

    while (true)
    {
        auto tag = samurai::make_field<bool, 1>("tag", ca);
        tag.fill(false);

        for (std::size_t level = min_level + 1; level <= max_level; ++level)
        {
            for (std::size_t level_below = min_level; level_below < level; ++level_below)
            {
                auto set = samurai::intersection(ca[level], ca[level_below]).on(level_below);
                set(
                    [&](const auto& i, const auto& index)
                    {
                        tag(level_below, i, index[0]) = true;
                    });
            }
        }

        samurai::CellList<dim> cl;
        samurai::for_each_cell(ca,
                               [&](auto cell)
                               {
                                   auto i = cell.indices[0];
                                   auto j = cell.indices[1];
                                   if (tag[cell])
                                   {
                                       cl[cell.level + 1][{2 * j}].add_interval({2 * i, 2 * i + 2});
                                       cl[cell.level + 1][{2 * j + 1}].add_interval({2 * i, 2 * i + 2});
                                   }
                                   else
                                   {
                                       cl[cell.level][{j}].add_point(i);
                                   }
                               });
        samurai::CellArray<dim> new_ca = {cl, true};

        if (new_ca == ca)
        {
            break;
        }

        std::swap(ca, new_ca);
    }
}

template <std::size_t dim>
void make_graduation(samurai::CellArray<dim>& ca)
{
    auto min_level = ca.min_level();
    auto max_level = ca.max_level();
    // xt::xtensor_fixed<int, xt::xshape<4, dim>> stencil{{1, 0}, {-1, 0}, {0,
    // 1}, {0, -1}};
    xt::xtensor_fixed<int, xt::xshape<4, dim>> stencil{
        {1,  1 },
        {-1, -1},
        {-1, 1 },
        {1,  -1}
    };
    while (true)
    {
        auto tag = samurai::make_field<bool, 1>("tag", ca);
        tag.fill(false);

        for (std::size_t level = min_level + 2; level <= max_level; ++level)
        {
            for (std::size_t level_below = min_level; level_below < level - 1; ++level_below)
            {
                for (std::size_t is = 0; is < stencil.shape()[0]; ++is)
                {
                    auto s   = xt::view(stencil, is);
                    auto set = samurai::intersection(samurai::translate(ca[level], s), ca[level_below]).on(level_below);
                    set(
                        [&](const auto& i, const auto& index)
                        {
                            tag(level_below, i, index[0]) = true;
                        });
                }
            }
        }

        samurai::CellList<dim> cl;
        samurai::for_each_cell(ca,
                               [&](auto cell)
                               {
                                   auto i = cell.indices[0];
                                   auto j = cell.indices[1];
                                   if (tag[cell])
                                   {
                                       cl[cell.level + 1][{2 * j}].add_interval({2 * i, 2 * i + 2});
                                       cl[cell.level + 1][{2 * j + 1}].add_interval({2 * i, 2 * i + 2});
                                   }
                                   else
                                   {
                                       cl[cell.level][{j}].add_point(i);
                                   }
                               });
        samurai::CellArray<dim> new_ca = {cl, true};

        if (new_ca == ca)
        {
            break;
        }

        std::swap(ca, new_ca);
    }
}

int main(int argc, char* argv[])
{
    constexpr std::size_t dim = 2;

    // Simulation parameters
    std::vector<double> min_corner_v = {0., 0.};
    std::vector<double> max_corner_v = {1., 1.};
    std::size_t nb_bubbles           = 10;
    double Tf                        = 100;
    double dt                        = 0.5;

    // Adaptation parameters
    std::size_t start_level = 4;
    std::size_t min_level   = 1;
    std::size_t max_level   = 9;

    // Output parameters
    fs::path path        = fs::current_path();
    std::string filename = "bubble_2d";
    std::size_t nfiles   = 1;

    CLI::App app{"2d bubble example from pablo (see "
                 "https://github.com/optimad/bitpit/blob/master/examples/"
                 "PABLO_bubbles_2D.cpp)"};
    app.add_option("--min-corner", min_corner_v, "The min corner of the box")->capture_default_str()->group("Simulation parameters");
    app.add_option("--max-corner", max_corner_v, "The max corner of the box")->capture_default_str()->group("Simulation parameters");
    app.add_option("--nb-bubbles", nb_bubbles, "Number of bubbles")->capture_default_str()->group("Simulation parameters");
    app.add_option("--Tf", Tf, "Final time")->capture_default_str()->group("Simulation parameters");
    app.add_option("--dt", dt, "Time step")->capture_default_str()->group("Simulation parameters");
    app.add_option("--start-level", start_level, "Start level of AMR")->capture_default_str()->group("Adaptation parameters");
    app.add_option("--min-level", min_level, "Minimum level of AMR")->capture_default_str()->group("Adaptation parameters");
    app.add_option("--max-level", max_level, "Maximum level of AMR")->capture_default_str()->group("Adaptation parameters");
    app.add_option("--path", path, "Output path")->capture_default_str()->group("Ouput");
    app.add_option("--filename", filename, "File name prefix")->capture_default_str()->group("Ouput");
    app.add_option("--nfiles", nfiles, "Number of output files")->capture_default_str()->group("Ouput");
    CLI11_PARSE(app, argc, argv);

    if (!fs::exists(path))
    {
        fs::create_directory(path);
    }

    auto min_corner(xt::adapt(min_corner_v));
    auto max_corner(xt::adapt(max_corner_v));
    const double x_length = max_corner(0) - min_corner(0);
    const double y_length = max_corner(1) - min_corner(1);

    xt::random::seed(42);
    using container_t             = xt::xtensor<double, 1>;
    container_t bb_xcenter        = 0.8 * x_length * xt::random::rand<double>({nb_bubbles}) + 0.1 * x_length;
    const container_t bb0_xcenter = bb_xcenter;
    container_t bb_ycenter        = y_length * xt::random::rand<double>({nb_bubbles}) - 0.5 * y_length;
    const container_t bb_radius   = 0.1 * xt::random::rand<double>({nb_bubbles}) + 0.02;
    const container_t dy          = 0.005 + 0.05 * xt::random::rand<double>({nb_bubbles});
    const container_t omega       = 0.5 * xt::random::rand<double>({nb_bubbles});
    const container_t aa          = 0.15 * xt::random::rand<double>({nb_bubbles});

    samurai::CellArray<dim> mesh;

    const samurai::Box<double, dim> box(min_corner, max_corner);
    mesh[start_level] = {start_level, box};

    const double dt_save = Tf / static_cast<double>(nfiles);
    double t             = 0.;

    std::size_t nsave = 1;
    std::size_t nt    = 0;

    while (t != Tf)
    {
        t += dt;
        if (t > Tf)
        {
            dt += Tf - t;
            t = Tf;
        }

        bb_xcenter = bb0_xcenter + aa * xt::cos(omega * t);
        bb_ycenter = bb_ycenter + dt * dy;

        std::cout << fmt::format("iteration -> {} t -> {}", nt++, t) << std::endl;

        for (std::size_t rep = 0; rep < 10; ++rep)
        {
            update_mesh(mesh, min_level, max_level, bb_xcenter, bb_ycenter, bb_radius);
            remove_intersection(mesh);
            make_graduation(mesh);
        }

        if (t >= static_cast<double>(nsave + 1) * dt_save || t == Tf)
        {
            std::string suffix = (nfiles != 1) ? fmt::format("_ite_{}", nsave++) : "";
            samurai::save(path, fmt::format("{}{}", filename, suffix), mesh);
        }
    }
    return 0;
}
