// Copyright 2021 SAMURAI TEAM. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.
#include "CLI/CLI.hpp"
#include <iostream>

#include <filesystem>

#include <samurai/cell_array.hpp>
#include <samurai/cell_list.hpp>
#include <samurai/field.hpp>
#include <samurai/hdf5.hpp>
#include <samurai/memory.hpp>
#include <samurai/mr/mesh.hpp>
#include <samurai/subset/subset_op.hpp>

namespace fs = std::filesystem;

/// Timer used in tic & toc
auto tic_timer = std::chrono::high_resolution_clock::now(); // NOLINT

/// Launching the timer
void tic()
{
    tic_timer = std::chrono::high_resolution_clock::now();
}

/// Stopping the timer and returning the duration in seconds
double toc()
{
    const auto toc_timer                          = std::chrono::high_resolution_clock::now();
    const std::chrono::duration<double> time_span = toc_timer - tic_timer;
    return time_span.count();
}

template <class mesh_t>
void refine_1(mesh_t& mesh, std::size_t max_level)
{
    constexpr std::size_t dim = mesh_t::dim;
    using cl_type             = typename mesh_t::cl_type;
    using coord_index_t       = typename mesh_t::interval_t::coord_index_t;

    for (std::size_t ite = 0; ite < max_level; ++ite)
    {
        auto cell_tag = samurai::make_field<bool, 1>("tag", mesh);
        cell_tag.fill(false);

        samurai::for_each_cell(mesh,
                               [&](auto cell)
                               {
                                   auto corner = cell.corner();
                                   auto x      = corner[0];
                                   auto y      = corner[1];

                                   if (cell.level < max_level)
                                   {
                                       if (x < 0.25 or (x == 0.75 and y == 0.75))
                                       {
                                           cell_tag[cell] = true;
                                       }
                                   }
                               });

        cl_type cl;
        samurai::for_each_interval(mesh,
                                   [&](std::size_t level, const auto& interval, const auto& index_yz)
                                   {
                                       auto itag = interval.start + interval.index;
                                       for (coord_index_t i = interval.start; i < interval.end; ++i)
                                       {
                                           if (cell_tag[itag])
                                           {
                                               samurai::static_nested_loop<dim - 1, 0, 2>(
                                                   [&](auto stencil)
                                                   {
                                                       auto index = 2 * index_yz + stencil;
                                                       cl[level + 1][index].add_interval({2 * i, 2 * i + 2});
                                                   });
                                           }
                                           else
                                           {
                                               cl[level][index_yz].add_point(i);
                                           }
                                           itag++;
                                       }
                                   });

        mesh = {cl};
    }
}

template <class mesh_t>
void refine_2(mesh_t& mesh, std::size_t max_level)
{
    constexpr std::size_t dim = mesh_t::dim;
    using mesh_id_t           = typename mesh_t::mesh_id_t;
    using cl_type             = typename mesh_t::cl_type;
    using coord_index_t       = typename mesh_t::interval_t::coord_index_t;

    for (std::size_t ite = 0; ite < max_level; ++ite)
    {
        auto cell_tag = samurai::make_field<bool, 1>("tag", mesh);
        cell_tag.fill(false);

        samurai::for_each_cell(mesh,
                               [&](auto cell)
                               {
                                   auto corner = cell.corner();
                                   auto x      = corner[0];
                                   auto y      = corner[1];

                                   if (cell.level < max_level)
                                   {
                                       if (x < 0.25 or (x == 0.75 and y == 0.75))
                                       {
                                           cell_tag[cell] = true;
                                       }
                                   }
                               });

        // graduation
        for (std::size_t level = max_level; level > 1; --level)
        {
            // xt::xtensor_fixed<int, xt::xshape<4, dim>> stencil{{1, 0}, {-1,
            // 0}, {0, 1}, {0, -1}};
            xt::xtensor_fixed<int, xt::xshape<4, dim>> stencil{
                {1,  1 },
                {-1, -1},
                {-1, 1 },
                {1,  -1}
            };

            for (std::size_t is = 0; is < stencil.shape()[0]; ++is)
            {
                auto s      = xt::view(stencil, is);
                auto subset = samurai::intersection(samurai::translate(mesh[mesh_id_t::cells][level], s), mesh[mesh_id_t::cells][level - 1])
                                  .on(level);

                subset(
                    [&](const auto& interval, const auto& index)
                    {
                        auto j_f = index[0];
                        auto i_f = interval.even_elements();

                        if (i_f.is_valid())
                        {
                            auto i_c = i_f >> 1;
                            auto j_c = j_f >> 1;
                            cell_tag(level - 1, i_c, j_c) |= cell_tag(level, i_f - s[0], j_f - s[1]);
                        }

                        i_f = interval.odd_elements();
                        if (i_f.is_valid())
                        {
                            auto i_c = i_f >> 1;
                            auto j_c = j_f >> 1;

                            cell_tag(level - 1, i_c, j_c) |= cell_tag(level, i_f - s[0], j_f - s[1]);
                        }
                    });
            }
        }

        cl_type cl;
        samurai::for_each_interval(mesh[mesh_id_t::cells],
                                   [&](std::size_t level, const auto& interval, const auto& index_yz)
                                   {
                                       auto itag = interval.start + interval.index;
                                       for (coord_index_t i = interval.start; i < interval.end; ++i)
                                       {
                                           if (cell_tag[itag])
                                           {
                                               samurai::static_nested_loop<dim - 1, 0, 2>(
                                                   [&](auto stencil)
                                                   {
                                                       auto index = 2 * index_yz + stencil;
                                                       cl[level + 1][index].add_interval({2 * i, 2 * i + 2});
                                                   });
                                           }
                                           else
                                           {
                                               cl[level][index_yz].add_point(i);
                                           }
                                           itag++;
                                       }
                                   });

        mesh = {cl, mesh.min_level(), mesh.max_level()};
    }
}

int main(int argc, char* argv[])
{
    constexpr size_t dim = 2;

    // Adaptation parameters
    std::size_t max_level = 9;

    // Output parameters
    fs::path path        = fs::current_path();
    std::string filename = "simple_2d";

    CLI::App app{"simple 2d p4est example (see "
                 "https://github.com/cburstedde/p4est/blob/master/example/"
                 "simple/simple2.c)"};
    app.add_option("--max-level", max_level, "Maximum level of the adaptation")->capture_default_str()->group("Adaptation parameters");
    app.add_option("--path", path, "Output path")->capture_default_str()->group("Ouput");
    app.add_option("--filename", filename, "File name prefix")->capture_default_str()->group("Ouput");
    CLI11_PARSE(app, argc, argv);

    if (!fs::exists(path))
    {
        fs::create_directory(path);
    }

    samurai::CellList<dim> cl;

    cl[1][{0}].add_interval({1, 2});
    cl[1][{1}].add_interval({0, 1});

    cl[2][{0}].add_interval({0, 2});
    cl[2][{1}].add_interval({0, 2});
    cl[2][{2}].add_interval({2, 4});
    cl[2][{3}].add_interval({2, 4});

    samurai::CellArray<dim> mesh_1(cl);
    std::cout << "nb_cells: " << mesh_1.nb_cells() << "\n";

    tic();
    refine_1(mesh_1, max_level);
    auto duration = toc();
    std::cout << "Version 1: " << duration << "s" << std::endl;
    toc();
    std::cout << "nb_cells: " << mesh_1.nb_cells() << "\n";

    // samurai::CellArray<dim> mesh_2(cl);
    using Config    = samurai::MRConfig<dim>;
    using mesh_id_t = typename samurai::MRMesh<Config>::mesh_id_t;
    samurai::MRMesh<Config> mesh_2(cl, 1, max_level);

    tic();
    refine_2(mesh_2, max_level);
    duration = toc();
    std::cout << "Version 2: " << duration << "s" << std::endl;
    std::cout << "nb_cells: " << mesh_2.nb_cells(mesh_id_t::cells) << "\n";

    std::cout << "Memory used " << std::endl;
    auto mem = samurai::memory_usage(mesh_2, /*verbose*/ true);
    std::cout << "Total: " << mem << std::endl;

    auto level = samurai::make_field<std::size_t, 1>("level", mesh_2);
    samurai::for_each_cell(mesh_2,
                           [&](auto cell)
                           {
                               level[cell] = cell.level;
                           });
    samurai::save(path, filename, mesh_2, level);

    return 0;
}
