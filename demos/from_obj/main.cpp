// Copyright 2021 SAMURAI TEAM. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.
#include "CLI/CLI.hpp"

#include <limits>
#include <tuple>

#include <CGAL/optimal_bounding_box.h>
#include <CGAL/Polygon_mesh_processing/IO/polygon_mesh_io.h>
#include <CGAL/polygon_mesh_processing.h>
#include <CGAL/Side_of_triangle_mesh.h>
#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
#include <CGAL/Polyhedron_3.h>

#include <samurai/box.hpp>
#include <samurai/cell_array.hpp>
#include <samurai/mr/mesh.hpp>
#include <samurai/field.hpp>
#include <samurai/hdf5.hpp>
#include <samurai/static_algorithm.hpp>
#include <samurai/algorithm/graduation.hpp>

namespace PMP = CGAL::Polygon_mesh_processing;
using Kernel = CGAL::Exact_predicates_inexact_constructions_kernel;

using K = CGAL::Exact_predicates_inexact_constructions_kernel;
using Point = typename Kernel::Point_3;
using Polyhedron = CGAL::Polyhedron_3<K>;

template <std::size_t dim>
auto init_mesh(std::size_t start_level, const Polyhedron& sm)
{
    std::array<Point, 8> obb_points;
    CGAL::oriented_bounding_box(sm, obb_points,
                                CGAL::parameters::use_convex_hull(true));

    xt::xtensor_fixed<double, xt::xshape<dim>> min_corner;
    xt::xtensor_fixed<double, xt::xshape<dim>> max_corner;
    min_corner.fill(std::numeric_limits<double>::max());
    max_corner.fill(std::numeric_limits<double>::min());

    double dx = 1./(1 << start_level);
    for(auto& p: obb_points)
    {
        for(std::size_t d = 0; d < dim; ++d)
        {
            double v = p[static_cast<int>(d)];
            if (min_corner[d] > v)
            {
                min_corner[d] = v;
            }
            if (max_corner[d] < v)
            {
                max_corner[d] = v;
            }
        }
    }

    double scale = max_corner[0] - min_corner[0];
    for(std::size_t d = 1; d < dim; ++d)
    {
        double tmp = max_corner[d] - min_corner[d];
        if (scale > tmp)
        {
            scale = tmp;
        }
    }
    if (scale > 1)
    {
        min_corner /= scale;
        max_corner /= scale;
    }
    else
    {
        scale = 1;
    }

    min_corner -= 2*dx;
    max_corner += 2*dx;

    std::cout << "scale factor " << scale << std::endl;

    samurai::Box<double, dim> box(min_corner, max_corner);
    samurai::CellArray<dim> ca;
    ca[start_level] = {start_level, box};

    return std::make_pair(scale, ca);
}

template <class Mesh>
void save_mesh(const fs::path& path, const std::string& filename, const Mesh& mesh)
{
    auto level = samurai::make_field<std::size_t, 1>("level", mesh);

    if (!fs::exists(path))
    {
        fs::create_directory(path);
    }

    samurai::for_each_cell(mesh, [&](auto &cell)
    {
        level[cell] = cell.level;
    });

    samurai::save(path, filename, mesh, level);
}

int main(int argc, char** argv)
{
    constexpr std::size_t dim = 2;
    std::size_t start_level = 2;
    std::size_t max_level = 8;

    // Output parameters
    fs::path path = fs::current_path();
    std::string input_file;

    CLI::App app{"Create an adapted mesh from an OBJ file"};
    app.add_option("--input", input_file, "input File")->required()->check(CLI::ExistingFile);;
    app.add_option("--start-level", start_level, "Start level of the output adaptive mesh")->capture_default_str();
    app.add_option("--max-level", max_level, "Maximum level of the output adaptive mesh")->capture_default_str();
    app.add_option("--path", path, "Output path")->capture_default_str();
    CLI11_PARSE(app, argc, argv);

    std::string output_file = fs::path(input_file).stem();

    Polyhedron sm;
    if(!PMP::IO::read_polygon_mesh(input_file, sm) || sm.is_empty())
    {
        std::cerr << "Invalid input file." << std::endl;
        return EXIT_FAILURE;
    }

    CGAL::Side_of_triangle_mesh<Polyhedron, K> inside(sm);

    samurai::CellArray<dim> ca;
    using interval_t = typename samurai::CellArray<dim>::interval_t;
    double scale;
    std::tie(scale, ca) = init_mesh<dim>(start_level, sm);

    std::size_t current_level = start_level;
    std::size_t ite = 0;
    while(current_level != max_level)
    {
        std::cout << "iteration: " << ite << std::endl;
        std::cout << "Number of cells " <<  ca.nb_cells() << std::endl;

        auto tag = samurai::make_field<int, 1>("tag", ca);
        tag.fill(static_cast<int>(samurai::CellFlag::keep));

        samurai::for_each_cell(ca[current_level], [&](auto cell)
        {
            auto center = scale*cell.center();
            auto corner = scale*cell.corner();
            double dx = scale*cell.length;
            std::vector<Point> points(1+(1<<dim));

            if constexpr (dim == 2)
            {
                points[0] = {center[0]   , center[1]   , 0.};
                points[1] = {corner[0]   , corner[1]   , 0.};
                points[2] = {corner[0]+dx, corner[1]   , 0.};
                points[3] = {corner[0]   , corner[1]+dx, 0.};
                points[4] = {corner[0]+dx, corner[1]+dx, 0.};
            }
            else
            {
                points[0] = {center[0]   , center[1]   , center[2]};
                points[1] = {corner[0]   , corner[1]   , corner[2]};
                points[2] = {corner[0]+dx, corner[1]   , corner[2]};
                points[3] = {corner[0]   , corner[1]+dx, corner[2]};
                points[4] = {corner[0]+dx, corner[1]+dx, corner[2]};
                points[5] = {corner[0]   , corner[1]   , corner[2]+dx};
                points[6] = {corner[0]+dx, corner[1]   , corner[2]+dx};
                points[7] = {corner[0]   , corner[1]+dx, corner[2]+dx};
                points[8] = {corner[0]+dx, corner[1]+dx, corner[2]+dx};
            }

            std::size_t npoints = 0;
            for (auto& p: points)
            {
                CGAL::Bounded_side res = inside(p);
                if (res == CGAL::ON_BOUNDED_SIDE)
                {
                    npoints++;
                }
                if (res == CGAL::ON_BOUNDARY)
                {
                    npoints++;
                }
            }
            if (npoints == 0)
            {
                tag[cell] = 0;
            }
            else if (npoints < 1 + (1<<dim))
            {
                tag[cell] = static_cast<int>(samurai::CellFlag::refine);
            }
        });

        samurai::CellList<dim> cl;
        samurai::for_each_interval(ca, [&](std::size_t level, const auto& interval, const auto& index_yz)
        {
            if (level < current_level)
            {
                cl[level][index_yz].add_interval(interval);
            }
            else
            {
                std::size_t itag = static_cast<std::size_t>(interval.start + interval.index);
                for (interval_t::value_t i = interval.start; i < interval.end; ++i, ++itag)
                {
                    if ((tag[itag] & static_cast<int>(samurai::CellFlag::refine)) && level < max_level)
                    {
                        samurai::static_nested_loop<dim - 1, 0, 2>([&](auto stencil)
                        {
                            auto index = 2 * index_yz + stencil;
                            cl[level + 1][index].add_interval({2 * i, 2 * i + 2});
                        });
                    }
                    else if (tag[itag] & static_cast<int>(samurai::CellFlag::keep))
                    {
                        cl[level][index_yz].add_point(i);
                    }
                }
            }
        });

        ca = {cl, true};
        std::cout << "New number of cells " <<  ca.nb_cells() << std::endl;

        save_mesh(path, fmt::format("mesh_{}_{}", output_file, ite++), ca);
        current_level++;
    }

    // // graduation

    // xt::xtensor_fixed<int, xt::xshape<6, dim>> stencil;
    // stencil = {{1, 0, 0}, {-1, 0, 0}, {0, 1, 0}, {0, -1, 0}, {0, 0, 1}, {0, 0, -1}};

    // std::size_t min_level = ca.min_level();
    // while(true)
    // {
    //     auto tag = samurai::make_field<bool, 1>("tag", ca);
    //     tag.fill(false);

    //     for(std::size_t level = min_level + 2; level <= max_level; ++level)
    //     {
    //         for(std::size_t level_below = min_level; level_below < level - 1; ++level_below)
    //         {
    //             for(std::size_t is = 0; is < stencil.shape()[0]; ++is)
    //             {
    //                 auto s = xt::view(stencil, is);
    //                 auto set = samurai::intersection(samurai::translate(ca[level], s), ca[level_below]).on(level_below);
    //                 set([&](const auto& i, const auto& index)
    //                 {
    //                     tag(level_below, i, index[0], index[1]) = true;
    //                 });
    //             }
    //         }
    //     }

    //     samurai::CellList<dim> cl;
    //     samurai::for_each_interval(ca, [&](std::size_t level, const auto& interval, const auto& index_yz)
    //     {
    //         std::size_t itag = static_cast<std::size_t>(interval.start + interval.index);
    //         for (interval_t::value_t i = interval.start; i < interval.end; ++i, ++itag)
    //         {
    //             if (tag[itag])
    //             {
    //                 samurai::static_nested_loop<dim - 1, 0, 2>([&](auto stencil)
    //                 {
    //                     auto index = 2 * index_yz + stencil;
    //                     cl[level + 1][index].add_interval({2 * i, 2 * i + 2});
    //                 });
    //             }
    //             else
    //             {
    //                 cl[level][index_yz].add_point(i);
    //             }
    //         }
    //     });
    //     samurai::CellArray<dim> new_ca = {cl, true};

    //     if(new_ca == ca)
    //     {
    //         break;
    //     }

    //     std::swap(ca, new_ca);
    // }

    // save_mesh(path, fmt::format("mesh_{}_graded", output_file), ca);

    return EXIT_SUCCESS;
}
