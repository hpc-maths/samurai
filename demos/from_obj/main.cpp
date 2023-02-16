// Copyright 2021 SAMURAI TEAM. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.
#include "CLI/CLI.hpp"

#include <limits>
#include <tuple>

#include <CGAL/Surface_mesh.h>
#include <CGAL/optimal_bounding_box.h>
#include <CGAL/Polygon_mesh_processing/IO/polygon_mesh_io.h>
#include <CGAL/polygon_mesh_processing.h>
#include <CGAL/Surface_mesh_simplification/edge_collapse.h>
#include <CGAL/Surface_mesh_simplification/Policies/Edge_collapse/Count_ratio_stop_predicate.h>

#include <samurai/box.hpp>
#include <samurai/cell_array.hpp>
#include <samurai/field.hpp>
#include <samurai/hdf5.hpp>
#include <samurai/static_algorithm.hpp>

namespace PMP = CGAL::Polygon_mesh_processing;
namespace SMS = CGAL::Surface_mesh_simplification;
using Kernel = CGAL::Exact_predicates_inexact_constructions_kernel;
using Point = typename Kernel::Point_3;
using Surface_mesh = CGAL::Surface_mesh<Point>;

template <std::size_t dim>
auto init_mesh(std::size_t start_level, const Surface_mesh& sm)
{
    std::array<Point, 8> obb_points;
    CGAL::oriented_bounding_box(sm, obb_points,
                                CGAL::parameters::use_convex_hull(true));

    xt::xtensor_fixed<double, xt::xshape<3>> min_corner;
    xt::xtensor_fixed<double, xt::xshape<3>> max_corner;
    min_corner.fill(std::numeric_limits<double>::max());
    max_corner.fill(std::numeric_limits<double>::min());

    double dx = 1./(1 << start_level);
    for(auto& p: obb_points)
    {
        for(std::size_t d = 0; d < 3; ++d)
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
    for(std::size_t d = 1; d < 3; ++d)
    {
        double tmp = max_corner[d] - min_corner[d];
        if (scale > tmp)
        {
            scale = tmp;
        }
    }
    min_corner /= scale;
    max_corner /= scale;

    min_corner -= 2*dx;
    max_corner += 2*dx;

    std::cout << "scale factor " << scale << std::endl;

    samurai::Box<double, dim> box(min_corner, max_corner);
    samurai::CellArray<dim> ca;
    ca[start_level] = {start_level, box};

    return std::make_pair(scale, ca);
}

auto build_tree(const Surface_mesh& sm)
{
    using Primitive = CGAL::AABB_face_graph_triangle_primitive<Surface_mesh>;
    using Traits = CGAL::AABB_traits<Kernel, Primitive>;
    using Tree = CGAL::AABB_tree<Traits>;

    Tree tree( faces(sm).first, faces(sm).second, sm);
    tree.accelerate_distance_queries();
    tree.build();

    return tree;
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
    constexpr std::size_t dim = 3;
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

    Surface_mesh sm;
    if(!PMP::IO::read_polygon_mesh(input_file, sm) || sm.is_empty())
    {
        std::cerr << "Invalid input file." << std::endl;
        return EXIT_FAILURE;
    }

    samurai::CellArray<dim> ca;
    using interval_t = typename samurai::CellArray<dim>::interval_t;
    double scale;
    std::tie(scale, ca) = init_mesh<dim>(start_level, sm);
    auto tree = build_tree(sm);

    std::cout << "Initial number of cells " <<  ca.nb_cells() << std::endl;
    std::size_t current_level = start_level;
    std::size_t ite = 0;

    while(current_level != max_level)
    {
        std::cout << "iteration: " << ite << std::endl;

        auto tag = samurai::make_field<bool, 1>("tag", ca);
        tag.fill(false);

        samurai::for_each_cell(ca[current_level], [&](auto cell)
        {

            auto center = scale*cell.center();
            double dx = scale*cell.length;
            Point p(center[0], center[1], center[2]);
            Kernel::FT dist = tree.squared_distance(p);

            if (CGAL::sqrt(dist) < 2*dx)
            {
                tag[cell] = true;
            }
        });

        samurai::CellList<dim> cl;
        samurai::for_each_interval(ca, [&](std::size_t level, const auto& interval, const auto& index_yz)
        {
            std::size_t itag = static_cast<std::size_t>(interval.start + interval.index);
            for (interval_t::value_t i = interval.start; i < interval.end; ++i, ++itag)
            {
                if (tag[itag] && level < max_level)
                {
                    samurai::static_nested_loop<dim - 1, 0, 2>([&](auto stencil) {
                        auto index = 2 * index_yz + stencil;
                        cl[level + 1][index].add_interval({2 * i, 2 * i + 2});
                    });
                }
                else
                {
                    cl[level][index_yz].add_point(i);
                }
            }
        });

        ca = {cl, true};
        std::cout << "Number of cells " <<  ca.nb_cells() << std::endl;

        save_mesh(path, fmt::format("mesh_{}_{}", output_file, ite++), ca);
        current_level++;
    }

    return EXIT_SUCCESS;
}
