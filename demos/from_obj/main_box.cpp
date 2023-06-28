// Copyright 2021 SAMURAI TEAM. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.
#include "CLI/CLI.hpp"

#define _HAS_AUTO_PTR_ETC false

#include <limits>
#include <tuple>

#include <CGAL/Bbox_3.h>
#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
#include <CGAL/Polygon_mesh_processing/IO/polygon_mesh_io.h>
#include <CGAL/Polygon_mesh_processing/triangulate_faces.h>
#include <CGAL/Polyhedron_3.h>
#include <CGAL/Side_of_triangle_mesh.h>
#include <CGAL/Simple_cartesian.h>
#include <CGAL/Surface_mesh.h>
#include <CGAL/Triangle_accessor_3.h>
#include <CGAL/box_intersection_d.h>
#include <CGAL/optimal_bounding_box.h>

// #include <CGAL/polygon_mesh_processing.h>

#include <samurai/algorithm/graduation.hpp>
#include <samurai/box.hpp>
#include <samurai/cell_array.hpp>
#include <samurai/field.hpp>
#include <samurai/hdf5.hpp>
#include <samurai/mr/mesh.hpp>
#include <samurai/static_algorithm.hpp>

#include "tri_overlap.hpp"

namespace PMP = CGAL::Polygon_mesh_processing;
using K       = CGAL::Exact_predicates_inexact_constructions_kernel;

// using K        = CGAL::Simple_cartesian<double>;
using Point    = typename K::Point_3;
using Mesh     = CGAL::Polyhedron_3<K>;
using Triangle = typename CGAL::GetGeomTraits<Mesh>::type::Triangle_3;
// using Mesh  = CGAL::Surface_mesh<Point>;

using Box  = CGAL::Box_intersection_d::Box_d<double, 3>;
using Bbox = CGAL::Bbox_3;

template <typename TriangleMesh, typename CGAL_NP_TEMPLATE_PARAMETERS>
typename CGAL::GetGeomTraits<TriangleMesh, CGAL_NP_CLASS>::type::Triangle_3
triangle(typename boost::graph_traits<TriangleMesh>::face_descriptor fd,
         const TriangleMesh& tmesh,
         const CGAL_NP_CLASS& np = CGAL::parameters::default_values())
{
    using CGAL::parameters::choose_parameter;
    using CGAL::parameters::get_parameter;

    CGAL_precondition(is_valid_face_descriptor(fd, tmesh));

    typename CGAL::GetVertexPointMap<TriangleMesh, CGAL_NP_CLASS>::const_type vpm = choose_parameter(
        get_parameter(np, CGAL::internal_np::vertex_point),
        get_const_property_map(CGAL::vertex_point, tmesh));

    typedef typename CGAL::GetGeomTraits<TriangleMesh, CGAL_NP_CLASS>::type GT;
    GT gt                                                = choose_parameter<GT>(get_parameter(np, CGAL::internal_np::geom_traits));
    typename GT::Construct_triangle_3 construct_triangle = gt.construct_triangle_3_object();

    typedef typename boost::graph_traits<TriangleMesh>::vertex_descriptor vertex_descriptor;
    typedef typename boost::graph_traits<TriangleMesh>::halfedge_descriptor halfedge_descriptor;

    halfedge_descriptor h = halfedge(fd, tmesh);
    vertex_descriptor v1  = target(h, tmesh);
    vertex_descriptor v2  = target(next(h, tmesh), tmesh);
    vertex_descriptor v3  = target(next(next(h, tmesh), tmesh), tmesh);
    return construct_triangle(get(vpm, v1), get(vpm, v2), get(vpm, v3));
}

void cross_product(xt::xtensor_fixed<double, xt::xshape<3>>& dest,
                   const xt::xtensor_fixed<double, xt::xshape<3>>& v1,
                   const xt::xtensor_fixed<double, xt::xshape<3>>& v2)
{
    dest(0) = v1(1) * v2(2) - v1(2) * v2(1);
    dest(1) = v1(2) * v2(0) - v1(0) * v2(2);
    dest(2) = v1(0) * v2(1) - v1(1) * v2(0);
}

auto normal(const Triangle& t)
{
    auto n = CGAL::normal(t.vertex(0), t.vertex(1), t.vertex(2));
    return xt::xtensor_fixed<double, xt::xshape<3>>{n.x(), n.y(), n.z()};
}

double dot(const xt::xtensor_fixed<double, xt::xshape<3>>& v1, const xt::xtensor_fixed<double, xt::xshape<3>>& v2)
{
    return v1(0) * v2(0) + v1(1) * v2(1) + v1(2) * v2(2);
}

bool planeBoxOverlap_new(double dx,
                         double scale,
                         const xt::xtensor_fixed<double, xt::xshape<3>>& normal,
                         const xt::xtensor_fixed<double, xt::xshape<3>>& v)
{
    xt::xtensor_fixed<double, xt::xshape<3>> vmin;
    xt::xtensor_fixed<double, xt::xshape<3>> vmax;

    for (std::size_t i = 0; i < 3; ++i)
    {
        if (normal(i) > 0.)
        {
            vmin(i) = -scale * dx - v(i);
            vmax(i) = scale * dx - v(i);
        }
        else
        {
            vmin(i) = scale * dx - v(i);
            vmax(i) = -scale * dx - v(i);
        }
    }
    if (dot(normal, vmin) > 0.)
    {
        return false;
    }
    if (dot(normal, vmax) >= 0.)
    {
        return true;
    }
    return false;
}

template <class Cell, class Triangle>
bool triBoxOverlap_new(double scale, const Cell& cell, const Triangle& triangle, double tol = 0)
{
    xt::xtensor_fixed<double, xt::xshape<3, 3>> ei = xt::eye<double>(3);
    xt::xtensor_fixed<double, xt::xshape<3, 3>> vi;
    for (std::size_t i = 0; i < 3; ++i)
    {
        auto& vertex = triangle.vertex(i);
        xt::xtensor_fixed<double, xt::xshape<3>> tri_coords{vertex.x(), vertex.y(), vertex.z()};
        xt::xtensor_fixed<double, xt::xshape<3>> center;
        if constexpr (Cell::dim == 2)
        {
            center = {scale * cell.center(0), scale * cell.center(1), 0};
        }
        else if constexpr (Cell::dim == 3)
        {
            center = {scale * cell.center(0), scale * cell.center(1), scale * cell.center(2)};
        }
        xt::view(vi, i) = tri_coords - center;
    }

    xt::xtensor_fixed<double, xt::xshape<3, 3>> fj;
    xt::view(fj, 0) = xt::view(vi, 1) - xt::view(vi, 0);
    xt::view(fj, 1) = xt::view(vi, 2) - xt::view(vi, 1);
    xt::view(fj, 2) = xt::view(vi, 0) - xt::view(vi, 2);

    // std::cout << "triangle: " << triangle << std::endl;
    // std::cout << "cell: " << cell << std::endl;
    xt::xtensor_fixed<double, xt::xshape<3>> aij;
    for (std::size_t i = 0; i < 3; ++i)
    {
        for (std::size_t j = 0; j < 3; ++j)
        {
            cross_product(aij, xt::view(ei, i), xt::view(fj, j));

            double p0     = dot(aij, xt::view(vi, 0));
            double p1     = dot(aij, xt::view(vi, 1));
            double p2     = dot(aij, xt::view(vi, 2));
            double radius = scale * .5 * cell.length * (std::abs(aij(0)) + std::abs(aij(1)) + std::abs(aij(2)));
            // std::cout << aij << " " << xt::view(ei, i) << " " << xt::view(fj, j) << std::endl;
            // std::cout << p0 << " " << p1 << " " << p2 << " " << radius << std::endl;

            if ((std::min(std::min(p0, p1), p2) > radius + tol) || (std::max(std::max(p0, p1), p2) < -radius - tol))
            {
                // std::cout << "false for " << i << " " << j << std::endl;
                return false;
            }
        }
    }

    // return planeBoxOverlap_new(cell.length, scale, normal(triangle), xt::view(vi, 0));
    return true;
}

template <std::size_t dim>
auto init_mesh(std::size_t start_level, const Mesh& sm)
{
    std::array<Point, 8> obb_points;
    CGAL::oriented_bounding_box(sm, obb_points, CGAL::parameters::use_convex_hull(true));

    xt::xtensor_fixed<double, xt::xshape<dim>> min_corner;
    xt::xtensor_fixed<double, xt::xshape<dim>> max_corner;
    min_corner.fill(std::numeric_limits<double>::max());
    max_corner.fill(std::numeric_limits<double>::min());

    double dx = 1. / (1 << start_level);
    for (auto& p : obb_points)
    {
        for (std::size_t d = 0; d < dim; ++d)
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
    for (std::size_t d = 1; d < dim; ++d)
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

    min_corner -= 2 * dx;
    max_corner += 2 * dx;

    std::cout << "scale factor " << scale << std::endl;

    samurai::Box<double, dim> box(min_corner, max_corner);
    samurai::CellArray<dim> ca;
    ca[start_level] = {start_level, box};

    return std::make_pair(scale, ca);
}

template <class Mesh>
void save_mesh(const fs::path& path, const std::string& filename, const Mesh& mesh)
{
    static constexpr std::size_t dim = Mesh::dim;
    auto level                       = samurai::make_field<std::size_t, 1>("level", mesh);
    auto coords                      = samurai::make_field<double, dim>("coords", mesh);
    auto coords_i                    = samurai::make_field<int, dim>("coords_i", mesh);

    if (!fs::exists(path))
    {
        fs::create_directory(path);
    }

    samurai::for_each_cell(mesh,
                           [&](auto& cell)
                           {
                               level[cell]    = cell.level;
                               coords[cell]   = cell.center();
                               coords_i[cell] = cell.indices;
                           });

    samurai::save(path, filename, mesh, level, coords, coords_i);
}

int main(int argc, char** argv)
{
    constexpr std::size_t dim = 3;
    std::size_t start_level   = 1;
    std::size_t max_level     = 8;
    bool keep_inside          = false;
    bool keep_outside         = false;

    // Output parameters
    fs::path path = fs::current_path();
    std::string input_file;

    CLI::App app{"Create an adapted mesh from an OBJ file"};
    app.add_option("--input", input_file, "input File")->required()->check(CLI::ExistingFile);
    ;
    app.add_option("--start-level", start_level, "Start level of the output adaptive mesh")->capture_default_str();
    app.add_flag("--keep-inside", keep_inside, "Keep the cells inside the form")->capture_default_str();
    app.add_flag("--keep-outside", keep_outside, "Keep the cells outside the form")->capture_default_str();
    app.add_option("--max-level", max_level, "Maximum level of the output adaptive mesh")->capture_default_str();
    app.add_option("--path", path, "Output path")->capture_default_str();
    CLI11_PARSE(app, argc, argv);

    std::string output_file = fs::path(input_file).stem();

    Mesh mesh;
    if (!PMP::IO::read_polygon_mesh(input_file, mesh) || mesh.is_empty())
    {
        std::cerr << "Invalid input file." << std::endl;
        return EXIT_FAILURE;
    }

    if (!CGAL::is_triangle_mesh(mesh))
    {
        PMP::triangulate_faces(mesh);
    }

    CGAL::Side_of_triangle_mesh<Mesh, K> inside(mesh);

    std::vector<Box> boxes;
    std::size_t start_boxes_id = std::numeric_limits<std::size_t>::max();
    for (auto& f : mesh.facet_handles())
    {
        boxes.push_back(CGAL::Polygon_mesh_processing::face_bbox(f, mesh));
        if (start_boxes_id == std::numeric_limits<std::size_t>::max())
        {
            // std::cout << "start " << boxes.back().id() << std::endl;
            start_boxes_id = boxes.back().id();
        }
        // std::cout << "boxes_id " << boxes.back().id() << std::endl;
    }

    std::vector<Triangle> triangles;
    // std::cout << "triangles" << std::endl;
    for (auto& f : mesh.facet_handles())
    {
        triangles.push_back(triangle(f, mesh));
        // std::cout << triangles.back() << std::endl;
    }
    // std::cout << std::endl << std::endl;

    samurai::CellArray<dim> ca;
    using cell_t     = samurai::Cell<int, dim>;
    using interval_t = typename samurai::CellArray<dim>::interval_t;
    double scale;
    std::tie(scale, ca) = init_mesh<dim>(start_level, mesh);

    save_mesh(path, fmt::format("mesh_{}_init", output_file), ca);

    std::size_t current_level = start_level;
    std::size_t ite           = 0;
    while (current_level != max_level + 1)
    {
        std::cout << "iteration: " << ite << std::endl;
        std::cout << "Number of cells " << ca.nb_cells() << std::endl;

        auto tag = samurai::make_field<int, 1>("tag", ca);
        // tag.fill(static_cast<int>(samurai::CellFlag::keep));
        tag.fill(0);

        for (std::size_t level = ca.min_level(); level < current_level; ++level)
        {
            samurai::for_each_interval(ca[level],
                                       [&](std::size_t level, const auto& i, const auto& index)
                                       {
                                           tag(level, i, index[0], index[1]) = static_cast<int>(samurai::CellFlag::keep);
                                       });
        }

        std::vector<Box> query;
        std::vector<cell_t> cells;
        std::cout << "query " << ca[current_level].nb_cells() << std::endl;
        query.reserve(ca[current_level].nb_cells());
        cells.reserve(ca[current_level].nb_cells());
        std::size_t start_id = std::numeric_limits<std::size_t>::max();
        samurai::for_each_cell(ca[current_level],
                               [&](auto cell)
                               {
                                   auto center = scale * cell.center();
                                   auto corner = scale * cell.corner();
                                   double dx   = scale * cell.length;
                                   // std::vector<Box> query{Bbox(corner[0], corner[1], corner[2], corner[0] + dx, corner[1] + dx, corner[2]
                                   // + dx)};
                                   if constexpr (dim == 2)
                                   {
                                       query.push_back(Bbox(corner[0], corner[1], 0, corner[0] + dx, corner[1] + dx, 0));
                                   }
                                   else if constexpr (dim == 3)
                                   {
                                       query.push_back(Bbox(corner[0], corner[1], corner[2], corner[0] + dx, corner[1] + dx, corner[2] + dx));
                                   }
                                   if (start_id == std::numeric_limits<std::size_t>::max())
                                   {
                                       start_id = query.back().id();
                                   }
                                   cells.push_back(cell);
                                   //    std::cout << "query_id " << query.back().id() << std::endl;
                                   // auto callback = [&](const Box& a, const Box& b)
                                   // {
                                   //     tag[cell] = static_cast<int>(samurai::CellFlag::refine);
                                   // };
                                   // CGAL::box_intersection_d(boxes.begin(), boxes.end(), query.begin(), query.end(), callback);
                               });

        auto callback = [&](const Box& a, const Box& b)
        {
            // std::cout << "---------------------------" << std::endl;
            // std::cout << a.id() << " " << b.id() << std::endl;
            // std::cout << a.id() - start_boxes_id << " " << b.id() - start_id << std::endl;
            // std::cout << triangles[a.id() - start_boxes_id] << std::endl;
            // std::cout << cells[b.id() - start_id] << std::endl;
            // std::cout << "---------------------------" << std::endl;

            if (triBoxOverlap_new(scale, cells[b.id() - start_id], triangles[a.id() - start_boxes_id]))
            {
                if (current_level != max_level)
                {
                    tag[cells[b.id() - start_id]] = static_cast<int>(samurai::CellFlag::refine);
                }
                else
                {
                    tag[cells[b.id() - start_id]] = static_cast<int>(samurai::CellFlag::keep);
                }
            }
        };
        CGAL::box_intersection_d(boxes.begin(), boxes.end(), query.begin(), query.end(), callback);

        samurai::for_each_cell(ca[current_level],
                               [&](auto cell)
                               {
                                   auto center = scale * cell.center();
                                   CGAL::Bounded_side res;
                                   if constexpr (dim == 2)
                                   {
                                       res = inside({center(0), center(1), 0});
                                   }
                                   else if constexpr (dim == 3)
                                   {
                                       res = inside({center(0), center(1), center(2)});
                                   }
                                   if (res == CGAL::ON_BOUNDED_SIDE && keep_inside)
                                   {
                                       tag[cell] |= static_cast<int>(samurai::CellFlag::keep);
                                   }
                                   if (res == CGAL::ON_UNBOUNDED_SIDE && keep_outside)
                                   {
                                       tag[cell] |= static_cast<int>(samurai::CellFlag::keep);
                                   }
                               });

        samurai::CellList<dim> cl;
        samurai::for_each_interval(ca,
                                   [&](std::size_t level, const auto& interval, const auto& index_yz)
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
                                                   samurai::static_nested_loop<dim - 1, 0, 2>(
                                                       [&](auto stencil)
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
        std::cout << "New number of cells " << ca.nb_cells() << std::endl;

        save_mesh(path, fmt::format("mesh_{}_{}", output_file, ite++), ca);
        current_level++;
    }

    // graduation

    xt::xtensor_fixed<int, xt::xshape<6, dim>> stencil;
    stencil = {
        {1,  0,  0 },
        {-1, 0,  0 },
        {0,  1,  0 },
        {0,  -1, 0 },
        {0,  0,  1 },
        {0,  0,  -1}
    };

    std::size_t min_level = ca.min_level();
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
                            tag(level_below, i, index[0], index[1]) = true;
                        });
                }
            }
        }

        samurai::CellList<dim> cl;
        samurai::for_each_interval(ca,
                                   [&](std::size_t level, const auto& interval, const auto& index_yz)
                                   {
                                       std::size_t itag = static_cast<std::size_t>(interval.start + interval.index);
                                       for (interval_t::value_t i = interval.start; i < interval.end; ++i, ++itag)
                                       {
                                           if (tag[itag])
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
                                       }
                                   });
        samurai::CellArray<dim> new_ca = {cl, true};

        if (new_ca == ca)
        {
            break;
        }

        std::swap(ca, new_ca);
    }

    save_mesh(path, fmt::format("mesh_{}_graded", output_file), ca);

    // auto field = samurai::make_field<bool, 1>("in_or_out", ca);
    // field.fill(false);
    // samurai::for_each_cell(ca,
    //                        [&](const auto& cell)
    //                        {
    //                            auto center = scale * cell.center();
    //                            auto corner = scale * cell.corner();
    //                            double dx   = scale * cell.length;
    //                            std::vector<Point> points(1 + (1 << dim));

    //                            if constexpr (dim == 2)
    //                            {
    //                                points[0] = {center[0], center[1], 0.};
    //                                points[1] = {corner[0], corner[1], 0.};
    //                                points[2] = {corner[0] + dx, corner[1], 0.};
    //                                points[3] = {corner[0], corner[1] + dx, 0.};
    //                                points[4] = {corner[0] + dx, corner[1] + dx, 0.};
    //                            }
    //                            else
    //                            {
    //                                points[0] = {center[0], center[1], center[2]};
    //                                points[1] = {corner[0], corner[1], corner[2]};
    //                                points[2] = {corner[0] + dx, corner[1], corner[2]};
    //                                points[3] = {corner[0], corner[1] + dx, corner[2]};
    //                                points[4] = {corner[0] + dx, corner[1] + dx, corner[2]};
    //                                points[5] = {corner[0], corner[1], corner[2] + dx};
    //                                points[6] = {corner[0] + dx, corner[1], corner[2] + dx};
    //                                points[7] = {corner[0], corner[1] + dx, corner[2] + dx};
    //                                points[8] = {corner[0] + dx, corner[1] + dx, corner[2] + dx};
    //                            }

    //                            std::size_t npoints = 0;
    //                            for (auto& p : points)
    //                            {
    //                                CGAL::Bounded_side res = inside(p);
    //                                if (res == CGAL::ON_BOUNDED_SIDE)
    //                                {
    //                                    npoints++;
    //                                }
    //                                if (res == CGAL::ON_BOUNDARY)
    //                                {
    //                                    npoints++;
    //                                }
    //                            }
    //                            if (npoints == 1 + (1 << dim))
    //                            {
    //                                field[cell] = true;
    //                            }
    //                        });

    // samurai::save(path, fmt::format("field_{}", output_file), ca, field);

    return EXIT_SUCCESS;
}
