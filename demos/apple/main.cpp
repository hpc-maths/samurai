#include <fstream>
#include <iostream>

// #define CGAL_USE_BASIC_VIEWER

#include <CGAL/Arr_circle_segment_traits_2.h>
#include <CGAL/Arrangement_2.h>
#include <CGAL/Bbox_3.h>
#include <CGAL/Cartesian.h>
#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
#include <CGAL/Exact_rational.h>
#include <CGAL/IO/File_header_OFF.h>
#include <CGAL/IO/OBJ_reader.h>
#include <CGAL/IO/binary_file_io.h>
#include <CGAL/Polygon_2.h>
#include <CGAL/Polygon_mesh_processing/internal/named_function_params.h>
#include <CGAL/Polygon_mesh_processing/internal/named_params_helper.h>
#include <CGAL/Polygon_mesh_processing/orient_polygon_soup.h>
#include <CGAL/Polygon_mesh_processing/polygon_soup_to_polygon_mesh.h>
#include <CGAL/Polygon_mesh_processing/repair.h>
#include <CGAL/Polyhedron_3.h>
#include <CGAL/Polyhedron_items_3.h>
#include <CGAL/Simple_cartesian.h>
#include <CGAL/Surface_mesh.h>
#include <CGAL/Surface_mesh_simplification/Policies/Edge_collapse/Count_ratio_stop_predicate.h>
#include <CGAL/Surface_mesh_simplification/edge_collapse.h>
#include <CGAL/array.h>
#include <algorithm>
#include <string>

#include <CGAL/draw_surface_mesh.h>

#include <CGAL/AABB_face_graph_triangle_primitive.h>
#include <CGAL/AABB_traits.h>
#include <CGAL/AABB_tree.h>
#include <CGAL/AABB_triangle_primitive.h>
#include <CGAL/Kernel_traits.h>

#include <samurai/box.hpp>
#include <samurai/cell_array.hpp>
#include <samurai/field.hpp>
#include <samurai/hdf5.hpp>
#include <samurai/static_algorithm.hpp>

typedef CGAL::Simple_cartesian<double> Kernel;
typedef Kernel::Point_3 Point_3;
typedef CGAL::Surface_mesh<Point_3> Surface_mesh;
namespace SMS = CGAL::Surface_mesh_simplification;
typedef CGAL::Surface_mesh<Point_3> Mesh_3;
typedef std::vector<std::size_t> Polygon_3;
using namespace std;

std::string path_extension(const string& filename)
{
    const auto pos = filename.rfind('.');
    if (pos == string::npos)
    {
        return "";
    }
    return filename.substr(pos);
}

Mesh_3* ReadMesh(const char* modelPath)
{
    const auto ext = path_extension(modelPath);

    Mesh_3* mesh = new Mesh_3();
    if (ext == ".obj" || ext == ".OBJ")
    {
        std::ifstream input(modelPath);

        vector<Point_3> points;
        std::vector<Polygon_3> faces;

        std::cout << modelPath << "\n";
        if (!CGAL::read_OBJ(input, points, faces))
        {
            return nullptr;
        }

        namespace PMP = CGAL::Polygon_mesh_processing;

        PMP::is_polygon_soup_a_polygon_mesh(faces);

        PMP::orient_polygon_soup(points, faces);

        try
        {
            PMP::polygon_soup_to_polygon_mesh(points, faces, *mesh);
        }
        catch (char* str)
        {
            std::cout << str << std::endl;
        }

        input.close();
    }
    else if (ext == ".off" || ext == ".OFF")
    {
        std::ifstream input(modelPath);

        if (!input || !(input >> *mesh))
        {
            std::cerr << "mesh is not a valid off file." << std::endl;
            return nullptr;
        }

        input.close();
    }
    else
    {
        std::cerr << "unsupported file type:" << ext << std::endl;
        return nullptr;
    }

    if (mesh->is_empty())
    {
        std::cerr << "mesh is empty." << std::endl;
        delete mesh;
        return nullptr;
    }
    SMS::Count_ratio_stop_predicate<Surface_mesh> stop(0.1);

    // int r = SMS::edge_collapse(*mesh, stop);
    return mesh;
}

int main(int argc, char** argv)
{
    auto p_m = ReadMesh("../demos/apple/Apple.obj");
    auto m   = *p_m;

    typedef CGAL::AABB_face_graph_triangle_primitive<Mesh_3> Primitive;
    typedef CGAL::AABB_traits<Kernel, Primitive> Traits;
    typedef CGAL::AABB_tree<Traits> Tree;

    Tree tree(faces(m).first, faces(m).second, m);
    tree.accelerate_distance_queries();
    tree.build();
    typedef typename boost::property_map<Mesh_3, boost::vertex_point_t>::const_type VPMap;
    VPMap vpmap                   = get(boost::vertex_point, m);
    typename Traits::Point_3 hint = get(vpmap, *vertices(m).begin());

    constexpr std::size_t dim = 3;
    std::size_t start_level   = 2;
    std::size_t max_level     = 8;
    int start                 = 1.2 * std::pow(start_level, 2);
    samurai::Box<int, dim> box({-start, -start, -start}, {start, start, start});
    samurai::CellArray<dim> ca;

    ca[start_level] = {start_level, box};

    for (std::size_t rep = 0; rep <= 10; ++rep)
    {
        std::cout << "iteration: " << rep << "\n";

        auto tag = samurai::make_field<bool, 1>("tag", ca);
        tag.fill(false);

        samurai::for_each_cell(ca,
                               [&](auto cell)
                               {
                                   auto center = 50 * cell.center();
                                   Kernel::Point_3 p(center[0], center[1], center[2]);
                                   hint            = tree.closest_point(p, hint);
                                   Kernel::FT dist = squared_distance(hint, p);
                                   double d        = CGAL::sqrt(dist);

                                   if (d < 2.5 && cell.level > 4)
                                   {
                                       tag[cell] = true;
                                   }

                                   if (d < 10 && cell.level <= 4)
                                   {
                                       tag[cell] = true;
                                   }
                               });

        samurai::CellList<dim> cl;
        samurai::for_each_interval(ca,
                                   [&](std::size_t level, const auto& interval, const auto& index_yz)
                                   {
                                       for (int i = interval.start; i < interval.end; ++i)
                                       {
                                           if (tag[i + interval.index] && level < max_level)
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

        ca = {cl, true};
    }

    auto level = samurai::make_field<std::size_t, 1>("level", ca);
    samurai::for_each_cell(ca,
                           [&](auto cell)
                           {
                               level[cell] = cell.level;
                           });
    std::cout << ca.nb_cells() << "\n";
    samurai::save("mesh_apple", ca, level);

    return EXIT_SUCCESS;
}
