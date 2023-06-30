#pragma once

#include <limits>

#include <CGAL/Bbox_3.h>
#include <CGAL/Polygon_mesh_processing/IO/polygon_mesh_io.h>
#include <CGAL/Polygon_mesh_processing/triangulate_faces.h>
#include <CGAL/Polyhedron_3.h>
#include <CGAL/Side_of_triangle_mesh.h>
#include <CGAL/Simple_cartesian.h>
#include <CGAL/box_intersection_d.h>
#include <CGAL/optimal_bounding_box.h>

#include <xtensor/xfixed.hpp>

#include "../box.hpp"
#include "../cell_array.hpp"

namespace samurai
{
    namespace cgal
    {
        namespace PMP               = CGAL::Polygon_mesh_processing;
        using K                     = CGAL::Simple_cartesian<double>;
        using Point                 = typename K::Point_3;
        using CGALMesh              = CGAL::Polyhedron_3<K>;
        using Triangle              = typename CGAL::GetGeomTraits<CGALMesh>::type::Triangle_3;
        using Side_of_triangle_mesh = typename CGAL::Side_of_triangle_mesh<CGALMesh, K>;
        using Box                   = CGAL::Box_intersection_d::Box_d<double, 3>;
        using Bbox                  = CGAL::Bbox_3;

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
        bool triBoxOverlap(double scale, const Cell& cell, const Triangle& triangle, double tol = 0)
        {
            xt::xtensor_fixed<double, xt::xshape<3, 3>> ei = xt::eye<double>(3);
            xt::xtensor_fixed<double, xt::xshape<3, 3>> vi;
            for (std::size_t i = 0; i < 3; ++i)
            {
                auto& vertex = triangle.vertex(static_cast<int>(i));
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

                    if ((std::min(std::min(p0, p1), p2) > radius + tol) || (std::max(std::max(p0, p1), p2) < -radius - tol))
                    {
                        return false;
                    }
                }
            }

            // return planeBoxOverlap(cell.length, scale, normal(triangle), xt::view(vi, 0));
            return true;
        }

        CGALMesh read_mesh(std::string input_file)
        {
            CGALMesh cgal_mesh;
            if (!PMP::IO::read_polygon_mesh(input_file, cgal_mesh) || cgal_mesh.is_empty())
            {
                std::cerr << "Invalid input file." << std::endl;
                return CGALMesh();
            }

            if (!CGAL::is_triangle_mesh(cgal_mesh))
            {
                PMP::triangulate_faces(cgal_mesh);
            }
            return cgal_mesh;
        }

        template <std::size_t dim>
        auto init_mesh(std::size_t start_level, const CGALMesh& cgal_mesh)
        {
            std::array<Point, 8> obb_points;
            CGAL::oriented_bounding_box(cgal_mesh, obb_points, CGAL::parameters::use_convex_hull(true));

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

            samurai::Box<double, dim> box(min_corner, max_corner);
            CellArray<dim> mesh;
            mesh[start_level] = {start_level, box};

            return std::make_pair(scale, mesh);
        }
    }

}
