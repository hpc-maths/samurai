#pragma once

#include <vector>

#include <CGAL/box_intersection_d.h>

#include "../algorithm.hpp"
#include "../cell_array.hpp"
#include "../cell_flag.hpp"
#include "../cell_list.hpp"
#include "../field.hpp"
#include "../graduation.hpp"
#include "../subset/subset_op.hpp"
#include "util.hpp"

namespace samurai
{
    template <std::size_t dim>
    auto
    from_geometry(std::string input_file, std::size_t start_level, std::size_t max_level, bool keep_outside = false, bool keep_inside = false);

    template <>
    auto from_geometry<3>(std::string input_file, std::size_t start_level, std::size_t max_level, bool keep_outside, bool keep_inside)
    {
        auto cgal_mesh = cgal::read_mesh(input_file);
        cgal::Side_of_triangle_mesh inside(cgal_mesh);

        std::vector<cgal::Box> boxes;
        std::size_t start_boxes_id = std::numeric_limits<std::size_t>::max();
        for (auto& f : cgal_mesh.facet_handles())
        {
            boxes.push_back(cgal::PMP::face_bbox(f, cgal_mesh));
            if (start_boxes_id == std::numeric_limits<std::size_t>::max())
            {
                start_boxes_id = boxes.back().id();
            }
        }

        std::vector<cgal::Triangle> triangles;
        for (auto& f : cgal_mesh.facet_handles())
        {
            triangles.push_back(cgal::triangle(f, cgal_mesh));
        }

        using Mesh = CellArray<3>;
        Mesh mesh;
        using interval_t = typename Mesh::interval_t;
        using cell_t     = Cell<3, interval_t>;
        double scale;
        std::tie(scale, mesh) = cgal::init_mesh<3>(start_level, cgal_mesh);

        std::size_t current_level = start_level;
        while (current_level != max_level + 1)
        {
            auto tag = make_field<int, 1>("tag", mesh);
            tag.fill(0);

            for (std::size_t level = mesh.min_level(); level < current_level; ++level)
            {
                for_each_interval(mesh[level],
                                  [&](std::size_t level, const auto& i, const auto& index)
                                  {
                                      tag(level, i, index[0], index[1]) = static_cast<int>(CellFlag::keep);
                                  });
            }

            std::vector<cgal::Box> query;
            std::vector<cell_t> cells;

            query.reserve(mesh[current_level].nb_cells());
            cells.reserve(mesh[current_level].nb_cells());
            std::size_t start_id = std::numeric_limits<std::size_t>::max();
            for_each_cell(mesh[current_level],
                          [&](auto cell)
                          {
                              auto center = scale * cell.center();
                              auto corner = scale * cell.corner();
                              double dx   = scale * cell.length;

                              query.push_back(cgal::Bbox(corner[0], corner[1], corner[2], corner[0] + dx, corner[1] + dx, corner[2] + dx));

                              if (start_id == std::numeric_limits<std::size_t>::max())
                              {
                                  start_id = query.back().id();
                              }
                              cells.push_back(cell);
                          });

            auto callback = [&](const cgal::Box& a, const cgal::Box& b)
            {
                if (cgal::triBoxOverlap(scale, cells[b.id() - start_id], triangles[a.id() - start_boxes_id]))
                {
                    if (current_level != max_level)
                    {
                        tag[cells[b.id() - start_id]] = static_cast<int>(CellFlag::refine);
                    }
                    else
                    {
                        tag[cells[b.id() - start_id]] = static_cast<int>(CellFlag::keep);
                    }
                }
            };
            CGAL::box_intersection_d(boxes.begin(), boxes.end(), query.begin(), query.end(), callback);

            for_each_cell(mesh[current_level],
                          [&](auto cell)
                          {
                              auto center = scale * cell.center();
                              CGAL::Bounded_side res;
                              res = inside({center(0), center(1), center(2)});
                              if (res == CGAL::ON_BOUNDED_SIDE && keep_inside)
                              {
                                  tag[cell] |= static_cast<int>(CellFlag::keep);
                              }
                              if (res == CGAL::ON_UNBOUNDED_SIDE && keep_outside)
                              {
                                  tag[cell] |= static_cast<int>(CellFlag::keep);
                              }
                          });

            CellList<3> cl;
            for_each_interval(mesh,
                              [&](std::size_t level, const auto& interval, const auto& index_yz)
                              {
                                  if (level < current_level)
                                  {
                                      cl[level][index_yz].add_interval(interval);
                                  }
                                  else
                                  {
                                      auto itag = interval.start + interval.index;
                                      for (typename interval_t::value_t i = interval.start; i < interval.end; ++i, ++itag)
                                      {
                                          if ((tag[itag] & static_cast<int>(CellFlag::refine)) && level < max_level)
                                          {
                                              static_nested_loop<2, 0, 2>(
                                                  [&](auto stencil)
                                                  {
                                                      auto index = 2 * index_yz + stencil;
                                                      cl[level + 1][index].add_interval({2 * i, 2 * i + 2});
                                                  });
                                          }
                                          else if (tag[itag] & static_cast<int>(CellFlag::keep))
                                          {
                                              cl[level][index_yz].add_point(i);
                                          }
                                      }
                                  }
                              });

            mesh = {cl, true};
            current_level++;
        }

        return mesh;
    }

}
