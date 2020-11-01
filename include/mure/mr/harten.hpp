#pragma once

#include <cmath>

#include "mesh.hpp"
#include "operators.hpp"
#include "criteria.hpp"

namespace mure
{
    template <class Field, class Func>
    bool harten(Field &u, Field &uold, Func&& update_bc_for_level, double eps, double regularity, std::size_t ite)
    {
        constexpr std::size_t dim = Field::dim;
        constexpr std::size_t size = Field::size;

        using value_t = typename Field::value_type;
        using mesh_t = typename Field::mesh_t;
        using interval_t = typename Field::interval_t;
        using coord_index_t = typename interval_t::coord_index_t;
        using cl_type = typename mesh_t::cl_type;

        auto mesh = u.mesh();
        std::size_t min_level = mesh.min_level(), max_level = mesh.max_level();

        auto detail = make_field<value_t, size>("detail", mesh);

        auto tag = make_field<int, 1>("tag", mesh);
        tag.fill(0);

        for_each_cell(mesh[MeshType::cells], [&](auto &cell)
        {
            tag[cell] = static_cast<int>(CellFlag::keep);
        });

        mr_projection(u);
        for (std::size_t level = min_level - 1; level <= max_level; ++level)
        {
            update_bc_for_level(u, level);
        }
        mr_prediction(u, update_bc_for_level);

        for (std::size_t level = min_level - 1; level < max_level - ite; ++level)
        {
            auto subset = intersection(mesh[MeshType::all_cells][level],
                                       mesh[MeshType::cells][level + 1])
                         .on(level);
            subset.apply_op(compute_detail(detail, u));
        }

        for (std::size_t level = min_level; level <= max_level - ite; ++level)
        {
            double exponent = dim * (max_level - level);
            double eps_l = std::pow(2., -exponent) * eps;

            double regularity_to_use = std::min(regularity, 3.0) + dim;

            auto subset_1 = intersection(mesh[MeshType::cells][level],
                                         mesh[MeshType::all_cells][level-1])
                           .on(level-1);

            subset_1.apply_op(to_coarsen_mr(detail, tag, eps_l, min_level)); // Derefinement
            subset_1.apply_op(to_refine_mr(detail, tag, (pow(2.0, regularity_to_use)) * eps_l, max_level)); // Refinement according to Harten
        }

        for (std::size_t level = min_level; level <= max_level - ite; ++level)
        {
            auto subset_2 = intersection(mesh[MeshType::cells][level],
                                         mesh[MeshType::cells][level]);
            auto subset_3 = intersection(mesh[MeshType::cells_and_ghosts][level],
                                         mesh[MeshType::cells_and_ghosts][level]);

            subset_2.apply_op(enlarge(tag));
            subset_2.apply_op(keep_around_refine(tag));
            subset_3.apply_op(tag_to_keep(tag));
        }

        // COARSENING GRADUATION
        for (std::size_t level = max_level; level > 0; --level)
        {
            auto keep_subset = intersection(mesh[MeshType::cells][level],
                                            mesh[MeshType::all_cells][level - 1])
                              .on(level - 1);

            keep_subset.apply_op(maximum(tag));

            xt::xtensor_fixed<int, xt::xshape<dim>> stencil;
            for (std::size_t d = 0; d < dim; ++d)
            {
                stencil.fill(0);
                for (int s = -1; s <= 1; ++s)
                {
                    if (s != 0)
                    {
                        stencil[d] = s;
                        auto subset = intersection(mesh[MeshType::cells][level],
                                                translate(mesh[MeshType::cells][level - 1], stencil))
                                    .on(level - 1);
                        subset.apply_op(balance_2to1(tag, stencil));
                    }
                }
            }
        }

        // REFINEMENT GRADUATION
        for (std::size_t level = max_level; level > min_level; --level)
        {
            auto subset_1 = intersection(mesh[MeshType::cells][level],
                                        mesh[MeshType::cells][level]);

            subset_1.apply_op(extend(tag));

            static_nested_loop<dim, -1, 2>(
                [&](auto stencil) {

                auto subset = intersection(translate(mesh[MeshType::cells][level], stencil),
                                           mesh[MeshType::cells][level-1])
                             .on(level);

                subset.apply_op(make_graduation(tag));

            });
        }

        for (std::size_t level = max_level; level > 0; --level)
        {
            auto keep_subset = intersection(mesh[MeshType::cells][level],
                                            mesh[MeshType::all_cells][level - 1])
                            .on(level - 1);

            keep_subset.apply_op(maximum(tag));
        }

        cl_type cell_list;

        for_each_interval(mesh[MeshType::cells], [&](std::size_t level, const auto& interval, const auto& index_yz)
        {
            for (coord_index_t i = interval.start; i < interval.end; ++i)
            {
                if (tag[i + interval.index] & static_cast<int>(CellFlag::refine))
                {
                    static_nested_loop<dim - 1, 0, 2>([&](auto stencil)
                    {
                        auto index = 2 * index_yz + stencil;
                        cell_list[level + 1][index].add_interval({2 * i, 2 * i + 2});
                    });
                }
                else if (tag[i + interval.index] & static_cast<int>(CellFlag::keep))
                {
                    cell_list[level][index_yz].add_point(i);
                }
                else
                {
                    cell_list[level-1][index_yz>>1].add_point(i>>1);
                }
            }
        });

        mesh_t new_mesh{cell_list, mesh.initial_mesh(), min_level, max_level};

        if (new_mesh == mesh)
        {
            return true;
        }

        auto new_u = make_field<value_t, size>(u.name(), new_mesh);
        new_u.fill(0.);

        for (std::size_t level = min_level; level <= max_level; ++level)
        {
            auto subset = intersection(union_(mesh[MeshType::cells][level],
                                              mesh[MeshType::proj_cells][level]),
                                       new_mesh[MeshType::cells][level]);

            subset.apply_op(copy(new_u, u));
        }

        for_each_interval(mesh[MeshType::cells], [&](std::size_t level, const auto& interval, const auto& index_yz)
        {
            for (coord_index_t i = interval.start; i < interval.end; ++i)
            {
                if (tag[i + interval.index] & static_cast<int>(CellFlag::refine))
                {
                    compute_prediction(level, interval_t{i, i + 1}, index_yz, u, new_u);
                }
            }
        });

        // START comment to the old fashion
        // which eliminates details of cells first deleted and then re-added by the refinement
        auto old_mesh = uold.mesh();
        for (std::size_t level = min_level; level <= max_level; ++level)
        {
            auto subset = intersection(intersection(old_mesh[MeshType::cells][level],
                                                    difference(new_mesh[MeshType::cells][level],
                                                               mesh[MeshType::cells][level])),
                                       mesh[MeshType::cells][level-1])
                         .on(level);

            subset.apply_op(copy(new_u, uold));
        }
        // END comment

        u.mesh_ptr()->swap(new_mesh);
        uold.mesh_ptr()->swap(new_mesh);

        std::swap(u.array(), new_u.array());
        std::swap(uold.array(), new_u.array());

        return false;
    }
}