#pragma once

#include "mesh.hpp"
#include "criteria.hpp"

namespace samurai
{
    template <class Field, class Func>
    bool refinement(Field &u, Func&& update_bc_for_level, double eps, double regularity, std::size_t ite)
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

        mesh.for_each_cell([&](auto &cell) {
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

        // Look carefully at how much of this we have to do...
        for (std::size_t level = min_level; level <= max_level - ite; ++level)
        {
            int exponent = dim * (level - max_level);
            auto eps_l = std::pow(2, exponent) * eps;

            // HARTEN HEURISTICS
            auto subset = intersection(mesh[MeshType::cells][level],
                                            mesh[MeshType::all_cells][level-1])
                         .on(level-1);

            double regularity_to_use = std::min(regularity, 3.0) + dim;

            subset.apply_op(to_refine_mr(detail, tag, (pow(2.0, regularity_to_use)) * eps_l, max_level));

        }
        // With the static nested loop, we also grade along the diagonals
        // and not only the axis
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
                else
                {
                    cell_list[level][index_yz].add_point(i);
                }
            }
        });

        mesh_t new_mesh{cell_list, mesh.initial_mesh(), min_level, max_level};

        if (new_mesh == mesh)
            return true;

        auto new_u = make_field<value_t, size>(u.name(), new_mesh);

        for (std::size_t level = min_level; level <= max_level; ++level)
        {
            auto subset = intersection(mesh[MeshType::all_cells][level],
                                       new_mesh[MeshType::cells][level]);
            subset.apply_op(copy(new_u, u));
        }

        for_each_interval(mesh[MeshType::cells], [&](std::size_t level, const auto& interval, const auto& index_yz)
        {
            for (coord_index_t i = interval.start; i < interval.end; ++i)
            {
                if (tag[i + interval.index] &
                    static_cast<int>(CellFlag::refine))
                {
                    compute_prediction(level, interval_t{i, i + 1}, index_yz, u, new_u);
                }
            }
        });

        // NOT NECESSARY BECAUSE THE NEXT
        // MR_PREDITION WILL DO EVERYTHING ALSO FOR THE
        // NECESSARY OVERLEAVES

        // Works but it does not do everything

        // // We have to predict on the overleaves which are not leaves, where the value must be kept.
        // for (std::size_t level = min_level; level < max_level; ++level)
        // {

        //     auto level_cell_array = mesh[MeshType::cells][level];

        //     if (!level_cell_array.empty())
        //     {
        //         level_cell_array.for_each_interval_in_x(
        //             [&](auto const &index_yz, auto const &interval) {
        //                 for (int i = interval.start; i < interval.end; ++i)
        //                 {
        //                     //std::cout<<std::endl<<"Prediction at level "<<level<<" of cell "<<i<<std::flush;
        //                     // We predict
        //                     compute_prediction(level, interval_t{i, i + 1},
        //                                                  index_yz, u, new_u);
        //                 }
        //             });
        //     }
        // }


        // We have to predict on the overleaves which are not leaves, where the value must be kept.
        // for (std::size_t level = min_level; level < max_level; ++level)
        // {
        //     // We are sure that the overleaves above no not intersect with leaves
        //     xt::xtensor_fixed<int, xt::xshape<1>> stencil_plus;
        //     stencil_plus = {{1}};

        //     xt::xtensor_fixed<int, xt::xshape<1>> stencil_minus;
        //     stencil_minus = {{1}}; // One is sufficient to get 2 at a finer level

        //     auto level_cell_array = union_(union_(mesh[MeshType::cells][level],
        //             translate(mesh[MeshType::cells][level], stencil_minus)),
        //             translate(mesh[MeshType::cells][level], stencil_plus));

        // }

        u.mesh_ptr()->swap(new_mesh);
        std::swap(u.array(), new_u.array());

        return false;
    }
}