#pragma once

#include <spdlog/spdlog.h>
#include <xtensor/xfixed.hpp>

#include "../field.hpp"
#include "cell_flag.hpp"
#include "mesh.hpp"

namespace mure
{
    template<class MRConfig>
    void harten_multi(std::vector<Field<MRConfig>> &field, double eps,
                      std::size_t ite)
    {
        constexpr auto max_refinement_level = MRConfig::max_refinement_level;
        constexpr auto dim = MRConfig::dim;
        using interval_t = typename MRConfig::interval_t;

        spdlog::info("Enter in Harten function");

        auto mesh = field[0].mesh();
        std::size_t max_level = mesh.max_level();
        std::size_t min_level = mesh.min_level();

        std::size_t field_size = field.size();

        std::vector<
            xt::xtensor_fixed<double, xt::xshape<max_refinement_level + 1>>>
            max_detail(field_size);

        Field<MRConfig, int> cell_flag_global{"cell_flag_global", mesh};
        cell_flag_global.array().fill(0);

        std::vector<Field<MRConfig, double>> detail;
        std::vector<Field<MRConfig, int>> cell_flag;

        for (std::size_t i = 0; i < field_size; ++i)
        {
            max_detail[i].fill(std::numeric_limits<double>::min());
            std::stringstream str;
            str << "detail_" << i;
            detail.push_back({str.str().data(), mesh});
            detail[i].array().fill(0);
            str << "cell_flag_" << i;
            cell_flag.push_back({str.str().data(), mesh});
            cell_flag[i].array().fill(0);
        }

        mesh.for_each_cell([&](auto &cell) {
            for (std::size_t i = 0; i < field_size; ++i)
                cell_flag[i][cell] = static_cast<int>(CellFlag::keep);
        });

        // Compute the detail
        spdlog::info("Compute detail");
        for (std::size_t level = min_level - 1; level < max_level - ite;
             ++level)
        {
            auto subset = intersection(mesh[MeshType::all_cells][level],
                                       mesh[MeshType::cells][level + 1])
                              .on(level);

            for (std::size_t i = 0; i < field.size(); ++i)
            {
                subset.apply_op(level, compute_detail(detail[i], field[i]),
                                compute_max_detail(detail[i], max_detail[i]));
            }
        }

        spdlog::info("Tag cells to be refine or coarsen");
        for (std::size_t level = 0; level < max_level - ite; ++level)
        {
            int exponent = dim * (level - max_level + 1);

            auto eps_l = std::pow(2, exponent) * eps;

            auto subset_1 = intersection(mesh[MeshType::all_cells][level],
                                         mesh[MeshType::cells][level + 1])
                                .on(level);
            // Harten
            auto subset_2 = intersection(mesh[MeshType::cells][level + 1],
                                         mesh[MeshType::cells][level + 1]);

            auto subset_3 =
                intersection(mesh[MeshType::cells_and_ghosts][level + 1],
                             mesh[MeshType::cells_and_ghosts][level + 1]);

            for (std::size_t i = 0; i < field.size(); ++i)
            {
                subset_1.apply_op(level, to_coarsen(cell_flag[i], detail[i],
                                                    max_detail[i], eps_l));

                subset_2.apply_op(level + 1, to_refine(cell_flag[i], detail[i],
                                                       max_detail[i], max_level,
                                                       4 * eps_l));

                subset_2.apply_op(level + 1,
                                  enlarge(cell_flag[i], CellFlag::keep));

                subset_3.apply_op(level + 1, tag_to_keep(cell_flag[i]));
            }

            auto subset = intersection(mesh[MeshType::cells][level],
                                       expand(mesh[MeshType::cells][level + 1]))
                              .on(level + 1);

            for (std::size_t i = 0; i < field.size(); ++i)
            {
                subset.apply_op(level, refine_ghost(cell_flag[i]));
            }
        }

        spdlog::info("Make 2 to 1 balance");
        for (std::size_t level = max_level; level > 0; --level)
        {
            auto keep_subset =
                intersection(mesh[MeshType::cells][level],
                             mesh[MeshType::all_cells][level - 1])
                    .on(level - 1);

            for (std::size_t i = 0; i < field.size(); ++i)
            {
                keep_subset.apply_op(level - 1, maximum(cell_flag[i]));
            }

            xt::xtensor_fixed<int, xt::xshape<dim>> stencil;
            for (std::size_t d = 0; d < dim; ++d)
            {
                for (std::size_t d1 = 0; d1 < dim; ++d1)
                    stencil[d1] = 0;
                for (int s = -1; s <= 1; ++s)
                {
                    if (s != 0)
                    {
                        stencil[d] = s;

                        auto subset =
                            intersection(
                                mesh[MeshType::cells][level],
                                translate_stencil(
                                    mesh[MeshType::cells][level - 1], stencil))
                                .on(level - 1);

                        for (std::size_t i = 0; i < field.size(); ++i)
                        {
                            subset.apply_op(
                                level - 1, balance_2to1(cell_flag[i], stencil));
                        }
                    }
                }
            }
        }

        for (std::size_t i = 0; i < field.size(); ++i)
        {
            cell_flag_global.array() |= cell_flag[i].array();
        }

        spdlog::info("Create new mesh");
        CellList<MRConfig> cell_list;
        for (std::size_t level = min_level; level <= max_level; ++level)
        {
            auto level_cell_array = mesh[MeshType::cells][level];

            if (!level_cell_array.empty())
            {
                level_cell_array.for_each_interval_in_x([&](auto const
                                                                &index_yz,
                                                            auto const
                                                                &interval) {
                    for (int i = interval.start; i < interval.end; ++i)
                    {
                        if (cell_flag_global.array()[i + interval.index] &
                            static_cast<int>(CellFlag::refine))
                        {
                            static_nested_loop<dim - 1, 0, 2>(
                                [&](auto stencil) {
                                    auto index =
                                        xt::eval(2 * index_yz + stencil);
                                    cell_list[level + 1][index].add_point(2 *
                                                                          i);
                                    cell_list[level + 1][index].add_point(
                                        2 * i + 1);
                                });
                        }
                        else if (cell_flag_global.array()[i + interval.index] &
                                 static_cast<int>(CellFlag::keep))
                        {
                            cell_list[level][index_yz].add_point(i);
                        }
                        else if (cell_flag_global.array()[i + interval.index] &
                                 static_cast<int>(CellFlag::coarsen))
                        {
                            if (level != min_level)
                            {
                                cell_list[level - 1][index_yz >> 1].add_point(
                                    i >> 1);
                            }
                            else
                            {
                                cell_list[level][index_yz].add_point(i);
                            }
                        }
                    }
                });
            }
        }

        Mesh<MRConfig> new_mesh{cell_list, mesh.initial_mesh(),
                                min_level, max_level};

        std::vector<Field<MRConfig>> new_field;

        for (std::size_t i = 0; i < field_size; ++i)
        {
            new_field.push_back({field[i].name(), new_mesh});
            new_field[i].array().fill(0);
        }

        spdlog::info("Copy the old field into the new field");
        for (std::size_t level = min_level; level <= max_level; ++level)
        {
            auto subset = intersection(mesh[MeshType::all_cells][level],
                                       new_mesh[MeshType::cells][level]);
            for (std::size_t i = 0; i < field_size; ++i)
            {
                subset.apply_op(level, copy(new_field[i], field[i]));
            }
        }

        spdlog::info("Compute the field of the refine cells using prediction");
        for (std::size_t level = min_level; level < max_level; ++level)
        {
            auto level_cell_array = mesh[MeshType::cells][level];

            if (!level_cell_array.empty())
            {
                level_cell_array.for_each_interval_in_x(
                    [&](auto const &index_yz, auto const &interval) {
                        for (int i = interval.start; i < interval.end; ++i)
                        {
                            if (cell_flag_global.array()[i + interval.index] &
                                static_cast<int>(CellFlag::refine))
                            {
                                compute_prediction(level, interval_t{i, i + 1},
                                                   index_yz, field, new_field);
                            }
                        }
                    });
            }
        }

        field[0].mesh_ptr()->swap(new_mesh);
        for (std::size_t i = 0; i < field_size; ++i)
        {
            field[i].array() = new_field[i].array();
        }
    }
}
