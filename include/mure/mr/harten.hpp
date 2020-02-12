#pragma once

#include "../field.hpp"
#include "cell_flag.hpp"
#include "mesh.hpp"

namespace mure
{
    template<class MRConfig>
    void harten(Field<MRConfig> &field, double eps, std::size_t ite)
    {
        constexpr auto max_refinement_level = MRConfig::max_refinement_level;
        constexpr auto dim = MRConfig::dim;
        using interval_t = typename MRConfig::interval_t;

        auto mesh = field.mesh();
        std::size_t max_level =
            mesh.initial_level(); // mesh[MeshType::cells].max_level();
        std::size_t min_level = mesh[MeshType::cells].min_level();

        xt::xtensor_fixed<double, xt::xshape<max_refinement_level + 1>>
            max_detail;
        max_detail.fill(std::numeric_limits<double>::min());

        Field<MRConfig, double> detail{"detail", mesh};
        Field<MRConfig, int> cell_flag{"flag", mesh};

        cell_flag.array().fill(0);
        mesh.for_each_cell([&](auto &cell) {
            cell_flag[cell] = static_cast<int>(CellFlag::keep);
        });

        // Compute the detail
        for (std::size_t level = min_level - 1; level < max_level - ite;
             ++level)
        {
            auto subset = intersection(mesh[MeshType::all_cells][level],
                                       mesh[MeshType::cells][level + 1])
                              .on(level);

            subset.apply_op(level, compute_detail(detail, field),
                            compute_max_detail(detail, max_detail));
        }

        for (std::size_t level = 0; level < max_level - ite; ++level)
        {
            int exponent = dim * (level - max_level + 1);

            auto eps_l = std::pow(2, exponent) * eps;

            auto subset_1 = intersection(mesh[MeshType::all_cells][level],
                                         mesh[MeshType::cells][level + 1])
                                .on(level);

            subset_1.apply_op(level,
                              to_coarsen(cell_flag, detail, max_detail, eps_l));

            // Harten
            auto subset_2 = intersection(mesh[MeshType::cells][level + 1],
                                         mesh[MeshType::cells][level + 1]);
            subset_2.apply_op(
                level + 1,
                to_refine(cell_flag, detail, max_detail, max_level, 4 * eps_l));

            subset_2.apply_op(level + 1, enlarge(cell_flag, CellFlag::keep));

            auto subset_right =
                intersection(
                    mesh[MeshType::cells][level],
                    translate_in_x<1>(mesh[MeshType::cells][level + 1]))
                    .on(level);

            subset_right([&](auto &index_yz, auto &interval, auto &) {
                auto i = interval[0];
                auto mask = cell_flag(level + 1, 2 * i) &
                            static_cast<int>(CellFlag::keep);
                xt::masked_view(cell_flag(level, i), mask) =
                    static_cast<int>(CellFlag::refine);
            });

            auto subset_left =
                intersection(
                    mesh[MeshType::cells][level],
                    translate_in_x<-1>(mesh[MeshType::cells][level + 1]))
                    .on(level);

            subset_left([&](auto &index_yz, auto &interval, auto &) {
                auto i = interval[0];
                auto mask = cell_flag(level + 1, 2 * i + 1) &
                            static_cast<int>(CellFlag::keep);
                xt::masked_view(cell_flag(level, i), mask) =
                    static_cast<int>(CellFlag::refine);
            });
        }

        for (std::size_t level = max_level; level > 0; --level)
        {
            auto keep_subset =
                intersection(mesh[MeshType::cells][level],
                             mesh[MeshType::all_cells][level - 1])
                    .on(level - 1);

            keep_subset.apply_op(level - 1, maximum(cell_flag));
            // 1D
            if (dim == 1)
            {
                auto subset_right =
                    intersection(
                        mesh[MeshType::cells][level],
                        translate_in_x<1>(mesh[MeshType::cells][level - 1]))
                        .on(level - 1);

                subset_right([&](auto &index_yz, auto &interval, auto &) {
                    auto i = interval[0];
                    cell_flag(level - 1, i - 1) |= cell_flag(level - 1, i);
                });

                auto subset_left =
                    intersection(
                        mesh[MeshType::cells][level],
                        translate_in_x<-1>(mesh[MeshType::cells][level - 1]))
                        .on(level - 1);

                subset_left([&](auto &index_yz, auto &interval, auto &) {
                    auto i = interval[0];
                    cell_flag(level - 1, i + 1) |= cell_flag(level - 1, i);
                });
            }
        }

        CellList<MRConfig> cell_list;
        for (std::size_t level = 0; level <= max_level; ++level)
        {
            auto level_cell_array = mesh[MeshType::cells][level];

            if (!level_cell_array.empty())
            {
                level_cell_array.for_each_interval_in_x(
                    [&](auto const &index_yz, auto const &interval) {
                        for (int i = interval.start; i < interval.end; ++i)
                        {
                            if (cell_flag.array()[i + interval.index] &
                                static_cast<int>(CellFlag::refine))
                            {
                                cell_list[level + 1][{}].add_point(2 * i);
                                cell_list[level + 1][{}].add_point(2 * i + 1);
                            }
                            else if (cell_flag.array()[i + interval.index] &
                                     static_cast<int>(CellFlag::keep))
                            {
                                cell_list[level][index_yz].add_point(i);
                            }
                            else if (cell_flag.array()[i + interval.index] &
                                     static_cast<int>(CellFlag::coarsen))
                            {
                                cell_list[level - 1][index_yz >> 1].add_point(
                                    i >> 1);
                            }
                        }
                    });
            }
        }

        Mesh<MRConfig> new_mesh{cell_list, mesh.initial_mesh(),
                                mesh.initial_level()};

        Field<MRConfig> new_field(field.name(), new_mesh);
        new_field.array().fill(0);

        for (std::size_t level = 0; level <= max_level; ++level)
        {
            auto subset = intersection(mesh[MeshType::all_cells][level],
                                       new_mesh[MeshType::cells][level]);

            subset.apply_op(level, copy(new_field, field));
        }

        for (std::size_t level = 0; level < max_level; ++level)
        {
            auto level_cell_array = mesh[MeshType::cells][level];

            if (!level_cell_array.empty())
            {
                level_cell_array.for_each_interval_in_x(
                    [&](auto const &index_yz, auto const &interval) {
                        for (int i = interval.start; i < interval.end; ++i)
                        {
                            if (cell_flag.array()[i + interval.index] &
                                static_cast<int>(CellFlag::refine))

                            {
                                auto ii = i << 1;
                                interval_t i_levelp1 = {ii, ii + 1};
                                interval_t i_level{i, i + 1};

                                auto qs_i = Qs_i<1>(field, level, i_level);

                                new_field(level + 1, i_levelp1) =
                                    (field(level, i_level) + qs_i);

                                new_field(level + 1, i_levelp1 + 1) =
                                    (field(level, i_level) - qs_i);
                            }
                        }
                    });
            }
        }

        field.mesh_ptr()->swap(new_mesh);
        field.array() = new_field.array();
    }

    template<class MRConfig>
    void harten(std::vector<Field<MRConfig>> &field, double eps,
                std::size_t ite, std::size_t min_level)
    {
        constexpr auto max_refinement_level = MRConfig::max_refinement_level;
        constexpr auto dim = MRConfig::dim;
        using interval_t = typename MRConfig::interval_t;

        auto mesh = field[0].mesh();
        std::size_t max_level =
            mesh.initial_level(); // mesh[MeshType::cells].max_level();

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

            for (std::size_t i = 0; i < field.size(); ++i)
            {
                subset_1.apply_op(level, to_coarsen(cell_flag[i], detail[i],
                                                    max_detail[i], eps_l));

                subset_2.apply_op(level + 1, to_refine(cell_flag[i], detail[i],
                                                       max_detail[i], max_level,
                                                       4 * eps_l));

                subset_2.apply_op(level + 1,
                                  enlarge(cell_flag[i], CellFlag::keep));
            }

            auto subset_right =
                intersection(
                    mesh[MeshType::cells][level],
                    translate_in_x<1>(mesh[MeshType::cells][level + 1]))
                    .on(level);

            subset_right([&](auto &index_yz, auto &interval, auto &) {
                auto i = interval[0];
                for (std::size_t ii = 0; ii < field.size(); ++ii)
                {
                    auto mask = cell_flag[ii](level + 1, 2 * i) &
                                static_cast<int>(CellFlag::keep);
                    xt::masked_view(cell_flag[ii](level, i), mask) =
                        static_cast<int>(CellFlag::refine);
                }
            });

            auto subset_left =
                intersection(
                    mesh[MeshType::cells][level],
                    translate_in_x<-1>(mesh[MeshType::cells][level + 1]))
                    .on(level);

            subset_left([&](auto &index_yz, auto &interval, auto &) {
                auto i = interval[0];
                for (std::size_t ii = 0; ii < field.size(); ++ii)
                {
                    auto mask = cell_flag[ii](level + 1, 2 * i + 1) &
                                static_cast<int>(CellFlag::keep);
                    xt::masked_view(cell_flag[ii](level, i), mask) =
                        static_cast<int>(CellFlag::refine);
                }
            });
        }

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
            // 1D
            if (dim == 1)
            {
                auto subset_right =
                    intersection(
                        mesh[MeshType::cells][level],
                        translate_in_x<1>(mesh[MeshType::cells][level - 1]))
                        .on(level - 1);

                subset_right([&](auto &index_yz, auto &interval, auto &) {
                    auto i = interval[0];
                    for (std::size_t ii = 0; ii < field.size(); ++ii)
                    {
                        cell_flag[ii](level - 1, i - 1) |=
                            cell_flag[ii](level - 1, i);
                    }
                });

                auto subset_left =
                    intersection(
                        mesh[MeshType::cells][level],
                        translate_in_x<-1>(mesh[MeshType::cells][level - 1]))
                        .on(level - 1);

                subset_left([&](auto &index_yz, auto &interval, auto &) {
                    auto i = interval[0];
                    for (std::size_t ii = 0; ii < field.size(); ++ii)
                    {
                        cell_flag[ii](level - 1, i + 1) |=
                            cell_flag[ii](level - 1, i);
                    }
                });
            }
        }

        for (std::size_t i = 0; i < field.size(); ++i)
        {
            cell_flag_global.array() |= cell_flag[i].array();
        }

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
                        // std::cout
                        //     << level << " " << i << " "
                        //     << cell_flag_global.array()[i + interval.index]
                        //     << "\n";
                        if (cell_flag_global.array()[i + interval.index] &
                            static_cast<int>(CellFlag::refine))
                        {
                            ;
                            // std::cout << "refine\n";
                            cell_list[level + 1][{}].add_point(2 * i);
                            cell_list[level + 1][{}].add_point(2 * i + 1);
                        }
                        else if (cell_flag_global.array()[i + interval.index] &
                                 static_cast<int>(CellFlag::keep))
                        {
                            // std::cout << "keep\n";
                            cell_list[level][index_yz].add_point(i);
                        }
                        else if (cell_flag_global.array()[i + interval.index] &
                                 static_cast<int>(CellFlag::coarsen))
                        {
                            if (level != min_level)
                            {
                                // std::cout << "coarsen\n";
                                cell_list[level - 1][index_yz >> 1].add_point(
                                    i >> 1);
                            }
                            else
                            {
                                // std::cout << "keep\n";
                                cell_list[level][index_yz].add_point(i);
                            }
                        }
                    }
                });
            }
        }

        Mesh<MRConfig> new_mesh{cell_list, mesh.initial_mesh(),
                                mesh.initial_level()};

        std::vector<Field<MRConfig>> new_field;

        for (std::size_t i = 0; i < field_size; ++i)
        {
            new_field.push_back({field[i].name(), new_mesh});
            new_field[i].array().fill(0);
        }

        for (std::size_t level = min_level; level <= max_level; ++level)
        {
            auto subset = intersection(mesh[MeshType::all_cells][level],
                                       new_mesh[MeshType::cells][level]);
            for (std::size_t i = 0; i < field_size; ++i)
            {
                subset.apply_op(level, copy(new_field[i], field[i]));
            }
        }

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
                                auto ii = i << 1;
                                interval_t i_levelp1 = {ii, ii + 1};
                                interval_t i_level{i, i + 1};

                                for (std::size_t i_f = 0; i_f < field_size;
                                     ++i_f)
                                {
                                    auto qs_i =
                                        Qs_i<1>(field[i_f], level, i_level);

                                    new_field[i_f](level + 1, i_levelp1) =
                                        (field[i_f](level, i_level) + qs_i);

                                    new_field[i_f](level + 1, i_levelp1 + 1) =
                                        (field[i_f](level, i_level) - qs_i);
                                }
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

    template<class MRConfig>
    void harten_2(std::vector<Field<MRConfig>> &field, double eps,
                  std::size_t ite, std::size_t min_level)
    {
        constexpr auto max_refinement_level = MRConfig::max_refinement_level;
        constexpr auto dim = MRConfig::dim;
        using interval_t = typename MRConfig::interval_t;

        auto mesh = field[0].mesh();
        std::size_t max_level =
            mesh.initial_level(); // mesh[MeshType::cells].max_level();

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

            for (std::size_t i = 0; i < field.size(); ++i)
            {
                subset_1.apply_op(level, to_coarsen(cell_flag[i], detail[i],
                                                    max_detail[i], eps_l));

                subset_2.apply_op(level + 1, to_refine(cell_flag[i], detail[i],
                                                       max_detail[i], max_level,
                                                       4 * eps_l));

                subset_2.apply_op(level + 1,
                                  enlarge(cell_flag[i], CellFlag::keep));
            }

            auto subset = intersection(mesh[MeshType::cells][level],
                                       expand(mesh[MeshType::cells][level + 1]))
                              .on(level + 1);

            subset([&](auto &index_yz, auto &interval, auto &) {
                auto i = interval[0];
                for (std::size_t ii = 0; ii < field.size(); ++ii)
                {
                    auto mask = cell_flag[ii](level + 1, i) &
                                static_cast<int>(CellFlag::keep);
                    xt::masked_view(cell_flag[ii](level, i / 2), mask) =
                        static_cast<int>(CellFlag::refine);
                }
            });
        }

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
            // 1D
            if (dim == 1)
            {
                auto subset_right =
                    intersection(
                        mesh[MeshType::cells][level],
                        translate_in_x<1>(mesh[MeshType::cells][level - 1]))
                        .on(level - 1);

                subset_right([&](auto &index_yz, auto &interval, auto &) {
                    auto i = interval[0];
                    for (std::size_t ii = 0; ii < field.size(); ++ii)
                    {
                        cell_flag[ii](level - 1, i - 1) |=
                            cell_flag[ii](level - 1, i);
                    }
                });

                auto subset_left =
                    intersection(
                        mesh[MeshType::cells][level],
                        translate_in_x<-1>(mesh[MeshType::cells][level - 1]))
                        .on(level - 1);

                subset_left([&](auto &index_yz, auto &interval, auto &) {
                    auto i = interval[0];
                    for (std::size_t ii = 0; ii < field.size(); ++ii)
                    {
                        cell_flag[ii](level - 1, i + 1) |=
                            cell_flag[ii](level - 1, i);
                    }
                });
            }
        }

        for (std::size_t i = 0; i < field.size(); ++i)
        {
            cell_flag_global.array() |= cell_flag[i].array();
        }

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
                            cell_list[level + 1][{}].add_point(2 * i);
                            cell_list[level + 1][{}].add_point(2 * i + 1);
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
                                mesh.initial_level()};

        std::vector<Field<MRConfig>> new_field;

        for (std::size_t i = 0; i < field_size; ++i)
        {
            new_field.push_back({field[i].name(), new_mesh});
            new_field[i].array().fill(0);
        }

        for (std::size_t level = min_level; level <= max_level; ++level)
        {
            auto subset = intersection(mesh[MeshType::all_cells][level],
                                       new_mesh[MeshType::cells][level]);
            for (std::size_t i = 0; i < field_size; ++i)
            {
                subset.apply_op(level, copy(new_field[i], field[i]));
            }
        }

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
                                auto ii = i << 1;
                                interval_t i_levelp1 = {ii, ii + 1};
                                interval_t i_level{i, i + 1};

                                for (std::size_t i_f = 0; i_f < field_size;
                                     ++i_f)
                                {
                                    auto qs_i =
                                        Qs_i<1>(field[i_f], level, i_level);

                                    new_field[i_f](level + 1, i_levelp1) =
                                        (field[i_f](level, i_level) + qs_i);

                                    new_field[i_f](level + 1, i_levelp1 + 1) =
                                        (field[i_f](level, i_level) - qs_i);
                                }
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
