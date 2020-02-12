#pragma once

namespace mure
{
    template<class MRConfig>
    void refinement(Field<MRConfig> &detail, Field<MRConfig> &field, double eps)
    {
        using interval_t = typename MRConfig::interval_t;
        constexpr auto max_refinement_level = MRConfig::max_refinement_level;
        constexpr auto dim = MRConfig::dim;

        auto mesh = field.mesh();
        std::size_t max_level = mesh[MeshType::cells].max_level();

        xt::xtensor_fixed<double, xt::xshape<max_refinement_level + 1>>
            max_detail;
        max_detail.fill(std::numeric_limits<double>::min());

        Field<MRConfig, bool> refine{"refine", mesh};
        refine.array().fill(false);

        for (std::size_t level = 0; level < max_level; ++level)
        {
            auto subset = intersection(mesh[MeshType::all_cells][level],
                                       mesh[MeshType::cells][level + 1])
                              .on(level);

            int exponent = dim * (level - max_level + 1);

            auto eps_l = 4 * std::pow(2, exponent) * eps;

            subset.apply_op(level, compute_detail(detail, field));

            auto subset_ref = intersection(mesh[MeshType::cells][level + 1],
                                           mesh[MeshType::cells][level + 1]);

            subset_ref.apply_op(level + 1,
                                compute_max_detail_(detail, max_detail));

            subset_ref.apply_op(level + 1,
                                to_refine(refine, detail, max_detail, eps_l));
        }

        // 1D
        for (std::size_t level = max_level; level > 0; --level)
        {
            auto subset_right =
                intersection(
                    mesh[MeshType::cells][level],
                    translate_in_x<-1>(mesh[MeshType::cells][level - 1]))
                    .on(level - 1);

            subset_right([&](auto &index_yz, auto &interval, auto &) {
                auto i = interval[0];
                refine(level - 1, i + 1) |= refine(level, 2 * i + 1);
            });

            auto subset_left =
                intersection(translate_in_x<-1>(mesh[MeshType::cells][level]),
                             mesh[MeshType::cells][level - 1])
                    .on(level - 1);

            subset_left([&](auto &index_yz, auto &interval, auto &) {
                auto i = interval[0];
                refine(level - 1, i) |= refine(level, 2 * (i + 1));
            });

            // 2D
            // for (std::size_t level = max_refinement_level; level > 0;
            // --level)
            // {
            //     auto subset_right =
            //         intersection(
            //             mesh[MeshType::cells][level],
            //             translate_in_x<-1>(mesh[MeshType::cells][level - 1]))
            //             .on(level - 1);

            //     subset_right([&](auto &index_yz, auto &interval, auto &) {
            //         auto i = interval[0];
            //         auto j = index_yz[0];
            //         refine(level - 1, i + 1, j) |=
            //             refine(level, 2 * i + 1, 2 * j) |
            //             refine(level, 2 * i + 1, 2 * j + 1);
            //     });

            //     auto subset_left =
            //         intersection(translate_in_x<-1>(mesh[MeshType::cells][level]),
            //                      mesh[MeshType::cells][level - 1])
            //             .on(level - 1);

            //     subset_left([&](auto &index_yz, auto &interval, auto &) {
            //         auto i = interval[0];
            //         auto j = index_yz[0];
            //         refine(level - 1, i, j) |=
            //             refine(level, 2 * (i + 1), 2 * j) |
            //             refine(level, 2 * (i + 1), 2 * j + 1);
            //     });

            //     auto subset_down =
            //         intersection(translate_in_y<-1>(mesh[MeshType::cells][level]),
            //                      mesh[MeshType::cells][level - 1])
            //             .on(level - 1);

            //     subset_down([&](auto &index_yz, auto &interval, auto &) {
            //         auto i = interval[0];
            //         auto j = index_yz[0];
            //         refine(level - 1, i, j) |=
            //             refine(level, 2 * i, 2 * (j + 1)) |
            //             refine(level, 2 * i + 1, 2 * (j + 1));
            //     });

            //     auto subset_up =
            //         intersection(translate_in_y<1>(mesh[MeshType::cells][level]),
            //                      mesh[MeshType::cells][level - 1])
            //             .on(level - 1);

            //     subset_up([&](auto &index_yz, auto &interval, auto &) {
            //         auto i = interval[0];
            //         auto j = index_yz[0];
            //         refine(level - 1, i, j) |= refine(level, 2 * i, 2 * j -
            //         1) |
            //                                    refine(level, 2 * i + 1, 2 * j
            //                                    - 1);
            //     });
        }

        CellList<MRConfig> cell_list;
        for (std::size_t level = 0; level < max_level; ++level)
        {
            auto level_cell_array = mesh[MeshType::cells][level];

            if (!level_cell_array.empty())
            {
                level_cell_array.for_each_interval_in_x(
                    [&](auto const &index_yz, auto const &interval) {
                        for (int i = interval.start; i < interval.end; ++i)
                        {
                            if (refine.array()[i + interval.index])
                            {
                                cell_list[level + 1][{}].add_point(2 * i);
                                cell_list[level + 1][{}].add_point(2 * i + 1);
                            }
                            else
                            {
                                cell_list[level][index_yz].add_point(i);
                            }

                            // if (refine.array()[i + interval.index])
                            // {
                            //     cell_list[level + 1][2 *
                            //     index_yz].add_point(2 *
                            //                                                  i);
                            //     cell_list[level + 1][2 * index_yz].add_point(
                            //         2 * i + 1);
                            //     cell_list[level + 1][2 * index_yz + 1]
                            //         .add_point(2 * i);
                            //     cell_list[level + 1][2 * index_yz + 1]
                            //         .add_point(2 * i + 1);
                            // }
                            // else
                            // {
                            //     cell_list[level][index_yz].add_point(i);
                            // }
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
                            if (refine.array()[i + interval.index])
                            {
                                auto ii = i << 1;
                                interval_t i_levelp1 = {ii, ii + 1};
                                interval_t i_level{i, i + 1};

                                auto qs_i = Qs_i<1>(field, level, i_level);

                                new_field(level + 1, i_levelp1) =
                                    (field(level, i_level) + qs_i);

                                new_field(level + 1, i_levelp1 + 1) =
                                    (field(level, i_level) - qs_i);

                                // auto ii = i << 1;
                                // auto jj = index_yz[0] << 1;
                                // interval_t i_levelp1 = {ii, ii + 1};
                                // auto j_levelp1 = jj;

                                // auto j = index_yz[0];
                                // interval_t i_level{i, i + 1};
                                // auto j_level = j;

                                // auto qs_i =
                                //     Qs_i<1>(field, level, i_level, j_level);
                                // auto qs_j =
                                //     Qs_j<1>(field, level, i_level, j_level);
                                // auto qs_ij =
                                //     Qs_ij<1>(field, level, i_level, j_level);

                                // new_field(level + 1, i_levelp1, j_levelp1) =
                                //     (field(level, i_level, j_level) + qs_i +
                                //      qs_j - qs_ij);

                                // new_field(level + 1, i_levelp1 + 1,
                                // j_levelp1) =
                                //     (field(level, i_level, j_level) - qs_i +
                                //      qs_j + qs_ij);

                                // new_field(level + 1, i_levelp1, j_levelp1 +
                                // 1) =
                                //     (field(level, i_level, j_level) + qs_i -
                                //      qs_j + qs_ij);

                                // new_field(level + 1, i_levelp1 + 1,
                                //           j_levelp1 + 1) =
                                //     (field(level, i_level, j_level) - qs_i -
                                //      qs_j - qs_ij);
                            }
                        }
                    });
            }
        }

        field.mesh_ptr()->swap(new_mesh);
        field.array() = new_field.array();
    }

    template<class MRConfig>
    void refinement(std::vector<Field<MRConfig>> &detail, std::vector<Field<MRConfig>> &field, double eps)
    {
        using interval_t = typename MRConfig::interval_t;
        constexpr auto max_refinement_level = MRConfig::max_refinement_level;
        constexpr auto dim = MRConfig::dim;

        auto mesh = field[0].mesh();
        std::size_t max_level = mesh[MeshType::cells].max_level();

        std::size_t field_size = field.size();
        std::vector<xt::xtensor_fixed<double, xt::xshape<max_refinement_level + 1>>>
            max_detail(field_size);

        Field<MRConfig, bool> refine_global{"refine_global", mesh};
        refine_global.array().fill(false);

        std::vector<Field<MRConfig, bool>> refine;

        for (std::size_t i = 0; i < field_size; ++i)
        {
            max_detail[i].fill(std::numeric_limits<double>::min());
            std::stringstream str;
            str << "refine_" << i;
            refine.push_back({str.str().data(), mesh});
            refine[i].array().fill(false);
        }

        for (std::size_t level = 0; level < max_level-1; ++level)
        {
            auto subset = intersection(mesh[MeshType::all_cells][level],
                                       mesh[MeshType::cells][level + 1])
                              .on(level);

            int exponent = dim * (level - max_level + 1);

            auto eps_l = std::pow(2, exponent) * eps;

            for (std::size_t i = 0; i < field.size(); ++i)
            {
                subset.apply_op(level, compute_detail(detail[i], field[i]));
            }

            auto subset_ref = intersection(mesh[MeshType::cells][level + 1],
                                           mesh[MeshType::cells][level + 1]);

            for (std::size_t i = 0; i < field.size(); ++i)
            {
                subset_ref.apply_op(level + 1,
                                compute_max_detail_(detail[i], max_detail[i]));

                subset_ref.apply_op(level + 1,
                                to_refine(refine[i], detail[i], max_detail[i], eps_l));
            }
        }

        for (std::size_t i = 0; i < field.size(); ++i)
        {
            refine_global.array() |= refine[i].array();
        }

        // 1D
        for (std::size_t level = max_level; level > 0; --level)
        {
            auto subset_right =
                intersection(
                    mesh[MeshType::cells][level],
                    translate_in_x<-1>(mesh[MeshType::cells][level - 1]))
                    .on(level - 1);

            subset_right([&](auto &index_yz, auto &interval, auto &) {
                auto i = interval[0];
                refine_global(level - 1, i + 1) |= refine_global(level, 2 * i + 1);
            });

            auto subset_left =
                intersection(translate_in_x<-1>(mesh[MeshType::cells][level]),
                             mesh[MeshType::cells][level - 1])
                    .on(level - 1);

            subset_left([&](auto &index_yz, auto &interval, auto &) {
                auto i = interval[0];
                refine_global(level - 1, i) |= refine_global(level, 2 * (i + 1));
            });
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
                            if (refine_global.array()[i + interval.index])
                            {
                                cell_list[level + 1][{}].add_point(2 * i);
                                cell_list[level + 1][{}].add_point(2 * i + 1);
                            }
                            else
                            {
                                cell_list[level][{}].add_point(i);
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

        for (std::size_t level = 0; level <= max_level; ++level)
        {
            auto subset = intersection(mesh[MeshType::all_cells][level],
                                       new_mesh[MeshType::cells][level]);

            for (std::size_t i = 0; i < field_size; ++i)
            {
                subset.apply_op(level, copy(new_field[i], field[i]));
            }
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
                            if (refine_global.array()[i + interval.index])
                            {
                                auto ii = i << 1;
                                interval_t i_levelp1 = {ii, ii + 1};
                                interval_t i_level{i, i + 1};

                                for (std::size_t ifield = 0; ifield < field_size; ++ifield)
                                {
                                auto qs_i = Qs_i<1>(field[ifield], level, i_level);

                                new_field[ifield](level + 1, i_levelp1) =
                                    (field[ifield](level, i_level) + qs_i);

                                new_field[ifield](level + 1, i_levelp1 + 1) =
                                    (field[ifield](level, i_level) - qs_i);
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