#pragma once

namespace mure
{
    template<class MRConfig>
    void refinement(Field<MRConfig> &detail, Field<MRConfig> &field, double eps)
    {
        using interval_t = typename MRConfig::interval_t;
        constexpr auto max_refinement_level = MRConfig::max_refinement_level;
        constexpr auto dim = MRConfig::dim;

        xt::xtensor_fixed<double, xt::xshape<max_refinement_level + 1>>
            max_detail;
        max_detail.fill(std::numeric_limits<double>::min());

        auto mesh = field.mesh();
        Field<MRConfig, bool> refine{"refine", mesh};
        refine.array().fill(false);

        for (std::size_t level = 0; level < max_refinement_level; ++level)
        {
            auto subset = intersection(mesh[MeshType::all_cells][level],
                                       mesh[MeshType::cells][level + 1])
                              .on(level);

            int exponent =
                dim * (level - mesh[MeshType::cells].max_level() - 1);
            // int exponent = dim * (level - max_refinement_level);

            auto eps_l = std::pow(2, exponent) * eps;

            subset.apply_op(level, compute_detail(detail, field));

            auto subset_ref = intersection(mesh[MeshType::cells][level + 1],
                                           mesh[MeshType::cells][level + 1]);

            subset_ref.apply_op(level + 1,
                                compute_max_detail_(detail, max_detail));

            subset_ref.apply_op(level + 1,
                                to_refine(refine, detail, max_detail, eps_l));
        }

        // {
        //     auto h5file = mure::Hdf5("to_refine");
        //     h5file.add_mesh(mesh);
        //     h5file.add_field(refine);
        //     h5file.add_field(detail);
        //     h5file.add_field(field);
        //     // h5file.add_field_by_level(mesh, field);
        //     auto h5file_d = mure::Hdf5("to_refine_u");
        //     h5file_d.add_field_by_level(mesh, field);
        // }

        // 1D
        for (std::size_t level = max_refinement_level; level > 0; --level)
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
        for (std::size_t level = 0; level < max_refinement_level; ++level)
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

        for (std::size_t level = 0; level <= max_refinement_level; ++level)
        {
            auto subset = intersection(mesh[MeshType::all_cells][level],
                                       new_mesh[MeshType::cells][level]);

            subset.apply_op(level, copy(new_field, field));
        }

        for (std::size_t level = 0; level < max_refinement_level; ++level)
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
}