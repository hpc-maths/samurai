#pragma once

namespace mure
{
    template<class MRConfig>
    void coarsening(Field<MRConfig> &detail, Field<MRConfig> &field, double eps,
                    std::size_t ite)
    {
        constexpr auto max_refinement_level = MRConfig::max_refinement_level;
        constexpr auto dim = MRConfig::dim;

        xt::xtensor_fixed<double, xt::xshape<max_refinement_level + 1>>
            max_detail;
        max_detail.fill(std::numeric_limits<double>::min());

        auto mesh = field.mesh();
        Field<MRConfig, bool> keep{"keep", mesh};

        keep.array().fill(false);
        mesh.for_each_cell([&](auto &cell) { keep[cell] = true; });

        for (std::size_t level = 0; level < max_refinement_level; ++level)
        {
            auto subset = intersection(mesh[MeshType::all_cells][level],
                                       mesh[MeshType::cells][level + 1])
                              .on(level);

            // int exponent =
            //     dim * (level - mesh[MeshType::cells].max_level() - 1);
            int exponent = dim * (level - max_refinement_level);

            auto eps_l = std::pow(2, exponent) * eps;

            subset.apply_op(level, compute_detail(detail, field),
                            compute_max_detail(detail, max_detail));

            subset.apply_op(level, to_coarsen(keep, detail, max_detail, eps_l));
        }

        for (std::size_t level = max_refinement_level; level > 0; --level)
        {
            auto keep_subset =
                intersection(mesh[MeshType::cells][level],
                             mesh[MeshType::all_cells][level - 1])
                    .on(level - 1);

            keep_subset.apply_op(level - 1, maximum(keep));

            // 1D
            auto subset_right =
                intersection(
                    mesh[MeshType::cells][level],
                    translate_in_x<1>(mesh[MeshType::cells][level - 1]))
                    .on(level - 1);

            subset_right([&](auto &index_yz, auto &interval, auto &) {
                auto i = interval[0];
                keep(level - 1, i - 1) |= keep(level - 1, i);
            });

            auto subset_left =
                intersection(
                    mesh[MeshType::cells][level],
                    translate_in_x<-1>(mesh[MeshType::cells][level - 1]))
                    .on(level - 1);

            subset_left([&](auto &index_yz, auto &interval, auto &) {
                auto i = interval[0];
                keep(level - 1, i + 1) |= keep(level - 1, i);
            });

            // 2D
            // auto subset_right =
            //     intersection(
            //         mesh[MeshType::cells][level],
            //         translate_in_x<1>(mesh[MeshType::cells][level - 1]))
            //         .on(level - 1);

            // subset_right([&](auto &index_yz, auto &interval, auto &) {
            //     auto i = interval[0];
            //     auto j = index_yz[0];
            //     keep(level - 1, i - 1, j) |= keep(level - 1, i, j);
            // });

            // auto subset_left =
            //     intersection(
            //         mesh[MeshType::cells][level],
            //         translate_in_x<-1>(mesh[MeshType::cells][level - 1]))
            //         .on(level - 1);

            // subset_left([&](auto &index_yz, auto &interval, auto &) {
            //     auto i = interval[0];
            //     auto j = index_yz[0];
            //     keep(level - 1, i + 1, j) |= keep(level - 1, i, j);
            // });

            // auto subset_down =
            //     intersection(translate_in_y<-1>(mesh[MeshType::cells][level]),
            //                  mesh[MeshType::cells][level - 1])
            //         .on(level - 1);

            // subset_down([&](auto &index_yz, auto &interval, auto &) {
            //     auto i = interval[0];
            //     auto j = index_yz[0];
            //     keep(level - 1, i, j) |= keep(level - 1, i, j + 1);
            // });

            // auto subset_up =
            //     intersection(translate_in_y<1>(mesh[MeshType::cells][level]),
            //                  mesh[MeshType::cells][level - 1])
            //         .on(level - 1);

            // subset_up([&](auto &index_yz, auto &interval, auto &) {
            //     auto i = interval[0];
            //     auto j = index_yz[0];
            //     keep(level - 1, i, j) |= keep(level - 1, i, j - 1);
            // });
        }

        CellList<MRConfig> cell_list;
        for (std::size_t level = 0; level <= max_refinement_level; ++level)
        {
            auto level_cell_array = mesh[MeshType::cells][level];

            if (!level_cell_array.empty())
            {
                level_cell_array.for_each_interval_in_x(
                    [&](auto const &index_yz, auto const &interval) {
                        for (int i = interval.start; i < interval.end; ++i)
                        {
                            if (keep.array()[i + interval.index])
                            {
                                cell_list[level][{}].add_point(i);
                            }
                            else
                            {
                                cell_list[level - 1][{}].add_point(i >> 1);
                            }
                            // if (keep.array()[i + interval.index])
                            // {
                            //     cell_list[level][index_yz].add_point(i);
                            // }
                            // else
                            // {
                            //     cell_list[level - 1][index_yz >>
                            //     1].add_point(
                            //         i >> 1);
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

        field.mesh_ptr()->swap(new_mesh);
        field.array() = new_field.array();
    }
}