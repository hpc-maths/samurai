#pragma once

#include <algorithm>
#include <iostream>

#include <xtensor/xfixed.hpp>
#include <xtensor/xio.hpp>
#include <xtensor/xmasked_view.hpp>
#include <xtensor/xview.hpp>

#include "box.hpp"
#include "cell_array.hpp"
#include "cell_list.hpp"
#include "intervals_operator.hpp"
#include "operators.hpp"
#include "static_algorithm.hpp"
#include "subset.hpp"

#include "hdf5.hpp"
#include "mesh_type.hpp"

namespace mure
{
    template<class MRConfig, class value_t>
    class Field;

    template<class MRConfig, std::size_t dim>
    struct MeshCellsArray : private std::array<CellArray<MRConfig>, dim>
    {
        using base = std::array<CellArray<MRConfig>, dim>;
        using base::operator[];

        CellArray<MRConfig> const &operator[](MeshType mesh_type) const
        {
            return operator[](static_cast<std::size_t>(mesh_type));
        }

        CellArray<MRConfig> &operator[](MeshType mesh_type)
        {
            return operator[](static_cast<std::size_t>(mesh_type));
        }
    };

    template<class MRConfig>
    class Mesh {
      public:
        static constexpr auto dim = MRConfig::dim;
        static constexpr auto max_refinement_level =
            MRConfig::max_refinement_level;
        static constexpr auto ghost_width = std::max(
            std::max(2 * static_cast<int>(MRConfig::graduation_width) - 1,
                     static_cast<int>(MRConfig::max_stencil_width)),
            static_cast<int>(MRConfig::default_s_for_prediction));
        using index_t = typename MRConfig::index_t;
        using coord_index_t = typename MRConfig::coord_index_t;
        using point_t = typename Box<int, dim>::point_t;
        using interval_t = typename MRConfig::interval_t;

        Mesh(Mesh const &) = default;
        Mesh &operator=(Mesh const &) = default;

        Mesh(Box<double, dim> b, std::size_t init_level)
        : m_init_level{init_level}
        {
            using box_t = Box<coord_index_t, dim>;
            point_t start = b.min_corner() * std::pow(2, init_level);
            point_t end = b.max_corner() * std::pow(2, init_level);

            m_cells[MeshType::cells][init_level] = {box_t{start, end}};
            m_cells[MeshType::cells_and_ghosts][init_level] = {
                box_t{start - 1, end + 1}};
            m_cells[MeshType::all_cells][init_level] = {
                box_t{start - 1, end + 1}};
            m_cells[MeshType::all_cells][init_level - 1] = {
                box_t{(start >> 1) - 1, (end >> 1) + 1}};
            m_cells[MeshType::proj_cells][init_level - 1] = {
                box_t{(start >> 1), (end >> 1)}};
            m_init_cells = {box_t{start, end}};
            update_x0_and_nb_ghosts();
            // update_ghost_nodes();
        }

        Mesh(const CellList<MRConfig> &dcl)
        {
            m_cells[MeshType::cells] = {dcl};
            update_ghost_nodes();
        }

        void projection(Field<MRConfig> &field) const
        {
            auto expr = intersection(_1, _2);

            for (std::size_t level = max_refinement_level - 1; level >= 1;
                 --level)
            {
                if (!m_cells[MeshType::proj_cells][level - 1].empty())
                {
                    std::array<LevelCellArray<MRConfig>, 2> set_array{
                        m_cells[MeshType::all_cells][level],
                        m_cells[MeshType::proj_cells][level - 1]};
                    auto set = make_subset<MRConfig>(
                        expr, level - 1, {level, level - 1}, set_array);

                    set.apply([&](auto &index, auto &interval, auto &) {
                        auto op =
                            Projection<MRConfig>(level - 1, index, interval[0]);
                        op.apply(field);
                    });
                }
            }
        }

        void prediction(Field<MRConfig> &field) const
        {
            auto expr = intersection(_1, _2);
            for (std::size_t level = 0; level < max_refinement_level; ++level)
            {
                auto set = mure::make_subset<MRConfig>(
                    expr, level, {level, level + 1},
                    m_cells[MeshType::all_cells][level],
                    m_cells[MeshType::cells][level + 1]);

                set.apply([&](auto & /*index*/, auto &interval,
                              auto & /*interval_index*/) {
                    auto i = interval[0];
                    field(level + 1, 2 * i) =
                        field(level, i) -
                        1. / 8 * (field(level, i + 1) - field(level, i - 1));
                    field(level + 1, 2 * i + 1) =
                        field(level, i) +
                        1. / 8 * (field(level, i + 1) - field(level, i - 1));
                });
            }
        }

        void refinment(Field<MRConfig> &detail, Field<MRConfig> &field,
                       std::size_t /*ite*/)
        {}

        void coarsening(Field<MRConfig> &detail, Field<MRConfig> &field,
                        std::size_t ite)
        {

            Field<MRConfig, bool> keep{"keep", *this};
            Field<MRConfig, bool> keep_old{"keep_old", *this};
            xt::xtensor_fixed<double, xt::xshape<max_refinement_level + 1>>
                max_detail;
            max_detail.fill(std::numeric_limits<double>::min());
            keep.array().fill(false);

            for_each_cell([&](auto &cell) { keep[cell] = true; });

            for (std::size_t level = 0; level < max_refinement_level; ++level)
            {
                auto expr = intersection(_1, _2);
                std::array<LevelCellArray<MRConfig>, 2> set_array{
                    m_cells[MeshType::all_cells][level],
                    m_cells[MeshType::cells][level + 1]};

                auto set = mure::make_subset<MRConfig>(
                    expr, level, {level, level + 1}, set_array);

                double eps = 1e-2;
                std::size_t exponent =
                    dim * (m_cells[MeshType::cells].max_level() - level - 1);
                auto eps_l = std::pow(2, -exponent) * eps;

                set.apply([&](auto &index, auto &interval,
                              auto & /*interval_index*/) {
                    auto op = Detail_op<MRConfig>(level, index, interval[0]);
                    op.compute_detail(detail, field);
                    op.compute_max_detail(max_detail, detail);
                });

                set.apply([&](auto &index, auto &interval,
                              auto & /*interval_index*/) {
                    auto op = Detail_op<MRConfig>(level, index, interval[0]);
                    op.to_coarsen(keep, detail, max_detail, eps_l);
                });
            }

            // {
            //     std::stringstream ss1;
            //     ss1 << "detail_" << ite;
            //     auto h5file = Hdf5(ss1.str().data());
            //     h5file.add_mesh(*this);
            //     h5file.add_field(detail);
            // }

            // {
            //     std::stringstream ss1;
            //     ss1 << "demo_to_coarsen_all_" << ite;
            //     auto h5file = Hdf5(ss1.str().data(), 1);
            //     h5file.add_field_by_level(*this, keep);
            //     std::stringstream ss2;
            //     ss2 << "demo_to_coarsen_" << ite;
            //     auto h5file2 = Hdf5(ss2.str().data());
            //     h5file2.add_field_by_level(*this, keep);
            //     std::stringstream ss3;
            //     ss3 << "demo_to_coarsen_proj" << ite;
            //     auto h5file3 = Hdf5(ss3.str().data(), 2);
            //     h5file3.add_field_by_level(*this, keep);
            // }

            for (std::size_t level = max_refinement_level; level > 0; --level)
            {
                auto keep_expr = intersection(_1, _2);
                std::array<LevelCellArray<MRConfig>, 2> keep_set_array{
                    m_cells[MeshType::cells][level],
                    m_cells[MeshType::all_cells][level - 1]};

                auto keep_set = make_subset<MRConfig>(
                    keep_expr, level - 1, {level, level - 1}, keep_set_array);

                // if (!m_cells[MeshType::all_cells][level].empty())
                // {
                //     std::stringstream ss1;
                //     ss1 << "demo_before_graded_" << ite << "_level_"
                //         << level - 1;
                //     auto h5file = Hdf5(ss1.str().data(),
                //     MeshType::all_cells); h5file.add_field_by_level(*this,
                //     keep);
                // }

                keep_set.apply([&](auto &index, auto &interval,
                                   auto & /*interval_index*/) {
                    auto op = Maximum<MRConfig>(level - 1, index, interval[0]);
                    op.apply(keep);
                    auto op_graded =
                        Graded_op<MRConfig>(level - 1, index, interval);
                    op_graded.apply(keep);
                });

                // if (!m_cells[MeshType::all_cells][level].empty())
                // {
                //     std::stringstream ss1;
                //     ss1 << "demo_maximum_graded_" << ite << "_level_"
                //         << level - 1;
                //     auto h5file = Hdf5(ss1.str().data(),
                //     MeshType::all_cells); h5file.add_field_by_level(*this,
                //     keep);
                // }

                // keep_old = keep;
                // keep_set.apply([&](auto &index, auto &interval,
                //                    auto & /*interval_index*/) {
                //     auto op_graded =
                //         Graded_op<MRConfig>(level - 1, index, interval);
                //     // op_graded.apply(keep, keep_old);
                //     op_graded.apply(keep);
                // });

                // if (!m_cells[MeshType::all_cells][level].empty())
                // {
                //     std::stringstream ss1;
                //     ss1 << "demo_graded_" << ite << "_level_" << level - 1;
                //     auto h5file = Hdf5(ss1.str().data(),
                //     MeshType::all_cells); h5file.add_field_by_level(*this,
                //     keep);
                // }

                auto clean_expr = difference(_1, _2);
                std::array<LevelCellArray<MRConfig>, 2> clean_set_array{
                    m_cells[MeshType::all_cells][level - 1],
                    m_cells[MeshType::cells][level - 1]};
                auto clean_set = mure::make_subset<MRConfig>(
                    clean_expr, level - 1, {level - 1, level - 1},
                    clean_set_array);

                clean_set.apply([&](auto &index, auto &interval,
                                    auto & /*interval_index*/) {
                    auto op = Clean<MRConfig>(level - 1, index, interval[0]);
                    op.apply(keep);
                });

                // if (!m_cells[MeshType::all_cells][level].empty())
                // {
                //     std::stringstream ss1;
                //     ss1 << "demo_clean_" << ite << "_level_" << level - 1;
                //     auto h5file = Hdf5(ss1.str().data(),
                //     MeshType::all_cells); h5file.add_field_by_level(*this,
                //     keep);
                // }
            }

            // {
            //     std::stringstream ss1;
            //     ss1 << "demo_graded_all_" << ite;
            //     auto h5file = Hdf5(ss1.str().data(), 1);
            //     h5file.add_field_by_level(*this, keep);
            //     std::stringstream ss2;
            //     ss2 << "demo_graded_" << ite;
            //     auto h5file2 = Hdf5(ss2.str().data());
            //     h5file2.add_field_by_level(*this, keep);
            //     std::stringstream ss3;
            //     ss3 << "demo_to_graded_proj" << ite;
            //     auto h5file3 = Hdf5(ss3.str().data(), 2);
            //     h5file3.add_field_by_level(*this, keep);
            // }

            // for (std::size_t level = 0; level <= max_refinement_level;
            // ++level)
            // {
            //     const LevelCellArray<MRConfig> &level_cell_array =
            //         m_cells[MeshType::all_cells][level];

            //     if (!level_cell_array.empty())
            //     {
            //         auto clean_expr = difference(_1, _2);
            //         std::array<LevelCellArray<MRConfig>, 2> set_array{
            //             m_cells[MeshType::all_cells][level],
            //             m_cells[MeshType::cells][level]};
            //         auto clean_set = mure::make_subset<MRConfig>(
            //             clean_expr, level, {level, level}, set_array);

            //         clean_set.apply([&](auto &index, auto &interval,
            //                             auto & /*interval_index*/) {
            //             auto op = Clean<MRConfig>(level, index, interval[0]);
            //             op.apply(keep);
            //         });
            //     }
            // }

            // {
            //     std::stringstream ss1;
            //     ss1 << "demo_clean_all_" << ite;
            //     auto h5file = Hdf5(ss1.str().data(), 1);
            //     h5file.add_field_by_level(*this, keep);
            //     std::stringstream ss2;
            //     ss2 << "demo_clean_" << ite;
            //     auto h5file2 = Hdf5(ss2.str().data());
            //     h5file2.add_field_by_level(*this, keep);
            //     std::stringstream ss3;
            //     ss3 << "demo_clean_proj" << ite;
            //     auto h5file3 = Hdf5(ss3.str().data(), 2);
            //     h5file3.add_field_by_level(*this, keep);
            // }

            for (std::size_t level = max_refinement_level; level > 0; --level)
            {
                auto keep_expr = intersection(_1, _2);
                std::array<LevelCellArray<MRConfig>, 2> set_array{
                    m_cells[MeshType::all_cells][level - 1],
                    m_cells[MeshType::cells][level]};
                auto keep_set = make_subset<MRConfig>(
                    keep_expr, level - 1, {level - 1, level}, set_array);

                keep_set.apply([&](auto &index, auto &interval,
                                   auto & /*interval_index*/) {
                    auto op = Test<MRConfig>(level, index, interval[0]);
                    op.apply(keep);
                });
            }

            // {
            //     std::stringstream ss1;
            //     ss1 << "demo_to_keep_all_" << ite;
            //     auto h5file = Hdf5(ss1.str().data(), 1);
            //     h5file.add_field_by_level(*this, keep);
            //     std::stringstream ss2;
            //     ss2 << "demo_to_keep_" << ite;
            //     auto h5file2 = Hdf5(ss2.str().data());
            //     h5file2.add_field_by_level(*this, keep);
            //     std::stringstream ss3;
            //     ss3 << "demo_to_keep_proj" << ite;
            //     auto h5file3 = Hdf5(ss3.str().data(), 2);
            //     h5file3.add_field_by_level(*this, keep);
            // }

            CellList<MRConfig> cell_list;

            for (std::size_t level = 0; level <= max_refinement_level; ++level)
            {
                const LevelCellArray<MRConfig> &level_cell_array =
                    m_cells[MeshType::all_cells][level];

                if (!level_cell_array.empty())
                {
                    level_cell_array.for_each_interval_in_x(
                        [&](auto const &index_yz, auto const &interval) {
                            for (int i = interval.start; i < interval.end; ++i)
                            {
                                if (keep.array()[static_cast<std::size_t>(
                                        i + interval.index)])
                                    cell_list[level][index_yz].add_point(i);
                            }
                        });
                }
            }

            Mesh<MRConfig> new_mesh{cell_list};
            Field<MRConfig> new_field(field.name(), new_mesh);
            new_field.array().fill(0);
            auto expr_update = intersection(_1, _2);

            for (std::size_t level = 0; level <= max_refinement_level; ++level)
            {
                std::array<LevelCellArray<MRConfig>, 2> set_array{
                    m_cells[MeshType::all_cells][level],
                    new_mesh.m_cells[MeshType::cells][level]};
                auto set = mure::make_subset<MRConfig>(
                    expr_update, level, {level, level}, set_array);

                set.apply([&](auto &index, auto &interval,
                              auto & /*interval_index*/) {
                    auto op = Copy<MRConfig>(level, index, interval[0]);
                    op.apply(new_field, field);
                });
            }

            // std::stringstream ss1;
            // ss1 << "demo_all_" << ite;
            // auto h5file = Hdf5(ss1.str().data(), 1);
            // h5file.add_field_by_level(*this, {detail, field});
            // std::stringstream ss2;
            // ss2 << "demo_cells_" << ite;
            // auto h5file2 = Hdf5(ss2.str().data());
            // h5file2.add_field_by_level(*this, {detail, field});

            m_cells = std::move(new_mesh.m_cells);
            // m_all_cells = std::move(new_mesh.m_all_cells);
            // m_cells_and_ghosts = std::move(new_mesh.m_cells_and_ghosts);
            // m_proj_cells = std::move(new_mesh.m_proj_cells);
            // std::cout << *this << "\n";
            field.array() = new_field.array();
            // {
            //     std::stringstream ss1;
            //     ss1 << "demo_new_mesh_all_" << ite;
            //     auto h5file = Hdf5(ss1.str().data(), MeshType::all_cells);
            //     h5file.add_field_by_level(*this, field);
            //     std::stringstream ss2;
            //     ss2 << "demo_new_mesh_" << ite;
            //     auto h5file2 = Hdf5(ss2.str().data());
            //     h5file2.add_field_by_level(*this, field);
            //     std::stringstream ss3;
            //     ss3 << "demo_new_mesh_proj" << ite;
            //     auto h5file3 = Hdf5(ss3.str().data(), MeshType::proj_cells);
            //     h5file3.add_field_by_level(*this, field);
            // }
        }

        void coarsening() const
        {
            auto expr = union_(intersection(difference(_1, _2), _4),
                               intersection(difference(_3, _4), _2));
            for (int level = max_refinement_level - 1; level >= 0; --level)
            {
                auto set = mure::make_subset<MRConfig>(
                    expr, level, {level, level, level + 1, level + 1},
                    m_cells[MeshType::cells_and_ghosts][level],
                    m_cells[MeshType::cells][level],
                    m_cells[MeshType::cells_and_ghosts][level + 1],
                    m_cells[MeshType::cells][level + 1]);

                // set.apply([&](auto& index_yz, auto& interval, auto&
                // /*interval_index*/)
                //          {
                //             std::cout << level << " " << interval[0] << "\n";
                //          });
            }
        }

        inline std::size_t nb_cells(MeshType mesh_type) const
        {
            return m_cells[mesh_type].nb_cells();
        }

        inline std::size_t nb_cells(std::size_t level, MeshType mesh_type) const
        {
            return m_cells[mesh_type][level].nb_cells();
        }

        inline std::size_t nb_total_cells() const
        {
            return m_cells[MeshType::all_cells].nb_cells();
        }

        auto const &get_cells(std::size_t i) const
        {
            return m_cells[i];
        }

        template<typename... T>
        auto get_interval(std::size_t level, interval_t interval,
                          T... index) const
        {
            auto lca = m_cells[MeshType::all_cells][level];
            auto row = lca.find({interval.start, index...});
            return lca[0][static_cast<std::size_t>(row)];
        }

        void to_stream(std::ostream &os) const
        {
            os << "Cells\n";
            m_cells[MeshType::cells].to_stream(os);
            os << "\nCells and ghosts\n";
            m_cells[MeshType::cells_and_ghosts].to_stream(os);
            os << "\nProjection cells\n";
            m_cells[MeshType::proj_cells].to_stream(os);
            os << "\nAll cells\n";
            m_cells[MeshType::all_cells].to_stream(os);
        }

        template<class Func>
        inline void for_each_cell(Func &&func,
                                  MeshType mesh_type = MeshType::cells) const
        {
            m_cells[mesh_type].for_each_cell(std::forward<Func>(func));
        }

        template<class Func>
        inline void for_each_cell(std::size_t level, Func &&func,
                                  MeshType mesh_type = MeshType::cells) const
        {
            m_cells[mesh_type][level].for_each_cell(level,
                                                    std::forward<Func>(func));
        }

      private:
        void update_ghost_nodes();
        void add_ng_ghosts_and_get_nb_leaves(CellList<MRConfig> &cell_list);
        void add_ghosts_for_level_m1(CellList<MRConfig> &cell_list);
        void update_x0_and_nb_ghosts();

        MeshCellsArray<MRConfig, 4> m_cells;
        LevelCellArray<MRConfig> m_init_cells;
        std::size_t m_init_level;
    };

    template<class MRConfig>
    void Mesh<MRConfig>::update_ghost_nodes()
    {
        // +/- w ghosts in level + 0 and 1, computation of _nb_local_leaf_cells
        CellList<MRConfig> cell_list;
        add_ng_ghosts_and_get_nb_leaves(cell_list);

        m_cells[MeshType::cells_and_ghosts] = {cell_list};

        // optionnal +/- w nodes in level - 1
        if (MRConfig::need_pred_from_proj)
        {
            add_ghosts_for_level_m1(cell_list);
        }

        // compaction
        m_cells[MeshType::all_cells] = {cell_list};

        CellList<MRConfig> cell_list_1;
        for (std::size_t level = max_refinement_level - 1; level > 0; --level)
        {
            if (!m_cells[MeshType::all_cells][level - 1].empty())
            {
                std::array<LevelCellArray<MRConfig>, 7> set_array{
                    m_cells[MeshType::all_cells][level],
                    m_cells[MeshType::all_cells][level + 1],
                    m_cells[MeshType::cells][level],
                    m_cells[MeshType::all_cells][level - 1],
                    m_cells[MeshType::all_cells][level],
                    m_cells[MeshType::cells][level - 1],
                    m_init_cells};
                auto expr = intersection(intersection(union_(intersection(_1, _2), _3),
                                                      union_(intersection(_4, _5), _6)), _7);
                auto set = make_subset<MRConfig>(
                    expr, level - 1,
                    {level, level + 1, level, level - 1, level, level - 1, m_init_level},
                    set_array);
                set.apply([&](auto &index_yz, auto &interval,
                              auto & /*interval_index*/) {
                    cell_list_1[level - 1]
                               [xt::eval(xt::view(index_yz, xt::range(1, dim)))]
                                   .add_interval(
                                       {interval[0].start, interval[0].end});
                });
            }
        }
        m_cells[MeshType::proj_cells] = {cell_list_1};

        // update of x0_indices, _leaf_to_ghost_indices,
        // _nb_local_ghost_and_leaf_cells
        update_x0_and_nb_ghosts();
    }

    template<class MRConfig>
    void Mesh<MRConfig>::add_ng_ghosts_and_get_nb_leaves(
        CellList<MRConfig> &cell_list)
    {
        // +/- w nodes in the current level
        for (std::size_t level = 0; level <= max_refinement_level; ++level)
        {
            const LevelCellArray<MRConfig> &level_cell_array =
                m_cells[MeshType::cells][level];
            if (!level_cell_array.empty())
            {
                LevelCellList<MRConfig> &level_cell_list = cell_list[level];

                level_cell_array.for_each_interval_in_x(
                    [&](xt::xtensor_fixed<coord_index_t,
                                          xt::xshape<dim - 1>> const &index_yz,
                        interval_t const &interval) {
                        static_nested_loop<dim - 1, -ghost_width,
                                           ghost_width + 1>([&](auto stencil) {
                            auto index = xt::eval(index_yz + stencil);
                            level_cell_list[index].add_interval(
                                {interval.start - static_cast<int>(ghost_width),
                                 interval.end + static_cast<int>(ghost_width)});
                        });
                    });
            }
        }
    }

    template<class MRConfig>
    void Mesh<MRConfig>::add_ghosts_for_level_m1(CellList<MRConfig> &cell_list)
    {
        for (std::size_t level = 1; level <= max_refinement_level; ++level)
        {
            const LevelCellArray<MRConfig> &level_cell_array =
                m_cells[MeshType::cells][level];
            if (level_cell_array.empty() == false)
            {
                LevelCellList<MRConfig> &level_cell_list = cell_list[level - 1];
                constexpr index_t s = MRConfig::default_s_for_prediction;

                level_cell_array.for_each_interval_in_x(
                    [&](xt::xtensor_fixed<coord_index_t,
                                          xt::xshape<dim - 1>> const &index_yz,
                        interval_t const &interval) {
                        static_nested_loop<dim - 1, -s, s + 1>(
                            [&](auto stencil) {
                                level_cell_list[(index_yz >> 1) + stencil]
                                    .add_interval({(interval.start >> 1) -
                                                       static_cast<int>(s),
                                                   ((interval.end + 1) >> 1) +
                                                       static_cast<int>(s)});
                            });
                    });
            }
        }
    }

    template<class MRConfig>
    void Mesh<MRConfig>::update_x0_and_nb_ghosts()
    {
        // get x0_index for each ghost_and_leaf node
        std::size_t ghost_index = 0;
        for (std::size_t level = 0; level <= max_refinement_level; ++level)
        {
            m_cells[MeshType::all_cells][level].for_each_interval_in_x(
                [&](auto &, const interval_t &interval) {
                    // FIXME: remove this const_cast !!
                    const_cast<interval_t &>(interval).index =
                        static_cast<index_t>(ghost_index) - interval.start;
                    // interval.index = ghost_index - interval.start;
                    ghost_index += interval.size();
                });
        }

        // get x0_index for each leaf node (ranges are not the same than in
        // target) index_t cpt_leaf = 0;
        // m_cell_to_ghost_indices.resize(nb_local_cells());
        auto expr = intersection(_1, _2);
        for (std::size_t level = 0; level <= max_refinement_level; ++level)
        {
            if (!m_cells[MeshType::cells][level].empty())
            {
                std::array<LevelCellArray<MRConfig>, 2> set_array{
                    m_cells[MeshType::all_cells][level],
                    m_cells[MeshType::cells][level]};
                auto set = mure::make_subset<MRConfig>(
                    expr, level, {level, level}, set_array);

                set.apply([&](auto & /*index_yz*/, auto & /*interval*/,
                              auto &interval_index) {
                    m_cells[MeshType::cells][level][0]
                           [static_cast<std::size_t>(interval_index(0, 1))]
                               .index =
                        m_cells[MeshType::all_cells][level][0]
                               [static_cast<std::size_t>(interval_index(0, 0))]
                                   .index;
                });
            }

            if (!m_cells[MeshType::cells_and_ghosts][level].empty())
            {
                std::array<LevelCellArray<MRConfig>, 2> set_array{
                    m_cells[MeshType::all_cells][level],
                    m_cells[MeshType::cells_and_ghosts][level]};
                auto set = mure::make_subset<MRConfig>(
                    expr, level, {level, level}, set_array);

                set.apply([&](auto & /*index_yz*/, auto & /*interval*/,
                              auto &interval_index) {
                    m_cells[MeshType::cells_and_ghosts][level][0]
                           [static_cast<std::size_t>(interval_index(0, 1))]
                               .index =
                        m_cells[MeshType::all_cells][level][0]
                               [static_cast<std::size_t>(interval_index(0, 0))]
                                   .index;
                });
            }

            if (!m_cells[MeshType::proj_cells][level].empty())
            {
                std::array<LevelCellArray<MRConfig>, 2> set_array{
                    m_cells[MeshType::all_cells][level],
                    m_cells[MeshType::proj_cells][level]};
                auto set = mure::make_subset<MRConfig>(
                    expr, level, {level, level}, set_array);

                set.apply([&](auto & /*index_yz*/, auto & /*interval*/,
                              auto &interval_index) {
                    m_cells[MeshType::proj_cells][level][0]
                           [static_cast<std::size_t>(interval_index(0, 1))]
                               .index =
                        m_cells[MeshType::all_cells][level][0]
                               [static_cast<std::size_t>(interval_index(0, 0))]
                                   .index;
                });
            }
        }
    }

    template<class MRConfig>
    std::ostream &operator<<(std::ostream &out, const Mesh<MRConfig> &mesh)
    {
        mesh.to_stream(out);
        return out;
    }
}
