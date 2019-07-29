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
#include "operators.hpp"
#include "static_algorithm.hpp"
#include "subset/subset_op.hpp"

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

            m_cells[MeshType::cells][init_level] = {init_level,
                                                    box_t{start, end}};
            m_cells[MeshType::cells_and_ghosts][init_level] = {
                init_level, box_t{start - 1, end + 1}};
            m_cells[MeshType::all_cells][init_level] = {
                init_level, box_t{start - 1, end + 1}};
            m_cells[MeshType::all_cells][init_level - 1] = {
                init_level - 1, box_t{(start >> 1) - 1, (end >> 1) + 1}};
            m_cells[MeshType::proj_cells][init_level - 1] = {
                init_level - 1, box_t{(start >> 1), (end >> 1)}};
            m_init_cells = {init_level, box_t{start, end}};
            update_x0_and_nb_ghosts();
            // update_ghost_nodes();
        }

        Mesh(const CellList<MRConfig> &dcl,
             const LevelCellArray<dim> &init_cells, std::size_t init_level)
            : m_init_cells{init_cells}, m_init_level{init_level}
        {
            m_cells[MeshType::cells] = {dcl};
            update_ghost_nodes();
        }

        void make_projection(Field<MRConfig> &field) const
        {

            for (std::size_t level = max_refinement_level; level >= 1; --level)
            {
                auto expr =
                    intersection(m_cells[MeshType::all_cells][level],
                                 m_cells[MeshType::proj_cells][level - 1])
                        .on(level - 1);
                expr.apply_op(level - 1, projection(field));
            }
        }

        void make_prediction(Field<MRConfig> &field) const
        {
            for (std::size_t level = 1; level <= max_refinement_level; ++level)
            {

                if (!m_cells[MeshType::cells][level].empty())
                {
                    auto expr =
                        intersection(
                            difference(
                                m_cells[MeshType::all_cells][level],
                                union_(m_cells[MeshType::cells][level],
                                       m_cells[MeshType::proj_cells][level])),
                            m_init_cells)
                            .on(level);

                    expr.apply_op(level, prediction(field));
                }
            }
        }

        void refinment(Field<MRConfig> &detail, Field<MRConfig> &field,
                       std::size_t /*ite*/)
        {}

        void coarsening(Field<MRConfig> &detail, Field<MRConfig> &field,
                        std::size_t ite)
        {

            Field<MRConfig, bool> keep{"keep", *this};
            xt::xtensor_fixed<double, xt::xshape<max_refinement_level + 1>>
                max_detail;
            max_detail.fill(std::numeric_limits<double>::min());
            keep.array().fill(false);

            for_each_cell([&](auto &cell) { keep[cell] = true; });

            for (std::size_t level = 0; level < max_refinement_level; ++level)
            {
                auto subset = intersection(m_cells[MeshType::all_cells][level],
                                           m_cells[MeshType::cells][level + 1])
                                  .on(level);

                double eps = 1e-2;
                int exponent =
                    dim * (level - m_cells[MeshType::cells].max_level() - 1);
                auto eps_l = std::pow(2, exponent) * eps;

                subset.apply_op(level, compute_detail(detail, field),
                                compute_max_detail(detail, max_detail));

                subset.apply_op(level,
                                to_coarsen(keep, detail, max_detail, eps_l));
            }

            for (std::size_t level = max_refinement_level; level > 0; --level)
            {
                auto keep_subset =
                    intersection(m_cells[MeshType::cells][level],
                                 m_cells[MeshType::all_cells][level - 1])
                        .on(level - 1);

                keep_subset.apply_op(level - 1, maximum(keep));

                auto subset_right =
                    intersection(
                        m_cells[MeshType::cells][level],
                        translate_in_x<1>(m_cells[MeshType::cells][level - 1]))
                        .on(level - 1);

                subset_right([&](auto &index_yz, auto &interval, auto &) {
                    auto i = interval[0];
                    auto j = index_yz[0];
                    keep(level - 1, i - 1, j) |= keep(level - 1, i, j);
                });

                auto subset_left =
                    intersection(
                        m_cells[MeshType::cells][level],
                        translate_in_x<-1>(m_cells[MeshType::cells][level - 1]))
                        .on(level - 1);

                subset_left([&](auto &index_yz, auto &interval, auto &) {
                    auto i = interval[0];
                    auto j = index_yz[0];
                    keep(level - 1, i + 1, j) |= keep(level - 1, i, j);
                });

                auto subset_down =
                    intersection(
                        translate_in_y<-1>(m_cells[MeshType::cells][level]),
                        m_cells[MeshType::cells][level - 1])
                        .on(level - 1);

                subset_down([&](auto &index_yz, auto &interval, auto &) {
                    auto i = interval[0];
                    auto j = index_yz[0];
                    keep(level - 1, i, j) |= keep(level - 1, i, j + 1);
                });

                auto subset_up =
                    intersection(
                        translate_in_y<1>(m_cells[MeshType::cells][level]),
                        m_cells[MeshType::cells][level - 1])
                        .on(level - 1);

                subset_up([&](auto &index_yz, auto &interval, auto &) {
                    auto i = interval[0];
                    auto j = index_yz[0];
                    keep(level - 1, i, j) |= keep(level - 1, i, j - 1);
                });
            }

            CellList<MRConfig> cell_list;
            for (std::size_t level = 0; level <= max_refinement_level; ++level)
            {
                const LevelCellArray<dim> &level_cell_array =
                    m_cells[MeshType::cells][level];

                if (!level_cell_array.empty())
                {
                    level_cell_array.for_each_interval_in_x(
                        [&](auto const &index_yz, auto const &interval) {
                            for (int i = interval.start; i < interval.end; ++i)
                            {
                                if (keep.array()[i + interval.index])
                                {
                                    cell_list[level][index_yz].add_point(i);
                                }
                                else
                                {
                                    cell_list[level - 1][index_yz >> 1]
                                        .add_point(i >> 1);
                                }
                            }
                        });
                }
            }

            Mesh<MRConfig> new_mesh{cell_list, m_init_cells, m_init_level};
            Field<MRConfig> new_field(field.name(), new_mesh);
            new_field.array().fill(0);

            for (std::size_t level = 0; level <= max_refinement_level; ++level)
            {
                auto subset =
                    intersection(m_cells[MeshType::all_cells][level],
                                 new_mesh.m_cells[MeshType::cells][level]);

                subset.apply_op(level, copy(new_field, field));
            }

            m_cells = std::move(new_mesh.m_cells);
            field.array() = new_field.array();
        }

        void refinement(Field<MRConfig> &detail, Field<MRConfig> &field,
                        std::size_t ite)
        {

            Field<MRConfig, bool> refine{"refine", *this};
            xt::xtensor_fixed<double, xt::xshape<max_refinement_level + 1>>
                max_detail;
            max_detail.fill(std::numeric_limits<double>::min());
            refine.array().fill(false);

            for (std::size_t level = 0; level < max_refinement_level - 1;
                 ++level)
            {
                auto subset = intersection(m_cells[MeshType::all_cells][level],
                                           m_cells[MeshType::cells][level + 1])
                                  .on(level);

                double eps = 1e-2;
                int exponent =
                    dim * (level - m_cells[MeshType::cells].max_level() - 1);
                auto eps_l = std::pow(2, exponent) * eps;

                subset.apply_op(level, compute_detail(detail, field),
                                compute_max_detail(detail, max_detail));

                auto subset_ref =
                    intersection(m_cells[MeshType::cells][level + 1],
                                 m_cells[MeshType::cells][level + 1]);

                subset_ref.apply_op(
                    level + 1, to_refine(refine, detail, max_detail, eps_l));
            }

            for (std::size_t level = max_refinement_level; level > 0; --level)
            {
                // auto refine_subset =
                //     intersection(m_cells[MeshType::cells][level],
                //                  m_cells[MeshType::all_cells][level - 1])
                //         .on(level - 1);

                // refine_subset.apply_op(level - 1, maximum_ref(refine));

                auto subset_right =
                    intersection(
                        m_cells[MeshType::cells][level],
                        translate_in_x<-1>(m_cells[MeshType::cells][level - 1]))
                        .on(level - 1);

                subset_right([&](auto &index_yz, auto &interval, auto &) {
                    auto i = interval[0];
                    auto j = index_yz[0];
                    refine(level - 1, i + 1, j) |=
                        refine(level, 2 * i + 1, 2 * j) |
                        refine(level, 2 * i + 1, 2 * j + 1);
                });

                auto subset_left =
                    intersection(
                        translate_in_x<-1>(m_cells[MeshType::cells][level]),
                        m_cells[MeshType::cells][level - 1])
                        .on(level - 1);

                subset_left([&](auto &index_yz, auto &interval, auto &) {
                    auto i = interval[0];
                    auto j = index_yz[0];
                    refine(level - 1, i, j) |=
                        refine(level, 2 * (i + 1), 2 * j) |
                        refine(level, 2 * (i + 1), 2 * j + 1);
                });

                auto subset_down =
                    intersection(
                        translate_in_y<-1>(m_cells[MeshType::cells][level]),
                        m_cells[MeshType::cells][level - 1])
                        .on(level - 1);

                subset_down([&](auto &index_yz, auto &interval, auto &) {
                    auto i = interval[0];
                    auto j = index_yz[0];
                    refine(level - 1, i, j) |=
                        refine(level, 2 * i, 2 * (j + 1)) |
                        refine(level, 2 * i + 1, 2 * (j + 1));
                });

                auto subset_up =
                    intersection(
                        translate_in_y<1>(m_cells[MeshType::cells][level]),
                        m_cells[MeshType::cells][level - 1])
                        .on(level - 1);

                subset_up([&](auto &index_yz, auto &interval, auto &) {
                    auto i = interval[0];
                    auto j = index_yz[0];
                    refine(level - 1, i, j) |=
                        refine(level, 2 * i, 2 * j - 1) |
                        refine(level, 2 * i + 1, 2 * j - 1);
                });
            }

            CellList<MRConfig> cell_list;
            for (std::size_t level = 0; level <= max_refinement_level; ++level)
            {
                const LevelCellArray<dim> &level_cell_array =
                    m_cells[MeshType::cells][level];

                if (!level_cell_array.empty())
                {
                    level_cell_array.for_each_interval_in_x(
                        [&](auto const &index_yz, auto const &interval) {
                            for (int i = interval.start; i < interval.end; ++i)
                            {
                                if (refine.array()[i + interval.index])
                                {
                                    cell_list[level + 1][2 * index_yz]
                                        .add_point(2 * i);
                                    cell_list[level + 1][2 * index_yz]
                                        .add_point(2 * i + 1);
                                    cell_list[level + 1][2 * index_yz + 1]
                                        .add_point(2 * i);
                                    cell_list[level + 1][2 * index_yz + 1]
                                        .add_point(2 * i + 1);
                                }
                                else
                                {
                                    cell_list[level][index_yz].add_point(i);
                                }
                            }
                        });
                }
            }

            Mesh<MRConfig> new_mesh{cell_list, m_init_cells, m_init_level};
            Field<MRConfig> new_field(field.name(), new_mesh);
            new_field.array().fill(0);

            for (std::size_t level = 0; level <= max_refinement_level; ++level)
            {
                auto subset =
                    intersection(m_cells[MeshType::all_cells][level],
                                 new_mesh.m_cells[MeshType::cells][level]);

                subset.apply_op(level, copy(new_field, field));
            }

            for (std::size_t level = 0; level <= max_refinement_level; ++level)
            {
                const LevelCellArray<dim> &level_cell_array =
                    m_cells[MeshType::cells][level];

                if (!level_cell_array.empty())
                {
                    level_cell_array.for_each_interval_in_x(
                        [&](auto const &index_yz, auto const &interval) {
                            for (int i = interval.start; i < interval.end; ++i)
                            {
                                if (refine.array()[i + interval.index])
                                {
                                    interval_t ii{i, i + 1};
                                    interval_t iv = 2 * ii;
                                    auto jj = index_yz[0];
                                    auto j = 2 * index_yz[0];
                                    auto qs_i =
                                        xt::eval(Qs_i<1>(field, level, ii, jj));
                                    auto qs_j =
                                        xt::eval(Qs_j<1>(field, level, ii, jj));
                                    auto qs_ij = xt::eval(
                                        Qs_ij<1>(field, level, ii, jj));

                                    new_field(level + 1, iv, j) =
                                        // detail(level, ii, jj) / 2 +
                                        (field(level, ii, jj) + qs_i + qs_j -
                                         qs_ij);

                                    new_field(level + 1, iv + 1, j) =
                                        // detail(level, ii, jj) / 2 +
                                        (field(level, ii, jj) - qs_i + qs_j +
                                         qs_ij);

                                    new_field(level + 1, iv, j + 1) =
                                        // detail(level, ii, jj) / 2 +
                                        (field(level, ii, jj) + qs_i - qs_j +
                                         qs_ij);

                                    new_field(level + 1, iv + 1, j + 1) =
                                        // detail(level, ii, jj) / 2 +
                                        (field(level, ii, jj) - qs_i - qs_j -
                                         qs_ij);
                                }
                            }
                        });
                }
            }

            m_cells = std::move(new_mesh.m_cells);
            field.array() = new_field.array();
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

        auto operator[](MeshType mesh_type) const
        {
            return m_cells[mesh_type];
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
        LevelCellArray<dim> m_init_cells;
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

        for (std::size_t level = max_refinement_level; level > 0; --level)
        {
            if (!m_cells[MeshType::cells][level].empty())
            {
                LevelCellList<dim, interval_t> lcl{level - 1};
                auto expr =
                    intersection(m_cells[MeshType::all_cells][level - 1],
                                 union_(m_cells[MeshType::cells][level],
                                        m_cells[MeshType::proj_cells][level]))
                        .on(level - 1);

                expr([&](auto &index_yz, auto &interval,
                         auto & /*interval_index*/) {
                    lcl[index_yz].add_interval(
                        {interval[0].start, interval[0].end});
                });
                m_cells[MeshType::proj_cells][level - 1] = {lcl};
            }
        }

        // update of x0_indices, _leaf_to_ghost_indices,
        update_x0_and_nb_ghosts();
    }

    template<class MRConfig>
    void Mesh<MRConfig>::add_ng_ghosts_and_get_nb_leaves(
        CellList<MRConfig> &cell_list)
    {
        // +/- w nodes in the current level
        for (std::size_t level = 0; level <= max_refinement_level; ++level)
        {
            const LevelCellArray<dim> &level_cell_array =
                m_cells[MeshType::cells][level];
            if (!level_cell_array.empty())
            {
                LevelCellList<dim> &level_cell_list = cell_list[level];

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
            const LevelCellArray<dim> &level_cell_array =
                m_cells[MeshType::cells][level];
            if (level_cell_array.empty() == false)
            {
                LevelCellList<dim> &level_cell_list = cell_list[level - 1];
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
        std::size_t ghost_index = 0;
        for (std::size_t level = 0; level <= max_refinement_level; ++level)
        {
            m_cells[MeshType::all_cells][level].for_each_interval_in_x(
                [&](auto &, const interval_t &interval) {
                    // FIXME: remove this const_cast !!
                    const_cast<interval_t &>(interval).index =
                        static_cast<index_t>(ghost_index) - interval.start;
                    ghost_index += interval.size();
                });
        }

        for (std::size_t level = 0; level <= max_refinement_level; ++level)
        {
            if (!m_cells[MeshType::cells][level].empty())
            {
                auto expr = intersection(m_cells[MeshType::all_cells][level],
                                         m_cells[MeshType::cells][level]);

                expr([&](auto & /*index_yz*/, auto & /*interval*/,
                         auto &interval_index) {
                    m_cells[MeshType::cells][level][0]
                           [static_cast<std::size_t>(interval_index[1])]
                               .index =
                        m_cells[MeshType::all_cells][level][0]
                               [static_cast<std::size_t>(interval_index[0])]
                                   .index;
                });
            }

            if (!m_cells[MeshType::cells_and_ghosts][level].empty())
            {
                auto expr =
                    intersection(m_cells[MeshType::all_cells][level],
                                 m_cells[MeshType::cells_and_ghosts][level]);

                expr([&](auto & /*index_yz*/, auto & /*interval*/,
                         auto &interval_index) {
                    m_cells[MeshType::cells_and_ghosts][level][0]
                           [static_cast<std::size_t>(interval_index[1])]
                               .index =
                        m_cells[MeshType::all_cells][level][0]
                               [static_cast<std::size_t>(interval_index[0])]
                                   .index;
                });
            }

            if (!m_cells[MeshType::proj_cells][level].empty())
            {
                auto expr = intersection(m_cells[MeshType::all_cells][level],
                                         m_cells[MeshType::proj_cells][level]);

                expr([&](auto & /*index_yz*/, auto & /*interval*/,
                         auto &interval_index) {
                    m_cells[MeshType::proj_cells][level][0]
                           [static_cast<std::size_t>(interval_index[1])]
                               .index =
                        m_cells[MeshType::all_cells][level][0]
                               [static_cast<std::size_t>(interval_index[0])]
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
