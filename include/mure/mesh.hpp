#pragma once

#include <iostream>
#include <algorithm>

#include <xtensor/xfixed.hpp>
#include <xtensor/xview.hpp>
#include <xtensor/xio.hpp>
#include <xtensor/xmasked_view.hpp>

#include "box.hpp"
#include "cell_list.hpp"
#include "cell_array.hpp"
#include "intervals_operator.hpp"
#include "static_algorithm.hpp"
#include "stencil.hpp"
#include "subset.hpp"

namespace mure
{
    template<class MRConfig>
    class Field;

    template<class MRConfig>
    class Mesh
    {
    public:

        static constexpr auto dim = MRConfig::dim;
        static constexpr auto max_refinement_level = MRConfig::max_refinement_level;
        static constexpr auto ghost_width = std::max(std::max(2*(int)MRConfig::graduation_width - 1,
                                                     (int)MRConfig::max_stencil_width),
                                                     (int)MRConfig::default_s_for_prediction);
        using index_t = typename MRConfig::index_t;
        using coord_index_t = typename MRConfig::coord_index_t;
        using point_t = typename Box<int, dim>::point_t;
        using interval_t = typename MRConfig::interval_t;

        Mesh(Mesh const&) = default;
        Mesh& operator=(Mesh const&) = default;

        Mesh(Box<double, dim> b, std::size_t init_level)
        {
            using box_t = Box<coord_index_t, dim>;
            point_t start = b.min_corner()*std::pow(2, init_level);
            point_t end = b.max_corner()*std::pow(2, init_level);

            m_cells[init_level] = {box_t{start, end}};
            m_cells_and_ghosts[init_level] = {box_t{start - 1, end + 1}};
            m_all_cells[init_level] = {box_t{start - 1 , end + 1}};
            m_all_cells[init_level-1] = {box_t{(start>>1) - 1 , (end>>1) + 1}};
            update_x0_and_nb_ghosts(m_nb_local_cells_and_ghosts, m_all_cells, m_cells_and_ghosts, m_cells);
            // update_ghost_nodes();
        }

        Mesh(const CellList<MRConfig>& dcl)
        {
            m_cells = {dcl};
            update_ghost_nodes();
        }

        void projection(Field<MRConfig> &field) const
        {
            // auto expr = intersection(difference(_1, _2), _3);
            auto expr = intersection(difference(_1, _2), difference(_3, difference(_4, _5)));
            for(std::size_t level = max_refinement_level; level >=1; --level)
            {
                auto set = mure::make_subset<MRConfig>(expr,
                                                       level-1,
                                                       {level-1, level-1, level, level, level},
                                                       m_all_cells[level-1],
                                                       m_cells[level-1],
                                                       m_all_cells[level],
                                                       m_cells_and_ghosts[level],
                                                       m_cells[level]);


                set.apply([&](auto& index, auto& interval, auto& interval_index)
                {
                    index[0] = interval.start;

                    // auto op = op_projection(index, interval, interval_index, level);
                    // op.apply(field);

                    auto i = interval;
                    field(level - 1, i) = .5*(field(level, 2*i) + field(level, 2*i + 1));
                });
            }
        }

        void prediction(Field<MRConfig> &field) const
        {
            auto expr = intersection(_1, _2);
            for(std::size_t level = 0; level < max_refinement_level; ++level)
            {
                auto set = mure::make_subset<MRConfig>(expr,
                                                       level, {level, level+1},
                                                       m_all_cells[level],
                                                       m_cells[level+1]);

                set.apply([&](auto& index, auto& interval, auto& interval_index)
                {
                    auto i = interval;
                    field(level + 1, 2*i) = field(level, i) - 1./8*(field(level, i + 1) - field(level, i - 1));
                    field(level + 1, 2*i + 1) = field(level, i) + 1./8*(field(level, i + 1) - field(level, i - 1));
                });
            }
        }

        void detail(Field<MRConfig> &detail, Field<MRConfig> &field)
        {
            auto expr = intersection(_1, _2);

            Field<MRConfig> keep{"keep", *this};
            keep.array().fill(1);

            for(std::size_t level = 0; level < max_refinement_level; ++level)
            {
                auto set = mure::make_subset<MRConfig>(expr,
                                                       level, {level, level+1},
                                                       m_all_cells[level],
                                                       m_cells[level+1]);

                double eps = 1e-4;
                int exponent = level + 1 - m_cells.max_level();
                auto eps_l = std::pow(2, exponent)*eps;
            
                set.apply([&](auto& index, auto& interval, auto& interval_index)
                {
                    auto i = interval;
                    detail(level + 1, 2*i) = field(level + 1, 2*i) - (field(level, i) - 1./8*(field(level, i + 1) - field(level, i - 1)));
                    detail(level + 1, 2*i + 1) = field(level + 1, 2*i + 1) - (field(level, i) + 1./8*(field(level, i + 1) - field(level, i - 1)));

                    auto ii = 2*interval;
                    ii.step = 1;
                    auto mask = xt::abs(detail(level + 1, ii)) < eps_l;
                    xt::masked_view(keep(level + 1, ii), mask) = 0;
                });
            }

            for(std::size_t level = max_refinement_level; level > 0; --level)
            {
                auto keep_expr = intersection(difference(_1, _2), _3);
                auto set_keep = mure::make_subset<MRConfig>(keep_expr,
                                                            level-1,
                                                            {level-1, level-1, level},
                                                            m_all_cells[level-1],
                                                            m_cells_and_ghosts[level-1],
                                                            m_all_cells[level]);

                set_keep.apply([&](auto& index, auto& interval, auto& interval_index)
                {
                    auto i = interval;
                    if (i.start&1)
                        i.start+=1;
                    if (i.end&1)
                        i.end-=1;
                    auto ii = 2*i;
                    keep(level - 1, i) = xt::maximum(keep(level, 2*i), keep(level, 2*i + 1));
                });

                // graded tree
                auto graded_expr_1 = intersection(difference(_1, _2), _3);
                auto graded_expr_2 = intersection(difference(_3, _4), _2);
                auto graded_expr = union_(graded_expr_1, graded_expr_2);

                auto set = mure::make_subset<MRConfig>(graded_expr,
                                                       level-1, {level, level, level-1, level-1},
                                                       m_cells_and_ghosts[level],
                                                       m_cells[level],
                                                       m_cells_and_ghosts[level-1],
                                                       m_cells[level-1]);

                set.apply([&](auto& index, auto& interval, auto& interval_index)
                {
                    auto i = interval;
                    if (interval.size() == 2)
                        if (xt::any(keep(level-1, i) < 1))
                            keep(level-1, i) = 1;
                });
            }

            CellList<MRConfig> cell_list;

            for(std::size_t level = 0; level <= max_refinement_level; ++level)
            {
                const LevelCellArray<MRConfig> &level_cell_array = m_cells[level];

                if (!level_cell_array.empty())
                {
                    level_cell_array.for_each_interval_in_x([&](auto const& index_yz, auto const& interval)
                    {
                        if (interval.start&1)
                        {
                            cell_list[level][index_yz].add_point(interval.start);
                        }

                        if (interval.end&1)
                        {
                            cell_list[level][index_yz].add_point(interval.end - 1);
                        }

                        auto start = (interval.start&1)? interval.start + 1: interval.start;
                        auto end = (interval.end&1)? interval.end - 1: interval.end;
                        for (int i = start; i < end; i+=2)
                        {
                            if ((keep.array()[i + interval.index] == 0) and (keep.array()[i + 1 + interval.index] == 0))
                            {
                                cell_list[level-1][index_yz/2].add_point(i >> 1);
                            }
                            else
                            {
                                cell_list[level][index_yz].add_point(i);
                                cell_list[level][index_yz].add_point(i+1);
                            }
                        }
                   });
                }
            }

            Mesh<MRConfig> new_mesh{cell_list};
            // std::cout << m_all_cells << "\n";
            // std::cout << field.array() << "\n";
            // std::cout << new_mesh.m_all_cells << "\n";
            // std::cout << new_mesh.m_cells << "\n";
            Field<MRConfig> new_field(field.name(), new_mesh);
            auto expr_update = intersection(_1, _2);

            for(std::size_t level = 0; level <= max_refinement_level; ++level)
            {
                auto set = mure::make_subset<MRConfig>(expr_update,
                                                       level, {level, level},
                                                       m_all_cells[level],
                                                       new_mesh.m_cells[level]);

                set.apply([&](auto& index, auto& interval, auto& interval_index)
                {
                    auto i = interval;
                    new_field(level, i) = field(level, i);
                });
            }

            // std::cout << new_mesh << "\n";
            m_cells = {cell_list};
            update_ghost_nodes();
            field.array() = new_field.array();
        }

        void coarsening() const
        {
            auto expr = union_(intersection(difference(_1, _2), _4),
                               intersection(difference(_3, _4), _2));
            for(std::size_t level = max_refinement_level - 1; level >= 0; --level)
            {
                auto set = mure::make_subset<MRConfig>(expr,
                                                       level, {level, level, level+1, level+1},
                                                       m_cells_and_ghosts[level],
                                                       m_cells[level],
                                                       m_cells_and_ghosts[level+1],
                                                       m_cells[level+1]);

                set.apply([&](auto& index_yz, auto& interval, auto& interval_index)
                         {
                            std::cout << level << " " << interval << "\n";
                         });
            }
        }

        inline std::size_t nb_cells() const
        {
            return m_cells.nb_cells();
        }

        inline std::size_t nb_total_cells() const
        {
            return m_all_cells.nb_cells();
        }

        auto const& get_cells(std::size_t i) const
        {
            return m_cells[i];
        }

        template<typename... T>
        auto get_interval(std::size_t level, interval_t interval, T... index) const
        {
            auto lca = m_all_cells[level];
            auto row = lca.find({interval.start, index... });
            return lca[0][row];
        }

        void to_stream(std::ostream &os) const
        {
            os << "Cells\n";
            m_cells.to_stream(os);
            os << "\nCells and ghosts\n";
            m_cells_and_ghosts.to_stream(os);
            os << "\nAll cells\n";
            m_all_cells.to_stream(os);
        }

        template<class Func>
        inline void for_each_cell(Func&& func) const
        {
            m_cells.for_each_cell(std::forward<Func>(func));
        }

    private:
        void update_ghost_nodes();
        void add_ng_ghosts_and_get_nb_leaves(CellList<MRConfig>& cell_list);
        void add_ghosts_for_level_m1(CellList<MRConfig>& cell_list);
        void update_x0_and_nb_ghosts(index_t& target_nb_local_cells_and_ghosts,
                                     CellArray<MRConfig>& target_all_cells,
                                     CellArray<MRConfig>& target_cells_and_ghosts,
                                     CellArray<MRConfig>& target_cells);

        CellArray<MRConfig> m_cells;
        CellArray<MRConfig> m_cells_and_ghosts;
        CellArray<MRConfig> m_all_cells;
        index_t m_nb_local_cells;
        index_t m_nb_local_cells_and_ghosts;
    };


    template<class MRConfig>
    void Mesh<MRConfig>::update_ghost_nodes() {
        // FIXME: m_nb_local_cells is set in add_ng_ghosts_and_get_nb_leaves
        //        Then, do we need this line??
        // m_nb_local_cells = _leaves.nb_cells();

        // +/- w ghosts in level + 0 and 1, computation of _nb_local_leaf_cells
        CellList<MRConfig> cell_list;
        add_ng_ghosts_and_get_nb_leaves(cell_list);

        m_cells_and_ghosts = {cell_list};

        // optionnal +/- w nodes in level - 1
        if (MRConfig::need_pred_from_proj)
        {
            add_ghosts_for_level_m1(cell_list);
        }

        // // // // get external (remote or replicated) cells needed to compute the ghosts.
        // // // // Updates also _leaves_to_recv and _leaves_to_send (with leaf indices at this stage)
        // // // std::vector<std::vector<Cell<Carac>>> needed_nodes( mpi->size() );
        // // // _add_needed_or_replicated_cells( dcl, needed_nodes );

        // compaction
        m_all_cells = {cell_list};

        // update of x0_indices, _leaf_to_ghost_indices, _nb_local_ghost_and_leaf_cells
        update_x0_and_nb_ghosts(m_nb_local_cells_and_ghosts, m_all_cells, m_cells_and_ghosts, m_cells);

        // // // update of _leaves_to_recv and _leaves_to_send (using needed_nodes)
        // // _update_leaves_to_send_and_recv( needed_nodes );

        // // // update of _extended_leavesv (union of leaves + remote leaves + symmetry leaves, with correct ghost indices)
        // // _update_of_extended_leaves( needed_nodes );
    }

    template<class MRConfig>
    void Mesh<MRConfig>::add_ng_ghosts_and_get_nb_leaves(CellList<MRConfig>& cell_list)
    {
        // +/- w nodes in the current level
        m_nb_local_cells = 0;
        for(std::size_t level = 0; level <= max_refinement_level; ++level)
        {
            const LevelCellArray<MRConfig> &level_cell_array = m_cells[level];
            if (!level_cell_array.empty())
            {
                LevelCellList<MRConfig> &level_cell_list = cell_list[level];

                level_cell_array.for_each_interval_in_x([&](xt::xtensor_fixed<coord_index_t, xt::xshape<dim-1>> const& index_yz,
                                                            interval_t const& interval)
                {
                    static_nested_loop<dim-1, -ghost_width, ghost_width+1>([&](auto stencil)
                    {
                        auto index = xt::eval(index_yz + stencil);
                        level_cell_list[index].add_interval({interval.start - static_cast<int>(ghost_width),
                                                             interval.end + static_cast<int>(ghost_width)});
                    });
                    m_nb_local_cells += interval.size();
                } );
            }
        }
    }

    template<class MRConfig>
    void Mesh<MRConfig>::add_ghosts_for_level_m1(CellList<MRConfig>& cell_list)
    {
        for(int level = 1; level <= max_refinement_level; ++level)
        {
            const LevelCellArray<MRConfig> &level_cell_array = m_cells[level];
            if (level_cell_array.empty() == false)
            {
                LevelCellList<MRConfig> &level_cell_list = cell_list[level - 1];
                constexpr index_t s = MRConfig::default_s_for_prediction;

                level_cell_array.for_each_interval_in_x([&](xt::xtensor_fixed<coord_index_t, xt::xshape<dim-1>> const& index_yz,
                                                            interval_t const& interval)
                {
                    static_nested_loop<dim-1, -s, s+1>([&](auto stencil)
                    {
                        level_cell_list[(index_yz >> 1) + stencil].add_interval({(interval.start >> 1) - static_cast<int>(s),
                                                                                 ((interval.end + 1) >> 1) + static_cast<int>(s)});
                    });
                });
            }
        }
    }

    template<class MRConfig>
    void Mesh<MRConfig>::update_x0_and_nb_ghosts(index_t& target_nb_local_cells_and_ghosts,
                                                 CellArray<MRConfig>& target_all_cells,
                                                 CellArray<MRConfig>& target_cells_and_ghosts,
                                                 CellArray<MRConfig>& target_cells)
    {
        // get x0_index for each ghost_and_leaf node
        index_t ghost_index = 0;
        for(int level = 0; level <= max_refinement_level; ++level )
        {
            target_all_cells[level].for_each_interval_in_x([&](auto& pos_yz, const interval_t& interval)
            {
                // FIXME: remove this const_cast !!
                const_cast<interval_t&>(interval).index = ghost_index - interval.start;
                // interval.index = ghost_index - interval.start;
                ghost_index += interval.size();
            } );
        }
        target_nb_local_cells_and_ghosts = ghost_index;

        // get x0_index for each leaf node (ranges are not the same than in target)
        // index_t cpt_leaf = 0;
        // m_cell_to_ghost_indices.resize(nb_local_cells());
        auto expr = intersection(_1, _2);
        for( int level = 0; level <= max_refinement_level; ++level )
        {
            if (!target_cells[level].empty())
            {
                auto set = mure::make_subset<MRConfig>(expr,
                                                    target_all_cells[level],
                                                    target_cells[level]);

                set.apply([&](auto& index_yz, auto& interval, auto& interval_index)
                {
                    target_cells[level][0][interval_index[0, 1]].index = target_all_cells[level][0][interval_index[0, 0]].index;
                });
            }

            if (!target_cells_and_ghosts[level].empty())
            {
                auto set = mure::make_subset<MRConfig>(expr,
                                                    target_all_cells[level],
                                                    target_cells_and_ghosts[level]);

                set.apply([&](auto& index_yz, auto& interval, auto& interval_index)
                {
                    target_cells_and_ghosts[level][0][interval_index[0, 1]].index = target_all_cells[level][0][interval_index[0, 0]].index;
                });
            }
        }
    }

    template<class MRConfig>
    std::ostream& operator<<(std::ostream& out, const Mesh<MRConfig>& mesh)
    {
        mesh.to_stream(out);
        return out;
    }
}