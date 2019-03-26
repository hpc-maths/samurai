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
#include "operators.hpp"
#include "static_algorithm.hpp"
#include "subset.hpp"

#include "hdf5.hpp"

namespace mure
{
    template<class MRConfig, class value_t>
    class Field;

    template<class MRConfig>
    class Mesh
    {
    public:

        static constexpr auto dim = MRConfig::dim;
        static constexpr auto max_refinement_level = MRConfig::max_refinement_level;
        static constexpr auto ghost_width = std::max(std::max(2*static_cast<int>(MRConfig::graduation_width) - 1,
                                                     static_cast<int>(MRConfig::max_stencil_width)),
                                                     static_cast<int>(MRConfig::default_s_for_prediction));
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
            m_proj_cells[init_level-1] = {box_t{(start>>1), (end>>1)}};
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

            for(std::size_t level = max_refinement_level-1; level >=1; --level)
            {
                if (!m_proj_cells[level-1].empty())
                {
                    auto expr = intersection(_1, _2);
                    auto set = make_subset<MRConfig>(expr,
                                                    level-1,
                                                    {level, level-1},
                                                    m_all_cells[level],
                                                    m_proj_cells[level-1]);

                    // auto expr = union_(intersection(_1, _2), _3);
                    // auto set = make_subset<MRConfig>(expr,
                    //                                         level-1,
                    //                                         {level, level+1, level},
                    //                                         m_all_cells[level],
                    //                                         m_all_cells[level+1],
                    //                                         m_cells[level]);

                    set.apply([&](auto& index, auto& interval, auto&)
                    {
                        auto op = Projection<MRConfig>(level-1, index, interval[0]);
                        op.apply(field);
                    });
                }
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

                set.apply([&](auto& /*index*/, auto& interval, auto& /*interval_index*/)
                {
                    auto i = interval[0];
                    field(level + 1, 2*i) = field(level, i) - 1./8*(field(level, i + 1) - field(level, i - 1));
                    field(level + 1, 2*i + 1) = field(level, i) + 1./8*(field(level, i + 1) - field(level, i - 1));
                });
            }
        }

        void detail(Field<MRConfig> &detail, Field<MRConfig> &field, std::size_t /*ite*/)
        {

            Field<MRConfig, bool> keep{"keep", *this};
            xt::xtensor_fixed<double, xt::xshape<max_refinement_level+1>> max_detail;
            max_detail.fill(std::numeric_limits<double>::min());
            keep.array().fill(false);

            for_each_cell([&](auto& cell){
                keep[cell] = true;
            });

            for(std::size_t level = 0; level < max_refinement_level; ++level)
            {
                auto expr = intersection(_1, _2);
                auto set = mure::make_subset<MRConfig>(expr,
                                                       level, {level, level+1},
                                                       m_all_cells[level],
                                                       m_cells[level+1]);

                double eps = 1e-2;
                std::size_t exponent = dim*(m_cells.max_level() - level - 1 );
                auto eps_l = std::pow(2, -exponent)*eps;
            
                set.apply([&](auto& index, auto& interval, auto& /*interval_index*/)
                {
                    auto op = Detail_op<MRConfig>(level, index, interval[0]);
                    op.compute_detail(detail, field);
                    op.compute_max_detail(max_detail, detail);
                });

                set.apply([&](auto& index, auto& interval, auto& /*interval_index*/)
                {
                    auto op = Detail_op<MRConfig>(level, index, interval[0]);
                    op.to_coarsen(keep, detail, max_detail, eps_l);
                });
            }

            // {
            //     std::stringstream ss1;
            //     ss1 << "demo_to_coarsen_all_" << ite;
            //     auto h5file = Hdf5(ss1.str().data(), 1);
            //     h5file.add_field_by_level(*this, keep);
            //     std::stringstream ss2;
            //     ss2 << "demo_to_coarsen_" << ite;
            //     auto h5file2 = Hdf5(ss2.str().data());
            //     h5file2.add_field_by_level(*this, keep);
            // }

            for(std::size_t level = max_refinement_level; level > 0; --level)
            {
                auto keep_expr = intersection(_1, _2);
                auto keep_set = make_subset<MRConfig>(keep_expr,
                                                level-1,
                                                {level, level-1},
                                                m_cells[level],
                                                m_all_cells[level-1]);

                keep_set.apply([&](auto& index, auto& interval, auto& /*interval_index*/)
                {
                    auto op = Maximum<MRConfig>(level-1, index, interval[0]);
                    op.apply(keep);
                    auto op_graded = Graded_op<MRConfig>(level-1, index, interval);
                    op_graded.apply(keep);
                });
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
            // }

            for(std::size_t level = 0; level <= max_refinement_level; ++level)
            {
                const LevelCellArray<MRConfig> &level_cell_array = m_all_cells[level];

                if (!level_cell_array.empty())
                {
                    auto clean_expr = difference(_1, _2);
                    auto clean_set = mure::make_subset<MRConfig>(clean_expr,
                                                                level,
                                                                {level, level},
                                                                m_all_cells[level],
                                                                m_cells[level]);

                    clean_set.apply([&](auto& index, auto& interval, auto& /*interval_index*/)
                    {
                        auto op = Clean<MRConfig>(level, index, interval[0]);
                        op.apply(keep);
                    });
                }
            }

            // {
            //     std::stringstream ss1;
            //     ss1 << "demo_clean_all_" << ite;
            //     auto h5file = Hdf5(ss1.str().data(), 1);
            //     h5file.add_field_by_level(*this, keep);
            //     std::stringstream ss2;
            //     ss2 << "demo_clean_" << ite;
            //     auto h5file2 = Hdf5(ss2.str().data());
            //     h5file2.add_field_by_level(*this, keep);
            // }

            for(std::size_t level = max_refinement_level; level > 0; --level)
            {
                auto keep_expr = intersection(_1, _2);
                auto keep_set = make_subset<MRConfig>(keep_expr,
                                                level-1,
                                                {level-1, level},
                                                m_all_cells[level-1],
                                                m_cells[level]);

                keep_set.apply([&](auto& index, auto& interval, auto& /*interval_index*/)
                {
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
            // }

            CellList<MRConfig> cell_list;

            for(std::size_t level = 0; level <= max_refinement_level; ++level)
            {
                const LevelCellArray<MRConfig> &level_cell_array = m_all_cells[level];

                if (!level_cell_array.empty())
                {
                    level_cell_array.for_each_interval_in_x([&](auto const& index_yz, auto const& interval)
                    {
                        for (int i = interval.start; i < interval.end; ++i)
                        {
                            if (keep.array()[static_cast<std::size_t>(i + interval.index)])
                                cell_list[level][index_yz].add_point(i);
                        }
                   });
                }
            }

            Mesh<MRConfig> new_mesh{cell_list};
            Field<MRConfig> new_field(field.name(), new_mesh);
            new_field.array().fill(0);
            auto expr_update = intersection(_1, _2);

            for(std::size_t level = 0; level <= max_refinement_level; ++level)
            {
                auto set = mure::make_subset<MRConfig>(expr_update,
                                                       level, {level, level},
                                                       m_all_cells[level],
                                                       new_mesh.m_cells[level]);

                set.apply([&](auto& index, auto& interval, auto& /*interval_index*/)
                {
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
            m_all_cells = std::move(new_mesh.m_all_cells);
            m_cells_and_ghosts = std::move(new_mesh.m_cells_and_ghosts);
            m_proj_cells = std::move(new_mesh.m_proj_cells);
            // std::cout << *this << "\n";
            field.array() = new_field.array();
        }

        void coarsening() const
        {
            auto expr = union_(intersection(difference(_1, _2), _4),
                               intersection(difference(_3, _4), _2));
            for(int level = max_refinement_level - 1; level >= 0; --level)
            {
                auto set = mure::make_subset<MRConfig>(expr,
                                                       level, {level, level, level+1, level+1},
                                                       m_cells_and_ghosts[level],
                                                       m_cells[level],
                                                       m_cells_and_ghosts[level+1],
                                                       m_cells[level+1]);

                // set.apply([&](auto& index_yz, auto& interval, auto& /*interval_index*/)
                //          {
                //             std::cout << level << " " << interval[0] << "\n";
                //          });
            }
        }

        inline std::size_t nb_cells(std::size_t type=0) const
        {
            if (type == 0)
                return m_cells.nb_cells();
            if (type == 1)
                return m_all_cells.nb_cells();
            // if (type == 2)
            return m_proj_cells.nb_cells();
        }

        inline std::size_t nb_cells_for_level(std::size_t level, std::size_t type=0) const
        {
            if (type == 0)
                return m_cells[level].nb_cells();
            if (type == 1)
                return m_all_cells[level].nb_cells();
            if (type == 2)
                return m_proj_cells[level].nb_cells();
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
            return lca[0][static_cast<std::size_t>(row)];
        }

        void to_stream(std::ostream &os) const
        {
            os << "Cells\n";
            m_cells.to_stream(os);
            os << "\nCells and ghosts\n";
            m_cells_and_ghosts.to_stream(os);
            os << "\nProjection cells\n";
            m_proj_cells.to_stream(os);
            os << "\nAll cells\n";
            m_all_cells.to_stream(os);
        }

        template<class Func>
        inline void for_each_cell(Func&& func, std::size_t type=0) const
        {
            if (type == 0)
                m_cells.for_each_cell(std::forward<Func>(func));
            if (type == 1)
                m_all_cells.for_each_cell(std::forward<Func>(func));
            if (type == 2)
                m_proj_cells.for_each_cell(std::forward<Func>(func));
        }

        template<class Func>
        inline void for_each_cell_on_level(std::size_t level, Func&& func, std::size_t type=0) const
        {
            if (type == 0)
                m_cells[level].for_each_cell(std::forward<Func>(func), level);
            if (type == 1)
                m_all_cells[level].for_each_cell(std::forward<Func>(func), level);
            if (type == 2)
                m_proj_cells[level].for_each_cell(std::forward<Func>(func), level);
        }

    private:
        void update_ghost_nodes();
        void add_ng_ghosts_and_get_nb_leaves(CellList<MRConfig>& cell_list);
        void add_ghosts_for_level_m1(CellList<MRConfig>& cell_list);
        void update_x0_and_nb_ghosts(std::size_t& target_nb_local_cells_and_ghosts,
                                     CellArray<MRConfig>& target_all_cells,
                                     CellArray<MRConfig>& target_cells_and_ghosts,
                                     CellArray<MRConfig>& target_cells);

        CellArray<MRConfig> m_cells;
        CellArray<MRConfig> m_cells_and_ghosts;
        CellArray<MRConfig> m_all_cells;
        CellArray<MRConfig> m_proj_cells;
        std::size_t m_nb_local_cells;
        std::size_t m_nb_local_cells_and_ghosts;
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

        CellList<MRConfig> cell_list_1;
        for(std::size_t level = max_refinement_level-1; level > 0; --level)
        {
            if (!m_all_cells[level-1].empty())
            {
                auto expr = intersection(union_(intersection(_1, _2), _3), union_(intersection(_4, _5), _6));
                auto set = make_subset<MRConfig>(expr,
                                                level-1,
                                                {level, level+1, level, level-1, level, level-1},
                                                m_all_cells[level],
                                                m_all_cells[level+1],
                                                m_cells[level],
                                                m_all_cells[level-1],
                                                m_all_cells[level],
                                                m_cells[level-1]);
                set.apply([&](auto& index_yz, auto& interval, auto& /*interval_index*/)
                {
                    cell_list_1[level-1][xt::eval(xt::view(index_yz, xt::range(1, dim)))].add_interval({interval[0].start, interval[0].end});
                });
            }
        }
        m_proj_cells = {cell_list_1};

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
        for(std::size_t level = 1; level <= max_refinement_level; ++level)
        {
            const LevelCellArray<MRConfig> &level_cell_array = m_cells[level];
            if (level_cell_array.empty() == false)
            {
                LevelCellList<MRConfig> &level_cell_list = cell_list[level - 1];
                constexpr index_t s = MRConfig::default_s_for_prediction;

                level_cell_array.for_each_interval_in_x([&](xt::xtensor_fixed<coord_index_t, xt::xshape<dim-1>> const& index_yz,
                                                            interval_t const& interval)
                {
                    static_nested_loop<dim-1, -2*s, 2*s+1>([&](auto stencil)
                    {
                        level_cell_list[(index_yz >> 1) + stencil].add_interval({(interval.start >> 1) - 2*static_cast<int>(s),
                                                                                 ((interval.end + 1) >> 1) + 2*static_cast<int>(s)});
                    });
                });
            }
        }
    }

    template<class MRConfig>
    void Mesh<MRConfig>::update_x0_and_nb_ghosts(std::size_t& target_nb_local_cells_and_ghosts,
                                                 CellArray<MRConfig>& target_all_cells,
                                                 CellArray<MRConfig>& target_cells_and_ghosts,
                                                 CellArray<MRConfig>& target_cells)
    {
        // get x0_index for each ghost_and_leaf node
        std::size_t ghost_index = 0;
        for(std::size_t level = 0; level <= max_refinement_level; ++level )
        {
            target_all_cells[level].for_each_interval_in_x([&](auto&, const interval_t& interval)
            {
                // FIXME: remove this const_cast !!
                const_cast<interval_t&>(interval).index = static_cast<index_t>(ghost_index) - interval.start;
                // interval.index = ghost_index - interval.start;
                ghost_index += interval.size();
            } );
        }
        target_nb_local_cells_and_ghosts = ghost_index;

        // get x0_index for each leaf node (ranges are not the same than in target)
        // index_t cpt_leaf = 0;
        // m_cell_to_ghost_indices.resize(nb_local_cells());
        auto expr = intersection(_1, _2);
        for(std::size_t level = 0; level <= max_refinement_level; ++level )
        {
            if (!target_cells[level].empty())
            {
                auto set = mure::make_subset<MRConfig>(expr,
                                                    level,
                                                    {level, level},
                                                    m_all_cells[level],
                                                    m_cells[level]);

                set.apply([&](auto& /*index_yz*/, auto& /*interval*/, auto& interval_index)
                {   
                    m_cells[level][0][static_cast<std::size_t>(interval_index(0, 1))].index = m_all_cells[level][0][static_cast<std::size_t>(interval_index(0, 0))].index;
                });
            }

            if (!target_cells_and_ghosts[level].empty())
            {
                auto set = mure::make_subset<MRConfig>(expr,
                                                    level,
                                                    {level, level},
                                                    target_all_cells[level],
                                                    target_cells_and_ghosts[level]);

                set.apply([&](auto& /*index_yz*/, auto& /*interval*/, auto& interval_index)
                {
                    target_cells_and_ghosts[level][0][static_cast<std::size_t>(interval_index(0, 1))].index = target_all_cells[level][0][static_cast<std::size_t>(interval_index(0, 0))].index;
                });
            }

            if (!m_proj_cells[level].empty())
            {
                auto set = mure::make_subset<MRConfig>(expr,
                                                    level,
                                                    {level, level},
                                                       target_all_cells[level],
                                                       m_proj_cells[level]);

                set.apply([&](auto& /*index_yz*/, auto& /*interval*/, auto& interval_index)
                {
                    m_proj_cells[level][0][static_cast<std::size_t>(interval_index(0, 1))].index = target_all_cells[level][0][static_cast<std::size_t>(interval_index(0, 0))].index;
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