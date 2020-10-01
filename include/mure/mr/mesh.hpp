#pragma once

#include <algorithm>
#include <iostream>

#include <xtensor/xfixed.hpp>
#include <xtensor/xio.hpp>
#include <xtensor/xmasked_view.hpp>
#include <xtensor/xview.hpp>

#include "../box.hpp"
#include "../cell_array.hpp"
#include "../cell_list.hpp"
#include "../static_algorithm.hpp"
// #include "../subset_old/subset_op.hpp"
#include "../subset_new/subset_op.hpp"
#include "operators.hpp"

#include "../hdf5.hpp"
#include "mesh_type.hpp"

namespace mure
{
    template<class MRConfig, std::size_t dim>
    struct MeshCellsArray : private std::array<CellArray<MRConfig>, dim>
    {
        using base = std::array<CellArray<MRConfig>, dim>;
        using base::operator[];

        inline const CellArray<MRConfig>& operator[](MeshType mesh_type) const
        {
            return operator[](static_cast<std::size_t>(mesh_type));
        }

        inline CellArray<MRConfig>& operator[](MeshType mesh_type)
        {
            return operator[](static_cast<std::size_t>(mesh_type));
        }
    };

    template<class MRConfig>
    class Mesh {
      public:
        static constexpr auto dim = MRConfig::dim;
        static constexpr auto max_refinement_level = MRConfig::max_refinement_level;
        static constexpr std::size_t ghost_width = std::max(std::max(2 * static_cast<int>(MRConfig::graduation_width) - 1,
                                                              static_cast<int>(MRConfig::max_stencil_width)),
                                                     static_cast<int>(MRConfig::default_s_for_prediction));

        // static constexpr auto ghost_width = std::max(std::max(2 * static_cast<int>(MRConfig::graduation_width) - 1,
        //                                                       static_cast<int>(MRConfig::max_stencil_width)),
        //                                              static_cast<int>(MRConfig::default_s_for_prediction));
        using index_t = typename MRConfig::index_t;
        using coord_index_t = typename MRConfig::coord_index_t;
        using point_t = typename Box<int, dim>::point_t;
        using interval_t = typename MRConfig::interval_t;

        Mesh(Mesh const &) = default;
        Mesh &operator=(Mesh const &) = default;

        inline Mesh(Box<double, dim> b, std::size_t min_level, std::size_t max_level)
          : m_min_level{min_level}, m_max_level{max_level}
        {
            using box_t = Box<coord_index_t, dim>;
            point_t start = b.min_corner() * std::pow(2, max_level);
            point_t end = b.max_corner() * std::pow(2, max_level);

            // m_cells[MeshType::cells][max_level] = {max_level, box_t{start, end}};
            // m_cells[MeshType::cells_and_ghosts][max_level] = {max_level, box_t{start - 1, end + 1}};
            // m_cells[MeshType::all_cells][max_level] = {max_level, box_t{start - 1, end + 1}};
            // m_cells[MeshType::all_cells][max_level - 1] = {max_level - 1, box_t{(start >> 1) - 1, (end >> 1) + 1}};
            // m_cells[MeshType::proj_cells][max_level - 1] = {max_level - 1, box_t{(start >> 1), (end >> 1)}};
            // m_cells[MeshType::union_cells][max_level - 1] = {max_level - 1, box_t{(start >> 1), (end >> 1)}};
            // m_init_cells = {max_level, box_t{start, end}};
            // update_x0_and_nb_ghosts();

            auto gw = static_cast<int>(ghost_width); // Just to cast it...

            m_cells[MeshType::cells][max_level] = {max_level, box_t{start, end}};
            m_cells[MeshType::cells_and_ghosts][max_level] = {max_level, box_t{start - gw, end + gw}};
            m_cells[MeshType::all_cells][max_level] = {max_level, box_t{start - gw, end + gw}};
            m_cells[MeshType::all_cells][max_level - 1] = {max_level - 1, box_t{(start >> 1) - gw, (end >> 1) + gw}};
            m_cells[MeshType::proj_cells][max_level - 1] = {max_level - 1, box_t{(start >> 1), (end >> 1)}};
            m_cells[MeshType::union_cells][max_level - 1] = {max_level - 1, box_t{(start >> 1), (end >> 1)}};
            
            m_cells[MeshType::overleaves][max_level] = {max_level, box_t{start, end}};

            m_init_cells = {max_level, box_t{start, end}};
            update_x0_and_nb_ghosts();
        }

        inline Mesh(const CellList<MRConfig> &dcl, const LevelCellArray<dim> &init_cells, std::size_t min_level, std::size_t max_level)
            : m_init_cells{init_cells}, m_min_level{min_level}, m_max_level{max_level}
        {
            m_cells[MeshType::cells] = {dcl};
            update_ghost_nodes();
        }

        inline Mesh(const CellList<MRConfig> &dcl, std::size_t min_level, std::size_t max_level)
        : m_min_level{min_level}, m_max_level{max_level}
        {
            m_cells[MeshType::cells] = {dcl};
            update_ghost_nodes();
        }

        inline void put_overleaves(const CellList<MRConfig> &dcl)
        {
            m_cells[MeshType::overleaves] = {dcl};
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

        inline auto const &get_cells(std::size_t i) const
        {
            return m_cells[i];
        }

        inline const CellArray<MRConfig>& operator[](MeshType mesh_type) const
        {
            return m_cells[mesh_type];
        }

        inline const LevelCellArray<dim>& initial_mesh() const
        {
            return m_init_cells;
        }

        inline auto max_level() const
        {
            return m_max_level;
        }

        inline auto min_level() const
        {
            return m_min_level;
        }

        inline void swap(Mesh<MRConfig> &mesh) noexcept
        {
            m_cells = std::move(mesh.m_cells);
            // swap(m_init_cells, mesh.m_init_cells);
            // swap(m_max_level, mesh.m_max_level);
        }

        template<typename... T>
        inline auto get_interval(std::size_t level, interval_t interval, T... index) const
        {
            const auto& lca = m_cells[MeshType::all_cells][level];
            auto row = lca.find({interval.start, index...});
            return lca[0][static_cast<std::size_t>(row)];
        }

        template<typename... T>
        inline auto exists(MeshType type, std::size_t level, interval_t interval, T... index) const
        {
            const auto& lca = m_cells[type][level];
            std::size_t size = interval.size()/interval.step;
            xt::xtensor<bool, 1> out = xt::empty<bool>({size});
            std::size_t iout = 0;
            for(coord_index_t i = interval.start; i < interval.end; i+=interval.step)
            {

                auto row = lca.find({i, index...});

                // std::cout<<std::endl<<"(***) level = "<<level<<" i = "<<i<<"  row = "<<row<<std::endl;

                if (row == -1)
                {
                    out[iout++] = false;
                }
                else
                {
                    out[iout++] = true;
                }
            }
            return out;
        }


        inline void to_stream(std::ostream &os) const
        {
            os << "Cells\n";
            m_cells[MeshType::cells].to_stream(os);
            os << "\nCells and ghosts\n";
            m_cells[MeshType::cells_and_ghosts].to_stream(os);
            os << "\nUnion cells\n";
            m_cells[MeshType::union_cells].to_stream(os);
            os << "\nProjection cells\n";
            m_cells[MeshType::proj_cells].to_stream(os);
            os << "\nAll cells\n";
            m_cells[MeshType::all_cells].to_stream(os);
            
            os << "\nOverleaves\n";
            m_cells[MeshType::overleaves].to_stream(os);
        }

        template<class Func>
        inline void for_each_cell(Func &&func, MeshType mesh_type = MeshType::cells) const
        {
            m_cells[mesh_type].for_each_cell(std::forward<Func>(func));
        }

        template<class Func>
        inline void for_each_cell(std::size_t level, Func &&func, MeshType mesh_type = MeshType::cells) const
        {
            m_cells[mesh_type][level].for_each_cell(level, std::forward<Func>(func));
        }

      private:
        void update_ghost_nodes();
            
        void add_overleaves(CellList<MRConfig> &, CellList<MRConfig> &);

        void add_ng_ghosts_and_get_nb_leaves(CellList<MRConfig> &cell_list);
        void add_ghosts_for_level_m1(CellList<MRConfig> &cell_list);
        void update_x0_and_nb_ghosts();

        //MeshCellsArray<MRConfig, 5> m_cells;
        MeshCellsArray<MRConfig, 6> m_cells; // Since we have added one type....

        LevelCellArray<dim> m_init_cells;
        std::size_t m_max_level;
        std::size_t m_min_level;
    };

    template<class MRConfig>
    inline void Mesh<MRConfig>::update_ghost_nodes()
    {
        {
            auto max_level = m_cells[MeshType::cells].max_level();
            auto min_level = m_cells[MeshType::cells].min_level();
            m_cells[MeshType::union_cells][max_level] = {max_level};
            for (std::size_t level = max_level; level >= ((min_level == 0) ? 1 : min_level); --level)
            {
                LevelCellList<dim, interval_t> lcl{level - 1};
                auto expr =
                    union_(m_cells[MeshType::cells][level], m_cells[MeshType::union_cells][level]).on(level - 1);

                expr([&](auto &index_yz, auto &interval, auto & /*interval_index*/) {

                    lcl[index_yz].add_interval({interval[0].start, interval[0].end});
                });
                m_cells[MeshType::union_cells][level - 1] = {lcl};
            }
        }
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
            LevelCellList<dim> &lcl = cell_list[level];

            if (!m_cells[MeshType::cells][level].empty())
            {
                auto expr =
                    intersection(m_cells[MeshType::union_cells][level],
                                 difference(m_cells[MeshType::all_cells][level], m_cells[MeshType::cells][level]))
                        .on(level - 1);

                expr([&](auto &index_yz, auto &interval, auto &
                         /*interval_index*/) {
                    static_nested_loop<dim - 1, 0, 2>([&](auto stencil) {
                        lcl[(index_yz << 1) + stencil].add_interval({interval[0].start << 1, interval[0].end << 1});
                    });
                });
            }
        }
        m_cells[MeshType::all_cells] = {cell_list};

        {
            auto max_level = m_cells[MeshType::cells].max_level();
            auto min_level = m_cells[MeshType::cells].min_level();
            for (std::size_t level = max_level; level >= ((min_level == 0) ? 1 : min_level); --level)
            {
                LevelCellList<dim, interval_t> lcl{level - 1};
                auto expr =
                    intersection(m_cells[MeshType::all_cells][level - 1], m_cells[MeshType::union_cells][level - 1]);

                expr([&](auto &index_yz, auto &interval, auto & /*interval_index*/) {

                    lcl[index_yz].add_interval({interval[0].start, interval[0].end});
                });
                m_cells[MeshType::proj_cells][level - 1] = {lcl};
            }
        }

        // update of x0_indices, _leaf_to_ghost_indices,


        //PUT MY UPDATE
        CellList<MRConfig> overleaves_list;
        add_overleaves(overleaves_list, cell_list);
        m_cells[MeshType::overleaves] = {overleaves_list};
        
        m_cells[MeshType::all_cells] = {cell_list}; // We must put the overleaves in the all cells to store them



        update_x0_and_nb_ghosts(); // MODIFY INSIDE
    }

    template<class MRConfig>
    inline void Mesh<MRConfig>::add_overleaves(CellList<MRConfig> &overleaves_list, CellList<MRConfig> &cell_list)
    {

        //const int cells_to_add_1D = 2; // To be changed according to the numerical scheme
        const int cells_to_add = 1; // To be changed according to the numerical scheme

        for (std::size_t level = 0; level < max_level(); ++level)
        {
            const LevelCellArray<dim> &level_cell_array = m_cells[MeshType::cells][level];
            if (!level_cell_array.empty())
            {
                LevelCellList<dim> &level_overleaves_list = overleaves_list[level + 1]; // We have to put it at the higher level
                LevelCellList<dim> &level_cell_list = cell_list[level + 1]; // We have to put it at the higher level

                level_cell_array.for_each_interval_in_x(
                    [&](xt::xtensor_fixed<coord_index_t, xt::xshape<dim - 1>> const &index_yz,
                        interval_t const &interval) {

                        
                        // THIS WORKED FOR THE 1D ONLY. BUT WORKED !!!
                        
                        // auto index = xt::eval(index_yz);
                        
                        // level_overleaves_list[index].add_interval({2*interval.start - cells_to_add_1D,
                        //                                            2*interval.end   + cells_to_add_1D});

                                                                                           
                        // level_cell_list[index].add_interval({2*interval.start - cells_to_add_1D,
                        //                                            2*interval.end   + cells_to_add_1D});




                        static_nested_loop<dim - 1, -cells_to_add, cells_to_add + 1, 1>([&](auto stencil) {
                            auto index = xt::eval(index_yz + stencil);

                            //std::cout<<std::endl<<"Debug = "<<(2 * index + 1)<<std::flush;
                        
                            level_overleaves_list[2 * index].add_interval({2 * (interval.start - cells_to_add),
                                                                       2 * (interval.end   + cells_to_add)});
                            level_overleaves_list[2 * index + 1].add_interval({2 * (interval.start - cells_to_add),
                                                                       2 * (interval.end   + cells_to_add)});
             
                            level_cell_list[2 * index].add_interval({2 * (interval.start - cells_to_add),
                                                                 2 * (interval.end   + cells_to_add)});
                            level_cell_list[2 * index + 1].add_interval({2 * (interval.start - cells_to_add),
                                                                 2 * (interval.end   + cells_to_add)});
                        });
                    });
            }
        }
    }

    template<class MRConfig>
    inline void Mesh<MRConfig>::add_ng_ghosts_and_get_nb_leaves(CellList<MRConfig> &cell_list)
    {
        // +/- w nodes in the current level
        for (std::size_t level = 0; level <= max_refinement_level; ++level)
        {
            const LevelCellArray<dim> &level_cell_array = m_cells[MeshType::cells][level];
            if (!level_cell_array.empty())
            {
                LevelCellList<dim> &level_cell_list = cell_list[level];

                level_cell_array.for_each_interval_in_x(
                    [&](xt::xtensor_fixed<coord_index_t, xt::xshape<dim - 1>> const &index_yz,
                        interval_t const &interval) {
                        static_nested_loop<dim - 1, -ghost_width, ghost_width + 1>([&](auto stencil) {
                            auto index = xt::eval(index_yz + stencil);
                            level_cell_list[index].add_interval({interval.start - static_cast<int>(ghost_width),
                                                                 interval.end + static_cast<int>(ghost_width)});
                        });
                    });
            }
        }
    }

    template<class MRConfig>
    inline void Mesh<MRConfig>::add_ghosts_for_level_m1(CellList<MRConfig> &cell_list)
    {
        for (std::size_t level = 1; level <= max_refinement_level; ++level)
        {
            const LevelCellArray<dim> &level_cell_array = m_cells[MeshType::cells][level];
            if (level_cell_array.empty() == false)
            {
                LevelCellList<dim> &level_cell_list = cell_list[level - 1];
                constexpr index_t s = MRConfig::default_s_for_prediction;

                level_cell_array.for_each_interval_in_x(
                    [&](xt::xtensor_fixed<coord_index_t, xt::xshape<dim - 1>> const &index_yz,
                        interval_t const &interval) {
                        static_nested_loop<dim - 1, -ghost_width - s, ghost_width + s + 1>([&](auto stencil) {
                            int beg = (interval.start >> 1) - static_cast<int>(s + ghost_width);
                            int end = ((interval.end + 1) >> 1) + static_cast<int>(s + ghost_width);

                            level_cell_list[(index_yz >> 1) + stencil].add_interval({beg, end});
                        });
                    });
            }
        }
    }

    template<class MRConfig>
    inline void Mesh<MRConfig>::update_x0_and_nb_ghosts()
    {
        std::size_t ghost_index = 0;
        for (std::size_t level = 0; level <= max_refinement_level; ++level)
        {
            m_cells[MeshType::all_cells][level].for_each_interval_in_x([&](auto &, const interval_t &interval) {
                // FIXME: remove this const_cast !!
                const_cast<interval_t &>(interval).index = static_cast<index_t>(ghost_index) - interval.start;
                ghost_index += interval.size();
            });
        }

        for (std::size_t level = 0; level <= max_refinement_level; ++level)
        {
            if (!m_cells[MeshType::cells][level].empty())
            {
                auto expr = intersection(m_cells[MeshType::all_cells][level], m_cells[MeshType::cells][level]);

                expr([&](auto & /*index_yz*/, auto & /*interval*/, auto &interval_index) {
                    m_cells[MeshType::cells][level][0][static_cast<std::size_t>(interval_index[1])].index =
                        m_cells[MeshType::all_cells][level][0][static_cast<std::size_t>(interval_index[0])].index;
                });
            }

            if (!m_cells[MeshType::cells_and_ghosts][level].empty())
            {
                auto expr =
                    intersection(m_cells[MeshType::all_cells][level], m_cells[MeshType::cells_and_ghosts][level]);

                expr([&](auto & /*index_yz*/, auto & /*interval*/, auto &interval_index) {
                    m_cells[MeshType::cells_and_ghosts][level][0][static_cast<std::size_t>(interval_index[1])].index =
                        m_cells[MeshType::all_cells][level][0][static_cast<std::size_t>(interval_index[0])].index;
                });
            }

            if (!m_cells[MeshType::proj_cells][level].empty())
            {
                auto expr = intersection(m_cells[MeshType::all_cells][level], m_cells[MeshType::proj_cells][level]);

                expr([&](auto & /*index_yz*/, auto & /*interval*/, auto &interval_index) {
                    m_cells[MeshType::proj_cells][level][0][static_cast<std::size_t>(interval_index[1])].index =
                        m_cells[MeshType::all_cells][level][0][static_cast<std::size_t>(interval_index[0])].index;
                });
            }

            if (!m_cells[MeshType::union_cells][level].empty())
            {
                auto expr = intersection(m_cells[MeshType::all_cells][level], m_cells[MeshType::union_cells][level]);

                expr([&](auto & /*index_yz*/, auto & /*interval*/, auto &interval_index) {
                    m_cells[MeshType::union_cells][level][0][static_cast<std::size_t>(interval_index[1])].index =
                        m_cells[MeshType::all_cells][level][0][static_cast<std::size_t>(interval_index[0])].index;
                });
            }



            if (!m_cells[MeshType::overleaves][level].empty())
            {
                auto expr = intersection(m_cells[MeshType::all_cells][level], m_cells[MeshType::overleaves][level]);

                expr([&](auto & /*index_yz*/, auto & /*interval*/, auto &interval_index) {
                    m_cells[MeshType::overleaves][level][0][static_cast<std::size_t>(interval_index[1])].index =
                        m_cells[MeshType::all_cells][level][0][static_cast<std::size_t>(interval_index[0])].index;
                });
            }
        }
    }

    template<class MRConfig>
    inline std::ostream &operator<<(std::ostream &out, const Mesh<MRConfig> &mesh)
    {
        mesh.to_stream(out);
        return out;
    }

    template<class MRConfig>
    inline bool operator==(const Mesh<MRConfig> &mesh1, const Mesh<MRConfig> &mesh2)
    {
        if (mesh1.max_level() != mesh2.max_level() or
            mesh1.min_level() != mesh2.min_level())
            return false;

        for(std::size_t level=mesh1.max_level(); level >= mesh1.min_level(); --level)
        {
            if (!(mesh1[MeshType::cells][level] == mesh2[MeshType::cells][level]))
            {
                return false;
            }
        }
        return true;
    }
}

namespace std
{
    template<class MRConfig>
    inline void swap(mure::Mesh<MRConfig> &lhs, mure::Mesh<MRConfig> &rhs) noexcept
    {
        lhs.swap(rhs);
    }
}