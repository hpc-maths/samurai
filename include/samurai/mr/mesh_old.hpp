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
#include "../subset/subset_op.hpp"
#include "operators.hpp"

#include "../hdf5.hpp"
#include "mesh_type.hpp"

namespace samurai
{
    namespace detail
    {
        template<class CellArray, std::size_t nb_sub_mesh, class MeshType>
        struct MeshCellArray : private std::array<CellArray, nb_sub_mesh>
        {
            using base_type = std::array<CellArray, nb_sub_mesh>;
            using base_type::operator[];

            inline const CellArray& operator[](MeshType mesh_type) const
            {
                return operator[](static_cast<std::size_t>(mesh_type));
            }

            inline CellArray& operator[](MeshType mesh_type)
            {
                return operator[](static_cast<std::size_t>(mesh_type));
            }
        };
    }

    /////////////////////
    // Mesh definition //
    /////////////////////

    template<class MRConfig>
    class Mesh
    {
    public:
        static constexpr auto dim = MRConfig::dim;
        static constexpr auto max_refinement_level = MRConfig::max_refinement_level;
        static constexpr std::size_t ghost_width = std::max(std::max(2 * static_cast<int>(MRConfig::graduation_width) - 1,
                                                                     static_cast<int>(MRConfig::max_stencil_width)),
                                                                     static_cast<int>(MRConfig::default_s_for_prediction));

        using index_t = typename MRConfig::index_t;
        using coord_index_t = typename MRConfig::coord_index_t;
        using point_t = typename Box<int, dim>::point_t;
        using interval_t = typename MRConfig::interval_t;

        using cl_type = CellList<dim, interval_t, max_refinement_level>;
        using lcl_type = typename cl_type::lcl_type;

        using ca_type = CellArray<dim, interval_t, max_refinement_level>;
        using lca_type = typename ca_type::lca_type;

        Mesh() = default;

        Mesh(const Mesh&) = default;
        Mesh& operator=(const Mesh&) = default;

        Mesh(Mesh&&) = default;
        Mesh& operator=(Mesh&&) = default;

        Mesh(const Box<double, dim>& b, std::size_t min_level, std::size_t max_level);
        Mesh(const cl_type &cl, const lca_type& init_cells, std::size_t min_level, std::size_t max_level);
        Mesh(const cl_type &cl, std::size_t min_level, std::size_t max_level);

        std::size_t nb_cells(MeshType mesh_type=MeshType::all_cells) const;
        std::size_t nb_cells(std::size_t level, MeshType mesh_type=MeshType::all_cells) const;

        const ca_type& operator[](MeshType mesh_type) const;
        const lca_type& initial_mesh() const;
        std::size_t max_level() const;
        std::size_t min_level() const;
        void swap(Mesh& mesh) noexcept;

        template<typename... T>
        const interval_t& get_interval(std::size_t level, const interval_t& interval, T... index) const;

        template<typename... T>
        xt::xtensor<bool, 1> exists(MeshType type, std::size_t level, interval_t interval, T... index) const;

        void to_stream(std::ostream &os) const;

      private:
        void update_ghost_nodes();

        void add_overleaves(cl_type&, cl_type&);

        void add_ng_ghosts_and_get_nb_leaves(cl_type& cl);
        void add_ghosts_for_level_m1(cl_type& cl);
        void update_x0_and_nb_ghosts();

        //detail::MeshCellsArray<MRConfig, 5> m_cells;
        detail::MeshCellArray<ca_type, 6, MeshType> m_cells; // Since we have added one type....

        LevelCellArray<dim, interval_t> m_init_cells;
        std::size_t m_max_level;
        std::size_t m_min_level;
    };

    /////////////////////////
    // Mesh implementation //
    /////////////////////////

    template<class MRConfig>
    inline Mesh<MRConfig>::Mesh(const Box<double, dim>& b, std::size_t min_level, std::size_t max_level)
    : m_min_level{min_level}, m_max_level{max_level}
    {
        using box_t = Box<coord_index_t, dim>;
        point_t start = b.min_corner() * std::pow(2, max_level);
        point_t end = b.max_corner() * std::pow(2, max_level);

        auto gw = static_cast<int>(ghost_width); // Just to cast it...

        m_cells[MeshType::cells][max_level] = {max_level, box_t{start, end}};
        m_cells[MeshType::cells_and_ghosts][max_level] = {max_level, box_t{start - gw, end + gw}};
        m_cells[MeshType::all_cells][max_level] = {max_level, box_t{start - gw, end + gw}};
        m_cells[MeshType::all_cells][max_level - 1] = {max_level - 1, box_t{(start >> 1) - gw, (end >> 1) + gw}};
        m_cells[MeshType::proj_cells][max_level - 1] = {max_level - 1, box_t{(start >> 1), (end >> 1)}};
        m_cells[MeshType::union_cells][max_level - 1] = {max_level - 1, box_t{(start >> 1), (end >> 1)}};

        m_cells[MeshType::overleaves][max_level] = {max_level, box_t{start, end}};

        m_init_cells = {max_level, box_t{start, end}};

        m_cells[MeshType::all_cells].update_index();

        update_x0_and_nb_ghosts();
    }

    template<class MRConfig>
    inline Mesh<MRConfig>::Mesh(const cl_type& cl, const lca_type& init_cells, std::size_t min_level, std::size_t max_level)
    : m_init_cells{init_cells}, m_min_level{min_level}, m_max_level{max_level}
    {
        m_cells[MeshType::cells] = {cl, false};
        update_ghost_nodes();
    }

    template<class MRConfig>
    inline Mesh<MRConfig>::Mesh(const cl_type &cl, std::size_t min_level, std::size_t max_level)
    : m_min_level{min_level}, m_max_level{max_level}
    {
        m_cells[MeshType::cells] = {cl, false};
        update_ghost_nodes();
    }

    template<class MRConfig>
    inline std::size_t Mesh<MRConfig>::nb_cells(MeshType mesh_type) const
    {
        return m_cells[mesh_type].nb_cells();
    }

    template<class MRConfig>
    inline std::size_t Mesh<MRConfig>::nb_cells(std::size_t level, MeshType mesh_type) const
    {
        return m_cells[mesh_type][level].nb_cells();
    }

    template<class MRConfig>
    inline auto Mesh<MRConfig>::operator[](MeshType mesh_type) const -> const ca_type&
    {
        return m_cells[mesh_type];
    }

    template<class MRConfig>
    inline auto Mesh<MRConfig>::initial_mesh() const -> const lca_type&
    {
        return m_init_cells;
    }

    template<class MRConfig>
    inline std::size_t Mesh<MRConfig>::max_level() const
    {
        return m_max_level;
    }

    template<class MRConfig>
    inline std::size_t Mesh<MRConfig>::min_level() const
    {
        return m_min_level;
    }

    template<class MRConfig>
    inline void Mesh<MRConfig>::swap(Mesh<MRConfig> &mesh) noexcept
    {
        using std::swap;
        swap(m_cells, mesh.m_cells);
        swap(m_init_cells, mesh.m_init_cells);
        swap(m_max_level, mesh.m_max_level);
        swap(m_min_level, mesh.m_min_level);
    }

    template<class MRConfig>
    template<typename... T>
    inline auto Mesh<MRConfig>::get_interval(std::size_t level, const interval_t& interval, T... index) const -> const interval_t&
    {
        return m_cells[MeshType::all_cells].get_interval(level, interval, index...);
    }

    // TODO: put this method outside
    template<class MRConfig>
    template<typename... T>
    inline xt::xtensor<bool, 1> Mesh<MRConfig>::exists(MeshType type, std::size_t level, interval_t interval, T... index) const
    {
        const auto& lca = m_cells[type][level];
        std::size_t size = interval.size()/interval.step;
        xt::xtensor<bool, 1> out = xt::empty<bool>({size});
        std::size_t iout = 0;
        for(coord_index_t i = interval.start; i < interval.end; i+=interval.step)
        {
            auto row = find(lca, {i, index...});
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

    template<class MRConfig>
    inline void Mesh<MRConfig>::to_stream(std::ostream& os) const
    {
        std::map<MeshType, std::string> m {{MeshType::cells, "Cells"},
                                           {MeshType::cells_and_ghosts, "Cells and Ghosts"},
                                           {MeshType::union_cells, "Union cells"},
                                           {MeshType::proj_cells, "Projection cells"},
                                           {MeshType::all_cells, "All cells"},
                                           {MeshType::overleaves, "Overleaves"}};

        for (auto e: m)
        {
            fmt::print(os, fmt::format(fmt::emphasis::bold, "{}\n{:─^50}\n", e.second, ""));
            fmt::print(os, fmt::format("{}", m_cells[e.first]));
        }
    }

    template<class MRConfig>
    inline void Mesh<MRConfig>::update_ghost_nodes()
    {
        auto max_level = m_cells[MeshType::cells].max_level();
        auto min_level = m_cells[MeshType::cells].min_level();

        m_cells[MeshType::union_cells][max_level] = {max_level};

        for (std::size_t level = max_level; level >= ((min_level == 0) ? 1 : min_level); --level)
        {
            lcl_type lcl{level - 1};
            auto expr = union_(m_cells[MeshType::cells][level], m_cells[MeshType::union_cells][level]).on(level - 1);

            expr([&](auto& interval, auto& index_yz)
            {
                lcl[index_yz].add_interval({interval.start, interval.end});
            });

            m_cells[MeshType::union_cells][level - 1] = {lcl};
        }

        // +/- w ghosts in level + 0 and 1, computation of _nb_local_leaf_cells
        cl_type cell_list;
        add_ng_ghosts_and_get_nb_leaves(cell_list);

        m_cells[MeshType::cells_and_ghosts] = {cell_list, false};

        // optionnal +/- w nodes in level - 1
        if (MRConfig::need_pred_from_proj)
        {
            add_ghosts_for_level_m1(cell_list);
        }

        // compaction
        m_cells[MeshType::all_cells] = {cell_list, false};

        for (std::size_t level = max_level; level >= ((min_level == 0) ? 1 : min_level); --level)
        {
            lcl_type& lcl = cell_list[level];

            if (!m_cells[MeshType::cells][level].empty())
            {
                auto expr = intersection(m_cells[MeshType::union_cells][level],
                                         difference(m_cells[MeshType::all_cells][level],
                                                    m_cells[MeshType::cells][level]))
                           .on(level - 1);

                expr([&](auto& interval, auto& index_yz)
                {
                    static_nested_loop<dim - 1, 0, 2>([&](auto stencil)
                    {
                        lcl[(index_yz << 1) + stencil].add_interval({interval.start << 1, interval.end << 1});
                    });
                });
            }
        }

        m_cells[MeshType::all_cells] = {cell_list, false};
        for (std::size_t level = max_level; level >= ((min_level == 0) ? 1 : min_level); --level)
        {
            lcl_type lcl{level - 1};
            auto expr = intersection(m_cells[MeshType::all_cells][level - 1], m_cells[MeshType::union_cells][level - 1]);

            expr([&](auto& interval, auto& index_yz)
            {
                lcl[index_yz].add_interval({interval.start, interval.end});
            });
            m_cells[MeshType::proj_cells][level - 1] = {lcl};
        }

        // update of x0_indices, _leaf_to_ghost_indices,


        //PUT MY UPDATE
        cl_type overleaves_list;
        add_overleaves(overleaves_list, cell_list);
        m_cells[MeshType::overleaves] = {overleaves_list, false};

        m_cells[MeshType::all_cells] = {cell_list}; // We must put the overleaves in the all cells to store them

        update_x0_and_nb_ghosts(); // MODIFY INSIDE
    }

    template<class MRConfig>
    inline void Mesh<MRConfig>::add_overleaves(cl_type& overleaves_list, cl_type& cell_list)
    {

        //const int cells_to_add_1D = 2; // To be changed according to the numerical scheme
        const int cells_to_add = 1; // To be changed according to the numerical scheme

        for_each_interval(m_cells[samurai::MeshType::cells], [&](std::size_t level, const auto& interval, const auto& index_yz)
        {
            lcl_type& level_overleaves_list = overleaves_list[level + 1]; // We have to put it at the higher level
            lcl_type& level_cell_list = cell_list[level + 1]; // We have to put it at the higher level

            static_nested_loop<dim - 1, -cells_to_add, cells_to_add + 1, 1>([&](auto stencil)
            {
                auto index = xt::eval(index_yz + stencil);

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

    template<class MRConfig>
    inline void Mesh<MRConfig>::add_ng_ghosts_and_get_nb_leaves(cl_type &cl)
    {
        for_each_interval(m_cells[samurai::MeshType::cells], [&](std::size_t level, const auto& interval, const auto& index_yz)
        {
            lcl_type& lcl = cl[level];
            static_nested_loop<dim - 1, -ghost_width, ghost_width + 1>([&](auto stencil)
            {
                auto index = xt::eval(index_yz + stencil);
                lcl[index].add_interval({interval.start - static_cast<int>(ghost_width),
                                         interval.end + static_cast<int>(ghost_width)});
            });
        });
    }

    template<class MRConfig>
    inline void Mesh<MRConfig>::add_ghosts_for_level_m1(cl_type &cl)
    {
        constexpr index_t s = MRConfig::default_s_for_prediction;
        auto min_level = m_cells[MeshType::cells].min_level();
        auto max_level = m_cells[MeshType::cells].max_level();

        for (std::size_t level = ((min_level == 0)? 1 : min_level); level <= max_level; ++level)
        {
            lca_type& lca = m_cells[MeshType::cells][level];
            lcl_type& lcl = cl[level - 1];

            for_each_interval(lca, [&](std::size_t level, const auto& interval, const auto& index_yz)
            {
                // static_nested_loop<dim - 1, -ghost_width - s, ghost_width + s + 1>([&](auto stencil) {
                //     int beg = (interval.start >> 1) - static_cast<int>(s + ghost_width);
                //     int end = ((interval.end + 1) >> 1) + static_cast<int>(s + ghost_width);

                //     level_cell_list[(index_yz >> 1) + stencil].add_interval({beg, end});
                // });
                static_nested_loop<dim - 1, -ghost_width, ghost_width + 1>([&](auto stencil) {
                    int beg = (interval.start >> 1) - static_cast<int>(ghost_width);
                    int end = ((interval.end + 1) >> 1) + static_cast<int>(ghost_width);

                    lcl[(index_yz >> 1) + stencil].add_interval({beg, end});
                });
            });
        }
    }

    template<class MRConfig>
    inline void Mesh<MRConfig>::update_x0_and_nb_ghosts()
    {
        std::vector<MeshType> mesh_type {MeshType::cells,
                                         MeshType::cells_and_ghosts,
                                         MeshType::proj_cells,
                                         MeshType::union_cells,
                                         MeshType::overleaves};

        for(auto mt: mesh_type)
        {
            for (std::size_t level = 0; level <= max_refinement_level; ++level)
            {
                lca_type& lhs = m_cells[mt][level];
                const lca_type& rhs = m_cells[MeshType::all_cells][level];

                auto expr = intersection(lhs, rhs);

                expr.apply_interval_index([&](auto& interval_index)
                {
                    lhs[0][interval_index[0]].index = rhs[0][interval_index[1]].index;
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
        if (mesh1.max_level() != mesh2.max_level() ||
            mesh1.min_level() != mesh2.min_level())
        {
            return false;
        }

        for(std::size_t level=mesh1.min_level(); level <= mesh1.max_level(); ++level)
        {
            if (!(mesh1[MeshType::cells][level] == mesh2[MeshType::cells][level]))
            {
                return false;
            }
        }
        return true;
    }
}

// namespace std
// {
    // template<class MRConfig>
    // inline void swap(samurai::Mesh<MRConfig> &lhs, samurai::Mesh<MRConfig> &rhs) noexcept
    // {
    //     lhs.swap(rhs);
    // }
// }