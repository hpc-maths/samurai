#pragma once
#include "stencil.hpp"

namespace samurai_new
{
    template <class Mesh, typename TIndex>
    inline auto get_index_start(const Mesh& mesh, std::size_t level, const typename Mesh::interval_t& i, const TIndex& index)
    {
        static constexpr std::size_t dim = Mesh::dim;
        using coord_index_t = typename Mesh::coord_index_t;

        std::array<coord_index_t, dim> coord;
        std::copy(index.cbegin(), index.end(), coord.begin()+1);
        coord[0] = i.start;
        return mesh.get_index(level, coord);
    }

    template <typename DesiredIndexType, class Mesh, typename TIndex, class Func>
    inline void for_each_cell(const Mesh& mesh, std::size_t level, const typename Mesh::interval_t& i, const TIndex& index, Func &&f)
    {
        auto i_start = static_cast<DesiredIndexType>(get_index_start(mesh, level, i, index));
        for(DesiredIndexType ii=0; ii<static_cast<DesiredIndexType>(i.size()); ++ii)
        {
            f(i_start + ii);
        }
    }

    template <typename DesiredIndexType, class Mesh, class Set, class Func>
    inline void for_each_cell(const Mesh& mesh, const Set& set, Func &&f)
    {
        for_each_interval(set, [&](std::size_t level, const auto& i, const auto& index)
        {
            for_each_cell<DesiredIndexType>(mesh, level, i, index, std::forward<Func>(f));
        });
    }

    template <typename DesiredIndexType, class Mesh, typename TIndex, class StencilType, class Func>
    inline void for_each_stencil(const Mesh& mesh, std::size_t level, const typename Mesh::interval_t i, const TIndex& index, StencilType& stencil, Func &&f)
    {
        stencil.init(mesh, level, i, index);
        f(stencil.indices());
        for(DesiredIndexType ii=1; ii<static_cast<DesiredIndexType>(i.size()); ++ii)
        {
            stencil.move_next();
            f(stencil.indices());
        }
    }

    template <typename DesiredIndexType, class Mesh, class Set, class StencilType, class Func>
    inline void for_each_stencil(const Mesh& mesh, const Set& set, std::size_t level, StencilType& stencil, Func &&f)
    {
        for_each_interval(set[level], [&](std::size_t level, const auto& i, const auto& index)
        {
            for_each_stencil<DesiredIndexType>(mesh, level, i, index, stencil, std::forward<Func>(f));
        });
    }

    template <typename DesiredIndexType, class Mesh, class Set, class StencilType, class Func>
    inline void for_each_stencil(const Mesh& mesh, const Set& set, StencilType& stencil, Func &&f)
    {
        for_each_level(set, [&](std::size_t level)
        {
            for_each_stencil<DesiredIndexType>(mesh, set, level, stencil, std::forward<Func>(f));
        });
    }

    /**
     * Used to define the projection operator.
     */
    template <typename DesiredIndexType, class Mesh, class Func>
    inline void for_each_cell_and_children(const Mesh& mesh, Func &&f)
    {
        using mesh_id_t = typename Mesh::mesh_id_t;

        auto min_level = mesh[mesh_id_t::cells].min_level();
        auto max_level = mesh[mesh_id_t::cells].max_level();

        for(std::size_t level=min_level; level<max_level; ++level)
        {
            auto set = samurai::intersection(mesh[mesh_id_t::cells_and_ghosts][level],
                                             mesh[mesh_id_t::cells][level+1])
                        .on(level);

            set([&](const auto& i, const auto& index)
            {
                auto j = index[0];

                auto cell_start       = static_cast<DesiredIndexType>(mesh.get_index(level    ,   i.start,   j    ));
                auto child_2j___start = static_cast<DesiredIndexType>(mesh.get_index(level + 1, 2*i.start, 2*j    ));
                auto child_2jp1_start = static_cast<DesiredIndexType>(mesh.get_index(level + 1, 2*i.start, 2*j + 1));

                for(DesiredIndexType ii=0; ii<static_cast<DesiredIndexType>(i.size()); ++ii)
                {
                    auto cell = cell_start + ii;

                    // Children following the the direct orientation
                    std::array<DesiredIndexType, 4> children;
                    children[0] = child_2j___start + 2*ii;   // bottom-left:  (2i  , 2j  )
                    children[1] = child_2j___start + 2*ii+1; // bottom-right: (2i+1, 2j  )
                    children[2] = child_2jp1_start + 2*ii+1; // top-right:    (2i+1, 2j+1)
                    children[3] = child_2jp1_start + 2*ii;   // top-left:     (2i  , 2j+1)
                    f(cell, children);
                }
            });
        }
    }

    /**
     * Used for the allocation of the matrix rows where the projection operator is used.
     */
    template <typename DesiredIndexType, class Mesh, class Func>
    inline void for_each_cell_having_children(const Mesh& mesh, Func &&f)
    {
        using mesh_id_t = typename Mesh::mesh_id_t;

        auto min_level = mesh[mesh_id_t::cells].min_level();
        auto max_level = mesh[mesh_id_t::cells].max_level();

        for(std::size_t level=min_level; level<max_level; ++level)
        {
            auto set = samurai::intersection(mesh[mesh_id_t::cells_and_ghosts][level],
                                             mesh[mesh_id_t::cells][level+1])
                        .on(level);

            set([&](const auto& i, const auto& index)
            {
                for_each_cell<DesiredIndexType>(mesh, level, i, index, std::forward<Func>(f));
            });
        }
    }

    /**
     * Used for the allocation of the matrix rows where the prediction operator is used.
     */
    template <typename DesiredIndexType, class Mesh, class Func>
    inline void for_each_cell_having_parent(const Mesh& mesh, Func &&f)
    {
        using mesh_id_t = typename Mesh::mesh_id_t;

        auto min_level = mesh[mesh_id_t::cells].min_level();
        auto max_level = mesh[mesh_id_t::cells].max_level();

        for(std::size_t level=min_level+1; level<=max_level; ++level)
        {
            auto set = samurai::intersection(mesh[mesh_id_t::cells_and_ghosts][level],
                                             mesh[mesh_id_t::cells][level-1])
                    .on(level);

            set([&](const auto& i, const auto& index)
            {
                for_each_cell<DesiredIndexType>(mesh, level, i, index, std::forward<Func>(f));
            });
        }
    }
}