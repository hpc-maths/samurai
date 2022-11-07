#pragma once
#include "stencil.hpp"

namespace samurai_new
{
    template <typename DesiredIndexType, class Mesh, class Func>
    inline void for_each_cell(const Mesh& mesh, const MeshInterval<Mesh>& mesh_interval, Func &&f)
    {
        auto i_start = static_cast<DesiredIndexType>(get_index_start(mesh, mesh_interval));
        for(DesiredIndexType ii=0; ii<static_cast<DesiredIndexType>(mesh_interval.i.size()); ++ii)
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

    template <typename DesiredIndexType, class Mesh, class StencilType, class Func>
    inline void for_each_stencil(const Mesh& mesh, const MeshInterval<Mesh>& mesh_interval, StencilType& stencil, Func &&f)
    {
        stencil.init(mesh, mesh_interval);
        f(stencil.indices());
        for(DesiredIndexType ii=1; ii<static_cast<DesiredIndexType>(mesh_interval.i.size()); ++ii)
        {
            stencil.move_next();
            f(stencil.indices());
        }
    }

    template <class Mesh, class StencilType, class Func>
    inline void for_each_stencil(const Mesh& mesh, const MeshInterval<Mesh>& mesh_interval, StencilType& stencil, Func &&f)
    {
        stencil.init(mesh, mesh_interval);
        f(stencil.cells());
        for(std::size_t ii=1; ii<mesh_interval.i.size(); ++ii)
        {
            stencil.move_next();
            f(stencil.cells());
        }
    }

    template <typename DesiredIndexType, class Mesh, class Set, class StencilType, class Func>
    inline void for_each_stencil(const Mesh& mesh, const Set& set, std::size_t level, StencilType& stencil, Func &&f)
    {
        for_each_interval(set[level], [&](std::size_t level, const auto& i, const auto& index)
        {
            MeshInterval<Mesh> mesh_interval(level, i, index);
            for_each_stencil<DesiredIndexType>(mesh, mesh_interval, stencil, std::forward<Func>(f));
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
        static constexpr std::size_t dim = Mesh::dim;
        static constexpr std::size_t number_of_children = (1 << dim);

        auto min_level = mesh[mesh_id_t::cells].min_level();
        auto max_level = mesh[mesh_id_t::cells].max_level();

        for(std::size_t level=min_level; level<max_level; ++level) // TO PUT BACK!!!!!!!!!!!!!!!!!!
        //for(std::size_t level=min_level; level<=max_level; ++level)
        {
            auto set = samurai::intersection(mesh[mesh_id_t::cells_and_ghosts][level],
                                             mesh[mesh_id_t::cells][level+1])
                        .on(level);
            /*auto set = samurai::intersection(mesh[mesh_id_t::cells][level],
                                             mesh[mesh_id_t::cells][level])
                        .on(level);*/
            MeshInterval<Mesh> mesh_interval(level);

            set([&](const auto& i, const auto& index)
            {
                mesh_interval.i = i;
                mesh_interval.index = index;
                //auto j = index[0];

                /*auto cell_start       = static_cast<DesiredIndexType>(mesh.get_index(level    ,   i.start,   j    ));
                auto child_2j___start = static_cast<DesiredIndexType>(mesh.get_index(level + 1, 2*i.start, 2*j    ));
                auto child_2jp1_start = static_cast<DesiredIndexType>(mesh.get_index(level + 1, 2*i.start, 2*j + 1));*/

                auto cell_start = static_cast<DesiredIndexType>(get_index_start(mesh, mesh_interval));
                std::array<DesiredIndexType, dim> children_yz_start;
                for (std::size_t yz=0; yz<dim; ++yz)
                {
                    std::array<int, dim> translation_vect { 0 };
                    children_yz_start[yz * ((1<<dim) - 1)] = static_cast<DesiredIndexType>(get_index_start_children(mesh, mesh_interval, translation_vect));
                    for (std::size_t other_dim=1; other_dim<dim; ++other_dim)
                    {
                        translation_vect[other_dim] = 1;
                        for (std::size_t cell=0; cell<(1<<dim); ++cell)
                        {
                            children_yz_start[yz * ((1<<dim) - 1) + cell] = static_cast<DesiredIndexType>(get_index_start_children(mesh, mesh_interval, translation_vect));
                        }
                    }
                }

                for(DesiredIndexType ii=0; ii<static_cast<DesiredIndexType>(i.size()); ++ii)
                {
                    auto cell = cell_start + ii;

                    // Children following the direct orientation
                    std::array<DesiredIndexType, number_of_children> children;

                    /*
                    children[0] = child_2j___start + 2*ii;   // bottom-left:  (2i  , 2j  )
                    children[1] = child_2j___start + 2*ii+1; // bottom-right: (2i+1, 2j  )
                    children[2] = child_2jp1_start + 2*ii+1; // top-right:    (2i+1, 2j+1)
                    children[3] = child_2jp1_start + 2*ii;   // top-left:     (2i  , 2j+1)
                    */

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

            MeshInterval<Mesh> mesh_interval(level);
            set([&](const auto& i, const auto& index)
            {
                mesh_interval.i = i;
                mesh_interval.index = index;
                for_each_cell<DesiredIndexType>(mesh, mesh_interval, std::forward<Func>(f));
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

            MeshInterval<Mesh> mesh_interval(level);
            set([&](const auto& i, const auto& index)
            {
                mesh_interval.i = i;
                mesh_interval.index = index;
                for_each_cell<DesiredIndexType>(mesh, mesh_interval, std::forward<Func>(f));
            });
        }
    }
}