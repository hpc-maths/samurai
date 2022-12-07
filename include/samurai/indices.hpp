#pragma once
#include "algorithm.hpp"

namespace samurai
{
    template <class Mesh>
    inline auto get_index_start(const Mesh& mesh, const typename Mesh::mesh_interval_t& mesh_interval)
    {
        static constexpr std::size_t dim = Mesh::dim;
        using coord_index_t = typename Mesh::coord_index_t;

        xt::xtensor_fixed<coord_index_t, xt::xshape<dim>> coord;
        std::copy(mesh_interval.index.cbegin(), mesh_interval.index.end(), coord.begin()+1);
        coord[0] = mesh_interval.i.start;
        return mesh.get_index(mesh_interval.level, coord);
    }

    template <class Mesh, class Vector>
    inline auto get_index_start_translated(const Mesh& mesh, const typename Mesh::mesh_interval_t& mesh_interval, const Vector& translation_vect)
    {
        static constexpr std::size_t dim = Mesh::dim;
        using coord_index_t = typename Mesh::coord_index_t;

        xt::xtensor_fixed<coord_index_t, xt::xshape<dim>> coord;
        std::copy(mesh_interval.index.cbegin(), mesh_interval.index.end(), coord.begin()+1);
        coord[0] = mesh_interval.i.start;
        for (std::size_t d=0; d<dim; ++d)
        {
            coord[d] += translation_vect[d];
        }
        return mesh.get_index(mesh_interval.level, coord);
    }

    template <class Mesh, class Vector>
    inline auto get_index_start_children(const Mesh& mesh, typename Mesh::mesh_interval_t& mesh_interval, const Vector& translation_vect)
    {
        static constexpr std::size_t dim = Mesh::dim;
        using coord_index_t = typename Mesh::coord_index_t;

        xt::xtensor_fixed<coord_index_t, xt::xshape<dim>> coord;
        std::copy(mesh_interval.index.cbegin(), mesh_interval.index.cend(), coord.begin()+1);
        coord[0] = mesh_interval.i.start;
        for (std::size_t d=0; d<dim; ++d)
        {
            coord[d] = 2*coord[d] + translation_vect[d];
        }

        return mesh.get_index(mesh_interval.level+1, coord);
    }


    //
    // Functions that return cell indices
    //
    template <typename DesiredIndexType, class Mesh, class Func>
    inline void for_each_cell_index(const Mesh& mesh, const typename Mesh::mesh_interval_t& mesh_interval, Func &&f)
    {
        auto i_start = static_cast<DesiredIndexType>(get_index_start(mesh, mesh_interval));
        for(DesiredIndexType ii=0; ii<static_cast<DesiredIndexType>(mesh_interval.i.size()); ++ii)
        {
            f(i_start + ii);
        }
    }

    template <typename DesiredIndexType, class Mesh, class Func>
    inline void for_each_cell_index(const Mesh& mesh, const typename Mesh::ca_type& set, Func &&f)
    {
        for_each_meshinterval(set, [&](const auto mesh_interval)//std::size_t level, const auto& i, const auto& index)
        {
            //typename Mesh::mesh_interval_t mesh_interval(level, i, index);
            for_each_cell_index<DesiredIndexType>(mesh, mesh_interval, std::forward<Func>(f));
        });
    }

    template <typename DesiredIndexType, class Mesh, class Func>
    inline void for_each_cell_index(const Mesh& mesh, Func &&f)
    {
        using mesh_id_t = typename Mesh::mesh_id_t;
        for_each_cell_index<DesiredIndexType>(mesh, mesh[mesh_id_t::cells], std::forward<Func>(f));
    }

    template <typename DesiredIndexType, class Mesh, class Subset, class Func>
    inline void for_each_cell_index(const Mesh& mesh, std::size_t level, Subset& subset, Func &&f)
    {
        typename Mesh::mesh_interval_t mesh_interval(level);
        subset([&](const auto& i, const auto& index)
        {
            mesh_interval.i = i;
            mesh_interval.index = index;
            for_each_cell_index<DesiredIndexType>(mesh, mesh_interval, std::forward<Func>(f));
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


        static_assert(dim >= 1 || dim <= 3, "for_each_cell_and_children() not implemented for this dimension");

        auto min_level = mesh[mesh_id_t::cells].min_level();
        auto max_level = mesh[mesh_id_t::cells].max_level();

        for(std::size_t level=min_level; level<max_level; ++level)
        {
            auto set = intersection(mesh[mesh_id_t::cells_and_ghosts][level],
                                             mesh[mesh_id_t::cells][level+1])
                        .on(level);
            typename Mesh::mesh_interval_t mesh_interval(level);

            set([&](const auto& i, const auto& index)
            {
                mesh_interval.i = i;
                mesh_interval.index = index;

                auto cell_start     = static_cast<DesiredIndexType>(get_index_start(mesh, mesh_interval));
                std::array<DesiredIndexType, number_of_children> children;
                if constexpr (dim == 1)
                {
                    std::array<int, dim> translation_0 { 0 };
                    auto child_start = static_cast<DesiredIndexType>(get_index_start_children(mesh, mesh_interval, translation_0));
                    for(DesiredIndexType ii=0; ii<static_cast<DesiredIndexType>(i.size()); ++ii)
                    {
                        auto cell = cell_start + ii;
                        children[0] = child_start + 2*ii;   // left:  (2i  )
                        children[1] = child_start + 2*ii+1; // right: (2i+1)
                        f(cell, children);
                    }
                }
                else if constexpr (dim == 2)
                {
                    std::array<int, dim> j   { 0, 0 };
                    auto children_row_2j___start = static_cast<DesiredIndexType>(get_index_start_children(mesh, mesh_interval, j));
                    std::array<int, dim> jp1 { 0, 1 };
                    auto children_row_2jp1_start = static_cast<DesiredIndexType>(get_index_start_children(mesh, mesh_interval, jp1));
                    for(DesiredIndexType ii=0; ii<static_cast<DesiredIndexType>(i.size()); ++ii)
                    {
                        auto cell = cell_start + ii;
                        children[0] = children_row_2j___start + 2*ii;   // bottom-left:  (2i  , 2j  )
                        children[1] = children_row_2j___start + 2*ii+1; // bottom-right: (2i+1, 2j  )
                        children[2] = children_row_2jp1_start + 2*ii;   // top-left:     (2i  , 2j+1)
                        children[3] = children_row_2jp1_start + 2*ii+1; // top-right:    (2i+1, 2j+1)
                        f(cell, children);
                    }
                }
                else if constexpr (dim == 3)
                {
                    std::array<int, dim> j___k   { 0, 0, 0 };
                    auto children_2j___2k___start = static_cast<DesiredIndexType>(get_index_start_children(mesh, mesh_interval, j___k  ));
                    std::array<int, dim> jp1_k   { 0, 1, 0 };
                    auto children_2jp1_2k___start = static_cast<DesiredIndexType>(get_index_start_children(mesh, mesh_interval, jp1_k  ));
                    std::array<int, dim> j___kp1 { 0, 0, 1 };
                    auto children_2j___2kp1_start = static_cast<DesiredIndexType>(get_index_start_children(mesh, mesh_interval, j___kp1));
                    std::array<int, dim> jp1_kp1 { 0, 1, 1 };
                    auto children_2jp1_2kp1_start = static_cast<DesiredIndexType>(get_index_start_children(mesh, mesh_interval, jp1_kp1));
                    for(DesiredIndexType ii=0; ii<static_cast<DesiredIndexType>(i.size()); ++ii)
                    {
                        auto cell = cell_start + ii;
                        children[0] = children_2j___2k___start + 2*ii;   // (2i  , 2j  , 2k  )
                        children[1] = children_2j___2k___start + 2*ii+1; // (2i+1, 2j  , 2k  )
                        children[2] = children_2jp1_2k___start + 2*ii;   // (2i  , 2j+1, 2k  )
                        children[3] = children_2jp1_2k___start + 2*ii+1; // (2i+1, 2j+1, 2k  )
                        children[4] = children_2jp1_2kp1_start + 2*ii;   // (2i  , 2j+1, 2k+1)
                        children[5] = children_2jp1_2kp1_start + 2*ii+1; // (2i+1, 2j+1, 2k+1)
                        children[6] = children_2j___2kp1_start + 2*ii;   // (2i  , 2j  , 2k+1)
                        children[7] = children_2j___2kp1_start + 2*ii+1; // (2i+1, 2j  , 2k+1)
                        f(cell, children);
                    }
                }
            });
        }
    }

    /**
     * Used for the allocation of the matrix rows where the projection operator is used.
     */
    template <class Mesh, class Func>
    inline void for_each_cell_having_children(const Mesh& mesh, Func &&f)
    {
        using mesh_id_t = typename Mesh::mesh_id_t;

        auto min_level = mesh[mesh_id_t::cells].min_level();
        auto max_level = mesh[mesh_id_t::cells].max_level();

        for(std::size_t level=min_level; level<max_level; ++level)
        {
            auto set = intersection(mesh[mesh_id_t::cells_and_ghosts][level],
                                             mesh[mesh_id_t::cells][level+1])
                        .on(level);

            for_each_cell(mesh, set, std::forward<Func>(f));
        }
    }

    /**
     * Used for the allocation of the matrix rows where the prediction operator is used.
     */
    template <class Mesh, class Func>
    inline void for_each_cell_having_parent(const Mesh& mesh, Func &&f)
    {
        using mesh_id_t = typename Mesh::mesh_id_t;

        auto min_level = mesh[mesh_id_t::cells].min_level();
        auto max_level = mesh[mesh_id_t::cells].max_level();

        for(std::size_t level=min_level+1; level<=max_level; ++level)
        {
            auto set = intersection(mesh[mesh_id_t::cells_and_ghosts][level],
                                             mesh[mesh_id_t::cells][level-1])
                    .on(level);

            for_each_cell(mesh, set, std::forward<Func>(f));
        }
    }

    template <typename DesiredIndexType, class Mesh, class Func>
    inline void for_each_outside_ghost_index(const Mesh& mesh, Func &&f)
    {
        using mesh_id_t = typename Mesh::mesh_id_t;
        for_each_level(mesh, [&](std::size_t level, double)
        {
            auto boundary_ghosts = difference(mesh[mesh_id_t::cells_and_ghosts][level], mesh.domain()).on(level);
            for_each_cell_index<DesiredIndexType>(mesh, level, boundary_ghosts, std::forward<Func>(f));
        });
    }
}