#pragma once
#include "algorithm.hpp"

namespace samurai
{
    template <class Mesh>
    inline auto get_index_start(const Mesh& mesh, const typename Mesh::mesh_interval_t& mesh_interval)
    {
        static constexpr std::size_t dim = Mesh::dim;
        using value_t                    = typename Mesh::value_t;

        xt::xtensor_fixed<value_t, xt::xshape<dim>> coord;
        std::copy(mesh_interval.index.cbegin(), mesh_interval.index.end(), coord.begin() + 1);
        coord[0] = mesh_interval.i.start;
        return mesh.get_index(mesh_interval.level, coord);
    }

    template <class Mesh, class Vector>
    inline auto
    get_index_start_translated(const Mesh& mesh, const typename Mesh::mesh_interval_t& mesh_interval, const Vector& translation_vect)
    {
        static constexpr std::size_t dim = Mesh::dim;
        using value_t                    = typename Mesh::value_t;

        xt::xtensor_fixed<value_t, xt::xshape<dim>> coord;
        std::copy(mesh_interval.index.cbegin(), mesh_interval.index.end(), coord.begin() + 1);
        coord[0] = mesh_interval.i.start;
        for (std::size_t d = 0; d < dim; ++d)
        {
            coord[d] += translation_vect[d];
        }
        return mesh.get_index(mesh_interval.level, coord);
    }

    template <class Mesh, class Vector>
    inline auto get_index_start_children(const Mesh& mesh, typename Mesh::mesh_interval_t& mesh_interval, const Vector& translation_vect)
    {
        static constexpr std::size_t dim = Mesh::dim;
        using value_t                    = typename Mesh::value_t;

        xt::xtensor_fixed<value_t, xt::xshape<dim>> coord;
        std::copy(mesh_interval.index.cbegin(), mesh_interval.index.cend(), coord.begin() + 1);
        coord[0] = mesh_interval.i.start;
        for (std::size_t d = 0; d < dim; ++d)
        {
            coord[d] = 2 * coord[d] + translation_vect[d];
        }
        return mesh.get_index(mesh_interval.level + 1, coord);
    }

    /**
     * Used to define the projection operator.
     */
    template <typename DesiredIndexType, class Mesh, class Func>
    inline void for_each_projection_ghost_and_children_cells(const Mesh& mesh, Func&& f)
    {
        using mesh_id_t                                 = typename Mesh::mesh_id_t;
        using mesh_interval_t                           = typename Mesh::mesh_interval_t;
        static constexpr std::size_t dim                = Mesh::dim;
        static constexpr std::size_t number_of_children = (1 << dim);

        static_assert(dim >= 1 && dim <= 3,
                      "for_each_projection_ghost_and_children_cells() not "
                      "implemented for this dimension");

        auto min_level = mesh[mesh_id_t::cells].min_level();
        auto max_level = mesh[mesh_id_t::cells].max_level();

        for (std::size_t level = min_level; level < max_level; ++level)
        {
            auto projection_ghosts = intersection(mesh[mesh_id_t::cells_and_ghosts][level], mesh[mesh_id_t::cells][level + 1]).on(level);
            for_each_meshinterval<mesh_interval_t>(
                projection_ghosts,
                [&](auto mesh_interval)
                {
                    auto& i = mesh_interval.i;

                    auto cell_start = static_cast<DesiredIndexType>(get_index_start(mesh, mesh_interval));
                    std::array<DesiredIndexType, number_of_children> children;
                    if constexpr (dim == 1)
                    {
                        std::array<int, dim> translation_0{0};
                        auto child_start = static_cast<DesiredIndexType>(get_index_start_children(mesh, mesh_interval, translation_0));
                        for (DesiredIndexType ii = 0; ii < static_cast<DesiredIndexType>(i.size()); ++ii)
                        {
                            auto cell   = cell_start + ii;
                            children[0] = child_start + 2 * ii;     // left:  (2i  )
                            children[1] = child_start + 2 * ii + 1; // right: (2i+1)
                            f(level, cell, children);
                        }
                    }
                    else if constexpr (dim == 2)
                    {
                        std::array<int, dim> j{0, 0};
                        auto children_row_2j_start = static_cast<DesiredIndexType>(get_index_start_children(mesh, mesh_interval, j));
                        std::array<int, dim> jp1{0, 1};
                        auto children_row_2jp1_start = static_cast<DesiredIndexType>(get_index_start_children(mesh, mesh_interval, jp1));
                        for (DesiredIndexType ii = 0; ii < static_cast<DesiredIndexType>(i.size()); ++ii)
                        {
                            auto cell   = cell_start + ii;
                            children[0] = children_row_2j_start + 2 * ii;       // bottom-left:  (2i  , 2j  )
                            children[1] = children_row_2j_start + 2 * ii + 1;   // bottom-right: (2i+1, 2j  )
                            children[2] = children_row_2jp1_start + 2 * ii;     // top-left:     (2i  , 2j+1)
                            children[3] = children_row_2jp1_start + 2 * ii + 1; // top-right:    (2i+1, 2j+1)
                            f(level, cell, children);
                        }
                    }
                    else if constexpr (dim == 3)
                    {
                        std::array<int, dim> j_k{0, 0, 0};
                        auto children_2j_2k_start = static_cast<DesiredIndexType>(get_index_start_children(mesh, mesh_interval, j_k));
                        std::array<int, dim> jp1_k{0, 1, 0};
                        auto children_2jp1_2k_start = static_cast<DesiredIndexType>(get_index_start_children(mesh, mesh_interval, jp1_k));
                        std::array<int, dim> j_kp1{0, 0, 1};
                        auto children_2j_2kp1_start = static_cast<DesiredIndexType>(get_index_start_children(mesh, mesh_interval, j_kp1));
                        std::array<int, dim> jp1_kp1{0, 1, 1};
                        auto children_2jp1_2kp1_start = static_cast<DesiredIndexType>(get_index_start_children(mesh, mesh_interval, jp1_kp1));
                        for (DesiredIndexType ii = 0; ii < static_cast<DesiredIndexType>(i.size()); ++ii)
                        {
                            auto cell   = cell_start + ii;
                            children[0] = children_2j_2k_start + 2 * ii;         // (2i  , 2j  , 2k  )
                            children[1] = children_2j_2k_start + 2 * ii + 1;     // (2i+1, 2j  , 2k  )
                            children[2] = children_2jp1_2k_start + 2 * ii;       // (2i  , 2j+1, 2k  )
                            children[3] = children_2jp1_2k_start + 2 * ii + 1;   // (2i+1, 2j+1, 2k  )
                            children[4] = children_2jp1_2kp1_start + 2 * ii;     // (2i  , 2j+1, 2k+1)
                            children[5] = children_2jp1_2kp1_start + 2 * ii + 1; // (2i+1, 2j+1, 2k+1)
                            children[6] = children_2j_2kp1_start + 2 * ii;       // (2i  , 2j  , 2k+1)
                            children[7] = children_2j_2kp1_start + 2 * ii + 1;   // (2i+1, 2j  , 2k+1)
                            f(level, cell, children);
                        }
                    }
                });
        }
    }

    /**
     * Used for the allocation of the matrix rows where the projection operator
     * is used.
     */
    template <class Mesh, class Func>
    inline void for_each_projection_ghost(const Mesh& mesh, Func&& f)
    {
        using mesh_id_t = typename Mesh::mesh_id_t;

        auto min_level = mesh[mesh_id_t::cells].min_level();
        auto max_level = mesh[mesh_id_t::cells].max_level();

        for (std::size_t level = min_level; level < max_level; ++level)
        {
            auto set = intersection(mesh[mesh_id_t::cells_and_ghosts][level], mesh[mesh_id_t::cells][level + 1]).on(level);

            for_each_cell(mesh, set, std::forward<Func>(f));
        }
    }

    /**
     * Used for the allocation of the matrix rows where the prediction operator
     * is used.
     */
    template <class Mesh, class Func>
    inline void for_each_prediction_ghost(const Mesh& mesh, Func&& f)
    {
        using mesh_id_t = typename Mesh::mesh_id_t;

        auto min_level = mesh[mesh_id_t::cells].min_level();
        auto max_level = mesh[mesh_id_t::cells].max_level();

        for (std::size_t level = min_level + 1; level <= max_level; ++level)
        {
            auto set = intersection(mesh[mesh_id_t::cells_and_ghosts][level], mesh[mesh_id_t::cells][level - 1]).on(level);

            for_each_cell(mesh, set, std::forward<Func>(f));
        }
    }

    template <class Mesh, class Func>
    inline void for_each_outside_ghost(const Mesh& mesh, Func&& f)
    {
        using mesh_id_t = typename Mesh::mesh_id_t;

        auto min_level = mesh[mesh_id_t::reference].min_level();
        auto max_level = mesh[mesh_id_t::reference].max_level();

        for (std::size_t level = min_level; level <= max_level; ++level)
        {
            auto set = difference(mesh[mesh_id_t::reference][level], mesh.domain()).on(level);

            for_each_cell(mesh, set, std::forward<Func>(f));
        }
    }
}
