#pragma once
#include "indices.hpp"

namespace samurai
{
    template <std::size_t stencil_size, std::size_t dim>
    using Stencil = xt::xtensor_fixed<int, xt::xshape<stencil_size, dim>>;

    template <std::size_t dim>
    using DirectionVector = xt::xtensor_fixed<int, xt::xshape<dim>>;

    template <std::size_t stencil_size, std::size_t dim>
    struct DirectionalStencil
    {
        DirectionVector<dim> direction;
        Stencil<stencil_size, dim> stencil;
    };

    template <std::size_t stencil_size, std::size_t dim>
    int find_stencil_origin(const Stencil<stencil_size, dim>& stencil)
    {
        for (unsigned int id = 0; id < stencil_size; ++id)
        {
            auto d              = xt::view(stencil, id);
            bool is_zero_vector = true;
            for (unsigned int i = 0; i < dim; ++i)
            {
                if (d[i] != 0)
                {
                    is_zero_vector = false;
                    break;
                }
            }
            if (is_zero_vector)
            {
                return static_cast<int>(id);
            }
        }
        return -1;
    }

    template <std::size_t stencil_size, std::size_t dim>
    int find(const Stencil<stencil_size, dim>& stencil, const DirectionVector<dim>& vector)
    {
        for (unsigned int id = 0; id < stencil_size; ++id)
        {
            auto d     = xt::view(stencil, id);
            bool found = true;
            for (unsigned int i = 0; i < dim; ++i)
            {
                if (d[i] != vector[i])
                {
                    found = false;
                    break;
                }
            }
            if (found)
            {
                return static_cast<int>(id);
            }
        }
        return -1;
    }

    template <class Mesh, std::size_t stencil_size>
    class IteratorStencil
    {
      public:

        static constexpr std::size_t dim = Mesh::dim;
        using mesh_interval_t            = typename Mesh::mesh_interval_t;
        using coord_index_t              = typename Mesh::config::interval_t::coord_index_t;
        using cell_t                     = Cell<dim, typename Mesh::interval_t>;

      private:

        const Mesh& m_mesh;                         // NOLINT(cppcoreguidelines-avoid-const-or-ref-data-members)
        const Stencil<stencil_size, dim> m_stencil; // NOLINT(cppcoreguidelines-avoid-const-or-ref-data-members)
        std::array<cell_t, stencil_size> m_cells;
        unsigned int m_origin_cell;

      public:

        IteratorStencil(const Mesh& mesh, const Stencil<stencil_size, dim>& stencil)
            : m_mesh(mesh)
            , m_stencil(stencil)
        {
            int origin_index = find_stencil_origin(stencil);
            assert(origin_index >= 0 && "the zero vector is required in the stencil definition.");
            m_origin_cell = static_cast<unsigned int>(origin_index);
        }

        void init(const mesh_interval_t& mesh_interval)
        {
            double length = cell_length(mesh_interval.level);
            for (cell_t& cell : m_cells)
            {
                cell.level  = mesh_interval.level;
                cell.length = length;
            }

            // origin of the stencil
            cell_t& origin_cell    = m_cells[m_origin_cell];
            origin_cell.indices[0] = mesh_interval.i.start;
            for (unsigned int d = 0; d < dim - 1; ++d)
            {
                origin_cell.indices[d + 1] = mesh_interval.index[d];
            }
            origin_cell.index = get_index_start(m_mesh, mesh_interval);

            for (unsigned int id = 0; id < stencil_size; ++id)
            {
                if (id == m_origin_cell)
                {
                    continue;
                }

                auto d = xt::view(m_stencil, id);

                // Translate the coordinates according the direction d
                cell_t& cell = m_cells[id];
                for (unsigned int k = 0; k < dim; ++k)
                {
                    cell.indices[k] = origin_cell.indices[k] + d[k];
                }

                // We are on the same row as the stencil origin if d = {d[0], 0,..., 0}
                bool same_row = true;
                for (std::size_t k = 1; k < dim; ++k)
                {
                    if (d[k] != 0)
                    {
                        same_row = false;
                        break;
                    }
                }
                if (same_row) // same row as the stencil origin
                {
                    // translation on the row
                    cell.index = origin_cell.index + d[0];
                }
                else
                {
                    cell.index = get_index_start_translated(m_mesh, mesh_interval, d);
                }
            }
        }

        void move_next()
        {
            for (cell_t& cell : m_cells)
            {
                cell.index++;      // increment cell index
                cell.indices[0]++; // increment x-coordinate
            }
        }

        std::array<cell_t, stencil_size>& cells()
        {
            return m_cells;
        }
    };

    template <class Mesh, std::size_t stencil_size>
    auto make_stencil_iterator(const Mesh& mesh, const Stencil<stencil_size, Mesh::dim>& stencil)
    {
        return IteratorStencil<Mesh, stencil_size>(mesh, stencil);
    }

    template <class iterator_stencil, class Func>
    inline void for_each_stencil_sliding_in_interval(const typename iterator_stencil::mesh_interval_t& mesh_interval,
                                                     iterator_stencil& stencil_it,
                                                     Func&& f)
    {
        stencil_it.init(mesh_interval);
        for (std::size_t ii = 0; ii < mesh_interval.i.size(); ++ii)
        {
            f(stencil_it.cells());
            stencil_it.move_next();
        }
    }

    template <class Mesh, std::size_t stencil_size, class Func>
    inline void for_each_stencil_sliding_in_interval(const Mesh& mesh,
                                                     const typename Mesh::mesh_interval_t& mesh_interval,
                                                     const Stencil<stencil_size, Mesh::dim>& stencil,
                                                     Func&& f)
    {
        auto stencil_it = make_stencil_iterator(mesh, stencil);
        for_each_stencil_sliding_in_interval(mesh_interval, stencil_it, std::forward<Func>(f));
    }

    template <class Mesh, std::size_t stencil_size, class Func>
    inline void for_each_stencil(const Mesh& mesh, std::size_t level, IteratorStencil<Mesh, stencil_size>& stencil_it, Func&& f)
    {
        using mesh_id_t = typename Mesh::mesh_id_t;
        for_each_meshinterval(mesh[mesh_id_t::cells][level],
                              [&](auto mesh_interval)
                              {
                                  for_each_stencil_sliding_in_interval(mesh_interval, stencil_it, std::forward<Func>(f));
                              });
    }

    template <class Set, class iterator_stencil, class Func>
    inline void for_each_stencil(Set& set, iterator_stencil& stencil_it, Func&& f)
    {
        using mesh_interval_t = typename iterator_stencil::mesh_interval_t;
        for_each_meshinterval<mesh_interval_t>(set,
                                               [&](auto mesh_interval)
                                               {
                                                   for_each_stencil_sliding_in_interval(mesh_interval, stencil_it, std::forward<Func>(f));
                                               });
    }

    template <class Mesh, class Set, std::size_t stencil_size, class Func>
    inline void for_each_stencil(const Mesh& mesh, Set& set, const Stencil<stencil_size, Mesh::dim>& stencil, Func&& f)
    {
        auto stencil_it = make_stencil_iterator(mesh, stencil);
        for_each_stencil(set, stencil_it, std::forward<Func>(f));
    }

    //-----------------------//
    //    Useful stencils    //
    //-----------------------//

    template <std::size_t dim, std::size_t neighbourhood_width = 1>
    constexpr Stencil<1 + 2 * dim * neighbourhood_width, dim> star_stencil()
    {
        static_assert(dim >= 1 && dim <= 3, "Star stencil not implemented for this dimension");
        static_assert(neighbourhood_width >= 0 && neighbourhood_width <= 2, "Star stencil not implemented for this neighbourhood width");

        if constexpr (neighbourhood_width == 0)
        {
            Stencil<1, dim> s;
            s.fill(0);
            return s;
        }
        else if constexpr (neighbourhood_width == 1)
        {
            // clang-format off
            if constexpr (dim == 1)
            {
                //    left, center, right
                return {{-1}, {0}, {1}};
            }
            else if constexpr (dim == 2)
            {
                //       left,   center,  right,  bottom,  top
                return {{-1, 0}, {0, 0}, {1, 0}, {0, -1}, {0, 1} };
            }
            else if constexpr (dim == 3)
            {
                //        left,      center,     right,     front,       back,     bottom,      top
                return {{-1, 0, 0}, {0, 0, 0}, {1, 0, 0}, {0, -1, 0}, {0, 1, 0}, {0, 0, -1}, {0, 0, 1}};
            }
            // clang-format on
        }
        else if constexpr (neighbourhood_width == 2)
        {
            // clang-format off
            if constexpr (dim == 1)
            {
                //   left2, left, center, right, right2
                return {{-2}, {-1}, {0}, {1}, {2}};
            }
            else if constexpr (dim == 2)
            {
                //       left2,   left,   center,  right, right2  bottom2, bottom,   top,    top2
                return {{-2, 0}, {-1, 0}, {0, 0}, {1, 0}, {2, 0}, {0, -2}, {0, -1}, {0, 1}, {0, 2}};
            }
            else if constexpr (dim == 3)
            {
                //        left2,       left,     center,     right,    right2,    front2,      front,      back,      back2,    bottom2,     bottom,      top,      top2
                return {{-2, 0, 0}, {-1, 0, 0}, {0, 0, 0}, {1, 0, 0}, {2, 0, 0}, {0, -2, 0}, {0, -1, 0}, {0, 1, 0}, {0, 2, 0}, {0, 0, -2}, {0, 0, -1}, {0, 0, 1}, {0, 0, 2}};
            }
            // clang-format on
        }
        return Stencil<1 + 2 * dim * neighbourhood_width, dim>();
    }

    template <std::size_t dim>
    constexpr Stencil<2 * dim, dim> cartesian_directions()
    {
        static_assert(dim >= 1 && dim <= 3, "cartesian_directions() not implemented for this dimension");

        // clang-format off
        if constexpr (dim == 1)
        {
            //     left, right
            return {{-1}, {1}};
        }
        else if constexpr (dim == 2)
        {
            //       left,   right,   bottom,  top
            return {{-1, 0}, {1, 0}, {0, -1}, {0, 1}};
        }
        else if constexpr (dim == 3)
        {
            //         left,      right,     front,      back,      bottom,      top
            return {{-1, 0, 0}, {1, 0, 0}, {0, -1, 0}, {0, 1, 0}, {0, 0, -1}, {0, 0, 1}};
        }
        // clang-format on
        return Stencil<2 * dim, dim>();
    }

    template <std::size_t dim>
    constexpr Stencil<dim, dim> positive_cartesian_directions()
    {
        static_assert(dim >= 1 && dim <= 3, "positive_cartesian_directions() not implemented for this dimension");
        // clang-format off
        if constexpr (dim == 1)
        {
            //     right
            return {{1}};
        }
        else if constexpr (dim == 2)
        {
            //      right,   top
            return {{1, 0}, {0, 1}};
        }
        else if constexpr (dim == 3)
        {
            //        right,     back,       top
            return {{1, 0, 0}, {0, 1, 0}, {0, 0, 1}};
        }
        // clang-format on
        return Stencil<dim, dim>();
    }

    /**
     * Returns a table of the form
     *      dir_stencils[i].direction = {direction};
     *      dir_stencils[i].stencil   = {{center}, {-directions...}, {directions...}};
     * so that on the boundary, we have     |             |                 |
     *                                current cell      cells             ghosts
     */
    template <std::size_t dim, std::size_t neighbourhood_width = 1>
    auto directional_stencils()
    {
        static_assert(dim >= 1 && dim <= 2, "directional_stencils() not implemented for this dimension");
        static_assert(neighbourhood_width >= 0 && neighbourhood_width <= 2,
                      "directional_stencils() not implemented for this neighbourhood width");

        static constexpr std::size_t stencil_size      = 1 + 2 * neighbourhood_width;
        static constexpr std::size_t n_cart_directions = 2 * dim;

        std::array<DirectionalStencil<stencil_size, dim>, n_cart_directions> dir_stencils;

        if constexpr (neighbourhood_width == 0)
        {
            dir_stencils[0].direction.fill(0);
            dir_stencils[0].stencil.fill(0);
        }
        else if constexpr (neighbourhood_width == 1)
        {
            // clang-format off
            if constexpr (dim == 1)
            {
                // Left
                dir_stencils[0].direction = {-1};
                dir_stencils[0].stencil   = {{0}, {1}, {-1}};
                // Right
                dir_stencils[1].direction = {1};
                dir_stencils[1].stencil   = {{0}, {-1}, {1}};
            }
            else if constexpr (dim == 2)
            {
                // Left
                dir_stencils[0].direction = {-1, 0};
                dir_stencils[0].stencil   = {{0,  0}, {1,  0}, {-1, 0}};
                // Top
                dir_stencils[1].direction = {0, 1};
                dir_stencils[1].stencil   = {{0, 0 }, {0, -1}, {0, 1 }};
                // Right
                dir_stencils[2].direction = {1, 0};
                dir_stencils[2].stencil   = {{0,  0}, {-1, 0}, {1,  0}};
                // Bottom
                dir_stencils[3].direction = {0, -1};
                dir_stencils[3].stencil   = {{0, 0 }, {0, 1 }, {0, -1}};
            }
            // clang-format on
        }
        else if constexpr (neighbourhood_width == 2)
        {
            // clang-format off
            if constexpr (dim == 1)
            {
                // Left
                dir_stencils[0].direction = {-1};
                dir_stencils[0].stencil   = {{0}, {1}, {2}, {-1}, {-2}};
                // right
                dir_stencils[1].direction = {1};
                dir_stencils[1].stencil   = {{0}, {-1}, {-2}, {1}, {2}};
            }
            else if constexpr (dim == 2)
            {
                // Left
                dir_stencils[0].direction = {-1, 0};
                dir_stencils[0].stencil   = {{0,  0}, {1,  0}, {2,  0}, {-1, 0}, {-2, 0}};
                // Top
                dir_stencils[1].direction = {0, 1};
                dir_stencils[1].stencil   = {{0, 0 }, {0, -1}, {0, -2}, {0, 1 }, {0, 2 }};
                // Right
                dir_stencils[2].direction = {1, 0};
                dir_stencils[2].stencil   = {{0,  0}, {-1, 0}, {-2, 0}, {1,  0}, {2,  0}};
                // Bottom
                dir_stencils[3].direction = {0, -1};
                dir_stencils[3].stencil   = {{0, 0 }, {0, 1 }, {0, 2 }, {0, -1}, {0, -2}};
            }
            // clang-format on
        }
        return dir_stencils;
    }

    template <std::size_t dim>
    constexpr Stencil<1, dim> center_only_stencil()
    {
        return star_stencil<dim, 0>();
    }

    template <std::size_t dim, class Vector>
    Stencil<2, dim> in_out_stencil(const Vector& towards_out_from_in)
    {
        auto stencil_shape         = Stencil<2, dim>();
        xt::view(stencil_shape, 0) = 0;
        xt::view(stencil_shape, 1) = towards_out_from_in;
        return stencil_shape;
    }
}
