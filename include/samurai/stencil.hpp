#pragma once
#include "indices.hpp"
#include "static_algorithm.hpp"

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

    template <class T>
    bool is_cartesian(const T& direction)
    {
        return xt::sum(xt::abs(direction))[0] == 1; // only one non-zero in the vector
    }

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

    template <std::size_t dim>
    std::size_t find_direction_index(const DirectionVector<dim>& direction)
    {
        for (std::size_t i = 0; i < dim; ++i)
        {
            if (direction(i) != 0)
            {
                return i;
            }
        }
        return 0;
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
            if (origin_cell.index > 0 && static_cast<std::size_t>(origin_cell.index) > m_mesh.nb_cells())
            {
                std::cout << "Cell not found in the mesh: " << origin_cell << std::endl;
                assert(false);
            }

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
                    if (cell.index > 0 && static_cast<std::size_t>(cell.index) > m_mesh.nb_cells())
                    {
                        std::cout << "Non-existing neighbour for " << origin_cell << " in the direction " << d << std::endl;
                        assert(false);
                    }
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

    template <class Mesh, std::size_t stencil_size, class Func>
    inline void for_each_stencil(const Mesh& mesh, const Stencil<stencil_size, Mesh::dim>& stencil, Func&& f)
    {
        auto stencil_it = make_stencil_iterator(mesh, stencil);
        for_each_level(mesh,
                       [&](std::size_t level)
                       {
                           for_each_stencil(mesh, level, stencil_it, std::forward<Func>(f));
                       });
    }

    template <class Set, class Mesh, std::size_t stencil_size, class Func>
    inline void for_each_stencil(Set& set, IteratorStencil<Mesh, stencil_size>& stencil_it, Func&& f)
    {
        using mesh_interval_t = typename IteratorStencil<Mesh, stencil_size>::mesh_interval_t;
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

    template <std::size_t dim, std::size_t d, class... Ints>
    auto line_stencil(Ints... neighbours)
    {
        static constexpr std::size_t stencil_size = sizeof...(Ints);

        std::array<int, stencil_size> neighbours_array = {{neighbours...}};

        Stencil<stencil_size, dim> s;
        s.fill(0);
        static_for<0, stencil_size>::apply( // for (int i=0; i<stencil_size; i++)
            [&](auto integral_constant_i)
            {
                static constexpr std::size_t i = decltype(integral_constant_i)::value;

                s(i, d) = neighbours_array[i];
            });
        return s;
    }

    template <std::size_t dim, std::size_t d, std::size_t stencil_size>
    auto line_stencil_from(int from)
    {
        Stencil<stencil_size, dim> s;
        s.fill(0);
        static_for<0, stencil_size>::apply( // for (int i=0; i<stencil_size; i++)
            [&](auto integral_constant_i)
            {
                static constexpr std::size_t i = decltype(integral_constant_i)::value;

                s(i, d) = from + static_cast<int>(i);
            });

        return s;
    }

    template <std::size_t dim, std::size_t d, std::size_t stencil_size>
    auto line_stencil()
    {
        return line_stencil_from<dim, d, stencil_size>(-static_cast<int>(stencil_size) / 2 + 1);
    }

    template <std::size_t stencil_size, std::size_t dim>
    bool is_line_stencil([[maybe_unused]] Stencil<stencil_size, dim>& stencil)
    {
        if constexpr (dim > 1)
        {
            std::size_t dir      = 0;
            bool direction_found = false;
            for (std::size_t i = 0; i < stencil_size; ++i)
            {
                for (std::size_t d = 0; d < dim; ++d)
                {
                    if (stencil(i, d) != 0)
                    {
                        if (!direction_found)
                        {
                            dir             = d;
                            direction_found = true;
                        }
                        else if (dir != d) // another direction found
                        {
                            return false;
                        }
                    }
                }
            }
        }
        return true;
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
        static_assert(dim >= 1 && dim <= 3, "directional_stencils() not implemented for this dimension");
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
            else if constexpr (dim == 3)
            {
                // Left
                dir_stencils[0].direction = {-1, 0, 0};
                dir_stencils[0].stencil   = {{0,  0, 0}, {1,  0, 0}, {-1, 0, 0}};
                // Top
                dir_stencils[1].direction = {0, 1, 0};
                dir_stencils[1].stencil   = {{0, 0 ,0}, {0, -1,0}, {0, 1,0 }};
                // Right
                dir_stencils[2].direction = {1, 0,0};
                dir_stencils[2].stencil   = {{0,  0,0}, {-1, 0,0}, {1,  0,0}};
                // Bottom
                dir_stencils[3].direction = {0, -1,0};
                dir_stencils[3].stencil   = {{0, 0,0 }, {0, 1,0 }, {0, -1,0}};
                // Back
                dir_stencils[4].direction = {0, 0,-1};
                dir_stencils[4].stencil   = {{0,  0,0}, {0, 0,1}, {0,  0,-1}};
                // Front
                dir_stencils[5].direction = {0, 0, 1};
                dir_stencils[5].stencil   = {{0, 0,0 }, {0, 0, -1}, {0, 0,1}};
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
            else if constexpr (dim == 3)
            {
                // Left
                dir_stencils[0].direction = {-1, 0, 0};
                dir_stencils[0].stencil   = {{0,  0, 0}, {1,  0, 0}, {2,  0, 0}, {-1, 0, 0}, {-2, 0, 0}};
                // Top
                dir_stencils[1].direction = {0, 1, 0};
                dir_stencils[1].stencil   = {{0, 0 , 0}, {0, -1, 0}, {0, -2, 0}, {0, 1 , 0}, {0, 2 , 0}};
                // Right
                dir_stencils[2].direction = {1, 0, 0};
                dir_stencils[2].stencil   = {{0,  0, 0}, {-1, 0, 0}, {-2, 0, 0}, {1,  0, 0}, {2,  0, 0}};
                // Bottom
                dir_stencils[3].direction = {0, -1, 0};
                dir_stencils[3].stencil   = {{0, 0 , 0}, {0, 1 , 0}, {0, 2 , 0}, {0, -1, 0}, {0, -2, 0}};
                // Back
                dir_stencils[4].direction = {0, 0, -1};
                dir_stencils[4].stencil   = {{0,  0, 0}, {0, 0, 1}, {0, 0, 2}, {0,  0, -1}, {0,  0, -2}};
                // Front
                dir_stencils[5].direction = {0, 0, 1};
                dir_stencils[5].stencil   = {{0,  0, 0}, {0, 0, -1}, {0, 0, -2}, {0,  0, 1}, {0,  0, 2}};
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

    template <std::size_t stencil_size, std::size_t dim>
    Stencil<stencil_size, dim> convert_for_direction(const Stencil<stencil_size, dim>& stencil_in_x, const DirectionVector<dim>& direction)
    {
        Stencil<stencil_size, dim> stencil_in_d;
        if constexpr (dim == 1)
        {
            stencil_in_d = direction[0] * stencil_in_x;
        }
        else
        {
            // We apply to each vector in the stencil the rotation matrix
            // that converts the first canonical vector, i.e. {1, 0}, into 'direction'.

            auto d = find_direction_index(direction);
            // If d is the x-direction, then we choose any other direction (e.g. the next one)
            // to perform the rotation.
            d = d == 0 ? d + 1 : d;

            stencil_in_d = stencil_in_x; // 1 on the diagonal of the rotation matrix

            int cos = direction(0);
            int sin = direction(d);
            static_for<0, stencil_size>::apply( // for (int i=0; i<stencil_size; i++)
                [&](auto integral_constant_i)
                {
                    static constexpr std::size_t i = decltype(integral_constant_i)::value;

                    auto v_in_x = xt::view(stencil_in_x, i); // vector in direction x (canonical basis)
                    auto v_in_d = xt::view(stencil_in_d, i); // vector in direction d (=rotation of v_in_x)

                    v_in_d(0) = cos * v_in_x(0) - sin * v_in_x(d);
                    v_in_d(d) = sin * v_in_x(0) + cos * v_in_x(d);
                });
        }
        return stencil_in_d;
    }
}
