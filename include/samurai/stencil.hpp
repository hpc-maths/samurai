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
    struct StencilAnalyzer
    {
        std::size_t origin_index = 0;
        bool has_origin          = false;
        std::array<bool, stencil_size> same_row_as_origin;
        Stencil<stencil_size, dim> stencil;

        StencilAnalyzer()
        {
            same_row_as_origin.fill(false);
        }

        explicit StencilAnalyzer(const Stencil<stencil_size, dim>& stencil_)
            : stencil(stencil_)
        {
            init();
        }

        explicit StencilAnalyzer(Stencil<stencil_size, dim>&& stencil_)
            : stencil(std::move(stencil_))
        {
            init();
        }

        void operator=(Stencil<stencil_size, dim>&& stencil_)
        {
            stencil = std::move(stencil_);
            init();
        }

        void init()
        {
            for (std::size_t id = 0; id < stencil_size; ++id)
            {
                auto dir = xt::view(stencil, id);

                bool is_zero_vector = true;
                bool same_row       = true;
                for (std::size_t i = 0; i < dim; ++i)
                {
                    if (dir[i] != 0)
                    {
                        is_zero_vector = false;
                        if (i > 0)
                        {
                            // We are on the same row as the stencil origin if dir = {dir[0], 0,..., 0}
                            same_row = false;
                            break;
                        }
                    }
                }
                if (is_zero_vector)
                {
                    has_origin   = true;
                    origin_index = id;
                }
                same_row_as_origin[id] = same_row;
            }
        }

        int find(const DirectionVector<dim>& vector) const
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
    };

    template <std::size_t stencil_size, std::size_t dim>
    auto make_stencil_analyzer(const Stencil<stencil_size, dim>& stencil)
    {
        return StencilAnalyzer<stencil_size, dim>(stencil);
    }

    template <std::size_t stencil_size, std::size_t dim>
    struct DirectionalStencil
    {
        DirectionVector<dim> direction;
        StencilAnalyzer<stencil_size, dim> stencil;
    };

    //-----------------------//
    //    Useful functions   //
    //-----------------------//

    template <class T>
    bool is_cartesian(const T& direction)
    {
        return xt::sum(xt::abs(direction))[0] == 1; // only one non-zero in the vector
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

    template <std::size_t dim, class Func>
    void for_each_cartesian_direction(Func&& f)
    {
        DirectionVector<dim> direction;
        direction.fill(0);
        for (std::size_t d = 0; d < dim; ++d)
        {
            direction[d] = 1;
            f(d, direction);
            direction[d] = -1;
            f(d, direction);
            direction[d] = 0;
        }
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
        Stencil<dim, dim> s;
        s.fill(0);
        for (std::size_t i = 0; i < dim; ++i)
        {
            s(i, i) = 1;
        }
        return s;
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
        Stencil<1, dim> s;
        s.fill(0);
        return s;
    }

    template <std::size_t dim, class Vector>
    Stencil<2, dim> in_out_stencil(const Vector& towards_out_from_in)
    {
        auto stencil_shape         = Stencil<2, dim>();
        xt::view(stencil_shape, 0) = 0;
        xt::view(stencil_shape, 1) = towards_out_from_in;
        return stencil_shape;
    }

    template <std::size_t dim>
    static const StencilAnalyzer<1, dim> center_only_stencil_analyzer = StencilAnalyzer<1, dim>(center_only_stencil<dim>());

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

    template <class Mesh, std::size_t stencil_size_>
    class IteratorStencil
    {
      public:

        static constexpr std::size_t dim          = Mesh::dim;
        static constexpr std::size_t stencil_size = stencil_size_;
        using mesh_t                              = Mesh;
        using mesh_interval_t                     = typename Mesh::mesh_interval_t;
        using coord_index_t                       = typename Mesh::config::interval_t::coord_index_t;
        using cell_t                              = Cell<dim, typename Mesh::interval_t>;

      private:

        const Mesh& m_mesh; // NOLINT(cppcoreguidelines-avoid-const-or-ref-data-members)
        const mesh_interval_t* m_mesh_interval = nullptr;
        const StencilAnalyzer<stencil_size, dim>& m_stencil_analyzer;
        std::array<cell_t, stencil_size> m_cells;

      public:

        IteratorStencil(const Mesh& mesh, const StencilAnalyzer<stencil_size, dim>& stencil_analyzer)
            : m_mesh(mesh)
            , m_stencil_analyzer(stencil_analyzer)
        {
            assert(m_stencil_analyzer.has_origin && "the zero vector is required in the stencil definition.");
            auto length = mesh.cell_length(mesh.min_level());
            for (cell_t& cell : m_cells)
            {
                cell.origin_point = mesh.origin_point();
                cell.level        = mesh.min_level();
                cell.length       = length;
            }
        }

        void init(const mesh_interval_t& origin_mesh_interval)
        {
            if (origin_mesh_interval.level != m_cells[0].level)
            {
                double length = m_mesh.cell_length(origin_mesh_interval.level);
                for (cell_t& cell : m_cells)
                {
                    cell.level  = origin_mesh_interval.level;
                    cell.length = length;
                }
            }
            m_mesh_interval = &origin_mesh_interval;

            // origin of the stencil
            cell_t& origin_cell    = m_cells[m_stencil_analyzer.origin_index];
            origin_cell.indices[0] = origin_mesh_interval.i.start;
            for (unsigned int d = 0; d < dim - 1; ++d)
            {
                origin_cell.indices[d + 1] = origin_mesh_interval.index[d];
            }
            origin_cell.index = get_index_start(m_mesh, origin_mesh_interval);
#ifndef NDEBUG
            if (origin_cell.index > 0 && static_cast<std::size_t>(origin_cell.index) > m_mesh.nb_cells()) // nb_cells() is very costly
            {
                std::cout << "Cell not found in the mesh: " << origin_cell << std::endl;
                assert(false);
            }
#endif
            if constexpr (stencil_size > 1)
            {
                for (unsigned int i = 0; i < stencil_size; ++i)
                {
                    if (i == m_stencil_analyzer.origin_index)
                    {
                        continue;
                    }

                    // Translate the coordinates according the direction d
                    cell_t& cell = m_cells[i];
                    for (unsigned int k = 0; k < dim; ++k)
                    {
                        cell.indices[k] = origin_cell.indices[k] + m_stencil_analyzer.stencil(i, k);
                    }

                    // Find cell index
                    if (m_stencil_analyzer.same_row_as_origin[i])
                    {
                        // translation on the row
                        cell.index = origin_cell.index + m_stencil_analyzer.stencil(i, 0);
                    }
                    else
                    {
                        DirectionVector<dim> dir;
                        for (std::size_t k = 0; k < dim; ++k)
                        {
                            dir(k) = m_stencil_analyzer.stencil(i, k);
                        }
                        cell.index = get_index_start_translated(m_mesh, origin_mesh_interval, dir);
#ifndef NDEBUG
                        if (cell.index > 0 && static_cast<std::size_t>(cell.index) > m_mesh.nb_cells()) // nb_cells() is very costly
                        {
                            std::cout << "Non-existing neighbour for " << origin_cell << " in the direction " << dir << std::endl;
                            assert(false);
                        }
#endif
                    }
                }
            }
        }

        inline const auto& mesh() const
        {
            return m_mesh;
        }

        inline auto& mesh_interval() const
        {
            return *m_mesh_interval;
        }

        inline auto& interval() const
        {
            return m_mesh_interval->i;
        }

        inline auto& level() const
        {
            return m_mesh_interval->level;
        }

        inline const auto& stencil() const
        {
            return m_stencil_analyzer.stencil;
        }

        inline const auto& cells() const
        {
            return m_cells;
        }

        inline auto& cells()
        {
            return m_cells;
        }

        inline void move_next()
        {
            for (cell_t& cell : m_cells)
            {
                ++cell.index;      // increment cell index
                ++cell.indices[0]; // increment x-coordinate
            }
        }
    };

    template <std::size_t index_coarse_cell, class Mesh, std::size_t stencil_size>
    class LevelJumpIterator
    {
      public:

        static constexpr std::size_t dim = Mesh::dim;
        using interval_t                 = typename Mesh::interval_t;
        using cell_t                     = Cell<dim, interval_t>;
        using mesh_interval_t            = typename Mesh::mesh_interval_t;
        using interval_value_t           = typename interval_t::value_t;
        using cell_index_t               = typename cell_t::index_t;

        static constexpr std::size_t coarse = index_coarse_cell;
        static constexpr std::size_t fine   = (index_coarse_cell + 1) % 2;

      private:

        std::size_t m_direction_index;

        IteratorStencil<Mesh, 1> m_coarse_it;
        const IteratorStencil<Mesh, stencil_size>* m_fine_it = nullptr;
        std::array<cell_t, 2> m_cells;
        bool m_move_coarse_cell = false;

      public:

        LevelJumpIterator(const IteratorStencil<Mesh, stencil_size>& fine_it, std::size_t direction_index)
            : m_direction_index(direction_index)
            , m_coarse_it(fine_it.mesh(), center_only_stencil_analyzer<dim>)
            , m_fine_it(&fine_it)
        {
        }

        void init(const mesh_interval_t& fine_mesh_interval)
        {
            mesh_interval_t coarse_mesh_interval(fine_mesh_interval.level - 1, fine_mesh_interval.i >> 1, fine_mesh_interval.index >> 1);

            m_coarse_it.init(coarse_mesh_interval);

            m_cells[coarse] = m_coarse_it.cells()[0];
            m_cells[fine]   = m_fine_it->cells()[m_direction_index];

            m_move_coarse_cell = false;
        }

        inline auto& interval() const
        {
            return m_fine_it->interval();
        }

        inline const auto& cells() const
        {
            return m_cells;
        }

        inline void move_next()
        {
            // Move fine cell
            ++m_cells[fine].index;      // increment cell index
            ++m_cells[fine].indices[0]; // increment x-coordinate

            // Move coarse cell only once every two iterations
            m_cells[coarse].index += static_cast<cell_index_t>(m_move_coarse_cell);
            m_cells[coarse].indices[0] += static_cast<interval_value_t>(m_move_coarse_cell);
            m_move_coarse_cell = !m_move_coarse_cell;
        }
    };

    template <class Mesh, std::size_t stencil_size>
    auto make_stencil_iterator(const Mesh& mesh, const StencilAnalyzer<stencil_size, Mesh::dim>& stencil)
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
                                                     const StencilAnalyzer<stencil_size, Mesh::dim>& stencil,
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
    inline void for_each_stencil(const Mesh& mesh, const StencilAnalyzer<stencil_size, Mesh::dim>& stencil, Func&& f)
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
    inline void for_each_stencil(const Mesh& mesh, Set& set, const StencilAnalyzer<stencil_size, Mesh::dim>& stencil, Func&& f)
    {
        auto stencil_it = make_stencil_iterator(mesh, stencil);
        for_each_stencil(set, stencil_it, std::forward<Func>(f));
    }

    template <std::size_t index_coarse_cell, class Mesh, std::size_t stencil_size>
    auto make_leveljump_iterator(const IteratorStencil<Mesh, stencil_size>& fine_iterator, std::size_t direction_index)
    {
        return LevelJumpIterator<index_coarse_cell, Mesh, stencil_size>(fine_iterator, direction_index);
    }
}
