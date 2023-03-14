#pragma once
#include "indices.hpp"

namespace samurai
{
    template<std::size_t stencil_size, std::size_t dim>
    using Stencil = xt::xtensor_fixed<int, xt::xshape<stencil_size, dim>>;

    template<std::size_t dim>
    using StencilVector = xt::xtensor_fixed<int, xt::xshape<dim>>;


    template<std::size_t stencil_size, std::size_t dim>
    int find_stencil_origin(const Stencil<stencil_size, dim>& stencil)
    {
        for (unsigned int id = 0; id<stencil_size; ++id)
        {
            auto d = xt::view(stencil, id);
            bool is_zero_vector = true;
            for (unsigned int i=0; i<dim; ++i)
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

    
    template<class Mesh, std::size_t stencil_size>
    class IteratorStencil
    {
        static constexpr std::size_t dim = Mesh::dim;
        using coord_index_t = typename Mesh::config::interval_t::coord_index_t;
        using Cell = typename samurai::Cell<coord_index_t, dim>;
    private:
        const Stencil<stencil_size, dim> m_stencil;
        std::array<Cell, stencil_size> m_cells;
        unsigned int m_origin_cell;

    public:
        IteratorStencil(const Stencil<stencil_size, dim>& stencil)
        : m_stencil(stencil)
        {
            int origin_index = find_stencil_origin(stencil);
            assert(origin_index >= 0 && "the zero vector is required in the stencil definition.");
            m_origin_cell = static_cast<unsigned int>(origin_index);
        }

        void init(const Mesh& mesh, const typename Mesh::mesh_interval_t& mesh_interval)
        {
            double length = cell_length(mesh_interval.level);
            for (Cell& cell : m_cells)
            {
                cell.level = mesh_interval.level;
                cell.length = length;
            }

            // origin of the stencil
            Cell& origin_cell = m_cells[m_origin_cell];
            origin_cell.indices[0] = mesh_interval.i.start; 
            for(unsigned int d = 0; d < dim - 1; ++d)
            {
                origin_cell.indices[d + 1] = mesh_interval.index[d];
            }
            origin_cell.index = get_index_start(mesh, mesh_interval);

            for (unsigned int id = 0; id<stencil_size; ++id)
            {
                if (id == m_origin_cell)
                    continue;

                auto d = xt::view(m_stencil, id);

                // Translate the coordinates according the direction d
                Cell& cell = m_cells[id];
                for (unsigned int k = 0; k < dim; ++k)
                {
                    cell.indices[k] = origin_cell.indices[k] + d[k];
                }

                // We are on the same row as the stencil origin if d = {d[0], 0, ..., 0}
                bool same_row = true;
                for (std::size_t k=1; k<dim; ++k)
                {
                    if (d[k] != 0)
                    {
                        same_row = false;
                        break;
                    }
                }
                if (same_row) // same row as the stencil origin
                {
                    cell.index = origin_cell.index + d[0]; // translation on the row
                }
                else
                {
                    cell.index = get_index_start_translated(mesh, mesh_interval, d);
                }
            }
        }

        void move_next()
        {
            for (Cell& cell : m_cells)
            {
                cell.index++;      // increment cell index
                cell.indices[0]++; // increment x-coordinate
            }
        }

        std::array<Cell, stencil_size>& cells()
        {
            return m_cells;
        }
    };

    template <class Mesh, std::size_t stencil_size, class Func>
    inline void for_each_stencil_sliding_in_interval(const Mesh& mesh, const typename Mesh::mesh_interval_t& mesh_interval, IteratorStencil<Mesh, stencil_size>& stencil_it, Func &&f)
    {
        stencil_it.init(mesh, mesh_interval);
        for(std::size_t ii=0; ii<mesh_interval.i.size(); ++ii)
        {
            f(stencil_it.cells());
            stencil_it.move_next();
        }
    }

    template <class Mesh, std::size_t stencil_size, class Func>
    inline void for_each_stencil_sliding_in_interval(const Mesh& mesh, const typename Mesh::mesh_interval_t& mesh_interval, const Stencil<stencil_size, Mesh::dim>& stencil, Func &&f)
    {
        IteratorStencil<Mesh, stencil_size> stencil_it(stencil);
        for_each_stencil_sliding_in_interval(mesh, mesh_interval, stencil_it, std::forward<Func>(f));
    }

    template <class Mesh, std::size_t stencil_size, class Func>
    inline void for_each_stencil(const Mesh& mesh, std::size_t level, IteratorStencil<Mesh, stencil_size>& stencil_it, Func &&f)
    {
        using mesh_id_t = typename Mesh::mesh_id_t;
        for_each_meshinterval(mesh[mesh_id_t::cells][level], [&](auto mesh_interval)
        {
            for_each_stencil_sliding_in_interval(mesh, mesh_interval, stencil_it, std::forward<Func>(f));
        });
    }

    template <class Mesh, class Set, std::size_t stencil_size, class Func>
    inline void for_each_stencil(const Mesh& mesh, Set& set, IteratorStencil<Mesh, stencil_size>& stencil_it, Func &&f)
    {
        using mesh_interval_t = typename Mesh::mesh_interval_t;
        for_each_meshinterval<mesh_interval_t>(set, [&](auto mesh_interval)
        {
            for_each_stencil_sliding_in_interval(mesh, mesh_interval, stencil_it, std::forward<Func>(f));
        });
    }

    template <class Mesh, std::size_t stencil_size, class GetCoeffsFunc, class Func>
    inline void for_each_stencil(const Mesh& mesh, const Stencil<stencil_size, Mesh::dim>& stencil, GetCoeffsFunc&& get_coefficients, Func &&f)
    {
        IteratorStencil<Mesh, stencil_size> stencil_it(stencil);

        for_each_level(mesh, [&](std::size_t level)
        {
            auto coeffs = get_coefficients(cell_length(level));

            for_each_stencil(mesh, level, stencil_it,
            [&] (auto& cells)
            {
                f(cells, coeffs);
            });
        });
    }

    template <class Mesh, class Set, std::size_t stencil_size, class Func>
    inline void for_each_stencil(const Mesh& mesh, Set& set, const Stencil<stencil_size, Mesh::dim>& stencil, Func &&f)
    {
        IteratorStencil<Mesh, stencil_size> stencil_it(stencil);
        for_each_stencil(mesh, set, stencil_it,
        [&] (auto& cells)
        {
            f(cells);
        });
    }


    //-----------------------//
    //    Useful stencils    //
    //-----------------------//


    template<std::size_t dim, std::size_t neighbourhood_width=1>
    constexpr Stencil<1+2*dim*neighbourhood_width, dim> star_stencil()
    {
        static_assert(dim >= 1 || dim <= 3, "Star stencil not implemented for this dimension");
        static_assert(neighbourhood_width >= 0 || neighbourhood_width <= 2, "Star stencil not implemented for this neighbourhood width");

        if constexpr (neighbourhood_width == 0)
        {
            Stencil<1, dim> s;
            s.fill(0);
            return s;
        }
        else if constexpr (neighbourhood_width == 1)
        {
            if constexpr (dim == 1)
            {
                //    left, center, right
                return {{-1}, {0}, {1}};
            }
            else if constexpr (dim == 2)
            {
                //       left,   center,  right,   bottom,  top 
                return {{-1, 0}, {0, 0},  {1, 0}, {0, -1}, {0, 1}};
            }
            else if constexpr (dim == 3)
            {
                //       left,   center,    right,   front,    back,    bottom,    top
                return {{-1,0,0}, {0,0,0},  {1,0,0}, {0,-1,0}, {0,1,0}, {0,0,-1}, {0,0,1}};
            }
        }
        else if constexpr (neighbourhood_width == 2)
        {
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
                //        left2,    left,    center,  right,   right2,  front2,    front,   back,    back2,   bottom2,  bottom,    top,     top2
                return {{-2,0,0}, {-1,0,0}, {0,0,0}, {1,0,0}, {2,0,0}, {0,-2,0}, {0,-1,0}, {0,1,0}, {0,2,0}, {0,0,-2}, {0,0,-1}, {0,0,1}, {0,0,2}};
            }
        }
        return Stencil<1+2*dim*neighbourhood_width, dim>();
    }

    template<std::size_t dim>
    constexpr Stencil<2*dim, dim> cartesian_directions()
    {
        static_assert(dim >= 1 || dim <= 3, "cartesian_directions() not implemented for this dimension");

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
            //       left,    right,   front,    back,    bottom,    top
            return {{-1,0,0}, {1,0,0}, {0,-1,0}, {0,1,0}, {0,0,-1}, {0,0,1}};
        }
        return Stencil<2*dim, dim>();
    }

    template<std::size_t dim>
    constexpr Stencil<1, dim> center_only_stencil()
    {
        return star_stencil<dim, 0>();
    }

    template<std::size_t dim, class Vector>
    Stencil<2, dim> in_out_stencil(const Vector& towards_out_from_in)
    {
        auto stencil_shape = Stencil<2, dim>();
        xt::view(stencil_shape, 0) = 0;
        xt::view(stencil_shape, 1) = towards_out_from_in;
        return stencil_shape;
    }
}