#pragma once
#include "indices.hpp"

namespace samurai
{
    template<std::size_t stencil_size, std::size_t dim>
    using Stencil = xt::xtensor_fixed<int, xt::xshape<stencil_size, dim>>;


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
        const Stencil<stencil_size, dim> _stencil;
        std::array<Cell, stencil_size> _cells;
        unsigned int _origin_cell;

    public:
        IteratorStencil(const Stencil<stencil_size, dim>& stencil)
        : _stencil(stencil)
        {
            int origin_index = find_stencil_origin(stencil);
            assert(origin_index >= 0 && "the zero vector is required in the stencil definition.");
            _origin_cell = static_cast<unsigned int>(origin_index);
        }

        void init(const Mesh& mesh, const typename Mesh::mesh_interval_t& mesh_interval)
        {
            double length = cell_length(mesh_interval.level);
            for (Cell& cell : _cells)
            {
                cell.level = mesh_interval.level;
                cell.length = length;
            }

            // origin of the stencil
            Cell& origin_cell = _cells[_origin_cell];
            origin_cell.indices[0] = mesh_interval.i.start; 
            for(unsigned int d = 0; d < dim - 1; ++d)
            {
                origin_cell.indices[d + 1] = mesh_interval.index[d];
            }
            origin_cell.index = get_index_start(mesh, mesh_interval);

            for (unsigned int id = 0; id<stencil_size; ++id)
            {
                if (id == _origin_cell)
                    continue;

                auto d = xt::view(_stencil, id);

                // Translate the coordinates according the direction d
                Cell& cell = _cells[id];
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
            for (Cell& cell : _cells)
            {
                cell.index++;      // increment cell index
                cell.indices[0]++; // increment x-coordinate
            }
        }

        std::array<Cell, stencil_size>& cells()
        {
            return _cells;
        }
    };

    template <class Mesh, std::size_t stencil_size, class Func>
    inline void for_each_stencil(const Mesh& mesh, const typename Mesh::mesh_interval_t& mesh_interval, IteratorStencil<Mesh, stencil_size>& stencil_it, Func &&f)
    {
        stencil_it.init(mesh, mesh_interval);
        for(std::size_t ii=0; ii<mesh_interval.i.size(); ++ii)
        {
            f(stencil_it.cells());
            stencil_it.move_next();
        }
    }

    template <class Mesh, std::size_t stencil_size, class Func>
    inline void for_each_stencil(const Mesh& mesh, const typename Mesh::mesh_interval_t& mesh_interval, const Stencil<stencil_size, Mesh::dim>& stencil, Func &&f)
    {
        IteratorStencil<Mesh, stencil_size> stencil_it(stencil);
        for_each_stencil(mesh, mesh_interval, stencil_it, std::forward<Func>(f));
    }

    template <class Mesh, std::size_t stencil_size, class Func>
    inline void for_each_stencil(const Mesh& mesh, std::size_t level, IteratorStencil<Mesh, stencil_size>& stencil_it, Func &&f)
    {
        using mesh_id_t = typename Mesh::mesh_id_t;
        for_each_meshinterval(mesh[mesh_id_t::cells][level], [&](auto mesh_interval)
        {
            for_each_stencil(mesh, mesh_interval, stencil_it, std::forward<Func>(f));
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


    //-----------------------//
    //    Useful stencils    //
    //-----------------------//


    template<std::size_t dim>
    constexpr Stencil<dim, 1+2*dim> star_stencil()
    {
        static_assert(dim >= 1 || dim <= 3, "Star stencil not implemented for this dimension");

        if constexpr (dim == 1)
        {
            // 3-point stencil:
            //    left, center, right
            return {{-1}, {0}, {1}};
        }
        else if constexpr (dim == 2)
        {
            // 5-point stencil:
            //       left,   center,  right,   bottom,  top 
            return {{-1, 0}, {0, 0},  {1, 0}, {0, -1}, {0, 1}};
        }
        else if constexpr (dim == 3)
        {
            // 7-point stencil:
            //       left,   center,    right,   front,    back,    bottom,    top
            return {{-1,0,0}, {0,0,0},  {1,0,0}, {0,-1,0}, {0,1,0}, {0,0,-1}, {0,0,1}};
        }
        return Stencil<dim, 1+2*dim>();
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