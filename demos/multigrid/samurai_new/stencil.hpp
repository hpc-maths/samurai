#pragma once
#include <xtensor/xfixed.hpp>

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

    template <class Mesh, typename TIndex, class Vector>
    inline auto get_index_start_translated(const Mesh& mesh, std::size_t level, const typename Mesh::interval_t& i, const TIndex& index, const Vector translation_vect)
    {
        static constexpr std::size_t dim = Mesh::dim;
        using coord_index_t = typename Mesh::coord_index_t;

        std::array<coord_index_t, dim> coord;
        std::copy(index.cbegin(), index.end(), coord.begin()+1);
        coord[0] = i.start;
        for (std::size_t d=0; d<dim; ++d)
        {
            coord[d] += translation_vect[d];
        }
        return mesh.get_index(level, coord);
    }

    template <class Mesh, typename TIndex, class Vector>
    inline auto get_index_start_children(const Mesh& mesh, std::size_t level, const typename Mesh::interval_t& i, const TIndex& index, const Vector translation_vect)
    {
        static constexpr std::size_t dim = Mesh::dim;
        using coord_index_t = typename Mesh::coord_index_t;

        std::array<coord_index_t, dim> coord;
        std::copy(index.cbegin(), index.end(), coord.begin()+1);
        coord[0] = i.start;
        for (std::size_t d=0; d<dim; ++d)
        {
            coord[d] = 2*coord[d] + translation_vect[d];
        }

        return mesh.get_index(level+1, coord);
    }




    template<std::size_t dim, std::size_t stencil_size>
    using StencilShape = xt::xtensor_fixed<int, xt::xshape<stencil_size, dim>>;




    template<typename DesiredIndexType, std::size_t dim, std::size_t stencil_size>
    class StencilIndices
    {
    private:
        const StencilShape<dim, stencil_size> _stencil;
        std::array<DesiredIndexType, stencil_size> _cell_indices;
        std::array<int, stencil_size>  _origin_in_row;
        unsigned int _origin_cell;

    public:
        StencilIndices(const StencilShape<dim, stencil_size>& stencil)
        : _stencil(stencil)
        {
            //using coord_index_t = typename Mesh::config::interval_t::coord_index_t;
            bool origin_found = false;
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
                    origin_found = true;
                    _origin_cell = id;
                    break;
                }
            }
            assert(origin_found && "the zero vector is required in the stencil definition.");
        }

        template<class Mesh, typename Coords>
        void init(const Mesh& mesh, std::size_t level, const typename Mesh::config::interval_t& i, const Coords& index)
        {
            _cell_indices[_origin_cell] = static_cast<DesiredIndexType>(get_index_start(mesh, level, i, index)); // origin of the stencil
            for (unsigned int id = 0; id<stencil_size; ++id)
            {
                if (id == _origin_cell)
                    continue;

                auto d = xt::view(_stencil, id);

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
                    _cell_indices[id] = _cell_indices[_origin_cell] + d[0]; // translation on the row
                }
                else
                {
                    _cell_indices[id] = static_cast<DesiredIndexType>(get_index_start_translated(mesh, level, i, index, d));
                }
            }
        }

        void move_next()
        {
            for (unsigned int cell = 0; cell < stencil_size; ++cell)
            {
                _cell_indices[cell]++;
            }
        }

        std::array<DesiredIndexType, stencil_size>& indices()
        {
            return _cell_indices;
        }
    };



    template<class Mesh, std::size_t stencil_size>
    class StencilCells
    {
        static constexpr std::size_t dim = Mesh::dim;
        using coord_index_t = typename Mesh::config::interval_t::coord_index_t;
        using Cell = typename samurai::Cell<coord_index_t, dim>;
    private:
        const StencilShape<dim, stencil_size> _stencil;
        std::array<Cell, stencil_size> _cells;
        std::array<int, stencil_size>  _origin_in_row;
        unsigned int _origin_cell;

    public:
        StencilCells(const StencilShape<dim, stencil_size>& stencil)
        : _stencil(stencil)
        {
            //using coord_index_t = typename Mesh::config::interval_t::coord_index_t;
            bool origin_found = false;
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
                    origin_found = true;
                    _origin_cell = id;
                    break;
                }
            }
            assert(origin_found && "the zero vector is required in the stencil definition.");
        }

        template<typename Coords>
        void init(const Mesh& mesh, std::size_t level, const typename Mesh::config::interval_t& i, const Coords& index)
        {
            double length = 1./(1 << level);
            for (Cell& cell : _cells)
            {
                cell.level = level;
                cell.length = length;
            }

            // origin of the stencil
            Cell& origin_cell = _cells[_origin_cell];
            origin_cell.indices[0] = i.start; 
            for(unsigned int d = 0; d < dim - 1; ++d)
            {
                origin_cell.indices[d + 1] = index[d];
            }
            origin_cell.index = get_index_start(mesh, level, i, index);

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
                    cell.index = get_index_start_translated(mesh, level, i, index, d);
                }
            }
        }

        void move_next()
        {
            for (Cell& cell : _cells)
            {
                cell.index++;
                cell.indices[0]++;
            }
        }

        std::array<Cell, stencil_size>& cells()
        {
            return _cells;
        }
    };
}