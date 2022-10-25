
#pragma once
#include <xtensor/xfixed.hpp>

namespace samurai_new
{
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
            auto j = index[0];
            _cell_indices[_origin_cell] = static_cast<DesiredIndexType>(mesh.get_index(level, i.start, j)); // origin of the stencil
            for (unsigned int id = 0; id<stencil_size; ++id)
            {
                if (id == _origin_cell)
                    continue;

                auto d = xt::view(_stencil, id);
                if (d[1] == 0) // same row as the stencil origin
                {
                    _cell_indices[id] = _cell_indices[_origin_cell] + d[0];
                }
                else
                {
                    _cell_indices[id] = static_cast<DesiredIndexType>(mesh.get_index(level, i.start + d[0], j + d[1]));
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
}