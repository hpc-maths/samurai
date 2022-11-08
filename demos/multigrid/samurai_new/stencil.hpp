#pragma once
#include <xtensor/xfixed.hpp>

namespace samurai_new
{
    /**
     * Stores the triplet (level, i, index)
    */
    template<class Mesh>
    struct MeshInterval
    {
        using interval_t = typename Mesh::interval_t;
        using coord_type = typename Mesh::lca_type::iterator::coord_type;

        std::size_t level;
        interval_t i;
        coord_type index;
        double cell_length;

        MeshInterval(std::size_t l) 
        : level(l) 
        {
            cell_length = 1./(1 << level);
        }

        MeshInterval(std::size_t l, const interval_t& _i, const coord_type& _index) 
        : level(l) , i(_i), index(_index)
        {
            cell_length = 1./(1 << level);
        }
    };


    template <class Mesh>
    inline auto get_index_start(const Mesh& mesh, const MeshInterval<Mesh>& mesh_interval)
    {
        static constexpr std::size_t dim = Mesh::dim;
        using coord_index_t = typename Mesh::coord_index_t;

        std::array<coord_index_t, dim> coord;
        std::copy(mesh_interval.index.cbegin(), mesh_interval.index.end(), coord.begin()+1);
        coord[0] = mesh_interval.i.start;
        return mesh.get_index(mesh_interval.level, coord);
    }

    template <class Mesh, class Vector>
    inline auto get_index_start_translated(const Mesh& mesh, const MeshInterval<Mesh>& mesh_interval, const Vector& translation_vect)
    {
        static constexpr std::size_t dim = Mesh::dim;
        using coord_index_t = typename Mesh::coord_index_t;

        std::array<coord_index_t, dim> coord;
        std::copy(mesh_interval.index.cbegin(), mesh_interval.index.end(), coord.begin()+1);
        coord[0] = mesh_interval.i.start;
        for (std::size_t d=0; d<dim; ++d)
        {
            coord[d] += translation_vect[d];
        }
        return mesh.get_index(mesh_interval.level, coord);
    }

    template <class Mesh, class Vector>
    inline auto get_index_start_children(const Mesh& mesh, MeshInterval<Mesh>& mesh_interval, const Vector& translation_vect)
    {
        static constexpr std::size_t dim = Mesh::dim;
        using coord_index_t = typename Mesh::coord_index_t;

        std::array<coord_index_t, dim> coord;
        std::copy(mesh_interval.index.cbegin(), mesh_interval.index.cend(), coord.begin()+1);
        coord[0] = mesh_interval.i.start;
        for (std::size_t d=0; d<dim; ++d)
        {
            coord[d] = 2*coord[d] + translation_vect[d];
        }

        return mesh.get_index(mesh_interval.level+1, coord);
    }



    // TODO inverser les dimensions
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

        template<class Mesh>
        void init(const Mesh& mesh, const MeshInterval<Mesh>& mesh_interval)
        {
            _cell_indices[_origin_cell] = static_cast<DesiredIndexType>(get_index_start(mesh, mesh_interval)); // origin of the stencil
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
                    _cell_indices[id] = static_cast<DesiredIndexType>(get_index_start_translated(mesh, mesh_interval, d));
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

        void init(const Mesh& mesh, const MeshInterval<Mesh>& mesh_interval)
        {
            double length = 1./(1 << mesh_interval.level);
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
                cell.index++;
                cell.indices[0]++;
            }
        }

        std::array<Cell, stencil_size>& cells()
        {
            return _cells;
        }
    };



    template <std::size_t dim>
    inline StencilShape<dim, 2*dim> cartesian_directions()
    {
        static_assert((dim >= 1 && dim <=3), "cartesian_directions() not implemented in this dimension");

        // !!! The order is important: the opposite of a vector must be located 'dim' indices after.
        if constexpr (dim == 1)
        {
            //                       left, right
            return StencilShape<1, 2>{{-1}, {1}};
        }
        else if constexpr (dim == 2)
        {
            //                        bottom,   right,  top,    left
            return StencilShape<2, 4>{{0, -1}, {1, 0}, {0, 1}, {-1, 0}};
        }
        else if constexpr (dim == 3)
        {
            //                         bottom,   front,   right,    top,     back,     left
            return StencilShape<3, 6>{{0,0,-1}, {0,1,0}, {1,0,0}, {0,0,1}, {0,-1,0}, {-1,0,0}};
        }
        return StencilShape<dim, 2*dim>();
    }


    template<std::size_t dim, class Vector>
    samurai_new::StencilShape<dim, 2> out_in_stencil(const Vector& out_normal_vect)
    {
        auto stencil_shape = samurai_new::StencilShape<dim, 2>();
        xt::view(stencil_shape, 0) = 0;
        xt::view(stencil_shape, 1) = -out_normal_vect;
        return stencil_shape;
    }
}