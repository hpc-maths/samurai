#pragma once
#include <samurai/algorithm.hpp>

namespace samurai_new
{
    template <class Mesh>
    Mesh coarsen(const Mesh& mesh)
    {
        using mesh_id_t = typename Mesh::mesh_id_t;

        typename Mesh::cl_type coarse_cell_list;
        if (Mesh::dim == 1)
        {
            samurai::for_each_interval(mesh[mesh_id_t::cells],
                                       [&](size_t level, const auto& i, const auto&)
                                       {
                                           coarse_cell_list[level - 1][{}].add_interval(i >> 1);
                                       });
        }
        else if (Mesh::dim == 2)
        {
            samurai::for_each_interval(mesh[mesh_id_t::cells],
                                       [&](size_t level, const auto& i, const auto& index)
                                       {
                                           auto j = index[0];
                                           if (j % 2 == 0)
                                           {
                                               coarse_cell_list[level - 1][{j / 2}].add_interval(i / 2);
                                           }
                                       });
        }
        return Mesh(coarse_cell_list, mesh.min_level() - 1, mesh.max_level() - 1);
    }

    /*template <class Mesh>
    Mesh refine(const Mesh& mesh)
    {
        assert(false && "refine() not implemented");
    }*/

} // namespace samurai_new
