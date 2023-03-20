#pragma once
#include "stencil.hpp"

namespace samurai
{
    template <class Mesh, class Vector, class Func>
    inline void for_each_interface(const Mesh& mesh, Vector direction, Func&& f)
    {
        static constexpr std::size_t dim = Mesh::dim;

        Stencil<2, dim> comput_stencil = in_out_stencil<dim>(direction);
        for_each_interface(mesh, direction, comput_stencil, std::forward<Func>(f));
    }

    template <class Mesh, class Vector, std::size_t comput_stencil_size, class Func>
    inline void for_each_interface(const Mesh& mesh, Vector direction, const Stencil<comput_stencil_size, Mesh::dim>& comput_stencil, Func&& f)
    {
        static constexpr std::size_t dim = Mesh::dim;
        using mesh_id_t = typename Mesh::mesh_id_t;
        using mesh_interval_t = typename Mesh::mesh_interval_t;
        using coord_index_t = typename Mesh::config::interval_t::coord_index_t;
        using Cell = typename samurai::Cell<coord_index_t, dim>;

        int direction_index_int = find(comput_stencil, direction);
        std::size_t direction_index = static_cast<std::size_t>(direction_index_int);

        Stencil<2, dim> interface_stencil = in_out_stencil<dim>(direction);
        auto interface_it = make_stencil_iterator(mesh, interface_stencil);
        auto comput_stencil_it = make_stencil_iterator(mesh, comput_stencil);

        // Same level
        for_each_level(mesh, [&](auto level)
        {
            auto& cells = mesh[mesh_id_t::cells][level];
            auto shifted_cells = translate(cells, -direction);
            auto intersect = intersection(cells, shifted_cells);
            //for_each_interface(intersect, interface_it, comput_stencil_it, std::forward<Func>(f));
            for_each_meshinterval<mesh_interval_t>(intersect, [&](auto mesh_interval)
            {
                interface_it.init(mesh_interval);
                comput_stencil_it.init(mesh_interval);
                for (std::size_t ii=0; ii<mesh_interval.i.size(); ++ii)
                {
                    f(interface_it.cells(), comput_stencil_it.cells());
                    interface_it.move_next();
                    comput_stencil_it.move_next();
                }
            });
        });

        // Level jumps
        Stencil<1, dim> coarse_cell_stencil = center_only_stencil<dim>();
        auto coarse_it = make_stencil_iterator(mesh, coarse_cell_stencil);
        for_each_level(mesh, [&](auto level)
        {
            if (level < mesh.max_level())
            {
                auto& coarse_cells = mesh[mesh_id_t::cells][level];
                auto& fine_cells = mesh[mesh_id_t::cells][level+1];
                auto shifted_fine_cells = translate(fine_cells, -direction);
                auto fine_intersect = intersection(coarse_cells, shifted_fine_cells).on(level+1);
                for_each_meshinterval<mesh_interval_t>(fine_intersect, [&](auto fine_mesh_interval)
                {
                    mesh_interval_t coarse_mesh_interval(level, fine_mesh_interval.i / 2, fine_mesh_interval.index / 2);
                    comput_stencil_it.init(fine_mesh_interval);
                    coarse_it.init(coarse_mesh_interval);
                    for (std::size_t ii=0; ii<fine_mesh_interval.i.size(); ++ii)
                    {
                        std::array<Cell, 2> interface_cells;
                        interface_cells[0] = coarse_it.cells()[0];

                        interface_cells[1] = comput_stencil_it.cells()[direction_index];
                        f(interface_cells, comput_stencil_it.cells());
                        comput_stencil_it.move_next();

                        interface_cells[1] = comput_stencil_it.cells()[direction_index];
                        f(interface_cells, comput_stencil_it.cells());
                        comput_stencil_it.move_next();
                    }
                });
            }
        });
    }

    /*template <class SetType, class InterfaceIterator, class ComputStencilIterator, class Func>
    inline void for_each_interface(SetType& set, InterfaceIterator& interface_it, ComputStencilIterator& comput_stencil_it, Func&& f)
    {
        using mesh_interval_t = typename InterfaceIterator::interval_t;
        for_each_meshinterval<mesh_interval_t>(set, [&](auto mesh_interval)
        {
            interface_it.init(mesh_interval);
            comput_stencil_it.init(mesh_interval);
            for (std::size_t ii=0; ii<mesh_interval.i.size(); ++ii)
            {
                f(interface_it.cells(), comput_stencil_it.cells());
                interface_it.move_next();
                comput_stencil_it.move_next();
            }
        });
    }*/
}