#pragma once
#include "boundary.hpp"
#include "stencil.hpp"

namespace samurai
{
    /**
     * Iterates over the interfaces of the mesh in the chosen direction.
     * @param direction: positive Cartesian direction defining, for each cell, which neighbour defines the desired interface.
     *                   In 2D: {1,0} to browse vertical interfaces, {0,1} to browse horizontal interfaces.
     *
     * The provided callback @param f has the following signature:
     *           void f(auto& interface_cells, auto& comput_cells)
     * where
     *       'interface_cells' is an array containing the two real cells on both sides of the interface,
     *       'comput_cells'    is an array containing the two cells that must be used for the computation.
     * If there is no level jump, then 'interface_cells' = 'comput_cells'.
     * In case of level jump l/l+1, the cells of 'interface_cells' are of different levels,
     * while both cells of 'comput_cells' are at level l+1 and one of them is a ghost.
     */
    template <class Mesh, class Vector, class Func>
    void for_each_interior_interface(const Mesh& mesh, Vector direction, Func&& f)
    {
        static constexpr std::size_t dim = Mesh::dim;

        Stencil<2, dim> comput_stencil = in_out_stencil<dim>(direction);
        for_each_interior_interface(mesh, direction, comput_stencil, std::forward<Func>(f));
    }

    /**
     * This function does the same as the preceding one, but allows to define the computational stencil.
     * @param comput_stencil defines the set of cells returned in the callback function.
     * If no stencil is defined (--> preceding function), then the default is {{0,0}, direction},
     * i.e. the current cell and its neighbour in the desired @param direction.
     *
     * The provided callback @param f has the following signature:
     *           void f(auto& interface_cells, auto& comput_cells)
     * where
     *       'interface_cells' is an array containing the two real cells on both sides of the interface (might be of different levels),
     *       'comput_cells'    is an array containing the set of cells/ghosts defined by @param comput_stencil (all of same level).
     */
    template <class Mesh, class Vector, std::size_t comput_stencil_size, class Func>
    void
    for_each_interior_interface(const Mesh& mesh, Vector direction, const Stencil<comput_stencil_size, Mesh::dim>& comput_stencil, Func&& f)
    {
        for_each_level(mesh,
                       [&](auto level)
                       {
                           for_each_interior_interface(mesh, level, direction, comput_stencil, std::forward<Func>(f));
                       });
    }

    /**
     * This function does the same as the preceding one, but on one level only.
     * @param level: the browsed interfaces will be defined by two cells of same level,
     *               or one cell of that level and another one level higher.
     *
     * The provided callback @param f has the following signature:
     *           void f(auto& interface_cells, auto& comput_cells)
     * where
     *       'interface_cells' is an array containing the two real cells on both sides of the interface (might be of different levels).
     *       'comput_cells'    is an array containing the set of cells/ghosts defined by @param comput_stencil (all of same level).
     */
    template <class Mesh, class Vector, std::size_t comput_stencil_size, class Func>
    void for_each_interior_interface(const Mesh& mesh,
                                     std::size_t level,
                                     Vector direction,
                                     const Stencil<comput_stencil_size, Mesh::dim>& comput_stencil,
                                     Func&& f)
    {
        for_each_interior_interface___same_level(mesh, level, direction, comput_stencil, std::forward<Func>(f));
        for_each_interior_interface___level_jump_direction(mesh, level, direction, comput_stencil, std::forward<Func>(f));
        for_each_interior_interface___level_jump_opposite_direction(mesh, level, direction, comput_stencil, std::forward<Func>(f));
    }

    /**
     * Iterates over the interfaces of same level only (no level jump).
     * Same parameters as the preceding function.
     */
    template <class Mesh, class Vector, std::size_t comput_stencil_size, class Func>
    void for_each_interior_interface___same_level(const Mesh& mesh,
                                                  std::size_t level,
                                                  Vector direction,
                                                  const Stencil<comput_stencil_size, Mesh::dim>& comput_stencil,
                                                  Func&& f)
    {
        static constexpr std::size_t dim = Mesh::dim;
        using mesh_id_t                  = typename Mesh::mesh_id_t;
        using mesh_interval_t            = typename Mesh::mesh_interval_t;

        Stencil<2, dim> interface_stencil = in_out_stencil<dim>(direction);
        auto interface_it                 = make_stencil_iterator(mesh, interface_stencil);
        auto comput_stencil_it            = make_stencil_iterator(mesh, comput_stencil);

        auto& cells        = mesh[mesh_id_t::cells][level];
        auto shifted_cells = translate(cells, -direction);
        auto intersect     = intersection(cells, shifted_cells);

        for_each_meshinterval<mesh_interval_t>(intersect,
                                               [&](auto mesh_interval)
                                               {
                                                   interface_it.init(mesh_interval);
                                                   comput_stencil_it.init(mesh_interval);
                                                   for (std::size_t ii = 0; ii < mesh_interval.i.size(); ++ii)
                                                   {
                                                       f(interface_it.cells(), comput_stencil_it.cells());
                                                       interface_it.move_next();
                                                       comput_stencil_it.move_next();
                                                   }
                                               });
    }

    /**
     * Iterates over the level jumps (level --> level+1) that occur in the chosen direction.
     *
     *         |__|   l+1
     *    |____|      l
     *    --------->
     *    direction
     *
     * The provided callback @param f has the following signature:
     *           void f(auto& interface_cells, auto& comput_cells)
     * where
     *       'interface_cells' = [cell_{l}, cell_{l+1}].
     */
    template <class Mesh, class Vector, std::size_t comput_stencil_size, class Func>
    void for_each_interior_interface___level_jump_direction(const Mesh& mesh,
                                                            std::size_t level,
                                                            Vector direction,
                                                            const Stencil<comput_stencil_size, Mesh::dim>& comput_stencil,
                                                            Func&& f)
    {
        static constexpr std::size_t dim = Mesh::dim;
        using mesh_id_t                  = typename Mesh::mesh_id_t;
        using mesh_interval_t            = typename Mesh::mesh_interval_t;
        using cell_t                     = Cell<dim, typename Mesh::interval_t>;

        if (level >= mesh.max_level())
        {
            return;
        }

        Stencil<1, dim> coarse_cell_stencil = center_only_stencil<dim>();
        auto coarse_it                      = make_stencil_iterator(mesh, coarse_cell_stencil);

        auto comput_stencil_it = make_stencil_iterator(mesh, comput_stencil);

        int direction_index_int     = find(comput_stencil, direction);
        std::size_t direction_index = static_cast<std::size_t>(direction_index_int);

        auto& coarse_cells = mesh[mesh_id_t::cells][level];
        auto& fine_cells   = mesh[mesh_id_t::cells][level + 1];

        auto shifted_fine_cells = translate(fine_cells, -direction);
        auto fine_intersect     = intersection(coarse_cells, shifted_fine_cells).on(level + 1);

        for_each_meshinterval<mesh_interval_t>(
            fine_intersect,
            [&](auto fine_mesh_interval)
            {
                mesh_interval_t coarse_mesh_interval(level, fine_mesh_interval.i >> 1, fine_mesh_interval.index >> 1);

                comput_stencil_it.init(fine_mesh_interval);
                coarse_it.init(coarse_mesh_interval);

                for (std::size_t ii = 0; ii < fine_mesh_interval.i.size(); ++ii)
                {
                    std::array<cell_t, 2> interface_cells;
                    interface_cells[0] = coarse_it.cells()[0];
                    interface_cells[1] = comput_stencil_it.cells()[direction_index];

                    f(interface_cells, comput_stencil_it.cells());
                    comput_stencil_it.move_next();

                    if (ii % 2 == 1)
                    {
                        coarse_it.move_next();
                    }
                }
            });
    }

    /**
     * Iterates over the level jumps (level --> level+1) that occur in the OPPOSITE direction of @param direction.
     *
     *    |__|        l+1
     *       |____|   l
     *    --------->
     *    direction
     *
     * The provided callback @param f has the following signature:
     *           void f(auto& interface_cells, auto& comput_cells)
     * where
     *       'interface_cells' = [cell_{l+1}, cell_{l}].
     */
    template <class Mesh, class Vector, std::size_t comput_stencil_size, class Func>
    void for_each_interior_interface___level_jump_opposite_direction(const Mesh& mesh,
                                                                     std::size_t level,
                                                                     Vector direction,
                                                                     const Stencil<comput_stencil_size, Mesh::dim>& comput_stencil,
                                                                     Func&& f)
    {
        static constexpr std::size_t dim = Mesh::dim;
        using mesh_id_t                  = typename Mesh::mesh_id_t;
        using mesh_interval_t            = typename Mesh::mesh_interval_t;
        using cell_t                     = Cell<dim, typename Mesh::interval_t>;

        if (level >= mesh.max_level())
        {
            return;
        }

        Stencil<1, dim> coarse_cell_stencil = center_only_stencil<dim>();
        auto coarse_it                      = make_stencil_iterator(mesh, coarse_cell_stencil);

        Stencil<comput_stencil_size, dim> minus_comput_stencil = -xt::flip(comput_stencil, 0);
        Vector minus_direction                                 = -direction;
        int minus_direction_index_int                          = find(minus_comput_stencil, minus_direction);
        std::size_t minus_direction_index                      = static_cast<std::size_t>(minus_direction_index_int);
        auto minus_comput_stencil_it                           = make_stencil_iterator(mesh, minus_comput_stencil);

        auto& coarse_cells = mesh[mesh_id_t::cells][level];
        auto& fine_cells   = mesh[mesh_id_t::cells][level + 1];

        auto shifted_fine_cells = translate(fine_cells, direction);
        auto fine_intersect     = intersection(coarse_cells, shifted_fine_cells).on(level + 1);

        for_each_meshinterval<mesh_interval_t>(
            fine_intersect,
            [&](auto fine_mesh_interval)
            {
                mesh_interval_t coarse_mesh_interval(level, fine_mesh_interval.i >> 1, fine_mesh_interval.index >> 1);

                minus_comput_stencil_it.init(fine_mesh_interval);
                coarse_it.init(coarse_mesh_interval);

                for (std::size_t ii = 0; ii < fine_mesh_interval.i.size(); ++ii)
                {
                    std::array<cell_t, 2> interface_cells;
                    interface_cells[0] = minus_comput_stencil_it.cells()[minus_direction_index];
                    interface_cells[1] = coarse_it.cells()[0];

                    f(interface_cells, minus_comput_stencil_it.cells());
                    minus_comput_stencil_it.move_next();

                    if (ii % 2 == 1)
                    {
                        coarse_it.move_next();
                    }
                }
            });
    }

    /**
     * Iterates over the boundary interface in @param direction and its opposite direction.
     *
     * The provided callback @param f has the following signature:
     *           void f(auto& cell, auto& comput_cells)
     * where
     *       'cell'         is the inner cell at the boundary.
     *       'comput cells' is the set of cells/ghosts defined by @param comput_stencil
     *                      (typically, the inner cell and the outside ghost).
     */
    template <class Mesh, class Vector, std::size_t comput_stencil_size, class Func>
    void
    for_each_boundary_interface(const Mesh& mesh, Vector direction, const Stencil<comput_stencil_size, Mesh::dim>& comput_stencil, Func&& f)
    {
        for_each_level(mesh,
                       [&](auto level)
                       {
                           for_each_boundary_interface(mesh, level, direction, comput_stencil, std::forward<Func>(f));
                       });
    }

    /**
     * Same as the preceding function, but for @param level only.
     */
    template <class Mesh, class Vector, std::size_t comput_stencil_size, class Func>
    void for_each_boundary_interface(const Mesh& mesh,
                                     std::size_t level,
                                     Vector direction,
                                     const Stencil<comput_stencil_size, Mesh::dim>& comput_stencil,
                                     Func&& f)
    {
        for_each_boundary_interface___direction(mesh, level, direction, comput_stencil, std::forward<Func>(f));
        for_each_boundary_interface___opposite_direction(mesh, level, direction, comput_stencil, std::forward<Func>(f));
    }

    template <class Mesh, class Vector, std::size_t comput_stencil_size, class Func>
    void for_each_boundary_interface___direction(const Mesh& mesh,
                                                 std::size_t level,
                                                 Vector direction,
                                                 const Stencil<comput_stencil_size, Mesh::dim>& comput_stencil,
                                                 Func&& f)
    {
        static constexpr std::size_t dim = Mesh::dim;
        using mesh_interval_t            = typename Mesh::mesh_interval_t;

        Stencil<2, dim> interface_stencil = in_out_stencil<dim>(direction);
        auto interface_it                 = make_stencil_iterator(mesh, interface_stencil);
        auto comput_stencil_it            = make_stencil_iterator(mesh, comput_stencil);

        auto bdry = boundary(mesh, level, direction);
        for_each_meshinterval<mesh_interval_t>(bdry,
                                               [&](auto mesh_interval)
                                               {
                                                   interface_it.init(mesh_interval);
                                                   comput_stencil_it.init(mesh_interval);
                                                   for (std::size_t ii = 0; ii < mesh_interval.i.size(); ++ii)
                                                   {
                                                       f(interface_it.cells()[0], comput_stencil_it.cells());
                                                       interface_it.move_next();
                                                       comput_stencil_it.move_next();
                                                   }
                                               });
    }

    template <class Mesh, class Vector, std::size_t comput_stencil_size, class Func>
    void for_each_boundary_interface___opposite_direction(const Mesh& mesh,
                                                          std::size_t level,
                                                          Vector direction,
                                                          const Stencil<comput_stencil_size, Mesh::dim>& comput_stencil,
                                                          Func&& f)
    {
        Vector opposite_direction                        = -direction;
        decltype(comput_stencil) opposite_comput_stencil = -xt::flip(comput_stencil, 0);
        for_each_boundary_interface___direction(mesh, level, opposite_direction, opposite_comput_stencil, std::forward<Func>(f));
    }

}
