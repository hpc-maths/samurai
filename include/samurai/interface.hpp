#pragma once
#include "boundary.hpp"
#include "stencil.hpp"

namespace samurai
{
    template <class Mesh, class Vector, class Func>
    void for_each_interior_interface(const Mesh& mesh, Vector direction, Func&& f)
    {
        static constexpr std::size_t dim = Mesh::dim;

        Stencil<2, dim> comput_stencil = in_out_stencil<dim>(direction);
        for_each_interior_interface(mesh, direction, comput_stencil, std::forward<Func>(f));
    }

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

        // Same level
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

    // Jumps level --> level+1
    //
    //         |__|   l+1
    //    |____|      l
    //    --------->
    //    direction
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
        using coord_index_t              = typename Mesh::config::interval_t::coord_index_t;
        using cell_t                     = typename samurai::Cell<coord_index_t, dim>;

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

    // Jumps level+1 --> level
    //
    //    |__|        l+1
    //       |____|   l
    //    --------->
    //    direction
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
        using coord_index_t              = typename Mesh::config::interval_t::coord_index_t;
        using cell_t                     = typename samurai::Cell<coord_index_t, dim>;

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

    template <class Mesh, class Vector, std::size_t comput_stencil_size, class GetFluxCoeffsFunc, class GetCellCoeffsFunc, class Func>
    void for_each_interior_interface(const Mesh& mesh,
                                     Vector direction,
                                     const Stencil<comput_stencil_size, Mesh::dim>& comput_stencil,
                                     GetFluxCoeffsFunc get_flux_coeffs,
                                     GetCellCoeffsFunc get_coeffs,
                                     GetCellCoeffsFunc get_coeffs_opposite_direction,
                                     Func&& f)
    {
        static constexpr std::size_t dim = Mesh::dim;

        Stencil<2, dim> interface_stencil = in_out_stencil<dim>(direction);
        auto interface_it                 = make_stencil_iterator(mesh, interface_stencil);
        auto comput_stencil_it            = make_stencil_iterator(mesh, comput_stencil);

        // Same level
        for_each_level(mesh,
                       [&](auto level)
                       {
                           auto h                                  = cell_length(level);
                           auto flux_coeffs                        = get_flux_coeffs(h);
                           decltype(flux_coeffs) minus_flux_coeffs = -flux_coeffs;

                           auto left_cell_coeffs  = get_coeffs(flux_coeffs, h, h);
                           auto right_cell_coeffs = get_coeffs_opposite_direction(minus_flux_coeffs, h, h);

                           for_each_interior_interface___same_level(mesh,
                                                                    level,
                                                                    direction,
                                                                    comput_stencil,
                                                                    [&](auto& interface_cells, auto& comput_cells)
                                                                    {
                                                                        f(interface_cells, comput_cells, left_cell_coeffs, right_cell_coeffs);
                                                                    });
                       });

        // Level jumps
        for_each_level(mesh,
                       [&](auto level)
                       {
                           if (level < mesh.max_level())
                           {
                               auto h_l                                = cell_length(level);
                               auto h_lp1                              = cell_length(level + 1);
                               auto flux_coeffs                        = get_flux_coeffs(h_lp1); // flux computed at level l+1
                               decltype(flux_coeffs) minus_flux_coeffs = -flux_coeffs;

                               // Jumps level --> level+1
                               //
                               //         |__|   l+1
                               //    |____|      l
                               //    --------->
                               //    direction
                               {
                                   auto left_cell_coeffs  = get_coeffs(flux_coeffs, h_lp1, h_l);
                                   auto right_cell_coeffs = get_coeffs_opposite_direction(minus_flux_coeffs, h_lp1, h_lp1);

                                   for_each_interior_interface___level_jump_direction(
                                       mesh,
                                       level,
                                       direction,
                                       comput_stencil,
                                       [&](auto& interface_cells, auto& comput_cells)
                                       {
                                           f(interface_cells, comput_cells, left_cell_coeffs, right_cell_coeffs);
                                       });
                               }
                               // Jumps level+1 --> level
                               //
                               //    |__|        l+1
                               //       |____|   l
                               //    --------->
                               //    direction
                               {
                                   auto left_cell_coeffs  = get_coeffs(flux_coeffs, h_lp1, h_lp1);
                                   auto right_cell_coeffs = get_coeffs_opposite_direction(minus_flux_coeffs, h_lp1, h_l);

                                   for_each_interior_interface___level_jump_opposite_direction(
                                       mesh,
                                       level,
                                       direction,
                                       comput_stencil,
                                       [&](auto& interface_cells, auto& comput_cells)
                                       {
                                           f(interface_cells, comput_cells, left_cell_coeffs, right_cell_coeffs);
                                       });
                               }
                           }
                       });
    }

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

    template <class Mesh, class Vector, std::size_t comput_stencil_size, class GetFluxCoeffsFunc, class GetCellCoeffsFunc, class Func>
    void for_each_boundary_interface(const Mesh& mesh,
                                     Vector direction,
                                     const Stencil<comput_stencil_size, Mesh::dim>& comput_stencil,
                                     GetFluxCoeffsFunc get_flux_coeffs,
                                     GetCellCoeffsFunc get_cell_coeffs,
                                     GetCellCoeffsFunc get_coeffs_opposite_direction,
                                     Func&& f)
    {
        for_each_level(mesh,
                       [&](auto level)
                       {
                           auto h = cell_length(level);

                           // Boundary in direction
                           auto flux_coeffs = get_flux_coeffs(h);
                           auto cell_coeffs = get_cell_coeffs(flux_coeffs, h, h);
                           for_each_boundary_interface___direction(mesh,
                                                                   level,
                                                                   direction,
                                                                   comput_stencil,
                                                                   [&](auto& cell, auto& comput_cells)
                                                                   {
                                                                       f(cell, comput_cells, cell_coeffs);
                                                                   });

                           // Boundary in opposite direction
                           decltype(flux_coeffs) minus_flux_coeffs = -flux_coeffs;
                           cell_coeffs                             = get_coeffs_opposite_direction(minus_flux_coeffs, h, h);
                           for_each_boundary_interface___opposite_direction(mesh,
                                                                            level,
                                                                            direction,
                                                                            comput_stencil,
                                                                            [&](auto& cell, auto& comput_cells)
                                                                            {
                                                                                f(cell, comput_cells, cell_coeffs);
                                                                            });
                       });
    }

}
