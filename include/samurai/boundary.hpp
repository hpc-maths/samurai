#pragma once
#include "stencil.hpp"
#include <utility>

namespace samurai
{
    // OBSOLETE
    template <class Mesh, class Vector>
    auto boundary(const Mesh& mesh, std::size_t level, const Vector& direction)
    {
        using mesh_id_t = typename Mesh::mesh_id_t;

        auto& cells  = mesh[mesh_id_t::cells][level];
        auto& domain = mesh.domain();

        auto max_level    = domain.level(); // domain.level();//mesh[mesh_id_t::cells].max_level();
        auto one_interval = 1 << (max_level - level);

        return difference(cells, translate(domain, -one_interval * direction)).on(level);
    }

    // OBSOLETE
    template <class Mesh, class Vector>
    auto boundary_mr(const Mesh& mesh, std::size_t level, const Vector& direction)
    {
        using mesh_id_t = typename Mesh::mesh_id_t;
        auto& domain    = mesh.domain();

        auto max_level    = domain.level();
        auto one_interval = 1 << (max_level - level);

        auto boundary_cells = difference(domain, translate(domain, -one_interval * direction));

        auto& all_on_level = mesh[mesh_id_t::reference][level];

        return intersection(boundary_cells, all_on_level).on(level);
    }

    // OBSOLETE
    template <class Mesh, class Vector, class Func>
    auto for_each_meshinterval_on_boundary(const Mesh& mesh, const Vector& direction, Func&& func)
    {
        using mesh_interval_t = typename Mesh::mesh_interval_t;

        for_each_level(mesh,
                       [&](std::size_t level)
                       {
                           auto bdry = boundary(mesh, level, direction);
                           for_each_meshinterval<mesh_interval_t>(bdry, std::forward<Func>(func));
                       });
    }

    // OBSOLETE
    template <class Mesh, std::size_t stencil_size, class Func>
    inline void for_each_meshinterval_on_boundary(const Mesh& mesh,
                                                  std::size_t level,
                                                  const Stencil<stencil_size, Mesh::dim>& boundary_directions,
                                                  Func&& func)
    {
        using mesh_interval_t = typename Mesh::mesh_interval_t;

        for (unsigned int is = 0; is < stencil_size; ++is)
        {
            auto direction = xt::view(boundary_directions, is);
            if (xt::any(direction)) // if (direction != 0)
            {
                auto bdry = boundary(mesh, level, direction);
                for_each_meshinterval<mesh_interval_t>(bdry,
                                                       [&](auto& mesh_interval)
                                                       {
                                                           func(mesh_interval, direction);
                                                       });
            }
        }
    }

    // OBSOLETE
    template <class Mesh, std::size_t stencil_size, class Func>
    inline void for_each_meshinterval_on_boundary_mr(const Mesh& mesh,
                                                     std::size_t level,
                                                     const Stencil<stencil_size, Mesh::dim>& boundary_directions,
                                                     Func&& func)
    {
        using mesh_interval_t = typename Mesh::mesh_interval_t;

        for (unsigned int is = 0; is < stencil_size; ++is)
        {
            auto direction = xt::view(boundary_directions, is);
            if (xt::any(direction)) // if (direction != 0)
            {
                auto bdry = boundary_mr(mesh, level, direction);
                for_each_meshinterval<mesh_interval_t>(bdry,
                                                       [&](auto& mesh_interval)
                                                       {
                                                           func(mesh_interval, direction);
                                                       });
            }
        }
    }

    // OBSOLETE
    template <class Mesh, class local_matrix_t, std::size_t stencil_size, class Func>
    inline void for_each_meshinterval_on_boundary(const Mesh& mesh,
                                                  std::size_t level,
                                                  const Stencil<stencil_size, Mesh::dim>& boundary_directions,
                                                  std::array<local_matrix_t, stencil_size>& coefficients,
                                                  Func&& func)
    {
        using mesh_interval_t = typename Mesh::mesh_interval_t;

        for (unsigned int is = 0; is < stencil_size; ++is)
        {
            auto direction = xt::view(boundary_directions, is);
            if (xt::any(direction)) // if (direction != 0)
            {
                auto coeff = coefficients[is];
                auto bdry  = boundary(mesh, level, direction);
                for_each_meshinterval<mesh_interval_t>(bdry,
                                                       [&](auto& mesh_interval)
                                                       {
                                                           func(mesh_interval, direction, coeff);
                                                       });
            }
        }
    }

    // OBSOLETE
    template <class Mesh, std::size_t stencil_size, class GetCoeffsFunc, class Func>
    void for_each_meshinterval_on_boundary(const Mesh& mesh,
                                           const Stencil<stencil_size, Mesh::dim>& boundary_directions,
                                           GetCoeffsFunc&& get_coefficients,
                                           Func&& func)
    {
        using mesh_id_t = typename Mesh::mesh_id_t;

        auto& cells = mesh[mesh_id_t::cells];
        for (std::size_t level = cells.min_level(); level <= cells.max_level(); ++level)
        {
            if (!cells[level].empty())
            {
                auto coeffs = get_coefficients(cell_length(level));
                for_each_meshinterval_on_boundary(mesh, level, boundary_directions, coeffs, std::forward<Func>(func));
            }
        }
    }

    template <class Mesh, std::size_t stencil_size, class Func>
    void for_each_meshinterval_on_boundary(const Mesh& mesh, const Stencil<stencil_size, Mesh::dim>& boundary_directions, Func&& func)
    {
        using mesh_id_t = typename Mesh::mesh_id_t;

        auto& cells = mesh[mesh_id_t::cells];
        for (std::size_t level = cells.min_level(); level <= cells.max_level(); ++level)
        {
            if (!cells[level].empty())
            {
                for_each_meshinterval_on_boundary(mesh, level, boundary_directions, std::forward<Func>(func));
            }
        }
    }

    template <class Mesh, std::size_t stencil_size, class GetCoeffsFunc, class Func>
    void for_each_cell_on_boundary(const Mesh& mesh,
                                   const Stencil<stencil_size, Mesh::dim>& boundary_directions,
                                   GetCoeffsFunc&& get_coefficients,
                                   Func&& func)
    {
        for_each_meshinterval_on_boundary(mesh,
                                          boundary_directions,
                                          get_coefficients,
                                          [&](auto& mesh_interval, auto& stencil_vector, double out_coeff)
                                          {
                                              for_each_cell(mesh,
                                                            mesh_interval.level,
                                                            mesh_interval.i,
                                                            mesh_interval.index,
                                                            [&](auto& cell)
                                                            {
                                                                func(cell, stencil_vector, out_coeff);
                                                            });
                                          });
    }

    template <class Mesh, std::size_t stencil_size, class GetCoeffsFunc, class Func>
    void for_each_stencil_center_and_outside_ghost(const Mesh& mesh,
                                                   const Stencil<stencil_size, Mesh::dim>& stencil,
                                                   GetCoeffsFunc&& get_coefficients,
                                                   Func&& func)
    {
        for_each_level(mesh,
                       [&](std::size_t level)
                       {
                           auto coeffs = get_coefficients(cell_length(level));

                           for_each_meshinterval_on_boundary(mesh,
                                                             level,
                                                             stencil,
                                                             coeffs,
                                                             [&](const auto& mesh_interval, const auto& towards_bdry_ghost, auto& out_coeff)
                                                             {
                                                                 for_each_stencil_sliding_in_interval(
                                                                     mesh,
                                                                     mesh_interval,
                                                                     in_out_stencil<Mesh::dim>(towards_bdry_ghost),
                                                                     [&](auto& cells)
                                                                     {
                                                                         func(cells, towards_bdry_ghost, out_coeff);
                                                                     });
                                                             });
                       });
    }

    template <class Mesh, std::size_t stencil_size, class Func>
    void for_each_stencil_center_and_outside_ghost(const Mesh& mesh, const Stencil<stencil_size, Mesh::dim>& stencil, Func&& func)
    {
        for_each_level(mesh,
                       [&](std::size_t level)
                       {
                           for_each_meshinterval_on_boundary(mesh,
                                                             level,
                                                             stencil,
                                                             [&](const auto& mesh_interval, const auto& towards_bdry_ghost)
                                                             {
                                                                 for_each_stencil_sliding_in_interval(
                                                                     mesh,
                                                                     mesh_interval,
                                                                     in_out_stencil<Mesh::dim>(towards_bdry_ghost),
                                                                     [&](auto& cells)
                                                                     {
                                                                         func(cells, towards_bdry_ghost);
                                                                     });
                                                             });
                       });
    }

    // OBSOLETE
    template <class Mesh, std::size_t stencil_size, class Func>
    void for_each_stencil_on_boundary(const Mesh& mesh,
                                      const DirectionVector<Mesh::dim>& boundary_direction,
                                      const Stencil<stencil_size, Mesh::dim>& stencil,
                                      Func&& func)
    {
        for_each_level(mesh,
                       [&](std::size_t level)
                       {
                           auto bdry = boundary(mesh, level, boundary_direction);
                           for_each_stencil(mesh, bdry, stencil, std::forward<Func>(func));
                       });
    }

    // NEW
    template <class Mesh, class Subset, std::size_t stencil_size, class Func>
    void
    for_each_stencil_on_boundary(const Mesh& mesh, const Subset& boundary_region, const Stencil<stencil_size, Mesh::dim>& stencil, Func&& func)
    {
        using mesh_id_t = typename Mesh::mesh_id_t;

        for_each_level(mesh,
                       [&](std::size_t level)
                       {
                           auto bdry = intersection(mesh[mesh_id_t::cells][level], boundary_region).on(level);
                           for_each_stencil(mesh, bdry, stencil, std::forward<Func>(func));
                       });
    }

    // NEW
    template <class Mesh, class Subset, std::size_t stencil_size, class GetCoeffsFunc, class Func>
    void for_each_stencil_on_boundary(const Mesh& mesh,
                                      const Subset& boundary_region,
                                      const Stencil<stencil_size, Mesh::dim>& stencil,
                                      GetCoeffsFunc&& get_coefficients,
                                      Func&& func)
    {
        using mesh_id_t = typename Mesh::mesh_id_t;

        for_each_level(mesh,
                       [&](std::size_t level)
                       {
                           auto bdry   = intersection(mesh[mesh_id_t::cells][level], boundary_region).on(level);
                           auto coeffs = get_coefficients(cell_length(level));
                           for_each_stencil(mesh,
                                            bdry,
                                            stencil,
                                            [&](auto& cells)
                                            {
                                                func(cells, coeffs);
                                            });
                       });
    }

    template <class Mesh, std::size_t n_directions, std::size_t stencil_size, class Func>
    void for_each_stencil_on_boundary(const Mesh& mesh,
                                      const std::array<DirectionVector<Mesh::dim>, n_directions>& boundary_directions,
                                      const std::array<Stencil<stencil_size, Mesh::dim>, n_directions>& stencils,
                                      Func&& func)
    {
        for (std::size_t i = 0; i < n_directions; ++i)
        {
            for_each_stencil_on_boundary(mesh,
                                         boundary_directions[i],
                                         stencils[i],
                                         [&](auto& cells)
                                         {
                                             func(cells, boundary_directions[i]);
                                         });
        }
    }

    template <class Mesh, std::size_t n_directions, std::size_t stencil_size, class GetCoeffsFunc, class Func>
    void for_each_stencil_on_boundary(const Mesh& mesh,
                                      const std::array<DirectionVector<Mesh::dim>, n_directions>& boundary_directions,
                                      const std::array<Stencil<stencil_size, Mesh::dim>, n_directions>& stencils,
                                      GetCoeffsFunc&& get_coefficients,
                                      Func&& func)
    {
        for (std::size_t i = 0; i < n_directions; ++i)
        {
            for_each_stencil_on_boundary(mesh,
                                         boundary_directions[i],
                                         stencils[i],
                                         get_coefficients,
                                         [&](auto& cells, auto& coeffs)
                                         {
                                             func(cells, boundary_directions[i], coeffs);
                                         });
        }
    }

}
