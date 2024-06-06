#pragma once
#include "stencil.hpp"

namespace samurai
{
    template <class Mesh, class Vector>
    auto boundary(const Mesh& mesh, std::size_t level, const Vector& direction)
    {
        using mesh_id_t = typename Mesh::mesh_id_t;

        auto& cells  = mesh[mesh_id_t::cells][level];
        // auto& domain = mesh.domain();
        auto& domain = mesh.subdomain();

        auto max_level    = domain.level(); // domain.level();//mesh[mesh_id_t::cells].max_level();
        auto one_interval = 1 << (max_level - level);

        return difference(cells, translate(domain, -one_interval * direction)).on(level);
    }

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

    template <class Mesh, class Subset, std::size_t stencil_size, class Equation, std::size_t nb_equations, class Func>
    void for_each_stencil_on_boundary(const Mesh& mesh,
                                      const Subset& boundary_region,
                                      const Stencil<stencil_size, Mesh::dim>& stencil,
                                      std::array<Equation, nb_equations> equations,
                                      Func&& func)
    {
        using mesh_id_t         = typename Mesh::mesh_id_t;
        using equation_coeffs_t = typename Equation::equation_coeffs_t;

        for_each_level(mesh,
                       [&](std::size_t level)
                       {
                           auto bdry = intersection(mesh[mesh_id_t::cells][level], boundary_region).on(level);

                           std::array<equation_coeffs_t, nb_equations> equations_coeffs;
                           for (std::size_t i = 0; i < nb_equations; ++i)
                           {
                               equations_coeffs[i].ghost_index    = equations[i].ghost_index;
                               equations_coeffs[i].stencil_coeffs = equations[i].get_stencil_coeffs(cell_length(level));
                               equations_coeffs[i].rhs_coeffs     = equations[i].get_rhs_coeffs(cell_length(level));
                           }
                           for_each_stencil(mesh,
                                            bdry,
                                            stencil,
                                            [&](auto& cells)
                                            {
                                                func(cells, equations_coeffs);
                                            });
                       });
    }
}
