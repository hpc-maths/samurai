#pragma once
#include "stencil.hpp"

namespace samurai
{
    template <class Mesh, class Vector>
    auto
    boundary_layer(const Mesh& mesh, const typename Mesh::lca_type& domain, std::size_t level, const Vector& direction, std::size_t layer_width)
    {
        using mesh_id_t = typename Mesh::mesh_id_t;

        const auto& cells = mesh[mesh_id_t::cells][level];

        return difference(cells, translate(self(domain).on(level), -layer_width * direction));
    }

    template <class Mesh, class Vector>
    inline auto domain_boundary_layer(const Mesh& mesh, std::size_t level, const Vector& direction, std::size_t layer_width)
    {
        return boundary_layer(mesh, mesh.domain(), level, direction, layer_width);
    }

    template <class Mesh, class Vector>
    inline auto subdomain_boundary_layer(const Mesh& mesh, std::size_t level, const Vector& direction, std::size_t layer_width)
    {
        return boundary_layer(mesh, mesh.subdomain(), level, direction, layer_width);
    }

    template <class Mesh, class Vector>
    inline auto domain_boundary(const Mesh& mesh, std::size_t level, const Vector& direction)
    {
        return domain_boundary_layer(mesh, level, direction, 1);
    }

    template <class Mesh, class Vector>
    inline auto subdomain_boundary(const Mesh& mesh, std::size_t level, const Vector& direction)
    {
        return subdomain_boundary_layer(mesh, level, direction, 1);
    }

    template <class Mesh>
    inline auto domain_boundary(const Mesh& mesh, std::size_t level)
    {
        using mesh_id_t = typename Mesh::mesh_id_t;

        const auto& cells = mesh[mesh_id_t::cells][level];

        return difference(cells, contract(self(mesh.domain()).on(level), 1));
    }

    template <class Mesh>
    auto domain_boundary_outer_layer(const Mesh& mesh, std::size_t level, std::size_t layer_width)
    {
        using lca_t = typename Mesh::lca_type;
        using lcl_t = typename Mesh::lcl_type;

        auto domain = self(mesh.domain()).on(level);

        lcl_t outer_boundary_lcl(level, mesh.origin_point(), mesh.scaling_factor());

        for_each_cartesian_direction<Mesh::dim>(
            [&](const auto& direction)
            {
                auto inner_boundary = domain_boundary(mesh, level, direction);
                for (std::size_t layer = 1; layer <= layer_width; ++layer)
                {
                    auto outer_layer = difference(translate(inner_boundary, layer * direction), domain);
                    outer_layer(
                        [&](const auto& i, const auto& index)
                        {
                            outer_boundary_lcl[index].add_interval({i});
                        });
                }
            });
        return lca_t(outer_boundary_lcl);
    }

    template <class Mesh>
    auto
    domain_boundary_outer_layer(const Mesh& mesh, std::size_t level, const DirectionVector<Mesh::dim>& direction, std::size_t layer_width)
    {
        using lca_t = typename Mesh::lca_type;
        using lcl_t = typename Mesh::lcl_type;

        auto domain = self(mesh.domain()).on(level);

        lcl_t outer_boundary_lcl(level, mesh.origin_point(), mesh.scaling_factor());

        auto inner_boundary = domain_boundary(mesh, level, direction);
        for (std::size_t layer = 1; layer <= layer_width; ++layer)
        {
            auto outer_layer = difference(translate(inner_boundary, layer * direction), domain);
            outer_layer(
                [&](const auto& i, const auto& index)
                {
                    outer_boundary_lcl[index].add_interval({i});
                });
        }
        return lca_t(outer_boundary_lcl);
    }

    template <class Mesh, class Subset, std::size_t stencil_size, class Equation, std::size_t nb_equations, class Func>
    void for_each_stencil_on_boundary(const Mesh& mesh,
                                      const Subset& boundary_region,
                                      const StencilAnalyzer<stencil_size, Mesh::dim>& stencil,
                                      const std::array<Equation, nb_equations>& equations,
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
                               equations_coeffs[i].stencil_coeffs = equations[i].get_stencil_coeffs(mesh.cell_length(level));
                               equations_coeffs[i].rhs_coeffs     = equations[i].get_rhs_coeffs(mesh.cell_length(level));
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
